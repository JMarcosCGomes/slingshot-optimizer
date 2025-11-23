import numpy as np
import math #for arctang
import matplotlib as plt
import matplotlib.pyplot as plt

from matplotlib.animation import FuncAnimation

from Corpo_Celeste import Corpo_Celeste

from scipy.integrate import solve_ivp



class Universo:
    def __init__(self, duracao_padrao=4e6, ganho_duracao=100, intervalo_animacao=100):
        self.G = 6.67430e-11
        self.duracao = duracao_padrao * ganho_duracao
        self.intervalo_animacao = intervalo_animacao
        self.limite_x = 4e11
        self.limite_y = 4e11
        self.maneuver = []
        self.criar_corpos_celestes()


    def criar_corpos_celestes(self):
        # === Inicialização das listas ===
        self.corpos_celestes = []
        self.corpos_names = []
        self.corpos_massas = []
        
        # === Sol ===
        self.Sol = Corpo_Celeste(massa=2e30, raio=6.957e8, color="yellow", name="Sol", orbit_radius=0.0, angle_deg=0,)
        self.corpos_celestes.append(self.Sol)
        self.fixed_body_index = len(self.corpos_celestes) - 1
        self.fixed_body_name = self.corpos_celestes[self.fixed_body_index].name
        
        # === Terra ===
        self.Terra = Corpo_Celeste(massa=5.972e24, raio=6.371e6, color="blue", name="Terra", orbit_radius=1.49e11, angle_deg=-125,wir_id=self.fixed_body_index)
        covp = self.corpos_celestes[self.Terra.wir_id].return_cov_parameters()
        self.Terra.vel_x, self.Terra.vel_y = self.Terra.Calculate_Orbital_Velocity(**covp)
        self.corpos_celestes.append(self.Terra)
        self.planet_index = len(self.corpos_celestes) - 1
        self.planet_name = self.corpos_celestes[self.planet_index].name

        # === Probe ===
        #pro foguete (que lanca o probe) tambem ir na mesma direcao da velocidade da terra
        planet_vel_angle_rad = math.atan2(self.corpos_celestes[self.planet_index].vel_y, self.corpos_celestes[self.planet_index].vel_x)
        planet_vel_angle_deg = math.degrees(planet_vel_angle_rad)

        rocket_vel_module = 5.8e3
        rocket_vel_x = rocket_vel_module * np.sin(planet_vel_angle_deg)
        rocket_vel_y = rocket_vel_module * np.cos(planet_vel_angle_deg)


        #roda mas demora mt, o bom seria colocar a distancia como 1e7 ou 1e7, mas acho que deve ficar MUITO lento
        test_or1 = 5.0e7
        #roda e demora menos
        test_or2 = 2.0e8

        test_or = test_or2
        self.Probe = Corpo_Celeste(massa=5.9e6, raio=5e2, color="yellow", name="Probe", orbit_radius=test_or, angle_deg=planet_vel_angle_deg, wir_id=self.planet_index, is_orbiting=False)
        self.Probe.pos_x += self.Terra.pos_x
        self.Probe.pos_y += self.Terra.pos_y
        self.Probe.vel_x = self.corpos_celestes[self.planet_index].vel_x + rocket_vel_x
        self.Probe.vel_y = self.corpos_celestes[self.planet_index].vel_y + rocket_vel_y
        self.corpos_celestes.append(self.Probe)
        self.probe_index = len(self.corpos_celestes) - 1
        self.probe_name = self.corpos_celestes[self.probe_index].name

        # === Lua ===
        self.Lua = Corpo_Celeste(massa=7.346e22, raio=1.737e6, color="white", name="Lua", orbit_radius=3.84e8, angle_deg=30, wir_id=self.corpos_celestes.index(self.Terra))
        self.Lua.pos_x += self.Terra.pos_x
        self.Lua.pos_y += self.Terra.pos_y
        covp = self.corpos_celestes[self.Lua.wir_id].return_cov_parameters()
        self.Lua.vel_x, self.Lua.vel_y = self.Lua.Calculate_Orbital_Velocity(**covp)
        self.corpos_celestes.append(self.Lua)

        for cc in self.corpos_celestes:
            self.corpos_names.append(cc.name)
            self.corpos_massas.append(cc.massa)

        self.y0 = self.get_y0()


    # y0 = vetor inicial
    def get_y0(self):
        y0 = []
        for index, cc in enumerate(self.corpos_celestes):
            if index != self.fixed_body_index:
                estado = cc.return_estado()
                y0.extend(estado)
        return np.array(y0)
    
    
    def criar_plot(self, ax):
        ax.set_xlim(-self.limite_x, self.limite_x)
        ax.set_ylim(-self.limite_y, self.limite_y)
        ax.set_aspect("equal", adjustable="datalim")
        ax.set_facecolor("gray")
        ax.set_title("Sistema Solar")
        ax.set_xlabel("X (m)")
        ax.set_ylabel("Y (m)")


    def plotar_corpos_celestes(self, ax):

        for cc in self.corpos_celestes:
            if (cc.name == "Lua") | (cc.name == "Probe"):
                ax.plot([cc.pos_x], [cc.pos_y], "o", markersize=6, label=f"{cc.name}", markerfacecolor=cc.color, markeredgecolor="black", markeredgewidth=1.0)
            else:
                ax.plot([cc.pos_x], [cc.pos_y], "o", markersize=12, label=f"{cc.name}", markerfacecolor=cc.color, markeredgecolor="black", markeredgewidth=1.0)


    def simple_plot(self):
        fig, ax = plt.subplots(figsize=(15, 8))
        self.criar_plot(ax)
        self.plotar_corpos_celestes(ax)
        ax.set_title("Plot simples para visualizar as posicoes iniciais")
        plt.legend()
        plt.show()


    def capturar_posicao_inicial(self):
        fig, ax = plt.subplots(figsize=(15, 8))
        self.criar_plot(ax)
        ax.set_title('Clique para definir a posicao final desejada')
        self.plotar_corpos_celestes(ax)
        posicao = []

        def clique(evento):
            if evento.inaxes == ax:
                posicao.append(evento.xdata)
                posicao.append(evento.ydata)
                plt.close()
        fig.canvas.mpl_connect('button_press_event', clique)
        plt.legend()
        plt.show()
        return posicao if len(posicao) == 2 else None


    # equacoes_movimento_setup
    def equacoes_movimento_setup(self, y):
        all_positions = []
        all_velocities = []

        # como temos xa,ya,vxa,vya,xb,yb,vxb,vyb,xc,yc,vxc,vyc
        # precisamos de um ponteiro = ptr pra ir em cada valor de forma organizada
        ptr = 0
        for i, cc in enumerate(self.corpos_celestes):
            if i == self.fixed_body_index:
                all_positions.append(np.array(cc.return_pos()))
                all_velocities.append(np.array(cc.return_vel()))
            else:
                all_positions.append(np.array([y[ptr], y[ptr + 1]]))
                all_velocities.append(np.array([y[ptr + 2], y[ptr + 3]]))
                ptr += 4

        dydt = np.zeros_like(y)
        return all_positions, all_velocities, dydt


    # tem que deixar t aqui pelo Solver
    def equacoes_movimento(self, t, y):
        all_positions, all_velocities, dydt = self.equacoes_movimento_setup(y)

        # dessa vez o ptr eh pro vx,vy,ax,ay,...
        ptr = 0
        for i in range(len(self.corpos_celestes)):
            # se for o index do sol ele passa pra prox iteracao
            if i == self.fixed_body_index:
                continue

            acc_i = np.array([0.0, 0.0])

            for j in range(len(self.corpos_celestes)):
                if i == j:
                    continue
                r = all_positions[j] - all_positions[i]
                r2 = np.dot(r, r)
                if r2 < 1e-10:
                    continue
                r3 = r2**1.5
                acc_i += self.G * self.corpos_massas[j] * r / r3

            dydt[ptr] = all_velocities[i][0]
            dydt[ptr + 1] = all_velocities[i][1]
            dydt[ptr + 2] = acc_i[0]
            dydt[ptr + 3] = acc_i[1]

            ptr += 4

        return dydt


    def get_current_state(self, y_in_t):
        current_states = []
        ptr = 0
        for i, cc in enumerate(self.corpos_celestes):
            if i == self.fixed_body_index:
                current_states.append({"pos_x": cc.pos_x, "pos_y": cc.pos_y, "vel_x": cc.vel_x, "vel_y": cc.vel_y})
            else:
                current_states.append({"pos_x": y_in_t[ptr], "pos_y": y_in_t[ptr + 1], "vel_x": y_in_t[ptr + 2], "vel_y": y_in_t[ptr + 3]})
                ptr += 4
        return current_states


    #se eu otimizar pra chegar numa posicao posso usar o clique novamente
    def simular_setup(self):
        posicao_inicial = self.capturar_posicao_inicial()
        if posicao_inicial is None:
            print("Nenhum clique detectado.")
            return None


    def run_until_aphelion(self):
        solve_ivp_parameters = self.get_solveivp_params(simulation_segment="initial")
        sol1 = solve_ivp(**solve_ivp_parameters)
        return sol1


    def run_after_aphelion(self, new_y0):
        solve_ivp_parameters = self.get_solveivp_params(simulation_segment="next", new_y0=new_y0)
        sol2 = solve_ivp(**solve_ivp_parameters)
        return sol2


    #to optimizer in the future
    def get_probe_v_indexes(self):
        probe_vx_index = (self.probe_index-1)*4+2
        probe_vy_index = (self.probe_index-1)*4+3
        return probe_vx_index, probe_vy_index


    #test if it'll work in the optimizer
    def simular_optimized(self, params):
        dvx, dvy = params
        #dvx = 5e3
        #dvy = -5e3

        sol1 = self.run_until_aphelion()

        new_y0 = sol1.y[:, -1].copy()
        probe_vx_index, probe_vy_index = self.get_probe_v_indexes()
        new_y0[probe_vx_index] += dvx
        new_y0[probe_vy_index] += dvy

        sol2 = self.run_after_aphelion(new_y0=new_y0)

        t_full = np.concatenate((sol1.t, sol2.t))
        y_full = np.concatenate((sol1.y, sol2.y), axis=1)

        solucao_array = y_full.T

        #"""get trajectory trace
        for step in range(len(t_full)):
            y_in_t = solucao_array[step]
            current_states = self.get_current_state(y_in_t)

            for i, state in enumerate(current_states):
                self.corpos_celestes[i].pos_x = state["pos_x"]
                self.corpos_celestes[i].pos_y = state["pos_y"]
                self.corpos_celestes[i].vel_x = state["vel_x"]
                self.corpos_celestes[i].vel_y = state["vel_y"]
                self.corpos_celestes[i].trace.append((state["pos_x"], state["pos_y"]))
        #"""

        return solucao_array
    

    def simular_simple(self):
        #self.simular_setup()

        solve_ivp_parameters = self.get_solveivp_params(simulation_segment="initial")
        solucao1 = solve_ivp(**solve_ivp_parameters)

        new_y02 = solucao1.y[:, -1].copy() #ultimo estado
        deltavx = 5e2
        deltavy = -2e3
        new_y02[(self.probe_index-1)*4+2] += deltavx
        new_y02[(self.probe_index-1)*4+3] += deltavy

        solve_ivp_parameters = self.get_solveivp_params(simulation_segment="next", new_y0=new_y02)
        solucao2 = solve_ivp(**solve_ivp_parameters)

        t_full = np.concatenate((solucao1.t, solucao2.t))
        y_full = np.concatenate((solucao1.y, solucao2.y), axis=1)

        solucao_array = y_full.T

        for step in range(len(t_full)):
            y_in_t = solucao_array[step]
            current_states = self.get_current_state(y_in_t)

            for i, state in enumerate(current_states):
                self.corpos_celestes[i].pos_x = state["pos_x"]
                self.corpos_celestes[i].pos_y = state["pos_y"]
                self.corpos_celestes[i].vel_x = state["vel_x"]
                self.corpos_celestes[i].vel_y = state["vel_y"]
                self.corpos_celestes[i].trace.append((state["pos_x"], state["pos_y"]))

        """
        #tem que iterar nas duas solucoes, faco isso depois se necessario

        print("[INFO] DETECTED EVENTS")
        print(f"[INFO] solucao.t_events: {solucao.t_events}")
        if len(solucao.t_events) > 0:
            print("[INFO] STATES IN EVENT:")
            for idx_event, t_list in enumerate(solucao.t_events):
                for t_event in t_list:
                    state = solucao.sol(t_event)
                    probe_x = state[self.probe_index*4]
                    probe_y = state[self.probe_index*4 + 1]
                    print(f"Event {idx_event} – t = {t_event:.2e} s – Probe ({probe_x:.3e}, {probe_y:.3e})")
        """

        return solucao_array
    

    def get_solveivp_params(self, simulation_segment, new_y0=None):
        #fazer uma self.coisa pra ver se ta otimizando ou não, quando n ta otimizando eu queria ver melhor esse slingshot
        #events = self.create_event_functions()
        #t_eval = np.linspace(0, self.duracao, 20000)
        
        #simulation segment can be. initial: events are event_aphelion; next: for now events: None
        if simulation_segment == "initial":
            t_max =  self.duracao
            t_eval = np.linspace(0, t_max, 20000)
            y0 = self.y0
            events = self.create_event_functions()

        elif simulation_segment == "next":
            UMANOEMEIO = 7.884e7
            DOISANOSEMEIO = 4.73e7
            t_max = UMANOEMEIO
            #t_max = self.duracao/4
            ratio = t_max / self.duracao
            t_eval = np.linspace(0, t_max, int(20000 * ratio))
            y0 = new_y0
            events = None #i could use an "event time cap here"
        else:
            print("YOU FAILED, error in get_solveivp_params, unexpected simulation_segment")

        sivp_params = {
            "fun": self.equacoes_movimento,
            "t_span": (0, t_max),
            "y0": y0,
            "method": 'RK45', 
            "t_eval": t_eval,
            "events": events,
            "dense_output": True,
            "max_step": 86400, #isso dá 1/1 dia por passo. 
            "rtol": 1e-9,
            "atol": 1e-12,
        }
        

        return sivp_params
    

    def animar(self, solucao_array):
            if solucao_array is None:
                print("[ERRO] solucao_array chegou em universo.animar() vazio")
                return

            fig, ax = plt.subplots(figsize=(15, 8))
            self.criar_plot(ax)

            t_eval = np.linspace(0, self.duracao, len(solucao_array))


            linhas = []
            pontos = []
            for cc in self.corpos_celestes:
                (linha,) = ax.plot([], [], "-", lw=1, color=cc.color, alpha=0.5)
                if (cc.name == "Lua") | (cc.name == "Probe"):
                    (ponto,) = ax.plot([],[],"o",markersize=6,label=f"{cc.name}",markerfacecolor=cc.color,markeredgecolor="black",markeredgewidth=1.0)
                else:
                    (ponto,) = ax.plot([],[],"o",markersize=12,label=f"{cc.name}",markerfacecolor=cc.color)
                linhas.append(linha)
                pontos.append(ponto)

            def update(frame):

                current_t = t_eval[frame]

                current_states = self.get_current_state(solucao_array[frame])
                for i, cc in enumerate(self.corpos_celestes):
                    linhas[i].set_data([p[0] for p in cc.trace[:frame + 1]], [p[1] for p in cc.trace[:frame + 1]])
                    pontos[i].set_data([current_states[i]["pos_x"]], [current_states[i]["pos_y"]])


                return linhas + pontos
            anim = FuncAnimation(fig, update, frames=len(solucao_array), interval=self.intervalo_animacao, blit=False, repeat=False)
            plt.legend()
            plt.show()


    def create_event_functions(self):
    
        def event_closest_approach_earth(t, y):

            earth_pos = None
            probe_pos = None

            ptr = 0

            #for i, cc in enumerate(self.corpos_celestes):
            for i in range(len(self.corpos_celestes)):
                if i == self.fixed_body_index:
                    continue
                
                if i == self.planet_index: #planeta relacionada ao gravity assist
                    earth_pos = np.array([y[ptr], y[ptr+1]])
                elif i == self.probe_index: #Probe
                    probe_pos = np.array([y[ptr], y[ptr+1]])
                
                ptr += 4

            if (earth_pos is not None) and (probe_pos is not None):
                distance = np.linalg.norm(probe_pos - earth_pos)
                return distance - 1e8
            
            return 1e10

        event_closest_approach_earth.terminal = False
        event_closest_approach_earth.direction = -1

        
        def event_probe_escape_velocity(t, y):
            ptr = 0
            earth_pos = None
            probe_pos = None
            probe_vel = None

            #for i, cc in enumerate(self.corpos_celestes):
            for i in range(len(self.corpos_celestes)):

                if i == self.fixed_body_index:
                    continue
                
                if i == self.planet_index:
                    earth_pos = np.array([y[ptr], y[ptr+1]])

                elif i == self.probe_index:
                    probe_pos = np.array([y[ptr], y[ptr+1]])
                    probe_vel = np.array([y[ptr+2], y[ptr+3]])
                ptr += 4

            if earth_pos is not None and probe_pos is not None:
                r_vec = probe_pos - earth_pos
                r = np.linalg.norm(r_vec)
                v = np.linalg.norm(probe_vel)
                v_escape = np.sqrt(2 * self.G * self.Terra.massa / r)

                return v - v_escape

            return -1e10
    
        event_probe_escape_velocity.terminal = False
        event_probe_escape_velocity.direction = 1

        def event_aphelion(t, y):
            #pra achar o aphelion entre probe e fixed_body
            #como o return_pos e return_vel tao retornando listas vou deixar assim, mas seria só deixar um paramtro pra isso, daqueles blabla=normal
            fixed_body_pos = np.array([self.corpos_celestes[self.fixed_body_index].pos_x, self.corpos_celestes[self.fixed_body_index].pos_y])
            fixed_body_vel = np.array([self.corpos_celestes[self.fixed_body_index].vel_x, self.corpos_celestes[self.fixed_body_index].vel_y])
            probe_pos = None
            probe_vel = None
            ptr = 0

            for i in range(len(self.corpos_celestes)):
                if i == self.fixed_body_index: 
                    continue
                elif i == self.probe_index: #Probe
                    probe_pos = np.array([y[ptr], y[ptr+1]])
                    probe_vel = np.array([y[ptr+2], y[ptr+3]])
                ptr += 4
                if probe_pos is not None and probe_vel is not None:
                    r_rel = probe_pos - fixed_body_pos
                    v_rel = probe_vel - fixed_body_vel
                    return np.dot(r_rel, v_rel)
            
        event_aphelion.terminal = True
        event_aphelion.direction = -1

        return [event_aphelion]
        return [event_closest_approach_earth, event_probe_escape_velocity, event_aphelion]
    