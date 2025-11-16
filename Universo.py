import numpy as np
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
        self.criar_corpos_celestes()


    def criar_corpos_celestes(self):
        # === Inicialização das listas ===
        self.corpos_celestes = []
        self.corpos_names = []
        self.corpos_massas = []
        
        # === Sol ===
        self.Sol = Corpo_Celeste(massa=2e30, raio=6.957e8, color="yellow", name="Sol", orbit_radius=0.0, angle_deg=0,)
        self.fixed_body_name = self.Sol.name
        self.fixed_body_index = 0
        self.corpos_celestes.append(self.Sol)
        
        # === Terra ===
        self.Terra = Corpo_Celeste(massa=5.972e24, raio=6.371e6, color="blue", name="Terra", orbit_radius=1.49e11, angle_deg=0,)
        self.Terra.vel_x, self.Terra.vel_y = self.Terra.Calculate_SunPlanet_Speed()
        self.planet_name = self.Terra.name
        self.planet_index = 1
        self.corpos_celestes.append(self.Terra)

        # === Probe ===
        altitude_to_earth = 2.347e6
        test_or1 = self.Terra.raio + altitude_to_earth
        test_or2 = 6.371e6 + 2.347e6
        test_or3 = 8.7e6
        #roda mas demora mt
        test_or4 = 5.0e7
        #roda e demora menos
        test_or5 = 2.0e8

        test_or = test_or5
        self.Probe = Corpo_Celeste(massa=5.9e6, raio=5e2, color="yellow", name="Probe", orbit_radius=test_or, angle_deg=30,)
        self.Probe.pos_x += self.Terra.pos_x
        self.Probe.pos_y += self.Terra.pos_y
        cpss_params = self.Terra.return_cpss_params()
        self.Probe.vel_x, self.Probe.vel_y = self.Probe.Calculate_PlanetSatelite_Speed(**cpss_params)
        self.probe_name = self.Probe.name
        self.probe_index = 2
        self.corpos_celestes.append(self.Probe)

        # === Lua ===
        self.Lua = Corpo_Celeste(massa=7.346e22, raio=1.737e6, color="white", name="Lua", orbit_radius=3.84e8, angle_deg=30,)
        self.Lua.pos_x += self.Terra.pos_x
        self.Lua.pos_y += self.Terra.pos_y
        cpss_params = self.Terra.return_cpss_params()
        self.Lua.vel_x, self.Lua.vel_y = self.Lua.Calculate_PlanetSatelite_Speed(**cpss_params)
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

    
    def simular_setup(self):
        posicao_inicial = self.capturar_posicao_inicial()
        if posicao_inicial is None:
            print("Nenhum clique detectado.")
            return None
        #se adicionar algum curpo novo depois de criar_corpos_celestes
        #self.y0 = self.get_y0()


    def simular(self):
        self.simular_setup()
        events = self.create_event_functions()

        t_eval = np.linspace(0, self.duracao, 20000)
        solucao = solve_ivp(self.equacoes_movimento, (0, self.duracao), self.y0, method='RK45', t_eval=t_eval, events=events, dense_output=True, rtol=1e-9, atol=1e-12,)
        

        solucao_array = solucao.y.T

        for step in range(len(solucao.t)):
            y_in_t = solucao_array[step]
            current_states = self.get_current_state(y_in_t)

            for i, state in enumerate(current_states):
                self.corpos_celestes[i].pos_x = state["pos_x"]
                self.corpos_celestes[i].pos_y = state["pos_y"]
                self.corpos_celestes[i].vel_x = state["vel_x"]
                self.corpos_celestes[i].vel_y = state["vel_y"]
                self.corpos_celestes[i].trace.append((state["pos_x"], state["pos_y"]))

        print("[INFO] DETECTED EVENTS")
        print(f"[INFO] solucao.t_events: {solucao.t_events}")
        

        if len(solucao.t_events) > 0:
            print("[INFO] STATES IN EVENT:")
            for idx_event, t_list in enumerate(solucao.t_events):
                for t_event in t_list:
                    state = solucao.sol(t_event)
                    # probe está no segundo estado, 4,5,6,7
                    probe_x = state[4]
                    probe_y = state[5]
                    print(f"Evento {idx_event} – t = {t_event:.2e} s – Probe ({probe_x:.3e}, {probe_y:.3e})")


        return solucao_array
    

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

            for i, cc in enumerate(self.corpos_celestes):
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

            for i, cc in enumerate(self.corpos_celestes):

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


        """
        #a simple event that finished the code when t = 1e6
        def event_test(t, y):
            return t - 1e6
        
        #True to stop when happens and False to don't
        event_test.terminal = False
        event_test.direction = 1

        return [event_closest_approach_earth, event_probe_escape_velocity, event_test]
        #"""

        return [event_closest_approach_earth, event_probe_escape_velocity]


"""
TODO 1: Adicionar manobra/impulso; (Inicialmente testar com valores fixos pra ver o efeito)
TODO 2: Criar optimizer.py pra fazer a otimizacao
TODO 3: Aplicar a otiimzação em si no resto do codigo
"""