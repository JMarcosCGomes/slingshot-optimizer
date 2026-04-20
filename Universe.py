import numpy as np
import math #for arctang
from scipy.integrate import solve_ivp

from CelestialBody import CelestialBody


class Universe:
    def __init__(self, planet_angle_deg=0, max_step = 86400, duration=4e8):
        self.planet_angle_deg = planet_angle_deg
        self.max_step = max_step
        self.G = 6.67430e-11
        self.duration = duration
        self.create_celestial_bodies()


    def create_celestial_bodies(self):
        # === Inicialização das listas ===
        self.celestial_bodies = []
        self.body_names = []
        self.body_masses = []
        
        # === Sun ===
        self.Sun = CelestialBody(mass=2e30, radius=6.957e8, color="yellow", name="Sun", orbit_radius=0.0, angle_deg=0,)
        self.celestial_bodies.append(self.Sun)
        self.fixed_body_index = len(self.celestial_bodies) - 1
        self.fixed_body_name = self.celestial_bodies[self.fixed_body_index].name
        
        # === Earth ===
        self.Earth = CelestialBody(mass=5.972e24, radius=6.371e6, color="blue", name="Earth", orbit_radius=1.49e11, angle_deg=self.planet_angle_deg,wir_id=self.fixed_body_index)
        covp = self.celestial_bodies[self.Earth.wir_id].get_cov_parameters()
        self.Earth.vel_x, self.Earth.vel_y = self.Earth.Calculate_Orbital_Velocity(**covp)
        self.celestial_bodies.append(self.Earth)
        self.planet_index = len(self.celestial_bodies) - 1
        self.planet_name = self.celestial_bodies[self.planet_index].name

        # === Probe ===
        #pro foguete (que lanca o probe) ir na frente do planeta
        planet_vel = [self.celestial_bodies[self.planet_index].vel_x, self.celestial_bodies[self.planet_index].vel_y]
        planet_vel_norm = np.linalg.norm(planet_vel)
        unit_tangent = planet_vel / planet_vel_norm
        probe_angle_rad = math.atan2(unit_tangent[1], unit_tangent[0])
        probe_angle_deg = math.degrees(probe_angle_rad)

        rocket_vel_module = 5.8e3
        rocket_vel = rocket_vel_module * unit_tangent
        test_or2 = 2.0e8

        test_or = test_or2
        self.Probe = CelestialBody(mass=5.9e6, radius=5e2, color="yellow", name="Probe", orbit_radius=test_or, angle_deg=probe_angle_deg, wir_id=self.planet_index, is_orbiting=False)
        self.Probe.pos_x += self.Earth.pos_x
        self.Probe.pos_y += self.Earth.pos_y
        self.Probe.vel_x = self.celestial_bodies[self.planet_index].vel_x + rocket_vel[0]
        self.Probe.vel_y = self.celestial_bodies[self.planet_index].vel_y + rocket_vel[1]
        self.celestial_bodies.append(self.Probe)
        self.probe_index = len(self.celestial_bodies) - 1
        self.probe_name = self.celestial_bodies[self.probe_index].name

        # === Moon ===
        self.Moon = CelestialBody(mass=7.346e22, radius=1.737e6, color="white", name="Moon", orbit_radius=3.84e8, angle_deg=30, wir_id=self.celestial_bodies.index(self.Earth))
        self.Moon.pos_x += self.Earth.pos_x
        self.Moon.pos_y += self.Earth.pos_y
        covp = self.celestial_bodies[self.Moon.wir_id].get_cov_parameters()
        self.Moon.vel_x, self.Moon.vel_y = self.Moon.Calculate_Orbital_Velocity(**covp)
        self.celestial_bodies.append(self.Moon)

        for cb in self.celestial_bodies:
            self.body_names.append(cb.name)
            self.body_masses.append(cb.mass)

        self.y0 = self.get_y0()


    # y0 = vetor inicial
    def get_y0(self):
        y0 = []
        for index, cb in enumerate(self.celestial_bodies):
            if index != self.fixed_body_index:
                state = cb.get_state()
                y0.extend(state)
        return np.array(y0)
    

    def get_celestial_bodies(self):
        return self.celestial_bodies


    # equations_of_motion_setup
    def equations_of_motion_setup(self, y):
        all_positions = []
        all_velocities = []

        # como temos xa,ya,vxa,vya,xb,yb,vxb,vyb,xc,yc,vxc,vyc
        # precisamos de um ponteiro = ptr pra ir em cada valor de forma organizada
        ptr = 0
        for i, cb in enumerate(self.celestial_bodies):
            if i == self.fixed_body_index:
                all_positions.append(np.array(cb.get_pos()))
                all_velocities.append(np.array(cb.get_vel()))
            else:
                all_positions.append(np.array([y[ptr], y[ptr + 1]]))
                all_velocities.append(np.array([y[ptr + 2], y[ptr + 3]]))
                ptr += 4

        dydt = np.zeros_like(y)
        return all_positions, all_velocities, dydt


    # tem que deixar t aqui pelo Solver
    def equations_of_motion(self, t, y):
        all_positions, all_velocities, dydt = self.equations_of_motion_setup(y)

        # dessa vez o ptr eh pro vx,vy,ax,ay,...
        ptr = 0
        for i in range(len(self.celestial_bodies)):
            # se for o index do sol ele passa pra prox iteracao
            if i == self.fixed_body_index:
                continue

            acc_i = np.array([0.0, 0.0])

            for j in range(len(self.celestial_bodies)):
                if i == j:
                    continue
                r = all_positions[j] - all_positions[i]
                r2 = np.dot(r, r)
                if r2 < 1e-10:
                    continue
                r3 = r2**1.5
                acc_i += self.G * self.body_masses[j] * r / r3

            dydt[ptr] = all_velocities[i][0]
            dydt[ptr + 1] = all_velocities[i][1]
            dydt[ptr + 2] = acc_i[0]
            dydt[ptr + 3] = acc_i[1]

            ptr += 4

        return dydt


    def get_current_state(self, y_in_t):
        current_states = []
        ptr = 0
        for i, cb in enumerate(self.celestial_bodies):
            if i == self.fixed_body_index:
                current_states.append({"pos_x": cb.pos_x, "pos_y": cb.pos_y, "vel_x": cb.vel_x, "vel_y": cb.vel_y})
            else:
                current_states.append({"pos_x": y_in_t[ptr], "pos_y": y_in_t[ptr + 1], "vel_x": y_in_t[ptr + 2], "vel_y": y_in_t[ptr + 3]})
                ptr += 4
        return current_states


    def run_until_aphelion(self, add_trace=False):
        solve_ivp_parameters = self.get_solveivp_params(simulation_segment="initial")
        sol1 = solve_ivp(**solve_ivp_parameters)

        if add_trace == True:
            solution_array = sol1.y.T
            for step in range(len(sol1.t)):
                y_in_t = solution_array[step]
                current_states = self.get_current_state(y_in_t)
                for i, state in enumerate(current_states):
                    self.celestial_bodies[i].pos_x = state["pos_x"]
                    self.celestial_bodies[i].pos_y = state["pos_y"]
                    self.celestial_bodies[i].vel_x = state["vel_x"]
                    self.celestial_bodies[i].vel_y = state["vel_y"]
                    self.celestial_bodies[i].trace.append((state["pos_x"], state["pos_y"]))

        return sol1


    def run_after_aphelion(self, new_y0):
        solve_ivp_parameters = self.get_solveivp_params(simulation_segment="next", new_y0=new_y0)
        sol2 = solve_ivp(**solve_ivp_parameters)
        return sol2


    def get_probe_v_indexes(self):
        probe_vx_index = (self.probe_index-1)*4+2
        probe_vy_index = (self.probe_index-1)*4+3
        return probe_vx_index, probe_vy_index


    #test if it'll work in the optimizer
    def simulate_optimized(self, params):
        dvx, dvy = params

        sol1 = self.run_until_aphelion()

        new_y0 = sol1.y[:, -1].copy()
        probe_vx_index, probe_vy_index = self.get_probe_v_indexes()
        new_y0[probe_vx_index] += dvx
        new_y0[probe_vy_index] += dvy

        sol2 = self.run_after_aphelion(new_y0=new_y0)

        t_full = np.concatenate((sol1.t, sol2.t))
        y_full = np.concatenate((sol1.y, sol2.y), axis=1)

        solution_array = y_full.T

        for step in range(len(t_full)):
            y_in_t = solution_array[step]
            current_states = self.get_current_state(y_in_t)

            for i, state in enumerate(current_states):
                self.celestial_bodies[i].pos_x = state["pos_x"]
                self.celestial_bodies[i].pos_y = state["pos_y"]
                self.celestial_bodies[i].vel_x = state["vel_x"]
                self.celestial_bodies[i].vel_y = state["vel_y"]
                self.celestial_bodies[i].trace.append((state["pos_x"], state["pos_y"]))

        return solution_array
    

    def simulate_simple(self):

        solve_ivp_parameters = self.get_solveivp_params(simulation_segment="initial")
        sol1 = solve_ivp(**solve_ivp_parameters)

        new_y02 = sol1.y[:, -1].copy() #ultimo state
        deltavx = 5e2
        deltavy = -2e3
        new_y02[(self.probe_index-1)*4+2] += deltavx
        new_y02[(self.probe_index-1)*4+3] += deltavy

        solve_ivp_parameters = self.get_solveivp_params(simulation_segment="next", new_y0=new_y02)
        sol2 = solve_ivp(**solve_ivp_parameters)

        t_full = np.concatenate((sol1.t, sol2.t))
        y_full = np.concatenate((sol1.y, sol2.y), axis=1)

        solution_array = y_full.T

        for step in range(len(t_full)):
            y_in_t = solution_array[step]
            current_states = self.get_current_state(y_in_t)

            for i, state in enumerate(current_states):
                self.celestial_bodies[i].pos_x = state["pos_x"]
                self.celestial_bodies[i].pos_y = state["pos_y"]
                self.celestial_bodies[i].vel_x = state["vel_x"]
                self.celestial_bodies[i].vel_y = state["vel_y"]
                self.celestial_bodies[i].trace.append((state["pos_x"], state["pos_y"]))

        return solution_array
    

    def get_solveivp_params(self, simulation_segment, new_y0=None):
        if simulation_segment == "initial":
            t_max =  self.duration
            t_eval = np.linspace(0, t_max, 20000)
            y0 = self.y0
            events = self.create_event_functions()

        elif simulation_segment == "next":
            ONE_YEAR_IN_SECONDS = 3.154e7
            t_max = ONE_YEAR_IN_SECONDS * 1.5
            ratio = t_max / self.duration
            t_eval = np.linspace(0, t_max, int(20000 * ratio))
            y0 = new_y0
            events = None
        else:
            print("YOU FAILED, error in get_solveivp_params, unexpected simulation_segment")

        sivp_params = {
            "fun": self.equations_of_motion,
            "t_span": (0, t_max),
            "y0": y0,
            "method": 'RK45', 
            "t_eval": t_eval,
            "events": events,
            "dense_output": True,
            "max_step": self.max_step, #aqui eh onde aumenta pra ver melhor a Moon orbitando por exemplo 
            "rtol": 1e-9,
            "atol": 1e-12,
        }
        

        return sivp_params
    

    def create_event_functions(self):
    
        def event_aphelion(t, y):
            probe_pos = None
            probe_vel = None
            ptr = 0

            for i in range(len(self.celestial_bodies)):
                if i == self.fixed_body_index: 
                    continue
                if i == self.planet_index:
                    planet_pos = np.array([y[ptr], y[ptr+1]])
                    planet_vel = np.array([y[ptr+2], y[ptr+3]])
                elif i == self.probe_index: #Probe
                    probe_pos = np.array([y[ptr], y[ptr+1]])
                    probe_vel = np.array([y[ptr+2], y[ptr+3]])
                ptr += 4
                if probe_pos is not None and probe_vel is not None:
                    r_rel = probe_pos - planet_pos
                    v_rel = probe_vel - planet_vel
                    return np.dot(r_rel, v_rel)
            
        event_aphelion.terminal = True
        event_aphelion.direction = -1

        return [event_aphelion]