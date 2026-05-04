import numpy as np
from scipy.integrate import solve_ivp

from CelestialBody import CelestialBody


class Universe:
    def __init__(self, config):
        self.config = config
        self.max_step = float(config["simulation"]["max_step"])
        self.G = 6.67430e-11
        self.duration = float(config["simulation"]["duration"])
        self.create_celestial_bodies()


    def create_celestial_bodies(self):
        self.celestial_bodies = []
        self.body_names = []
        self.body_masses = []

        body_map = {} #used to iterate using wir names
        #name, role, mass, radius, color, orbit_radius, angle_deg, wir, is_orbiting
        for data in self.config["celestial_bodies"]:
            cb = CelestialBody(
                name=data["name"],
                role=data["role"],
                mass=float(data["mass"]),
                radius=float(data["radius"]),
                color=data["color"],
                orbit_radius=float(data["orbit_radius"]),
                angle_deg=data["angle_deg"],
                wir=data["wir"],
                is_orbiting=data["is_orbiting"],
            )

            self.celestial_bodies.append(cb)
            self.body_names.append(cb.name)
            self.body_masses.append(cb.mass)
            body_map[cb.name] = cb
            if cb.role == "fixed":
                self.fixed_body_index = len(self.celestial_bodies) - 1
            elif cb.role == "probe":
                self.probe_index = len(self.celestial_bodies) - 1               
        
        #loop to set planets
        for cb in self.celestial_bodies:
            if cb.role in ("fixed", "probe", "satellite"):
                continue
            cb_wir = body_map[cb.wir]
            cb.pos_x += cb_wir.pos_x
            cb.pos_y += cb_wir.pos_y
            if cb.is_orbiting:
                covp = cb_wir.get_cov_parameters()
                cb.vel_x, cb.vel_y = cb.Calculate_Orbital_Velocity(**covp)      

        #role = probe or satellite
        for cb in self.celestial_bodies:
            if cb.role in ("fixed", "planet", "generic"):
                continue
            
            if cb.role == "satellite":
                cb_wir = body_map[cb.wir]
                cb.pos_x += cb_wir.pos_x
                cb.pos_y += cb_wir.pos_y
                if cb.is_orbiting:
                    covp = cb_wir.get_cov_parameters()
                    cb.vel_x, cb.vel_y = cb.Calculate_Orbital_Velocity(**covp)   

            elif cb.role == "probe":
                cb_wir = body_map[cb.wir]
                planet_vel = np.array(cb_wir.get_vel())
                self.planet_index = self.celestial_bodies.index(cb_wir)
                self.planet_name = cb_wir.name
                launch_speed = 5.8e3
                cb.angle_deg = cb.Calculate_Probe_Angle(planet_vel)
                cb.pos_x, cb.pos_y = cb.Recalculate_Probe_Position(planet_vel)
                cb.pos_x += cb_wir.pos_x
                cb.pos_y += cb_wir.pos_y
                cb.vel_x, cb.vel_y = cb.Calculate_Probe_Velocity(launch_speed, planet_vel)

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


    def run_until_aphelion(self):
        solve_ivp_parameters = self.get_solveivp_params(simulation_segment="initial")
        sol1 = solve_ivp(**solve_ivp_parameters)
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
            ptr = 0
            planet_pos = planet_vel = probe_pos = probe_vel = None

            for i in range(len(self.celestial_bodies)):
                if i == self.fixed_body_index:
                    continue
                if i == self.planet_index:
                    planet_pos = np.array([y[ptr], y[ptr+1]])
                    planet_vel = np.array([y[ptr+2], y[ptr+3]])
                elif i == self.probe_index:
                    probe_pos = np.array([y[ptr], y[ptr+1]])
                    probe_vel = np.array([y[ptr+2], y[ptr+3]])
                ptr += 4

            r_rel = probe_pos - planet_pos
            v_rel = probe_vel - planet_vel
            return np.dot(r_rel, v_rel)


        event_aphelion.terminal = True
        event_aphelion.direction = -1

        return [event_aphelion]