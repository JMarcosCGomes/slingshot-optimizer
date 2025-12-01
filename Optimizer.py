from Universo import Universo
import numpy as np
from scipy.optimize import minimize

class Optimizer:

    def __init__(self, planet_angle_deg=0, max_step = 86400, max_dv=5e3, initial_guess=[0.0, 0.0]):
        self.max_dv = max_dv
        self.initial_guess = initial_guess #[guess_dvx, guess_dvy]
        self.optimization_attempts = 0
        #cache
        self.last_params = None
        self.last_y_full = None 
        #universe setup
        self.universo = Universo(planet_angle_deg=planet_angle_deg, max_step=max_step)
        self.sol1 = self.universo.run_until_aphelion()
        self.post_aphelion_y = self.sol1.y[:, -1].copy()
        #constraints
        self.set_constraints()


    def run_simulation_if_needed(self, params):
        if (self.last_params is not None) and (np.allclose(params, self.last_params, atol=1e-12, rtol=0)):
            return self.last_y_full
        dvx, dvy = params
        post_aphelion_y = self.post_aphelion_y.copy()
        post_aphelion_y[(self.universo.probe_index-1)*4+2] += dvx
        post_aphelion_y[(self.universo.probe_index-1)*4+3] += dvy

        #preciso dele pra pegar a distancia do pos aphelion
        self.sol2 = self.universo.run_after_aphelion(new_y0=post_aphelion_y)
        y_full = np.concatenate((self.sol1.y, self.sol2.y), axis=1)
        self.last_params = params.copy()
        self.last_y_full = y_full
        return y_full


    def objective(self, params):
        dvx, dvy = params
        self.optimization_attempts += 1
        y_full = self.run_simulation_if_needed(params)
        final_y = y_full[:, -1]
        
        probe_final_x = final_y[(self.universo.probe_index-1)*4]
        probe_final_y = final_y[(self.universo.probe_index-1)*4+1]
        probe_final_vx = final_y[(self.universo.probe_index-1)*4+2]
        probe_finaL_vy = final_y[(self.universo.probe_index-1)*4+3]
        fixed_body_x = self.universo.corpos_celestes[self.universo.fixed_body_index].pos_x
        fixed_body_y = self.universo.corpos_celestes[self.universo.fixed_body_index].pos_y

        dx = probe_final_x - fixed_body_x
        dy = probe_final_y - fixed_body_y
        r_module = np.sqrt(dx**2 + dy**2)
        v_module = np.sqrt(probe_final_vx**2 + probe_finaL_vy**2)
        mu_fixed_body = self.universo.G * self.universo.corpos_celestes[self.universo.fixed_body_index].massa
        #energia mecanica especifica heliocentrica
        energy = (v_module**2)/2 - (mu_fixed_body / r_module)

        #y pós afélio
        y_p2 = self.sol2.y
        planet_id = self.universo.planet_index
        probe_id = self.universo.probe_index
        probe_all_x = y_p2[(probe_id-1)*4]
        probe_all_y = y_p2[(probe_id-1)*4 + 1]
        planet_all_x = y_p2[(planet_id-1)*4]
        planet_all_y = y_p2[(planet_id-1)*4 + 1]
        dists = np.sqrt((probe_all_x - planet_all_x)**2 + (probe_all_y - planet_all_y)**2)
        minimal_distance = np.min(dists)

        energy_weight = 2e-6
        score_energy = - energy * energy_weight
        
        log_distance = np.log10(minimal_distance)
        penalty_factor = 5e2
        #minimal_distance=1e7 -> log_distance = 7.0
        distance_penalty = max(0, (log_distance - 7.0)) * penalty_factor
        score = score_energy + distance_penalty
        
        #if you don't want logs, just comment this parte
        #VVVV
        print(f"attempt: {self.optimization_attempts}")
        print(f"dvx: {dvx}")
        print(f"dvy: {dvy}")
        print(f"energy: {energy}")
        print(f"Minimal distance: {minimal_distance}")
        print(f"Score: {score}")
        print("------------------")
        #^^^^

        return score


    def dv_constraint(self, params):
        return self.max_dv - np.linalg.norm(params)
    

    def planet_collision_constraint(self, params):
        planet_id = self.universo.planet_index
        probe_id = self.universo.probe_index

        y_full = self.run_simulation_if_needed(params)

        probe_all_x = y_full[(probe_id-1)*4]
        probe_all_y = y_full[(probe_id-1)*4 + 1]
        planet_all_x = y_full[(planet_id-1)*4]
        planet_all_y = y_full[(planet_id-1)*4 + 1]
        dist = np.sqrt((probe_all_x - planet_all_x)**2 + (probe_all_y - planet_all_y)**2)
        minimal_distance_found = np.min(dist)
        planet_radius = self.universo.corpos_celestes[self.universo.planet_index].raio
        safety_margin = 2e6
        
        return minimal_distance_found - (planet_radius + safety_margin)


    def set_constraints(self):
        constraint_dv = {'type': 'ineq', 'fun': self.dv_constraint}
        constraint_planet_collision = {'type': 'ineq', 'fun': self.planet_collision_constraint}

        self.constraints = (constraint_dv, constraint_planet_collision)

    def optimize(self, maxiter=120, ftol=1e-4): 
        bounds = [(-self.max_dv, self.max_dv), (-self.max_dv, self.max_dv)]

        method = 'SLSQP'
        options_slsqp = {
            'maxiter': maxiter,
            'ftol': ftol,
            #'disp': True,
            'eps': 5.0,
        }
        options = options_slsqp

        res = minimize(self.objective, self.initial_guess, method=method, bounds=bounds, constraints=self.constraints, options=options)
        self.final_score = -res.fun
        self.best_dv = res.x #dvx, dvy


        print("================= SUMMARY =================")
        print(f"Method: {method}")
        print(f"best deltaV: {self.best_dv}")
        print(f"final score (energy - log(minimal_distance): {self.final_score}")
        print("============ END OF OPTIMIZATION ============") 

        return self.best_dv