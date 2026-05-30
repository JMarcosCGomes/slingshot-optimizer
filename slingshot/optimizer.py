import numpy as np
from scipy.optimize import minimize

class Optimizer:

    def __init__(self, universe, max_dv=5e3, initial_guess=[0.0, 0.0]):
        self.max_dv = float(max_dv)
        self.initial_guess = initial_guess #[guess_dvx, guess_dvy]
        self.optimization_attempts = 0
        self.optimization_attempts_distance = 0
        self.optimization_attempts_energy = 0
        #cache
        self.last_params = None
        self.last_y_full = None 
        #universe setup
        self.universe = universe
        self.sol1 = self.universe.run_until_aphelion()
        self.post_aphelion_y = self.sol1.y[:, -1].copy()
        #constraints
        self.set_constraints()
        self._best_energy_score = 2e10
        self._best_energy_dv = [0.0, 0.0]
        self.flyby_threshold_dynamic = 1e9


    def run_simulation_if_needed(self, params):
        if (self.last_params is not None) and (np.allclose(params, self.last_params, atol=1e-12, rtol=0)):
            return self.last_y_full, self.last_y_post
        dvx, dvy = params
        post_aphelion_y = self.post_aphelion_y.copy()
        post_aphelion_y[(self.universe.probe_index-1)*4+2] += dvx
        post_aphelion_y[(self.universe.probe_index-1)*4+3] += dvy

        #preciso dele pra pegar a distancia do pos aphelion
        self.sol2 = self.universe.run_after_aphelion(new_y0=post_aphelion_y)
        y_post = self.sol2.y
        y_full = np.concatenate((self.sol1.y, self.sol2.y), axis=1)
        self.last_params = params.copy()
        self.last_y_full = y_full
        self.last_y_post = y_post
        return y_full, y_post


    def objective_distance(self, params):
        dvx, dvy = params
        self.optimization_attempts_distance += 1

        _, y_post = self.run_simulation_if_needed(params)
        target_id = self.universe.target_index
        probe_id = self.universe.probe_index
        probe_all_x = y_post[(probe_id-1)*4]
        probe_all_y = y_post[(probe_id-1)*4 + 1]
        target_all_x = y_post[(target_id-1)*4]
        target_all_y = y_post[(target_id-1)*4 + 1]
        dists = np.sqrt((probe_all_x - target_all_x)**2 + (probe_all_y - target_all_y)**2)
        minimal_distance = np.min(dists)
        
        scale_ua = 1.49e11 # 1 UA
        #distance_weight = 1e10
        #distance_score = ((minimal_distance / scale_ua) ** 2) * distance_weight
        distance_score = minimal_distance / 1e6
        score = distance_score

        print(f"Distance Optimization")
        print(f"attempt: {self.optimization_attempts_distance}")
        print(f"dvx: {dvx}")
        print(f"dvy: {dvy}")
        print(f"Minimal distance: {minimal_distance}")
        print(f"Score: {score}")
        print("------------------")

        return score


    def objective_energy(self, params):
        dvx, dvy = params
        self.optimization_attempts_energy += 1
        y_full, _ = self.run_simulation_if_needed(params)
        final_y = y_full[:, -1]

        probe_final_x = final_y[(self.universe.probe_index-1)*4]
        probe_final_y = final_y[(self.universe.probe_index-1)*4+1]
        probe_final_vx = final_y[(self.universe.probe_index-1)*4+2]
        probe_finaL_vy = final_y[(self.universe.probe_index-1)*4+3]
        fixed_body_x = self.universe.celestial_bodies[self.universe.fixed_body_index].pos_x
        fixed_body_y = self.universe.celestial_bodies[self.universe.fixed_body_index].pos_y

        dx = probe_final_x - fixed_body_x
        dy = probe_final_y - fixed_body_y
        r_module = np.sqrt(dx**2 + dy**2)
        v_module = np.sqrt(probe_final_vx**2 + probe_finaL_vy**2)
        mu_fixed_body = self.universe.G * self.universe.celestial_bodies[self.universe.fixed_body_index].mass

        energy = (v_module**2)/2 - (mu_fixed_body / r_module)

        energy_weight = 1e-6
        score = - energy * energy_weight

        if score < self._best_energy_score:
            self._best_energy_score = score
            self._best_energy_dv = params.copy()


        print(f"Energy Optimization")
        print(f"attempt: {self.optimization_attempts_energy}")
        print(f"dvx: {dvx}")
        print(f"dvy: {dvy}")
        print(f"energy: {energy}")
        print(f"Score: {score}")
        print("------------------")

        return score


    def dv_constraint(self, params):
        return self.max_dv - np.linalg.norm(params)
    

    def target_collision_constraint(self, params):
        target_id = self.universe.target_index
        probe_id = self.universe.probe_index

        _, y_post = self.run_simulation_if_needed(params)

        probe_all_x = y_post[(probe_id-1)*4]
        probe_all_y = y_post[(probe_id-1)*4 + 1]
        target_all_x = y_post[(target_id-1)*4]
        target_all_y = y_post[(target_id-1)*4 + 1]
        dist = np.sqrt((probe_all_x - target_all_x)**2 + (probe_all_y - target_all_y)**2)
        minimal_distance_found = np.min(dist)
        target_radius = self.universe.celestial_bodies[self.universe.target_index].radius
        safety_margin = 2e6
        
        return minimal_distance_found - (target_radius + safety_margin)


    def max_distance_constraint(self, params):
        target_id = self.universe.target_index
        probe_id = self.universe.probe_index

        _, y_post = self.run_simulation_if_needed(params)

        probe_all_x = y_post[(probe_id-1)*4]
        probe_all_y = y_post[(probe_id-1)*4 + 1]
        target_all_x = y_post[(target_id-1)*4]
        target_all_y = y_post[(target_id-1)*4 + 1]
        dists = np.sqrt((probe_all_x - target_all_x)**2 + (probe_all_y - target_all_y)**2)
        minimal_distance = np.min(dists)

        return self.flyby_threshold_dynamic - minimal_distance


    def set_constraints(self):
        constraint_dv = {'type': 'ineq', 'fun': self.dv_constraint}
        constraint_target_collision = {'type': 'ineq', 'fun': self.target_collision_constraint}
        constraint_max_distance = {'type': 'ineq', 'fun': self.max_distance_constraint}

        self.constraints_distance = (constraint_dv, constraint_target_collision)
        self.constraints_energy = (constraint_dv, constraint_target_collision, constraint_max_distance)


    def optimize(self, maxiter): 
        bounds = [(-self.max_dv, self.max_dv), (-self.max_dv, self.max_dv)]

        method_distance = 'SLSQP'
        options_distance = {
            'maxiter': maxiter,
            'ftol': 1e-3,
            #'disp': True,
            'eps': 250.0,
        }

        method_energy = 'SLSQP'
        options_energy = {
            'maxiter': maxiter,
            'ftol': 1e-3,
            #'disp': True,
            'eps': 75.0,
        }

        result_distance = minimize(self.objective_distance, self.initial_guess, method=method_distance, bounds=bounds, constraints=self.constraints_distance, options=options_distance)
        self.distance_score = result_distance.fun
        self.best_dv_distance = result_distance.x

        print("======= DISTANCE OPTIMIZATION CONCLUDED ===========")
        print(f"Methods: distance-{method_distance}")
        print(f"local best deltaV (distance only): {self.best_dv_distance}")    
        print(f"distance optimization score: {self.distance_score}")

        _, y_post = self.run_simulation_if_needed(self.best_dv_distance)
        probe_id = self.universe.probe_index
        target_id = self.universe.target_index
        px = y_post[(probe_id-1)*4]
        py = y_post[(probe_id-1)*4 + 1]
        tx = y_post[(target_id-1)*4]
        ty = y_post[(target_id-1)*4 + 1]
        minimal_distance_phase1 = np.min(np.sqrt((px - tx)**2 + (py - ty)**2))
        #flyby_margin = 1e7
        flyby_margin = 5e9
        self.flyby_threshold_dynamic = minimal_distance_phase1 + flyby_margin
        print(f" Dynamic flyby threshold set to: {self.flyby_threshold_dynamic} meters")
        # ============================================================


        if not result_distance.success:
            print(f"[WARN] Distance optimization did not converge: {result_distance.message}")
            return [0.0, 0.0]

        print("============ ENERGY OPTIMIZATION STARTING ============") 

        result_energy = minimize(self.objective_energy, self.best_dv_distance, method=method_energy, bounds=bounds, constraints=self.constraints_energy, options=options_energy)
        self.energy_score = - self._best_energy_score
        self.best_dv = self._best_energy_dv


        print("================= SUMMARY =================")
        print(f"Methods: distance-{method_distance} ; energy-{method_energy}")
        print(f"local best deltaV (distance only): {self.best_dv_distance}")    
        print(f"best deltaV: {self.best_dv}")
        print(f"distance optimization score: {self.distance_score}")
        print(f"final score (energy score): {self.energy_score}")
        print("============ END OF OPTIMIZATION ============") 

        return self.best_dv