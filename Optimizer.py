from Universo import Universo
import numpy as np
from scipy.optimize import minimize

class Optimizer:

    def __init__(self, max_dv=5e3, initial_guess=[0.0, 0.0]):
        self.max_dv = max_dv #integer / float 
        self.initial_guess = initial_guess #[guess_dvx, guess_dvy]
        self.set_constraints()
        self.the_fixed_part()
        self.optimization_attempts = 0
        self.last_params = None
        self.last_y_full = None #i think i can substitute for sol2


    def the_fixed_part(self):
        self.universo = Universo()
        self.probe_vx_index, self.probe_vy_index = self.universo.get_probe_v_indexes()
        self.sol1 = self.universo.run_until_aphelion()
        


    #objective function, to optimize the speed
    def objective(self, params):
        dvx, dvy = params
        post_aphelion_y = self.sol1.y[:, -1].copy()
        post_aphelion_y[self.probe_vx_index] += dvx
        post_aphelion_y[self.probe_vy_index] += dvy

        sol2 = self.universo.run_after_aphelion(new_y0=post_aphelion_y)
        y_full = np.concatenate((self.sol1.y, sol2.y), axis=1)
        final_y = y_full[:, -1].copy()
        probe_final_vx = final_y[self.probe_vx_index]
        probe_finaL_vy = final_y[self.probe_vy_index]
        final_velocity = np.sqrt(probe_final_vx**2 + probe_finaL_vy**2)

        #CACHE TO SUN CONSTRAINT
        self.last_params = params.copy()
        self.last_y_full = y_full
        # if its taking too long, i can change to sol2 part

        self.optimization_attempts += 1
        print(f"attempt: {self.optimization_attempts}")
        print(f"dvx: {dvx}")
        print(f"dvy: {dvy}")
        print(f"final_v: {final_velocity}")
        print("------------------")
        #as far as I know, scipy can only minimize
        #I can use as score the final_velocity OR final_velocity - penalization_to_dv_use
        return -final_velocity


    def dv_constraint(self, params):
        return self.max_dv - (abs(params[0]) + abs(params[1]))
    

    def sun_distance_constraint(self, params):

        if (self.last_params is None):
            dvx, dvy = params
            post_aphelion_y = self.sol1.y[:, -1].copy()
            post_aphelion_y[self.probe_vx_index] += dvx
            post_aphelion_y[self.probe_vy_index] += dvy
            sol2 = self.universo.run_after_aphelion(new_y0=post_aphelion_y)
            self.last_y_full = np.concatenate((self.sol1.y, sol2.y), axis=1)
            self.last_params = params.copy()

        probe_x_index = self.probe_vx_index - 2
        probe_y_index = self.probe_vy_index - 2
        probe_all_x_positions = self.last_y_full[probe_x_index]
        probe_all_y_positions = self.last_y_full[probe_y_index]
        dist = np.sqrt(probe_all_x_positions**2 + probe_all_y_positions**2)
        minimal_distance_found = np.min(dist)
        minimal_distance_required = 0.31 * 1.496e11 #or 0.25 * 
        #if minimal_distance_found is lower than minimal_distance required. The condition isn't satisfied; mdf >= mdr --> mdf - mdr >= 0
        return minimal_distance_found - minimal_distance_required


    def set_constraints(self):
        #self.constraint = ({'type': 'ineq', 'fun': self.dv_constraint},) 
        constraint_dv = {'type': 'ineq', 'fun': self.dv_constraint}
        constraint_sun_dist = {'type': 'ineq', 'fun': self.sun_distance_constraint}
        self.constraints = (constraint_dv, constraint_sun_dist)


    def optimize(self, maxiter=120, ftol=1e-3): 
        #method = 'SLSQP'
        #methods_to_test = ['SLSQP', 'trust-constr']
        methods_to_test = ['SLSQP']
        
        bounds = [(-self.max_dv, self.max_dv), (-self.max_dv, self.max_dv)]
        options_slsqp = {
            'maxiter': maxiter,
            'ftol': ftol,
            'disp': True,
        }

        options_trustconstr = {
            "maxiter": maxiter,
            "verbose": 0,
            "gtol": 1e-4,
        }

        results = {}
        
        for method in methods_to_test:

            if method == 'SLSQP':
                options = options_slsqp
            elif method == 'trust-constr':
                options = options_trustconstr

            print("|||||||||||||||||||")
            print(f"METHOD: {method}")
            print("VVVVVVVVVVVVVVVVVVV")

            res = minimize(self.objective, self.initial_guess, method=method, bounds=bounds, constraints=self.constraints, options=options)
            final_velocity = -res.fun
            results[method] = (res.x, final_velocity)

            print("==============================================================")
            print(f"METHOD: {method} | best_dv = {res.x}, final_velocity = {final_velocity}")
            print("==============================================================")

            self.optimization_attempts = 0

        print("================= SUMMARY =================")
        for method, (dv, fv) in results.items():
            print(f"{method}: dv={dv}, final_velocity={fv}")
        print("================= BEST METHOD =================") 
        best_method = max(results.items(), key=lambda kv: kv[1][1]) #[1][1] eh final_velocity
        print(f"BEST METHOD FOUND: {best_method[0]}")
        
        self.best_dv = best_method[1][0]
        return self.best_dv


"""
self.best_dv = res.x #dvx, dvy
final_velocity = -res.fun
print(self.best_dv)
print(final_velocity)
return self.best_dv
"""

