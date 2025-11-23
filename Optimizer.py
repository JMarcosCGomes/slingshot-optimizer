from Universo import Universo
import numpy as np
from scipy.optimize import minimize

class Optimizer:

    def __init__(self, max_dv=5e3, initial_guess=[0.0, 0.0]):
        self.max_dv = max_dv #integer / float 
        self.initial_guess = initial_guess #[guess_dvx, guess_dvy]
        self.constraint = ({'type': 'ineq', 'fun': self.dv_constraint},) #maybe an "set_cons?"
        self.the_fixed_part()
        self.optimization_attempts = 0


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
    

    def optimize(self, maxiter=120, ftol=1e-3):
        #i think i can test different methods 
        method = 'SLSQP'
        #amanha faço isso bonitinho, nao e possivel
        bounds = [(-self.max_dv, self.max_dv), (-self.max_dv, self.max_dv)]
        options = {
            'maxiter': maxiter,
            'ftol': ftol,
            'disp': True,
        }
        
        res = minimize(self.objective, self.initial_guess, method=method, bounds=bounds, constraints=self.constraint, options=options)
        self.best_dv = res.x #dvx, dvy
        final_velocity = -res.fun
        print(self.best_dv)
        print(final_velocity)
        return self.best_dv


