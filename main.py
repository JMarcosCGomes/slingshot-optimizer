from Universo import Universo
from Optimizer import Optimizer

PLANET_ANGLE_DEG = -120
MAX_DV = 4e3
day_in_seconds = 86400
MAX_STEP_SOLVEIVP = day_in_seconds / 4

if __name__ == "__main__":
    universo = Universo(planet_angle_deg=PLANET_ANGLE_DEG, max_step=MAX_STEP_SOLVEIVP, intervalo_animacao=5)
    optimizer = Optimizer(planet_angle_deg=PLANET_ANGLE_DEG, max_step=MAX_STEP_SOLVEIVP, max_dv=MAX_DV, initial_guess=[0.0, 0.0])

    # so pra ver as posicoes
    #teste = universo.simple_plot()

    #to visualize the "aphelion event", with trace
    #sol = universo.run_until_aphelion(add_trace=True)
    #solucao_array = sol.y.T
    #universo.animar(solucao_array)

    #without optimizing anything
    #solucao = universo.simular_simple()
    #universo.animar(solucao)

    #with "manual" deltav, to visualize
    #dv = [0.0, 0.0]
    #dv = [-873, -2181]
    #sol = universo.simular_optimized(dv)
    #universo.animar(sol)


    #with optimization, "reduce" max_steps from blabla/4 to blabla
    dv = optimizer.optimize(maxiter=30)
    sol = universo.simular_optimized(dv)
    universo.animar(sol)


"""
MAX_STEP_SOLVEIVP = day_in_seconds / 4
PLANET_ANGLE_DEG = -120
'eps': 5.0

attempt: 56
dvx: -873.6464240618799
dvy: -2181.4687480351304
energy: -111468006.49790066
Minimal distance: 22844542.860160097
Score: 402.3272489236322
------------------
================= SUMMARY =================
Method: SLSQP
best deltaV: [ -873.64642406 -2181.46874804]
final score (energy - log(minimal_distance): -402.3272489236322
============ END OF OPTIMIZATION ============
"""