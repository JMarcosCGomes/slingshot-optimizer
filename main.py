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

    #without optimizing anything
    #solucao = universo.simular_simple()
    #universo.animar(solucao)

    #with "manual" deltav, to visualize
    #dv = [0.0, 0.0]
    #dv = [2985, 765]
    #sol = universo.simular_optimized(dv)
    #universo.animar(sol)


    #with optimization, "reduce" max_steps from blabla/4 to blabla
    dv = optimizer.optimize(maxiter=30)
    sol = universo.simular_optimized(dv)
    universo.animar(sol)


"""
MAX_STEP_SOLVEIVP = day_in_seconds / 4
PLANET_ANGLE_DEG = -120
'eps': 4.0, (obs: esp 5 improved)

attempt: 97
dvx: -300.6846277367216
dvy: -2235.1855204010185
energy: -12930885.513949633
Minimal distance: 96280125.26643053
Score: 504.6992088376943
"""