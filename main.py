from Universo import Universo
from Optimizer import Optimizer

PLANET_ANGLE_DEG = -50
MAX_DV = 4e3

if __name__ == "__main__":
    universo = Universo(planet_angle_deg=PLANET_ANGLE_DEG, intervalo_animacao=5)
    optimizer = Optimizer(planet_angle_deg=PLANET_ANGLE_DEG, max_dv=MAX_DV, initial_guess=[0.0, 0.0])

    # so pra ver as posicoes
    #teste = universo.simple_plot()

    #without optimizing anything
    #solucao = universo.simular_simple()
    #universo.animar(solucao)

    #with "manual" optimization, to visualize
    #dv = [0.0, 0.0]
    #dv = [2985, 765]
    #sol = universo.simular_optimized(dv)
    #universo.animar(sol)


    #with optimization, "reduce" max_steps from blabla/4 to blabla
    dv = optimizer.optimize(maxiter=30)
    sol = universo.simular_optimized(dv)
    universo.animar(sol)


"""
Universo - get_solveivp_params  - max_step = 86400
PLANET_ANGLE_DEG = -50 

attempt: 89
dvx: 2985.0822615220627
dvy: 765.9270087066627
energy: 285229758.7288369
Minimal distance: 18047242.278514884
Report: OPTIMIZING SLINGSHOT | Score: -194.99354733626245
"""

#TODO URGENTE, CONSERTA MAX_STEP, bota que nem max_dv e planet_angledeg, passa como parametro pro universo (se n tiver, max_step fica 1 dia mesmo)
#TODO Menos urgente, aumenta o peso, pelos logs recentes o otimizador toda hora ta saindo de perto da terra
#TODO Igualmente urgente, acho que é o eps mt alto, acho que os pesos ja fazem o bastante
#TODO Testar métodos diferentes, mesmo achando que o SLSQP é melhor pra esse caso