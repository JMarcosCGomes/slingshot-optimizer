from Universo import Universo
from Optimizer import Optimizer


if __name__ == "__main__":
    universo = Universo(planet_angle_deg=-125, intervalo_animacao=5)
    optimizer = Optimizer(max_dv=4e3, initial_guess=[0.0, 0.0])

    # so pra ver as posicoes
    #teste = universo.simple_plot()

    #without optimizing anything
    #solucao = universo.simular_simple()
    #universo.animar(solucao)

    #with "manual" optimization, to visualize
    #dv = [0.0, 0.0]
    dv = [4868, -128]
    sol = universo.simular_optimized(dv)
    universo.animar(sol)


    #with optimization, "reduce" max_steps from blabla/4 to blabla
    #dv = optimizer.optimize(maxiter=40)
    #sol = universo.simular_optimized(dv)
    #universo.animar(sol)

    """
    attempt: 78
    dvx: 4868.354105154505
    dvy: -128.5439909272095
    final_v: 49786.02398612554
    """