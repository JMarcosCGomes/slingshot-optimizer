from Universo import Universo
from Optimizer import Optimizer

PLANET_ANGLE_DEG = -120
MAX_DV = 4e3
INITIAL_GUESS = [0.0, 0.0]

day_in_seconds = 86400
MAX_STEP_SOLVEIVP = day_in_seconds / 4

if __name__ == "__main__":
    universo = Universo(planet_angle_deg=PLANET_ANGLE_DEG, max_step=MAX_STEP_SOLVEIVP, intervalo_animacao=5)
    optimizer = Optimizer(planet_angle_deg=PLANET_ANGLE_DEG, max_step=MAX_STEP_SOLVEIVP, max_dv=MAX_DV, initial_guess=INITIAL_GUESS)

    #Mostra as posições iniciais, caso queira ver a sonda inicialmente na frente da terra
    #teste = universo.simple_plot()

    #Se quiser ver quando acontece o afélio
    #sol = universo.run_until_aphelion(add_trace=True)
    #solucao_array = sol.y.T
    #universo.animar(solucao_array)

    #Apenas a simulação, sem deltaV nem otimização
    #solucao = universo.simular_simple()
    #universo.animar(solucao)

    #Caso queira visualizar novamente, ajuste dv para os valores encontramos numa otimização
    #dv = [0.0, 0.0]
    #dv = [-1520, -2073]
    #sol = universo.simular_optimized(dv)
    #universo.animar(sol)

    #Otimizando e depois simulando
    dv = optimizer.optimize(maxiter=30)
    sol = universo.simular_optimized(dv)
    universo.animar(sol)
