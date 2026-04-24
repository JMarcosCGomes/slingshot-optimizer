from Universe import Universe
from Optimizer import Optimizer
from Visualizer import Visualizer

PLANET_ANGLE_DEG = -120
MAX_DV = 4e3
INITIAL_GUESS = [0.0, 0.0]

DURATION = 4e8

day_in_seconds = 86400
MAX_STEP_SOLVEIVP = day_in_seconds / 4

if __name__ == "__main__":
    universe = Universe(planet_angle_deg=PLANET_ANGLE_DEG, max_step=MAX_STEP_SOLVEIVP, duration=DURATION)
    visualizer = Visualizer(animation_interval=5, duration=DURATION)
    visualizer.set_celestial_bodies(universe.get_celestial_bodies(), universe.fixed_body_index)
    optimizer = Optimizer(planet_angle_deg=PLANET_ANGLE_DEG, max_step=MAX_STEP_SOLVEIVP, max_dv=MAX_DV, initial_guess=INITIAL_GUESS)

    #Mostra as posições iniciais, caso queira ver a sonda inicialmente na frente da terra
    simple_plot = visualizer.simple_plot()

    #Se quiser ver quando acontece o afélio
    #sol = universe.run_until_aphelion()
    #solution_array = sol.y.T
    #visualizer.set_solution_array(solution_array)
    #visualizer.animate()

    #Apenas a simulação, sem deltaV nem otimização
    #solution_array = universe.simulate_simple()
    #visualizer.set_solution_array(solution_array)
    #visualizer.animate()

    #Caso queira visualizar novamente, ajuste dv para os valores encontramos numa otimização
    #dv = [0.0, 0.0]
    #dv = [1234, -4321]
    #solution_array = universe.simulate_optimized(dv)
    #visualizer.set_solution_array(solution_array)
    #visualizer.animate()

    #Otimizando e depois simulando
    #dv = optimizer.optimize(maxiter=30)
    #solution_array = universe.simulate_optimized(dv)
    #visualizer.set_solution_array(solution_array)
    #visualizer.animate()



#TODO s , não ordenados.

#TODO #1 fazer README

#TODO #2 Talvez criar um arquivo de parametros com as infos dos corpos celestes
## Além disso é possível depois colocar opção pra escolher os cb que ficam na sim, qual cb vai ter o probe por ex
#TODO #3 Melhorar a iteração do create_celestial_bodies
#TODO #4 Fazer uma função pra calcular a velocidade do probe, até acho que ajuda pra melhorar a iteração

#TODO #5 melhorar a otimização, fazer em duas etapas
#TODO #? talvez dê pra fazer 1 otimização no afélio (2 etapas) e depois faço outra otimização no meio do caminho (mid-course correction)
## só tem que definir bem o evento pra isso funcionar (ao entrar na esfera de influencia SOI talvez)

#TODO #6 ver quais tipos de testes unitários seriam interessantes de adicionar