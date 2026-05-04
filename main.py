from Universe import Universe
from Optimizer import Optimizer
from Visualizer import Visualizer
from load_params import load_config

config = load_config()

DURATION = 4e8

if __name__ == "__main__":
    universe = Universe(config=config)
    visualizer = Visualizer(animation_interval=5, duration=DURATION)
    visualizer.set_celestial_bodies(universe.get_celestial_bodies(), universe.fixed_body_index)
    optimizer = Optimizer(universe=universe, max_dv=config["optimizer"]["max_dv"], initial_guess=config["optimizer"]["initial_guess"])

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

#TODO #2 trocar a "role" do parametro de generic pra planet, e faz planet pra target talvez

#TODO #3 melhorar a otimização, fazer em duas etapas
#TODO #? talvez dê pra fazer 1 otimização no afélio (2 etapas) e depois faço outra otimização no meio do caminho (mid-course correction)
## só tem que definir bem o evento pra isso funcionar (ao entrar na esfera de influencia SOI talvez)

#TODO #4 ver quais tipos de testes unitários seriam interessantes de adicionar