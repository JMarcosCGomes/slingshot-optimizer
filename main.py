from slingshot.universe import Universe
from slingshot.optimizer import Optimizer
from slingshot.visualizer import Visualizer
from slingshot.config import load_config

config = load_config()

if __name__ == "__main__":
    universe = Universe(config=config)
    visualizer = Visualizer(animation_interval=5, duration=float(config["simulation"]["duration"]))
    visualizer.set_celestial_bodies(universe.get_celestial_bodies(), universe.fixed_body_index)
    optimizer = Optimizer(universe=universe, max_dv=config["optimizer"]["max_dv"], initial_guess=config["optimizer"]["initial_guess"])

    #Mostra as posições iniciais, caso queira ver a sonda inicialmente na frente da terra
    #simple_plot = visualizer.simple_plot()

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
    dv = optimizer.optimize(maxiter=30)
    solution_array = universe.simulate_optimized(dv)
    visualizer.set_solution_array(solution_array)
    visualizer.animate()



#TODO s
#TODO #0 usar git issues

#TODO #1 faz um main.py mais sofisticado e menos poluido

#TODO #1 fazer README, adiciona depois o gif da simulação

#TODO #3 melhorar a otimização, fazer em duas etapas

#TODO #?(qualquer momento) verificar typos
