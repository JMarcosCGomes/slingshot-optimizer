from slingshot.universe import Universe
from slingshot.optimizer import Optimizer
from slingshot.visualizer import Visualizer
from slingshot.config import load_config

config = load_config()

DURATION = 4e8

if __name__ == "__main__":
    universe = Universe(config=config)
    visualizer = Visualizer(animation_interval=5, duration=DURATION)
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



#TODO s , não ordenados.
#TODO #0 usar git issues

#TODO #1 fazer README, adiciona depois o gif da simulação

#TODO #2 renomear alguns métodos pra snake case apenas

#TODO #3 faz um main.py mais sofisticado e menos poluido (enquanto não está usando CLI argparse ou arquivo de parametro pra escolher o que vai fazer)

#TODO #4 trocar a "role" do parametro de generic pra planet, e faz planet pra target talvez

#TODO #5 talvez criar um arquivo de constantes (G, ONE_YEAR_IN_SECONDS)

#TODO #6 adicionar type hints nas funções

#TODO #?(qualquer momento) verificar typos

#TODO #-2 melhorar a otimização, fazer em duas etapas
#TODO #? talvez dê pra fazer 1 otimização no afélio (2 etapas) e depois faço outra otimização no meio do caminho (mid-course correction), acho que melhor não
## só tem que definir bem o evento pra isso funcionar (ao entrar na esfera de influencia SOI talvez)

#TODO #-1 ver quais tipos de testes unitários seriam interessantes de adicionar

