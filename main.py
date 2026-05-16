from slingshot.universe import Universe
from slingshot.optimizer import Optimizer
from slingshot.visualizer import Visualizer
from slingshot.config import load_config

#this could be in my parameter file, however i think it's easier to test here
#simulation_option = "PLOT" # Simple plot
#simulation_option = "UNTIL_APHELION" # Run the simulation until aphelion
simulation_option = "FULL_SIMULATION" # Simulate using the chosen deltaV
#simulation_option = "OPTIMIZED_SIMULATION" # Run optimizer and simulate

chosen_dv = [0.0, 0.0] # To use in the "FULL_SIMULATION" option

config = load_config()

if __name__ == "__main__":
    universe = Universe(config=config)
    visualizer = Visualizer(animation_interval=5, duration=float(config["simulation"]["duration"]))
    visualizer.set_celestial_bodies(universe.get_celestial_bodies(), universe.fixed_body_index)
    optimizer = Optimizer(universe=universe, max_dv=config["optimizer"]["max_dv"], initial_guess=config["optimizer"]["initial_guess"])

    match simulation_option:
        case "PLOT":
            simple_plot = visualizer.simple_plot()

        case "UNTIL_APHELION":
            sol = universe.run_until_aphelion()
            solution_array = sol.y.T
            visualizer.set_solution_array(solution_array)
            visualizer.animate()

        case "FULL_SIMULATION":
            solution_array = universe.simulate(chosen_dv)
            visualizer.set_solution_array(solution_array)
            visualizer.animate()

        case "OPTIMIZED_SIMULATION":
            dv = optimizer.optimize(maxiter=30)
            solution_array = universe.simulate(dv)
            visualizer.set_solution_array(solution_array)
            visualizer.animate()

        case _:
            print(f"simulation option {simulation_option} doesn't exist or contains a typo")


#TODO #1 melhorar a otimização, fazer em duas etapas
#TODO #2 fazer README, adiciona depois o gif da simulação
#TODO #?(qualquer momento) verificar typos