import cantera as ct
import networkx as nx
import matplotlib.pyplot as plt
from faux import FauxFlameSolver

# Edge reduction threshold on the net rate of progress

thresholds = [1e-1, 1, 1e1]
# iterations = 5

# Assuming G is your graph
for threshold in thresholds:  # Add more threshold values as needed
    try:

        yamlfile = 'Peters_C3H8.yaml'
        fuel = 'C3H8'
        phi = 1.0

        # Define the flame parameters
        faux_solver = FauxFlameSolver(yamlfile, phi, fuel)

        flame_speed = faux_solver.flame_speed_computator()
        print(f"Flame Speed is: {flame_speed:.2f} cm/s")

        x_Tmin, x_Tmax, Tmin, Tmax, max_gradient_index, x_value = faux_solver.temperature_analysis()

        faux_solver.plot_fuel_rates()
        faux_solver.plot_closest_point(x_Tmin, x_Tmax, Tmin, Tmax, x_value)

        idx = max_gradient_index
        G, num_nodes = faux_solver.graph_generator(idx)

        # for iter in range(iterations):

        #     if iter >=2:
        #         threshold = threshold*10

        new_num_nodes = faux_solver.graph_reducer(G, threshold)

        if new_num_nodes < num_nodes:

            gas2 = faux_solver.mechanism_writer(G, idx, threshold)

            yamlfile = f"Mech_fuel-{fuel}-spec{len(gas2.species_names)}-reac{len(gas2.reactions())}-reduced.yaml"
            phi = 1.0

            # Define the flame parameters
            faux_solver = FauxFlameSolver(yamlfile, phi, fuel)
            flame_speed = faux_solver.flame_speed_computator()

            print(f"Flame Speed is: {flame_speed:.2f} cm/s \n")

            x_Tmin, x_Tmax, Tmin, Tmax, max_gradient_index, x_value = faux_solver.temperature_analysis()

            faux_solver.plot_fuel_rates()
            faux_solver.plot_closest_point(x_Tmin, x_Tmax, Tmin, Tmax, x_value)

    except Exception as e:
        print(f"Error processing with threshold {threshold}: {e}")

    input("Press Enter to close all plots...")
    plt.close('all')

