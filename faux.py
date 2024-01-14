import cantera as ct
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
import graphviz

# Auxiliary function used in the main code

class FauxFlameSolver:
    def __init__(self, yamlfile, phi, fuel, T0=300, P0=101325, width=0.02):
        self.yamlfile = yamlfile
        self.phi = phi
        self.fuel = fuel
        self.T0 = T0
        self.P0 = P0
        self.width = width
        self.flame = None

    def flame_speed_computator(self):
        # Define the flame parameters
        gas = ct.Solution(self.yamlfile)
        gas.set_equivalence_ratio(self.phi, self.fuel, {"O2": 1.0, "N2": 3.76})
        gas.TP = self.T0, self.P0

        # Create a Free Flame object
        self.flame = ct.FreeFlame(gas, width=self.width)

        # Solve the flame
        self.flame.solve(loglevel=0, refine_grid=True, auto=True)

        Su0 = self.flame.velocity[0]
        return Su0 * 100
    
    def temperature_analysis(self):
        if self.flame is None:
            raise ValueError("Flame not solved. Call flame_speed_computator first.")

        # Calculate the temperature gradient
        x_gradient = np.gradient(self.flame.T, self.flame.grid)

        # Find the x_value where the x_gradient is steepest
        max_gradient_index = np.argmax(np.abs(x_gradient))
        x_value = self.flame.grid[max_gradient_index]

        # Find the maximum temperature and calculate x_T01 and x_T99
        max_temperature = np.max(self.flame.T) - self.T0
        temperature_threshold_min = self.T0 + 0.1 * (max_temperature - self.T0)
        temperature_threshold_max = self.T0 + 0.9 * (max_temperature - self.T0)

        # Find indices where the temperature reaches 1% and 99% of the maximum temperature
        index_Tmin = np.argmin(np.abs(self.flame.T - temperature_threshold_min))
        index_Tmax = np.argmin(np.abs(self.flame.T - temperature_threshold_max))

        # Create an array between index_Tmin and index_Tmax (inclusive)
        indices_array = np.arange(index_Tmin, index_Tmax + 1)

        # Corresponding x values
        x_values_array = self.flame.grid[indices_array]

        # Corresponding x values
        x_Tmin = self.flame.grid[index_Tmin]
        x_Tmax = self.flame.grid[index_Tmax]

        Tmin = self.flame.T[index_Tmin]
        Tmax = self.flame.T[index_Tmax]

        # print(f"x_T01: {x_Tmin}, x_T99: {x_Tmax}")

        return x_Tmin, x_Tmax, Tmin, Tmax, max_gradient_index, x_value
    
    def plot_fuel_rates(self):
        if self.flame is None:
            raise ValueError("Flame not solved. Call flame_speed_computator first.")

        # Check if the specified fuel is in the species names
        if self.fuel in self.flame.gas.species_names:
            fuel_species_index = self.flame.gas.species_names.index(self.fuel)
            fuel_reaction_index = [index for index, reaction in enumerate(self.flame.gas.reaction_equations()) if self.fuel in reaction]

            # print(f"{self.fuel} species index: {fuel_species_index}")
            # print(f"{self.fuel} reaction indices: {fuel_reaction_index}")

            plt.figure()

            # Plot net rates of progress for each individual reaction involving the specified fuel
            for reaction_index in fuel_reaction_index:
                plt.plot(self.flame.grid, self.flame.net_rates_of_progress[reaction_index, :], label=f'Reaction {reaction_index+1}')

            plt.xlabel('Position (m)')
            plt.ylabel(f'Net Rate of Progress for {self.fuel} (mol/s)')
            plt.title(f'Net Rate of Progress Profile for Reactions Involving {self.fuel} Along the Flame')
            plt.legend()
            plt.show(block=False)
        else:
            print(f"Species {self.fuel} not found in the gas composition.")



    def plot_closest_point(self, x_Tmin, x_Tmax, Tmin, Tmax, x_value):
        if self.flame is None:
            raise ValueError("Flame not solved. Call flame_speed_computator first.")

        # Find the closest grid point to the specified x value
        closest_index = np.argmin(np.abs(self.flame.grid - x_value))
        closest_x = self.flame.grid[closest_index]
        closest_temperature = self.flame.T[closest_index]

        plt.figure()

        # Plot temperature vs grid
        plt.plot(self.flame.grid, self.flame.T)
        plt.xlabel('Position (m)')
        plt.ylabel('Temperature (K)')
        plt.title('Temperature Profile Along the Flame')
        plt.scatter([closest_x], [closest_temperature], color='red', label=f'Closest Point at x={closest_x:.4f}')
        plt.scatter([x_Tmin], [Tmin], color='red', label=f'Closest Point at x={x_Tmin:.4f}')
        plt.scatter([x_Tmax], [Tmax], color='red', label=f'Closest Point at x={x_Tmax:.4f}')

        plt.legend()
        plt.show(block=False)

    def graph_generator(self, idx):
        """
        Generate a directed graph representing the reaction network at a specific grid point.

        Parameters:
            - flame: Cantera FreeFlame object
            - idx: Index of the grid point

        Returns:
            - G: NetworkX DiGraph representing the reaction network
        """
        G = nx.DiGraph()

        # Add species as nodes to the graph
        species_names = self.flame.gas.species_names
        G.add_nodes_from(species_names)

        # Print the number of nodes in the original graph
        num_nodes = len(G.nodes)
        print(f'The number of nodes in the original graph is: {num_nodes}')

        # Iterate over reactions
        for i, reaction_equation in enumerate(self.flame.gas.reaction_equations()):
            # Initialize lists to store reactants and products
            reactants = []
            products = []

            # Extract reactants based on non-zero reactant stoichiometric coefficients
            reactant_indices = np.where(self.flame.gas.reactant_stoich_coeffs[:, i] != 0)[0]
            reactants.extend([species_names[k] for k in reactant_indices])

            # Extract products based on non-zero product stoichiometric coefficients
            product_indices = np.where(self.flame.gas.product_stoich_coeffs[:, i] != 0)[0]
            products.extend([species_names[k] for k in product_indices])

            for j, species in enumerate(species_names):
                if species in reactants:
                    for product_species in products:
                        edge_key = (species, product_species)
                        if G.has_edge(*edge_key):
                            G[species][product_species]['rate'] += self.flame.net_rates_of_progress[i, idx]/self.flame.gas.reactant_stoich_coeffs[j, i]
                        else:
                            G.add_edge(species, product_species, rate=self.flame.net_rates_of_progress[i, idx]/self.flame.gas.reactant_stoich_coeffs[j, i])
                elif species in products:
                    for reactant_species in reactants:
                        edge_key = (reactant_species, species)
                        if G.has_edge(*edge_key):
                            G[reactant_species][species]['rate'] += self.flame.net_rates_of_progress[i, idx]/self.flame.gas.product_stoich_coeffs[j, i]
                        else:
                            G.add_edge(reactant_species, species, rate= self.flame.net_rates_of_progress[i, idx]/self.flame.gas.product_stoich_coeffs[j, i])

        # Set layout for better visualization
        pos = nx.spring_layout(G, k=0.3)  # Adjust 'k' for better spacing

        # Draw the graph
        edge_labels = nx.get_edge_attributes(G, 'rate')
        filtered_edges = {edge: rate for edge, rate in edge_labels.items() if rate > 1e-3}  # Filter edges based on threshold

        plt.figure()

        # Add title to the graph
        plt.title(f'Reaction Network Graph at x = {self.flame.grid[idx]}')

        nx.draw(G, pos, with_labels=True, font_weight='bold', node_size=1000, font_size=8, edge_color='b', width=0.5)
        nx.draw_networkx_edge_labels(G, pos, edge_labels=filtered_edges, font_color='red')

        plt.show(block=False)

        # Create a graphviz graph
        dot = graphviz.Digraph(comment='Reaction Network Graph')

        # Add nodes to the graphviz graph
        for node in G.nodes():
            dot.node(str(node))

        # Add edges to the graphviz graph
        for edge, rate in edge_labels.items():
            if rate > 1e-3:
                dot.edge(str(edge[0]), str(edge[1]), label=str(rate), color='red')

        # Save the graphviz graph to a file and render it
        dot.render('graphviz_output', format='png', cleanup=True)

        return G, num_nodes

    def graph_reducer(self, G, threshold):
        """
        Visualize the modified graph after removing edges below a certain threshold.

        Parameters:
            - G: NetworkX DiGraph representing the graph
            - threshold: Threshold value for filtering edges

        Returns:
            - None (displays the graph plot)
        """
        # List of species names that should not be removed
        protected_species = [self.fuel, 'O2', 'N2']  # Add species names as needed

        # Create a copy of the graph to iterate over (to avoid modifying the graph while iterating)
        edges_to_remove = [(u, v) for u, v, d in G.edges(data=True) if abs(d['rate']) < threshold and u not in protected_species and v not in protected_species]

        # Remove edges below the threshold
        G.remove_edges_from(edges_to_remove)

        # Identify isolated nodes and remove them
        isolated_nodes = [node for node in G.nodes() if G.degree(node) == 0 and node not in protected_species]
        G.remove_nodes_from(isolated_nodes)


        plt.figure()

        # Visualize the modified graph
        pos = nx.spring_layout(G, k=0.3)  # Adjust 'k' for better spacing

        nx.draw(G, pos, with_labels=True, font_weight='bold', node_size=1000, font_size=8, edge_color='b', width=0.5)
        filtered_edges = {edge: rate for edge, rate in nx.get_edge_attributes(G, 'rate').items() if rate > threshold}  # Filter edges based on threshold
        nx.draw_networkx_edge_labels(G, pos, edge_labels=filtered_edges, font_color='red')

        # Print the number of nodes in the modified graph
        num_nodes = len(G.nodes)
        print(f'The number of nodes in the modified graph is: {num_nodes}')

        plt.show(block=False)

        return num_nodes

    def mechanism_writer(self, G, idx, threshold):
        """
        Write a reduced mechanism based on net rates of progress above a certain threshold.

        Parameters:
            - G: NetworkX DiGraph representing the graph
            - flame: Cantera FreeFlame object
            - threshold: Threshold value for selecting reactions

        Returns:
            - None (writes the reduced mechanism to a YAML file)
        """
        # Get the reactions with net rates of progress above the threshold
        selected_reactions = [reaction for i, reaction in enumerate(self.flame.gas.reactions()) if self.flame.net_rates_of_progress[i, idx] > threshold]

        # Collect species names from selected reactions
        species_names = set()
        for reaction in selected_reactions:
            species_names.update(reaction.reactants)
            species_names.update(reaction.products)

        # Get the species objects
        species = [self.flame.gas.species(name) for name in dict(G.nodes)]

        # Create the new reduced mechanism
        gas2 = ct.Solution(thermo='ideal-gas', kinetics='gas', transport_model="mixture-averaged", 
                           species=species, reactions=selected_reactions)

        # Save the reduced mechanism for later use
        gas2.write_yaml(f"Mech_fuel-{self.fuel}-spec{len(gas2.species_names)}-reac{len(gas2.reactions())}-reduced.yaml")

        return gas2
