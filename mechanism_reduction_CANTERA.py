"""
A simplistic approach to mechanism reduction which demonstrates Cantera's
features for dynamically manipulating chemical mechanisms.

Here, we use the full GRI 3.0 mechanism to simulate adiabatic, constant pressure
ignition of a lean methane/air mixture. We track the maximum reaction rates for
each reaction to determine which reactions are the most important, according to
a simple metric based on the relative net reaction rate.

We then create a sequence of reduced mechanisms including only the top reactions
and the associated species, and run the simulations again with these mechanisms
to see whether the reduced mechanisms with a certain number of species are able
to adequately simulate the ignition delay problem.

Requires: cantera >= 2.6.0, matplotlib >= 2.0
Keywords: kinetics, combustion, reactor network, editing mechanisms, ignition delay,
          plotting
"""

import cantera as ct
import numpy as np
from math import exp
import matplotlib.pyplot as plt
from faux import FauxFlameSolver

def remove_lines_with_string(file_path, string_to_remove):
        with open(file_path, 'r') as file:
            lines = file.readlines()

        # Remove lines containing the specified string
        lines = [line for line in lines if string_to_remove not in line]

        with open(file_path, 'w') as file:
            file.writelines(lines)


gas = ct.Solution('gri30.yaml')
initial_state = 1200, 5 * ct.one_atm, 'CH4:0.35, O2:1.0, N2:3.76'

flame_speeds = []

yamlfile = 'gri30.yaml'
fuel = 'CH4'
phi = 1.0

# Define the flame parameters
faux_solver = FauxFlameSolver(yamlfile, phi, fuel)

base_flame_speed = faux_solver.flame_speed_computator()
print(f"Flame Speed of the original mechanism is: {base_flame_speed:.2f} cm/s")

# Run a simulation with the full mechanism
gas.TPX = initial_state
r = ct.IdealGasConstPressureReactor(gas)
sim = ct.ReactorNet([r])

tt = []
TT = []
t = 0.0
# Rmax is the maximum relative reaction rate at any timestep
Rmax = np.zeros(gas.n_reactions)
while t < 0.02:
    t = sim.step()
    tt.append(1000 * t)
    TT.append(r.T)
    rnet = abs(gas.net_rates_of_progress)
    rnet /= max(rnet)
    Rmax = np.maximum(Rmax, rnet)

# plt.plot(tt, TT, label='K=53, R=325', color='k', lw=3, zorder=100)

# Get the reaction objects, and sort them so the most active reactions are first
R = sorted(zip(Rmax, gas.reactions()), key=lambda x: -x[0])

# Test reduced mechanisms with different numbers of reactions
# C = plt.cm.winter(np.linspace(0, 1, 11))

num_reactions = [50, 75, 100, 125, 150, 175, 200, 225, 250, 275, 300]

for i, N in enumerate(num_reactions):
    # Get the N most active reactions
    reactions = [r[1] for r in R[:N]]

    # find the species involved in these reactions. At a minimum, include all
    # species in the reactant mixture
    species_names = {'N2', 'CH4', 'O2'}
    for reaction in reactions:
        species_names.update(reaction.reactants)
        species_names.update(reaction.products)




    # Get the species objects
    species = [gas.species(name) for name in species_names]

    
    if N == 50 or N == 75 or N == 100 or N == 125 or N == 150 or N == 175:
        new_reac = ct.Reaction(equation=reactions[18].equation, 
                            rate={"A": reactions[18].rate.pre_exponential_factor, 
                                    "b": reactions[18].rate.temperature_exponent,
                                    "Ea": reactions[18].rate.activation_energy})
        reactions[18] = new_reac

    if N == 50 or N == 75 or N == 100:
        new_reac = ct.Reaction(equation=reactions[40].equation, 
                            rate={"A": reactions[40].rate.pre_exponential_factor, 
                                    "b": reactions[40].rate.temperature_exponent,
                                    "Ea": reactions[40].rate.activation_energy})

        reactions[40] = new_reac

    # create the new reduced mechanism
    gas2 = ct.Solution(thermo='ideal-gas', kinetics='gas', transport_model="mixture-averaged", 
                           species=species, reactions=reactions)


    # save the reduced mechanism for later use
    yaml_data = gas2.write_yaml(f"gri30-reduced-{N}-reaction-{len(species)}.yaml")

    # Let's insert a check here: it should see if the reaction is already there 
    # and toggle the duplicate field
        
    # WORK IN PROGRESS

    

    # remove_lines_with_string("gri30-reduced-{}-reaction.yaml".format(N), 'duplicate')


    try:

        yamlfile = f"gri30-reduced-{N}-reaction-{len(species)}.yaml"
        fuel = 'CH4'
        phi = 1.0

        # Define the flame parameters
        faux_solver = FauxFlameSolver(yamlfile, phi, fuel)

        flame_speed = faux_solver.flame_speed_computator()
        print(f"Flame Speed of the modified mechanism with {N} reactions is: {flame_speed:.2f} cm/s")
        flame_speeds.append(flame_speed)

    except Exception as e:
        print(f"Error processing with {N} reactions: {e}")

    # Re-run the ignition problem with the reduced mechanism
    gas2.TPX = initial_state
    r = ct.IdealGasConstPressureReactor(gas2)
    sim = ct.ReactorNet([r])

    t = 0.0

    tt = []
    TT = []
    while t < 0.02:
        t = sim.step()
        tt.append(1000 * t)
        TT.append(r.T)

    # plt.plot(tt, TT, lw=2, color=C[i],
    #          label='K={0}, R={1}'.format(gas2.n_species, N))
    # plt.xlabel('Time (ms)')
    # plt.ylabel('Temperature (K)')
    # plt.legend(loc='upper left')
    # plt.title('Reduced mechanism ignition delay times\n'
    #           'K: number of species; R: number of reactions')
    # plt.xlim(0, 20)

    
plt.plot(num_reactions, flame_speeds)
plt.axhline(y=base_flame_speed, color='r', linestyle='--', label=f'Y = {base_flame_speed}')
plt.xlabel('Number of reactions')
plt.ylabel('Flame speeds (cm/s)')
plt.title('Reduced mechanism flame speeds')
plt.tight_layout()
plt.show()
