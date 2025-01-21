# Authors: Mohammed Arab, Ben Cacic, Andrew Phan
import pandas as pd
import evaluation
import time
import numpy as np
import matplotlib.pyplot as plt
from genetic_algorithm import run_genetic_algorithm#, gen_sample_route
import pygad


def find_route(locations: pd.DataFrame, num_genes, num_cities, sol_per_pop, initial_population) -> list | np.ndarray:
    """
    Function that finds a good route through the provided travelling salesman problem
    :param locations: The x-y coordinates for each location in the TSP. Should be a pandas DataFrame
                        (as returned by pd.read_csv())
    :return: The route through the TSP. Should be a list or array of location indexes, in the desired order.
             The first entry should be the origin city.
             **DO NOT** include the origin city at the end of the list too - the route will be assumed to return to the
             origin city after the list's last entry
    """

    # locations = locations.reset_index(drop=True)
    best_route = run_genetic_algorithm(locations=locations,
                                       num_genes=num_genes,
                                       num_cities=num_cities,
                                       sol_per_pop=sol_per_pop,
                                       initial_population=initial_population)
    return best_route



if __name__ == '__main__':

    # here's an example of how to call find_route (and time it)
    tsp = pd.read_csv('./3625-assignment-1-team-main/data/250a.csv', index_col=0)

    tsp = tsp.reset_index(drop=True)

    num_cities = tsp.shape[0]
    num_genes = num_cities

    sol_per_pop = 40

    initial_population = []
    route = list(range(num_cities))
    initial_population = []
    for _ in range(sol_per_pop):
        initial_population.append(route)

    initial_population = np.array(initial_population)

    start_time = time.time()
    route = find_route(locations=tsp, num_genes=num_genes, num_cities=num_cities, sol_per_pop=sol_per_pop,
                       initial_population=initial_population)
    elapsed = time.time() - start_time

    # use the provided distance calculation function to measure route distance
    distance = evaluation.measure_distance(tsp, route)
    print(f'found a route with distance {distance:.2f} in {elapsed:.4f}s')

    # plot route
    evaluation.plot_route(tsp, route)
    plt.title(f'distance={distance:.2f}')
    plt.xticks([])
    plt.yticks([])
    plt.show()

