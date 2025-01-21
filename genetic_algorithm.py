# Authors: Mohammed Arab, Ben Cacic, Andrew Phan
import pygad
import numpy as np
import pandas as pd
import time
from evaluation import measure_distance

def selection_func(fitness, num_parents, ga_instance):
    """
    Function that selects the individual solutions that will be used to create the next
    generation. This uses a tourbnament to determine which offspring wins where k is the
    number of participants in the tournament. High pressure means the next generation will
    be less diverse, and composed mostly of the fittest individuals.
    :param: fitness: the quality of a particular solution, num_parents: number of parents, 
    ga_instance
    :return: parents, parents' indices
    """
    k = 21 # a higher k value is more exploitative, so can result in faster solutions.
    parents_indices = []
    population_size = len(fitness)
    
    for _ in range(num_parents):
        # Randomly select 'k' contestants from the population
        contestants = np.random.choice(np.arange(population_size), k, replace=False)
        contestant_fitness = fitness[contestants]
        
        winner = np.argmax(contestant_fitness)
        parents_indices.append(contestants[winner])
    
    # Get the actual parent solutions from the population using the indices
    parents = ga_instance.population[parents_indices, :]
    
    # Return the parents and their indices
    return parents, np.array(parents_indices)

    
# OX crossover
def crossover_func(parents, offspring_size, ga_instance):
    """
    Function that crosses over the genes of two parents. Takes a subsequence from parent 1
    and then fill the missing cities from parent 2 in the child
    :param: parents, offspring_size, ga_instance
    :returns: offspring
    """
    offspring = np.empty(offspring_size, dtype=int)
    num_genes = parents.shape[1]  # Length of each route

    for k in range(offspring_size[0]):
        # Select parents
        parent1 = parents[k % parents.shape[0]]
        parent2 = parents[(k + 1) % parents.shape[0]]

        # Child array initialized with -1
        child = np.full(num_genes, -1)

        # Select two random crossover points
        start, end = np.sort(np.random.choice(num_genes, size=2, replace=False))

        # Copy the segment from parent1 to child
        child[start:end + 1] = parent1[start:end + 1]

        # Set for constant-time lookups
        parent1_genes = set(child[start:end + 1])

        # Fill in the rest from parent2
        p2_idx = 0
        for i in range(num_genes):
            if child[i] == -1:
                while parent2[p2_idx] in parent1_genes:
                    p2_idx += 1
                child[i] = parent2[p2_idx]
                p2_idx += 1

        # Save the offspring
        offspring[k, :] = child

    return offspring

    
def mutation_func(offspring, ga_instance):
    """
    Randomly switches two cities around to give a new permutation
    :param: offspring, ga_instance
    :return: offspring

    not used as less efficient than library function in pygad
    """

    num_cities = offspring.shape[1]  # Number of genes (cities) in each route
    mutation_prob = 0.1  # Mutation probability

    for idx in range(len(offspring)):
        if np.random.rand() < mutation_prob:
            # Select two random cities to swap
            swap_idx1, swap_idx2 = np.random.choice(num_cities, size=2, replace=False)
            # Swap the cities in the route
            offspring[idx][swap_idx1], offspring[idx][swap_idx2] = (
                offspring[idx][swap_idx2], 
                offspring[idx][swap_idx1]
            )
    
    return offspring

   
def on_generation(ga_instance):
    """
    Prints out fitness of solution after each generation
    :param ga_instance
    :returns nothing
    """
    # print the fitness of the best solution in each generation.
    print(f"Generation = {ga_instance.generations_completed}, Fitness = {ga_instance.best_solution()[1]:.5f}")

def run_genetic_algorithm(locations, num_genes, num_cities, sol_per_pop, initial_population) -> list:
    """
    Runs the pygad instance and holds the fitness function
    :param locations: The x-y coordinates for each location in the TSP
    :param num_genes: Number of genes in a solution which means the number of cities to visit in TSP
    :param num_cities: Number of cities to visit in TSP. Defines range of valid city indices
    :param sol_per_pop: Number of solutions (routes) in the population for each generation
    :param initial_population: Array representing initial population of solution.
    :returns the best solution in a list
    """
    def fitness_func(ga_instance, solution, solution_idx):
        distance = measure_distance(locations, solution)
        fitness = 1.0 / distance
        return fitness

    ga_instance = pygad.GA(
        num_generations=700,
        num_parents_mating=20,
        fitness_func=fitness_func,
        sol_per_pop=sol_per_pop,
        num_genes=num_genes,
        initial_population=initial_population,
        gene_type=int,
        mutation_type='inversion',
        crossover_type=crossover_func,
        parent_selection_type=selection_func,
        keep_parents=15,
        gene_space=list(range(num_cities))  # Ensures genes are valid city indices
    )

    ga_instance.run()

    solution, _, _ = ga_instance.best_solution()
    best_route = solution.tolist()
    return best_route