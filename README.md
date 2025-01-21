# Genetic Algorithm for Solving the Traveling Salesman Problem (TSP)

## Project Overview
This project implements a Genetic Algorithm (GA) to solve the Traveling Salesman Problem (TSP). The algorithm optimizes the route through a given set of cities to minimize the total distance traveled. The project includes the following key components:

1. **genetic_algorithm.py**: Contains the genetic algorithm logic, including custom functions for selection, crossover, and mutation.
2. **main.py**: Acts as the entry point for running the algorithm and visualizing the results.
3. **evaluation.py**: Provides utility functions for measuring route distances and plotting routes.

## Authors
- Mohammed Arab
- Ben Cacic
- Andrew Phan

---

## Key Features
### Genetic Algorithm Components
- **Selection Function**: Implements tournament selection with adjustable pressure to choose parent solutions for crossover.
- **Crossover Function**: Uses Order Crossover (OX) to combine parent routes and generate offspring while preserving valid permutations of cities.
- **Mutation Function**: Swaps two cities in the route with a small probability to introduce diversity (not actively used as Pygad provides a more efficient built-in mutation function).
- **Fitness Function**: Evaluates solutions based on the inverse of the route distance (shorter distances yield higher fitness).
- **Callback**: Outputs the best solution's fitness after each generation.

### Main Script Features
- **Input Data**: Reads city coordinates from a CSV file.
- **Initialization**: Generates an initial population of solutions.
- **Route Optimization**: Runs the genetic algorithm to find an optimized route through the cities.
- **Visualization**: Plots the final route on a graph and displays the total distance.

---

## Prerequisites
### Libraries
This project requires the following Python libraries:
- `pygad`: For implementing the genetic algorithm.
- `numpy`: For numerical computations.
- `pandas`: For data manipulation.
- `matplotlib`: For plotting routes.

To install the required libraries, run:
```bash
pip install pygad numpy pandas matplotlib
```

### Data
The algorithm expects a CSV file with city coordinates. Each row represents a city with its `x` and `y` coordinates. An example file path is `./3625-assignment-1-team-main/data/250a.csv`.

---

## How to Run the Project
1. **Prepare the Input Data**:
   - Place a CSV file with city coordinates in the appropriate folder.
   - Example file: `./3625-assignment-1-team-main/data/250a.csv`.

2. **Run the Main Script**:
   ```bash
   python main.py
   ```

3. **Output**:
   - The console will display the optimized route distance and the time taken to find the solution.
   - A plot of the route will appear, showing the sequence of cities.

---

## Key Functions and Parameters
### **genetic_algorithm.py**
#### `selection_func(fitness, num_parents, ga_instance)`
- Selects parent solutions using tournament selection.
- **Parameters**:
  - `fitness`: Array of fitness values.
  - `num_parents`: Number of parents to select.
  - `ga_instance`: Current GA instance.
- **Returns**:
  - Selected parents and their indices.

#### `crossover_func(parents, offspring_size, ga_instance)`
- Combines two parent routes using Order Crossover (OX).
- **Parameters**:
  - `parents`: Selected parent solutions.
  - `offspring_size`: Shape of the offspring array.
  - `ga_instance`: Current GA instance.
- **Returns**:
  - Array of offspring solutions.

#### `mutation_func(offspring, ga_instance)`
- Randomly swaps two cities in a route (not used by default).
- **Parameters**:
  - `offspring`: Array of offspring solutions.
  - `ga_instance`: Current GA instance.
- **Returns**:
  - Mutated offspring solutions.

#### `run_genetic_algorithm(locations, num_genes, num_cities, sol_per_pop, initial_population)`
- Runs the genetic algorithm and returns the best solution.
- **Parameters**:
  - `locations`: DataFrame of city coordinates.
  - `num_genes`: Number of cities in a route.
  - `num_cities`: Total number of cities.
  - `sol_per_pop`: Number of solutions in each generation.
  - `initial_population`: Initial population of routes.
- **Returns**:
  - List representing the best route.

### **main.py**
#### `find_route(locations, num_genes, num_cities, sol_per_pop, initial_population)`
- Wrapper function to call `run_genetic_algorithm`.
- **Parameters**:
  - `locations`: DataFrame of city coordinates.
  - `num_genes`: Number of cities in a route.
  - `num_cities`: Total number of cities.
  - `sol_per_pop`: Number of solutions in each generation.
  - `initial_population`: Initial population of routes.
- **Returns**:
  - Optimized route as a list or array.

---

## Example Workflow
1. Load the dataset:
   ```python
   tsp = pd.read_csv('./data/250a.csv', index_col=0)
   tsp = tsp.reset_index(drop=True)
   ```

2. Set algorithm parameters:
   ```python
   num_cities = tsp.shape[0]
   sol_per_pop = 40
   initial_population = np.array([list(range(num_cities)) for _ in range(sol_per_pop)])
   ```

3. Find the optimized route:
   ```python
   route = find_route(locations=tsp, num_genes=num_cities, num_cities=num_cities, sol_per_pop=sol_per_pop, initial_population=initial_population)
   ```

4. Measure the distance:
   ```python
   distance = evaluation.measure_distance(tsp, route)
   ```

5. Visualize the route:
   ```python
   evaluation.plot_route(tsp, route)
   ```

---

## Acknowledgments
This project uses the Pygad library for implementing genetic algorithms and matplotlib for route visualization.

