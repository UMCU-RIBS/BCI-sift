import time

import numpy as np
import ray
from deap import base, creator, tools
from deap.algorithms import varAnd

ngen = 10

# Initialize Ray
ray.init()

# Define the problem
creator.create("FitnessMax", base.Fitness, weights=(1.0,))
creator.create("Individual", list, fitness=creator.FitnessMax)


@ray.remote
def crossvalidation(individual):
    time.sleep(5)
    return np.sum(np.square(individual))


def evaluate_candidates(individual, cv_folds=10):
    return np.mean(
        ray.get([crossvalidation.remote(individual) for _ in range(cv_folds)])
    )


# Define your objective function
@ray.remote
def evaluate(individual):
    time.sleep(1)  # Simulating a long evaluation
    return (evaluate_candidates(individual),)


# Set up DEAP toolbox
toolbox = base.Toolbox()
toolbox.register(
    "individual",
    tools.initRepeat,
    creator.Individual,
    lambda: np.random.rand(10),
    n=1000,
)
toolbox.register("population", tools.initRepeat, list, toolbox.individual)
toolbox.register("evaluate", evaluate)
toolbox.register("mate", tools.cxTwoPoint)
toolbox.register("mutate", tools.mutGaussian, mu=0, sigma=1, indpb=0.1)
toolbox.register("select", tools.selTournament, tournsize=3)


# Custom map function using Ray
def ray_map(func, iterable):
    return ray.get([func.remote(x) for x in iterable])


toolbox.register("map", ray_map)

# Example of running a simple evolutionary algorithm
population = toolbox.population(n=128)

# Begin the generational process
for gen in range(1, ngen + 1):
    start = time.time()

    # Select the next generation individuals
    offspring = toolbox.select(population, len(population))

    # Vary the pool of individuals
    offspring = varAnd(offspring, toolbox, 0.5, 0.2)

    # Evaluate the individuals with an invalid fitness
    invalid_ind = [ind for ind in offspring if not ind.fitness.valid]

    if invalid_ind:  # Only evaluate if there are invalid individuals
        fitnesses = toolbox.map(toolbox.evaluate, invalid_ind)
        for ind, fit in zip(invalid_ind, fitnesses):
            ind.fitness.values = fit

    # Replace the current population by the offspring
    population[:] = offspring
    print(f"Generation {gen} completed in {time.time() - start:.2f} seconds")

ray.shutdown()
