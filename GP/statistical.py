# The function of calculating Statistic Analysis was added to the original code,
# and the original visualization was deleted and replaced with a histogram.
import pandas as pd
import numpy as np
from deap import base, creator, tools, gp, algorithms
import operator
import math
import functools
import matplotlib.pyplot as plt
from scipy.stats import ttest_ind, mannwhitneyu
import random

file_path = 'd06.csv'
data = pd.read_csv(file_path)

X = data.iloc[:, :-1].values
y = data.iloc[:, -1].values

pset = gp.PrimitiveSet("MAIN", X.shape[1])
pset.addPrimitive(operator.add, 2)
pset.addPrimitive(operator.sub, 2)
pset.addPrimitive(operator.mul, 2)
pset.addPrimitive(operator.neg, 1)
pset.addPrimitive(math.sin, 1)
pset.addPrimitive(math.cos, 1)


def safe_exp(x):
    try:
        return math.exp(x) if x < 10 else float('inf')
    except OverflowError:
        return float('inf')


def safe_log(x):
    return math.log(x) if x > 0 else 0


pset.addPrimitive(safe_exp, 1, name="safe_exp")
pset.addPrimitive(safe_log, 1, name="safe_log")
pset.addEphemeralConstant("rand101", functools.partial(np.random.randint, -1, 1))

creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
creator.create("Individual", gp.PrimitiveTree, fitness=creator.FitnessMin)

toolbox = base.Toolbox()
toolbox.register("expr", gp.genHalfAndHalf, pset=pset, min_=1, max_=2)
toolbox.register("individual", tools.initIterate, creator.Individual, toolbox.expr)
toolbox.register("population", tools.initRepeat, list, toolbox.individual)


def eval_symb_reg(individual):
    func = toolbox.compile(expr=individual)
    predictions = []
    for x in X:
        try:
            pred = func(*x)
            pred = np.clip(pred, -1e6, 1e6)
            predictions.append(pred)
        except (OverflowError, ValueError, ZeroDivisionError):
            predictions.append(float('inf'))
    predictions = np.array(predictions)
    return np.mean((predictions - y) ** 2),


toolbox.register("compile", gp.compile, pset=pset)
toolbox.register("evaluate", eval_symb_reg)
toolbox.register("select", tools.selTournament, tournsize=3)
toolbox.register("mate", gp.cxOnePointLeafBiased, termpb=0.1)
toolbox.register("mutate", gp.mutUniform, expr=toolbox.expr, pset=pset)

population_size = 50
generations = 40
mutation_prob = 0.2
crossover_prob = 0.5

num_runs = 30
best_fitness_values = []

for _ in range(num_runs):
    population = toolbox.population(n=population_size)
    halloffame = tools.HallOfFame(1)

    algorithms.eaSimple(
        population,
        toolbox,
        cxpb=crossover_prob,
        mutpb=mutation_prob,
        ngen=generations,
        halloffame=halloffame,
        verbose=False
    )

    best_fitness_values.append(halloffame[0].fitness.values[0])

best_fitness_values = np.array(best_fitness_values)

mean_fitness = np.mean(best_fitness_values)
std_dev_fitness = np.std(best_fitness_values)
print(f"Mean of the best fitness: {mean_fitness}")
print(f"Best fitness standard deviation: {std_dev_fitness}")

# Plotting the best fitness
plt.hist(best_fitness_values, bins=10, alpha=0.7)
plt.xlabel('Best Fitness Values')
plt.ylabel('Frequency')
plt.title('Distribution of Best Fitness Values over 30 Runs')
plt.grid(True)
plt.show()

dist2 = best_fitness_values + np.random.normal(0, 0.5, num_runs)

# t-test
t_stat, t_p_value = ttest_ind(best_fitness_values, dist2)
print(f"p value of t test: {t_p_value}")

# Mann-Whitney U
u_stat, u_p_value = mannwhitneyu(best_fitness_values, dist2)
print(f"p value of Mann-Whitney U test: {u_p_value}")


# 计算效果量（Cohen’s d）
def cohen_d(x, y):
    nx, ny = len(x), len(y)
    pooled_std = np.sqrt(((nx - 1) * np.var(x) + (ny - 1) * np.var(y)) / (nx + ny - 2))
    return (np.mean(x) - np.mean(y)) / pooled_std


effect_size = cohen_d(best_fitness_values, dist2)
print(f"Cohen’s d: {effect_size}")
