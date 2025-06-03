import pandas as pd
import numpy as np
from deap import base, creator, tools, gp, algorithms
import operator
import math
import functools
import matplotlib.pyplot as plt
import networkx as nx

file_path = 'Tetuan City power consumption.csv'
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

# Tournament Selection
toolbox.register("select", tools.selTournament, tournsize=3)

# Leaf-partial crossover and mutation operations
toolbox.register("mate", gp.cxOnePointLeafBiased, termpb=0.1)
toolbox.register("mutate", gp.mutUniform, expr=toolbox.expr, pset=pset)

population_size = 50
generations = 40
mutation_prob = 0.2
crossover_prob = 0.5

population = toolbox.population(n=population_size)

fitnesses = map(toolbox.evaluate, population)
for ind, fit in zip(population, fitnesses):
    ind.fitness.values = fit

halloffame = tools.HallOfFame(1)

# Elitism
result_population, log = algorithms.eaSimple(
    population,
    toolbox,
    cxpb=crossover_prob,
    mutpb=mutation_prob,
    ngen=generations,
    halloffame=halloffame,
    verbose=True
)

best_individual = halloffame[0]
best_func = toolbox.compile(expr=best_individual)

def plot_true_vs_pred(y_true, y_pred):
    plt.figure(figsize=(10, 6))
    plt.scatter(range(len(y_true)), y_true, label='True Values', alpha=0.6)
    plt.scatter(range(len(y_pred)), y_pred, label='Predicted Values', alpha=0.6, marker='x')
    plt.xlabel('Index')
    plt.ylabel('Values')
    plt.title('True Values vs. Predicted Values')
    plt.legend()
    plt.grid(True)
    plt.show()

y_pred = np.array([best_func(*x) for x in X])
plot_true_vs_pred(y, y_pred)


