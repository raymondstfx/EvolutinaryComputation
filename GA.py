import json
import matplotlib.pyplot as plt
import random
import math
import numpy as np


class GA(object):
    def __init__(self, num_city, num_total, iteration, data):
        self.num_city = num_city
        self.num_total = num_total
        self.scores = []
        self.iteration = iteration
        self.location = data
        self.ga_choose_ratio = 0.2
        self.mutate_ratio = 0.05
        self.dis_mat = self.compute_dis_mat(num_city, data)
        self.fruits = self.greedy_init(self.dis_mat, num_total, num_city)

        scores = self.compute_adp(self.fruits)
        sort_index = np.argsort(-scores)
        init_best = self.fruits[sort_index[0]]
        init_best = self.location[init_best]

        self.iter_x = [0]
        self.iter_y = [1. / scores[sort_index[0]]]

    def greedy_init(self, dis_mat, num_total, num_city):
        # Greedy initialization to create a starting population
        start_index = 0
        result = []
        for i in range(num_total):
            rest = [x for x in range(num_city)]
            if start_index >= num_city:
                start_index = np.random.randint(0, num_city)
                result.append(result[start_index].copy())
                continue
            current = start_index
            rest.remove(current)
            result_one = [current]
            while len(rest) != 0:
                tmp_min = math.inf
                tmp_choose = -1
                for x in rest:
                    if dis_mat[current][x] < tmp_min:
                        tmp_min = dis_mat[current][x]
                        tmp_choose = x
                current = tmp_choose
                result_one.append(tmp_choose)
                rest.remove(tmp_choose)
            result.append(result_one)
            start_index += 1
        return result

    def compute_dis_mat(self, num_city, location):
        # Calculate the distance matrix between cities
        dis_mat = np.zeros((num_city, num_city))
        for i in range(num_city):
            for j in range(num_city):
                if i == j:
                    dis_mat[i][j] = np.inf
                    continue
                a = location[i]
                b = location[j]
                tmp = np.sqrt(sum([(x[0] - x[1]) ** 2 for x in zip(a, b)]))
                dis_mat[i][j] = tmp
        return dis_mat

    def compute_pathlen(self, path, dis_mat):
        # Calculate the length of a given path
        try:
            a = path[0]
            b = path[-1]
        except:
            import pdb
            pdb.set_trace()
        result = dis_mat[a][b]
        for i in range(len(path) - 1):
            a = path[i]
            b = path[i + 1]
            result += dis_mat[a][b]
        return result

    def compute_adp(self, fruits):
        # Calculate the fitness of the population
        adp = []
        for fruit in fruits:
            if isinstance(fruit, int):
                import pdb
                pdb.set_trace()
            length = self.compute_pathlen(fruit, self.dis_mat)
            adp.append(1.0 / length)
        return np.array(adp)

    def ga_cross(self, x, y, num_points=3):
        # Perform crossover between two individuals using multiple crossover points
        len_ = len(x)
        assert len(x) == len(y)

        # Select multiple crossover points
        path_list = [t for t in range(len_)]
        order = sorted(random.sample(path_list, num_points))
        x_new, y_new = x.copy(), y.copy()

        for i in range(0, len(order) - 1, 2):
            start, end = order[i], order[i + 1]

            x_section = x_new[start:end].copy()
            y_section = y_new[start:end].copy()

            x_new[start:end] = y_section
            y_new[start:end] = x_section

        x_new = self.resolve_conflicts(x_new, y)
        y_new = self.resolve_conflicts(y_new, x)

        return list(x_new), list(y_new)

    def resolve_conflicts(self, offspring, other_parent):
        seen = set()
        conflicts = []

        for i, city in enumerate(offspring):
            if city in seen:
                conflicts.append(i)
            else:
                seen.add(city)

        missing_cities = [city for city in other_parent if city not in seen]

        for i, conflict_index in enumerate(conflicts):
            offspring[conflict_index] = missing_cities[i]

        return offspring

    def ga_parent(self, scores, ga_choose_ratio):
        sort_index = np.argsort(-scores).copy()
        sort_index = sort_index[0:int(ga_choose_ratio * len(sort_index))]
        parents = []
        parents_score = []
        for index in sort_index:
            parents.append(self.fruits[index])
            parents_score.append(scores[index])
        return parents, parents_score

    def ga_choose(self, genes_score, genes_choose):
        sum_score = sum(genes_score)
        score_ratio = [sub * 1.0 / sum_score for sub in genes_score]
        rand1 = np.random.rand()
        rand2 = np.random.rand()
        for i, sub in enumerate(score_ratio):
            if rand1 >= 0:
                rand1 -= sub
                if rand1 < 0:
                    index1 = i
            if rand2 >= 0:
                rand2 -= sub
                if rand2 < 0:
                    index2 = i
            if rand1 < 0 and rand2 < 0:
                break
        return list(genes_choose[index1]), list(genes_choose[index2])

    def ga_mutate(self, gene):
        path_list = [t for t in range(len(gene))]
        order = list(random.sample(path_list, 2))
        start, end = min(order), max(order)
        tmp = gene[start:end]
        tmp = tmp[::-1]
        gene[start:end] = tmp
        return list(gene)

    def ga(self):
        scores = self.compute_adp(self.fruits)
        parents, parents_score = self.ga_parent(scores, self.ga_choose_ratio)
        tmp_best_one = parents[0]
        tmp_best_score = parents_score[0]
        fruits = parents.copy()
        while len(fruits) < self.num_total:
            gene_x, gene_y = self.ga_choose(parents_score, parents)
            gene_x_new, gene_y_new = self.ga_cross(gene_x, gene_y)
            if np.random.rand() < self.mutate_ratio:
                gene_x_new = self.ga_mutate(gene_x_new)
            if np.random.rand() < self.mutate_ratio:
                gene_y_new = self.ga_mutate(gene_y_new)
            x_adp = 1. / self.compute_pathlen(gene_x_new, self.dis_mat)
            y_adp = 1. / self.compute_pathlen(gene_y_new, self.dis_mat)
            if x_adp > y_adp and (not gene_x_new in fruits):
                fruits.append(gene_x_new)
            elif x_adp <= y_adp and (not gene_y_new in fruits):
                fruits.append(gene_y_new)
        # Ensure the best individuals are retained in the population
        if tmp_best_one not in fruits:
            fruits.append(tmp_best_one)
        self.fruits = fruits
        return tmp_best_one, tmp_best_score

    def run(self):
        # Run the genetic algorithm for the specified number of iterations
        BEST_LIST = None
        best_score = -math.inf
        self.best_record = []
        for i in range(1, self.iteration + 1):
            tmp_best_one, tmp_best_score = self.ga()
            self.iter_x.append(i)
            self.iter_y.append(1. / tmp_best_score)
            if tmp_best_score > best_score:
                best_score = tmp_best_score
                BEST_LIST = tmp_best_one
            self.best_record.append(1. / best_score)
            print(i, 1. / best_score)
        print(1. / best_score)
        return self.location[BEST_LIST], 1. / best_score


# Read TSP data from a JSON file
def read_tsp(path):
    with open(path, 'r') as file:
        data = json.load(file)  # Load JSON data

    node_coordinates = []
    for coord in data['NODE_COORD_SECTION']:
        node_coordinates.append([coord[0], coord[1], coord[2]])  # Append x and y coordinates

    return np.array(node_coordinates)

data = read_tsp('data/berlin52.json')
data = data[:, 1:]
Best, Best_path = math.inf, None
model = GA(num_city=data.shape[0], num_total=25, iteration=1000, data=data.copy())
path, path_len = model.run()
if path_len < Best:
    Best = path_len
    Best_path = path

fig, axs = plt.subplots(2, 1, sharex=False, sharey=False)
axs[0].scatter(Best_path[:, 0], Best_path[:, 1])
Best_path = np.vstack([Best_path, Best_path[0]])
axs[0].plot(Best_path[:, 0], Best_path[:, 1])
axs[0].set_title('Optimal Path')
iterations = range(model.iteration)
best_record = model.best_record
axs[1].plot(iterations, best_record)
axs[1].set_title('Convergence Curve')
plt.subplots_adjust(hspace=0.5)

plt.suptitle('Genetic Algorithm Solution for TSP Problem', fontsize=16)
plt.show()
