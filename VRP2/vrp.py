# encoding: utf-8

import sys
import random
import math
import matplotlib.pyplot as plt


def read_vrp_from_file(filename):
    vrp = {}
    with open(filename, 'r') as f:
        lines = f.readlines()

    lines = [line.strip() for line in lines if line.strip() and not line.startswith('#')]

    if lines[0].lower() != 'params:':
        raise ValueError('Invalid input: params section missing')

    index = 1
    while lines[index].lower() != 'nodes:':
        inputs = lines[index].split()
        if inputs[0].lower() == 'capacity':
            vrp['capacity'] = float(inputs[1])
        index += 1

    vrp['nodes'] = [{'label': 'depot', 'demand': 0, 'posX': 0, 'posY': 0}]
    index += 1
    for line in lines[index:]:
        inputs = line.split()
        if len(inputs) < 4:
            raise ValueError('Invalid input: node data is incomplete')
        node = {
            'label': inputs[0],
            'demand': float(inputs[1]),
            'posX': float(inputs[2]),
            'posY': float(inputs[3])
        }
        vrp['nodes'].append(node)

    return vrp


def distance(n1, n2):
    dx = n2['posX'] - n1['posX']
    dy = n2['posY'] - n1['posY']
    return math.sqrt(dx * dx + dy * dy)


def fitness(p, vrp):
    s = distance(vrp['nodes'][0], vrp['nodes'][p[0]])
    for i in range(len(p) - 1):
        prev = vrp['nodes'][p[i]]
        next = vrp['nodes'][p[i + 1]]
        s += distance(prev, next)
    s += distance(vrp['nodes'][p[len(p) - 1]], vrp['nodes'][0])
    return s


def adjust(p, vrp):
    repeated = True
    while repeated:
        repeated = False
        for i1 in range(len(p)):
            for i2 in range(i1):
                if p[i1] == p[i2]:
                    haveAll = True
                    for nodeId in range(len(vrp['nodes'])):
                        if nodeId not in p:
                            p[i1] = nodeId
                            haveAll = False
                            break
                    if haveAll:
                        del p[i1]
                    repeated = True
                if repeated: break
            if repeated: break
    i = 0
    s = 0.0
    cap = vrp['capacity']
    while i < len(p):
        s += vrp['nodes'][p[i]]['demand']
        if s > cap:
            p.insert(i, 0)
            s = 0.0
        i += 1
    i = len(p) - 2
    while i >= 0:
        if p[i] == 0 and p[i + 1] == 0:
            del p[i]
        i -= 1


def plot_vrp_route(vrp, route):
    depot = vrp['nodes'][0]
    x_coords = [depot['posX']]
    y_coords = [depot['posY']]

    for node_idx in route:
        node = vrp['nodes'][node_idx]
        x_coords.append(node['posX'])
        y_coords.append(node['posY'])

    x_coords.append(depot['posX'])
    y_coords.append(depot['posY'])

    plt.figure(figsize=(8, 8))
    plt.plot(x_coords, y_coords, 'bo-', label='Route')

    plt.text(depot['posX'], depot['posY'], 'depot', fontsize=12, ha='right')
    for i, node_idx in enumerate(route):
        node = vrp['nodes'][node_idx]
        plt.text(node['posX'], node['posY'], node['label'], fontsize=12, ha='right')

    plt.title('Vehicle Route')
    plt.grid(True)
    plt.legend()
    plt.show()


def solve_vrp(vrp, popsize, iterations):
    pop = []

    for i in range(popsize):
        p = list(range(1, len(vrp['nodes'])))
        random.shuffle(p)
        pop.append(p)
    for p in pop:
        adjust(p, vrp)

    for i in range(iterations):
        nextPop = []
        for j in range(int(len(pop) / 2)):
            parentIds = set()
            while len(parentIds) < 4:
                parentIds |= {random.randint(0, len(pop) - 1)}
            parentIds = list(parentIds)
            parent1 = pop[parentIds[0]] if fitness(pop[parentIds[0]], vrp) < fitness(pop[parentIds[1]], vrp) else pop[
                parentIds[1]]
            parent2 = pop[parentIds[2]] if fitness(pop[parentIds[2]], vrp) < fitness(pop[parentIds[3]], vrp) else pop[
                parentIds[3]]
            cutIdx1, cutIdx2 = random.randint(1, min(len(parent1), len(parent2)) - 1), random.randint(1,
                                                                                                      min(len(parent1),
                                                                                                          len(parent2)) - 1)
            cutIdx1, cutIdx2 = min(cutIdx1, cutIdx2), max(cutIdx1, cutIdx2)
            child1 = parent1[:cutIdx1] + parent2[cutIdx1:cutIdx2] + parent1[cutIdx2:]
            child2 = parent2[:cutIdx1] + parent1[cutIdx1:cutIdx2] + parent2[cutIdx2:]
            nextPop += [child1, child2]
        if random.randint(1, 15) == 1:
            ptomutate = nextPop[random.randint(0, len(nextPop) - 1)]
            i1 = random.randint(0, len(ptomutate) - 1)
            i2 = random.randint(0, len(ptomutate) - 1)
            ptomutate[i1], ptomutate[i2] = ptomutate[i2], ptomutate[i1]
        for p in nextPop:
            adjust(p, vrp)
        pop = nextPop

    better = None
    bf = float('inf')
    for p in pop:
        f = fitness(p, vrp)
        if f < bf:
            bf = f
            better = p

    print("route:")
    print('Depot')
    for nodeIdx in better:
        print(vrp['nodes'][nodeIdx]['label'])
    print('Depot')
    print('cost:')
    print(f'{bf:.2f}')

    plot_vrp_route(vrp, better)


if __name__ == "__main__":
    vrp = read_vrp_from_file('vrp_data')
    popsize = 40
    iterations = 500

    solve_vrp(vrp, popsize, iterations)
