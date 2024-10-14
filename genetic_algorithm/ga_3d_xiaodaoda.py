import numpy as np
import matplotlib.pyplot as plt
from ypstruct import structure


def run(problem, params):
    costfunc = problem.costfunc
    var_shape = problem.var_shape
    maxit = params.maxit
    npop = params.npop
    beta = params.beta
    pc = params.pc
    nc = int(np.round(pc * npop / 2) * 2)

    empty_individual = structure()
    empty_individual.position = None
    empty_individual.cost = None

    bestsol = empty_individual.deepcopy()
    bestsol.cost = -np.inf

    pop = empty_individual.repeat(npop)
    for i in range(npop):
        pop[i].position = initialize(var_shape)
        pop[i].cost = costfunc(pop[i].position)
        if pop[i].cost > bestsol.cost:
            bestsol = pop[i].deepcopy()

    bestcost = np.empty(maxit)

    for it in range(maxit):
        costs = np.array([x.cost for x in pop])
        avg_cost = np.mean(costs)
        if avg_cost != 0:
            costs = costs / avg_cost
        probs = np.exp(-beta * costs)

        popc = []
        for _ in range(nc // 2):
            p1 = pop[roulette_wheel_selection(probs)]
            p2 = pop[roulette_wheel_selection(probs)]
            c1, c2 = crossover(p1, p2, var_shape)
            c1 = mutate(c1)
            c2 = mutate(c2)
            c1.cost = costfunc(c1.position)
            if c1.cost > bestsol.cost:
                bestsol = c1.deepcopy()
            c2.cost = costfunc(c2.position)
            if c2.cost > bestsol.cost:
                bestsol = c2.deepcopy()
            popc.append(c1)
            popc.append(c2)

        pop += popc
        pop = sorted(pop, key=lambda x: x.cost, reverse=True)
        pop = pop[:npop]
        bestcost[it] = bestsol.cost
        print(f"Iteration {it}: Best Cost = {bestcost[it]}")

    out = structure()
    out.pop = pop
    out.bestsol = bestsol
    out.bestcost = bestcost
    return out

'''
def initialize(shape):
    position = np.zeros(shape, dtype=int)
    for i in range(shape[0]):
        for j in range(shape[1]):
            position[i, j, np.random.choice(shape[2])] = 1
    return position
'''
def initialize(var_shape):
        individual = np.zeros(var_shape, dtype=np.float32)
        for t in range(var_shape[0]):
            available_satellites = np.arange(var_shape[2])
            for k in range(var_shape[1]):
                if len(available_satellites) > 0:
                    chosen_satellite = np.random.choice(len(available_satellites))
                    satellite_index = available_satellites[chosen_satellite]
                    individual[t, k, satellite_index] = np.random.choice([0, 1])
                    available_satellites = np.concatenate((
                        available_satellites[:chosen_satellite],
                        available_satellites[chosen_satellite+1:]
                    ))
        return individual


def crossover(p1, p2, shape):
    c1 = p1.deepcopy()
    c2 = p2.deepcopy()
    for i in range(shape[0]):
        for j in range(shape[1]):
            if np.random.rand() < 0.5:
                c1.position[i, j] = p2.position[i, j]
                c2.position[i, j] = p1.position[i, j]
    return c1, c2


def mutate(x):
    y = x.deepcopy()
    for i in range(x.position.shape[0]):
        for j in range(x.position.shape[1]):
            if np.random.rand() < 0.1:  # Mutation probability
                cols = np.where(y.position[i, j] == 1)[0]
                if len(cols) > 0:
                    k = cols[0]
                    new_k = np.random.choice([col for col in range(y.position.shape[2]) if col != k])
                    y.position[i, j, k] = 0
                    y.position[i, j, new_k] = 1
    return y


def roulette_wheel_selection(p):
    c = np.cumsum(p)
    r = sum(p) * np.random.rand()
    ind = np.argwhere(r <= c)
    return ind[0][0]

'''
# Example Cost Function
def example_cost_function(x):
    return np.sum(x ** 2)
'''

# 计算转换次数 已验证其正确性
def example_cost_function(individual):

    # 在第一个尺度下计算元素变换的次数
    axis0_changes = np.count_nonzero(np.diff(individual, axis=0))
    print(axis0_changes)
    print(np.sum(individual ** 2))

    return -0.4*axis0_changes+ 0.6*np.sum(individual ** 2)

# Problem Definition
problem = structure()
problem.costfunc = example_cost_function
problem.var_shape = (60, 10, 301)

# GA Parameters
params = structure()
params.maxit = 200
params.npop = 50
params.beta = 1
params.pc = 1
params.gamma = 0.1
params.mu = 0.01
params.sigma = 0.1

# Run GA
out = run(problem, params)

# Results
plt.plot(out.bestcost)
print(out.bestsol.position)
plt.xlim(0, params.maxit)
plt.xlabel('Iterations')
plt.ylabel('Best Cost')
plt.title('Genetic Algorithm (GA)')
plt.grid(True)
plt.show()
