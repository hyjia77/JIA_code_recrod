import numpy as np
from ypstruct import structure

def run(problem, params):
    # Problem Information
    costfunc = problem.costfunc
    var_shape = problem.var_shape

    # Parameters
    maxit = params.maxit
    npop = params.npop
    beta = params.beta
    pc = params.pc
    nc = int(np.round(pc * npop / 2) * 2)

    # Empty Individual Template
    empty_individual = structure()
    empty_individual.position = None
    empty_individual.cost = None

    # Best Solution Ever Found
    bestsol = empty_individual.deepcopy()
    bestsol.cost = -np.inf

    # Initialize Population
    pop = empty_individual.repeat(npop)
    for i in range(npop):
        pop[i].position = np.random.randint(0, 2, var_shape)  # Random 0 or 1 matrix
        pop[i].cost = costfunc(pop[i].position)
        if pop[i].cost > bestsol.cost:
            bestsol = pop[i].deepcopy()

    # Best Cost of Iterations
    bestcost = np.empty(maxit)

    # Main Loop
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

            # Perform Crossover
            c1, c2 = crossover(p1, p2)

            # Perform Mutation
            c1 = mutate(c1)
            c2 = mutate(c2)

            # Evaluate First Offspring
            c1.cost = costfunc(c1.position)
            if c1.cost < bestsol.cost:
                bestsol = c1.deepcopy()

            # Evaluate Second Offspring
            c2.cost = costfunc(c2.position)
            if c2.cost < bestsol.cost:
                bestsol = c2.deepcopy()

            # Add Offsprings to popc
            popc.append(c1)
            popc.append(c2)

        # Merge, Sort and Select
        pop += popc
        pop = sorted(pop, key=lambda x: x.cost)
        pop = pop[:npop]

        # Store Best Cost
        bestcost[it] = bestsol.cost

        # Show Iteration Information
        print(f"Iteration {it}: Best Cost = {bestcost[it]}")

    # Output
    out = structure()
    out.pop = pop
    out.bestsol = bestsol
    out.bestcost = bestcost
    return out

def crossover(p1, p2):
    c1 = p1.deepcopy()
    c2 = p2.deepcopy()
    # Uniform crossover
    mask = np.random.randint(0, 2, p1.position.shape).astype(bool)
    c1.position[mask] = p2.position[mask]
    c2.position[mask] = p1.position[mask]
    return c1, c2

def mutate(x):
    y = x.deepcopy()
    mutation_prob = 0.1  # Mutation probability
    mask = np.random.rand(*x.position.shape) < mutation_prob
    y.position[mask] = 1 - y.position[mask]  # Flip bits
    return y

def roulette_wheel_selection(p):
    c = np.cumsum(p)
    r = sum(p) * np.random.rand()
    ind = np.argwhere(r <= c)
    return ind[0][0]
