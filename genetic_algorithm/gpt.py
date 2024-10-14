import torch

# 定义覆盖和接入变量的尺寸
N, K, U = 10, 20, 30  # 示例值，根据实际情况调整

# 生成与 np.random.randint() 相同的随机整数数组，并将其移动到GPU上
c_torch = torch.randint(0, 2, size=(N, K, U), device='cuda')

def initialize_population(pop_size, var_shape):
    population = []
    for _ in range(pop_size):
        individual = torch.zeros(var_shape, dtype=torch.int, device='cuda')
        for t in range(var_shape[2]):
            for k in range(var_shape[1]):
                n = torch.randint(var_shape[0], (1,), device='cuda')
                individual[n, k, t] = 1
        population.append(individual)
    return population

def calculate_fitness(individual, c):
    throughput = torch.sum(individual * c)
    return throughput

def select_parents(population, fitness):
    fitness_tensor = torch.tensor(fitness, device='cuda')  # 将fitness列表转换为PyTorch的Tensor并移动到GPU
    indices = torch.argsort(fitness_tensor)[-2:]
    return [population[i] for i in indices]

def crossover(parent1, parent2):
    point = torch.randint(1, parent1.shape[2] - 1, (1,), device='cuda')
    child1 = torch.cat((parent1[:, :, :point], parent2[:, :, point:]), dim=2)
    child2 = torch.cat((parent2[:, :, :point], parent1[:, :, point:]), dim=2)
    return child1, child2

def mutate(individual, mutation_rate=0.01):
    for t in range(individual.shape[2]):
        for k in range(individual.shape[1]):
            if torch.rand(1, device='cuda') < mutation_rate:
                n = torch.randint(individual.shape[0], (1,), device='cuda')
                individual[:, k, t] = 0
                individual[n, k, t] = 1
    return individual

# Genetic Algorithm Parameters
pop_size = 50
max_generations = 500

# Initialize population并将其移动到GPU上
population = initialize_population(pop_size, (N, K, U))

# Run Genetic Algorithm
for generation in range(max_generations):
    fitness = [calculate_fitness(ind, c_torch) for ind in population]
    parents = select_parents(population, fitness)

    new_population = []
    for _ in range(pop_size // 2):
        child1, child2 = crossover(parents[0], parents[1])
        new_population.append(mutate(child1))
        new_population.append(mutate(child2))

    population = new_population

    best_fitness = max(fitness)
    print(f"Generation {generation}: Best Fitness = {best_fitness}")
