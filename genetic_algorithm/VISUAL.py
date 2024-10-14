import pandas as pd
import torch

# 检查CUDA是否可用，如果可用则将设备设置为CUDA
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 定义基础变量
T = 60   # 定义时隙数量
sta_num = 301  # 定义卫星数量
client_num = 10  # 定义用户数量

# 定义场景参数
angle_threshold = 15  # 定义倾角阈值
w1 = 0.4  # 定义切换次数权重
w2 = 0.6# 定义用户速率权重
communication_frequency = torch.tensor(18.5e9) # 通信频率为18.5GHz
total_bandwidth = 250e6  # 总带宽为250MHz
noise_temperature = 213.15  # 系统的噪声温度为213.15开尔文
Polarization_isolation_factor = 12  # 单位dB
receive_benefit_ground = 15.4  # 单位dB
EIRP = 73.1  # 单位:dBm
k = 1.380649e-23  # 单位:J/K
radius_earth = 6731  # 单位:km
EIRP_watts = 10 ** ((EIRP - 30) / 10)  # 将 EIRP 从 dBm 转换为瓦特
noise_power = k * noise_temperature * total_bandwidth  # 噪声功率计算

# 初始化种群，形状为60 * 10 * 300 ,约束条件也已满足
def initialize_population(pop_size, var_shape):
    population = []
    for _ in range(pop_size):
        individual = torch.zeros(var_shape, dtype=torch.float32, device=device)
        for t in range(var_shape[0]):
            available_satellites = torch.arange(var_shape[2], device=device)
            for k in range(var_shape[1]):
                if len(available_satellites) > 0:
                    chosen_satellite = torch.randint(len(available_satellites), (1,), device=device).item()
                    satellite_index = available_satellites[chosen_satellite]
                    individual[t, k, satellite_index] = torch.randint(0, 2, (1,), dtype=torch.float32).item()
                    available_satellites = torch.cat((
                        available_satellites[:chosen_satellite],
                        available_satellites[chosen_satellite+1:]
                    ))
        population.append(individual)
    return population


def calculate_hk(individual):

    # 在第一个尺度下计算元素变换的次数
    axis0_changes = torch.nonzero(torch.diff(individual, dim=0)).size(0)

    return axis0_changes
'''
# 定义适应度函数，即就是优化目标
def calculate_fitness(individual):

    fitness = torch.sum(individual ** 2)
    return fitness
'''
def calculate_fitness(individual):
    hk = calculate_hk(individual)

    fitness = hk
    return -fitness
def select_parents(population, fitness):
    fitness_tensor = torch.tensor(fitness, device='cuda')
    # 选择适应度最高的两个个体的索引
    # indices = torch.argsort(fitness_tensor)[-2:]

    # 选择适应度最低的两个个体的索引
    indices = torch.argsort(fitness_tensor)[:2]
    return [population[i] for i in indices]

def crossover(parent1, parent2):
    point1 = torch.randint(1, parent1.shape[2] - 2, (1,), device='cuda')
    point2 = torch.randint(point1 + 1, parent1.shape[2] - 1, (1,), device='cuda')
    child1 = torch.cat((parent1[:, :, :point1], parent2[:, :, point1:point2], parent1[:, :, point2:]), dim=2)
    child2 = torch.cat((parent2[:, :, :point1], parent1[:, :, point1:point2], parent2[:, :, point2:]), dim=2)
    return child1, child2

def mutate(individual, generation, max_generations, mutation_rate=0.1):
    for t in range(individual.shape[2]):
        for k in range(individual.shape[1]):
            if torch.rand(1, device='cuda') < mutation_rate * (1 - generation / max_generations):
                n = torch.randint(individual.shape[0], (1,), device='cuda')
                individual[:, k, t] = 0
                individual[n, k, t] = 1
    return individual


# Genetic Algorithm Parameters
pop_size = 20
max_generations = 100
mutation_rate = 0.1

# Initialize population
population = initialize_population(pop_size, (T, client_num, sta_num))

# Run Genetic Algorithm
for generation in range(max_generations):
    #print("population:")
    #print(population)
    fitness = [calculate_fitness(ind) for ind in population]
    parents = select_parents(population, fitness)

    new_population = []
    for _ in range(pop_size // 2):
        child1, child2 = crossover(parents[0], parents[1])
        new_population.append(mutate(child1, generation, max_generations, mutation_rate))
        new_population.append(mutate(child2, generation, max_generations, mutation_rate))

    population = new_population

    best_fitness = min(fitness)
    print(f"Generation {generation}: Best Fitness = {best_fitness}")