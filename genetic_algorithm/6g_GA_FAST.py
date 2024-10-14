import os

import numpy as np
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
radius_earth = 6731e3  # 单位:m
EIRP_watts = 10 ** ((EIRP - 30) / 10)  # 将 EIRP 从 dBm 转换为瓦特
noise_power = k * noise_temperature * total_bandwidth  # 噪声功率计算

# 初始化种群，形状为60 * 10 * 300
'''
def initialize_population(pop_size, var_shape):
    population = []
    for _ in range(pop_size):
        individual = torch.zeros(var_shape, dtype=torch.float32, device=device)
        for i in range(var_shape[0]):
            row = torch.randint(var_shape[1], (1,), device=device)
            col = torch.randint(var_shape[2], (1,), device=device)
            individual[i, row, col] = 1
        population.append(individual)
    return population
'''
def initialize_population(pop_size, var_shape):
    population = []
    for _ in range(pop_size):
        individual = torch.zeros(var_shape, dtype=torch.float32, device=device)
        for t in range(var_shape[0]):
            for k in range(var_shape[2]):
                n = torch.randint(var_shape[1], (1,), device=device)
                individual[t, :, k] = 0  # 确保每行只有一个元素为1
                individual[t, n, k] = torch.rand(1, device=device)  # 生成随机在0和1之间的值

        # 随机打乱每行和每列的顺序
        for t in range(var_shape[0]):
            for _ in range(var_shape[1]):
                torch.randperm(var_shape[1], device=device)

        population.append(individual)
    return population

# 获取仰角数据,返回一个60*10*301的numpy数组
def gain_evl(T, client_num, sta_num):
    df = pd.read_csv('ev_data.csv')
    var_shape = (T, client_num, sta_num)
    evl = torch.zeros(var_shape, dtype=torch.float32, device=device)

    for t in range(T):
        for n in range(client_num):
            for k in range(sta_num):
                evl[t, n, k] = df.iloc[k * client_num + n, t]
    return evl

def initialize_coverage(T, client_num, sta_num):
    csv_file = 'coverge_data.csv'

    df = pd.read_csv(csv_file, header=None, skiprows=1)

    # print(f"cov DataFrame shape: {df.shape}")

    # 从CSV文件读取覆盖数据
    coverage = torch.zeros((T, client_num, sta_num, 2), dtype=torch.float32)

    # 填充 coverage 数组
    for time_slot in range(T):
        for i in range(sta_num):
            for j in range(client_num):
                # 获取覆盖字符串并解析成整数
                coverage_str = df.iloc[j + i * client_num, T]
                beam_1, beam_2 = map(int, coverage_str.strip('()').split(','))
                coverage[time_slot, j, i, 0] = beam_1
                coverage[time_slot, j, i, 1] = beam_2

        # print(f"Initialized coverage with shape: {coverage.shape}")
        return coverage
# 计算倾角覆盖矩阵
def calculate_coverge(T, client_num, sta_num):
    evl = gain_evl(T, client_num, sta_num)
    var_shape = (T, client_num, sta_num)
    coverage_indicator = torch.zeros(var_shape, dtype=torch.float32, device=device)
    for time_slot in range(T):
        for user_index in range(client_num):
            for satellite_index in range(sta_num):
                if evl[time_slot, user_index, satellite_index] > angle_threshold:
                    coverage_indicator[time_slot, user_index, satellite_index] = 1
                else:
                    coverage_indicator[time_slot, user_index, satellite_index] = 0
    return coverage_indicator

coverage_indicator_ind = calculate_coverge(T, client_num, sta_num)
# 计算转换次数
def calculate_hk(individual, coverage_indicator):
    # 对应元素相乘
    result_tensor = torch.mul(individual, coverage_indicator)

    # 在第一个尺度下计算元素变换的次数
    axis0_changes = torch.nonzero(torch.diff(result_tensor, dim=0)).size(0)

    return axis0_changes

# 获取卫星高度
def gain_alt(T, sta_num):
    df = pd.read_csv('alt_data.csv')
    var_shape = (T, sta_num)
    alt = torch.zeros(var_shape, dtype=torch.float32, device=device)

    for t in range(T):
        for k in range(sta_num):
            # 确保从 DataFrame 中读取的数据是数值类型
            value = float(df.iloc[k, t])
            alt[t, k] = int(value)

    return alt

# 计算距离
def calculate_distance_matrix(T, client_num, sta_num)-> torch.Tensor :
    # 获取所有时间段的卫星高度和仰角
    sat_heights = gain_alt(T, sta_num)  # 形状: [61, 301]
    eval_angles = gain_evl(T, client_num, sta_num)  # 形状: [61, 10, 301]

    # 通过调整形状来启用广播
    sat_heights_expanded = torch.unsqueeze(torch.tensor(sat_heights).clone().detach(), dim=1)
    # 注意：这里不再对 eval_angles 进行形状调整，因为它已经是预期形状


    # 计算距离公式
    distance = radius_earth * (radius_earth + sat_heights_expanded) / torch.sqrt(
        (radius_earth + sat_heights_expanded) ** 2 - radius_earth ** 2 * torch.cos(torch.deg2rad(eval_angles)) ** 2)

        # print(f"[calculate_distance_matrix] Distance matrix shape: {distance.shape}")
        # 断言验证最终形状
        # assert distance.shape == (61, 10, 301), f"Unexpected shape: {distance.shape}"

    return distance

def calculate_DL_pathloss_matrix(distance_matrix)-> torch.Tensor:
    # 计算路径损耗矩阵
    pathloss = 20 * torch.log10(distance_matrix) + 20 * torch.log10(torch.tensor(communication_frequency).clone().detach()) - 147.55

    # print(f"Pathloss matrix shape: {pathloss.shape}")
    return pathloss

#CNR的计算需要根据决策变量来决定，所以应该只记录当前slot下的CNR情况
def calculate_CNR_matrix(distance_matrix)-> torch.Tensor :
    # 计算路径损耗矩阵，其形状为 [NUM_TIME_SLOTS, NUM_SATELLITES, NUM_GROUND_USER]
    loss = calculate_DL_pathloss_matrix(distance_matrix)

    # 计算接收功率（单位：瓦特），假设 self.EIRP_watts 和 self.receive_benefit_ground 是标量
    received_power_watts = EIRP_watts * 10 ** (receive_benefit_ground / 10) / (10 ** (loss / 10))
    # print(f"received power watts:",{received_power_watts})

    # 计算 CNR（线性值），假设 self.noise_power 是标量
    CNR_linear = received_power_watts / noise_power
        # print(f"CNR Linear:",{CNR_linear})
        # 返回 CNR 的对数值（单位：dB），保持矩阵形状
        # CNR = 10 * torch.log10(CNR_linear)
        # print(f"CNR:",{CNR})
        # print(f"[calculate_CNR_matrix] CNR matrix shape: {CNR.shape}")  # [10,301,301]
    return CNR_linear

def calculate_interference_matrix(individual, client_num, sta_num)-> torch.Tensor:
    interference_matrix = torch.zeros((T, client_num, sta_num), dtype=torch.float32, device=device)
    interference_matrix = calculate_interference(T, client_num, sta_num)
    #interference_matrix = interference_matrix.transpose(0, 1)
        # print(f"[calculate_interference_matrix] Interference matrix shape: {interference_matrix.shape}")
    return interference_matrix

# 先计算整个时间序列的距离矩阵
distance_matrix_all = calculate_distance_matrix(T, client_num, sta_num)
coverage_indicator = initialize_coverage(T, client_num, sta_num)
def calculate_interference(T, client_num, sta_num) -> float:
        # 先计算整个时间序列的距离矩阵
        #distance_matrix = calculate_distance_matrix(T, client_num, sta_num)
        # 只取当前时间槽的距离矩阵
        #current_distance_matrix = distance_matrix_all[t]
        #coverage_indicator = calculate_coverge(T, client_num, sta_num)
        # 计算路径损耗矩阵，传递正确的距离矩阵
        loss = calculate_DL_pathloss_matrix(distance_matrix_all)
        #print('loss:')
        #print(loss)
        total_interference_power_watts = torch.tensor(0.0)

        #loss_value = loss[user_index, satellite_index]
        EIRP_watts = 10 ** ((EIRP - 30) / 10)
        interference_power_watts = EIRP_watts * (10 ** (receive_benefit_ground / 10)) / (
                            10 ** (loss / 10))
        total_interference_power_watts = total_interference_power_watts + interference_power_watts

            # 计算总干扰（单位：dBm）
        total_interference_dBm = 10 * torch.log10(total_interference_power_watts) + 30
        return total_interference_dBm.item()
def update_rates_and_capacity(individual,t):
    distance_matrix = distance_matrix_all[t]
    # 计算 CNR 矩阵，假设其形状为 [NUM_SATELLITES, NUM_GROUND_USERS]

    CNR = calculate_CNR_matrix(distance_matrix)
    # print(f"CNR matrix shape: {CNR.shape}, values: {CNR}")
    #print(CNR)
    # 计算 INR 矩阵，假设其形状为 [NUM_SATELLITES, NUM_GROUND_USERS]
    INR = calculate_interference_matrix(individual, t, client_num, sta_num)
    #print('INR:')
    #print(INR)
    # 确保 CNR 和 INR 的形状一致
    assert CNR.shape == INR.shape, f"CNR shape {CNR.shape} does not match INR shape {INR.shape}"

    # 直接更新信道容量，不考虑时间维度
    channel_capacity = total_bandwidth * torch.log2(1.0 + CNR / (INR + 1.0))
    #print(channel_capacity)
    # 确保 channel_capacity 形状正确
    if channel_capacity.shape != (client_num,sta_num):
        channel_capacity = channel_capacity.transpose(0, 1)
    return channel_capacity

# 计算信道容量
def calculate_R(individual, t):
    reward = 0
    channel_capacity = update_rates_and_capacity(individual,t)
    #print('channel_capacity:')
    #print(channel_capacity)
    #print('individual[t]:')
    #print(individual[t])
    nonzero_indices_individual = torch.nonzero(individual[t])
    # 提取出非零元素
    #nonzero_elements_individual = individual[t][nonzero_indices_individual[:, 0], nonzero_indices_individual[:, 1]]
    #channel_capacity_individual = channel_capacity[nonzero_indices_individual[:, 0], nonzero_indices_individual[:, 1]]
    #print('nonzero_elements_individual:')
    #print(nonzero_elements_individual)
    #print('channel_capacity_individual:')
    #print(channel_capacity_individual)

    result = torch.mul(individual[t], coverage_indicator_ind[t])
    # 对应元素相乘
    result = torch.mul(result, channel_capacity)
    #print(result)
    # print(result)
    # 找到张量中非零元素的索引
    nonzero_indices = torch.nonzero(result)

    # 提取出非零元素
    nonzero_elements = result[nonzero_indices[:, 0], nonzero_indices[:, 1]]
    #print('nonzero_elements:')
    #print(nonzero_elements)
    # 对非零元素进行相加
    capacity = torch.sum(nonzero_elements)

    reward += capacity*1e-8

    return reward

# 定义适应度函数，即就是优化目标
def calculate_fitness(individual, T, client_num, sta_num):
    coverage_indicator = calculate_coverge(T, client_num, sta_num)
    hk = calculate_hk(individual, coverage_indicator) * 1e-4
    r = 0
    for t in range(2):
        rt = calculate_R(individual, t)
        r += rt
    #print(hk)
    print(r)
    # 基于切换次数和用户速率计算适应度
    fitness = -w1*hk + w2*r
    # 假设吞吐量需要最大化
    return fitness

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
    # 随机变异
    for t in range(individual.shape[2]):
        for k in range(individual.shape[1]):
            if torch.rand(1, device='cuda') < mutation_rate:
                n = torch.randint(individual.shape[0], (1,), device='cuda')
                individual[:, k, t] = 0
                individual[n, k, t] = 1
                #individual[t, k, :] = 0
                #individual[t, k, n] = 1
    return individual

# Example coverage matrix



# Genetic Algorithm Parameters
pop_size = 4
max_generations = 50

# Initialize population
population = initialize_population(pop_size, (T, client_num, sta_num))

# Run Genetic Algorithm
for generation in range(max_generations):
    fitness = [calculate_fitness(ind, T, client_num, sta_num) for ind in population]
    parents = select_parents(population, fitness)

    new_population = []
    for _ in range(pop_size // 2):
        child1, child2 = crossover(parents[0], parents[1])
        new_population.append(mutate(child1))
        new_population.append(mutate(child2))

    population = new_population

    best_fitness = max(fitness)
    print(f"Generation {generation}: Best Fitness = {best_fitness}")