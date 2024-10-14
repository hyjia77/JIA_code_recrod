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
total_bandwidth = 250e4  # 总带宽为250MHz
noise_temperature = 213.15  # 系统的噪声温度为213.15开尔文
Polarization_isolation_factor = 12  # 单位dB
receive_benefit_ground = 15.4  # 单位dB
EIRP = 73.1  # 单位:dBm
k = 1.380649e-23  # 单位:J/K
radius_earth = 6371.0  # 单位:km
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
#获取覆盖性指标 但是测试显示得到的结果中没有非零元素
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
                coverage_str = df.iloc[j + i * client_num, time_slot]
                beam_1, beam_2 = map(int, coverage_str.strip('()').split(','))
                coverage[time_slot, j, i, 0] = beam_1
                coverage[time_slot, j, i, 1] = beam_2

        # print(f"Initialized coverage with shape: {coverage.shape}")
    return coverage
# 计算倾角覆盖矩阵 分析了一下逻辑 感觉这个没有必要了
'''
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
'''
# 计算转换次数 已验证其正确性
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
    #sat_heights_expanded = torch.unsqueeze(torch.tensor(sat_heights).clone().detach(), dim=1)
    sat_heights_expanded = torch.unsqueeze(sat_heights.clone().detach(), dim=1)
    # 注意：这里不再对 eval_angles 进行形状调整，因为它已经是预期形状


    # 计算距离公式
    distance = radius_earth * (radius_earth + sat_heights_expanded) / torch.sqrt(
        (radius_earth + sat_heights_expanded) ** 2 - radius_earth ** 2 * torch.cos(torch.deg2rad(eval_angles)))
    # 计算距离矩阵
   # distances = torch.sqrt((sat_heights + radius_earth) ** 2 - radius_earth ** 2 * torch.cos(eval_angles) ** 2)
        # print(f"[calculate_distance_matrix] Distance matrix shape: {distance.shape}")
        # 断言验证最终形状
        # assert distance.shape == (61, 10, 301), f"Unexpected shape: {distance.shape}"

    return distance

distance = calculate_distance_matrix(T, client_num, sta_num)
print("distance")
print(distance)


def calculate_DL_pathloss_matrix(distance_matrix)-> torch.Tensor:
    # 计算路径损耗矩阵
    #pathloss = 20 * torch.log10(distance_matrix) + 20 * torch.log10(torch.tensor(communication_frequency).clone().detach()) - 147.55
    pathloss = 20 * torch.log10(distance_matrix.clone().detach()*1e3) + 20 * torch.log10(communication_frequency.clone().detach()) - 147.55
    # print(f"Pathloss matrix shape: {pathloss.shape}")
    return pathloss
pathloss = calculate_DL_pathloss_matrix(distance)
print("pathloss")
print(pathloss)


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
    CNR_linear = 10 * torch.log10(CNR_linear)
        # print(f"CNR:",{CNR})
        # print(f"[calculate_CNR_matrix] CNR matrix shape: {CNR.shape}")  # [10,301,301]
    return CNR_linear

#distance_matrix_all = calculate_distance_matrix(T, client_num, sta_num)
#coverage_indicator = initialize_coverage(T, client_num, sta_num)

population = initialize_population(1, (T, client_num, sta_num))
CNR_linear = calculate_CNR_matrix(distance)
result = CNR_linear.mul(population[0])
print("CNR_linear")
print(CNR_linear)
#print("result")
#print(result)
#nonzero_indices_INR = torch.nonzero(result)
#print('nonzero_indices_INR:')
#print(nonzero_indices_INR)

def calculate_interference_matrix(T, client_num, sta_num)-> torch.Tensor:
    interference_matrix = torch.zeros((T,client_num, sta_num), dtype=torch.float32, device=device)
    for t in range(T):
        row_sums = result[t].sum(dim=1)
        print(row_sums)
        row_sums_expanded = row_sums.unsqueeze(1).expand_as(result[t])
        result2 = (row_sums_expanded - result[t]).mul(population[0][t])
        print(result2)
        interference_matrix[t] =  result2
    return interference_matrix
interference_matrix = calculate_interference_matrix(T, client_num, sta_num)
print("interference_matrix")
print(interference_matrix)

def update_rates_and_capacity():

    # 直接更新信道容量，不考虑时间维度
    channel_capacity = total_bandwidth * torch.log2(1.0 + result / (interference_matrix + 1.0))
    #channel_capacity = total_bandwidth * torch.log2(1.0 + CNR)
    #print(channel_capacity)
    # 确保 channel_capacity 形状正确
    return channel_capacity
channel_capacity = update_rates_and_capacity()
print("channel_capacity")
print(channel_capacity)

nonzero_indices_INR = torch.nonzero(channel_capacity)
print('nonzero_indices_INR:')
print(nonzero_indices_INR)
# 找到张量中非零元素的索引
nonzero_indiceschannel_capacity = torch.nonzero(channel_capacity)
print(nonzero_indiceschannel_capacity)
# 提取出非零元素
nonzero_elementschannel_capacity = channel_capacity[nonzero_indiceschannel_capacity[:, 0], nonzero_indiceschannel_capacity[:, 1]]
print(nonzero_elementschannel_capacity)
'''
def calculate_interference(t: int, user_index: int, accessed_satellite_index: int) -> float:
        # 先计算整个时间序列的距离矩阵
        # distance_matrix = calculate_distance_matrix(T, client_num, sta_num)
        # 只取当前时间槽的距离矩阵
        current_distance_matrix = distance_matrix_all[t]
        # coverage_indicator = calculate_coverge(T, client_num, sta_num)
        # 计算路径损耗矩阵，传递正确的距离矩阵
        loss = calculate_DL_pathloss_matrix(current_distance_matrix)
        # print('loss:')
        # print(loss)
        total_interference_power_watts = torch.tensor(0.0)

        for satellite_index in range(sta_num):
            if satellite_index != accessed_satellite_index and (
                    coverage_indicator[t, user_index, satellite_index, 0] == 1 or
                    coverage_indicator[t, user_index, satellite_index, 1] == 1):

                if satellite_index >= loss.shape[0] or user_index >= loss.shape[1]:
                    continue

                loss_value = loss[user_index, satellite_index]

                EIRP_watts = 10 ** ((EIRP - 30) / 10)
                receive_benefit_ground = max(receive_benefit_ground, 1e-10)  # 避免除零情况
                interference_power_watts = EIRP_watts * (10 ** (receive_benefit_ground / 10)) / (
                    max(10 ** (loss_value / 10), 1e-10))  # 避免除零情况

                total_interference_power_watts = total_interference_power_watts + interference_power_watts

        total_interference_power_watts = max(total_interference_power_watts, 1e-10)  # 避免数值过小导致负无穷
        total_interference_dBm = 10 * torch.log10(total_interference_power_watts) + 30

        return total_interference_dBm.item()

def calculate_interference_matrix(individual,t, client_num, sta_num)-> torch.Tensor:
    interference_matrix = torch.zeros((client_num, sta_num), dtype=torch.float32, device=device)
    individual = torch.tensor(individual, dtype=torch.float32, device=device)  # 将individual转换为张量
    for user_index in range(client_num):
        for satellite_index in range(sta_num):
            if individual[t, user_index, satellite_index] == 1:
                interference_matrix[user_index, satellite_index] = calculate_interference(t, user_index, satellite_index)
    #interference_matrix = interference_matrix.transpose(0, 1)
        # print(f"[calculate_interference_matrix] Interference matrix shape: {interference_matrix.shape}")
    return interference_matrix



population = initialize_population(1, (T, client_num, sta_num))
for ind in population:
    for t in range(T):
        for satellite_index in range(sta_num):
            for j in range(client_num):
                interference_matrix = calculate_interference_matrix(ind, t, client_num, sta_num)
'''