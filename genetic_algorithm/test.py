from matplotlib import pyplot as plt
from torch import tensor

fitness_history = [tensor(7.2795e+08, device='cuda:0'), tensor(7.6580e+08, device='cuda:0'), tensor(8.0856e+08, device='cuda:0'), tensor(8.5247e+08, device='cuda:0'), tensor(8.9106e+08, device='cuda:0'), tensor(9.2867e+08, device='cuda:0'), tensor(9.7230e+08, device='cuda:0'), tensor(1.0005e+09, device='cuda:0'), tensor(1.0308e+09, device='cuda:0'), tensor(1.0641e+09, device='cuda:0'), tensor(1.0870e+09, device='cuda:0'), tensor(1.1101e+09, device='cuda:0'), tensor(1.1447e+09, device='cuda:0'), tensor(1.1667e+09, device='cuda:0'), tensor(1.2016e+09, device='cuda:0'), tensor(1.2255e+09, device='cuda:0'), tensor(1.2476e+09, device='cuda:0'), tensor(1.2579e+09, device='cuda:0'), tensor(1.2745e+09, device='cuda:0'), tensor(1.2928e+09, device='cuda:0'), tensor(1.3048e+09, device='cuda:0'), tensor(1.3263e+09, device='cuda:0'), tensor(1.3399e+09, device='cuda:0'), tensor(1.3532e+09, device='cuda:0'), tensor(1.3684e+09, device='cuda:0'), tensor(1.3892e+09, device='cuda:0'), tensor(1.4025e+09, device='cuda:0'), tensor(1.4104e+09, device='cuda:0'), tensor(1.4190e+09, device='cuda:0'), tensor(1.4230e+09, device='cuda:0'), tensor(1.4300e+09, device='cuda:0'), tensor(1.4385e+09, device='cuda:0'), tensor(1.4503e+09, device='cuda:0'), tensor(1.4548e+09, device='cuda:0'), tensor(1.4563e+09, device='cuda:0'), tensor(1.4700e+09, device='cuda:0'), tensor(1.4796e+09, device='cuda:0'), tensor(1.4863e+09, device='cuda:0'), tensor(1.4921e+09, device='cuda:0'), tensor(1.4947e+09, device='cuda:0'), tensor(1.4989e+09, device='cuda:0'), tensor(1.5000e+09, device='cuda:0'), tensor(1.5035e+09, device='cuda:0'), tensor(1.5070e+09, device='cuda:0'), tensor(1.5107e+09, device='cuda:0'), tensor(1.5234e+09, device='cuda:0'), tensor(1.5243e+09, device='cuda:0'), tensor(1.5292e+09, device='cuda:0'), tensor(1.5403e+09, device='cuda:0'), tensor(1.5457e+09, device='cuda:0'), tensor(1.5519e+09, device='cuda:0'), tensor(1.5519e+09, device='cuda:0'), tensor(1.5568e+09, device='cuda:0'), tensor(1.5569e+09, device='cuda:0'), tensor(1.5594e+09, device='cuda:0'), tensor(1.5620e+09, device='cuda:0'), tensor(1.5654e+09, device='cuda:0'), tensor(1.5673e+09, device='cuda:0'), tensor(1.5722e+09, device='cuda:0'), tensor(1.5736e+09, device='cuda:0'), tensor(1.5736e+09, device='cuda:0'), tensor(1.5747e+09, device='cuda:0'), tensor(1.5758e+09, device='cuda:0'), tensor(1.5764e+09, device='cuda:0'), tensor(1.5780e+09, device='cuda:0'), tensor(1.5813e+09, device='cuda:0'), tensor(1.5838e+09, device='cuda:0'), tensor(1.5878e+09, device='cuda:0'), tensor(1.5878e+09, device='cuda:0'), tensor(1.5908e+09, device='cuda:0'), tensor(1.5924e+09, device='cuda:0'), tensor(1.5927e+09, device='cuda:0'), tensor(1.5951e+09, device='cuda:0'), tensor(1.6004e+09, device='cuda:0'), tensor(1.6017e+09, device='cuda:0'), tensor(1.6021e+09, device='cuda:0'), tensor(1.6034e+09, device='cuda:0'), tensor(1.6036e+09, device='cuda:0'), tensor(1.6062e+09, device='cuda:0'), tensor(1.6062e+09, device='cuda:0'), tensor(1.6070e+09, device='cuda:0'), tensor(1.6070e+09, device='cuda:0'), tensor(1.6104e+09, device='cuda:0'), tensor(1.6120e+09, device='cuda:0'), tensor(1.6135e+09, device='cuda:0'), tensor(1.6145e+09, device='cuda:0'), tensor(1.6154e+09, device='cuda:0'), tensor(1.6165e+09, device='cuda:0'), tensor(1.6187e+09, device='cuda:0'), tensor(1.6195e+09, device='cuda:0'), tensor(1.6209e+09, device='cuda:0'), tensor(1.6234e+09, device='cuda:0'), tensor(1.6240e+09, device='cuda:0'), tensor(1.6248e+09, device='cuda:0'), tensor(1.6265e+09, device='cuda:0'), tensor(1.6292e+09, device='cuda:0'), tensor(1.6304e+09, device='cuda:0'), tensor(1.6315e+09, device='cuda:0'), tensor(1.6317e+09, device='cuda:0'), tensor(1.6320e+09, device='cuda:0')]
fitness_history = [x.cpu().numpy() for x in fitness_history]
def smooth_curve(points, factor=0.9):
    smoothed_points = []
    for point in points:
        if smoothed_points:
            previous = smoothed_points[-1]
            smoothed_points.append(previous * factor + point * (1 - factor))
        else:
            smoothed_points.append(point)
    return smoothed_points

smoothed_fitness_history = smooth_curve(fitness_history)
plt.plot(smoothed_fitness_history)

#print(fitness_history)
# Plot fitness history
#plt.plot(fitness_history)
plt.xlabel('Generation')
plt.ylabel('Best Fitness')
plt.show()