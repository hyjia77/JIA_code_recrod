import numpy as np
import matplotlib.pyplot as plt
from ypstruct import structure
import ga

# Sphere Test Function
def sphere(x):
    return np.sum(x**2)

# Problem Definition
problem = structure()
problem.costfunc = sphere
problem.varmin = -100       # Lower bound (scalar)
problem.varmax = 100        # Upper bound (scalar)
problem.var_shape = (10, 301)

# GA Parameters
params = structure()
params.maxit = 700
params.npop = 5
params.beta = 1
params.pc = 1
params.gamma = 0.1
params.mu = 0.01
params.sigma = 0.1

# Run GA
out = ga.run(problem, params)

# Results
plt.plot(out.bestcost)
print(out.bestsol.position)
plt.xlim(0, params.maxit)
plt.xlabel('Iterations')
plt.ylabel('Best Cost')
plt.title('Genetic Algorithm (GA)')
plt.grid(True)
plt.show()
