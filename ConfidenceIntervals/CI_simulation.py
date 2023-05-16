import numpy as np

# The number of Monte Carlo simulations
MC = 10000

k = 0
l = 0

for i in range(1, MC + 1):
    # Draw two random bubbles from a uniform distribution
    x = np.random.uniform(-5, 5, size=2)
    
    # Confidence interval based on the sampling distribution
    CI1 = np.mean(x) + np.array([-1, 1]) * (5 - 5 / np.sqrt(2))
    contained = (CI1[0] < 0) & (CI1[1] > 0)
    k += contained
    
    # Non-parametric confidence interval (distance between bubbles)
    CI2 = np.mean(x) + np.array([-1, 1]) * np.abs(np.diff(x)) / 2
    contained = (CI2[0] < 0) & (CI2[1] > 0)
    l += contained

print("Method 1:", f"{round(k/MC * 100, 1)}%")
print("Method 2:", f"{round(l/MC * 100, 1)}%")
