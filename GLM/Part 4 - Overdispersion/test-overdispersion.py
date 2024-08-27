import pandas as pd
import numpy as np
import statsmodels.api as sm
import matplotlib.pyplot as plt

# Read the data
DF = pd.read_csv("example_data_overdispersed_counts.csv")

# Fit a Poisson GLM
X = sm.add_constant(DF['age'])
y = DF['apples']
GLM = sm.GLM(y, X, family = sm.families.Poisson()).fit()

# A cursory check: Is the deviance ratio close to 1? (No, it is much larger.)
GLM.deviance / GLM.df_resid

# Simulation-based overdispersion test
# Explanation:
#  - Use Monte Carlo simulation of new outcomes, based on the model
#  - Compute the residual deviance for each simulation
#  - The proportion of simulated deviance larger than observed is a p-value

# Observed deviance
obsDev = GLM.deviance

# Sample size, and fitted values
n = DF.shape[0]
yhat = GLM.fittedvalues

# Function to compute simulated deviance (here: Poisson deviance)
def Poisson_deviance(sim_y, yhat):
    return 2 * np.sum(
        np.where(sim_y == 0, 0, sim_y * np.log(sim_y / yhat)) - (sim_y - yhat)
    )

np.random.seed(1234)
MC = 10000
simDev = np.zeros(MC)
for i in range(MC):
    sim_y = np.random.poisson(lam = yhat)
    simDev[i] = Poisson_deviance(sim_y, yhat)

# p-value
np.mean(obsDev < simDev)

plt.hist(simDev, bins = 30, color = 'grey', edgecolor = 'black',
         label = 'Simulated Deviance',
         range = (min(simDev.min(), obsDev), max(simDev.max(), obsDev)))
plt.axvline(obsDev, color = 'red', linewidth = 2, label='Observed Deviance')
plt.xlabel("Deviance")
plt.ylabel("Frequency")
plt.legend()
plt.show()
