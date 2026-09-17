import pandas as pd
import numpy as np
import statsmodels.api as sm
import matplotlib.pyplot as plt

# Read the data
DF = pd.read_csv("example_data_ratios.csv")

# Fit the model
# (feel free to contact me if you know a shorter way to specify this)
GLM = sm.GLM(
    DF["correct"] / 10,
    sm.add_constant(DF["time"]),
    family=sm.families.Binomial(),
    freq_weights=np.repeat(10, len(DF))
).fit()

# Regression table
print(GLM.summary2())

# Baseline odds
odds0 = np.exp(GLM.params.iloc[0])
print(odds0)

# Multiplicative change in odds
oddsx = np.exp(GLM.params.iloc[1])
print(oddsx)

# Probabilities at any given point (e.g., x = 2)
# Explanation: 1 = use intercept, 2 = value of x, [0] strips the array.
p = GLM.predict([1, 2])[0]
print(p)

# Rough measure of overdispersion
d = GLM.deviance / GLM.df_resid
print(d)

# Deviance-based pseudo-R^2 (deviance explained)
deviance_explained = 1 - GLM.deviance / GLM.null_deviance
print(deviance_explained)

# Nagelkerke pseudo R^2 (likelihood improvement over intercept-only model)
n = GLM.nobs
ll0 = GLM.llnull
ll = GLM.llf
Nagelkerke = (1 - np.exp((2 / n) * (ll0 - ll))) / (1 - np.exp((2 / n) * ll0))
print(Nagelkerke)

