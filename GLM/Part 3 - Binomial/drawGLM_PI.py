import pandas as pd
import numpy as np
import statsmodels.api as sm
import matplotlib.pyplot as plt
from scipy.stats import norm
from scipy.stats import binom

# Read the data
DF = pd.read_csv("example_data_ratios.csv")

# Fit the model
n = 10
GLM = sm.GLM(np.column_stack((DF['correct'], n - DF['correct'])),
             sm.add_constant(DF['time']),
             family=sm.families.Binomial(link=sm.families.links.logit())).fit()

# Enter the estimates
intercept = GLM.params[0]
slope = GLM.params[1]

# Create a large sequence of numbers along the x-axis
x = np.linspace(0, 20, 1000)

# Get predictions on the scale of the linear predictor
pred = GLM.get_prediction(exog=sm.add_constant(x)).linpred

# Extract the predicted mean and standard errors
eta    = pred.predicted_mean
se_fit = pred.se_mean

# Compute the confidence intervals on the scale of the linear predictor
eta_lwr = eta + norm.ppf(0.025) * se_fit
eta_upr = eta + norm.ppf(0.975) * se_fit

# Compute the resulting values on the y-axis
# For the prediction interval, we want the lowest 2.5% of a Poisson distribution
# with lambda equal to the lower bound of the confidence interval, and the 
# highest 2.5% o a Poisson distribution with lambda equal to the upper bound.
y     = 1 / (np.exp(-eta) + 1) * n
y_lwr = binom.ppf(0.025, 10, 1 / (np.exp(-eta_lwr) + 1))
y_upr = binom.ppf(0.975, 10, 1 / (np.exp(-eta_upr) + 1))

# Plot the resulting model
plt.plot(x, y)

# Add the confidence band
plt.plot(x, y_lwr, linestyle = 'dashed')
plt.plot(x, y_upr, linestyle = 'dashed')

plt.show()
