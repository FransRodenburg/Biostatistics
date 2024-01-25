# Note: There might be faster/more efficient ways to do this in Pyton, 
#       but I wanted to show the steps involved as explained in the video.

import pandas as pd
import statsmodels.api as sm
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import norm
from scipy.stats import poisson

# Read the data
DF = pd.read_csv("example_data_counts.csv")

# Fit the Poisson regression model
GLM = sm.GLM(DF['apples'], sm.add_constant(DF['age']), family=sm.families.Poisson()).fit()

# Enter the estimates
intercept   = GLM.params[0]
slope       = GLM.params[1]
inverselink = np.exp

# Create a large sequence of numbers along the x-axis
x = np.linspace(0, 3, 10)

# Create a large sequence of numbers along the x-axis
x = np.linspace(0, 3, 1000)

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
y     = inverselink(eta)
y_lwr = poisson.ppf(0.025, inverselink(eta_lwr))
y_upr = poisson.ppf(0.975, inverselink(eta_upr))

# Plot the resulting model
plt.plot(x, y)

# Add the confidence band
plt.plot(x, y_lwr, linestyle = 'dashed')
plt.plot(x, y_upr, linestyle = 'dashed')

plt.show()
