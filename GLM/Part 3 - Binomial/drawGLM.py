import pandas as pd
import numpy as np
import statsmodels.api as sm
import matplotlib.pyplot as plt

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

# Compute the predictions on the scale of the linear predictor
eta = intercept + slope * x

# Compute the resulting values on the y-axis (and multiply by n)
y = 1 / (np.exp(-eta) + 1) * n

# Plot the resulting model
plt.plot(x, y)

# Show the plot
plt.show()
