import pandas as pd
import numpy as np
import statsmodels.api as sm
import matplotlib.pyplot as plt

# Read the data
DF = pd.read_csv("example_data_counts.csv")

# Fit the model
GLM = sm.GLM(DF['apples'], sm.add_constant(DF['age']), family=sm.families.Poisson()).fit()

# Enter the estimates
intercept = GLM.params[0]
slope = GLM.params[1]
inverselink = np.exp

# Create a large sequence of numbers along the x-axis
x = np.linspace(0, 3, 1000)

# Compute the predictions on the scale of the linear predictor
eta = intercept + slope * x

# Compute the resulting values on the y-axis
y = inverselink(eta)

# Plot the resulting model
plt.plot(x, y, '-')

# Show the plot
plt.show()
