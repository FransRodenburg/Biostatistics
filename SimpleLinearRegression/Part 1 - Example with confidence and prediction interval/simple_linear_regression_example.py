import pandas as pd
import statsmodels.api as sm
import matplotlib.pyplot as plt
import numpy as np

# Read the CSV data
DF = pd.read_csv("example_data_simple_linear_regression.csv")

# Fit a linear model
LM = sm.OLS(DF['height'], sm.add_constant(DF['weight'])).fit()

# Create an empty canvas
fig, ax = plt.subplots(constrained_layout = True)
ax.set_xlim(0, 160)
ax.set_ylim(125, 215)
ax.set_xlabel("Weight (kg)")
ax.set_ylabel("Height (cm)")

# Add prediction band
x_values = np.arange(0, 160, 0.1)
newdata = sm.add_constant(pd.DataFrame({'weight': x_values}))
y_pred = LM.get_prediction(newdata).summary_frame()
ax.fill_between(x_values, y_pred['obs_ci_lower'], y_pred['obs_ci_upper'], 
                color = "coral", alpha = 0.5)

# Add confidence band
ax.fill_between(x_values, y_pred['mean_ci_lower'], y_pred['mean_ci_upper'], 
                color = "white")
ax.fill_between(x_values, y_pred['mean_ci_lower'], y_pred['mean_ci_upper'], 
                color = "steelblue", alpha = 0.5)

# Add the regression line
ax.plot(x_values, y_pred['mean'], color = "steelblue", linewidth = 2)

# Add the data points on top
ax.scatter(DF['weight'], DF['height'], marker = 'o', color = 'black')

# Add axes
ax.xaxis.set_ticks_position('bottom')
ax.yaxis.set_ticks_position('left')

plt.show()
