import pandas as pd
import statsmodels.api as sm

# Read the CSV data
DF = pd.read_csv("example_data_simple_linear_regression.csv")

# Fit a linear model
LM = sm.OLS(DF['height'], sm.add_constant(DF['weight'])).fit()

# Print a regression table
print(LM.summary2())
