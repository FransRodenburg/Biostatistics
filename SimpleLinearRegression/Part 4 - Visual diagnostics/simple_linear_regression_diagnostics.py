import numpy as np
import pandas as pd
import statsmodels.api as sm
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.graphics.gofplots import qqplot
from statsmodels.stats.outliers_influence import OLSInfluence

# Read the CSV data
DF = pd.read_csv("example_data_simple_linear_regression.csv")

# Fit a linear model
LM = sm.OLS(DF['height'], sm.add_constant(DF['weight'])).fit()

# Perform visual diagnostics
fig, axes = plt.subplots(2, 2, figsize=(10, 8))

# Residuals vs Fitted values plot
sns.residplot(x = LM.fittedvalues, y = LM.resid, lowess = True, ax = axes[0, 0])
axes[0, 0].set_xlabel('Fitted values')
axes[0, 0].set_ylabel('Residuals')
axes[0, 0].set_title('Residuals vs Fitted')

# QQ-plot
qqplot(LM.resid, line = 's', ax = axes[0, 1])
axes[0, 1].set_title('Normal Q-Q')

# Scale-Location plot
sqrt_abs_standardized_residuals = (abs(LM.resid) ** 0.5) / LM.resid.std()
sns.regplot(x = LM.fittedvalues, y = sqrt_abs_standardized_residuals,
            lowess = True, ax = axes[1, 0])
axes[1, 0].set_xlabel('Fitted values')
axes[1, 0].set_ylabel('Sqrt(|Standardized Residuals|)')
axes[1, 0].set_title('Scale-Location')

# Residuals vs Leverage plot
influence = OLSInfluence(LM)
residuals = influence.resid_studentized_internal
leverage = influence.hat_matrix_diag
sns.scatterplot(x = leverage, y = residuals, ax = axes[1, 1])
axes[1, 1].set_xlabel('Leverage')
axes[1, 1].set_ylabel('Standardized Residuals')
axes[1, 1].set_title('Residuals vs Leverage')

# Adjust y-limit of Residuals vs Leverage plot
y_range = max(residuals) - min(residuals)
axes[1, 1].set_ylim(min(residuals) - 0.1 * y_range,
                    max(residuals) + 0.1 * y_range)

# Add boundaries for Cook's distance (translation from R's plot.lm)
p = len(LM.params)
cook_levels = [0.5, 1.0]
hii = influence.hat_matrix_diag
r_hat = (min(hii), max(hii))
hh = np.linspace(min(r_hat[0], r_hat[1] / 100), r_hat[1], num = 101)
for crit in cook_levels:
    cl_h = np.sqrt(crit * p * (1 - hh) / hh)
    axes[1, 1].plot(hh, cl_h, linestyle = '--', color = 'r')
    axes[1, 1].plot(hh, -cl_h, linestyle = '--', color = 'r')

# Adjust layout and add more margin for x- and y-labels
plt.subplots_adjust(wspace = 0.25, hspace = 0.5, bottom = 0.1, 
                    left = 0.1, right = 0.95, top = 0.95)

# Show the plots
plt.show()
