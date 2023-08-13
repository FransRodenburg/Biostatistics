library("car") # For the qqPlot function. Install if missing.
# Store this script as well as the data set in the same location, then go to:
# Session > Set Working Directory > To Source File Location
DF <- read.csv("example_data_simple_linear_regression.csv")

# Fit a linear model
LM <- lm(height ~ weight, data = DF)

# Perform visual diagnostics
par(mfrow = c(2, 2))   # Plot in a 2x2 grid
plot(LM, which = 1)    # Residuals vs fitted
qqPlot(LM, reps = 1e4) # QQ-plot
mtext("QQ-plot", 3, 0.25)
plot(LM, which = 3)    # Scale-location
plot(LM, which = 5)    # Cook's distance
par(mfrow = c(1, 1))   # Restore the default
