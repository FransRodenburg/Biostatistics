# Store this script as well as the data set in the same location, then go to:
# Session > Set Working Directory > To Source File Location
DF <- read.csv("example_data_simple_linear_regression.csv")

# Fit a linear model
LM <- lm(height ~ weight, data = DF)

# Print a regression table
summary(LM)