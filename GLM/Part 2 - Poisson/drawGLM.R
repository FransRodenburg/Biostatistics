# Read the data (Session --> Set Working Directory --> To source file location)
DF <- read.csv("example_data_counts.csv")

# Fit the model
GLM <- glm(apples ~ age, family = poisson(link = "log"), data = DF)

# Enter the estimates
intercept   <- coef(GLM)[1]
slope       <- coef(GLM)[2]
inverselink <- exp

# Create a large sequence of numbers along the x-axis
x <- seq(0, 3, length.out = 1000)

# Compute the predictions on the scale of the linear predictor
eta <- intercept + slope * x
  
# Compute the resulting values on the y-axis
y <- inverselink(eta)

# Plot the resulting model
plot(x, y, type = "l")