# Read the data (Session --> Set Working Directory --> To source file location)
DF <- read.csv("example_data_ratios.csv")

# Fit the model
n   <- 10
GLM <- glm(cbind(correct, n - correct) ~ time,
           family = binomial(link = "logit"), data = DF)

# Enter the estimates
intercept   <- coef(GLM)[1]
slope       <- coef(GLM)[2]
inverselink <- function(eta){
  1 / (exp(-eta) + 1)
}

# Create a large sequence of numbers along the x-axis
x <- seq(0, 20, length.out = 1000)

# Compute the predictions on the scale of the linear predictor
eta <- intercept + slope * x
  
# Compute the resulting values on the y-axis (and multiply by n)
y <- inverselink(eta) * n

# Plot the resulting model
plot(x, y, type = "l")
