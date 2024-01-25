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

# The following gives us two things:
#  - Predictions of lambda for every value of x
#  - A standard error at that value of x
pred <- predict(GLM, newdata = list(age = x), type = "link", se.fit = TRUE)

# Combined, those two can create a 95% CI on the scale of linear predictor.
# On this scale (not on the original scale!), we can use a normal-approximation.
eta     <- pred$fit
eta_lwr <- pred$fit + qnorm(0.025) * pred$se.fit
eta_upr <- pred$fit + qnorm(0.975) * pred$se.fit
  
# Compute the resulting values on the y-axis
y     <- inverselink(eta)
y_lwr <- inverselink(eta_lwr)
y_upr <- inverselink(eta_upr)

# Plot the resulting model
plot(x, y, type = "l")

# Add the confidence band
lines(x, y_lwr, lty = "dashed")
lines(x, y_upr, lty = "dashed")
