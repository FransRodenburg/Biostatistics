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

# The following gives us two things:
#  - Predictions of lambda for every value of x
#  - A standard error at that value of x
pred <- predict(GLM, newdata = list(time = x), type = "link", se.fit = TRUE)

# Combined, those two can create a 95% CI on the scale of linear predictor.
# On this scale (not on the original scale!), we can use a normal-approximation.
eta     <- pred$fit
eta_lwr <- pred$fit + qnorm(0.025) * pred$se.fit
eta_upr <- pred$fit + qnorm(0.975) * pred$se.fit
  
# Compute the resulting values on the y-axis (and multiply by n)
y     <- inverselink(eta) * n
y_lwr <- inverselink(eta_lwr) * n
y_upr <- inverselink(eta_upr) * n

# Plot the resulting model
plot(x, y, type = "l")

# Add the confidence band
lines(x, y_lwr, lty = "dashed")
lines(x, y_upr, lty = "dashed")
