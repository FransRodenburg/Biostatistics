# Read the data
DF <- read.csv("example_data_counts.csv")

# Fit the model
GLM <- glm(apples ~ age, data = DF, family = "poisson")

# Regression table
summary(GLM)

# Rough measure of overdispersion
S <- summary(GLM)
S$deviance / S$df.residual

# Deviance-based pseudo-R^2 (deviance explained)
1 - S$deviance / S$null.deviance

# Nagelkerke pseudo R^2 (likelihood improvement over intercept-only model)
n   <- nrow(DF)
ll  <- logLik(GLM)[1]
ll0 <- logLik(update(GLM, . ~ 1))[1]
Nagelkerke <- (1 - exp((2 / n) * (ll0 - ll))) / (1 - exp((2 / n) * ll0))
Nagelkerke
