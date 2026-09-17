# Read the data
DF <- read.csv("example_data_ratios.csv")

# Fit the model
GLM <- glm(cbind(correct, 10 - correct) ~ time, data = DF, family = "binomial")

# Regression table
summary(GLM)

# Baseline odds
exp(coef(GLM)[1])

# Multiplicative change in odds
exp(coef(GLM)[2])

# Probability at any given point (e.g., x = 2)
# qlogis = logit, plogis = inverse logit
plogis(coef(GLM)[1] + coef(GLM)[2] * 2)

# ... or just use predict:
predict(GLM, list(time = 2), type = "response")

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
