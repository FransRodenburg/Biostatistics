# Read the data
DF <- read.csv("example_data_overdispersed_counts.csv")

# Fit a Poisson GLM
GLM <- glm(apples ~ age, family = "poisson", data = DF)

# A cursory check: Is the deviance ratio close to 1? (No, it is much larger.)
deviance(GLM) / df.residual(GLM)

# Option 1 (recommended): Use the simulation-based approach in DHARMa:
library("DHARMa")
simres <- simulateResiduals(GLM)
testDispersion(simres)

# Option 2: Implement your own (for comparison with Python)

# Explanation:
#  - Use Monte Carlo simulation of new outcomes, based on the model
#  - Compute the residual deviance for each simulation
#  - The proportion of simulated deviance larger than observed is a p-value

# Observed deviance
obsDev <- deviance(GLM)

# Predictor(s), sample size, and fitted values
X <- DF$age
n <- nrow(DF)
yhat <- fitted(GLM)

set.seed(1234)
MC <- 10000
simDev <- numeric(MC)
for(i in 1:MC){
  sim_y     <- rpois(n = n, lambda = yhat)
  simDev[i] <- sum(poisson()$dev.resids(sim_y, yhat, 1))
}

# p-value
mean(obsDev < simDev)

hist(simDev, xlim = range(c(simDev, obsDev)), breaks = 30,
     main = "Simulated (grey) vs observed deviance (red)", xlab = "Deviance")
abline(v = obsDev, col = 2, lwd = 2)
