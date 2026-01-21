# Save both files (data and this script) in the same folder. Then:
# Session > Set Working Directory > To Source File Location
# Then you can run the script below. Do not forget to install DHARMa if needed.
library("DHARMa")

# Example (fill in 01, 02, or 03)
DF <- read.csv(paste0("example-data-01.csv"))
# 1: No violations
# 2: Non-linear (on the scale of eta)
# 3: Overdispersed

# Fit a GLM
GLM <- glm(y ~ x, data = DF, family = "poisson")

# DHARMa plots
simres <- simulateResiduals(GLM)
plot(simres)
