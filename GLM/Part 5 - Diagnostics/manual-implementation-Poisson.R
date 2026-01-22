require("car")
require("qgam")
require("ragg")
require("sfsmisc")

# Example (fill in 01, 02, or 03)
DF <- read.csv(paste0("example-Poisson-data-01.csv"))
# 1: No violations
# 2: Non-linear (on the scale of eta)
# 3: Overdispersed

# To enable bootstrapped outlier test, set to TRUE. Computationally intensive.
bootstrap <- TRUE

# Fit a model
GLM <- glm(y ~ x, data = DF, family = "poisson")

# Rank-transformed predictions
yhat     <- fitted(GLM)
normrank <- function(x){
  x <- rank(x, ties.method = "average")
  x <- x / max(x)
  return(unname(x))
}
yhatrank <- normrank(yhat)

# Monte Carlo simulated outcomes
seed <- 2026
nsim <- 250  # DHARMa's default
sim  <- simulate(GLM, nsim = nsim, seed = seed)
sim  <- matrix(unlist(sim, use.names = FALSE), ncol = nsim)

# Simulated randomized residuals:
n   <- nrow(DF)
SRR <- numeric(n)

# Edge cases
one  <- numeric(n)
zero <- numeric(n)

for(i in 1:n){
  lwr <- mean(sim[i, ] < DF$y[i]) 
  upr <- mean(sim[i, ] <= DF$y[i])
  if(upr == 0){
    zero[i] <- 1
  }
  if(lwr == 1){
    one[i] <- 1
  }
  SRR[i] <- runif(1, lwr, upr)
}

# DHARMa's distributional test
KS <- ks.test(SRR, ppoints(n, a = 0))$p.value

# DHARMa's overdispersion test
simVar <- apply(sim, 2, function(x){
  var(x - yhat)
}) / var(c(sim))
obsVar <- var(DF$y - yhat) / var(c(sim))
OD <- mean(obsVar <= simVar)

# DHARMa's bootstrapped outlier test
if(bootstrap == TRUE){
  B <- nsim
  one_zero_boot <- numeric(B)
  for(i in 1:B){
    wh       <- sample((1:B)[-i], size = nsim, replace = TRUE)
    current  <- sim[, wh]
    SRR_boot <- numeric(n)
    for(j in 1:n){
      lwr <- mean(current[j, ] < sim[j, i])
      upr <- mean(current[j, ] <= sim[j, i])
      SRR_boot[j] <- runif(1, lwr, upr)
    }
    one_zero_boot[i] <- mean(SRR_boot <= 0 | SRR_boot >= 1)
  }
  one_zero_real <- mean(one + zero)
  BO <- mean(one_zero_boot >= one_zero_real)
} else{
  BO <- NA
}

pval_left <- c(KS, OD, BO)
FDR_left  <- p.adjust(pval_left, method = "BH")

# Plot
par(mfrow = c(1, 2), mar = c(5, 3, 3, 2) + 0.1, mgp = c(2, 0.75, 0))

# QQ-plot (a = 0 corresponds to what gap::qqunif uses for uniform order stats)
plot(ppoints(n, a = 0), sort(SRR), pch = 2, las = 1, bty = "n", axes = FALSE,
     xlim = c(0, 1), ylim = c(0, 1),
     xlab = "Expected", ylab = "Observed", col = "black", 
     panel.first = list(abline(a = 0, b = 1, col = "red", lty = 1)))
eaxis(1, cex.axis = 0.8)
eaxis(2, cex.axis = 0.8)

fmt_p_left <- sprintf("%.3f", signif(pval_left, 3))
text(0.33, 0.9, bquote("KS test: p ="~.(fmt_p_left[1])),
     col = ifelse(KS < 0.05, "red", "black"), cex = 0.9, xpd = TRUE)
text(0.5, 0.5, bquote("Dispersion test: p ="~.(fmt_p_left[2])),
     col = ifelse(OD < 0.05, "red", "black"), cex = 0.9, xpd = TRUE)
text(0.67, 0.1, labels = bquote("Outlier test: p ="~.(fmt_p_left[3])),
     col = ifelse(BO < 0.05, "red", "black"), cex = 0.9, xpd = TRUE)
if(any(FDR_left < 0.05, na.rm = TRUE)){
  label <- "Significant problems detected"
  col  <- "red"
  h    <- 0
} else if(any(pval_left < 0.05, na.rm = TRUE)){
  label <- "Significant problems detected" # \nCombined adjusted distributional test n.s."
  col  <- "red"
  h    <- 0 # 0.5
} else{
  label <- "No significant problems detected"
  col  <- "grey50"
  h    <- 0
}
mtext("QQ-plot residuals", 3, 1.25 + h, font = 2)
mtext(label, 3, 0.25, col = col, cex = 0.75)

# Residuals vs fitted
qu  <- c(0.25, 0.5, 0.75)
pch <- ifelse(zero | one, 8, 1)
col <- ifelse(zero | one, "red", "black")
plot(yhatrank, SRR, pch = pch, col = col, las = 1, bty = "n", axes = FALSE,
     xlim = c(0, 1), ylim = c(0, 1),
     xlab = "Model predictions (rank transformed)",
     ylab = "",
     panel.first = list(abline(h = qu, lty = 3, col = "grey75")))
mtext("DHARMa residual", 2, 2.5)

# Quantile lines
newx <- seq(0, 1, l = 1000)
pval_right <- numeric(3)
for(i in 1:3){
  data <- data.frame(y = SRR - qu[i], yhatrank)
  k    <- min(length(unique(data$yhatrank)), 10)
  invisible(capture.output({
    Qi <- qgam(y ~ s(yhatrank, k = k), qu = qu[i], data = data)
  }))
  pred     <- predict(Qi, list(yhatrank = newx), se.fit = TRUE)
  pred$fit <- pred$fit + qu[i]
  lwr      <- pred$fit - pred$se.fit # DHARMa 0.4.7 does not scale by 1.96
  upr      <- pred$fit + pred$se.fit
  S        <- summary(Qi)
  pval_right[i]  <- min(p.adjust(c(S$p.pv, S$s.pv), method = "BH"))
  red      <- ifelse(pval_right[i] < 0.05, 1, 0)
  polygon(x = c(newx, rev(newx)), y = c(lwr, rev(upr)),
          border = NA, col = rgb(0, 0, 0, 0.1))
  lines(pred$fit ~ newx, col = rgb(red, 0, 0), lwd = 2, lend = 1)
}
FDR_right <- p.adjust(pval_right, method = "BH")
if(any(FDR_right < 0.05, na.rm = TRUE)){
  label <- "Quantile deviations detected (red curves)\nCombined adjusted quantile test significant"
  col  <- "red"
  h    <- 0.5
} else if(any(pval_right < 0.05, na.rm = TRUE)){
  label <- "Quantile deviations detected (red curves)\nCombined adjusted quantile test n.s."
  col  <- "red"
  h    <- 0.5
} else{
  label <- "No significant problems detected"
  col  <- "grey50"
  h    <- 0
}
eaxis(1, cex.axis = 0.8)
eaxis(2, at = seq(0, 1, 0.25), cex.axis = 0.8)

mtext("DHARMa residual vs. predicted", 3, 1.25 + h, font = 2)
mtext(label, 3, 0.25, col = col, cex = 0.75)
par(mfrow = c(1, 1))