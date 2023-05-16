# The number of Monte Carlo simulations
MC <- 10000

k <- 0
l <- 0
for(i in 1:MC){
  # Draw two random bubbles from a uniform distribution
  x <- runif(2, -5, 5)
  
  # Confidence interval based on the sampling distribution
  CI1 <- mean(x) + c(-1, 1) * (5 - 5 / sqrt(2))
  contained <- (CI1[1] < 0 & CI1[2] > 0)
  k <- k + contained
  
  # Non-parametric confidence interval (distance between bubbles)
  CI2 <- mean(x) + c(-1, 1) * abs(diff(x)) / 2
  contained <- (CI2[1] < 0 & CI2[2] > 0)
  l <- l + contained
}

cat("Method 1:", paste0(round(k/MC * 100, 1), "%\n"))
cat("Method 2:", paste0(round(l/MC * 100, 1), "%"))