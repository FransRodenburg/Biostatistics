# Store this script as well as the data set in the same location, then go to:
# Session > Set Working Directory > To Source File Location
DF <- read.csv("example_data_simple_linear_regression.csv")

# Fit a linear model
LM <- lm(height ~ weight, data = DF)

# Create an empty canvas
plot(NA, bty = "n", axes = FALSE, 
     xlab = "Weight (kg)",
     ylab = "Height (cm)",
     xlim = c(0, 160), xaxs = "i",
     ylim = c(125, 215), yaxs = "i")

# Create transparent colors
coralalpha <- rgb(t(col2rgb("coral")), alpha = 128, maxColorValue = 255)
steelbluealpha <- rgb(t(col2rgb("steelblue")), alpha = 128, maxColorValue = 255)

# Add prediction band
x_values <- seq(0, 160, 0.1)
newdata  <- data.frame(weight = x_values)
y_pred   <- predict(LM, newdata, interval = "pred")
polygon(c(x_values, rev(x_values)),
        c(y_pred[, 2], rev(y_pred[, 3])), col = coralalpha, border = NA)

# Add confidence band
y_conf   <- predict(LM, newdata, interval = "conf")
polygon(c(x_values, rev(x_values)),
        c(y_conf[, 2], rev(y_conf[, 3])), col = "white", border = NA)
polygon(c(x_values, rev(x_values)),
        c(y_conf[, 2], rev(y_conf[, 3])), col = steelbluealpha, border = NA)

# Add the regression line
abline(coef(LM), lwd = 2, col = "steelblue")

# Add the data points on top
points(height ~ weight, DF, pch = 19)

# Add axes
axis(1)
axis(2, las = 1)
