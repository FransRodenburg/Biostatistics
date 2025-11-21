## 1

import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import linregress

def plot_with_optional_smoother(x, y):
    plt.scatter(x, y, label="data")

    try:
        slope, intercept, *_ = linregress(x, y)
        xs = np.linspace(min(x), max(x), 200)
        ys = intercept + slope * xs
        plt.plot(xs, ys, label="smoother")
    except Exception as e:
        print(f"Warning: could not compute smoothing line: {e}")

    plt.legend()
    plt.show()

# Normal run
x = np.random.randn(50)
y = x + np.random.randn(50) * 0.3
plot_with_optional_smoother(x, y)

# Failure run (degenerate x)
x = np.zeros(50)
y = np.random.randn(50)
plot_with_optional_smoother(x, y)


## 2
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import UnivariateSpline
from scipy.stats import linregress

def fit_spline_with_fallback(x, y):
    try:
        # Try a smoothing spline
        spline = UnivariateSpline(x, y, s=0.5)
        print("Spline model fitted.")
        return spline(x)
    except Exception as e:
        print(f"Spline failed: {e}")
        print("Falling back to linear regression.")
        slope, intercept, *_ = linregress(x, y)
        return intercept + slope * x

# Normal run (spline works)
x = np.linspace(0, 10, 50)
y = np.sin(x) + np.random.randn(50) * 0.1

print("=== Normal run ===")
yhat = fit_spline_with_fallback(x, y)

plt.scatter(x, y, label="data")
plt.plot(x, yhat, label="fit")
plt.legend()
plt.show()


# Failure run (spline guaranteed to fail)
# e.g. duplicate x-values (UnivariateSpline requires strictly increasing x)

x_bad = np.concatenate([np.linspace(0, 5, 25), np.linspace(2, 4, 25)])
y_bad = np.random.randn(50)

print("=== Failure run ===")
yhat_bad = fit_spline_with_fallback(x_bad, y_bad)

plt.scatter(x_bad, y_bad, label="data")
plt.plot(x_bad, yhat_bad, label="fallback fit")
plt.legend()
plt.show()


## 3
import pandas as pd

def robust_read(path):
    try:
        print("Trying CSV.")
        return pd.read_csv(path)
    except Exception:
        try:
            print("CSV failed. Trying Excel.")
            return pd.read_excel(path)
        except Exception:
            raise ValueError("Could not read file as CSV or Excel.")

# First, create two minimal examples:
pd.DataFrame({"a": [1, 2, 3]}).to_csv("test.csv", index=False)
pd.DataFrame({"a": [1, 2, 3]}).to_excel("test.xlsx", index=False)

# Normal run (CSV file)
df = pd.DataFrame({"a": [1, 2, 3]})
df.to_csv("test.csv", index=False)
robust_read("test.csv")

# Failure run (CSV fails, Excel succeeds)
df = pd.DataFrame({"a": [1, 2, 3]})
df.to_excel("test.xlsx", index=False)
robust_read("test.xlsx")


## 4
import requests
import time

def get_with_retry(url, retries=3):
    for attempt in range(1, retries + 1):
        try:
            print(f"Attempt {attempt}")
            return requests.get(url, timeout=2)
        except requests.exceptions.RequestException as e:
            print(f"Failed: {e}")
            time.sleep(1)
    raise RuntimeError("All attempts failed.")

# Normal run
get_with_retry("https://httpbin.org/get")

# Failure run
get_with_retry("http://nonexistent.example.abc")


## 5
def parse_number(x):
    try:
        return float(x)
    except ValueError:
        try:
            return float(x.replace(",", "."))
        except Exception:
            return None

# Normal run
print(parse_number("12.5"))

# Fallback run
print(parse_number("12,5"))

# Final fallback
print(parse_number("abc"))

## 6
def pipeline_step(fn, data):
    try:
        return fn(data)
    except Exception as e:
        print(f"Step skipped due to error: {e}")
        return data

# Normal run
print(pipeline_step(lambda d: d * 2, 10))

# Failure run
print(pipeline_step(lambda d: d / 0, 10))

