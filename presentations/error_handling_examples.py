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
import statsmodels.api as sm
import numpy as np

def fit_model_with_fallback(y, X_full, X_simple):
    try:
        model = sm.OLS(y, sm.add_constant(X_full)).fit()
        print("Full model fitted.")
        return model
    except Exception as e:
        print(f"Full model failed: {e}")
        print("Trying simpler model.")
        model = sm.OLS(y, sm.add_constant(X_simple)).fit()
        print("Simpler model fitted.")
        return model

# Normal run
n = 30
y = np.random.randn(n)
X_full = np.column_stack([np.random.randn(n), np.random.randn(n)])
X_simple = X_full[:, [0]]
fit_model_with_fallback(y, X_full, X_simple)

# Failure run (rank-deficient full model)
n = 30
y = np.random.randn(n)
bad_col = np.ones(n)
X_full = np.column_stack([bad_col, bad_col])
X_simple = bad_col.reshape(-1, 1)
fit_model_with_fallback(y, X_full, X_simple)

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

