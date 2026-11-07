import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats
import statsmodels.api as sm
from patsy import build_design_matrices, dmatrix
from statsmodels.regression.quantile_regression import QuantReg
from statsmodels.stats.multitest import multipletests

# Data
path = "example-binomial-data-01.csv" # Choose 01, 02, or 03:
                                      #  1. No violations
                                      #  2. Non-linear (on the scale of eta)
                                      #  3. Overdispersed

# To disable the bootstrapped outlier test, set to False.
bootstrap = True
nsim      = 1000
seed      = 2026

# Python does not have an equivalent of R's qgam, so I opted for a natural
# restricted spline inside QuantReg. For this you must choose an appropriate
# degrees of freedom (i.e., flexibility). Leave at 3 if unsure.
nu = 3

# Helper functions
def normrank(x):
    r = stats.rankdata(x, method='average')
    return r / r.max()

# Two-sided p-value of a simulation test, as in DHARMa
def two_sided(simulated, observed):
    return min(2 * min(np.mean(simulated <= observed), np.mean(simulated >= observed)), 1)

def fit_binomial_glm(y, n, x):
    ratio = np.column_stack((y, n - y))
    model = sm.GLM(ratio, x, family=sm.families.Binomial(link=sm.families.links.Logit()))
    return model.fit()

def simulate_binomial_from_mu(mu, n, nsim, rng):
    mu = np.asarray(mu)
    sims = rng.binomial(n=n, p=mu[:, None], size=(len(mu), nsim))
    return sims


# Read data
df = pd.read_csv(path)
x = df['x'].values
y = df['y'].values
n = len(y)
size = 10 # binomial trials, should be evident from study design

# Fit a GLM
X        = sm.add_constant(pd.DataFrame({'x': x}))
GLM      = fit_binomial_glm(y, size, X)
yhat     = GLM.fittedvalues  # numpy array-like
yhatrank = normrank(yhat)

# Monte Carlo simulated outcomes
rng  = np.random.default_rng(seed)
nsim = nsim
sim  = simulate_binomial_from_mu(np.asarray(yhat), size, nsim, rng)  # shape (n, nsim)

# Simulated randomized residuals
SRR = np.empty(n, dtype=float)
one = np.zeros(n, dtype=int)
zero = np.zeros(n, dtype=int)
for i in range(n):
    sims_i = sim[i, :]
    lwr = np.mean(sims_i < y[i])
    upr = np.mean(sims_i <= y[i])
    if upr == 0:
        zero[i] = 1
    if lwr == 1:
        one[i] = 1
    SRR[i] = rng.uniform(lwr, upr)

# DHARMa's distributional test
ks_stat, ks_p = stats.kstest(SRR, 'uniform')
KS = ks_p

# DHARMa's overdispersion test
flattened_var_sim = np.var(sim.flatten(), ddof=1)
simVar = np.array([np.var(sim[:, j] - yhat, ddof=1) / flattened_var_sim for j in range(nsim)])
obsVar = np.var(y - yhat, ddof=1) / flattened_var_sim
OD = two_sided(simVar, obsVar)

# DHARMa's bootstrapped outlier test
if bootstrap:
    B = nsim
    one_zero_boot = np.empty(B)
    wh = np.arange(nsim)
    for i in range(B):
        sel = rng.choice(wh[wh != i], size = nsim, replace = True)
        current = sim[:, sel]   # shape (n, nsim)
        SRR_boot = np.empty(n)
        for j in range(n):
            lwr = np.mean(current[j, :] < sim[j, i])
            upr = np.mean(current[j, :] <= sim[j, i])
            SRR_boot[j] = rng.uniform(lwr, upr)
        one_zero_boot[i] = np.mean((SRR_boot <= 0) | (SRR_boot >= 1))
    one_zero_real = np.mean((SRR <= 0) | (SRR >= 1))
    BO = two_sided(one_zero_boot, one_zero_real)
else:
    BO = np.nan

# FDR adjustment
# (For some reason, DHARMa does not use this for the figure on the left...)
pval_left = [KS, OD, BO]
pvals_for_fdr = [v for v in pval_left if not np.isnan(v)]
if len(pvals_for_fdr) > 0:
    rej, p_adj, _, _ = multipletests(pvals_for_fdr, method='fdr_bh')
    FDR_left = []
    idx = 0
    for v in pval_left:
        if np.isnan(v):
            FDR_left.append(np.nan)
        else:
            FDR_left.append(float(p_adj[idx]))
            idx += 1
else:
    FDR_left = [np.nan] * len(pval_left)

# QQ-plot
fig, axes = plt.subplots(1, 2, figsize=(12, 5))
ax = axes[0]
theor = np.linspace(1 / (n + 1), n / (n + 1), n) # Uniform order statistics
sr_sorted = np.sort(SRR)
ax.scatter(theor, sr_sorted, s=16)
ax.plot([0, 1], [0, 1], color='red')
fmt_p_left = [f"{v:.3f}" if not np.isnan(v) else "NA" for v in pval_left]
ax.text(0.1, 0.875, f"KS test p = {fmt_p_left[0]}", color=('red' if KS < 0.05 else 'black'), transform=ax.transAxes, fontsize=9)
ax.text(0.1, 0.8, f"Dispersion test p = {fmt_p_left[1]}", color=('red' if OD < 0.05 else 'black'), transform=ax.transAxes, fontsize=9)
ax.text(0.1, 0.725, f"Outlier test p = {fmt_p_left[2]}", color=('red' if (not np.isnan(BO) and BO < 0.05) else 'black'), transform=ax.transAxes, fontsize=9)
ax.set_xlabel("Expected")
ax.set_ylabel("Observed")
ax.set_title("QQ-plot residuals")
if any([(f is not None and (not np.isnan(f)) and f < 0.05) for f in pval_left]) :
    label = "Significant problems detected"
    col = "red"
else:
    label = "No significant problems detected"
    col = "#7F7F7F"
ax.text(0.5, 0.95, label, ha='center', color = col, transform = ax.transAxes)

# Residuals vs fitted
ax2 = axes[1]
outlier = (one + zero) > 0
ax2.scatter(yhatrank[~outlier], SRR[~outlier], s = 16, alpha = 0.8)
ax2.scatter(yhatrank[outlier], SRR[outlier], s = 80, marker = '*', color = 'red')
ax2.set_xlabel("Model predictions (rank transformed)")
ax2.set_ylabel("DHARMa residual")
ax2.set_title("DHARMa residual vs. predicted")
ax2.set_ylim(0, 1)
ax2.set_yticks([0, 0.25, 0.5, 0.75, 1])

# horizontal quantile lines
p = [0.25, 0.5, 0.75]
for q in p:
    ax2.axhline(q, linestyle="dashed", color="grey", linewidth=1, alpha=0.7)

pval_right = []

newx = np.linspace(0, 1, 1000)

# A centered spline plus an intercept (nu parameters in total)
X_spline = dmatrix(f"cr(x, df={nu - 1}, constraints='center', lower_bound=0, upper_bound=1)", {"x": yhatrank})
Xnew     = np.asarray(build_design_matrices([X_spline.design_info], {"x": newx})[0])
X_spline = np.asarray(X_spline)

for q in p:
    y_q = SRR - q

    # fit Quantile Regression at quantile q
    qr = QuantReg(y_q, X_spline)
    Qi = qr.fit(q = q)

    # Test the intercept and, jointly, the spline coefficients, as DHARMa does
    # (with fewer than 20 observations, statsmodels cannot compute these tests)
    try:
        p_pv = Qi.pvalues[0]
        s_pv = float(Qi.f_test(np.eye(nu)[1:]).pvalue)
        _, pv_pair_adj, _, _ = multipletests([p_pv, s_pv], method="fdr_bh")
        pval_q = float(np.min(pv_pair_adj))
    except ValueError:
        pval_q = np.nan
    pval_right.append(pval_q)
    red_flag = (pval_q < 0.05)

    ynew = Xnew @ Qi.params

    # Compute standard errors from covariance matrix
    covb = Qi.cov_params()
    XV = Xnew @ covb
    SE = np.sqrt(np.sum(XV * Xnew, axis = 1))

    # 95% confidence band, as in DHARMa 0.5.0
    lwr = ynew - 1.96 * SE
    upr = ynew + 1.96 * SE

    # add back the quantile baseline (we fitted SRR - q)
    ynew += q
    lwr  += q
    upr  += q

    color = "red" if red_flag else "black"
    ax2.fill_between(newx, lwr, upr, color = (1, 0, 0, 0.1) if red_flag else (0, 0, 0, 0.1), edgecolor = None)
    ax2.plot(newx, ynew, color = color, linewidth = 2)

# FDR adjustment across the 3 quantile tests
pval_right = np.array(pval_right)
if np.isnan(pval_right).any():
    p_adj2 = np.array([np.nan])
else:
    _, p_adj2, _, _ = multipletests(pval_right, method='fdr_bh')
if np.isnan(pval_right).any():
    label_right = "Some quantile regressions failed"
    col_right = "red"
elif any(p_adj2 < 0.05):
    label_right = "Quantile deviations detected (red curves)\nCombined adjusted quantile test significant"
    col_right = "red"
elif any(pval_right < 0.05):
    label_right = "Quantile deviations detected (red curves)\nCombined adjusted quantile test n.s."
    col_right = "red"
else:
    label_right = "No significant problems detected"
    col_right = "#7F7F7F"
ax2.text(0.5, 0.95, label_right, ha='center', color = col_right, transform = ax2.transAxes)

plt.show()
