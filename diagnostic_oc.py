"""Operating characteristics of the modifier-validity diagnostic.

    python diagnostic_oc.py --reps 500 --out results_diagnostic

Data: latent trait M ~ Bernoulli(0.3); log dose L ~ N(log 30, 1.5) (natural log, as in
simulate.py). The observed label follows
    logit P(M* = 1 | M, L) = a + 3 M + g(L)
so g = 0 is a valid label. Scenarios for g (z = log10 dose, centred):
    linear     gamma * z, with gamma = log(OR per tenfold)  for OR in 1, 0.8, 0.67, 0.5, 0.25, 0.05
    U-shaped   0.8 * z^2 - mean, a non-monotone dependence with no linear trend
    valid, trait tied to intake: g = 0 but M depends on L (producers eat more soy),
               which breaks the testable implication's premise that M is independent of L
Tests (alpha = 0.05):
    linear    Wald test of the slope in  label ~ log10 dose
    spline    likelihood-ratio test of a 4-df natural cubic spline against the null model
    deciles   likelihood-ratio test of 9 decile indicators against the null model
"""
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import statsmodels.api as sm
from patsy import dmatrix
from scipy.stats import chi2

ap = argparse.ArgumentParser()
ap.add_argument("--reps", type=int, default=500)
ap.add_argument("--sizes", type=int, nargs="*", default=[300, 500, 1000, 2600, 5000])
ap.add_argument("--out", default="results_diagnostic")
a = ap.parse_args()
out = Path(a.out); out.mkdir(exist_ok=True)

SCEN = [(f"linear OR={o}", "linear", o) for o in (1.0, 0.9, 0.8, 0.75, 0.67, 0.5, 0.25, 0.05)] + \
       [("U-shaped", "u", None), ("valid label, trait tied to intake", "tied", None)]


def draw(n, kind, orr, rng):
    L = rng.normal(np.log(30), 1.5, n)
    z = (L - np.log(30)) / np.log(10)
    pm = 0.3 if kind != "tied" else 1 / (1 + np.exp(-(-0.85 + 0.8 * z)))
    M = rng.binomial(1, pm)
    g = np.zeros(n)
    if kind == "linear":
        g = np.log(orr) * z
    elif kind == "u":
        g = 0.8 * (z ** 2 - np.mean(z ** 2))
    p = 1 / (1 + np.exp(-(-1.0 + 3 * M + g)))
    return rng.binomial(1, p), z


def lr(y, X1):
    ll0 = sm.Logit(y, np.ones((len(y), 1))).fit(disp=0).llf
    f1 = sm.Logit(y, X1).fit(disp=0)
    return chi2.sf(2 * (f1.llf - ll0), X1.shape[1] - 1)


rows = []
for name, kind, orr in SCEN:
    for n in a.sizes:
        rng = np.random.default_rng(hash((name, n)) % 2**32)
        rej = np.zeros(3)
        for _ in range(a.reps):
            y, z = draw(n, kind, orr, rng)
            try:
                p_lin = sm.Logit(y, sm.add_constant(z)).fit(disp=0).pvalues[1]
                S = np.asarray(dmatrix("cr(z, df=4)", {"z": z}, return_type="dataframe"))
                p_spl = lr(y, S)
                dec = pd.qcut(pd.Series(z).rank(method="first"), 10, labels=False)
                D = sm.add_constant(pd.get_dummies(dec, drop_first=True, dtype=float).values)
                p_dec = lr(y, D)
                rej += np.array([p_lin, p_spl, p_dec]) < 0.05
            except Exception:
                pass
        rows.append(dict(scenario=name, n=n, linear=rej[0] / a.reps, spline=rej[1] / a.reps,
                         deciles=rej[2] / a.reps))
    print(name, "done", flush=True)
R = pd.DataFrame(rows)
R.to_csv(out / "diagnostic_oc.csv", index=False)
pd.set_option("display.width", 200)
print(R.pivot(index="scenario", columns="n", values="linear").round(2).to_string())
print(R.round(2).to_string(index=False))
