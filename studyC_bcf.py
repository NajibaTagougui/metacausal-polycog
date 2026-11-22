"""Study C recommended design analysed with a CATE learner (BCF) instead of OLS.

    python studyC_bcf.py --reps 100 --out results_studyC_bcf

SIMULATED DATA ONLY. Same data-generating process as studyC_design.py for the
recommended design: soy-challenge classification (sensitivity = specificity = acc),
randomised exposure, balanced enrolment (50% producers), ENLH 40% (5% error).
H1 is the producer contrast of treatment effects.
  OLS : t-test of the T x EQP interaction (as in studyC_design.py)
  BCF : posterior of mean CATE among labelled producers minus labelled non-producers;
        reject when the 95% credible interval excludes 0. Propensity is known (0.5).
"""
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
from stochtree import BCFModel

ap = argparse.ArgumentParser()
ap.add_argument("--reps", type=int, default=100)
ap.add_argument("--sizes", type=int, nargs="*", default=[600, 1000, 1500])
ap.add_argument("--sds", type=float, nargs="*", default=[8.0, 13.0])
ap.add_argument("--delta", type=float, default=3.0)
ap.add_argument("--acc", type=float, default=0.95)
ap.add_argument("--out", default="results_studyC_bcf")
a = ap.parse_args()
out = Path(a.out); out.mkdir(exist_ok=True)


def draw(n, sd, rng):
    ses, age = rng.normal(0, 1, n), rng.uniform(60, 85, n)
    eqp, enlh = rng.binomial(1, 0.5, n), rng.binomial(1, 0.4, n)
    T = rng.binomial(1, 0.5, n)
    tau = a.delta * (eqp + 0.6 * enlh + 0.4 * eqp * enlh)
    y = -0.15 * (age - 70) + 1.5 * ses + tau * T + rng.normal(0, sd, n)
    E = np.where(rng.random(n) > a.acc, 1 - eqp, eqp)
    L = np.where(rng.random(n) > 0.95, 1 - enlh, enlh)
    return y, T, E, L, ses, age


def ols_z(y, T, E, L, ses, age):
    X = np.column_stack([np.ones_like(y), T, E, L, T * E, T * L, ses, age])
    b, *_ = np.linalg.lstsq(X, y, rcond=None)
    r = y - X @ b
    V = r @ r / (len(y) - X.shape[1]) * np.linalg.pinv(X.T @ X)
    return b[4] / np.sqrt(V[4, 4])


rows = []
for sd in a.sds:
    for n in a.sizes:
        rng = np.random.default_rng(int(1000 * sd + n))
        rej_ols = rej_bcf = 0
        for r in range(a.reps):
            y, T, E, L, ses, age = draw(n, sd, rng)
            rej_ols += abs(ols_z(y, T, E, L, ses, age)) > 1.96
            X = np.column_stack([E, L, ses, age])
            m = BCFModel()
            m.sample(X_train=X, Z_train=T.astype(float), y_train=y, propensity_train=np.full(n, 0.5),
                     num_gfr=10, num_burnin=0, num_mcmc=200, general_params={"random_seed": r})
            tau = np.asarray(m.predict(X, np.ones(n), propensity=np.full(n, 0.5), terms="cate")).reshape(n, -1)
            c = tau[E == 1].mean(axis=0) - tau[E == 0].mean(axis=0)
            lo, hi = np.percentile(c, [2.5, 97.5])
            rej_bcf += (lo > 0) or (hi < 0)
        rows.append(dict(sd=sd, n=n, power_ols=rej_ols / a.reps, power_bcf=rej_bcf / a.reps))
        print(rows[-1], flush=True)
R = pd.DataFrame(rows)
R.to_csv(out / "studyC_bcf.csv", index=False)
print(R.to_string(index=False))
