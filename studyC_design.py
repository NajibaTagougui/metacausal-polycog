"""Study C design simulation: which design choices make H1 and H2 answerable?

    python studyC_design.py --reps 500 --out results_studyC

SIMULATED DATA ONLY. No Study C participant data exist; every number this script
produces is a property of the assumed data-generating process below, which is
calibrated to what Study A (NHANES 1999-2002) showed.

Outcome: 2-year change in DSST (residual SD 8 points, assumed).
True metabotypes: equol producer (EQP, prevalence 30%), enterolactone-dominant
(ENLH, 40%). True effect of high intake on the 2-year change:
    tau = delta * (EQP + 0.6*ENLH + 0.4*EQP*ENLH)
(the protocol's additive pattern rescaled so EQP-only = delta; both = 2*delta).

Design factors
  classification  'spontaneous' : equol status from a spot urine, as in NHANES.
                    25% of people ate soy recently and are classified correctly
                    (sens = spec = 0.90). The rest: producers -> producer 70% /
                    unclassifiable 30%; non-producers -> producer 50% (equol at the
                    detection limit gives a ratio above the cut) / unclassifiable 50%.
                    Reproduces Study A: ~55% "producers", ~35% unclassifiable, and
                    non-producers seen only among high-soy eaters.
                  'challenge'   : status after a standardised soy challenge,
                    sens = spec = 0.95, nobody unclassifiable.
  exposure        'single_recall' : one 24-h recall, correlation 0.45 with usual intake
                  'repeated'      : repeated recalls + urinary biomarker, correlation 0.80
                  'randomised'    : supplement randomised 1:1 (no confounding, no error)
  enrichment      'none'  : metabotypes at population prevalence
                  'balanced' : screened and recruited 50% producers / 50% non-producers
Analysis: OLS with treatment x metabotype interactions, adjusted for the confounder
(SES) and age; H1 = T x EQP interaction; H2 = effect in both-favourable minus
neither cell. Power = share of replicates with two-sided p < 0.05.
"""
import argparse, itertools
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import norm

ap = argparse.ArgumentParser()
ap.add_argument("--reps", type=int, default=500)
ap.add_argument("--sizes", type=int, nargs="*", default=[600, 1000, 1500, 2600])
ap.add_argument("--deltas", type=float, nargs="*", default=[1.5, 3.0])
ap.add_argument("--out", default="results_studyC")
ap.add_argument("--sd", type=float, default=8.0, help="residual SD of the 2-year DSST change")
ap.add_argument("--p_eqp", type=float, default=0.30, help="population producer prevalence")
ap.add_argument("--acc", type=float, default=0.95, help="soy-challenge sensitivity = specificity")
ap.add_argument("--classifications", nargs="*", default=["spontaneous", "challenge"])
ap.add_argument("--exposures", nargs="*", default=["single_recall", "repeated", "randomised"])
ap.add_argument("--enrichments", nargs="*", default=["none", "balanced"])
a = ap.parse_args()


def simulate(n, delta, classification, exposure, enrichment, rng):
    ses = rng.normal(0, 1, n)
    age = rng.uniform(60, 85, n)
    p_eqp = 0.5 if enrichment == "balanced" else a.p_eqp
    eqp = rng.binomial(1, p_eqp, n)
    enlh = rng.binomial(1, 0.40, n)
    if exposure == "randomised":
        T = rng.binomial(1, 0.5, n); T_obs = T
    else:
        z = 0.5 * ses + rng.normal(0, np.sqrt(0.75), n)            # usual intake, confounded by SES
        T = (z > np.quantile(z, 2 / 3)).astype(int)
        r = 0.45 if exposure == "single_recall" else 0.80
        z_obs = r * z + np.sqrt(1 - r ** 2) * rng.normal(0, 1, n)
        T_obs = (z_obs > np.quantile(z_obs, 2 / 3)).astype(int)
    tau = delta * (eqp + 0.6 * enlh + 0.4 * eqp * enlh)
    y = -0.15 * (age - 70) + 1.5 * ses + tau * T + rng.normal(0, a.sd, n)

    if classification == "challenge":
        flip = rng.random(n) > a.acc
        eqp_obs = np.where(flip, 1 - eqp, eqp).astype(float)
    else:
        soy = rng.random(n) < 0.25
        u = rng.random(n)
        eqp_obs = np.full(n, np.nan)
        correct = np.where(rng.random(n) < 0.90, eqp, 1 - eqp)
        eqp_obs[soy] = correct[soy]
        prod, non = ~soy & (eqp == 1), ~soy & (eqp == 0)
        eqp_obs[prod & (u < 0.70)] = 1
        eqp_obs[non & (u < 0.50)] = 1
    enlh_obs = np.where(rng.random(n) > 0.95, 1 - enlh, enlh)
    keep = ~np.isnan(eqp_obs)
    return y[keep], T_obs[keep], eqp_obs[keep], enlh_obs[keep], ses[keep], age[keep]


def ols(X, y):
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    resid = y - X @ beta
    s2 = resid @ resid / (len(y) - X.shape[1])
    return beta, s2 * np.linalg.pinv(X.T @ X)


def analyse(y, T, E, L, ses, age):
    one = np.ones_like(y)
    # H1: y ~ T + E + L + T:E + T:L + ses + age
    X1 = np.column_stack([one, T, E, L, T * E, T * L, ses, age])
    b, V = ols(X1, y)
    z1 = b[4] / np.sqrt(V[4, 4])
    # H2: cell-specific treatment effects; both (E=1,L=1) minus neither (E=0,L=0)
    cells = [(E == e) & (L == l) for e in (0, 1) for l in (0, 1)]
    if min(c.sum() for c in cells) < 10:
        return z1, np.nan, int(cells[0].sum())
    C = np.column_stack(cells).astype(float)
    X2 = np.column_stack([C, C * T[:, None], ses, age])
    b, V = ols(X2, y)
    c = np.zeros(X2.shape[1]); c[7], c[4] = 1, -1                 # T x both  -  T x neither
    z2 = (c @ b) / np.sqrt(c @ V @ c)
    return z1, z2, int(cells[0].sum())


rows = []
grid = list(itertools.product(a.sizes, a.deltas, a.classifications, a.exposures, a.enrichments))
for i, (n, d, cl, ex, en) in enumerate(grid):
    rng = np.random.default_rng(1000 + i)
    res = [analyse(*simulate(n, d, cl, ex, en, rng)) for _ in range(a.reps)]
    z1, z2, nn = map(np.array, zip(*res))
    crit = norm.ppf(0.975)
    rows.append(dict(sd=a.sd, p_eqp=a.p_eqp, acc=a.acc, n_recruited=n, delta=d, classification=cl, exposure=ex, enrichment=en,
                     power_H1=np.mean(np.abs(z1) > crit),
                     power_H2=np.nanmean(np.abs(z2) > crit) if np.isfinite(z2).any() else np.nan,
                     mean_n_neither=nn.mean()))
R = pd.DataFrame(rows)
out = Path(a.out); out.mkdir(exist_ok=True)
R.to_csv(out / "studyC_power.csv", index=False)
pd.set_option("display.width", 200)
print(R.round(2).to_string(index=False))
