"""Comparison with a latent-class approach to the misclassified modifier.

    python latent_class_comparison.py --reps 50 --out results_latent

Latent-class methods treat the trait as an unobserved class and use the continuous
biomarkers instead of a thresholded label. Here: a two-component Gaussian mixture on
(log urinary daidzein, log urinary equol), whose components match the biology (producers:
equol proportional to daidzein; non-producers: background equol unrelated to daidzein).
The posterior P(producer | biomarkers) replaces the label. Following the mixture
identification of Lan, Kallus & Chou (2026), each person's AIPW pseudo-outcome satisfies
E[psi | p] = tau_0 + p * (tau_1 - tau_0) when the effect depends on the class only, so the
slope of psi on the posterior p estimates the trait contrast (OLS, HC1 SE).

Compared with the thresholded-label AIPW contrast on the same data, in the label
scenarios of benchmark_label.py (SES-driven and treatment-driven daidzein; trait
prognostic or not). The mixture uses biomarkers measured after exposure, so when
exposure moves them the posterior is a descendant of treatment too.
"""
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import statsmodels.api as sm
from sklearn.mixture import GaussianMixture
from simulate import simulate, COVARIATES
import pipeline as P

ap = argparse.ArgumentParser()
ap.add_argument("--reps", type=int, default=50)
ap.add_argument("--n", type=int, default=2600)
ap.add_argument("--out", default="results_latent")
a = ap.parse_args()
out = Path(a.out); out.mkdir(exist_ok=True)

SCEN = [("trait gates effect only", 0.0), ("prognostic trait, +10", 10.0), ("prognostic trait, -10", -10.0)]
rows = []
for scen, tb in SCEN:
    for label in ["confounded", "post"]:
        for s in range(a.reps):
            tr = simulate(a.n, seed=s, label=label, trait_baseline=tb)
            X, T, y = tr[COVARIATES].values, tr["T"].values, tr["y"].values
            psi = P.crossfit_nuisances(X, T, y, seed=s)["psi"]
            true_c = tr.tau_true[tr.EQP_true == 1].mean() - tr.tau_true[tr.EQP_true == 0].mean()
            e1, s1 = P.aipw_ate(psi[tr.EQP.values == 1]); e0, s0 = P.aipw_ate(psi[tr.EQP.values == 0])
            Z = tr[["log_daidzein", "log_equol"]].values
            gm = GaussianMixture(2, covariance_type="full", n_init=3, random_state=s).fit(Z)
            corr = [c[0, 1] / np.sqrt(c[0, 0] * c[1, 1]) for c in gm.covariances_]
            k = int(np.argmax(corr))                       # producer component: equol tracks daidzein
            p = gm.predict_proba(Z)[:, k]
            fit = sm.OLS(psi, sm.add_constant(p)).fit(cov_type="HC1")
            b, se = fit.params[1], fit.bse[1]
            rows.append(dict(scenario=scen, label=label, seed=s, true_contrast=true_c,
                             label_contrast=e1 - e0, label_cover=float(abs(e1 - e0 - true_c) < 1.96 * np.hypot(s1, s0)),
                             lc_contrast=b, lc_cover=float(abs(b - true_c) < 1.96 * se), lc_se=se,
                             posterior_auc=float(pd.Series(p).rank().pipe(lambda r: (r[tr.EQP_true.values == 1].sum()
                                 - (tr.EQP_true == 1).sum() * ((tr.EQP_true == 1).sum() + 1) / 2)
                                 / ((tr.EQP_true == 1).sum() * (tr.EQP_true == 0).sum()))),
                             posterior_mean_by_T=float(p[T == 1].mean() - p[T == 0].mean())))
            print(scen, label, s, flush=True)
        pd.DataFrame(rows).to_csv(out / "latent_raw.csv", index=False)
D = pd.DataFrame(rows)
S = D.assign(label_bias=D.label_contrast - D.true_contrast, lc_bias=D.lc_contrast - D.true_contrast) \
     .groupby(["scenario", "label"], sort=False)[["true_contrast", "label_contrast", "label_bias", "label_cover",
                                                  "lc_contrast", "lc_bias", "lc_cover", "lc_se", "posterior_auc",
                                                  "posterior_mean_by_T"]].mean()
S.to_csv(out / "latent_summary.csv")
pd.set_option("display.width", 220)
print(S.round(3).to_string())
