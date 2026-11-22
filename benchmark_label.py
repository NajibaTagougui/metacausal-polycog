"""Benchmark under a dose-dependent effect-modifier label (the paper's central scenario).

    python benchmark_label.py --reps 20 --n 2600 --out results_label

The true effect is gated by the latent equol-producer trait (EQP_true), as in
simulate.py's additive scenario. Learners see only the observed label EQP, which is
  true        the trait itself
  nondiff     the trait with 10% random misclassification
  confounded  a spot-urine ratio label; daidzein depends on SES, not on treatment
  post        a spot-urine ratio label; daidzein rises with treatment (post-treatment)
For each replicate we record:
  * CATE accuracy (sqrt PEHE, Spearman) on 5,000 held-out people, per learner
  * the label contrast: AIPW effect in label=1 minus label=0, against the TRUE trait
    contrast tau(EQP_true=1) - tau(EQP_true=0); bias, 95% coverage, rejection rate
  * the AIPW average effect: bias and coverage (the label is in the adjustment set)
  * the BLP heterogeneity test from out-of-fold KG-CF predictions (one-sided)
  * the modifier-validity diagnostic: logistic regression of the label on
    log urinary daidzein; rejection at p < 0.05 flags an invalid label
"""
import argparse, time, warnings
from pathlib import Path
import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy.stats import spearmanr
from simulate import simulate, COVARIATES
import pipeline as P

warnings.filterwarnings("ignore")
MODIFIERS = ["EQP", "ENLH", "age", "female", "diabetes", "bmi", "edu_college", "phys_active"]
MOD_IDX = [COVARIATES.index(c) for c in MODIFIERS]
LEARNERS = ["KG-CausalForest", "CausalForest", "OLS-interaction", "S-learner", "X-learner"]


def one_run(n, seed, label, effect_scale=1.0, trait_baseline=0.0, learners=None):
    kw = dict(label=label, effect_scale=effect_scale, trait_baseline=trait_baseline)
    tr = simulate(n, seed=seed, **kw)
    te = simulate(5000, seed=10_000 + seed, **kw)
    X, T, y = tr[COVARIATES].values, tr["T"].values, tr["y"].values
    out = dict(label=label, n=n, seed=seed, label_prev=tr.EQP.mean(),
               label_agreement=(tr.EQP == tr.EQP_true).mean())
    if learners:   # CATE accuracy of the named learners only (adds baselines cheaply)
        models = P.fit_learners(X, T, y, seed=seed, which=learners, modifier_idx=MOD_IDX)
        for name in learners:
            th = np.ravel(models[name](te[COVARIATES].values))
            out[f"pehe_{name}"] = float(np.sqrt(np.mean((th - te.tau_true.values) ** 2)))
            out[f"rho_{name}"] = float(spearmanr(th, te.tau_true.values).correlation)
        return out

    nuis = P.crossfit_nuisances(X, T, y, seed=seed)
    ate, se = P.aipw_ate(nuis["psi"])
    out.update(ate_bias=ate - tr.tau_true.mean(), ate_cover=float(abs(ate - tr.tau_true.mean()) < 1.96 * se))

    true_c = tr.tau_true[tr.EQP_true == 1].mean() - tr.tau_true[tr.EQP_true == 0].mean()
    e1, s1 = P.aipw_ate(nuis["psi"][tr.EQP.values == 1])
    e0, s0 = P.aipw_ate(nuis["psi"][tr.EQP.values == 0])
    c, cse = e1 - e0, np.hypot(s1, s0)
    out.update(true_contrast=true_c, label_contrast=c, contrast_bias=c - true_c,
               contrast_cover=float(abs(c - true_c) < 1.96 * cse), contrast_reject=float(abs(c / cse) > 1.96))

    models = P.fit_learners(X, T, y, seed=seed, which=LEARNERS, modifier_idx=MOD_IDX)
    for name in LEARNERS:
        th = np.ravel(models[name](te[COVARIATES].values))
        out[f"pehe_{name}"] = float(np.sqrt(np.mean((th - te.tau_true.values) ** 2)))
        out[f"rho_{name}"] = float(spearmanr(th, te.tau_true.values).correlation) if te.tau_true.std() > 0 else np.nan

    tau_oof = P.oof_cate(X, T, y, seed=seed, learner="KG-CausalForest", modifier_idx=MOD_IDX)
    b = P.blp_test(tau_oof, nuis, T)
    out.update(blp_beta2=b["beta2"], blp_reject=float(b["beta2_p"] < 0.05),
               blp_reject_twosided=float(b["beta2_p_twosided"] < 0.05))

    fit = sm.Logit(tr.EQP.values, sm.add_constant(tr.log_daidzein.values)).fit(disp=0)
    out.update(diag_coef=fit.params[1], diag_reject=float(fit.pvalues[1] < 0.05))
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--reps", type=int, default=20)
    ap.add_argument("--n", type=int, default=2600)
    ap.add_argument("--labels", nargs="*", default=["true", "nondiff", "confounded", "post"])
    ap.add_argument("--out", default="results_label")
    ap.add_argument("--effect_scale", type=float, default=1.0, help="0 = no treatment effect (null arm)")
    ap.add_argument("--learners", nargs="*", default=None, help="only CATE accuracy for these learners")
    ap.add_argument("--trait_baseline", type=float, default=0.0, help="trait effect on the untreated outcome")
    a = ap.parse_args()
    Path(a.out).mkdir(exist_ok=True)
    rows = []
    for label in a.labels:
        for s in range(a.reps):
            t0 = time.time()
            rows.append(one_run(a.n, s, label, a.effect_scale, a.trait_baseline, a.learners))
            print(f"{label} rep {s} done in {time.time() - t0:.0f}s", flush=True)
            pd.DataFrame(rows).to_csv(Path(a.out) / "label_raw.csv", index=False)
    R = pd.DataFrame(rows)
    summ = R.drop(columns=["seed", "n"]).groupby("label", sort=False).mean().T
    summ.to_csv(Path(a.out) / "label_summary.csv")
    print(summ.round(3).to_string())
