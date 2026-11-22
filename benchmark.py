"""
Semi-synthetic benchmark: which learner recovers metabotype-driven responders?
Usage: python benchmark.py --reps 10 --sizes 600 2600 --out results
"""
import argparse, json, time, warnings
from pathlib import Path
import numpy as np
import pandas as pd
import statsmodels.api as sm
from simulate import simulate, COVARIATES
import pipeline as P

warnings.filterwarnings("ignore")
MODIFIERS = ["EQP", "ENLH", "age", "female", "diabetes", "bmi", "edu_college", "phys_active"]
MOD_IDX = [COVARIATES.index(c) for c in MODIFIERS]


def one_run(n, seed, scenario="additive", only=None):
    tr = simulate(n, seed=seed, scenario=scenario)
    te = simulate(5000, seed=10_000 + seed, scenario=scenario)            # large held-out set for PEHE
    X, T, y = tr[COVARIATES].values, tr["T"].values, tr["y"].values
    rows = []
    if only:   # just the named learners (used to add baselines without re-running the rest)
        models = P.fit_learners(X, T, y, seed=seed, which=only, modifier_idx=MOD_IDX)
        draws = models.pop("_bcf_draws", None)
        for k in [k for k in models if k.startswith("_")]:
            models.pop(k)
        for name, f in models.items():
            m = P.cate_metrics(f(te[COVARIATES].values), te.tau_true.values)
            if name == "BCF":
                d = draws(te[COVARIATES].values)
                lo, hi = np.percentile(d, [2.5, 97.5], axis=1)
                m["ci_coverage"] = float(np.mean((te.tau_true.values >= lo) & (te.tau_true.values <= hi)))
            rows.append(dict(learner=name, n=n, seed=seed, **m))
        return rows
    # conventional analysis: adjusted OLS ATE (what most nutrition papers report)
    ols = sm.OLS(y, sm.add_constant(np.column_stack([T, X]))).fit(cov_type="HC3")
    nuis = P.crossfit_nuisances(X, T, y, seed=seed)
    ate, ate_se = P.aipw_ate(nuis["psi"])
    rows.append(dict(learner="OLS-adjusted (ATE only)", n=n, seed=seed,
                     sqrt_pehe=float(np.sqrt(np.mean((ols.params[1] - te.tau_true) ** 2))),
                     spearman=np.nan, ate_bias=ols.params[1] - tr.tau_true.mean(),
                     precision_top20=np.nan))
    rows.append(dict(learner="AIPW (ATE only)", n=n, seed=seed,
                     sqrt_pehe=float(np.sqrt(np.mean((ate - te.tau_true) ** 2))),
                     spearman=np.nan, ate_bias=ate - tr.tau_true.mean(),
                     precision_top20=np.nan, ate_cover=abs(ate - tr.tau_true.mean()) < 1.96 * ate_se))
    models = P.fit_learners(X, T, y, seed=seed, modifier_idx=MOD_IDX)
    cf = models.pop("_cf_model"); kg = models.pop("_kgcf_model")
    for name, f in models.items():
        m = P.cate_metrics(f(te[COVARIATES].values), te.tau_true.values)
        rows.append(dict(learner=name, n=n, seed=seed, **m))
    # pointwise 95% CI coverage of both forests on held-out points
    Xte, tt = te[COVARIATES].values, te.tau_true.values
    for r in rows:
        if r["learner"] == "CausalForest":
            lo, hi = cf.effect_interval(Xte, alpha=0.05)
        elif r["learner"] == "KG-CausalForest":
            lo, hi = kg.effect_interval(Xte[:, MOD_IDX], alpha=0.05)
        else:
            continue
        r["ci_coverage"] = float(np.mean((tt >= lo) & (tt <= hi)))
    return rows


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--reps", type=int, default=10)
    ap.add_argument("--sizes", type=int, nargs="+", default=[600, 2600])
    ap.add_argument("--out", default="results")
    ap.add_argument("--scenario", default="additive")
    ap.add_argument("--start", type=int, default=0, help="first replicate seed (to extend an earlier run)")
    ap.add_argument("--only", nargs="*", default=None, help="run only these learners, e.g. BCF R-learner")
    ap.add_argument("--drop_modifier", default=None,
                    help="misspecified KG-CF: leave this true modifier out of the list")
    a = ap.parse_args()
    if a.drop_modifier:
        MODIFIERS.remove(a.drop_modifier)
        MOD_IDX[:] = [COVARIATES.index(c) for c in MODIFIERS]
    Path(a.out).mkdir(exist_ok=True)
    allrows = []
    for n in a.sizes:
        for s in range(a.start, a.reps):
            t0 = time.time()
            allrows += one_run(n, s, a.scenario, a.only)
            print(f"n={n} rep={s} done in {time.time()-t0:.1f}s", flush=True)
            pd.DataFrame(allrows).to_csv(f"{a.out}/benchmark_raw.csv", index=False)
    df = pd.DataFrame(allrows)
    if "ate_cover" in df:
        df["ate_cover"] = df["ate_cover"].map({True: 1.0, False: 0.0})   # bool/NaN object column broke the mean
    summ = df.groupby(["n", "learner"]).agg(["mean", "std"]).drop(columns="seed")
    summ.to_csv(f"{a.out}/benchmark_summary.csv")
    print(summ.round(3).to_string())
