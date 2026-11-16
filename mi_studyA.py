"""Multiple-imputation sensitivity analysis for Study A (all 878 eligible people).

    python mi_studyA.py --data data_studyA/nhanes_studyA_full.csv --m 20 --out sens_mi

Imputation: chained equations (sklearn IterativeImputer, posterior draws), one
completed dataset per seed. Continuous quantities are imputed and binary
variables derived from them where a definition exists:
  * exposure  : tot_eadj imputed -> T = tot_eadj >= the observed-data cut
  * EQP       : log10(equol/daidzein) imputed -> EQP = ratio > cut. When equol was
                below the detection limit, the true ratio is BELOW the one computed
                from the LOD/sqrt(2) fill, so the imputed ratio is capped there.
  * ENLH      : ENL fraction imputed -> ENLH = fraction >= observed-data quantile
  * other binaries: imputed on the continuous scale, clipped and rounded
The outcome is never missing. Analyses per dataset: AIPW average effect, H1 and H2
AIPW contrasts, BLP from out-of-fold KG causal forest CATEs. Pooled by Rubin's rules.
"""
import argparse, json, warnings
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import norm, t as tdist
from sklearn.experimental import enable_iterative_imputer  # noqa: F401
from sklearn.impute import IterativeImputer
import pipeline as P
from simulate import COVARIATES
from analyze import MODIFIERS

warnings.filterwarnings("ignore")
ap = argparse.ArgumentParser()
ap.add_argument("--data", required=True)
ap.add_argument("--m", type=int, default=20)
ap.add_argument("--eqp_cut", type=float, default=-1.75)
ap.add_argument("--enl_quantile", type=float, default=0.6)
ap.add_argument("--tertile", type=float, default=2 / 3)
ap.add_argument("--out", default="sens_mi")
a = ap.parse_args()
out = Path(a.out); out.mkdir(exist_ok=True)

full = pd.read_csv(a.data)
covs = [c for c in COVARIATES + ["caffeine_mg"] if c in full.columns]
mods = [c for c in MODIFIERS if c in covs]
midx = [covs.index(c) for c in mods]
binary = [c for c in covs if c not in ("EQP", "ENLH") and set(full[c].dropna().unique()) <= {0, 1}]

t_cut = full.tot_eadj.quantile(a.tertile)
enl_cut = full.enl_fraction.quantile(a.enl_quantile)
for c in ["URXDAZ", "URXEQU", "URXETL", "URXETD", "URXGNS"]:
    full["log_" + c] = np.log(full[c])
ratio_cap = np.where(full.eqp_status == "equol_below_LOD_ambiguous", full.log_eq_daz, np.inf)
full.loc[full.eqp_status.isin(["equol_below_LOD_ambiguous", "equol_missing", "daidzein_below_LOD"]), "log_eq_daz"] = np.nan

imp_cols = ([c for c in covs if c not in ("EQP", "ENLH")] + ["tot_eadj", "log_eq_daz", "enl_fraction", "y"] +
            ["log_URXDAZ", "log_URXEQU", "log_URXETL", "log_URXETD", "log_URXGNS"])
say = []
print(f"n={len(full)}, m={a.m}; missing before imputation: " +
      ", ".join(f"{c}={full[c].isna().sum()}" for c in imp_cols if full[c].isna().any()))

rows = []
for m in range(a.m):
    imp = IterativeImputer(sample_posterior=True, max_iter=15, random_state=m)
    Z = pd.DataFrame(imp.fit_transform(full[imp_cols]), columns=imp_cols)
    for c in binary:
        Z[c] = Z[c].clip(0, 1).round()
    Z["log_eq_daz"] = np.minimum(Z.log_eq_daz, ratio_cap)
    Z["EQP"] = (Z.log_eq_daz > a.eqp_cut).astype(float)
    Z["ENLH"] = (Z.enl_fraction >= enl_cut).astype(float)
    Z["T"] = (Z.tot_eadj >= t_cut).astype(int)
    X, T, y, w = Z[covs].values, Z["T"].values, full.y.values, full.w.values

    nuis = P.crossfit_nuisances(X, T, y, w, seed=0)
    ate = P.aipw_ate(nuis["psi"], w)
    est = lambda msk: P.aipw_ate(nuis["psi"][msk], w[msk])
    e1, e0 = est(Z.EQP.values == 1), est(Z.EQP.values == 0)
    both = est(((Z.EQP == 1) & (Z.ENLH == 1)).values)
    neither = est(((Z.EQP == 0) & (Z.ENLH == 0)).values)
    tau = P.oof_cate(X, T, y, w, seed=0, learner="KG-CausalForest", modifier_idx=midx)
    blp = P.blp_test(tau, nuis, T, w)
    rows.append(dict(m=m, n_EQP1=int(Z.EQP.sum()), n_EQP0=int((Z.EQP == 0).sum()),
                     n_neither=int(((Z.EQP == 0) & (Z.ENLH == 0)).sum()),
                     ATE=ate[0], ATE_se=ate[1],
                     H1=e1[0] - e0[0], H1_se=np.hypot(e1[1], e0[1]),
                     H2=both[0] - neither[0], H2_se=np.hypot(both[1], neither[1]),
                     BLP_beta2=blp["beta2"], BLP_beta2_se=blp["beta2_se"]))
    r = rows[-1]
    print(f"  imputation {m + 1:>2}: EQP 1/0 = {r['n_EQP1']}/{r['n_EQP0']}, ATE {r['ATE']:.2f}, "
          f"H1 {r['H1']:.2f}, H2 {r['H2']:.2f}, BLP b2 {r['BLP_beta2']:.2f}", flush=True)
R = pd.DataFrame(rows); R.to_csv(out / "mi_per_imputation.csv", index=False)


def rubin(q, se, one_sided=False):
    q, u = np.asarray(q), np.asarray(se) ** 2
    m = len(q); qbar, ubar, b = q.mean(), u.mean(), q.var(ddof=1)
    tv = ubar + (1 + 1 / m) * b
    r = (1 + 1 / m) * b / ubar if ubar > 0 else np.inf
    df = (m - 1) * (1 + 1 / r) ** 2 if b > 0 else np.inf
    z = qbar / np.sqrt(tv)
    p = tdist.sf(z, df) if one_sided else 2 * tdist.sf(abs(z), df)
    return dict(estimate=qbar, se=float(np.sqrt(tv)), df=float(df), p=float(p),
                frac_missing_info=float((1 + 1 / m) * b / tv))


res = {"n": len(full), "m": a.m, "covariates": covs,
       "ATE": rubin(R.ATE, R.ATE_se), "H1_EQP1_minus_EQP0": rubin(R.H1, R.H1_se),
       "H2_both_minus_neither": rubin(R.H2, R.H2_se),
       "BLP_beta2_onesided": rubin(R.BLP_beta2, R.BLP_beta2_se, one_sided=True),
       "mean_n_EQP1": R.n_EQP1.mean(), "mean_n_EQP0": R.n_EQP0.mean(), "mean_n_neither": R.n_neither.mean()}
res = json.loads(json.dumps(res, default=float))
json.dump(res, open(out / "mi_pooled.json", "w"), indent=2)
print(json.dumps(res, indent=2))
