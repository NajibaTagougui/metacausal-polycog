"""Design-based standard errors, E-values and overlap for the NHANES average effects.

    python design_variance.py --out results_design

For each real-data analysis, recomputes the cross-fitted AIPW pseudo-outcomes exactly
as analyze.py does (same covariates, same seed), then reports:
  * the model-based SE used in the paper (weights, independent units)
  * a design-based SE: Taylor linearisation of the weighted mean of the AIPW
    pseudo-outcomes, with PSUs nested in strata (cycle x SDMVSTRA, SDMVPSU), as in
    NHANES analytic guidelines. Subgroup contrasts are treated as domain estimates.
  * the design effect (design variance / model variance)
  * E-values (VanderWeele & Ding 2017) for the point estimate and the CI limit nearest
    the null, using the standardised-difference approximation RR = exp(0.91 d)
  * the share of people whose propensity score sits at the 0.03 / 0.97 clip
"""
import argparse, json
from pathlib import Path
import numpy as np
import pandas as pd
import pipeline as P
from simulate import COVARIATES

ap = argparse.ArgumentParser()
ap.add_argument("--out", default="results_design")
a = ap.parse_args()
out = Path(a.out); out.mkdir(exist_ok=True)

RUNS = [  # name, file, treat, outcome, extra, drop, contrasts
    ("A: no EQP (main)", "data_studyA/nhanes_studyA_noEQP.csv", "T", "y", ["caffeine_mg"], ["EQP"], []),
    ("A: with EQP", "data_studyA/nhanes_studyA.csv", "T", "y", ["caffeine_mg"], [], ["EQP", "both"]),
    ("A: non-tea exposure", "data_studyA/nhanes_studyA_noEQP.csv", "T_nontea", "y", ["caffeine_mg"], ["EQP"], []),
    ("B: two-day (main)", "data_studyB/nhanes_studyB_twoday.csv", "T", "y", ["caffeine_mg"], [], []),
    ("B: non-tea exposure", "data_studyB/nhanes_studyB_twoday.csv", "T_nontea", "y", ["caffeine_mg"], [], []),
    ("B: animal fluency", "data_studyB/nhanes_studyB_twoday.csv", "T", "animal_fluency", ["caffeine_mg"], [], []),
    ("B: CERAD delayed", "data_studyB/nhanes_studyB_twoday.csv", "T", "cerad_delayed", ["caffeine_mg"], [], []),
]


def design_se(z, strata, psu):
    """Linearisation variance of a total of z: PSUs within strata, with replacement."""
    d = pd.DataFrame(dict(z=z, h=strata, j=psu)).groupby(["h", "j"]).z.sum().reset_index()
    v = 0.0
    for _, g in d.groupby("h"):
        nh = len(g)
        if nh > 1:
            v += nh / (nh - 1) * ((g.z - g.z.mean()) ** 2).sum()
    return np.sqrt(v)


def evalue(est, lo, hi, sd):
    def ev(d):
        rr = np.exp(0.91 * abs(d))
        return rr + np.sqrt(rr * (rr - 1))
    e_pt = ev(est / sd)
    e_ci = 1.0 if lo <= 0 <= hi else ev(min(abs(lo), abs(hi)) / sd)
    return e_pt, e_ci


rows = []
for name, f, treat, outcome, extra, drop, contrasts in RUNS:
    df = pd.read_csv(f)
    df["T"], df["y"] = df[treat], df[outcome]
    covs = [c for c in list(COVARIATES) + extra if c in df.columns and c not in drop]
    df = df.dropna(subset=covs + ["T", "y"]).reset_index(drop=True)
    X, T, y, w = df[covs].values.astype(float), df["T"].values.astype(int), df["y"].values, df["w"].values
    nuis = P.crossfit_nuisances(X, T, y, w, seed=0)
    psi = nuis["psi"]
    est, se_model = P.aipw_ate(psi, w)
    strata = df.cycle.astype(str) + "_" + df.strata.astype(int).astype(str)
    psu = df.psu.astype(int)
    z = w * (psi - est) / w.sum()
    se_des = design_se(z, strata, psu)
    lo, hi = est - 1.96 * se_des, est + 1.96 * se_des
    e_pt, e_ci = evalue(est, lo, hi, np.sqrt(np.cov(y, aweights=w)))
    clip = float(np.mean((nuis["e"] <= 0.0301) | (nuis["e"] >= 0.9699)))
    rows.append(dict(analysis=name, estimand="ATE", n=len(df), n_strata=strata.nunique(),
                     n_psu=(strata + "_" + psu.astype(str)).nunique(), estimate=est,
                     se_model=se_model, se_design=se_des, deff=(se_des / se_model) ** 2,
                     ci_low=lo, ci_high=hi, evalue_point=e_pt, evalue_ci=e_ci, share_at_clip=clip))
    for c in contrasts:
        if c == "EQP":
            g1, g0 = df.EQP == 1, df.EQP == 0
            lab = "EQP=1 minus EQP=0"
        else:
            g1, g0 = (df.EQP == 1) & (df.ENLH == 1), (df.EQP == 0) & (df.ENLH == 0)
            lab = "both minus neither"
        e1 = np.average(psi[g1], weights=w[g1]); e0 = np.average(psi[g0], weights=w[g0])
        zc = (g1 * w * (psi - e1) / w[g1].sum()) - (g0 * w * (psi - e0) / w[g0].sum())
        s1 = P.aipw_ate(psi[g1], w[g1])[1]; s0 = P.aipw_ate(psi[g0], w[g0])[1]
        sd = design_se(zc.values, strata, psu)
        rows.append(dict(analysis=name, estimand=lab, n=int(g1.sum() + g0.sum()), estimate=e1 - e0,
                         se_model=np.hypot(s1, s0), se_design=sd, deff=(sd / np.hypot(s1, s0)) ** 2,
                         ci_low=e1 - e0 - 1.96 * sd, ci_high=e1 - e0 + 1.96 * sd))
    print(name, "done", flush=True)

R = pd.DataFrame(rows)
R.to_csv(out / "design_based.csv", index=False)
pd.set_option("display.width", 250)
print(R.round(3).to_string(index=False))
