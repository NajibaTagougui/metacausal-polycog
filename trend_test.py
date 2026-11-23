"""Formal trend test: does the spot-urine equol-producer label depend on daidzein dose?

    python trend_test.py --xpt ../nhanes_xpt --out results_trend

A dose-invariant trait implies P(label = producer | daidzein) is flat. We test this
with survey-weighted logistic regression of the label on log10 urinary daidzein,
among classifiable adults (producer vs non-producer, same rules as replicate_eqp.py).
Standard errors are design-based: Taylor linearisation with PSUs nested in strata
(SDMVSTRA, SDMVPSU), via cluster-robust sandwich on stratum x PSU clusters.
Pooled models divide the subsample weight by the number of cycles and add cycle
fixed effects.

Models
  unadjusted   label ~ log10 daidzein (+ cycle)
  adjusted     + age, sex, race/ethnicity, log10 urinary creatinine
  identifiable  adjusted, restricted to daidzein above the cycle's identifiability
               threshold, where both labels are attainable; a trend here cannot be
               blamed on the detection limit alone
Figure: label share by within-cycle daidzein decile, one line per cycle.
"""
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import statsmodels.api as sm

ap = argparse.ArgumentParser()
ap.add_argument("--xpt", required=True)
ap.add_argument("--out", default="results_trend")
ap.add_argument("--cut", type=float, default=-1.75)
a = ap.parse_args()
out = Path(a.out); out.mkdir(exist_ok=True)

CYCLES = [("1999-2000", "DEMO", "PHPYPA"), ("2001-2002", "DEMO_B", "PHPYPA_B"),
          ("2003-2004", "DEMO_C", "L06PHY_C"), ("2005-2006", "DEMO_D", "PHYTO_D"),
          ("2007-2008", "DEMO_E", "PHYTO_E"), ("2009-2010", "DEMO_F", "PHYTO_F")]
rd = lambda f: pd.read_sas(Path(a.xpt) / f"{f}.xpt", format="xport")

frames = []
for cyc, demo, phy in CYCLES:
    D = rd(demo)[["SEQN", "RIDAGEYR", "RIAGENDR", "RIDRETH1", "SDMVSTRA", "SDMVPSU"]]
    P = rd(phy)
    wcol = [c for c in P.columns if c.startswith("WT")][0]
    P = P.rename(columns={wcol: "w"})
    crcol = [c for c in P.columns if c in ("URXUCR", "URXCRS")]
    keep = ["SEQN", "w", "URXDAZ", "URXEQU"] + crcol + [c for c in ("URDDAZLC", "URDEQULC") if c in P]
    m = D.merge(P[keep], on="SEQN")
    if not crcol:   # creatinine sits in a separate file in some cycles
        for f in ("LAB16", "L16_B", "L16_C", "ALB_CR_D", "ALB_CR_E", "ALB_CR_F"):
            p = Path(a.xpt) / f"{f}.xpt"
            if p.exists():
                C = pd.read_sas(p, format="xport")
                if "URXUCR" in C:
                    m = m.merge(C[["SEQN", "URXUCR"]], on="SEQN", how="left")
                break
    else:
        m = m.rename(columns={crcol[0]: "URXUCR"})
    m = m[(m.RIDAGEYR >= 20) & m.URXDAZ.notna() & m.URXEQU.notna() & (m.w > 0)].copy()
    if "URDEQULC" in m:
        m["eq_nd"] = m.URDEQULC > 0.5          # LC flags store 0 as ~5e-79
        m["daz_nd"] = m.URDDAZLC > 0.5
        lod_eq = m.loc[m.eq_nd, "URXEQU"].max() * np.sqrt(2) if m.eq_nd.any() else np.nan
    else:
        m["eq_nd"] = m.URXEQU <= m.URXEQU.min() + 1e-9
        m["daz_nd"] = m.URXDAZ <= m.URXDAZ.min() + 1e-9
        lod_eq = m.URXEQU.min() * np.sqrt(2)
    r = np.log10(m.URXEQU / m.URXDAZ)
    m["label"] = np.select([m.daz_nd, (r > a.cut) & m.eq_nd, r > a.cut], [np.nan, np.nan, 1.0], 0.0)
    m["cycle"] = cyc
    m["ident"] = m.URXDAZ > lod_eq / 10 ** a.cut
    m["decile"] = pd.qcut(m.URXDAZ.rank(method="first"), 10, labels=False) + 1
    frames.append(m)
A = pd.concat(frames, ignore_index=True)
A["log_daz"] = np.log10(A.URXDAZ)
A["log_cr"] = np.log10(A.URXUCR) if "URXUCR" in A else np.nan
A["female"] = (A.RIAGENDR == 2).astype(float)
for k, v in {1: "mexam", 2: "ohisp", 4: "nhblack", 5: "other"}.items():
    A[f"race_{v}"] = (A.RIDRETH1 == k).astype(float)
A["cluster"] = A.cycle + "_" + A.SDMVSTRA.astype(int).astype(str) + "_" + A.SDMVPSU.astype(int).astype(str)
C = A[A.label.notna()].copy()


def fit(df, adjusted, pooled):
    cols = ["log_daz"]
    if adjusted:
        cols += ["RIDAGEYR", "female", "race_mexam", "race_ohisp", "race_nhblack", "race_other", "log_cr"]
    d = df.dropna(subset=cols + ["label"])
    X = d[cols].copy()
    if pooled:
        X = X.join(pd.get_dummies(d.cycle, drop_first=True, dtype=float))
    X = sm.add_constant(X)
    w = d.w / (d.cycle.nunique() if pooled else 1)
    w = w / w.mean()
    res = sm.GLM(d.label, X, family=sm.families.Binomial(), var_weights=w).fit(
        cov_type="cluster", cov_kwds={"groups": pd.factorize(d.cluster)[0]})
    b, se = res.params["log_daz"], res.bse["log_daz"]
    return dict(n=len(d), n_psu=d.cluster.nunique(), OR_per_10x=np.exp(b),
                CI_low=np.exp(b - 1.96 * se), CI_high=np.exp(b + 1.96 * se), z=b / se, p=res.pvalues["log_daz"])


rows = []
for model, adj, sub in [("unadjusted", False, C), ("adjusted", True, C), ("identifiable", True, C[C.ident])]:
    rows.append(dict(cycle="pooled", model=model, **fit(sub, adj, True)))
    for cyc, d in sub.groupby("cycle"):
        rows.append(dict(cycle=cyc, model=model, **fit(d, adj, False)))
# Nonlinear version: 4-df natural spline in log10 daidzein, joint design-based Wald test
from patsy import dmatrix
def fit_spline(df, pooled):
    cols = ["RIDAGEYR", "female", "race_mexam", "race_ohisp", "race_nhblack", "race_other", "log_cr"]
    d = df.dropna(subset=cols + ["label", "log_daz"])
    S = dmatrix("cr(x, df=4) - 1", {"x": d.log_daz.values}, return_type="dataframe")
    S.columns = [f"s{i}" for i in range(S.shape[1])]; S.index = d.index
    X = d[cols].join(S.iloc[:, 1:])
    if pooled:
        X = X.join(pd.get_dummies(d.cycle, drop_first=True, dtype=float))
    X = sm.add_constant(X)
    w = d.w / (d.cycle.nunique() if pooled else 1); w = w / w.mean()
    res = sm.GLM(d.label, X, family=sm.families.Binomial(), var_weights=w).fit(
        cov_type="cluster", cov_kwds={"groups": pd.factorize(d.cluster)[0]})
    names = [c for c in X.columns if c.startswith("s")]
    from scipy.stats import chi2
    b = res.params[names].values
    V = res.cov_params().loc[names, names].values
    stat = float(b @ np.linalg.pinv(V) @ b)
    return dict(n=len(d), n_psu=d.cluster.nunique(), OR_per_10x=np.nan, CI_low=np.nan, CI_high=np.nan,
                z=stat, p=float(chi2.sf(stat, len(names))))
rows.append(dict(cycle="pooled", model="adjusted, spline (chi2, 3 df)", **fit_spline(C, True)))
for cyc, d in C.groupby("cycle"):
    try:   # 1999-2002 can show complete separation (low deciles 100% producers): MLE diverges
        r = fit_spline(d, False)
        if not np.isfinite(r["z"]):
            raise ValueError
    except Exception:
        r = dict(n=len(d), n_psu=d.cluster.nunique(), OR_per_10x=np.nan, CI_low=np.nan, CI_high=np.nan,
                 z=np.nan, p=np.nan)
    rows.append(dict(cycle=cyc, model="adjusted, spline (chi2, 3 df)", **r))
R = pd.DataFrame(rows)
R.to_csv(out / "trend_tests.csv", index=False)

share = (C.assign(ww=C.w).groupby(["cycle", "decile"])
         .apply(lambda g: np.average(g.label, weights=g.w)).unstack(0))
share.to_csv(out / "label_share_by_decile.csv")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
fig, ax = plt.subplots(figsize=(6.5, 4.2), dpi=200)
for cyc in share.columns:
    ax.plot(share.index, 100 * share[cyc], marker="o", ms=3, lw=1.4, label=cyc)
ax.set_xlabel("Within-cycle decile of urinary daidzein")
ax.set_ylabel("Labelled equol producers (%, weighted)")
ax.set_xticks(range(1, 11)); ax.set_ylim(0, 100)
ax.spines[["top", "right"]].set_visible(False)
ax.legend(frameon=False, fontsize=8, title="NHANES cycle", title_fontsize=8)
ax.set_title("A dose-invariant trait would give flat lines", fontsize=10, loc="left")
fig.tight_layout(); fig.savefig(out / "label_vs_daidzein.png")

pd.set_option("display.width", 200)
print(R.round(4).to_string(index=False))
print("\nWeighted label share by daidzein decile (%):\n", (100 * share).round(1).to_string())
