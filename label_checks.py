"""Two further checks on the urinary metabotype labels.

    python label_checks.py --xpt ../nhanes_xpt --out results_label_checks

1. Enterolactone label. Does the ENL-dominant label (top 40% of the within-cycle
   ENL / (ENL + END) fraction) depend on total enterolignan excretion, the lignan
   dose marker? Same model as trend_test.py: survey-weighted logistic regression
   with design-based SEs, adults 20+, six cycles 1999-2010, adjusted for age, sex,
   race/ethnicity and log urinary creatinine, cycle fixed effects.
2. Dairy equol. Cow's milk contains equol, so dairy intake could produce
   "producer" labels in non-producers. In 1999-2002 (24-h recall on the same exam),
   regress log10 urinary equol and the producer label on dairy grams (USDA food
   codes 11xxxxxx-14xxxxxx), adjusted for log10 daidzein and the covariates above.
"""
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import statsmodels.api as sm

ap = argparse.ArgumentParser()
ap.add_argument("--xpt", required=True)
ap.add_argument("--out", default="results_label_checks")
a = ap.parse_args()
out = Path(a.out); out.mkdir(exist_ok=True)
rd = lambda f: pd.read_sas(Path(a.xpt) / f"{f}.xpt", format="xport")
CYCLES = [("1999-2000", "DEMO", "PHPYPA", "DRXIFF"), ("2001-2002", "DEMO_B", "PHPYPA_B", "DRXIFF_B"),
          ("2003-2004", "DEMO_C", "L06PHY_C", None), ("2005-2006", "DEMO_D", "PHYTO_D", None),
          ("2007-2008", "DEMO_E", "PHYTO_E", None), ("2009-2010", "DEMO_F", "PHYTO_F", None)]
ADJ = ["RIDAGEYR", "female", "race_mexam", "race_ohisp", "race_nhblack", "race_other", "log_cr"]

frames = []
for cyc, demo, phy, iff in CYCLES:
    D = rd(demo)[["SEQN", "RIDAGEYR", "RIAGENDR", "RIDRETH1", "SDMVSTRA", "SDMVPSU"]]
    P = rd(phy)
    wcol = "WTSPH2YR" if "WTSPH2YR" in P else [c for c in P.columns if c.startswith("WT")][0]
    P = P.rename(columns={wcol: "w"})
    m = D.merge(P[["SEQN", "w", "URXDAZ", "URXEQU", "URXETL", "URXETD", "URXUCR"]], on="SEQN")
    if iff:
        F = rd(iff)[["SEQN", "DRDIFDCD", "DRXIGRMS"]]
        dairy = F[(F.DRDIFDCD >= 11e6) & (F.DRDIFDCD < 15e6)].groupby("SEQN").DRXIGRMS.sum()
        anyrec = F.groupby("SEQN").size()
        m["dairy_g"] = m.SEQN.map(dairy).fillna(0.0).where(m.SEQN.isin(anyrec.index))
    m["cycle"] = cyc
    m = m[(m.RIDAGEYR >= 20) & (m.w > 0)].copy()
    frames.append(m)
A = pd.concat(frames, ignore_index=True)
A["female"] = (A.RIAGENDR == 2).astype(float)
for k, v in {1: "mexam", 2: "ohisp", 4: "nhblack", 5: "other"}.items():
    A[f"race_{v}"] = (A.RIDRETH1 == k).astype(float)
A["log_cr"] = np.log10(A.URXUCR)
A["cluster"] = A.cycle + "_" + A.SDMVSTRA.astype(int).astype(str) + "_" + A.SDMVPSU.astype(int).astype(str)


def glm(d, yv, xv, family, pooled):
    d = d.dropna(subset=[yv, xv] + ADJ)
    X = d[[xv] + ADJ]
    if pooled:
        X = X.join(pd.get_dummies(d.cycle, drop_first=True, dtype=float))
    w = d.w / d.groupby("cycle").w.transform("mean")
    r = sm.GLM(d[yv], sm.add_constant(X), family=family, var_weights=w).fit(
        cov_type="cluster", cov_kwds={"groups": pd.factorize(d.cluster)[0]})
    b, se = r.params[xv], r.bse[xv]
    return dict(n=len(d), coef=b, ci_low=b - 1.96 * se, ci_high=b + 1.96 * se, p=r.pvalues[xv])


rows = []
# 1. enterolactone label vs total enterolignans
E = A.dropna(subset=["URXETL", "URXETD"]).copy()
E = E[(E.URXETL + E.URXETD) > 0]
E["enl_frac"] = E.URXETL / (E.URXETL + E.URXETD)
E["ENLH"] = (E.enl_frac > E.groupby("cycle").enl_frac.transform(lambda x: x.quantile(0.60))).astype(float)
E["log_tot"] = np.log10(E.URXETL + E.URXETD)
E["decile"] = E.groupby("cycle").log_tot.transform(lambda x: pd.qcut(x.rank(method="first"), 10, labels=False) + 1)
r = glm(E, "ENLH", "log_tot", sm.families.Binomial(), True)
rows.append(dict(check="ENLH label ~ log10 total enterolignans (pooled)", **r, OR_per_10x=np.exp(r["coef"])))
for cyc, d in E.groupby("cycle"):
    r = glm(d, "ENLH", "log_tot", sm.families.Binomial(), False)
    rows.append(dict(check=f"ENLH label ~ log10 total enterolignans ({cyc})", **r, OR_per_10x=np.exp(r["coef"])))
enl_share = E.groupby(["cycle", "decile"]).apply(lambda g: np.average(g.ENLH, weights=g.w)).unstack(0)

# 2. dairy and equol, 1999-2002
Dd = A[A.dairy_g.notna() & A.URXEQU.notna() & A.URXDAZ.notna()].copy()
Dd["log_eq"] = np.log10(Dd.URXEQU)
Dd["log_daz"] = np.log10(Dd.URXDAZ)
Dd["log_dairy"] = np.log10(Dd.dairy_g + 1)
Dd["label"] = (np.log10(Dd.URXEQU / Dd.URXDAZ) > -1.75).astype(float)
ADJ0 = ADJ.copy(); ADJ.append("log_daz")
r = glm(Dd, "log_eq", "log_dairy", sm.families.Gaussian(), True)
rows.append(dict(check="log10 equol ~ log10(dairy g + 1), adj. daidzein (1999-2002)", **r))
r = glm(Dd, "label", "log_dairy", sm.families.Binomial(), True)
rows.append(dict(check="producer label ~ log10(dairy g + 1), adj. daidzein (1999-2002)", **r, OR_per_10x=np.exp(r["coef"])))
low = Dd[Dd.log_daz < Dd.groupby("cycle").log_daz.transform("median")]
r = glm(low, "log_eq", "log_dairy", sm.families.Gaussian(), True)
rows.append(dict(check="log10 equol ~ log10(dairy g + 1), daidzein below median", **r))
Dd["dairy_q"] = pd.qcut(Dd.dairy_g.rank(method="first"), 4, labels=False) + 1
dq = Dd.groupby("dairy_q").apply(lambda g: pd.Series(dict(
    dairy_g_median=g.dairy_g.median(), equol_gmean=10 ** np.average(g.log_eq, weights=g.w),
    producer_share=np.average(g.label, weights=g.w), n=len(g))))
ADJ[:] = ADJ0

R = pd.DataFrame(rows)
R.to_csv(out / "label_checks.csv", index=False)
enl_share.to_csv(out / "enlh_share_by_decile.csv")
dq.to_csv(out / "equol_by_dairy_quartile.csv")
pd.set_option("display.width", 220)
print(R.round(4).to_string(index=False))
print("\nENLH share by total-enterolignan decile (%):\n", (100 * enl_share).round(1).to_string())
print("\nEquol by dairy quartile, 1999-2002:\n", dq.round(3).to_string())
