"""Negative-control outcome: dental caries experience (DMFT) in samples A and B.

    python negative_control.py --xpt ../nhanes_xpt --out results_negative_control

Outcome: DMFT over the 28 teeth excluding third molars. Missing teeth (tooth count 4) and
root fragments (5) come from the tooth-count field OHXxxTC / OHDxxTC, which is coded the
same way in every cycle; decayed or filled teeth are permanent teeth present (2) with
coronal-caries code Z. The missing-tooth codes in the caries field differ between
1999-2002 (R) and 2011-2014 (P), so they are not used. Older adults lose teeth mostly to
caries and periodontal disease, so all missing teeth are counted.
Adults with a complete or partial dental exam (both record tooth counts and caries codes) and at least 20 teeth assessed. Same covariates, weights, cross-fitted AIPW and
design-based SEs as the main analyses (design_variance.py).

Caveat: high flavonoid intake is mostly tea, and tea contains fluoride, so the main
exposure could plausibly lower caries. The non-tea flavonoid exposure is therefore the
cleaner negative-control contrast. Caries also tracks SES strongly, which is what makes
it a useful check on residual confounding.
"""
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import pipeline as P
from simulate import COVARIATES

ap = argparse.ArgumentParser()
ap.add_argument("--xpt", required=True)
ap.add_argument("--out", default="results_negative_control")
a = ap.parse_args()
out = Path(a.out); out.mkdir(exist_ok=True)
TEETH = [f"{i:02d}" for i in list(range(2, 16)) + list(range(18, 32))]


def dmft(f, prefix, status_col):
    d = pd.read_sas(Path(a.xpt) / f"{f}.xpt", format="xport")
    d = d[d[status_col].isin([1, 2])].copy()   # complete or partial exam; tooth codes present in both
    codes = pd.DataFrame({t: d[f"{prefix}{t}CTC"].str.decode("ascii").fillna("") for t in TEETH})
    tc = pd.DataFrame({t: d[f"{prefix}{t}TC"] for t in TEETH})
    # missing and root fragments from the tooth-count field (coded the same in every cycle);
    # decayed/filled from the caries code Z among teeth present
    out = pd.DataFrame({"SEQN": d.SEQN.values,
                        "dmft": ((tc == 4) | (tc == 5) | ((tc == 2) & (codes == "Z"))).sum(axis=1).values,
                        "n_assessed": tc.isin([1, 2, 3, 4, 5]).sum(axis=1).values})
    return out[out.n_assessed >= 20]


A = pd.concat([dmft("OHXDENT", "OHX", "OHAEXSTS"), dmft("OHXDEN_B", "OHD", "OHAEXSTS")])
B = pd.concat([dmft("OHXDEN_G", "OHX", "OHDDESTS"), dmft("OHXDEN_H", "OHX", "OHDDESTS")])


def design_se(z, strata, psu):
    d = pd.DataFrame(dict(z=z, h=strata, j=psu)).groupby(["h", "j"]).z.sum().reset_index()
    return np.sqrt(sum(len(g) / (len(g) - 1) * ((g.z - g.z.mean()) ** 2).sum()
                       for _, g in d.groupby("h") if len(g) > 1))


RUNS = [("A", "data_studyA/nhanes_studyA_noEQP.csv", A, ["EQP"]),
        ("B", "data_studyB/nhanes_studyB_twoday.csv", B, [])]
rows = []
for samp, f, D, drop in RUNS:
    base = pd.read_csv(f).merge(D, on="SEQN")
    for treat in ["T", "T_nontea"]:
        df = base.copy()
        df["T"], df["y"] = df[treat], df["dmft"]
        covs = [c for c in list(COVARIATES) + ["caffeine_mg"] if c in df.columns and c not in drop]
        df = df.dropna(subset=covs + ["T", "y"]).reset_index(drop=True)
        X, T, y, w = df[covs].values.astype(float), df["T"].values.astype(int), df["y"].values, df["w"].values
        nuis = P.crossfit_nuisances(X, T, y, w, seed=0)
        est, se_m = P.aipw_ate(nuis["psi"], w)
        strata = df.cycle.astype(str) + "_" + df.strata.astype(int).astype(str)
        se_d = design_se(w * (nuis["psi"] - est) / w.sum(), strata, df.psu.astype(int))
        crude = np.average(y[T == 1], weights=w[T == 1]) - np.average(y[T == 0], weights=w[T == 0])
        rows.append(dict(sample=samp, exposure=treat, n=len(df), mean_dmft=np.average(y, weights=w),
                         sd_dmft=np.sqrt(np.cov(y, aweights=w)), crude_diff=crude, aipw=est,
                         se_model=se_m, se_design=se_d, ci_low=est - 1.96 * se_d, ci_high=est + 1.96 * se_d,
                         std_effect=est / np.sqrt(np.cov(y, aweights=w))))
        print(rows[-1], flush=True)
R = pd.DataFrame(rows)
R.to_csv(out / "negative_control.csv", index=False)
pd.set_option("display.width", 220)
print(R.round(3).to_string(index=False))
