"""Comparison with the standard misclassification correction.

    python correction_comparison.py --out results_correction

The usual fix for a misclassified binary modifier rescales the label contrast by the
attenuation factor of Proposition 1, estimated in a validation subsample where the
true trait is measured (for equol, a soy challenge):
    corrected = label_contrast / (PPV_hat + NPV_hat - 1)
We apply it to the stored label contrasts of the label benchmarks (same seeds, so the
same simulated data) with an internal validation subsample of v people (10% by default)
drawn at random from each training set. The correction removes attenuation; it cannot
remove the selection term of Proposition 3, which is what this comparison shows.
"""
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
from simulate import simulate

ap = argparse.ArgumentParser()
ap.add_argument("--frac", type=float, default=0.10, help="validation fraction")
ap.add_argument("--out", default="results_correction")
a = ap.parse_args()
out = Path(a.out); out.mkdir(exist_ok=True)

RUNS = [("main (trait gates effect only)", "results_label", 0.0),
        ("prognostic trait, +10", "results_label_prog", 10.0),
        ("prognostic trait, -10", "results_label_prog_neg", -10.0)]
rows = []
for scen, folder, tb in RUNS:
    R = pd.read_csv(f"{folder}/label_raw.csv")
    for _, r in R.iterrows():
        tr = simulate(int(r.n), seed=int(r.seed), label=r.label, trait_baseline=tb)
        rng = np.random.default_rng(1000 + int(r.seed))
        v = tr.iloc[rng.choice(len(tr), int(a.frac * len(tr)), replace=False)]
        ppv = (v.EQP_true[v.EQP == 1] == 1).mean() if (v.EQP == 1).any() else np.nan
        npv = (v.EQP_true[v.EQP == 0] == 0).mean() if (v.EQP == 0).any() else np.nan
        k = ppv + npv - 1
        corr = r.label_contrast / k if k > 0.05 else np.nan
        rows.append(dict(scenario=scen, label=r.label, seed=r.seed, true_contrast=r.true_contrast,
                         label_contrast=r.label_contrast, factor_hat=k, corrected=corr))
D = pd.DataFrame(rows)
D.to_csv(out / "correction_raw.csv", index=False)
S = D.assign(bias_uncorrected=D.label_contrast - D.true_contrast, bias_corrected=D.corrected - D.true_contrast) \
     .groupby(["scenario", "label"], sort=False).agg(
        factor_hat=("factor_hat", "mean"), label_contrast=("label_contrast", "mean"),
        corrected=("corrected", "mean"), bias_uncorrected=("bias_uncorrected", "mean"),
        bias_corrected=("bias_corrected", "mean"), sd_corrected=("corrected", "std"),
        n_failed=("corrected", lambda s: int(s.isna().sum())))
S["mcse_bias_corrected"] = S.sd_corrected / np.sqrt(D.groupby(["scenario", "label"], sort=False).size())
S.to_csv(out / "correction_summary.csv")
pd.set_option("display.width", 200)
print(S.round(3).to_string())
