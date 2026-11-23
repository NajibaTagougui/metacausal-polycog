"""How strong must trait-intake dependence be to make a VALID label fail the diagnostic?

    python diagnostic_sensitivity.py --reps 400 --out results_diagnostic_sens

The label is valid (it depends on the trait only, sensitivity 0.9, specificity 0.9, no
dose term), but the trait itself varies with intake:
    logit P(M = 1 | D) = a + log(OR_trait) * log10(D)      (prevalence about 0.3)
For each OR_trait we report the label-on-dose odds ratio this induces and the rejection
rate of the linear diagnostic. A valid label can only reproduce a label-dose OR that is
closer to 1 than OR_trait (misclassification attenuates it), so the NHANES OR of 0.05 per
tenfold would require producers' odds to fall more than 20-fold per tenfold rise in soy.
"""
import argparse, zlib
from pathlib import Path
import numpy as np
import pandas as pd
import statsmodels.api as sm

ap = argparse.ArgumentParser()
ap.add_argument("--reps", type=int, default=400)
ap.add_argument("--sizes", type=int, nargs="*", default=[500, 2600])
ap.add_argument("--out", default="results_diagnostic_sens")
a = ap.parse_args()
out = Path(a.out); out.mkdir(exist_ok=True)

rows = []
for orr in (2.0, 1.5, 1.25, 1.1, 1.0, 0.9, 0.8, 0.67, 0.5, 0.25, 0.1, 0.05, 0.02):
    for n in a.sizes:
        rng = np.random.default_rng(zlib.crc32(f"{orr}|{n}".encode()))
        rej, ors = 0, []
        for _ in range(a.reps):
            z = rng.normal(0, 1.5, n) / np.log(10)                 # centred log10 daidzein, as in simulate.py
            M = rng.binomial(1, 1 / (1 + np.exp(-(-0.85 + np.log(orr) * z))))
            Ms = np.where(M == 1, rng.random(n) < 0.9, rng.random(n) < 0.1).astype(int)
            f = sm.Logit(Ms, sm.add_constant(z)).fit(disp=0)
            rej += f.pvalues[1] < 0.05
            ors.append(np.exp(f.params[1]))
        rows.append(dict(trait_intake_OR=orr, n=n, label_dose_OR=float(np.median(ors)), reject=rej / a.reps))
R = pd.DataFrame(rows)
R.to_csv(out / "diagnostic_sensitivity.csv", index=False)
print(R.round(3).to_string(index=False))
