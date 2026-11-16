# MetaCausal-PolyCog

Metabotype-aware causal machine learning for the polyphenol → cognition question
raised in Sbai et al., *Nutrients* 2026, 18:1697 (Sec. 6.8).

## Install
```
pip install econml shap scikit-learn statsmodels matplotlib pandas
```
Tested with econml 0.17.0, shap 0.54.1, scikit-learn 1.9.1 on 2 CPUs.

## Files
| File | What it does |
|---|---|
| `simulate.py` | NHANES-like semi-synthetic data with known individual effects (`scenario="additive"` or `"nonlinear"`) |
| `pipeline.py` | Cross-fitted nuisances, AIPW, CATE learners incl. the knowledge-guided causal forest, BLP / GATES / CLAN, policy tree |
| `benchmark.py` | Compares 7 CATE learners + 2 average-effect methods against the truth |
| `power.py` | Power of the confirmatory metabotype contrasts by sample size |
| `analyze.py` | Full single-dataset analysis + figures (`--demo` or `--data file.csv`) |
| `nhanes_prepare.py` | Downloads and merges NHANES 1999–2002 (run on a machine with internet access) |
| `make_figures.py` | Benchmark and power figures |

## Reproduce
```
python benchmark.py --reps 10 --sizes 600 2600 5000 --out results
python benchmark.py --reps 10 --sizes 2600 5000 --scenario nonlinear --out results_nonlinear
python power.py 30
python make_figures.py
python analyze.py --demo --n 5000 --out analysis_demo
```
Runtime on 2 CPUs: roughly 15 + 8 + 15 + 1 + 12 minutes.
For the paper, raise replicates (benchmark ≥ 50, power ≥ 200).

## Real data
```
python nhanes_prepare.py --polyphenol_table foodcode_polyphenols.csv --out nhanes_studyA.csv
python analyze.py --data nhanes_studyA.csv --out analysis_studyA
```
`foodcode_polyphenols.csv` (columns `food_code, mg_per_100g`) must be built from the USDA
flavonoid/isoflavone databases and Phenol-Explorer. Add BMI, diabetes, smoking etc. from the
BMX / DIQ / SMQ / MCQ / PAQ files before running (the script marks where).
Verify every variable against the CDC codebook for each cycle.

## Key design choices
- Metabotypes are defined by dose-invariant ratios (equol/daidzein; ENL/(ENL+END)) so that they
  are not downstream of polyphenol intake.
- Propensity = L2 logistic regression (boosted propensity overfitted at n = 600).
- Forests: honest, 1,000 trees, min leaf 50; KG-CF restricts heterogeneity to pre-registered modifiers.
- Policy tree charges a 3-point minimum worthwhile gain; otherwise it recommends treating everyone.

## Added 2026-11-16
- `nhanes_prepare.py --download_only nhanes_xpt` downloads all 24 files; full covariate merge, variable audit,
  `--local_dir`, `--eqp_cut`, `--enl_quantile`, unmatched food-code lists, energy adjustment (residual method).
- `build_polyphenol_table.py` builds `foodcode_polyphenols.csv` from the USDA FNDDS Flavonoid Database.
- `analyze.py --no_weights` (sensitivity); `benchmark.py --drop_modifier <name>` (misspecified KG-CF).
- Fix: propensity model now accepts survey weights (was a crash on weighted data).
