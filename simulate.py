"""
Semi-synthetic NHANES-like generator with KNOWN individual treatment effects.

Purpose: benchmark whether causal-ML learners recover metabotype-driven
heterogeneity in the polyphenol -> cognition effect (the gap stated in
Sbai et al., Nutrients 2026, 18:1697, Sec. 6.8), before touching real data.

Design choices mirror NHANES 2011-2014 / 1999-2002 older-adult samples:
  - outcome: DSST score (0-105 scale, SD ~ 17)
  - exposure T: high dietary polyphenol intake (top tertile of energy-adjusted
    flavonoid+lignan+isoflavone intake) -> binary
  - metabotypes (pre-exposure, dose-invariant traits):
      EQP  = equol producer (log10 equol/daidzein > -1.75; ~30% in Western adults)
      ENLH = high enterolactone producer (ENL / (ENL+END) ratio, top 40%)
  - confounding of T by SES, education, sex, BMI, smoking, energy intake
  - true CATE concentrated in producers -> small population ATE ("dilution")
"""
import numpy as np
import pandas as pd

COVARIATES = [
    "age", "female", "race_black", "race_hisp", "race_other", "edu_college",
    "edu_lt_hs", "pir", "bmi", "smoker", "diabetes", "hypertension", "cvd",
    "phys_active", "kcal", "alcohol", "phq9", "EQP", "ENLH",
]


def _sigmoid(z):
    return 1.0 / (1.0 + np.exp(-z))


def simulate(n=2600, seed=0, effect_scale=1.0, confounding=1.0, scenario="additive", label="true"):
    rng = np.random.default_rng(seed)
    age = rng.uniform(60, 85, n)
    female = rng.binomial(1, 0.53, n)
    race = rng.choice(4, n, p=[0.55, 0.20, 0.17, 0.08])
    race_black, race_hisp, race_other = (race == 1) * 1, (race == 2) * 1, (race == 3) * 1
    edu_z = rng.normal(0, 1, n) - 0.4 * race_hisp - 0.2 * race_black
    edu_college = (edu_z > 0.55) * 1
    edu_lt_hs = (edu_z < -0.9) * 1
    pir = np.clip(2.5 + 1.0 * edu_z + rng.normal(0, 1.1, n), 0, 5)
    bmi = np.clip(rng.normal(29, 6, n) + 1.2 * race_black - 0.6 * edu_college, 16, 60)
    smoker = rng.binomial(1, _sigmoid(-1.8 - 0.5 * edu_z), n)
    diabetes = rng.binomial(1, _sigmoid(-2.2 + 0.08 * (bmi - 29) + 0.03 * (age - 70)
                                        + 0.4 * race_black + 0.4 * race_hisp), n)
    hypertension = rng.binomial(1, _sigmoid(0.2 + 0.05 * (age - 70) + 0.06 * (bmi - 29)), n)
    cvd = rng.binomial(1, _sigmoid(-2.0 + 0.05 * (age - 70) + 0.6 * diabetes + 0.4 * smoker), n)
    phys_active = rng.binomial(1, _sigmoid(-0.2 + 0.5 * edu_z - 0.05 * (bmi - 29)), n)
    kcal = np.clip(rng.normal(1850, 550, n) + 350 * (1 - female), 600, 5000)
    alcohol = rng.binomial(1, _sigmoid(-0.5 + 0.4 * edu_z), n)
    phq9 = np.clip(rng.poisson(np.exp(0.9 - 0.15 * edu_z + 0.3 * cvd)), 0, 27)

    # Metabotypes: stable microbial traits (weakly related to ethnicity/BMI)
    EQP = rng.binomial(1, _sigmoid(-0.9 + 0.35 * race_other - 0.02 * (bmi - 29)), n)
    ENLH = rng.binomial(1, _sigmoid(-0.4 - 0.04 * (bmi - 29) - 0.4 * smoker + 0.3 * phys_active), n)

    # Exposure: high polyphenol intake, confounded
    lin_t = confounding * (0.6 * edu_z + 0.25 * (pir - 2.5) + 0.35 * female
                           - 0.04 * (bmi - 29) - 0.5 * smoker + 0.3 * phys_active
                           + 0.0002 * (kcal - 1850)) - 0.7
    ps = np.clip(_sigmoid(lin_t), 0.03, 0.97)
    T = rng.binomial(1, ps, n)

    # Baseline cognition mu0(X): DSST
    mu0 = (52 - 0.95 * (age - 70) + 7.5 * edu_college - 8 * edu_lt_hs + 1.8 * (pir - 2.5)
           - 6 * race_black - 4 * race_hisp + 2.5 * female - 3 * diabetes - 2 * cvd
           + 2 * phys_active - 0.4 * phq9 - 1.5 * smoker
           + 3 * np.sin((age - 60) / 8))                       # nonlinearity

    # TRUE individual effect tau(X): metabotype-gated, age-modified
    tau = effect_scale * (0.3 + 4.5 * EQP + 3.0 * ENLH + 1.5 * EQP * ENLH
                          - 0.10 * (age - 70) * (EQP + ENLH > 0)
                          - 1.5 * diabetes * (EQP + ENLH > 0))
    if scenario == "nonlinear":
        # threshold / non-additive gating: producers benefit only when metabolically
        # healthy (BMI<30) and not too old (<77); ENL benefit needs physical activity
        tau = effect_scale * (0.3 + 7.0 * EQP * (bmi < 30) * (age < 77)
                              + 4.0 * ENLH * phys_active
                              + 2.0 * EQP * ENLH * (diabetes == 0))
    y = mu0 + tau * T + rng.normal(0, 13, n)
    y = np.clip(y, 0, 105)

    # Observed equol label. Draws below happen after y, so label="true" reproduces
    # the original data exactly.
    #   true       : the label is the latent trait
    #   nondiff    : 10% random (non-differential) misclassification
    #   confounded : label from a spot-urine ratio; daidzein depends on SES, not on T
    #   post       : label from a spot-urine ratio; daidzein rises with T (soy is part
    #                of the exposure), so the label is a descendant of treatment
    # Spot-urine model, calibrated to NHANES 2005-2010: background equol ~ 1.5 ng/mL
    # in everyone; producers convert ~30% of daidzein; label = log10(equol/daidzein) > -1.75.
    EQP_true = EQP
    a_T, b_ses = (1.0, 0.3) if label == "post" else (0.0, 0.6)
    log_daz = np.log(30) + a_T * (T - T.mean()) + b_ses * edu_z + rng.normal(0, 1.5, n)
    daz = np.exp(log_daz)
    equol = np.exp(np.log(1.5) + rng.normal(0, 0.5, n)) + EQP_true * 0.3 * daz * np.exp(rng.normal(0, 0.5, n))
    if label == "nondiff":
        EQP = np.where(rng.random(n) < 0.10, 1 - EQP_true, EQP_true)
    elif label in ("confounded", "post"):
        EQP = (np.log10(equol / daz) > -1.75).astype(int)
    elif label != "true":
        raise ValueError(label)

    df = pd.DataFrame(dict(age=age, female=female, race_black=race_black,
                           race_hisp=race_hisp, race_other=race_other,
                           edu_college=edu_college, edu_lt_hs=edu_lt_hs, pir=pir,
                           bmi=bmi, smoker=smoker, diabetes=diabetes,
                           hypertension=hypertension, cvd=cvd,
                           phys_active=phys_active, kcal=kcal, alcohol=alcohol,
                           phq9=phq9, EQP=EQP, ENLH=ENLH, T=T, y=y,
                           tau_true=tau, ps_true=ps, mu0_true=mu0,
                           EQP_true=EQP_true, log_daidzein=log_daz))
    return df


if __name__ == "__main__":
    d = simulate()
    print(d.describe().T[["mean", "std", "min", "max"]].round(2))
    print("True ATE:", d.tau_true.mean().round(2),
          "| naive diff:", (d.y[d["T"] == 1].mean() - d.y[d["T"] == 0].mean()).round(2))
    print(d.groupby(["EQP", "ENLH"]).tau_true.mean().round(2))
