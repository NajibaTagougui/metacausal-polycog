"""
Full single-dataset analysis (the paper's main pipeline).
  python analyze.py --demo                 # semi-synthetic worked example
  python analyze.py --data nhanes_studyA.csv  # real data (needs columns below)
Outputs tables + figures into --out.
"""
import argparse, json, time, warnings
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import shap
import pipeline as P
from simulate import simulate, COVARIATES

warnings.filterwarnings("ignore")
MODIFIERS = ["EQP", "ENLH", "age", "female", "diabetes", "bmi", "edu_college"]
BLUE, ORANGE, AQUA, YELLOW, GREY = "#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#8a8984"
plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False,
                     "axes.edgecolor": "#52514e", "axes.labelcolor": "#0b0b0b",
                     "xtick.color": "#52514e", "ytick.color": "#52514e",
                     "axes.grid": True, "grid.color": "#e6e5e1", "grid.linewidth": 0.6,
                     "savefig.dpi": 300, "figure.facecolor": "white"})


def main(df, out, seed=0, extra_covs=(), drop_covs=(), shap_n=600):
    t0 = time.time()
    step = lambda m: print(f"[{time.time() - t0:6.0f}s] {m}", flush=True)
    out = Path(out); out.mkdir(exist_ok=True)
    covs = [c for c in list(COVARIATES) + list(extra_covs) if c in df.columns and c not in drop_covs]
    df = df.dropna(subset=covs + ["T", "y"]).reset_index(drop=True)
    print(f"n={len(df)} | covariates ({len(covs)}): {covs}")
    mods = [c for c in MODIFIERS if c in covs]
    midx = [covs.index(c) for c in mods]
    X, T, y = df[covs].values.astype(float), df["T"].values.astype(int), df["y"].values
    w = df["w"].values if "w" in df.columns else None
    res = {"n": int(len(df)), "treated_frac": float(T.mean()), "covariates": covs, "modifiers": mods}

    # 1. average effect (doubly robust)
    nuis = P.crossfit_nuisances(X, T, y, w, seed=seed)
    ate, se = P.aipw_ate(nuis["psi"], w)
    res["ATE_AIPW"] = [ate, se]
    res["overlap_e_range"] = [float(nuis["e"].min()), float(nuis["e"].max())]

    step("average effect done")
    # 2. CATE: knowledge-guided causal forest (primary) + unrestricted CF (sensitivity)
    tau_oof = P.oof_cate(X, T, y, w, seed=seed, learner="KG-CausalForest", modifier_idx=midx)
    fit = P.fit_learners(X, T, y, w, seed=seed, which=["KG-CausalForest"], modifier_idx=midx)
    kg = fit["_kgcf_model"]

    step("KG causal forest done")
    # 3. heterogeneity inference
    blp = P.blp_test(tau_oof, nuis, T, w); res["BLP"] = blp
    g, groups = P.gates(tau_oof, nuis, 5, w); g.to_csv(out / "gates.csv", index=False)
    cl = P.clan(df[covs], groups); cl.to_csv(out / "clan.csv")

    step("BLP/GATES/CLAN done")
    # 4. metabotype subgroup effects (pre-specified), AIPW within strata
    sub = []
    for eqp in ((0, 1) if "EQP" in covs and "ENLH" in covs else ()):
        for enl in (0, 1):
            m = ((df.EQP == eqp) & (df.ENLH == enl)).values
            e_, s_ = P.aipw_ate(nuis["psi"][m], None if w is None else w[m])
            row = dict(EQP=eqp, ENLH=enl, n=int(m.sum()), effect=e_, se=s_)
            if "tau_true" in df: row["true"] = df.tau_true[m].mean()
            sub.append(row)
    sub = pd.DataFrame(sub, columns=["EQP", "ENLH", "n", "effect", "se"])
    if len(sub) == 4:  # H1 / H2 contrasts: Wald tests of the difference
        e = sub.set_index(["EQP", "ENLH"])
        p_ = {k: (e.loc[k, "effect"], e.loc[k, "se"]) for k in e.index}
        m1 = (df.EQP == 1).values; m0 = (df.EQP == 0).values
        a1, s1 = P.aipw_ate(nuis["psi"][m1], None if w is None else w[m1])
        a0, s0 = P.aipw_ate(nuis["psi"][m0], None if w is None else w[m0])
        res["H1_EQP1_minus_EQP0"] = [a1 - a0, float(np.hypot(s1, s0))]
        res["H2_both_minus_neither"] = [p_[(1, 1)][0] - p_[(0, 0)][0], float(np.hypot(p_[(1, 1)][1], p_[(0, 0)][1]))]
    sub.to_csv(out / "metabotype_effects.csv", index=False)

    step("metabotype strata done")
    # 5. policy tree
    pt = P.policy_tree(X[:, midx], T, y, mods, depth=2, seed=seed, W=X)
    try:
        pt.plot(feature_names=mods, treatment_names=["no targeting", "target"])
        plt.gcf().set_size_inches(9, 4.5); plt.tight_layout()
        plt.savefig(out / "fig_policy_tree.png"); plt.close()
    except Exception as ex:
        print("policy tree plot skipped:", ex)

    step("policy tree done")
    json.dump(json.loads(json.dumps(res, default=float)), open(out / "results.json", "w"), indent=2)
    # 6. SHAP on CATE
    rs = np.random.default_rng(seed).choice(len(X), min(shap_n, len(X)), replace=False)
    sv = kg.shap_values(X[rs][:, midx], feature_names=mods, background_samples=100)
    key1 = list(sv.keys())[0]; key2 = list(sv[key1].keys())[0]
    vals = sv[key1][key2]
    plt.figure()
    shap.plots.beeswarm(vals, show=False, max_display=len(mods), color_bar=True)
    plt.gcf().set_size_inches(7, 4); plt.title("Drivers of the individual polyphenol effect (SHAP)")
    plt.tight_layout(); plt.savefig(out / "fig_shap_cate.png"); plt.close()
    res["mean_abs_shap"] = dict(zip(mods, np.abs(vals.values).mean(0).round(3).tolist()))

    step("SHAP done")
    # ---- figures
    fig, ax = plt.subplots(1, 2 if len(sub) else 1, figsize=(10 if len(sub) else 5.5, 3.8), squeeze=False)
    ax = ax[0]
    ax[0].errorbar(g.group, g.gate, yerr=1.96 * g.se, fmt="o", color=BLUE, capsize=3, ms=7,
                   label="Estimated group effect (95% CI)")
    ax[0].axhline(ate, color=GREY, ls="--", lw=1.5, label=f"Average effect = {ate:.2f}")
    if "tau_true" in df:
        tg = df.groupby(groups).tau_true.mean().values
        ax[0].plot(g.group, tg, "D", color=ORANGE, ms=6, label="True group effect")
    ax[0].set_xticks(g.group); ax[0].set_xlabel("Quintile of predicted benefit"); ax[0].set_ylabel("Effect on DSST (points)")
    ax[0].set_title("A. Sorted group effects (GATES)", loc="left"); ax[0].legend(frameon=False, fontsize=8)
    if len(sub):
        xs = np.arange(len(sub))
        ax[1].bar(xs, sub.effect, color=[GREY, AQUA, BLUE, ORANGE], width=0.6, edgecolor="white", lw=2)
        ax[1].errorbar(xs, sub.effect, yerr=1.96 * sub.se, fmt="none", ecolor="#0b0b0b", capsize=3)
        if "true" in sub: ax[1].plot(xs, sub.true, "D", color="#0b0b0b", ms=5, label="True effect")
        ax[1].set_xticks(xs, ["Non-producer", "ENL-high only", "Equol only", "Both"])
        ax[1].set_ylabel("Effect on DSST (points)")
        ax[1].set_title("B. Effect by microbial metabotype", loc="left")
        if "true" in sub: ax[1].legend(frameon=False, fontsize=8)
    plt.tight_layout(); plt.savefig(out / "fig_gates_metabotype.png"); plt.close()

    res = json.loads(json.dumps(res, default=float))
    json.dump(res, open(out / "results.json", "w"), indent=2)
    print(json.dumps(res, indent=2)); print(sub.round(2)); print(g.round(2)); print(cl.round(2))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--data"); ap.add_argument("--demo", action="store_true")
    ap.add_argument("--n", type=int, default=2600); ap.add_argument("--out", default="analysis_demo")
    ap.add_argument("--no_weights", action="store_true", help="sensitivity: ignore survey weights")
    ap.add_argument("--outcome", default="y", help="outcome column, e.g. animal_fluency")
    ap.add_argument("--treat", default="T", help="exposure column, e.g. T_nontea")
    ap.add_argument("--extra_covs", nargs="*", default=[], help="covariates beyond COVARIATES, e.g. caffeine_mg")
    ap.add_argument("--drop_covs", nargs="*", default=[], help="covariates to leave out, e.g. EQP")
    ap.add_argument("--shap_n", type=int, default=600, help="people sampled for SHAP (cost grows with this)")
    a = ap.parse_args()
    df = simulate(a.n, seed=2026) if a.demo else pd.read_csv(a.data)
    if a.no_weights and "w" in df.columns:
        df = df.drop(columns="w")
    if a.outcome != "y":
        df["y"] = df[a.outcome]
    if a.treat != "T":
        df["T"] = df[a.treat]
    main(df, a.out, extra_covs=a.extra_covs, drop_covs=a.drop_covs, shap_n=a.shap_n)
