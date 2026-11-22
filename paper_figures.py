"""Figures 2-4 of the paper (Figure 1 is made by trend_test.py).

    python paper_figures.py --out figures

  fig2_gates.png      GATES by predicted-effect quintile, sample A vs sample B
  fig3_learners.png   (a) sqrt-PEHE by learner and n; (b) rank correlation by label scenario
  fig4_power.png      design power for H1 against recruited n
"""
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ap = argparse.ArgumentParser()
ap.add_argument("--out", default="figures")
a = ap.parse_args()
out = Path(a.out); out.mkdir(exist_ok=True)

INK, MUTED, GRID = "#0b0b0b", "#52514e", "#e6e5e1"
CAT = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]
plt.rcParams.update({"font.size": 9, "axes.spines.top": False, "axes.spines.right": False,
                     "axes.edgecolor": MUTED, "axes.labelcolor": INK, "xtick.color": MUTED,
                     "ytick.color": MUTED, "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6,
                     "axes.axisbelow": True, "savefig.dpi": 300, "figure.facecolor": "white",
                     "legend.frameon": False, "lines.linewidth": 2})

# ---- Figure 2: GATES, sample A vs B
fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.0), sharey=True)
for ax, (d, title) in zip(axes, [("analysis_studyA_noEQP", "A. NHANES 1999–2002 (n = 685)"),
                                 ("analysis_studyB_twoday", "B. NHANES 2011–2014 (n = 2,253)")]):
    g = pd.read_csv(f"{d}/gates.csv")
    ax.axhline(0, color=MUTED, lw=0.8)
    ax.errorbar(g.group, g.gate, yerr=1.96 * g.se, fmt="o", ms=6, color=CAT[0], ecolor=CAT[0],
                elinewidth=1.5, capsize=0, mec="white", mew=1.5)
    ax.set_title(title, loc="left", fontsize=9, color=INK)
    ax.set_xticks(g.group); ax.set_xlabel("Quintile of out-of-fold predicted effect (1 = lowest)")
axes[0].set_ylabel("Effect on DSST, AIPW (points, 95% CI)")
fig.tight_layout(); fig.savefig(out / "fig2_gates.png"); plt.close(fig)

# ---- Figure 3: learners
NAMES = {"BCF": "Bayesian causal forest", "KG-CausalForest": "KG-CF", "CausalForest": "Causal forest",
         "OLS-interaction": "OLS + interactions", "S-learner": "S-learner", "X-learner": "X-learner",
         "R-learner": "R-learner"}
COLOR = dict(zip(NAMES, CAT))
B = pd.concat([pd.read_csv("results_bench50/benchmark_raw.csv"), pd.read_csv("results_baselines/benchmark_raw.csv")])
B = B[B.learner.isin(NAMES)].groupby(["learner", "n"]).sqrt_pehe.agg(["mean", "std", "count"]).reset_index()
L = pd.read_csv("results_label/label_summary.csv", index_col=0)
Lb = pd.read_csv("results_label_baselines/label_summary.csv", index_col=0)
labels = ["true", "nondiff", "confounded", "post"]
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9.0, 3.6), gridspec_kw={"width_ratios": [1, 1.1]})
order = B[B.n == 2600].sort_values("mean").learner.tolist()
for i, lname in enumerate(order):
    for n, mk in [(600, "o"), (2600, "s")]:
        r = B[(B.learner == lname) & (B.n == n)].iloc[0]
        ax1.plot(r["mean"], i, mk, ms=7, mec=COLOR[lname], mew=1.6,
                 mfc=COLOR[lname] if n == 2600 else "white")
ax1.set_yticks(range(len(order))); ax1.set_yticklabels([NAMES[l] for l in order]); ax1.invert_yaxis()
ax1.set_xlabel("√PEHE (DSST points; lower is better)")
ax1.plot([], [], "o", mfc="white", mec=MUTED, mew=1.6, ms=7, label="n = 600")
ax1.plot([], [], "s", mfc=MUTED, mec=MUTED, ms=7, label="n = 2,600")
ax1.legend(loc="upper center", bbox_to_anchor=(0.5, -0.2), ncol=2, fontsize=8)
ax1.set_title("a. Accuracy with the true modifier", loc="left", fontsize=9, color=INK)
x = np.arange(len(labels))
for lname in NAMES:
    src = Lb if lname in ("BCF", "R-learner") else L
    rho = [src.loc[f"rho_{lname}", lab] for lab in labels]
    ax2.plot(x, rho, "-o", color=COLOR[lname], ms=6, mec="white", mew=1.2, label=NAMES[lname])
ax2.set_xticks(x); ax2.set_xticklabels(["True\ntrait", "10% random\nerror", "Spot urine\n(SES)", "Spot urine\n(treatment)"])
ax2.set_ylabel("Spearman ρ with true effect"); ax2.set_ylim(0, 0.9)
ax2.set_title("b. Accuracy as the modifier label degrades (n = 2,600)", loc="left", fontsize=9, color=INK)
ax2.legend(fontsize=7.5, loc="upper left", bbox_to_anchor=(1.01, 1.0))
fig.tight_layout(); fig.savefig(out / "fig3_learners.png"); plt.close(fig)

# ---- Figure 4: design power
S = pd.read_csv("results_studyC/studyC_power.csv")
S = S[(S.delta == 3.0) & (S.enrichment == "balanced")]
DES = [("challenge", "randomised", "Soy challenge + randomised exposure"),
       ("challenge", "repeated", "Soy challenge + repeated recalls"),
       ("challenge", "single_recall", "Soy challenge + single recall"),
       ("spontaneous", "randomised", "Spot urine + randomised exposure")]
fig, ax = plt.subplots(figsize=(5.8, 4.0))
ax.axhline(0.8, color=MUTED, lw=0.8, ls="--")
ax.text(S.n_recruited.min(), 0.81, "80% power", color=MUTED, fontsize=8, va="bottom")
for i, (cl, ex, lab) in enumerate(DES):
    d = S[(S.classification == cl) & (S.exposure == ex)].sort_values("n_recruited")
    ax.plot(d.n_recruited, d.power_H1, "-o", color=CAT[i], ms=6, mec="white", mew=1.2, label=lab)
ax.set_xlabel("People recruited (balanced 50% producers)"); ax.set_ylabel("Power for H1 (equol interaction)")
ax.set_ylim(0, 1.02); ax.set_xticks(sorted(S.n_recruited.unique()))
ax.legend(fontsize=7.5, loc="upper center", bbox_to_anchor=(0.5, -0.2), ncol=2)
ax.set_title("Simulated prospective designs, δ = 3 DSST points, residual SD 8", loc="left", fontsize=9, color=INK)
fig.tight_layout(); fig.savefig(out / "fig4_power.png"); plt.close(fig)
print("written:", sorted(p.name for p in out.glob("fig[234]_*.png")))
