"""Benchmark + power figures for the manuscript (reads results*/ CSVs)."""
import pandas as pd, numpy as np, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False,
                     "axes.grid": True, "grid.color": "#e6e5e1", "grid.linewidth": 0.6,
                     "axes.edgecolor": "#52514e", "xtick.color": "#52514e", "ytick.color": "#52514e",
                     "savefig.dpi": 300})
COL = {"KG-CausalForest": "#2a78d6", "CausalForest": "#eb6834", "OLS-interaction": "#1baf7a",
       "S-learner": "#eda100", "X-learner": "#e87ba4", "DR-learner": "#008300", "T-learner": "#6250d6"}
LBL = {"KG-CausalForest": "KG-causal forest (ours)", "CausalForest": "Causal forest",
       "OLS-interaction": "OLS + interactions", "S-learner": "S-learner", "X-learner": "X-learner",
       "DR-learner": "DR-learner", "T-learner": "T-learner"}


def bench():
    fig, ax = plt.subplots(1, 2, figsize=(10, 4), sharey=True)
    for a, (f, title) in zip(ax, [("results", "A. Additive effects"),
                                  ("results_nonlinear", "B. Nonlinear (threshold) effects")]):
        d = pd.read_csv(f"{f}/benchmark_raw.csv")
        d = d[d.learner.isin(COL)]
        g = d.groupby(["learner", "n"]).spearman.agg(["mean", "std", "count"]).reset_index()
        for L in COL:
            s = g[g.learner == L]
            if s.empty: continue
            lw = 2.6 if L == "KG-CausalForest" else 1.5
            a.errorbar(s.n, s["mean"], yerr=1.96 * s["std"] / np.sqrt(s["count"]), marker="o",
                       ms=6 if L == "KG-CausalForest" else 4, lw=lw, capsize=2, color=COL[L], label=LBL[L])
        a.set_xscale("log"); a.set_xticks(sorted(d.n.unique()), [f"{n:,}" for n in sorted(d.n.unique())])
        a.minorticks_off(); a.set_xlabel("Sample size"); a.set_title(title, loc="left")
    ax[0].set_ylabel("Rank correlation with true individual effect")
    ax[1].legend(frameon=False, fontsize=8, loc="lower right")
    plt.tight_layout(); plt.savefig("figures/fig_benchmark.png"); plt.close()


def power():
    p = pd.read_csv("results/power.csv")
    fig, ax = plt.subplots(figsize=(7, 5))
    for scale, ls in [(1.0, "-"), (0.5, "--")]:
        s = p[p.effect_scale == scale]
        ax.plot(s.n, s.power_EQP, ls, marker="o", color="#2a78d6",
                label=f"Equol producer vs not ({'full' if scale == 1 else 'half'} effect)")
        ax.plot(s.n, s.power_both_vs_none, ls, marker="s", color="#eb6834",
                label=f"Both metabotypes vs neither ({'full' if scale == 1 else 'half'} effect)")
    ax.axhline(0.8, color="#8a8984", lw=1); ax.text(615, 0.815, "80% power", color="#52514e", fontsize=8)
    ax.axvspan(700, 900, color="#f0efec", zorder=0); ax.text(720, 0.03, "NHANES\nStudy A", fontsize=8, color="#52514e")
    ax.set_xscale("log"); ax.set_xticks(sorted(p.n.unique()), [f"{n:,}" for n in sorted(p.n.unique())])
    ax.minorticks_off(); ax.set_ylim(0, 1.02); ax.set_xlabel("Sample size"); ax.set_ylabel("Power (α = 0.05)")
    ax.set_title("Power of the confirmatory metabotype contrasts", loc="left")
    ax.legend(frameon=False, fontsize=8, loc="upper center", bbox_to_anchor=(0.5, -0.18), ncol=2)
    plt.tight_layout(); plt.savefig("figures/fig_power.png"); plt.close()


if __name__ == "__main__":
    import os; os.makedirs("figures", exist_ok=True)
    bench(); power()
