"""Replication: is spot-urine equol-producer status dose-dependent in every NHANES cycle?

    python replicate_eqp.py --xpt ../nhanes_xpt --out results_eqp_replication

Uses all six NHANES cycles with urinary phytoestrogens (1999-2010), adults >= 20
(and a 60+ subgroup). No cognition data needed.

Classification (Setchell & Cole cut-off, default -1.75 on log10 equol/daidzein):
  producer       equol detected and ratio > cut
  non_producer   ratio <= cut (if equol is below the LOD the true ratio is lower still)
  unclassifiable equol below LOD with a ratio above the cut, daidzein below LOD,
                 or equol missing
Detection limits: the URDxxxLC flags where they exist (2005-2010), otherwise the
cycle's LOD/sqrt(2) fill value (the cycle minimum).

The ratio can fall below the cut while equol is undetected only when
daidzein > LOD_equol / 10**cut. Below that "identifiability threshold" nobody can
be classified a non-producer. The script reports it per cycle.
"""
import argparse
from pathlib import Path
import numpy as np
import pandas as pd

ap = argparse.ArgumentParser()
ap.add_argument("--xpt", required=True)
ap.add_argument("--out", default="results_eqp_replication")
ap.add_argument("--cut", type=float, default=-1.75)
ap.add_argument("--min_age", type=int, default=20)
a = ap.parse_args()
out = Path(a.out); out.mkdir(exist_ok=True)

CYCLES = [("1999-2000", "DEMO", "PHPYPA"), ("2001-2002", "DEMO_B", "PHPYPA_B"),
          ("2003-2004", "DEMO_C", "L06PHY_C"), ("2005-2006", "DEMO_D", "PHYTO_D"),
          ("2007-2008", "DEMO_E", "PHYTO_E"), ("2009-2010", "DEMO_F", "PHYTO_F")]
rd = lambda f: pd.read_sas(Path(a.xpt) / f"{f}.xpt", format="xport")

frames = []
for cyc, demo, phy in CYCLES:
    D = rd(demo)[["SEQN", "RIDAGEYR"]]
    P = rd(phy)
    wcol = [c for c in P.columns if c.startswith("WT")][0]
    P = P.rename(columns={wcol: "w"})
    keep = ["SEQN", "w", "URXDAZ", "URXEQU", "URXETL", "URXETD"] + [c for c in ("URDDAZLC", "URDEQULC") if c in P]
    m = D.merge(P[keep], on="SEQN")
    m = m[(m.RIDAGEYR >= a.min_age) & m.URXDAZ.notna() & (m.w > 0)].copy()
    if "URDEQULC" in m:
        m["eq_nd"] = m.URDEQULC == 1
        m["daz_nd"] = m.URDDAZLC == 1
        lod_eq = m.loc[m.eq_nd, "URXEQU"].max() * np.sqrt(2) if m.eq_nd.any() else np.nan
    else:
        m["eq_nd"] = m.URXEQU <= m.URXEQU.min() + 1e-9
        m["daz_nd"] = m.URXDAZ <= m.URXDAZ.min() + 1e-9
        lod_eq = m.URXEQU.min() * np.sqrt(2)
    r = np.log10(m.URXEQU / m.URXDAZ)
    m["status"] = np.select(
        [m.URXEQU.isna(), m.daz_nd, (r > a.cut) & m.eq_nd, r > a.cut, r <= a.cut],
        ["unclassifiable", "unclassifiable", "unclassifiable", "producer", "non_producer"], "unclassifiable")
    m["cycle"] = cyc
    m["lod_equol"] = lod_eq
    m["ident_threshold"] = lod_eq / 10 ** a.cut
    frames.append(m)
A = pd.concat(frames, ignore_index=True)


def shares(df, weighted=False):
    w = df.w if weighted else pd.Series(1.0, index=df.index)
    s = w.groupby(df.status).sum() / w.sum()
    return s.reindex(["producer", "non_producer", "unclassifiable"]).fillna(0)


rows = []
for (cyc, grp), df in [((c, "all adults"), d) for c, d in A.groupby("cycle")] + \
                      [(("pooled", "all adults"), A), (("pooled", "aged 60+"), A[A.RIDAGEYR >= 60])]:
    s, sw = shares(df), shares(df, True)
    thr = df.ident_threshold.iloc[0] if cyc != "pooled" else np.nan
    above = df.URXDAZ > df.ident_threshold
    cl = df[above & (df.status != "unclassifiable")]
    rows.append(dict(cycle=cyc, group=grp, n=len(df),
                     producer=s.producer, non_producer=s.non_producer, unclassifiable=s.unclassifiable,
                     producer_weighted=sw.producer,
                     lod_equol_ng_ml=df.lod_equol.iloc[0] if cyc != "pooled" else np.nan,
                     ident_threshold_ng_ml=thr,
                     share_above_threshold=above.mean(),
                     non_producers_above_threshold=(df.status[df.status == "non_producer"].index.isin(df[above].index)).mean(),
                     producer_share_if_identifiable=(cl.status == "producer").mean(), n_identifiable=len(cl)))
S = pd.DataFrame(rows)
S.to_csv(out / "eqp_by_cycle.csv", index=False)

A["daz_quartile"] = A.groupby("cycle").URXDAZ.transform(lambda x: pd.qcut(x, 4, labels=False) + 1)
Q = (pd.crosstab([A.cycle, A.daz_quartile], A.status, normalize="index") * 100).round(1)
Q.to_csv(out / "eqp_by_daidzein_quartile.csv")
Qp = (pd.crosstab(A.daz_quartile, A.status, normalize="index") * 100).round(1)

pd.set_option("display.width", 220)
print(S.round(3).to_string(index=False))
print("\nPooled, status by within-cycle daidzein quartile (%):\n", Qp)
print("\nNon-producer share in daidzein quartile 4, by cycle (%):\n",
      Q.xs(4, level="daz_quartile")["non_producer"].to_string())
