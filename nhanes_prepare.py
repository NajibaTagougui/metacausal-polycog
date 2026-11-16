"""
Build the Study A analysis file from public NHANES 1999-2000 + 2001-2002.

  python nhanes_prepare.py --polyphenol_table foodcode_polyphenols.csv
  python nhanes_prepare.py --local_dir nhanes_xpt --polyphenol_table foodcode_polyphenols.csv

Files are read from --local_dir if present (name.xpt or NAME.XPT), otherwise downloaded
from wwwn.cdc.gov. The script prints a VARIABLE AUDIT: check every "MISSING" line against
the CDC codebook for that cycle and add the correct name to CANDIDATES below.

Outputs
  nhanes_studyA_full.csv     every eligible person, with missing values (for multiple imputation)
  nhanes_studyA.csv          complete cases on the analysis variables (input to analyze.py)
"""
import argparse, os
import numpy as np
import pandas as pd

BASE = "https://wwwn.cdc.gov/Nchs/Data/Nhanes/Public/{year}/DataFiles/{name}.xpt"
CYCLES = {1999: "", 2001: "_B"}
FILES = ["DEMO", "CFQ", "PHPYPA", "BMX", "DIQ", "BPQ", "SMQ", "MCQ", "PAQ", "ALQ", "DRXTOT"]

# variable -> candidate NHANES names (first found is used)
CANDIDATES = {
    "age": ["RIDAGEYR"], "sex": ["RIAGENDR"], "race": ["RIDRETH1"], "edu": ["DMDEDUC2"],
    "pir": ["INDFMPIR"], "strata": ["SDMVSTRA"], "psu": ["SDMVPSU"],
    "dsst": ["CFDRIGHT"],
    "w_sub": ["WTSB2YR"],
    "equol": ["URXEQU"], "daidzein": ["URXDAZ"], "daidzein_lod": ["URDDAZLC"],
    "enl": ["URXETL"], "end": ["URXETD"],
    "bmi": ["BMXBMI"], "diab": ["DIQ010"], "htn": ["BPQ020"],
    "smk100": ["SMQ020"], "smknow": ["SMQ040"],
    "chf": ["MCQ160B"], "chd": ["MCQ160C"], "mi": ["MCQ160E"], "stroke": ["MCQ160F"],
    "walk": ["PAD020"], "modpa": ["PAQ100"], "alc12": ["ALQ100", "ALQ101"],
    "kcal": ["DRXTKCAL", "DRDTKCAL"],
}
FOODCODE = ["DRDIFDCD", "DRXIFDCD", "DR1IFDCD"]
GRAMS = ["DRXIGRMS", "DR1IGRMS"]


def _read_xpt(p):
    try:
        return pd.read_sas(p, format="xport")
    except Exception:
        import pyreadstat                      # handles XPORT v8 and odd headers
        return pyreadstat.read_xport(p)[0]


def read(year, name, local_dir=None):
    if local_dir:
        for fn in (f"{name}.xpt", f"{name}.XPT", f"{name.lower()}.xpt"):
            p = os.path.join(local_dir, fn)
            if os.path.exists(p):
                return _read_xpt(p)
    url = BASE.format(year=year, name=name)
    print("  downloading", url)
    return pd.read_sas(url, format="xport")


def pick(df, key, audit):
    for c in CANDIDATES[key]:
        if c in df.columns:
            audit[key] = c
            return df[c]
    audit[key] = "MISSING"
    return pd.Series(np.nan, index=df.index)


def yes(s):            # NHANES coding: 1 = yes, 2 = no, 7/9 = refused/don't know
    return s.map({1: 1.0, 2: 0.0})


def build(local_dir=None, polyphenol_table=None, enl_quantile=0.60, eqp_cut=-1.75):
    parts, audit = [], {}
    for year, suf in CYCLES.items():
        print(f"cycle {year}")
        d = None
        for f in FILES:
            try:
                x = read(year, f + suf, local_dir)
            except Exception as ex:
                print(f"  WARNING could not read {f + suf}: {ex}")
                continue
            d = x if d is None else d.merge(x, on="SEQN", how="left", suffixes=("", f"_{f}"))
        if d is None or "CFDRIGHT" not in d.columns or "RIDAGEYR" not in d.columns:
            raise SystemExit(f"core files (DEMO{suf}, CFQ{suf}) could not be read for {year}: "
                             "download them into --local_dir (see the exact steps)")
        d["cycle"] = year
        parts.append(d)
    raw = pd.concat(parts, ignore_index=True)

    out = pd.DataFrame({"SEQN": raw.SEQN, "cycle": raw.cycle})
    v = {k: pick(raw, k, audit) for k in CANDIDATES}

    out["age"] = v["age"]
    out["female"] = (v["sex"] == 2).astype(float)
    out["race_black"] = (v["race"] == 4).astype(float)
    out["race_hisp"] = v["race"].isin([1, 2]).astype(float)
    out["race_other"] = (v["race"] == 5).astype(float)
    out["edu_college"] = np.where(v["edu"].isin([1, 2, 3, 4, 5]), (v["edu"] >= 4).astype(float), np.nan)
    out["edu_lt_hs"] = np.where(v["edu"].isin([1, 2, 3, 4, 5]), (v["edu"] <= 2).astype(float), np.nan)
    out["pir"] = v["pir"]
    out["bmi"] = v["bmi"]
    out["diabetes"] = yes(v["diab"])
    out["hypertension"] = yes(v["htn"])
    out["smoker"] = np.where(yes(v["smk100"]) == 0, 0.0, v["smknow"].map({1: 1.0, 2: 1.0, 3: 0.0}))
    cvd = pd.concat([yes(v[k]) for k in ("chf", "chd", "mi", "stroke")], axis=1)
    out["cvd"] = cvd.max(axis=1, skipna=True)
    out["phys_active"] = pd.concat([yes(v["walk"]), yes(v["modpa"])], axis=1).max(axis=1, skipna=True)
    out["alcohol"] = yes(v["alc12"])
    out["kcal"] = v["kcal"]
    out["y"] = v["dsst"]
    out["w"] = v["w_sub"] / len(CYCLES)              # combine 2 cycles (CDC rule)
    out["strata"], out["psu"] = v["strata"], v["psu"]

    # ---- metabotypes (dose-invariant)
    daz_ok = v["daidzein"] > 0
    if audit["daidzein_lod"] != "MISSING":
        daz_ok &= v["daidzein_lod"] != 1          # 1 = below limit of detection
    ratio = np.log10(v["equol"].clip(lower=1e-3) / v["daidzein"].where(daz_ok))
    out["eq_daz_log10"] = ratio
    out["EQP"] = np.where(ratio.notna(), (ratio > eqp_cut).astype(float), np.nan)
    enl_frac = v["enl"] / (v["enl"] + v["end"])
    out["enl_frac"] = enl_frac
    out["ENLH"] = np.where(enl_frac.notna(), (enl_frac >= enl_frac.quantile(enl_quantile)).astype(float), np.nan)

    # ---- eligibility
    elig = (out.age >= 60) & out.y.notna() & v["equol"].notna()
    print(f"\neligible (age>=60, DSST, phytoestrogens): {elig.sum()} of {len(out)}")
    out = out[elig].copy()

    # ---- exposure: energy-adjusted polyphenol intake (residual method), top tertile
    if polyphenol_table:
        tab = pd.read_csv(polyphenol_table)
        intake = []
        for year, suf in CYCLES.items():
            f = read(year, "DRXIFF" + suf, local_dir)
            code = next(c for c in FOODCODE if c in f.columns)
            grams = next(c for c in GRAMS if c in f.columns)
            f = f.merge(tab, left_on=code, right_on="food_code", how="left")
            matched = f.mg_per_100g.notna().mean()
            print(f"  {year}: food records matched to polyphenol table: {matched:.1%}")
            um = f[f.mg_per_100g.isna()].groupby(code)[grams].sum().sort_values(ascending=False)
            um.rename("grams_total").to_csv(f"unmatched_foodcodes_{year}.csv")
            print(f"  {year}: grams matched: {1 - um.sum() / f[grams].sum():.1%}; "
                  f"unmatched codes -> unmatched_foodcodes_{year}.csv")
            f["pp_mg"] = f[grams] * f.mg_per_100g.fillna(0) / 100
            intake.append(f.groupby("SEQN").pp_mg.sum())
        out = out.merge(pd.concat(intake).rename("pp_mg"), left_on="SEQN", right_index=True, how="left")
        ok = out.pp_mg.notna() & out.kcal.between(500, 5000)
        lp = np.log1p(out.loc[ok, "pp_mg"])
        b = np.polyfit(out.loc[ok, "kcal"], lp, 1)
        out.loc[ok, "pp_adj"] = lp - np.polyval(b, out.loc[ok, "kcal"]) + lp.mean()
        out["T"] = np.where(out.pp_adj.notna(), (out.pp_adj >= out.pp_adj.quantile(2 / 3)).astype(float), np.nan)
    else:
        print("WARNING: no --polyphenol_table: exposure T not built")

    print("\nVARIABLE AUDIT (check MISSING against the CDC codebook):")
    for k, c in audit.items():
        print(f"  {k:<13} -> {c}")
    return out


ANALYSIS_VARS = ["age", "female", "race_black", "race_hisp", "race_other", "edu_college", "edu_lt_hs",
                 "pir", "bmi", "smoker", "diabetes", "hypertension", "cvd", "phys_active", "kcal",
                 "alcohol", "EQP", "ENLH", "T", "y", "w"]

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--local_dir", default=None)
    ap.add_argument("--polyphenol_table", default=None)
    ap.add_argument("--out", default="nhanes_studyA.csv")
    ap.add_argument("--download_only", default=None, help="folder: save all .xpt files there and stop")
    ap.add_argument("--enl_quantile", type=float, default=0.60)
    ap.add_argument("--eqp_cut", type=float, default=-1.75)
    a = ap.parse_args()
    if a.download_only:
        import urllib.request
        os.makedirs(a.download_only, exist_ok=True)
        for year, suf in CYCLES.items():
            for f in FILES + ["DRXIFF"]:
                url = BASE.format(year=year, name=f + suf)
                dest = os.path.join(a.download_only, f + suf + ".xpt")
                print("downloading", url)
                urllib.request.urlretrieve(url, dest)
        raise SystemExit(f"done: {len(CYCLES) * (len(FILES) + 1)} files in {a.download_only}")
    df = build(a.local_dir, a.polyphenol_table, a.enl_quantile, a.eqp_cut)
    df.to_csv(a.out.replace(".csv", "_full.csv"), index=False)
    keep = [c for c in ANALYSIS_VARS if c in df.columns]
    cc = df.dropna(subset=keep)
    cc.to_csv(a.out, index=False)
    print(f"\nfull file: {len(df)} rows | complete cases: {len(cc)} rows -> {a.out}")
    print("missing per variable:\n", df[keep].isna().sum()[lambda s: s > 0].to_string())
    if len(cc):
        print("equol producers: %.1f%% | ENL-dominant: %.1f%%" % (100 * cc.EQP.mean(), 100 * cc.ENLH.mean()))
