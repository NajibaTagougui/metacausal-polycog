"""
Make foodcode_polyphenols.csv (food_code, mg_per_100g) from the USDA FNDDS Flavonoid
Database (Database of Flavonoid Values for USDA Food Codes, Food Surveys Research Group),
optionally adding lignans from a Phenol-Explorer-derived file you prepare.

  python build_polyphenol_table.py --flav FlavonoidDB_0710.xlsx --out foodcode_polyphenols.csv
  python build_polyphenol_table.py --flav FlavonoidDB_0710.xlsx --lignans lignans_by_foodcode.csv

The script finds the food-code column and the TOTAL flavonoid column by name
(case-insensitive); if it guesses wrong, pass --code_col / --total_col explicitly.
If no total column exists it sums every numeric flavonoid column (mg/100 g).
"""
import argparse
import pandas as pd

ap = argparse.ArgumentParser()
ap.add_argument("--flav", required=True)
ap.add_argument("--sheet", default=0)
ap.add_argument("--lignans", default=None, help="CSV: food_code, lignan_mg_per_100g")
ap.add_argument("--code_col", default=None)
ap.add_argument("--total_col", default=None)
ap.add_argument("--out", default="foodcode_polyphenols.csv")
a = ap.parse_args()

f = pd.read_excel(a.flav, sheet_name=a.sheet) if a.flav.endswith(("xls", "xlsx")) else pd.read_csv(a.flav)
cols = {c.lower(): c for c in f.columns}
code = a.code_col or next(cols[c] for c in cols if "food" in c and "code" in c)
total = a.total_col or next((cols[c] for c in cols if "total" in c and "flav" in c), None)
if total is None:
    num = [c for c in f.select_dtypes("number").columns if c != code]
    print("no total column found; summing:", num)
    f["mg_per_100g"] = f[num].sum(axis=1)
else:
    f["mg_per_100g"] = f[total]
out = f[[code, "mg_per_100g"]].rename(columns={code: "food_code"})
if a.lignans:
    lg = pd.read_csv(a.lignans)
    out = out.merge(lg, on="food_code", how="outer").fillna(0)
    out["mg_per_100g"] += out.pop("lignan_mg_per_100g")
out.to_csv(a.out, index=False)
print(f"{len(out)} food codes written to {a.out}; code column = {code}; total = {total or 'summed'}")
