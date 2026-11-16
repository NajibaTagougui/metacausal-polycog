"""Build the MetaCausal-PolyCog Study A analysis files (NHANES 1999-2002, age >= 60).

Usage:
    python build_studyA.py --xpt ../nhanes_xpt --poly ../usda/foodcode_polyphenols_0710.csv --out .

Outputs:
    nhanes_studyA_full.csv     all eligible people (for multiple imputation)
    nhanes_studyA.csv          complete cases on outcome, exposure and all covariates incl. EQP/ENLH
    nhanes_studyA_noEQP.csv    complete cases without requiring EQP (for non-metabotype analyses)
    audit.txt                  variable audit, exclusion flow and distributions

Changes from the original protocol (see audit.txt):
  * weight  WTSPH4YR (CDC 4-year phytoestrogen subsample weight) instead of WTSB2YR/2
  * no daidzein LOD flag exists; values below LOD are LOD/sqrt(2) fills and are detected by value
  * 2001-02 alcohol variable is ALD100 (ALQ100 in 1999-2000)
  * added covariate caffeine_mg (DRXTCAFF) because tea supplies ~77% of flavonoid mg
  * added exposure T_nontea (flavonoids excluding tea, food codes 923xxxxx)
  * EQP is missing when unclassifiable (equol < LOD with a ratio above the cut, or daidzein < LOD)
"""
import argparse, io, os
import numpy as np
import pandas as pd

p = argparse.ArgumentParser()
p.add_argument('--xpt', required=True)
p.add_argument('--poly', required=True)
p.add_argument('--out', default='.')
p.add_argument('--eqp_cut', type=float, default=-1.75)
p.add_argument('--enl_quantile', type=float, default=0.6, help='ENLH = ratio >= this quantile (0.6 = top 40%%)')
p.add_argument('--tertile', type=float, default=2 / 3)
a = p.parse_args()
os.makedirs(a.out, exist_ok=True)
log = io.StringIO()
def say(*x):
    print(*x); print(*x, file=log)

def rd(name, suf, cols):
    df = pd.read_sas(os.path.join(a.xpt, f'{name}{suf}.xpt'), format='xport')
    miss = [c for c in cols if c not in df.columns]
    say(f'  {name}{suf:<3} rows={len(df):>6}  ' + ('all found' if not miss else f'MISSING {miss}'))
    return df[['SEQN'] + [c for c in cols if c in df.columns]]

def na(s, bad=(7, 9, 77, 99, 777, 999)):
    return s.where(~s.isin(bad))

say('== Variable audit')
cyc = []
for year, suf in [(1999, ''), (2001, '_B')]:
    alq = 'ALQ100' if year == 1999 else 'ALD100'
    rec = 'DRDDRSTS' if year == 1999 else 'DRDDRSTZ'
    d = rd('DEMO', suf, ['RIDAGEYR', 'RIAGENDR', 'RIDRETH1', 'DMDEDUC2', 'INDFMPIR', 'SDMVSTRA', 'SDMVPSU'])
    for name, cols in [('CFQ', ['CFDRIGHT']),
                       ('PHPYPA', ['WTSPH4YR', 'URXEQU', 'URXDAZ', 'URXGNS', 'URXETL', 'URXETD', 'URXUCR']),
                       ('BMX', ['BMXBMI']), ('DIQ', ['DIQ010']), ('BPQ', ['BPQ020']),
                       ('SMQ', ['SMQ020', 'SMQ040']),
                       ('MCQ', ['MCQ160B', 'MCQ160C', 'MCQ160E', 'MCQ160F']),
                       ('PAQ', ['PAD020', 'PAD200', 'PAD320']), ('ALQ', [alq]),
                       ('DRXTOT', [rec, 'DRXTKCAL', 'DRXTCAFF', 'DRXTALCO'])]:
        d = d.merge(rd(name, suf, cols), on='SEQN', how='left')
    d = d.rename(columns={alq: 'ALQ100', rec: 'recall_status'})
    d['cycle'] = year
    cyc.append(d)
D = pd.concat(cyc, ignore_index=True)

say('\n== Eligibility flow')
say(f'  NHANES 1999-2002 participants: {len(D)}')
D = D[D.RIDAGEYR >= 60]; say(f'  age >= 60:                     {len(D)}')
D = D[D.CFDRIGHT.notna()]; say(f'  + DSST score:                  {len(D)}')
D = D[D.URXDAZ.notna() & (D.WTSPH4YR > 0)].copy(); say(f'  + urinary phytoestrogens:      {len(D)}  (by cycle {D.cycle.value_counts().to_dict()})')

# ---------------- exposure ----------------
poly = pd.read_csv(a.poly)
F = pd.concat([pd.read_sas(os.path.join(a.xpt, f'DRXIFF{s}.xpt'), format='xport')[['SEQN', 'DRDIFDCD', 'DRXIGRMS']]
               for s in ['', '_B']])
F = F[F.SEQN.isin(D.SEQN)].copy()
F['code'] = F.DRDIFDCD.astype('int64')
F = F.merge(poly, on='code', how='left')
matched = F.tot.notna()
say(f'\n== Food matching: {matched.mean():.1%} of records, {F.DRXIGRMS[matched].sum() / F.DRXIGRMS.sum():.1%} of grams matched')
classes = ['tot', 'anth', 'f3ol', 'flavanone', 'flavonol', 'iso']
for c in classes:
    F[c] = F.DRXIGRMS * F[c].fillna(0) / 100
F['tea'] = F.code.astype(str).str.startswith('923')
I = F.groupby('SEQN')[classes].sum().add_suffix('_mg')
I['tea_mg'] = F[F.tea].groupby('SEQN').tot.sum()
I['tea_any'] = F[F.tea].groupby('SEQN').DRXIGRMS.sum().gt(0).astype(int)
I = I.fillna({'tea_mg': 0, 'tea_any': 0})
I['nontea_mg'] = I.tot_mg - I.tea_mg
D = D.merge(I, left_on='SEQN', right_index=True, how='left')

valid = D.tot_mg.notna() & (D.recall_status == 1) & D.DRXTKCAL.between(500, 5000)
say(f'  with food records: {D.tot_mg.notna().sum()}; reliable recall and kcal 500-5000: {valid.sum()}')

def energy_adjust(x):
    """Residual method: log1p(x) regressed on kcal, residual + prediction at mean kcal."""
    y = np.log1p(D.loc[valid, x]); k = D.loc[valid, 'DRXTKCAL']
    b = np.polyfit(k, y, 1)
    out = pd.Series(np.nan, index=D.index)
    out[valid] = y - np.polyval(b, k) + np.polyval(b, k.mean())
    return out

for src, nm in [('tot_mg', 'tot'), ('nontea_mg', 'nontea'), ('iso_mg', 'iso')]:
    D[f'{nm}_eadj'] = energy_adjust(src)
for nm in ['tot', 'nontea']:
    cut = D[f'{nm}_eadj'].quantile(a.tertile)
    D[f'T_{nm}'] = np.where(D[f'{nm}_eadj'].notna(), (D[f'{nm}_eadj'] >= cut).astype(float), np.nan)
D['T'] = D['T_tot']

# ---------------- metabotypes ----------------
eq_nd = D.URXEQU <= D.groupby('cycle').URXEQU.transform('min') + 1e-9   # LOD/sqrt(2) fill = cycle minimum
daz_nd = D.URXDAZ <= D.URXDAZ.min() + 1e-9
r = np.log10(D.URXEQU / D.URXDAZ)
D['log_eq_daz'] = r
D['eqp_status'] = np.select(
    [D.URXEQU.isna(), daz_nd, (r > a.eqp_cut) & eq_nd, r > a.eqp_cut, r <= a.eqp_cut],
    ['equol_missing', 'daidzein_below_LOD', 'equol_below_LOD_ambiguous', 'producer', 'non_producer'], 'other')
D['EQP'] = D.eqp_status.map({'producer': 1.0, 'non_producer': 0.0})
f = D.URXETL / (D.URXETL + D.URXETD)
D['enl_fraction'] = f
D['ENLH'] = np.where(f.notna(), (f >= f.quantile(a.enl_quantile)).astype(float), np.nan)
for c, nm in [('URXDAZ', 'daz'), ('URXGNS', 'gns'), ('URXEQU', 'equ'), ('URXETL', 'enl'), ('URXETD', 'end')]:
    D[f'u_{nm}_ugg'] = D[c] / D.URXUCR * 100          # ng/mL per mg/dL creatinine -> ug/g creatinine

# ---------------- covariates ----------------
D['y'] = D.CFDRIGHT
D['age'] = D.RIDAGEYR
D['female'] = (D.RIAGENDR == 2).astype(float)
D['race_hisp'] = D.RIDRETH1.isin([1, 2]).astype(float)
D['race_black'] = (D.RIDRETH1 == 4).astype(float)
D['race_other'] = (D.RIDRETH1 == 5).astype(float)
edu = na(D.DMDEDUC2)
D['edu_college'] = np.where(edu.notna(), (edu == 5).astype(float), np.nan)
D['edu_lt_hs'] = np.where(edu.notna(), edu.isin([1, 2]).astype(float), np.nan)
D['pir'] = D.INDFMPIR
D['bmi'] = D.BMXBMI
s20, s40 = na(D.SMQ020), na(D.SMQ040)
D['smoker'] = np.where(s20 == 2, 0.0, np.where(s20 == 1, np.where(s40.isin([1, 2]), 1.0, np.where(s40 == 3, 0.0, np.nan)), np.nan))
D['diabetes'] = na(D.DIQ010).map({1: 1.0, 2: 0.0, 3: 0.0})
D['hypertension'] = na(D.BPQ020).map({1: 1.0, 2: 0.0})
cv = pd.concat([na(D[c]).map({1: 1.0, 2: 0.0}) for c in ['MCQ160B', 'MCQ160C', 'MCQ160E', 'MCQ160F']], axis=1)
D['cvd'] = np.where(cv.eq(1).any(axis=1), 1.0, np.where(cv.notna().all(axis=1), 0.0, np.nan))
pa = pd.concat([na(D[c]).map({1: 1.0, 2: 0.0, 3: 0.0}) for c in ['PAD020', 'PAD200', 'PAD320']], axis=1)
D['phys_active'] = np.where(pa.eq(1).any(axis=1), 1.0, np.where(pa.notna().all(axis=1), 0.0, np.nan))
D['kcal'] = D.DRXTKCAL.where(valid)
D['alcohol'] = na(D.ALQ100).map({1: 1.0, 2: 0.0})
D['caffeine_mg'] = D.DRXTCAFF.where(valid)
D['w'] = D.WTSPH4YR
D['strata'] = D.SDMVSTRA
D['psu'] = D.SDMVPSU

COVS = ['age', 'female', 'race_hisp', 'race_black', 'race_other', 'edu_college', 'edu_lt_hs', 'pir', 'bmi',
        'smoker', 'diabetes', 'hypertension', 'cvd', 'phys_active', 'kcal', 'alcohol',
        'caffeine_mg', 'EQP', 'ENLH']
keep = (['SEQN', 'cycle', 'y', 'T', 'T_tot', 'T_nontea', 'tot_eadj', 'nontea_eadj', 'iso_eadj'] + COVS +
        ['eqp_status', 'log_eq_daz', 'enl_fraction', 'w', 'strata', 'psu', 'recall_status',
         'tot_mg', 'nontea_mg', 'tea_mg', 'tea_any'] + [c + '_mg' for c in classes[1:]] +
        ['URXDAZ', 'URXGNS', 'URXEQU', 'URXETL', 'URXETD', 'URXUCR'] + [c for c in D.columns if c.startswith('u_')])
full = D[keep].reset_index(drop=True)
full.to_csv(os.path.join(a.out, 'nhanes_studyA_full.csv'), index=False)

say('\n== Missingness in eligible sample (n=%d)' % len(full))
for c in ['T', 'T_nontea'] + COVS:
    m = full[c].isna().sum()
    if m: say(f'  {c:<15} missing {m:>4} ({m / len(full):.0%})')
say('\n  EQP status: ' + str(full.eqp_status.value_counts().to_dict()))

cc = full.dropna(subset=['y', 'T', 'T_nontea'] + COVS)
cc.to_csv(os.path.join(a.out, 'nhanes_studyA.csv'), index=False)
cc2 = full.dropna(subset=['y', 'T', 'T_nontea'] + [c for c in COVS if c != 'EQP'])
cc2.to_csv(os.path.join(a.out, 'nhanes_studyA_noEQP.csv'), index=False)
say(f'\n== Complete cases: with EQP {len(cc)}; without requiring EQP {len(cc2)}')
for nm, df in [('with EQP', cc), ('without EQP', cc2)]:
    say(f'  [{nm}] T=1: {int(df["T"].sum())}, T_nontea=1: {int(df.T_nontea.sum())}, tea drinkers: {int(df.tea_any.sum())}, '
        f'EQP=1/0: {int((df.EQP == 1).sum())}/{int((df.EQP == 0).sum())}, ENLH=1: {int(df.ENLH.sum())}')
say('\n  EQP x ENLH (complete cases with EQP):\n' + pd.crosstab(cc.EQP, cc.ENLH, margins=True).to_string())

say('\n== Exposure contrasts (complete cases without EQP)')
g = cc2.groupby('T')
say('  ' + pd.DataFrame({'n': g.size(), 'tea_any': g.tea_any.mean(), 'caffeine_mg': g.caffeine_mg.mean(),
                         'DSST': g.y.mean(), 'edu_college': g.edu_college.mean(), 'age': g.age.mean()}).round(2).to_string().replace('\n', '\n  '))
x, t = cc2.y, cc2['T']
say(f'  crude DSST difference T=1 vs 0: {x[t == 1].mean() - x[t == 0].mean():.1f}')
x, t = cc2.y, cc2.T_nontea
say(f'  crude DSST difference T_nontea=1 vs 0: {x[t == 1].mean() - x[t == 0].mean():.1f}  '
    f'(tea drinkers in T_nontea=1: {cc2.tea_any[t == 1].mean():.0%})')
open(os.path.join(a.out, 'audit.txt'), 'w').write(log.getvalue())
