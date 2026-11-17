"""Build the Study B (replication) analysis file: NHANES 2011-2012 + 2013-2014, age >= 60.

    python build_studyB.py --xpt ../nhanes_xpt --flav0710 ../usda/foodcode_polyphenols_0710.csv \
        --flav1718 ../usda/flav1718/out/flavval.sas7bdat --out data_studyB

No metabotype data exist in these cycles (urinary phytoestrogens stopped after 2010),
so Study B tests only whether the null average effect and the absence of
non-metabotype heterogeneity replicate.

Exposure: mean flavonoid intake over the reliable recall days (two where available),
energy-adjusted by the residual method, top tertile = T. Primary file = people with
two reliable recalls (weight WTDR2D / 2); the day-1-only file uses WTDRD1 / 2.
Food codes: USDA 2017-18 flavonoid values first, 2007-10 values where 2017-18 lacks
the code. Outcome y = DSST (CFDDS); CERAD delayed recall and animal fluency are kept.
Covariates follow Study A plus PHQ-9 (absent in 1999-2002).
"""
import argparse, io, os
import numpy as np
import pandas as pd

p = argparse.ArgumentParser()
p.add_argument('--xpt', required=True)
p.add_argument('--flav0710', required=True)
p.add_argument('--flav1718', required=True)
p.add_argument('--out', default='data_studyB')
p.add_argument('--tertile', type=float, default=2 / 3)
a = p.parse_args()
os.makedirs(a.out, exist_ok=True)
log = io.StringIO()
def say(*x):
    print(*x); print(*x, file=log)
rd = lambda f, cols=None: pd.read_sas(os.path.join(a.xpt, f + '.xpt'), format='xport')[(['SEQN'] + cols) if cols else slice(None)]
na = lambda s, bad=(7, 9, 77, 99, 777, 999): s.where(~s.isin(bad))
zero = lambda s: s.where(s.abs() > 1e-20, 0.0)          # SAS xport stores 0 as ~5e-79

# ---- polyphenol table: 2017-18 first, 2007-10 fallback
CLS = {7000: 'tot', 7100: 'anth', 7300: 'f3ol', 7400: 'flavanone', 7600: 'flavonol', 7700: 'iso'}
v = pd.read_sas(a.flav1718)
v = v[v.Nutrient_Code.isin(CLS)]
t18 = v.pivot_table(index='Food_code', columns='Nutrient_Code', values='Nutrient_value', aggfunc='first').rename(columns=CLS)
t18.index = t18.index.astype('int64')
t07 = pd.read_csv(a.flav0710).set_index('code')
poly = pd.concat([t18, t07[~t07.index.isin(t18.index)]])
poly['source'] = np.where(poly.index.isin(t18.index), '1718', '0710')
poly.index.name = 'code'

say('== Eligibility and variables')
cyc = []
for year, s in [(2011, 'G'), (2013, 'H')]:
    D = rd(f'DEMO_{s}', ['RIDAGEYR', 'RIAGENDR', 'RIDRETH1', 'DMDEDUC2', 'INDFMPIR', 'SDMVSTRA', 'SDMVPSU'])
    for f, cols in [('CFQ', ['CFDDS', 'CFDCSR', 'CFDAST']), ('BMX', ['BMXBMI']), ('DIQ', ['DIQ010']),
                    ('BPQ', ['BPQ020']), ('SMQ', ['SMQ020', 'SMQ040']),
                    ('MCQ', ['MCQ160B', 'MCQ160C', 'MCQ160E', 'MCQ160F']),
                    ('PAQ', ['PAQ605', 'PAQ620', 'PAQ635', 'PAQ650', 'PAQ665']), ('ALQ', ['ALQ101']),
                    ('DPQ', [f'DPQ0{i}0' for i in range(1, 10)]),
                    ('DR1TOT', ['WTDRD1', 'WTDR2D', 'DR1DRSTZ', 'DR1TKCAL', 'DR1TCAFF']),
                    ('DR2TOT', ['DR2DRSTZ', 'DR2TKCAL', 'DR2TCAFF'])]:
        df = rd(f'{f}_{s}')
        miss = [c for c in cols if c not in df.columns]
        if miss: say(f'  {f}_{s}: MISSING {miss}')
        D = D.merge(df[['SEQN'] + [c for c in cols if c in df.columns]], on='SEQN', how='left')
    # food records, both days
    for d in (1, 2):
        F = rd(f'DR{d}IFF_{s}', [f'DR{d}IFDCD', f'DR{d}IGRMS'])
        F['code'] = F[f'DR{d}IFDCD'].astype('int64')
        F = F.merge(poly, left_on='code', right_index=True, how='left')
        if d == 1 and year == 2011: allF = []
        allF.append(F.assign(day=d, cycle=year).rename(columns={f'DR{d}IGRMS': 'grams'}))
    D['cycle'] = year
    cyc.append(D)
D = pd.concat(cyc, ignore_index=True)
F = pd.concat(allF, ignore_index=True)
for c in ['DR1DRSTZ', 'DR2DRSTZ', 'WTDRD1', 'WTDR2D', 'DR1TCAFF', 'DR2TCAFF']:
    D[c] = zero(D[c])

say(f'  NHANES 2011-2014 participants: {len(D)}')
D = D[D.RIDAGEYR >= 60]; say(f'  age >= 60:                     {len(D)}')
D = D[D.CFDDS.notna()]; say(f'  + DSST score:                  {len(D)}')
D = D[(D.DR1DRSTZ == 1) & D.DR1TKCAL.between(500, 5000)].copy()
say(f'  + reliable day-1 recall, 500-5,000 kcal: {len(D)}')
D['two_days'] = ((D.DR2DRSTZ == 1) & D.DR2TKCAL.between(500, 5000)).astype(int)
say(f'  ... of whom with a reliable day-2 recall: {D.two_days.sum()}')

F = F[F.SEQN.isin(D.SEQN)]
m = F.tot.notna()
say(f'\n== Food matching: {m.mean():.1%} of records, {F.grams[m].sum() / F.grams.sum():.1%} of grams '
    f'({F.grams[F.source == "1718"].sum() / F.grams.sum():.1%} from 2017-18 values)')
um = F[~m].groupby('code').grams.sum().sort_values(ascending=False)
um.rename('grams').reset_index().to_csv(os.path.join(a.out, 'unmatched_foodcodes_studyB.csv'), index=False)
cls = list(CLS.values())
for c in cls:
    F[c] = F.grams * F[c].fillna(0) / 100
F['tea'] = F.code.astype(str).str.startswith('923')
F['tea_mg'] = F.tot.where(F.tea, 0)
per_day = F.groupby(['SEQN', 'day'])[cls + ['tea_mg']].sum().reset_index()


D2 = D[D.two_days == 1].copy()
for name, df, days in [('two-day', D2, (1, 2)), ('day-1', D, (1,))]:
    x = per_day[per_day.day.isin(days) & per_day.SEQN.isin(df.SEQN)].groupby('SEQN')[cls + ['tea_mg']].mean()
    df[[c + '_mg' for c in cls] + ['tea_mg']] = x.reindex(df.SEQN).fillna(0).values
    df['kcal'] = df.DR1TKCAL if days == (1,) else (df.DR1TKCAL + df.DR2TKCAL) / 2
    df['caffeine_mg'] = df.DR1TCAFF if days == (1,) else (df.DR1TCAFF + df.DR2TCAFF) / 2
    df['nontea_mg'] = df.tot_mg - df.tea_mg
    df['tea_any'] = (df.tea_mg > 0).astype(int)
    for src, nm in [('tot_mg', 'tot'), ('nontea_mg', 'nontea'), ('iso_mg', 'iso')]:
        y = np.log1p(df[src]); b = np.polyfit(df.kcal, y, 1)
        df[f'{nm}_eadj'] = y - np.polyval(b, df.kcal) + np.polyval(b, df.kcal.mean())
    for nm in ['tot', 'nontea']:
        df[f'T_{nm}'] = (df[f'{nm}_eadj'] >= df[f'{nm}_eadj'].quantile(a.tertile)).astype(float)
    df['T'] = df.T_tot
    df['w'] = (df.WTDR2D if days == (1, 2) else df.WTDRD1) / 2

    df['y'] = df.CFDDS
    df['cerad_delayed'] = df.CFDCSR
    df['animal_fluency'] = df.CFDAST
    df['age'] = df.RIDAGEYR
    df['female'] = (df.RIAGENDR == 2).astype(float)
    df['race_hisp'] = df.RIDRETH1.isin([1, 2]).astype(float)
    df['race_black'] = (df.RIDRETH1 == 4).astype(float)
    df['race_other'] = (df.RIDRETH1 == 5).astype(float)
    edu = na(df.DMDEDUC2)
    df['edu_college'] = np.where(edu.notna(), (edu == 5).astype(float), np.nan)
    df['edu_lt_hs'] = np.where(edu.notna(), edu.isin([1, 2]).astype(float), np.nan)
    df['pir'] = df.INDFMPIR
    df['bmi'] = df.BMXBMI
    s20, s40 = na(df.SMQ020), na(df.SMQ040)
    df['smoker'] = np.where(s20 == 2, 0.0, np.where(s20 == 1, np.where(s40.isin([1, 2]), 1.0, np.where(s40 == 3, 0.0, np.nan)), np.nan))
    df['diabetes'] = na(df.DIQ010).map({1: 1.0, 2: 0.0, 3: 0.0})
    df['hypertension'] = na(df.BPQ020).map({1: 1.0, 2: 0.0})
    cv = pd.concat([na(df[c]).map({1: 1.0, 2: 0.0}) for c in ['MCQ160B', 'MCQ160C', 'MCQ160E', 'MCQ160F']], axis=1)
    df['cvd'] = np.where(cv.eq(1).any(axis=1), 1.0, np.where(cv.notna().all(axis=1), 0.0, np.nan))
    pa = pd.concat([na(df[c]).map({1: 1.0, 2: 0.0}) for c in ['PAQ605', 'PAQ620', 'PAQ635', 'PAQ650', 'PAQ665']], axis=1)
    df['phys_active'] = np.where(pa.eq(1).any(axis=1), 1.0, np.where(pa.notna().all(axis=1), 0.0, np.nan))
    df['alcohol'] = na(df.ALQ101).map({1: 1.0, 2: 0.0})
    dp = pd.concat([na(zero(df[f'DPQ0{i}0'])) for i in range(1, 10)], axis=1)
    df['phq9'] = dp.sum(axis=1).where(dp.notna().all(axis=1))
    df['strata'], df['psu'] = df.SDMVSTRA, df.SDMVPSU

    COVS = ['age', 'female', 'race_black', 'race_hisp', 'race_other', 'edu_college', 'edu_lt_hs', 'pir', 'bmi',
            'smoker', 'diabetes', 'hypertension', 'cvd', 'phys_active', 'kcal', 'alcohol', 'phq9', 'caffeine_mg']
    keep = (['SEQN', 'cycle', 'y', 'cerad_delayed', 'animal_fluency', 'T', 'T_tot', 'T_nontea', 'tot_eadj',
             'nontea_eadj', 'iso_eadj'] + COVS + ['w', 'strata', 'psu', 'two_days', 'tea_any', 'tea_mg', 'nontea_mg'] +
            [c + '_mg' for c in cls])
    full = df[keep].reset_index(drop=True)
    tag = 'twoday' if days == (1, 2) else 'day1'
    full.to_csv(os.path.join(a.out, f'nhanes_studyB_{tag}_full.csv'), index=False)
    cc = full.dropna(subset=['y', 'T', 'T_nontea'] + COVS)
    cc.to_csv(os.path.join(a.out, f'nhanes_studyB_{tag}.csv'), index=False)
    say(f'\n== {name} exposure: eligible {len(full)}, complete cases {len(cc)}')
    for c in COVS:
        mm = full[c].isna().sum()
        if mm: say(f'  {c:<13} missing {mm:>4} ({mm / len(full):.0%})')
    g = cc.groupby('T')
    say('  ' + pd.DataFrame({'n': g.size(), 'tea_any': g.tea_any.mean(), 'caffeine_mg': g.caffeine_mg.mean(),
                             'DSST': g.y.mean(), 'edu_college': g.edu_college.mean()}).round(2).to_string().replace('\n', '\n  '))
    say(f'  crude DSST difference T=1 vs 0: {cc.y[cc["T"] == 1].mean() - cc.y[cc["T"] == 0].mean():.1f}; '
        f'tea share of flavonoid mg: {cc.tea_mg.sum() / cc.tot_mg.sum():.1%}; '
        f'zero isoflavones: {(cc.iso_mg == 0).mean():.0%}')
open(os.path.join(a.out, 'audit.txt'), 'w').write(log.getvalue())
