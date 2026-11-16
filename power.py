"""
Power for the CONFIRMATORY test: does the polyphenol effect differ between
equol producers and non-producers? (AIPW contrast, two-sided alpha=0.05)
plus the omnibus BLP heterogeneity test is left to the full pipeline.
"""
import sys, warnings, numpy as np, pandas as pd
warnings.filterwarnings("ignore")
from simulate import simulate, COVARIATES
import pipeline as P
rows=[]
reps=int(sys.argv[1]) if len(sys.argv)>1 else 50
for scale in [0.5, 1.0]:
    for n in [600, 900, 1500, 2600, 5000]:
        hits=[]; hits_any=[]
        for s in range(reps):
            d=simulate(n, seed=7000+s, effect_scale=scale)
            X,T,y=d[COVARIATES].values,d['T'].values,d['y'].values
            nu=P.crossfit_nuisances(X,T,y,k=3,seed=s)
            m=d.EQP.values==1
            a1,s1=P.aipw_ate(nu['psi'][m]); a0,s0=P.aipw_ate(nu['psi'][~m])
            z=(a1-a0)/np.sqrt(s1**2+s0**2); hits.append(abs(z)>1.96)
            # joint metabotype: both-producers vs non-producers
            mb=(d.EQP.values==1)&(d.ENLH.values==1); mn=(d.EQP.values==0)&(d.ENLH.values==0)
            b1,t1=P.aipw_ate(nu['psi'][mb]); b0,t0=P.aipw_ate(nu['psi'][mn])
            hits_any.append(abs((b1-b0)/np.sqrt(t1**2+t0**2))>1.96)
        rows.append(dict(effect_scale=scale,n=n,power_EQP=np.mean(hits),power_both_vs_none=np.mean(hits_any)))
        print(rows[-1],flush=True)
pd.DataFrame(rows).to_csv("results/power.csv",index=False)
