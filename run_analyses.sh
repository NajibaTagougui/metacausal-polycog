#!/bin/bash
# Study A real-data analyses (NHANES 1999-2002, age >= 60).
# Run sequentially: the forests use every core, and parallel runs stall each other.
set -e
D=data_studyA
python -u analyze.py --data $D/nhanes_studyA.csv       --extra_covs caffeine_mg                                  --out analysis_studyA_cc_EQP        > logs_cc_EQP.txt 2>&1
python -u analyze.py --data $D/nhanes_studyA_noEQP.csv --extra_covs caffeine_mg --drop_covs EQP                  --out analysis_studyA_noEQP         > logs_noEQP.txt 2>&1
python -u analyze.py --data $D/nhanes_studyA_noEQP.csv --extra_covs caffeine_mg --drop_covs EQP --treat T_nontea --out analysis_studyA_noEQP_nontea  > logs_noEQP_nontea.txt 2>&1
python -u analyze.py --data $D/nhanes_studyA_noEQP.csv                          --drop_covs EQP                  --out analysis_studyA_noEQP_nocaff  > logs_noEQP_nocaff.txt 2>&1

# ---- sensitivity analyses
python -u analyze.py --data $D/nhanes_studyA.csv       --extra_covs caffeine_mg                 --no_weights --out sens_unweighted_cc_EQP  > logs_sens_unw_cc.txt 2>&1
python -u analyze.py --data $D/nhanes_studyA_noEQP.csv --extra_covs caffeine_mg --drop_covs EQP --no_weights --out sens_unweighted_noEQP   > logs_sens_unw_noEQP.txt 2>&1
python -u analyze.py --data data_studyA_eqp15/nhanes_studyA.csv --extra_covs caffeine_mg --out sens_eqp15 > logs_sens_eqp15.txt 2>&1
python -u analyze.py --data data_studyA_enl50/nhanes_studyA.csv --extra_covs caffeine_mg --out sens_enl50 > logs_sens_enl50.txt 2>&1
python -u mi_studyA.py --data $D/nhanes_studyA_full.csv --m 20 --out sens_mi > logs_sens_mi.txt 2>&1

# ---- Study B (replication, NHANES 2011-2014; no metabotypes)
B=data_studyB
python -u analyze.py --data $B/nhanes_studyB_twoday.csv --extra_covs caffeine_mg                       --out analysis_studyB_twoday        > logs_B_twoday.txt 2>&1
python -u analyze.py --data $B/nhanes_studyB_twoday.csv --extra_covs caffeine_mg --treat T_nontea      --out analysis_studyB_twoday_nontea > logs_B_nontea.txt 2>&1
python -u analyze.py --data $B/nhanes_studyB_day1.csv   --extra_covs caffeine_mg                       --out analysis_studyB_day1          > logs_B_day1.txt 2>&1
python -u analyze.py --data $B/nhanes_studyB_twoday.csv --extra_covs caffeine_mg --no_weights          --out sens_studyB_unweighted        > logs_B_unw.txt 2>&1
python -u analyze.py --data $B/nhanes_studyB_twoday.csv --extra_covs caffeine_mg --outcome animal_fluency --out analysis_studyB_fluency     > logs_B_flu.txt 2>&1
python -u analyze.py --data $B/nhanes_studyB_twoday.csv --extra_covs caffeine_mg --outcome cerad_delayed  --out analysis_studyB_cerad       > logs_B_cerad.txt 2>&1
