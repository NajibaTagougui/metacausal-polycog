#!/bin/bash
# Study A real-data analyses (NHANES 1999-2002, age >= 60)
set -e
D=data_studyA
python analyze.py --data $D/nhanes_studyA.csv       --extra_covs caffeine_mg                --out analysis_studyA_cc_EQP      > logs_cc_EQP.txt 2>&1 &
python analyze.py --data $D/nhanes_studyA_noEQP.csv --extra_covs caffeine_mg --drop_covs EQP --out analysis_studyA_noEQP     > logs_noEQP.txt 2>&1 &
python analyze.py --data $D/nhanes_studyA_noEQP.csv --extra_covs caffeine_mg --drop_covs EQP --treat T_nontea --out analysis_studyA_noEQP_nontea > logs_noEQP_nontea.txt 2>&1 &
python analyze.py --data $D/nhanes_studyA_noEQP.csv                          --drop_covs EQP --out analysis_studyA_noEQP_nocaff > logs_noEQP_nocaff.txt 2>&1 &
wait
