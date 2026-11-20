#!/bin/bash
# $1 = versão (controlo_af02d5d | corrigido), $2 = semente
V=$1; S=$2
if [ "$V" = controlo_af02d5d ]; then cd /home/user/wt_af02d5d; else cd /home/user/kambulu89/biuri_trepan_reloaded; fi
source /home/user/venv-biuri/bin/activate
D=/home/user/kambulu89/biuri_trepan_reloaded/data/ontologias_demo
OUT=/home/user/audit_runs/prod_bc_recalc/$V/seed$S
LOG=/home/user/audit_logs/prod_bc_recalc/${V}_seed$S.log
echo "commit=$(git rev-parse HEAD) dirty=$(git status --porcelain | wc -l) inicio=$(date -u +%FT%TZ)" > $LOG
( time python scripts/biuri_cli.py train-production --data $D/breast_cancer_wisconsin_diagnostic_dados.arff --target diagnosis \
   --owl $D/breast_cancer_wisconsin_diagnostic_enriquecida_biuri.owl --out $OUT --seed $S ) >> $LOG 2>&1
echo "rc=$? fim=$(date -u +%FT%TZ)" >> $LOG
