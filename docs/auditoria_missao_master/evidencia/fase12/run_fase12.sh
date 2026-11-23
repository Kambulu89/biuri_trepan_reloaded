#!/bin/bash
# $1 = A (sem gate) | B (com gate); $2 = semente
V=$1; S=$2
cd /home/user/kambulu89/biuri_trepan_reloaded && source /home/user/venv-biuri/bin/activate
D=data/ontologias_demo; OUT=/home/user/audit_runs/fase12/$V/seed$S; LOG=/home/user/audit_logs/fase12/${V}_seed$S.log
mkdir -p $(dirname $LOG)
FLAG=""; [ "$V" = B ] && FLAG="--semantic-attribution"
echo "commit=$(git rev-parse HEAD) dirty=$(git status --porcelain | grep -v '^??' | wc -l) inicio=$(date -u +%FT%TZ) flag=$FLAG" > $LOG
s0=$(date +%s)
python scripts/biuri_cli.py train-production --data $D/breast_cancer_wisconsin_diagnostic_dados.arff --target diagnosis \
  --owl $D/breast_cancer_wisconsin_diagnostic_enriquecida_biuri.owl --out $OUT --seed $S $FLAG >> $LOG 2>&1
echo "rc=$? segundos=$(( $(date +%s)-s0 )) fim=$(date -u +%FT%TZ)" >> $LOG
