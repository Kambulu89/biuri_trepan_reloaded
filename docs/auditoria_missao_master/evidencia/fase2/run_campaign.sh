#!/bin/bash
# uso: run_campaign.sh <modulo> <out> <workers> [args extra...]   — salta unidades já concluídas; limite de 5400 s por unidade
MOD=$1; OUT=$2; W=$3; shift 3
cd /home/user/kambulu89/biuri_trepan_reloaded && source /home/user/venv-biuri/bin/activate
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
mkdir -p $OUT/logs $OUT/status
for s in 42 7 123 2024 11; do for d in iris wine breast_cancer digits; do
  [ -f $OUT/raw/$d/seed$s/RUN_META.txt ] || echo "$d $s"
done; done | xargs -P $W -L 1 bash -c '
  d=$0; s=$1; t0=$(date +%s)
  timeout 5400 python -m '"$MOD"' --unit $d $s --out '"$OUT"' --reuse-checkpoint-from /home/user/audit_runs/baseline_main '"$*"' > '"$OUT"'/logs/${d}_seed${s}.log 2>&1
  rc=$?; echo "{\"dataset\":\"$d\",\"seed\":$s,\"returncode\":$rc,\"elapsed_s\":$(( $(date +%s)-t0 ))}" > '"$OUT"'/status/${d}_seed${s}.json
  echo "$d $s rc=$rc $(( $(date +%s)-t0 ))s"'
