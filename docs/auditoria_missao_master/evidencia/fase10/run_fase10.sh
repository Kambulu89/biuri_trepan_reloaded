#!/bin/bash
OUT=$1; W=$2
cd /home/user/kambulu89/biuri_trepan_reloaded && source /home/user/venv-biuri/bin/activate
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
mkdir -p $OUT/logs $OUT/status
for s in 42 7 123 2024 11; do for d in digits adult german_credit iris wine breast_cancer phys_power phys_bmi phys_kinetic; do
  [ -f $OUT/raw/$d/seed$s/RUN_META.txt ] || echo "$d $s"
done; done | xargs -P $W -L 1 bash -c '
  d=$0; s=$1; t0=$(date +%s)
  case $d in iris|wine|breast_cancer) SRC=/home/user/audit_runs/fase8;; *) SRC=/home/user/audit_runs/fase9;; esac
  timeout 7200 python -m validation.benchmark.run_fase10 --unit $d $s --out '"$OUT"' --reps 3 --reuse-checkpoint-from $SRC > '"$OUT"'/logs/${d}_seed${s}.log 2>&1
  rc=$?; echo "{\"dataset\":\"$d\",\"seed\":$s,\"returncode\":$rc,\"elapsed_s\":$(( $(date +%s)-t0 ))}" > '"$OUT"'/status/${d}_seed${s}.json
  echo "$d $s rc=$rc $(( $(date +%s)-t0 ))s"'
