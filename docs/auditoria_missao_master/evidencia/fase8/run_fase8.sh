#!/bin/bash
OUT=$1; W=$2
cd /home/user/kambulu89/biuri_trepan_reloaded && source /home/user/venv-biuri/bin/activate
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
mkdir -p $OUT/logs $OUT/status
for s in 42 7 123 2024 11; do for d in digits adult breast_cancer wine iris; do
  [ -f $OUT/raw/$d/seed$s/RUN_META.txt ] || echo "$d $s"
done; done | xargs -P $W -L 1 bash -c '
  d=$0; s=$1; t0=$(date +%s)
  timeout 7200 python -m validation.benchmark.run_fase8 --unit $d $s --out '"$OUT"' --reps 3 --reloaded O > '"$OUT"'/logs/${d}_seed${s}.log 2>&1
  rc=$?; echo "{\"dataset\":\"$d\",\"seed\":$s,\"returncode\":$rc,\"elapsed_s\":$(( $(date +%s)-t0 ))}" > '"$OUT"'/status/${d}_seed${s}.json
  echo "$d $s rc=$rc $(( $(date +%s)-t0 ))s"'
