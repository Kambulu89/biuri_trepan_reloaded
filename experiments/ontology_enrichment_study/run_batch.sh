#!/usr/bin/env bash
# Uso: run_batch.sh <python> <syn_dir> <out_dir> <ds> <seed>
# Um worker = um (dataset, seed); braços em sequência (none primeiro: aquece a cache do MLP, partilhada pelos braços).
PY=$1; SYN=$2; OUT=$3; ds=$4; seed=$5
HERE=$(cd "$(dirname "$0")" && pwd); mkdir -p "$OUT"
for arm in none correct perm rand; do
  tag=${ds}_seed${seed}_${arm}; [ -f "$OUT/res_$tag.json" ] && continue
  owl="$SYN/${ds}_seed${seed}_${arm}.owl"; [ $arm = none ] && owl="-"
  "$PY" -I "$HERE/run_biuri_headless.py" "$SYN/${ds}_seed${seed}.arff" "$owl" "$OUT/res_$tag.json" > "$OUT/log_$tag.txt" 2>&1
done
echo DONE > "$OUT/DONE_${ds}_${seed}"
