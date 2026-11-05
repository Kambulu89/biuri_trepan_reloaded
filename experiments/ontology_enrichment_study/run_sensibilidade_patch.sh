#!/usr/bin/env bash
export BIURI_ROOT=$SP/repo_patched BIURI_WAIT_S=14400
OUT=$SP/syn_runs_patched; mkdir -p $OUT
for arm in correct perm rand; do
  [ -f $OUT/res_s1_seed1_$arm.json ] && continue
  $SP/venv311/bin/python -I /home/user/biuri_trepan_reloaded/experiments/ontology_enrichment_study/run_biuri_headless.py $SP/syn/s1_seed1.arff $SP/syn/s1_seed1_$arm.owl $OUT/res_s1_seed1_$arm.json > $OUT/log_s1_seed1_$arm.txt 2>&1
done
