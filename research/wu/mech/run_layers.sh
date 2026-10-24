#!/usr/bin/env bash
# Band study (research/wu/mech): per-layer swaps on public mini 0 and 1. Base everywhere: the clean band (diag=exact,
# keep=exact, drop=zero at every band layer). At one layer L the dropped entries take the engine's exact values (x), the
# Monte Carlo truth (m), or the whole off-diagonal takes the truth (t). Each run keeps layer L's final C' and records
# layers L and L + 1, so the local effect (C' of layer L, T and t3 of layer L + 1) is read with the same upstream state.
#   cd ~/nova-arc && nohup bash research/r25a/mac_job25.sh mech_layers 7 1 60 -- bash research/wu/mech/run_layers.sh \
#       < /dev/null > research/wu/mech/logs/layers.log 2>&1 &
cd ~/nova-arc || exit 1
M=research/wu/mech
E=$M/est/mech_nott
BASE=diag=exact,keep=exact,drop=zero
run() {  # tag override rec_layers cov_layers
  grep -q "\"tag\": \"$1\"" $M/runs/layers.jsonl 2>/dev/null && return 0
  .venv-fs0123/bin/python -u $M/mech_run.py --est $E --ids 0,1 --tag "$1" --spec "$BASE" --override "$2" \
    --record $M/rec/loc --rec-layers "$3" --cov-layers "$4" --out $M/runs/layers.jsonl 2>&1 | grep -v Warning | grep -v '^  '
}
run Lbase "" "2,3,4,5,6,7,8,9,10,11,12,13,14" "2,3,4,5,6,7,8,9,10,11,12,13,14"
for L in 2 3 4 5 6 7 8 9 10 11 12 13 14; do
  N=$((L + 1))
  run "L${L}x" "$L:drop=exact" "$L,$N" "$L"
  run "L${L}m" "$L:drop=mc" "$L,$N" "$L"
  run "L${L}t" "$L:keep=mc,drop=mc" "$L,$N" "$L"
done
echo "done layers"
