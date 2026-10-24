#!/usr/bin/env bash
# Write-up: which product family the restriction's MSE gain comes from (coordinator 10-21 10:2x; recorded in ARC MISSION G2
# before any number): v_engine_nopush, v_engine_nott and v_engine_nolast (the three together are v_engine_noshape),
# each on the 48 untouched corpus networks after a smoke network (mini 0). No public-set read.
#   cd ~/nova-arc && nohup bash research/wu/run_shape.sh < /dev/null > /dev/null 2>&1 &
cd ~/nova-arc || exit 1
mkdir -p research/wu/kp research/wu/pub research/wu/smoke research/wu/logs
exec >> research/wu/logs/run_shape.log 2>&1
say() { echo "$(date '+%m-%d %H:%M:%S') [wu-shape] $*"; }
SNAP=$HOME/.cache/huggingface/hub/datasets--aicrowd--arc-whestbench-public-2026/snapshots/aa99830fdc09fad15407b10e8e3459d3e18bba0a
TAGS="engine_nopush engine_nott engine_nolast"
one() {  # job set id snapshot out log
  while :; do
    MJ_WAIT_S=3600 bash research/r25a/mac_job25.sh "$1" 5 1 30 -- env OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 \
      MKL_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 .venv-fs0123/bin/python -u research/wu/ablate.py --estimator "$EST" \
      --set "$2" --ids "$3" --snapshot "$4" --out "$5" > "$6" 2>&1 < /dev/null
    rc=$?
    { [ $rc -eq 6 ] || [ $rc -eq 76 ] || [ $rc -eq 4 ]; } || return $rc
    sleep 60
  done
}
for TAG in $TAGS; do
  EST=estimators/wu2/v_$TAG
  say "$TAG smoke start"
  one wu2_$TAG mini 0 "$SNAP" research/wu/smoke/$TAG.jsonl research/wu/smoke/$TAG.log
  rc=$?
  line=$(grep '"mlp_id": 0,' research/wu/smoke/$TAG.jsonl 2>/dev/null | tail -1)
  say "$TAG smoke rc=$rc $(echo "$line" | grep -o '"final_mse": [0-9.e-]*, "flops": [0-9]*')"
  if [ $rc -ne 0 ] || ! echo "$line" | grep -q '"finite": true'; then say "$TAG smoke failed: skip"; continue; fi
  for i in $(cat research/wu/kp_ids.txt); do
    grep -q "\"mlp_id\": $i," research/wu/kp/$TAG.jsonl 2>/dev/null && continue
    while [ -e research/wu/kp.hold ]; do sleep 300; done  # a score-lane step holds the Mac
    one wukp_$TAG kp "$i" research/r2_data research/wu/kp/$TAG.jsonl research/wu/kp/$TAG.last.log
    rc=$?
    [ $rc -eq 0 ] || say "$TAG kp_$i rc=$rc: $(tail -1 research/wu/kp/$TAG.last.log | cut -c1-200)"
  done
  say "$TAG kp done: $(grep -c '"mlp_id"' research/wu/kp/$TAG.jsonl) networks"
done
say end
