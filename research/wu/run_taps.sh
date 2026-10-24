#!/usr/bin/env bash
# Write-up factorial of the folds and the corrector's taps (coordinator 10-21 02:1x, after review round 4; recorded in ARC
# MISSION G2 before any number): v_notaps (folds on, taps off), v_tapsnofold (taps built, folds left out) and
# v_engine_notaps (the taps' own cost), next to the submitted file and v_nostar (folds and taps off). Same steps as
# research/wu/run_attr.sh for each tag: a smoke network (mini 0), the 48 untouched corpus networks, then the one-shot
# selection set after 2026-10-23 09:05 KST.
#   cd ~/nova-arc && nohup bash research/wu/run_taps.sh < /dev/null > /dev/null 2>&1 &
cd ~/nova-arc || exit 1
mkdir -p research/wu/kp research/wu/pub research/wu/smoke research/wu/logs
exec >> research/wu/logs/run_taps.log 2>&1
say() { echo "$(date '+%m-%d %H:%M:%S') [wu-taps] $*"; }
SNAP=$HOME/.cache/huggingface/hub/datasets--aicrowd--arc-whestbench-public-2026/snapshots/aa99830fdc09fad15407b10e8e3459d3e18bba0a
START=$(date -j -f "%Y-%m-%d %H:%M" "2026-10-23 09:05" +%s)
TAGS="notaps tapsnofold engine_notaps"
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
while [ "$(date +%s)" -lt "$START" ]; do sleep 300; done
for TAG in $TAGS; do
  EST=estimators/wu2/v_$TAG
  [ -e research/wu/kp/$TAG.jsonl ] || continue
  say "$TAG pub start"
  for part in "mini:$(seq -s ' ' 60 99)" "full:$(seq -s ' ' 304 351)"; do
    set_=${part%%:*}; ids=${part#*:}; out=research/wu/pub/${TAG}_${set_}.jsonl
    for i in $ids; do
      grep -q "\"mlp_id\": $i," "$out" 2>/dev/null && continue
      one wupub_$TAG "$set_" "$i" "$SNAP" "$out" research/wu/pub/$TAG.last.log
      rc=$?
      [ $rc -eq 0 ] || say "$TAG pub $set_ $i rc=$rc: $(tail -1 research/wu/pub/$TAG.last.log | cut -c1-200)"
    done
  done
  say "$TAG pub done: $(cat research/wu/pub/${TAG}_*.jsonl 2>/dev/null | grep -c '"mlp_id"') networks"
done
say end
