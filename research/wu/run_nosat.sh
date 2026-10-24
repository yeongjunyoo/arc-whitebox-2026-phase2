#!/usr/bin/env bash
# Write-up experiment (coordinator 10-20; recorded in ARC MISSION G2 before any number): the engine alone with every
# restricted product at full width (estimators/wu2/v_engine_nosat), so the restriction gets its own row next to
# `engine`. One smoke network (mini 0, a training-pool network) after the ablation smoke ends; if it runs and is
# finite, the 48 untouched corpus networks (same pause rule as research/wu/run_kp.sh), then the one-shot selection set
# after the nomination window closes (2026-10-23 09:05 KST), the same way as research/wu/run_pub.sh. One network per job
# through the shared job wrapper (5 GB, 1 thread). Its own log, so a failure here cannot stop run_kp.sh.
#   cd ~/nova-arc && nohup bash research/wu/run_nosat.sh < /dev/null > /dev/null 2>&1 &
cd ~/nova-arc || exit 1
mkdir -p research/wu/kp research/wu/pub research/wu/smoke research/wu/logs
exec >> research/wu/logs/run_nosat.log 2>&1
say() { echo "$(date '+%m-%d %H:%M:%S') [wu-nosat] $*"; }
SNAP=$HOME/.cache/huggingface/hub/datasets--aicrowd--arc-whestbench-public-2026/snapshots/aa99830fdc09fad15407b10e8e3459d3e18bba0a
EST=estimators/wu2/v_engine_nosat
TAG=engine_nosat
R33LOG=research/r33/logs/r33.log
RESUME=$(date -j -f "%Y-%m-%d %H:%M" "2026-10-22 02:30" +%s)
START=$(date -j -f "%Y-%m-%d %H:%M" "2026-10-23 09:05" +%s)
r33_busy() {  # R33's own pick line only (its 11:17 start line also names "V90 PICK"); research/wu/kp.release ends the pause
  [ -e research/wu/kp.release ] && return 1
  grep -q "\[r33\] STOP\|\[r33\] r33 STOP\|pre-registered rule: FAIL" "$R33LOG" && return 1
  grep -q "\[r33\] V90 PICK kdk05s3" "$R33LOG" && [ "$(date +%s)" -lt "$RESUME" ] && return 0
  return 1
}
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
until grep -q "\[wu2\] end" research/wu/smoke/smoke.log 2>/dev/null; do sleep 120; done
say "smoke start"
one wu2_$TAG mini 0 "$SNAP" research/wu/smoke/$TAG.jsonl research/wu/smoke/$TAG.log
rc=$?
line=$(grep '"mlp_id": 0,' research/wu/smoke/$TAG.jsonl 2>/dev/null | tail -1)
say "smoke rc=$rc $(echo "$line" | grep -o '"final_mse": [0-9.e-]*, "flops": [0-9]*')"
if [ $rc -ne 0 ] || ! echo "$line" | grep -q '"finite": true'; then say "smoke failed: stop"; exit 1; fi
say "kp start"
for i in $(cat research/wu/kp_ids.txt); do
  grep -q "\"mlp_id\": $i," research/wu/kp/$TAG.jsonl 2>/dev/null && continue
  while r33_busy; do sleep 300; done
  one wukp_$TAG kp "$i" research/r2_data research/wu/kp/$TAG.jsonl research/wu/kp/$TAG.last.log
  rc=$?
  [ $rc -eq 0 ] || say "kp_$i rc=$rc: $(tail -1 research/wu/kp/$TAG.last.log | cut -c1-200)"
done
say "kp done: $(grep -c '"mlp_id"' research/wu/kp/$TAG.jsonl) networks"
while [ "$(date +%s)" -lt "$START" ]; do sleep 300; done
say "pub start"
for part in "mini:$(seq -s ' ' 60 99)" "full:$(seq -s ' ' 304 351)"; do
  set_=${part%%:*}; ids=${part#*:}; out=research/wu/pub/${TAG}_${set_}.jsonl
  for i in $ids; do
    grep -q "\"mlp_id\": $i," "$out" 2>/dev/null && continue
    one wupub_$TAG "$set_" "$i" "$SNAP" "$out" research/wu/pub/$TAG.last.log
    rc=$?
    [ $rc -eq 0 ] || say "pub $set_ $i rc=$rc: $(tail -1 research/wu/pub/$TAG.last.log | cut -c1-200)"
  done
done
say "pub done: $(cat research/wu/pub/${TAG}_*.jsonl 2>/dev/null | grep -c '"mlp_id"') networks"
say end
