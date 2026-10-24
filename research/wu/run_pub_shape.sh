#!/usr/bin/env bash
# Write-up section 7.2: the three restriction families of section 5 (research/wu/run_shape.sh ran them on the corpus
# only) on the public one-shot selection set, public mini 60-99 and full 304-351, read after the nomination window
# closed (10-23 08:59 KST). The tags one after another, one network per job through the shared job wrapper (5 GB, 1
# thread). The harbor contest has the Mac first: a job is tried only with no harbor lock and memory free >= 40%, and
# with MJ_WAIT_S=1 so a refused job leaves at once and its exit trap clears arc_waiting.flag (the flag stops harbor's
# new splits; the ARC score lane that needed that priority closed 10-22).
#   cd ~/nova-arc && nohup bash research/wu/run_pub_shape.sh engine_nopush engine_nott engine_nolast < /dev/null > /dev/null 2>&1 &
cd ~/nova-arc || exit 1
mkdir -p research/wu/pub research/wu/logs
exec >> research/wu/logs/run_pub_shape.log 2>&1
say() { echo "$(date '+%m-%d %H:%M:%S') [wu-pubshape] $*"; }
SNAP=$HOME/.cache/huggingface/hub/datasets--aicrowd--arc-whestbench-public-2026/snapshots/aa99830fdc09fad15407b10e8e3459d3e18bba0a
P=$HOME/nova-adsb/prcm
free_pct() { memory_pressure 2>/dev/null | awk -F': ' '/free percentage/ {gsub(/%/, "", $2); print $2}'; }
harbor_busy() { ls $P/harbor_*.lock > /dev/null 2>&1 || [ "$(free_pct)" -lt 40 ]; }
for TAG in "$@"; do
  EST=estimators/wu2/v_$TAG
  [ -f "$EST/estimator.py" ] || { say "$TAG: no $EST"; continue; }
  say "$TAG start"
  for part in "mini:$(seq -s ' ' 60 99)" "full:$(seq -s ' ' 304 351)"; do
    set_=${part%%:*}; ids=${part#*:}; out=research/wu/pub/${TAG}_${set_}.jsonl
    for i in $ids; do
      grep -q "\"mlp_id\": $i," "$out" 2>/dev/null && continue
      while :; do
        while harbor_busy; do sleep 120; done
        MJ_WAIT_S=1 bash research/r25a/mac_job25.sh "wupub_$TAG" 5 1 10 -- env OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1           MKL_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 .venv-fs0123/bin/python -u research/wu/ablate.py --estimator "$EST"           --set "$set_" --ids "$i" --snapshot "$SNAP" --out "$out" > "research/wu/pub/$TAG.last.log" 2>&1 < /dev/null
        rc=$?
        { [ $rc -eq 6 ] || [ $rc -eq 76 ] || [ $rc -eq 4 ]; } || break
        sleep 120
      done
      [ $rc -eq 0 ] || say "$TAG $set_ $i rc=$rc: $(tail -1 research/wu/pub/$TAG.last.log | cut -c1-200)"
    done
  done
  say "$TAG done: $(cat research/wu/pub/${TAG}_*.jsonl 2>/dev/null | grep -c '"mlp_id"') networks"
done
say end
