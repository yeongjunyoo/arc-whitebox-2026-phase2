#!/usr/bin/env bash
# Write-up runs after the score lane closed (coordinator 10-24): a queue of ablate.py reads, one network per job through
# the shared job wrapper (5 GB, 1 thread), one job at a time. The harbor contest has the Mac first: a job is tried only
# with no harbor lock, and with MJ_WAIT_S=1 so a job the wrapper's budget refuses leaves at once and its exit trap clears
# arc_waiting.flag (the flag stops harbor's new splits). The wrapper's gate also quits Google Drive (mb_no_drive), which
# relaunches at login and holds memory; a free-memory test here would wait on Drive forever (10-24 after the restart). Items are tag=estimator-folder@set, in order; set kp is
# the 48 untouched corpus networks (research/wu/kp_ids.txt, results research/wu/kp/<tag>.jsonl), set pub the public
# one-shot selection set (mini 60-99 and full 304-351, results research/wu/pub/<tag>_<mini|full>.jsonl).
#   cd ~/nova-arc && nohup bash research/wu/run_friendly.sh r38sub=estimators/r38/r38_sub_x100_s2@kp ... < /dev/null > /dev/null 2>&1 &
cd ~/nova-arc || exit 1
mkdir -p research/wu/kp research/wu/pub research/wu/logs
exec >> research/wu/logs/run_friendly.log 2>&1
say() { echo "$(date '+%m-%d %H:%M:%S') [wu-friendly] $*"; }
SNAP=$HOME/.cache/huggingface/hub/datasets--aicrowd--arc-whestbench-public-2026/snapshots/aa99830fdc09fad15407b10e8e3459d3e18bba0a
P=$HOME/nova-adsb/prcm
harbor_busy() { ls $P/harbor_*.lock > /dev/null 2>&1; }
one() {  # tag est set snapshot ids out log
  local tag=$1 est=$2 set_=$3 snap=$4 out=$6 log=$7 i rc
  for i in $5; do
    grep -q "\"mlp_id\": $i," "$out" 2>/dev/null && continue
    while :; do
      while harbor_busy; do sleep 120; done
      MJ_WAIT_S=1 bash research/r25a/mac_job25.sh "wufr_$tag" 5 1 10 -- env OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 .venv-fs0123/bin/python -u research/wu/ablate.py --estimator "$est" --set "$set_" --ids "$i" --snapshot "$snap" --out "$out" > "$log" 2>&1 < /dev/null
      rc=$?
      { [ $rc -eq 6 ] || [ $rc -eq 76 ] || [ $rc -eq 4 ]; } || break
      sleep 120
    done
    [ $rc -eq 0 ] || say "$tag $set_ $i rc=$rc: $(tail -1 "$log" | cut -c1-200)"
  done
}
for item in "$@"; do
  tag=${item%%=*}; rest=${item#*=}; est=${rest%@*}; where=${rest##*@}
  [ -e "$est" ] || { say "$tag: no $est"; continue; }
  say "$tag ($est) on $where start"
  case $where in
    kp) one "$tag" "$est" kp research/r2_data "$(cat research/wu/kp_ids.txt)" research/wu/kp/$tag.jsonl research/wu/kp/$tag.last.log
        say "$tag kp done: $(grep -c '"mlp_id"' research/wu/kp/$tag.jsonl 2>/dev/null) networks";;
    pub) one "$tag" "$est" mini "$SNAP" "$(seq -s ' ' 60 99)" research/wu/pub/${tag}_mini.jsonl research/wu/pub/$tag.last.log
         one "$tag" "$est" full "$SNAP" "$(seq -s ' ' 304 351)" research/wu/pub/${tag}_full.jsonl research/wu/pub/$tag.last.log
         say "$tag pub done: $(cat research/wu/pub/${tag}_*.jsonl 2>/dev/null | grep -c '"mlp_id"') networks";;
    *) say "$tag: unknown set $where";;
  esac
done
say end
