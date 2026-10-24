#!/usr/bin/env bash
# Write-up ladder (coordinator 10-21; plan in ARC MISSION G2 of 10-20 and research/wu/ladder.py, unchanged): the seven
# variants at seeds 2, 3 and 4 with 30 epochs, through the shared job wrapper (3 GB and 2 threads each, two at a time),
# a job stopped for a harbor window (rc 76) or tired of waiting for room (rc 6) retried, then one eval of all of them on
# the 90 validation networks. A model already on disk is not trained again.
#   cd ~/nova-arc && nohup bash research/wu/run_ladder.sh < /dev/null > /dev/null 2>&1 &
cd ~/nova-arc || exit 1
mkdir -p research/wu/logs
exec >> research/wu/logs/ladder.log 2>&1
say() { echo "$(date '+%m-%d %H:%M:%S') $*"; }
POOLD=research/r25a/gru/qx3a15

one() {  # <variant> <seed>
  local v=$1 s=$2 rc try
  for try in 1 2 3 4 5 6; do
    [ -s "$POOLD/lc3_wmsg_lad_${v}_s${s}.npz" ] && { say "have $v s$s"; return 0; }
    while [ -e research/wu/kp.hold ]; do sleep 300; done  # a score-lane step holds the Mac
    MJ_WAIT_S=14400 bash research/r25a/mac_job25.sh "lad_${v}_s${s}" 3 2 15 -- .venv-lc3/bin/python -u \
      research/wu/ladder.py train --variant "$v" --seed "$s" --epochs 30 >> "research/wu/logs/lad_${v}_s${s}.log" 2>&1
    rc=$?
    say "train $v s$s try $try rc=$rc"
    [ "$rc" = 0 ] && return 0
    [ "$rc" = 76 ] || [ "$rc" = 6 ] || return "$rc"
    sleep 300
  done
  return 1
}

say "start"
i=0
for s in 2 3 4; do
  for v in ref nochain nochain_nomsg w1 notaps nomsg nocarry; do
    one "$v" "$s" &
    i=$((i + 1))
    [ $((i % 2)) = 0 ] && wait
  done
done
wait
MJ_WAIT_S=14400 bash research/r25a/mac_job25.sh lad_eval 3 2 30 -- .venv-lc3/bin/python -u research/wu/ladder.py eval
say "eval rc=$?"
say "end"
