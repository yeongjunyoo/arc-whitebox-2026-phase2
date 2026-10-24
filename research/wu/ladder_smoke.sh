#!/usr/bin/env bash
# Ladder smoke (coordinator 10-20): one epoch of the ref and nochain variants with a throwaway seed (99) and their V90
# read, after the ablation smoke ends, through the shared job wrapper (3 GB, 2 threads). The real ladder runs after
# 10-22 02:30.
#   cd ~/nova-arc && nohup bash research/wu/ladder_smoke.sh < /dev/null > /dev/null 2>&1 &
cd ~/nova-arc || exit 1
exec >> research/wu/logs/ladder_smoke.log 2>&1
until grep -q "\[wu2\] end" research/wu/smoke/smoke.log 2>/dev/null; do sleep 120; done
echo "$(date '+%m-%d %H:%M:%S') start"
for v in ref nochain; do
  MJ_WAIT_S=3600 bash research/r25a/mac_job25.sh lad_smoke 3 2 40 -- .venv-lc3/bin/python -u research/wu/ladder.py train \
    --variant "$v" --seed 99 --epochs 1
  echo "$(date '+%m-%d %H:%M:%S') train $v rc=$?"
done
MJ_WAIT_S=3600 bash research/r25a/mac_job25.sh lad_smoke_eval 3 2 20 -- .venv-lc3/bin/python -u research/wu/ladder.py eval \
  --variants ref,nochain --seeds 99
echo "$(date '+%m-%d %H:%M:%S') eval rc=$?"
