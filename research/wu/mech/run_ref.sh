#!/usr/bin/env bash
# Band study (research/wu/mech): reference predictions of the r38 engine cuts on public mini 0 and 1, then the
# recording runs of the hooked copies. Public rows only.
#   cd ~/nova-arc && nohup bash research/r25a/mac_job25.sh mech_ref 3 1 30 -- bash research/wu/mech/run_ref.sh \
#       < /dev/null > research/wu/mech/logs/ref.log 2>&1 &
cd ~/nova-arc || exit 1
SNAP=$HOME/.cache/huggingface/hub/datasets--aicrowd--arc-whestbench-public-2026/snapshots/aa99830fdc09fad15407b10e8e3459d3e18bba0a
M=research/wu/mech
mkdir -p $M/runs $M/rec
for v in v_engine v_engine_nott v_engine_nos7; do
  /usr/bin/time -l .venv-fs0123/bin/python -u research/wu/ablate.py --estimator $M/est/$v --set mini --ids 0,1 \
    --snapshot "$SNAP" --out $M/runs/ref_$v.jsonl --threads 1 2>&1 | grep -v '^ *[0-9]* *\(block\|page\|messages\|signals\|voluntary\|involuntary\|instructions\|cycles\|swaps\)' 
done
for e in mech_nott mech_prod mech_nos7; do
  /usr/bin/time -l .venv-fs0123/bin/python -u $M/mech_run.py --est $M/est/$e --ids 0,1 --tag rec_$e \
    --out $M/runs/rec.jsonl --record $M/rec 2>&1 | grep -v '^ *[0-9]* *\(block\|page\|messages\|signals\|voluntary\|involuntary\|instructions\|cycles\|swaps\)'
done
echo done
