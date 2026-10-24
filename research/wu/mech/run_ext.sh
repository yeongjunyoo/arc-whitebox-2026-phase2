#!/usr/bin/env bash
# Band study (research/wu/mech): does the band help the engine on public minis, and is it the same for the submitted r38
# file (337488, research/wu/mech/est cuts) and the opr30 file (336955, estimators/wu2 cuts, where kp_table.json measured
# v_engine_nott x1.101 MSE)? Band on (v_engine) and band off (v_engine_nott) of both on public minis 0-15, then the
# clean band (mech_nott dkz) of r38 on minis 2-15. Public training rows only (ablate.py meters as the grader does).
#   cd ~/nova-arc && nohup bash research/r25a/mac_job25.sh mech_ext 7 1 45 -- bash research/wu/mech/run_ext.sh \
#       < /dev/null > research/wu/mech/logs/ext.log 2>&1 &
cd ~/nova-arc || exit 1
SNAP=$HOME/.cache/huggingface/hub/datasets--aicrowd--arc-whestbench-public-2026/snapshots/aa99830fdc09fad15407b10e8e3459d3e18bba0a
M=research/wu/mech
P=.venv-fs0123/bin/python
ab() {  # estimator dir, ids, out
  $P -u research/wu/ablate.py --estimator "$1" --set mini --ids "$2" --snapshot "$SNAP" --out "$3" --threads 1 2>&1 \
    | grep '"mlp_id"' | cut -c1-120
}
ab $M/est/v_engine 2-15 $M/runs/ext_r38_v_engine.jsonl
ab $M/est/v_engine_nott 2-15 $M/runs/ext_r38_v_engine_nott.jsonl
ab estimators/wu2/v_engine 0-15 $M/runs/ext_opr30_v_engine.jsonl
ab estimators/wu2/v_engine_nott 0-15 $M/runs/ext_opr30_v_engine_nott.jsonl
grep -q '"tag": "dkz"' $M/runs/ext_dkz.jsonl 2>/dev/null || $P -u $M/mech_run.py --est $M/est/mech_nott \
  --ids 2,3,4,5,6,7,8,9,10,11,12,13,14,15 --tag dkz --spec diag=exact,keep=exact,drop=zero --out $M/runs/ext_dkz.jsonl 2>&1 \
  | grep -v Warning | grep -v '^  '
echo "done ext"
