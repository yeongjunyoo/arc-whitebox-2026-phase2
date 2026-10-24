#!/usr/bin/env bash
# Band study, second round f (research/wu/mech): (1) band on / off (ablate.py) on public minis 0-7 for the r38 file with
# the whole extraction at half strength (make_cuts.py est_r38x v_engine_x05, v_engine_nott_x05): a consistent engine
# between r38 without the extraction and r38; (2) the clean band split by why an entry is dropped (zrow: its single
# index is out, zcol: its doubled index is out) on minis 0-3 for r38 and for r38 without the extraction's dA part, and
# the clean band itself for the latter (the opr30 split is in run_round2b.sh). Public training rows only.
#   cd ~/nova-arc && nohup bash research/r25a/mac_job25.sh mech_round2f 7 1 20 -- bash research/wu/mech/run_round2f.sh \
#       < /dev/null > research/wu/mech/logs/round2f.log 2>&1 &
cd ~/nova-arc || exit 1
SNAP=$HOME/.cache/huggingface/hub/datasets--aicrowd--arc-whestbench-public-2026/snapshots/aa99830fdc09fad15407b10e8e3459d3e18bba0a
M=research/wu/mech
P=.venv-fs0123/bin/python
ab() {  # estimator dir, ids, out
  [ -s "$3" ] && return 0
  $P -u research/wu/ablate.py --estimator "$1" --set mini --ids "$2" --snapshot "$SNAP" --out "$3" --threads 1 2>&1 \
    | grep '"mlp_id"' | cut -c1-120
}
mr() {  # est tag spec out
  grep -q "\"tag\": \"$2\"" $4 2>/dev/null && return 0
  $P -u $M/mech_run.py --est $M/$1 --ids 0,1,2,3 --tag "$2" --spec "$3" --out $4 2>&1 | grep -v Warning | grep -v '^  '
}
ab $M/est_r38x/v_engine_x05 0-7 $M/runs/r38x_v_engine_x05.jsonl
ab $M/est_r38x/v_engine_nott_x05 0-7 $M/runs/r38x_v_engine_nott_x05.jsonl
mr est/mech_nott zrow drop=zrow $M/runs/r38_zsplit.jsonl
mr est/mech_nott zcol drop=zcol $M/runs/r38_zsplit.jsonl
mr est_r38x/mech_nott_noxa z drop=zero $M/runs/r38x_rec_noxa.jsonl
mr est_r38x/mech_nott_noxa zrow drop=zrow $M/runs/r38x_rec_noxa.jsonl
mr est_r38x/mech_nott_noxa zcol drop=zcol $M/runs/r38x_rec_noxa.jsonl
echo "done round2f"
