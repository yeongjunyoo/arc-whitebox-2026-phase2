#!/usr/bin/env bash
# Band study, second round d (research/wu/mech): which of the r38 extraction's four parts takes the band's gain. Band
# on / off (ablate.py) on public minis 0-7 for the r38 file with one part off at a time (make_cuts.py est_r38x
# v_engine_<mode>, v_engine_nott_<mode>; noxc: C', noxa: dA, noxw: w1, noxf: the feed pair). Public training rows only.
#   cd ~/nova-arc && nohup bash research/r25a/mac_job25.sh mech_round2d 7 1 35 -- bash research/wu/mech/run_round2d.sh \
#       < /dev/null > research/wu/mech/logs/round2d.log 2>&1 &
cd ~/nova-arc || exit 1
SNAP=$HOME/.cache/huggingface/hub/datasets--aicrowd--arc-whestbench-public-2026/snapshots/aa99830fdc09fad15407b10e8e3459d3e18bba0a
M=research/wu/mech
P=.venv-fs0123/bin/python
ab() {  # estimator dir, ids, out
  [ -s "$3" ] && return 0
  $P -u research/wu/ablate.py --estimator "$1" --set mini --ids "$2" --snapshot "$SNAP" --out "$3" --threads 1 2>&1 \
    | grep '"mlp_id"' | cut -c1-120
}
for mode in noxc noxa noxw noxf; do
  ab $M/est_r38x/v_engine_$mode 0-7 $M/runs/r38x_v_engine_$mode.jsonl
  ab $M/est_r38x/v_engine_nott_$mode 0-7 $M/runs/r38x_v_engine_nott_$mode.jsonl
done
echo "done round2d"
