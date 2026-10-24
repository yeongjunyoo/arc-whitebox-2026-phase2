#!/usr/bin/env bash
# Band study, second round (research/wu/mech, coordinator 10-22): (1) the opr30 file (submission 336955, the cuts behind
# kp_table.json) on public minis 0-3 with the Monte Carlo truth of mc_truth.py: recording runs (band off, band on) and
# the oracle settings z, m, dmm, s090, s125, dd; (2) band on / off on public minis 0-15 of the r37 file and of the r38
# file with R38's two changes toggled (make_cuts.py). Public training rows only.
#   cd ~/nova-arc && nohup bash research/r25a/mac_job25.sh mech_round2 7 1 90 -- bash research/wu/mech/run_round2.sh \
#       < /dev/null > research/wu/mech/logs/round2.log 2>&1 &
cd ~/nova-arc || exit 1
SNAP=$HOME/.cache/huggingface/hub/datasets--aicrowd--arc-whestbench-public-2026/snapshots/aa99830fdc09fad15407b10e8e3459d3e18bba0a
M=research/wu/mech
P=.venv-fs0123/bin/python
mr() {  # est tag spec ids out [extra...]
  local e=$1 tag=$2 spec=$3 ids=$4 out=$5; shift 5
  grep -q "\"tag\": \"$tag\"" $out 2>/dev/null && return 0
  $P -u $M/mech_run.py --est $M/$e --ids "$ids" --tag "$tag" --spec "$spec" --out $out "$@" 2>&1 | grep -v Warning | grep -v '^  '
}
ab() {  # estimator dir, ids, out
  $P -u research/wu/ablate.py --estimator "$1" --set mini --ids "$2" --snapshot "$SNAP" --out "$3" --threads 1 2>&1 \
    | grep '"mlp_id"' | cut -c1-120
}
mr est_opr30/mech_nott rec_mech_nott "" 0,1,2,3 $M/runs/opr30_rec.jsonl --record $M/rec_opr30
mr est_opr30/mech_prod rec_mech_prod "" 0,1,2,3 $M/runs/opr30_rec.jsonl --record $M/rec_opr30
for s in "z drop=zero" "m drop=mc" "dmm diag=exact,keep=mc,drop=mc" "s090 tscale=0.9,at=all" "s125 tscale=1.25,at=all" \
         "dd diag=mc,at=all"; do
  set -- $s
  mr est_opr30/mech_nott "$1" "$2" 0,1,2,3 $M/runs/opr30_swap.jsonl
done
ab $M/est_r37/v_engine 0-15 $M/runs/r37_v_engine.jsonl
ab $M/est_r37/v_engine_nott 0-15 $M/runs/r37_v_engine_nott.jsonl
ab $M/est_r38x/v_engine_lawon_noext 0 $M/runs/r38x_v_engine_lawon_noext.jsonl
ab $M/est_r38x/v_engine_nott_lawon_noext 0 $M/runs/r38x_v_engine_nott_lawon_noext.jsonl
for mode in noext lawon; do
  ab $M/est_r38x/v_engine_$mode 0-15 $M/runs/r38x_v_engine_$mode.jsonl
  ab $M/est_r38x/v_engine_nott_$mode 0-15 $M/runs/r38x_v_engine_nott_$mode.jsonl
done
echo "done round2"
