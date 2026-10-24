#!/usr/bin/env bash
# Band study (research/wu/mech): is the band's gain specific to the dropped entries, or does any reduction of the
# off-diagonal third cumulant help the engine? The band-off engine with every off-diagonal 3 T^T entry scaled (tscale,
# layers 1-14), the kept entries scaled with the dropped ones as computed (kscale), the Monte Carlo truth scaled, and
# the Monte Carlo t3 on the diagonal (alone, and with the truth off the diagonal).
# Public mini 0 and 1 only.
#   cd ~/nova-arc && nohup bash research/r25a/mac_job25.sh mech_scale 7 1 15 -- bash research/wu/mech/run_scale.sh \
#       < /dev/null > research/wu/mech/logs/scale.log 2>&1 &
cd ~/nova-arc || exit 1
M=research/wu/mech
run() {  # tag spec
  grep -q "\"tag\": \"$1\"" $M/runs/scale.jsonl 2>/dev/null && return 0
  .venv-fs0123/bin/python -u $M/mech_run.py --est $M/est/mech_nott --ids 0,1 --tag "$1" --spec "$2" \
    --out $M/runs/scale.jsonl 2>&1 | grep -v Warning | grep -v '^  '
}
run s090 tscale=0.9,at=all
run s075 tscale=0.75,at=all
run s050 tscale=0.5,at=all
run s000 tscale=0,at=all
run s125 tscale=1.25,at=all
run k075 kscale=0.75
run mm075 diag=exact,keep=mc,drop=mc,tscale=0.75
run mm050 diag=exact,keep=mc,drop=mc,tscale=0.5
run dd diag=mc,at=all
run ddmm diag=mc,keep=mc,drop=mc,at=all
echo "done scale"
