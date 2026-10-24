#!/usr/bin/env bash
# Band study (research/wu/mech): the oracle swap on public mini 2 and 3 (Monte Carlo truth from mc_truth.py): the
# recording runs (band off, band on) and the swap settings of run_swap.sh's main set. Public training rows only.
#   cd ~/nova-arc && nohup bash research/r25a/mac_job25.sh mech_more 7 1 20 -- bash research/wu/mech/run_more.sh \
#       < /dev/null > research/wu/mech/logs/more.log 2>&1 &
cd ~/nova-arc || exit 1
M=research/wu/mech
P=.venv-fs0123/bin/python
mr() {  # est tag spec ids out [extra...]
  local e=$1 tag=$2 spec=$3 ids=$4 out=$5; shift 5
  grep -q "\"tag\": \"$tag\"" $out 2>/dev/null && return 0
  $P -u $M/mech_run.py --est $M/est/$e --ids "$ids" --tag "$tag" --spec "$spec" --out $out "$@" 2>&1 | grep -v Warning | grep -v '^  '
}
mr mech_nott rec_mech_nott "" 2,3 $M/runs/rec23.jsonl --record $M/rec
mr mech_prod rec_mech_prod "" 2,3 $M/runs/rec23.jsonl --record $M/rec
for i in $(seq 1 180); do [ -f $M/mc/mc_mini2.json ] && [ -f $M/mc/mc_mini3.json ] && break; sleep 20; done
for s in "z drop=zero" "dkz diag=exact,keep=exact,drop=zero" "dkx diag=exact,keep=exact,drop=exact" "m drop=mc" \
         "dkm diag=exact,keep=exact,drop=mc" "dkmnf diag=exact,keep=exact,drop=mcnf" "dmm diag=exact,keep=mc,drop=mc"; do
  set -- $s
  mr mech_nott "$1" "$2" 2,3 $M/runs/swap23.jsonl
done
echo "done more"
