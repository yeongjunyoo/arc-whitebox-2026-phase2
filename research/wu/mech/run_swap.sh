#!/usr/bin/env bash
# Band study (research/wu/mech): the oracle swap on public mini 0 and 1, one hooked band-off engine run per setting
# (mech_run.py --spec). The per-layer swaps are run_layers.sh. Public rows only.
#   cd ~/nova-arc && nohup bash research/r25a/mac_job25.sh mech_swap 7 1 90 -- bash research/wu/mech/run_swap.sh \
#       < /dev/null > research/wu/mech/logs/swap.log 2>&1 &
cd ~/nova-arc || exit 1
M=research/wu/mech
E=$M/est/mech_nott
run() {  # tag spec [override]
  grep -q "\"tag\": \"$1\"" $M/runs/swap.jsonl 2>/dev/null && return 0
  .venv-fs0123/bin/python -u $M/mech_run.py --est $E --ids 0,1 --tag "$1" --spec "$2" --override "${3:-}" \
    --out $M/runs/swap.jsonl 2>&1 | grep -v Warning | grep -v '^  ' 
}
if [ "${1:-all}" = all ]; then
  run z      drop=zero
  run m      drop=mc
  run mnf    drop=mcnf
  run enf    drop=engnf
  run d      diag=exact
  run dz     diag=exact,drop=zero
  run dkz    diag=exact,keep=exact,drop=zero
  run dkx    diag=exact,keep=exact,drop=exact
  run dkm    diag=exact,keep=exact,drop=mc
  run dkmnf  diag=exact,keep=exact,drop=mcnf
  run dkxnf  diag=exact,keep=exact,drop=xnf
  run dkmn   diag=exact,keep=exact,drop=mcnoise
  run dmm    diag=exact,keep=mc,drop=mc
  run xall   diag=exact,keep=exact,drop=exact,at=all
fi
echo "done ${1:-all}"
