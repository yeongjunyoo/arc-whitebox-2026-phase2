#!/usr/bin/env bash
# Band study, second round b (research/wu/mech): where the opr30 band's gain comes from, on public minis 0-3: the truth
# on the kept entries with the dropped ones zeroed (mz) or true (dmm is in run_round2.sh), the dropped entries zeroed only
# at layers 2-10 (zlo) or only at 11-14 (zhi), only those dropped for their row (zrow) or for their column (zcol), and
# the whole truth with and without the dropped entries (ddmm, ddmz). Public training rows only.
#   cd ~/nova-arc && nohup bash research/r25a/mac_job25.sh mech_round2b 7 1 25 -- bash research/wu/mech/run_round2b.sh \
#       < /dev/null > research/wu/mech/logs/round2b.log 2>&1 &
cd ~/nova-arc || exit 1
M=research/wu/mech
P=.venv-fs0123/bin/python
mr() {  # tag spec [override]
  grep -q "\"tag\": \"$1\"" $M/runs/opr30_swap.jsonl 2>/dev/null && return 0
  $P -u $M/mech_run.py --est $M/est_opr30/mech_nott --ids 0,1,2,3 --tag "$1" --spec "$2" --override "${3:-}" \
    --out $M/runs/opr30_swap.jsonl 2>&1 | grep -v Warning | grep -v '^  '
}
HIAPA="11:drop=apa;12:drop=apa;13:drop=apa;14:drop=apa"
HIZERO="11:drop=zero;12:drop=zero;13:drop=zero;14:drop=zero"
mr mz diag=exact,keep=mc,drop=zero
mr zlo drop=zero "$HIAPA"
mr zhi "" "$HIZERO"
mr zrow drop=zrow
mr zcol drop=zcol
mr ddmm diag=mc,keep=mc,drop=mc,at=all
mr ddmz diag=mc,keep=mc,drop=zero,at=all
echo "done round2b"
