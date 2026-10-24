#!/usr/bin/env bash
# Band study, second round e (research/wu/mech): the band-on final means of r37 and of the r38 variants on public minis
# 0-3 (hooked band-on cuts of make_cuts.py; only the predictions are kept: --rec-layers 99), both final means of r38
# without the extraction's dA part, then mech_overlap.py: does the band move the final means the way the extraction
# does? Public training rows only.
#   cd ~/nova-arc && nohup bash research/r25a/mac_job25.sh mech_round2e 7 1 15 -- bash research/wu/mech/run_round2e.sh \
#       < /dev/null > research/wu/mech/logs/round2e.log 2>&1 &
cd ~/nova-arc || exit 1
M=research/wu/mech
P=.venv-fs0123/bin/python
mr() {  # est tag out record
  grep -q "\"tag\": \"$2\"" $3 2>/dev/null && return 0
  $P -u $M/mech_run.py --est $M/$1 --ids 0,1,2,3 --tag "$2" --out $3 --record $4 --rec-layers 99 2>&1 | grep -v Warning | grep -v '^  '
}
mr est_r37/mech_prod rec_mech_prod $M/runs/r37_rec.jsonl $M/pred_r37
mr est_r38x/mech_prod_noext rec_mech_prod $M/runs/r38x_rec_noext.jsonl $M/pred_r38noext
mr est_r38x/mech_prod_lawon rec_mech_prod $M/runs/r38x_rec_lawon.jsonl $M/pred_r38lawon
mr est_r38x/mech_nott_noxa rec_mech_nott $M/runs/r38x_rec_noxa.jsonl $M/pred_r38noxa
mr est_r38x/mech_prod_noxa rec_mech_prod $M/runs/r38x_rec_noxa.jsonl $M/pred_r38noxa
$P -u $M/mech_overlap.py
echo "done round2e"
