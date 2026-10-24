#!/usr/bin/env bash
# Band study, second round c (research/wu/mech): kept-vs-dropped entry statistics for the r37 file and for the r38 file
# with one of R38's two changes undone (make_cuts.py: est_r37/mech_nott, est_r38x/mech_nott_noext, mech_nott_lawon,
# band-off cuts with the hooks), on public minis 0-3 against the Monte Carlo truth of mc_truth.py; the band-off
# recordings must equal the unhooked cuts (runs/r37_v_engine_nott.jsonl, r38x_v_engine_nott_<mode>.jsonl); then the
# clean band (z) and dropped = truth (m) on each. Public training rows only.
#   cd ~/nova-arc && nohup bash research/r25a/mac_job25.sh mech_round2c 7 1 30 -- bash research/wu/mech/run_round2c.sh \
#       < /dev/null > research/wu/mech/logs/round2c.log 2>&1 &
cd ~/nova-arc || exit 1
M=research/wu/mech
P=.venv-fs0123/bin/python
mr() {  # est tag spec out [extra...]
  local e=$1 tag=$2 spec=$3 out=$4; shift 4
  grep -q "\"tag\": \"$tag\"" $out 2>/dev/null && return 0
  $P -u $M/mech_run.py --est $M/$e --ids 0,1,2,3 --tag "$tag" --spec "$spec" --out $out "$@" 2>&1 | grep -v Warning | grep -v '^  '
}
mr est_r37/mech_nott rec_mech_nott "" $M/runs/r37_rec.jsonl --record $M/rec_r37
mr est_r38x/mech_nott_noext rec_mech_nott "" $M/runs/r38x_rec_noext.jsonl --record $M/rec_r38noext
mr est_r38x/mech_nott_lawon rec_mech_nott "" $M/runs/r38x_rec_lawon.jsonl --record $M/rec_r38lawon
for s in "z drop=zero" "m drop=mc"; do
  set -- $s
  mr est_r37/mech_nott "$1" "$2" $M/runs/r37_rec.jsonl
  mr est_r38x/mech_nott_noext "$1" "$2" $M/runs/r38x_rec_noext.jsonl
  mr est_r38x/mech_nott_lawon "$1" "$2" $M/runs/r38x_rec_lawon.jsonl
done
echo "done round2c"
