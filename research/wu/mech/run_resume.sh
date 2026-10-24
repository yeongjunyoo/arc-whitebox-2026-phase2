#!/usr/bin/env bash
# Band study (research/wu/mech): the runs the 10-21 20:33 pause stopped, resumed at 06:30 KST 10-22 (coordinator):
# Monte Carlo truth for public mini 2 and 3 first (1.6 GB each, inside the 06:00-08:30 window's 4 GB ARC cap), then one
# engine chain (about 7 GB, so the gate holds it until the window ends): the per-layer swaps left (run_layers.sh skips
# finished tags), the scale test (run_scale.sh), the swap on mini 2 and 3 (run_more.sh), band on / off of r38 and opr30
# on minis 0-15 (run_ext.sh); then the reports (one thread, small).
#   cd ~/nova-arc && nohup bash research/wu/mech/run_resume.sh < /dev/null > research/wu/mech/logs/resume.log 2>&1 &
cd ~/nova-arc || exit 1
export MB_WINDOWS="2026-10-22 00:30 2026-10-22 02:30;2026-10-22 06:00 2026-10-22 08:30"
M=research/wu/mech
for i in 2 3; do
  rm -f $M/mc/mc_mini$i.npz $M/mc/mc_mini$i.json  # a stopped run leaves nothing usable; start clean
  MJ_WAIT_S=7200 bash research/r25a/mac_job25.sh mech_mc$i 1.6 4 35 -- env VECLIB_MAXIMUM_THREADS=4 OPENBLAS_NUM_THREADS=4 \
    .venv-fs0123/bin/python -u $M/mc_truth.py --id $i --n 10000000 --batch 50000 --out $M/mc < /dev/null > $M/logs/mc$i.log 2>&1 &
done
MJ_WAIT_S=14400 bash research/r25a/mac_job25.sh mech_resume 7 1 100 -- bash -c \
  "bash $M/run_layers.sh >> $M/logs/layers.log 2>&1; bash $M/run_scale.sh > $M/logs/scale.log 2>&1; bash $M/run_more.sh > $M/logs/more.log 2>&1; bash $M/run_ext.sh > $M/logs/ext.log 2>&1"
P=.venv-fs0123/bin/python
for s_ in mech_report mech_local mech_decomp fill_readme; do VECLIB_MAXIMUM_THREADS=1 $P $M/$s_.py > $M/logs/$s_.log 2>&1; done
wait
echo "resume done $(date '+%F %T')"
