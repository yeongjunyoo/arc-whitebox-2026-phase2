#!/usr/bin/env bash
# Write-up ablation on the public one-shot selection set (coordinator 10-20; plan in ARC MISSION G2 before any number):
# public mini 60-99 and full 304-351, held out of all training and acceptance reads and read once on 10-16 to choose the
# designated submissions. Starts only after the nomination window closes (2026-10-23 09:05 KST). The same thirteen
# estimator files as research/wu/run_kp.sh, one network per job through the shared job wrapper (5 GB, 1 thread).
# Optional arguments: a subset of the config tags (to run two instances on disjoint halves).
#   cd ~/nova-arc && nohup bash research/wu/run_pub.sh [tag ...] < /dev/null > /dev/null 2>&1 &
cd ~/nova-arc || exit 1
mkdir -p research/wu/pub research/wu/logs
exec >> research/wu/logs/run_pub.log 2>&1
say() { echo "$(date '+%m-%d %H:%M:%S') [wu-pub] $*"; }
B=k3_fast_aug_sc_2l_b_lc2k_rd_wm_ex35_satrole_fbv8_rsw_cf_sp_d6p_sb2_dc20x_r143bcxa8tv_r24wc10_r25ak4f10L_gA_r25ces_r25bc3e7x_r25agf05_r25aqx_r25av2a_opb30a15_r25av4kd05s3_opi175_opkcA_opslcs_opr30
SNAP=$HOME/.cache/huggingface/hub/datasets--aicrowd--arc-whestbench-public-2026/snapshots/aa99830fdc09fad15407b10e8e3459d3e18bba0a
CONFIGS="sub:estimators/$B engine:estimators/wu2/v_engine engine_nostar:estimators/wu2/v_engine_nostar
engine_nolaw:estimators/wu2/v_engine_nolaw engine_nok4f:estimators/wu2/v_engine_nok4f engine_nogf:estimators/wu2/v_engine_nogf
nostar:estimators/wu2/v_nostar noapa:estimators/wu2/v_noapa
ref_simple:estimators/k3_fast.py old_2leg:estimators/wu2/old/k3_fast_aug_sc_2l.py
old_3leg:estimators/wu2/old/k3_fast_aug_sc.py old_ex35:estimators/k3_fast_aug_sc_2l_b_lc2k_rd_wm_ex35
old_sat:estimators/k3_fast_aug_sc_2l_b_lc2k_rd_wm_ex35_satrole_f"
WANT="$*"
START=$(date -j -f "%Y-%m-%d %H:%M" "2026-10-23 09:05" +%s)
while [ "$(date +%s)" -lt "$START" ]; do sleep 300; done
say "start${WANT:+ (configs: $WANT)}"
for c in $CONFIGS; do
  tag=${c%%:*}; est=${c#*:}
  [ -n "$WANT" ] && ! echo " $WANT " | grep -q " $tag " && continue
  for part in "mini:$(seq -s ' ' 60 99)" "full:$(seq -s ' ' 304 351)"; do
    set_=${part%%:*}; ids=${part#*:}; out=research/wu/pub/${tag}_${set_}.jsonl
    for i in $ids; do
      grep -q "\"mlp_id\": $i," "$out" 2>/dev/null && continue
      while :; do
        MJ_WAIT_S=3600 bash research/r25a/mac_job25.sh "wupub_$tag" 5 1 10 -- env OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 \
          MKL_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 .venv-fs0123/bin/python -u research/wu/ablate.py --estimator "$est" \
          --set "$set_" --ids "$i" --snapshot "$SNAP" --out "$out" > "research/wu/pub/$tag.last.log" 2>&1 < /dev/null
        rc=$?
        { [ $rc -eq 6 ] || [ $rc -eq 76 ] || [ $rc -eq 4 ]; } || break
        sleep 60
      done
      [ $rc -eq 0 ] || say "$tag $set_ $i rc=$rc: $(tail -1 research/wu/pub/$tag.last.log | cut -c1-200)"
    done
  done
  say "$tag done: $(cat research/wu/pub/${tag}_*.jsonl 2>/dev/null | grep -c '"mlp_id"') networks"
done
say end
