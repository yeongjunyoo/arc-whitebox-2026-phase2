#!/usr/bin/env bash
# Write-up ablation smoke (coordinator 10-20): each new variant and old file on one public network (mini 0; the
# loader's full path on full 0, an acceptance-set network), so that the held-out run of 10-23 cannot fail on a
# typo. Starts after the R33 trainings end; every run goes through the shared job wrapper (5 GB, 1 thread).
#   cd ~/nova-arc && nohup bash research/wu/smoke.sh < /dev/null > /dev/null 2>&1 &
cd ~/nova-arc || exit 1
mkdir -p research/wu/smoke
exec >> research/wu/smoke/smoke.log 2>&1
say() { echo "$(date '+%m-%d %H:%M:%S') [wu2] $*"; }
SNAP=$HOME/.cache/huggingface/hub/datasets--aicrowd--arc-whestbench-public-2026/snapshots/aa99830fdc09fad15407b10e8e3459d3e18bba0a
B=k3_fast_aug_sc_2l_b_lc2k_rd_wm_ex35_satrole_fbv8_rsw_cf_sp_d6p_sb2_dc20x_r143bcxa8tv_r24wc10_r25ak4f10L_gA_r25ces_r25bc3e7x_r25agf05_r25aqx_r25av2a_opb30a15_r25av4kd05s3_opi175_opkcA_opslcs_opr30
until grep -q "teacher export of\|\[r33\] STOP" research/r33/logs/r33.log; do sleep 120; done
say "start (R33 trainings done)"
run() {  # tag estimator set id
  MJ_WAIT_S=1800 bash research/r25a/mac_job25.sh "wu2_$1" 5 1 30 -- env OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 \
    MKL_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 .venv-fs0123/bin/python -u research/wu/ablate.py --estimator "$2" \
    --set "$3" --ids "$4" --snapshot "$SNAP" --out "research/wu/smoke/$1.jsonl" > "research/wu/smoke/$1.log" 2>&1 < /dev/null
  say "$1 rc=$? $(grep -o '"final_mse": [0-9.e-]*, "flops": [0-9]*' research/wu/smoke/$1.jsonl 2>/dev/null | tail -1)"
}
run sub_full0 "estimators/$B" full 0
run sub estimators/$B mini 0
run engine estimators/wu2/v_engine mini 0
run engine_nostar estimators/wu2/v_engine_nostar mini 0
run engine_nolaw estimators/wu2/v_engine_nolaw mini 0
run nostar estimators/wu2/v_nostar mini 0
run noapa estimators/wu2/v_noapa mini 0
run ref_simple estimators/k3_fast.py mini 0
run old_3leg estimators/wu2/old/k3_fast_aug_sc.py mini 0
run old_2leg estimators/wu2/old/k3_fast_aug_sc_2l.py mini 0
run old_ex35 estimators/k3_fast_aug_sc_2l_b_lc2k_rd_wm_ex35 mini 0
run old_sat estimators/k3_fast_aug_sc_2l_b_lc2k_rd_wm_ex35_satrole_f mini 0
say end
