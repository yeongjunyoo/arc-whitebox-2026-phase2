#!/usr/bin/env bash
# Write-up ablation on 48 untouched networks of Keenan Pepper's public corpus (coordinator 10-20; plan in ARC MISSION G2
# before any number): idx 3408 + 8 k and 3409 + 8 k (k = 0..23, research/wu/kp_ids.txt), which no lane dumped, trained
# on or read. Thirteen estimator files, one network per job through the shared job wrapper (5 GB, 1 thread); pairing and
# the corpus floor come later (research/wu/paired.py --floor research/wu/kp_floor.json). Starts only after the smoke
# test passed. After R33's "V90 PICK" line it pauses until 10-22 02:30 KST (its S read, chain and last fire need a
# quiet Mac) unless R33 ends first with STOP or a failed rule.
#   cd ~/nova-arc && nohup bash research/wu/run_kp.sh < /dev/null > /dev/null 2>&1 &
cd ~/nova-arc || exit 1
mkdir -p research/wu/kp research/wu/logs
exec >> research/wu/logs/run_kp.log 2>&1
say() { echo "$(date '+%m-%d %H:%M:%S') [wu-kp] $*"; }
B=k3_fast_aug_sc_2l_b_lc2k_rd_wm_ex35_satrole_fbv8_rsw_cf_sp_d6p_sb2_dc20x_r143bcxa8tv_r24wc10_r25ak4f10L_gA_r25ces_r25bc3e7x_r25agf05_r25aqx_r25av2a_opb30a15_r25av4kd05s3_opi175_opkcA_opslcs_opr30
CONFIGS="sub:estimators/$B engine:estimators/wu2/v_engine engine_nostar:estimators/wu2/v_engine_nostar
engine_nolaw:estimators/wu2/v_engine_nolaw engine_nok4f:estimators/wu2/v_engine_nok4f engine_nogf:estimators/wu2/v_engine_nogf
nostar:estimators/wu2/v_nostar noapa:estimators/wu2/v_noapa
ref_simple:estimators/k3_fast.py old_2leg:estimators/wu2/old/k3_fast_aug_sc_2l.py
old_3leg:estimators/wu2/old/k3_fast_aug_sc.py old_ex35:estimators/k3_fast_aug_sc_2l_b_lc2k_rd_wm_ex35
old_sat:estimators/k3_fast_aug_sc_2l_b_lc2k_rd_wm_ex35_satrole_f"
IDS=$(cat research/wu/kp_ids.txt)
R33LOG=research/r33/logs/r33.log
RESUME=$(date -j -f "%Y-%m-%d %H:%M" "2026-10-22 02:30" +%s)
until grep -q "\[wu2\] end" research/wu/smoke/smoke.log 2>/dev/null; do sleep 120; done
if grep "\[wu2\]" research/wu/smoke/smoke.log | grep -q "rc=[1-9]"; then say "smoke test has a failed run: not starting"; exit 1; fi
r33_busy() {  # R33's own pick line only (its 11:17 start line also names "V90 PICK"); research/wu/kp.release ends the pause
  [ -e research/wu/kp.release ] && return 1
  grep -q "\[r33\] STOP\|\[r33\] r33 STOP\|pre-registered rule: FAIL" "$R33LOG" && return 1
  grep -q "\[r33\] V90 PICK kdk05s3" "$R33LOG" && [ "$(date +%s)" -lt "$RESUME" ] && return 0
  return 1
}
say "start: $(echo $IDS | wc -w) networks"
for c in $CONFIGS; do
  tag=${c%%:*}; est=${c#*:}; out=research/wu/kp/$tag.jsonl
  for i in $IDS; do
    grep -q "\"mlp_id\": $i," "$out" 2>/dev/null && continue
    while r33_busy; do sleep 300; done
    while :; do
      MJ_WAIT_S=3600 bash research/r25a/mac_job25.sh "wukp_$tag" 5 1 10 -- env OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 \
        MKL_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 .venv-fs0123/bin/python -u research/wu/ablate.py --estimator "$est" \
        --set kp --ids "$i" --snapshot research/r2_data --out "$out" > "research/wu/kp/$tag.last.log" 2>&1 < /dev/null
      rc=$?
      { [ $rc -eq 6 ] || [ $rc -eq 76 ] || [ $rc -eq 4 ]; } || break
      sleep 60
    done
    [ $rc -eq 0 ] || say "$tag kp_$i rc=$rc: $(tail -1 research/wu/kp/$tag.last.log | cut -c1-200)"
  done
  say "$tag done: $(grep -c '"mlp_id"' "$out") networks"
done
say end
