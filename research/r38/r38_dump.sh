#!/usr/bin/env bash
# R38b corrector retrain, step 1 (coordinator 10-21 12:0x): dump the 406-network fit pool (the qx3a15 tags less S =
# full 0-47) with the r38 engine (r37, the law masked in the engine, scale-mode extraction x100), two gru_sread2 streams.
#   cd ~/nova-arc && nohup bash research/r38/r38_dump.sh estimators/r38/r38_sub_x100 >> research/r38/logs/dump.log 2>&1 < /dev/null &
cd ~/nova-arc || exit 1
EST=$1; OUT=research/r25a/gru/r38x
mkdir -p "$OUT"
for i in 1 2; do
  bash research/r25a/gru_sread2.sh r38d$i "$EST" "$OUT" research/r25a/gru/tags_r38_$i.json 4 3 >> research/r25a/logs/sread_r38d$i.log 2>&1 &
done
wait
echo "$(date '+%F %T') r38 dump DONE ($EST -> $OUT)"
