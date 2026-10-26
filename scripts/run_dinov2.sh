#!/usr/bin/env bash
# DINOv2 person experiments, Smooth-AP comparisons and NightReID target checks.
# Run after preparing data/ with tools/prepare_data.py.
set -euo pipefail

declare -A EPOCHS=([market1501]=6 [msmt17]=4 [nightreid]=6)   # budgets chosen on held-out identities
SEEDS=(13 42 2026)

train_and_test () {   # dataset backbone name seed [extra train arguments]
  local data=$1 backbone=$2 name=$3 seed=$4; shift 4
  local out=runs/${data}/${backbone}/${name}_s${seed}
  python train.py --train-root data/${data}/train --output ${out} --backbone ${backbone} \
      --epochs ${EPOCHS[$data]} --seed ${seed} "$@"
  python evaluate.py --checkpoint ${out}/model.pth --query-root data/${data}/test/query \
      --gallery-root data/${data}/test/gallery --output ${out}/test --save-features
}

for data in market1501 msmt17 nightreid; do
  for backbone in dinov2_s14 dinov2_b14; do
    for seed in "${SEEDS[@]}"; do
      train_and_test ${data} ${backbone} m0              ${seed} --objective m0
      train_and_test ${data} ${backbone} tcrtd           ${seed} --objective tcrtd
      train_and_test ${data} ${backbone} smooth_ap       ${seed} --objective smooth_ap
      train_and_test ${data} ${backbone} smooth_ap_tcrtd ${seed} --objective tcrtd_smooth_ap
    done
  done
done

# NightReID target policies and AP-swap ranking policies.
for backbone in dinov2_s14 dinov2_b14; do
  for seed in "${SEEDS[@]}"; do
    for rank in candidate_order student_score; do
      train_and_test nightreid ${backbone} legacy_${rank} ${seed} --relation legacy --rank-policy ${rank}
    done
    train_and_test nightreid ${backbone} junk_aware_candidate_order ${seed} --rank-policy candidate_order
    train_and_test nightreid ${backbone} fixed_support ${seed} --relation fixed_support --rank-policy candidate_order
  done
done

# Paired interval for Market-1501 and DINOv2-B/14: for Market-1501 and DINOv2-B/14:
python tools/paired_bootstrap.py \
  --baseline runs/market1501/dinov2_b14/m0_s{13,42,2026}/test \
  --method   runs/market1501/dinov2_b14/tcrtd_s{13,42,2026}/test
