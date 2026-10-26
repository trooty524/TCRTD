#!/usr/bin/env bash
# ResNet-50/101 person experiments with the reported per-dataset settings.
# Run after preparing data/ with tools/prepare_data.py.
set -euo pipefail

SEEDS=(13 42 2026)
#        dataset     backbone  epochs  topology  swap      recipe
CELLS=(
  "market1501 resnet50  80 0.013333 0.002667 --sampler camera_aware"
  "market1501 resnet101 70 0.027429 0.004571 --sampler camera_aware"
  "msmt17     resnet50  80 0.026667 0.005333 --sampler random"
  "msmt17     resnet101 80 0.020000 0.012000 --sampler camera_aware --p 32 --optimizer adamw --lr-schedule cosine --weight-decay 0.005"
  "nightreid  resnet50  75 0.027429 0.004571 --sampler camera_aware"
  "nightreid  resnet101 55 0.020000 0.012000 --sampler random"
)

for cell in "${CELLS[@]}"; do
  read -r data backbone epochs wt ws recipe <<< "${cell}"
  for seed in "${SEEDS[@]}"; do
    for arm in m0 tcrtd; do
      out=runs/${data}/${backbone}/${arm}_s${seed}
      python train.py --train-root data/${data}/train --output ${out} --backbone ${backbone} \
          --objective ${arm} --epochs ${epochs} --seed ${seed} \
          --topology-weight ${wt} --swap-weight ${ws} ${recipe}
      python evaluate.py --checkpoint ${out}/model.pth --query-root data/${data}/test/query \
          --gallery-root data/${data}/test/gallery --output ${out}/test
    done
  done
done
