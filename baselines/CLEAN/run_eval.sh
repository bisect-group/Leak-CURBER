#!/bin/bash

cd ${PROJECT_ROOT}/baselines/CLEAN
export CUDA_VISIBLE_DEVICES=0

for train_json in emulator_bench/runs/*/seeds/*/train.json; do
  split_group="$(basename "$(dirname "$(dirname "$(dirname "$train_json")")")")"
  seed="$(basename "$(dirname "$train_json")")"

  conda run -n clean python -m emulator_bench.evaluate \
    --split-group "$split_group" \
    --runs-root emulator_bench/runs \
    --seed "$seed" \
    --eval-split both \
    --device cuda
done
