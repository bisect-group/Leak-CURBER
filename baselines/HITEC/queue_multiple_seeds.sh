#!/bin/bash
set -e

# Configure Task Spooler to run up to 8 simultaneous jobs
ts -S 8
ts --set_gpu_free_perc 70

echo "Queuing HITEC Leak-CURBER pipeline for multiple seeds..."
python -m emulator_bench.queue_pipeline \
    --split-group random_splits \
    --split-group enzyme_sequence_splits \
    --split-group enzyme_structure_splits \
    --split-group uniprot_time_splits \
    --env-name current \
    --spooler-bin ts \
    --epochs 80 \
    --batch-size 2 \
    --cache-gpus 0 \
    --train-gpus 1 \
    --eval-gpus 1 \
    --seed 0 \
    --seed 1 \
    --seed 2

echo "All HITEC seeds queued."
