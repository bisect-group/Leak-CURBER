#!/bin/bash
set -e

# Support simultaneous jobs across multiple GPUs
ts -S 8
ts --set_gpu_free_perc 70

# Three seeds: 0, 1, 2
echo "Queuing pipeline for multiple seeds..."
python -m emulator_bench.queue_pipeline \
    --split-group random_splits \
    --split-group enzyme_sequence_splits \
    --split-group enzyme_structure_splits \
    --split-group uniprot_time_splits \
    --env-name current \
    --spooler-bin ts \
    --gpus-per-job 1 \
    --seed 0 \
    --seed 1 \
    --seed 2

echo "All seeds queued."
