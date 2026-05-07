#!/bin/bash
set -e

# Configure Task Spooler to run up to 8 simultaneous jobs (adapt based on your CPU/GPUs)
# and consider GPUs free if they have 70% available memory.
ts -S 8
ts --set_gpu_free_perc 50

# Ensure the cache runs sequentially or let task-spooler decide based on dependencies.
# Three seeds: 0, 1, 2
echo "Queuing pipeline for multiple seeds..."
python -m emulator_bench.queue_pipeline \
    --split-group random_splits \
    --split-group enzyme_sequence_splits \
    --split-group enzyme_structure_splits \
    --split-group uniprot_time_splits \
    --split-group reaction_drfp_tanimoto_splits \
    --env-name current \
    --spooler-bin ts \
    --gpus-per-job 1 \
    --seed 0 \

echo "All seeds queued."