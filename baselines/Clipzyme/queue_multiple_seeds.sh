#!/bin/bash
set -e

# Support simultaneous jobs across multiple GPUs
ts -S 8
ts --set_gpu_free_perc 50

# Three seeds: 0, 1, 2
# The trainable 650M ESM encoder OOMs on 22 GB A10 jobs at batch size 8 before a
# checkpoint is written, which makes the dependent eval jobs skip. Use a smaller
# microbatch with accumulation to preserve the previous single-GPU optimizer
# update size: 8 batch * 2 accumulation = 1 batch * 16 accumulation.
echo "Queuing pipeline for multiple seeds..."
conda run -n clipzyme python -m emulator_bench.queue_pipeline \
    --split-group random_splits \
    --split-group enzyme_sequence_splits \
    --split-group enzyme_structure_splits \
    --split-group uniprot_time_splits \
    --split-group reaction_drfp_tanimoto_splits \
    --env-name clipzyme \
    --spooler-bin ts \
    --gpus-per-job 1 \
    --batch-size 1 \
    --accumulate-grad-batches 16 \
    --seed 0 \
    --seed 1 \
    --seed 2

echo "All seeds queued."
