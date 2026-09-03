#!/bin/bash
# The real training run — submit only after smoke.sbatch has completed cleanly. This is
# the one job the "one job per user" rule allows at a time, so don't submit this alongside
# anything else, and don't resubmit if it's sitting in PENDING — it starts on its own once
# resources free up.
#
# Written directly on the cluster with `nano train.sbatch` (see smoke.sbatch's header for
# why). Fill in --partition the same way — a GPU 3 light slice, never GPU 0.
#
# --fusion elementwise reproduces the paper's own best configuration. Once this finishes,
# copy this file (or just resubmit with --fusion cross_attention edited below) for the
# project's own experiment — comparing that result against this one is the actual
# contribution on top of the paper's baseline (see ml/README.md, "On GPU day" step 6).

#SBATCH --job-name=gvg_train
#SBATCH --output=train_%j.log
#SBATCH --error=train_%j.log
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --mem=16G
#SBATCH --time=04:00:00
#SBATCH --partition=REPLACE_WITH_GPU3_LIGHT_PARTITION

echo "Job ID: $SLURM_JOB_ID"
echo "Job started on: $(date)"
echo "Node: $(hostname)"
echo "Allocated GPUs: $CUDA_VISIBLE_DEVICES"

export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

source ~/miniconda3/etc/profile.d/conda.sh
conda activate geovision

cd "$SLURM_SUBMIT_DIR"

python scripts/train.py \
    --image-encoder xcit_nano \
    --fusion elementwise \
    --epochs 10 \
    --batch-size 512

echo "Job finished on: $(date)"
