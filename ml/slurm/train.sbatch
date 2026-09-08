#!/bin/bash
# The real training run — submit only after smoke.sbatch has completed cleanly. This is
# the one job the "one job per user" rule allows at a time, so don't submit this alongside
# anything else, and don't resubmit if it's sitting in PENDING — it starts on its own once
# resources free up.
#
# Written directly on the cluster with `nano train.sbatch` (see smoke.sbatch's header for
# the confirmed --gres/--partition values and why a named 1g.10gb slice is required to
# guarantee GPU 3 and never GPU 0 on this cluster).
#
# --fusion elementwise reproduces the paper's own best configuration. Once this finishes,
# copy this file (or just resubmit with --fusion cross_attention edited below) for the
# project's own experiment — comparing that result against this one is the actual
# contribution on top of the paper's baseline (see ml/README.md, "On GPU day" step 6).
#
# --batch-size 64, not the paper's 512: confirmed via job 4292 that 512 (tuned for a full
# 80GB A100) hits CUDA out of memory on this cluster's ~9.5GB usable 1g.10gb MIG slice.
# 64 is untested-but-conservative; if it fits with room to spare, this can go higher —
# check nvidia-smi's memory usage during the run.

#SBATCH --job-name=gvg_train
#SBATCH --output=train_%j.log
#SBATCH --error=train_%j.log
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1g.10gb:1
#SBATCH --mem=16G
#SBATCH --time=04:00:00
#SBATCH --partition=gpu

echo "Job ID: $SLURM_JOB_ID"
echo "Job started on: $(date)"
echo "Node: $(hostname)"
echo "Allocated GPUs (raw Slurm index): $CUDA_VISIBLE_DEVICES"

# MIG quirk, confirmed 2026-09-08 — see smoke.sbatch's comment for the full explanation.
MIG_UUID=$(nvidia-smi -L | grep -E "Device[[:space:]]*${CUDA_VISIBLE_DEVICES}:" | grep -oE 'MIG-[0-9a-f-]+')
if [ -n "$MIG_UUID" ]; then
    export CUDA_VISIBLE_DEVICES="$MIG_UUID"
fi
echo "Resolved CUDA_VISIBLE_DEVICES: $CUDA_VISIBLE_DEVICES"

export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

cd "$SLURM_SUBMIT_DIR"

~/.conda/envs/geovision/bin/python scripts/train.py \
    --image-encoder xcit_nano \
    --fusion elementwise \
    --epochs 10 \
    --batch-size 64

echo "Job finished on: $(date)"
