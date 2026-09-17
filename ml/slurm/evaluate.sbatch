#!/bin/bash
# Runs scripts/evaluate.py against a trained checkpoint on the held-out test split — the
# data the model never saw at all, not even for monitoring during training. This is the
# number that belongs in the report, not the val accuracy printed during training.
#
# Uses the same MIG-UUID resolution and direct-python-path approach as smoke.sbatch and
# train.sbatch — see those files' comments for why. Evaluation is inference-only (no
# training), so this finishes in well under a minute of actual compute.

#SBATCH --job-name=gvg_eval
#SBATCH --output=eval_%j.log
#SBATCH --error=eval_%j.log
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=2
#SBATCH --gres=gpu:1g.10gb:1
#SBATCH --mem=8G
#SBATCH --time=00:15:00
#SBATCH --partition=gpu

echo "Job ID: $SLURM_JOB_ID"
echo "Node: $(hostname)"
echo "Allocated GPUs (raw Slurm index): $CUDA_VISIBLE_DEVICES"

MIG_UUID=$(nvidia-smi -L | grep -E "Device[[:space:]]*${CUDA_VISIBLE_DEVICES}:" | grep -oE 'MIG-[0-9a-f-]+')
if [ -n "$MIG_UUID" ]; then
    export CUDA_VISIBLE_DEVICES="$MIG_UUID"
fi
echo "Resolved CUDA_VISIBLE_DEVICES: $CUDA_VISIBLE_DEVICES"

cd "$SLURM_SUBMIT_DIR"

~/.conda/envs/geovision/bin/python scripts/evaluate.py \
    --checkpoint checkpoints/xcit_nano_elementwise_epoch9.pt \
    --fusion elementwise \
    --split test \
    --batch-size 64
