#!/bin/bash
# Test-split evaluation for the cross-attention run — mirrors evaluate.sbatch (job 4323,
# which produced the elementwise baseline's final number: 72.74% overall / 75.16% average
# accuracy). Run this only after train_cross_attention.sbatch has finished and saved
# checkpoints/xcit_nano_cross_attention_epoch9.pt.

#SBATCH --job-name=gvg_eval_xattn
#SBATCH --output=eval_xattn_%j.log
#SBATCH --error=eval_xattn_%j.log
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
    --checkpoint checkpoints/xcit_nano_cross_attention_epoch9.pt \
    --fusion cross_attention \
    --split test \
    --batch-size 64
