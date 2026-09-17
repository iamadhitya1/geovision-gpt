#!/bin/bash
# The project's own experiment: identical setup to train.sbatch, only --fusion changed
# from elementwise to cross_attention. Comparing this result against the elementwise
# baseline (job 4297: 72.74% overall / 75.16% average accuracy on the RSVQA-LR test split,
# confirmed via evaluate.sbatch job 4323) is the actual original contribution on top of
# the LiT-4-RSVQA paper's published baseline.
#
# Same confirmed-working GPU/GRES/MIG-UUID/python-path setup as train.sbatch — see that
# file's comments for the full explanation of each.

#SBATCH --job-name=gvg_train_xattn
#SBATCH --output=train_xattn_%j.log
#SBATCH --error=train_xattn_%j.log
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
    --fusion cross_attention \
    --epochs 10 \
    --batch-size 64

echo "Job finished on: $(date)"
