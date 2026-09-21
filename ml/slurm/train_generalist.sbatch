#!/bin/bash
# Train the generalist model (RSVQA-LR + RSVQA-HR, augmentation, resumable).
#
# One script, configured by environment variables so it can serve the timing probe, the real run
# and the variants. sbatch passes your environment through, so:
#
#   TAG=timing_probe MAX_MINUTES=10 sbatch train_generalist.sbatch     # 10-minute speed probe
#   TAG=gen_xcit sbatch train_generalist.sbatch                        # the real run
#   TAG=gen_xcit_xattn FUSION=cross_attention sbatch train_generalist.sbatch
#
# RESUMING: the trainer stops itself cleanly after MAX_MINUTES (default 215, i.e. before the 4 hour
# limit) and saves its full state. If the log ends with "STOPPING ... Run again to resume", submit
# the SAME command again — it continues from the same step. Only when the log says "Training
# complete" is the run finished. Never submit while a job is PENDING or running (one job per user).
#
# Cluster settings are copied from train.sbatch, which is known to work: a named 1g.10gb slice
# (guarantees MIG slice of GPU 3, never the reserved GPU 0), the MIG UUID workaround, and offline
# mode for the cached pretrained weights.

#SBATCH --job-name=gvg_generalist
#SBATCH --output=generalist_%j.log
#SBATCH --error=generalist_%j.log
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1g.10gb:1
#SBATCH --mem=16G
#SBATCH --time=04:00:00
#SBATCH --partition=gpu

TAG="${TAG:?set TAG, e.g.  TAG=gen_xcit sbatch train_generalist.sbatch}"
DATASETS="${DATASETS:-lr+hr}"
FUSION="${FUSION:-elementwise}"
EPOCHS="${EPOCHS:-8}"
HR_PER_EPOCH="${HR_PER_EPOCH:-60000}"
MAX_MINUTES="${MAX_MINUTES:-215}"
EXTRA_ARGS="${EXTRA_ARGS:-}"

echo "Job ID: $SLURM_JOB_ID"
echo "Job started on: $(date)"
echo "Node: $(hostname)"
echo "Run: TAG=$TAG DATASETS=$DATASETS FUSION=$FUSION EPOCHS=$EPOCHS HR_PER_EPOCH=$HR_PER_EPOCH MAX_MINUTES=$MAX_MINUTES $EXTRA_ARGS"
echo "Allocated GPUs (raw Slurm index): $CUDA_VISIBLE_DEVICES"

MIG_UUID=$(nvidia-smi -L | grep -E "Device[[:space:]]*${CUDA_VISIBLE_DEVICES}:" | grep -oE 'MIG-[0-9a-f-]+')
if [ -n "$MIG_UUID" ]; then
    export CUDA_VISIBLE_DEVICES="$MIG_UUID"
fi
echo "Resolved CUDA_VISIBLE_DEVICES: $CUDA_VISIBLE_DEVICES"

export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

cd "$SLURM_SUBMIT_DIR"
echo "--- free space on this drive: $(df -h . | tail -1)"

~/.conda/envs/geovision/bin/python scripts/train_generalist.py \
    --tag "$TAG" \
    --datasets "$DATASETS" \
    --fusion "$FUSION" \
    --epochs "$EPOCHS" \
    --hr-per-epoch "$HR_PER_EPOCH" \
    --max-minutes "$MAX_MINUTES" \
    --num-workers 3 \
    --amp \
    $EXTRA_ARGS
CODE=$?

if [ $CODE -eq 3 ]; then
    echo ">>> NOT FINISHED (time budget reached, state saved). Submit the same command again to resume."
    CODE=0
fi

echo "Job finished on: $(date)   exit code: $CODE"
exit $CODE
