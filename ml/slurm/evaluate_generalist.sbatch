#!/bin/bash
# Evaluate a train_generalist.py checkpoint on LR test, HR test, and HR test_phili (the unseen
# city) in one job. Fast (inference only) — a small time/GPU allocation is enough.
#
#   TAG=gen_xcit sbatch evaluate_generalist.sbatch
#
# Same cluster settings as the other jobs here (named 1g.10gb slice, MIG UUID resolution).

#SBATCH --job-name=gvg_eval_generalist
#SBATCH --output=eval_generalist_%j.log
#SBATCH --error=eval_generalist_%j.log
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1g.10gb:1
#SBATCH --mem=16G
#SBATCH --time=00:30:00
#SBATCH --partition=gpu

TAG="${TAG:?set TAG, e.g.  TAG=gen_xcit sbatch evaluate_generalist.sbatch}"
WHICH="${WHICH:-best}"

echo "Job ID: $SLURM_JOB_ID"
echo "Job started on: $(date)"
echo "Evaluating: TAG=$TAG WHICH=$WHICH"

MIG_UUID=$(nvidia-smi -L | grep -E "Device[[:space:]]*${CUDA_VISIBLE_DEVICES}:" | grep -oE 'MIG-[0-9a-f-]+')
if [ -n "$MIG_UUID" ]; then
    export CUDA_VISIBLE_DEVICES="$MIG_UUID"
fi
echo "Resolved CUDA_VISIBLE_DEVICES: $CUDA_VISIBLE_DEVICES"

export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

cd "$SLURM_SUBMIT_DIR"
~/.conda/envs/geovision/bin/python scripts/evaluate_generalist.py --tag "$TAG" --which "$WHICH"

echo "Job finished on: $(date)"
