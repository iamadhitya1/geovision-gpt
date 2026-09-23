#!/bin/bash
# Robustness check for a train_generalist.py checkpoint, on the same perturbations and test
# images used for the original baseline/cross-attention comparison in ml/README.md.
#
#   TAG=gen_xcit sbatch robustness_eval_generalist.sbatch
#
# Small and fast (inference only over ~1,100 questions x 8 perturbations) — the default 30
# minutes is generous.

#SBATCH --job-name=gvg_robust_generalist
#SBATCH --output=robust_generalist_%j.log
#SBATCH --error=robust_generalist_%j.log
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1g.10gb:1
#SBATCH --mem=16G
#SBATCH --time=00:30:00
#SBATCH --partition=gpu

TAG="${TAG:?set TAG, e.g.  TAG=gen_xcit sbatch robustness_eval_generalist.sbatch}"
WHICH="${WHICH:-best}"

echo "Job ID: $SLURM_JOB_ID"
echo "Job started on: $(date)"
echo "Evaluating robustness: TAG=$TAG WHICH=$WHICH"

MIG_UUID=$(nvidia-smi -L | grep -E "Device[[:space:]]*${CUDA_VISIBLE_DEVICES}:" | grep -oE 'MIG-[0-9a-f-]+')
if [ -n "$MIG_UUID" ]; then
    export CUDA_VISIBLE_DEVICES="$MIG_UUID"
fi
echo "Resolved CUDA_VISIBLE_DEVICES: $CUDA_VISIBLE_DEVICES"

export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

cd "$SLURM_SUBMIT_DIR"
~/.conda/envs/geovision/bin/python scripts/robustness_eval_generalist.py --tag "$TAG" --which "$WHICH"

echo "Job finished on: $(date)"
