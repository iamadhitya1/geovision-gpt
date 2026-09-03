#!/bin/bash
# Run this FIRST, before train.sbatch. It proves the cluster environment itself works
# (conda env activates, torch sees the GPU, the HF cache resolves, the data loads) on a
# ~30-example / 1-epoch run that finishes in well under a minute of actual compute. Only
# spend the real training job's slot (see "one job per user" in the coordinator's email)
# once this has printed "Training complete." in its log with no errors.
#
# Written directly on the cluster with `nano smoke.sbatch` — the manual says job scripts
# must be typed/pasted into the cluster's own editor, not uploaded as a file. The rest of
# ml/ (code, dataset, HF cache) uploads fine as a zip via MobaXterm or scp/rsync.
#
# Before submitting, fill in --partition below — run `sinfo` on the master node first and
# pick one of GPU 3's small (10GB) slices. Never GPU 0 (reserved) or GPU 1/2 (for jobs
# needing >20GB — this model doesn't).

#SBATCH --job-name=gvg_smoke
#SBATCH --output=smoke_%j.log
#SBATCH --error=smoke_%j.log
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=2
#SBATCH --gres=gpu:1
#SBATCH --mem=8G
#SBATCH --time=00:15:00
#SBATCH --partition=REPLACE_WITH_GPU3_LIGHT_PARTITION

echo "Job ID: $SLURM_JOB_ID"
echo "Node: $(hostname)"
echo "Allocated GPUs: $CUDA_VISIBLE_DEVICES"

# If the HF weights cache was uploaded (see ml/README.md "On GPU day"), this keeps the job
# from depending on the GPU node having outbound internet — the loader reads the local
# cache exactly like it would hit the network, but never tries to. Comment both lines out
# on the first attempt if you haven't uploaded the cache yet and want to test live internet
# access instead.
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

source ~/miniconda3/etc/profile.d/conda.sh   # adjust if using module-load anaconda3 instead
conda activate geovision

cd "$SLURM_SUBMIT_DIR"   # run sbatch from inside ml/ so this lands in the right place
python scripts/train.py --smoke
