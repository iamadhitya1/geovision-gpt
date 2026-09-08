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
# Confirmed via `sinfo` + `scontrol show node gpunode` on 2026-09-08: this cluster has a
# single Slurm partition ("gpu") covering one node with Gres=gpu:ampere:3 (GPU 0/1/2 — full
# GPUs, indistinguishable by name, so a generic request risks landing on reserved GPU 0),
# gpu:1g.20gb:2 and gpu:1g.10gb:4 (GPU 3's six MIG slices, matching the coordinator's email
# exactly). Requesting a named 1g.10gb slice is the only way to guarantee GPU 3's small
# tier and never GPU 0 — a generic `--gres=gpu:1` cannot make that guarantee here.

#SBATCH --job-name=gvg_smoke
#SBATCH --output=smoke_%j.log
#SBATCH --error=smoke_%j.log
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=2
#SBATCH --gres=gpu:1g.10gb:1
#SBATCH --mem=8G
#SBATCH --time=00:15:00
#SBATCH --partition=gpu

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

cd "$SLURM_SUBMIT_DIR"   # run sbatch from inside ml/ so this lands in the right place
~/.conda/envs/geovision/bin/python scripts/train.py --smoke
