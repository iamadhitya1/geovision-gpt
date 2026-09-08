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
echo "Allocated GPUs (raw Slurm index): $CUDA_VISIBLE_DEVICES"

# MIG quirk, confirmed 2026-09-08: Slurm hands CUDA_VISIBLE_DEVICES as a bare MIG device
# index (e.g. "5"), but CUDA needs the MIG UUID for that index to actually see the device
# — a bare index alone left torch.cuda.is_available() False with no error message. Resolve
# it here at runtime so this works regardless of which MIG slice gets allocated.
MIG_UUID=$(nvidia-smi -L | grep -E "Device[[:space:]]*${CUDA_VISIBLE_DEVICES}:" | grep -oE 'MIG-[0-9a-f-]+')
if [ -n "$MIG_UUID" ]; then
    export CUDA_VISIBLE_DEVICES="$MIG_UUID"
fi
echo "Resolved CUDA_VISIBLE_DEVICES: $CUDA_VISIBLE_DEVICES"

# If the HF weights cache was uploaded (see ml/README.md "On GPU day"), this keeps the job
# from depending on the GPU node having outbound internet — the loader reads the local
# cache exactly like it would hit the network, but never tries to. Comment both lines out
# on the first attempt if you haven't uploaded the cache yet and want to test live internet
# access instead.
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

cd "$SLURM_SUBMIT_DIR"   # run sbatch from inside ml/ so this lands in the right place
~/.conda/envs/geovision/bin/python scripts/train.py --smoke
