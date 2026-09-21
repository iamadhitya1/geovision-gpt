#!/bin/bash
#SBATCH --job-name=gvg_download_hr
#SBATCH --partition=gpu
#SBATCH --cpus-per-task=2
#SBATCH --mem=4G
#SBATCH --time=03:00:00
#SBATCH --output=download_hr_%j.log
#
# Fetch RSVQA-HR (Lobry et al., CC-BY-4.0, Zenodo record 6344367) into dataset/rsvqa_hr/.
# CPU-only: no GPU is requested, because a download does not need one. Resumable (wget -c), so
# if the job is cut off, submit it again and it continues.
#
# Only what training and evaluation need is fetched: the 12 split files (~540 MB) and the image
# archive (13.5 GB). The three whole-dataset question/answer files and the people file are not
# needed and are skipped. The archive is deleted after extraction ONLY if the expected number of
# images (10,659) is found, to keep the footprint on the shared drive small.

set -euo pipefail

echo "Job ID: ${SLURM_JOB_ID:-none}   started: $(date)   node: $(hostname)"

DEST="${SLURM_SUBMIT_DIR:-$PWD}/dataset/rsvqa_hr"
BASE="https://zenodo.org/records/6344367/files"
mkdir -p "$DEST"
cd "$DEST"

echo "--- free space on this drive before:"
df -h "$DEST" | tail -1

SPLITS=(
  USGS_split_train_questions.json USGS_split_train_answers.json USGS_split_train_images.json
  USGS_split_val_questions.json   USGS_split_val_answers.json   USGS_split_val_images.json
  USGS_split_test_questions.json  USGS_split_test_answers.json  USGS_split_test_images.json
  USGS_split_test_phili_questions.json USGS_split_test_phili_answers.json USGS_split_test_phili_images.json
)

fetch() {
  # -c resumes a partial file; a few retries because Zenodo occasionally drops connections.
  wget -c --tries=8 --waitretry=20 --timeout=60 --progress=dot:giga -O "$1" "$BASE/$1?download=1"
}

for f in "${SPLITS[@]}"; do
  echo "--- fetching $f"
  fetch "$f"
done

echo "--- split files done: $(date)"
ls -la "${SPLITS[@]}"

if [ ! -d Images ] || [ "$(find Images -type f 2>/dev/null | wc -l)" -lt 10000 ]; then
  echo "--- fetching Images.tar (13.5 GB)"
  fetch Images.tar
  echo "--- extracting: $(date)"
  tar -xf Images.tar
fi

COUNT=$(find . -type f \( -iname '*.png' -o -iname '*.tif' -o -iname '*.tiff' -o -iname '*.jpg' -o -iname '*.jpeg' \) | wc -l)
echo "--- image files found: $COUNT (expected 10659)"
echo "--- layout:"
ls | head -20
find . -maxdepth 3 -type d | head -10
find . -type f \( -iname '*.png' -o -iname '*.tif' -o -iname '*.tiff' -o -iname '*.jpg' -o -iname '*.jpeg' \) | head -3

if [ "$COUNT" -ge 10000 ] && [ -f Images.tar ]; then
  echo "--- image count OK, deleting Images.tar to save shared space"
  rm Images.tar
fi

# Show the JSON structure so the loader can be written from the real format.
for f in USGS_split_val_questions.json USGS_split_val_answers.json USGS_split_val_images.json; do
  echo "=== first 1500 bytes of $f"
  head -c 1500 "$f"
  echo
done

echo "--- free space on this drive after:"
df -h "$DEST" | tail -1
echo "Job finished on: $(date)"
