#!/bin/bash
#SBATCH --job-name=unified-histo
#SBATCH --time=06:00:00
#SBATCH --cpus-per-task=32
#SBATCH --mem=64G

set -euo pipefail

# Args:
#   1: IMAGE_PATH
#   2: OUTPUT_ROOT
#   3: MODE            [all|clahe|stain|nuclei|stain+nuclei]
#   4: MAGNIFICATION   [default: 40]
#   5: TILE_SIZE       [default: 512]
#   6: VENV_PATH       [default: /well/parkkinen/users/nfw313/python-venvs/histomics-env]
#   7: WORKERS         [default: SLURM_CPUS_PER_TASK or 4]
#   8: SCRIPT_DIR      [optional override for unified_pipeline directory]

usage() {
  cat <<USAGE
Usage:
  sbatch run_slide_job.sh IMAGE_PATH OUTPUT_ROOT [MODE] [MAGNIFICATION] [TILE_SIZE] [VENV_PATH] [WORKERS] [SCRIPT_DIR]

Examples:
  sbatch /exafs1/well/parkkinen/users/nfw313/final_scripts/unified_pipeline/run_slide_job.sh \
    /exafs1/well/parkkinen/users/nfw313/images/00-1006-A-IBA1.svs \
    /exafs1/well/parkkinen/users/nfw313/histoprocessor_outputs/unified_test/00-1006-A-IBA1 \
    all

  sbatch /exafs1/well/parkkinen/users/nfw313/final_scripts/unified_pipeline/run_slide_job.sh \
    /exafs1/well/parkkinen/users/nfw313/images/00-1006-A-IBA1.svs \
    /exafs1/well/parkkinen/users/nfw313/histoprocessor_outputs/unified_test/00-1006-A-IBA1 \
    stain+nuclei 40 512 /exafs1/well/parkkinen/users/nfw313/python-venvs/histomics-env 8
USAGE
}

log() {
  printf '%s [histoprocessor] %s\n' "$(date '+%Y-%m-%d %H:%M:%S')" "$*"
}

if [[ $# -lt 2 ]]; then
  echo "ERROR: IMAGE_PATH and OUTPUT_ROOT are required." >&2
  usage >&2
  exit 2
fi

IMAGE_PATH="$1"
OUTPUT_ROOT="$2"
MODE="${3:-all}"
MAG="${4:-40}"
TILE_SIZE="${5:-512}"
VENV_PATH="${6:-/well/parkkinen/users/nfw313/python-venvs/histomics-env}"
WORKERS="${7:-${SLURM_CPUS_PER_TASK:-4}}"
SCRIPT_DIR_OVERRIDE="${8:-}"

if [[ -n "$SCRIPT_DIR_OVERRIDE" ]]; then
  SCRIPT_DIR="$SCRIPT_DIR_OVERRIDE"
else
  SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
fi
RUNNER="$SCRIPT_DIR/run_slide.py"

if [[ ! -f "$RUNNER" ]]; then
  for candidate in \
    "/exafs1/well/parkkinen/users/nfw313/final_scripts/unified_pipeline" \
    "/well/parkkinen/users/nfw313/final_scripts/unified_pipeline"; do
    if [[ -f "$candidate/run_slide.py" ]]; then
      SCRIPT_DIR="$candidate"
      RUNNER="$SCRIPT_DIR/run_slide.py"
      break
    fi
  done
fi

if [[ ! -f "$RUNNER" ]]; then
  log "ERROR: unified runner not found: $RUNNER" >&2
  exit 1
fi

if [[ ! -f "$IMAGE_PATH" ]]; then
  log "ERROR: image not found: $IMAGE_PATH" >&2
  exit 1
fi

mkdir -p "$OUTPUT_ROOT"

log "Node: ${SLURMD_NODENAME:-unknown}"
log "Image: $IMAGE_PATH"
log "Output: $OUTPUT_ROOT"
log "Mode: $MODE"
log "Magnification: $MAG"
log "Tile size: $TILE_SIZE"
log "Workers: $WORKERS"

export OPENSLIDE_THREAD_COUNT=1
export VIPS_CONCURRENCY=1
export VIPS_MAX_MEM=256m
export VIPS_MAX_PAGES=1
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export PYTHONUNBUFFERED=1

# shellcheck disable=SC1090
source "$VENV_PATH/bin/activate"

stage_args=()
case "$MODE" in
  all)
    stage_args+=(--all)
    ;;
  clahe)
    stage_args+=(--clahe)
    ;;
  stain)
    stage_args+=(--stain)
    ;;
  nuclei)
    stage_args+=(--nuclei)
    ;;
  stain+nuclei)
    stage_args+=(--stain --nuclei)
    ;;
  *)
    log "ERROR: unsupported mode '$MODE'" >&2
    log "Valid modes: all, clahe, stain, nuclei, stain+nuclei" >&2
    exit 2
    ;;
esac

python -u "$RUNNER" \
  "$(realpath "$IMAGE_PATH")" \
  "$(realpath "$OUTPUT_ROOT")" \
  "${stage_args[@]}" \
  --magnification "$MAG" \
  --tile-size "$TILE_SIZE" \
  --workers "$WORKERS"

deactivate || true

log "Done: $IMAGE_PATH"
