#!/usr/bin/env bash
#SBATCH --job-name=grpo_sglang_hpu
#SBATCH --partition=gaudi
#SBATCH --qos=class_gaudi
#SBATCH -A class_cse59827694spring2026
#SBATCH --gres=gpu:hl225:4
#SBATCH --cpus-per-task=72
#SBATCH --mem=240G
#SBATCH --time=08:00:00
#SBATCH --export=NONE
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

# 4 HPUs: modules 0,1 train (FSDP), modules 2,3 serve rollouts (SGLang).
# They cannot be colocated — Habana allows one process per module.

set -xeuo pipefail

REPO_ROOT=${REPO_ROOT:-/scratch/rarya124/sglang-verl-hpu}
SIF=${SIF:-/data/sse/gaudi/containers/vllm-gaudi.sif}
ENV_FILE=${ENV_FILE:-/scratch/rarya124/inoculation_her/.env}

mkdir -p "${REPO_ROOT}/logs"

# WANDB_API_KEY / HF_TOKEN. APPTAINERENV_* forwards them through to the container.
if [[ -f "${ENV_FILE}" ]]; then
    set -a
    source "${ENV_FILE}"
    set +a
    export APPTAINERENV_WANDB_API_KEY="${WANDB_API_KEY:-}"
    export APPTAINERENV_HF_TOKEN="${HF_TOKEN:-}"
fi

export APPTAINERENV_HF_HOME=/scratch/rarya124/hf_cache

apptainer exec \
    --bind /scratch:/scratch \
    --bind /data:/data \
    "${SIF}" \
    bash "${REPO_ROOT}/gaudi/recipe/grpo_qwen3_4b_gaudi/run_grpo_sglang_hpu.sh" "$@"
