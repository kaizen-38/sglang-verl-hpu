#!/usr/bin/env bash
# GRPO | Qwen3-4B-Base | FSDP training | SGLang rollout | Intel Gaudi 2 (HL-225)
#
# Derived from verl's examples/grpo_trainer/run_qwen3_4b_fsdp.sh, with the
# Gaudi-specific settings that were established the hard way (see gaudi/README.md).
#
# Success bar for step 1 — anything less means it is NOT learning:
#   critic/score/mean > -1.0 with varied rewards across the group
#   => nonzero critic/advantages => nonzero actor/pg_loss

set -xeuo pipefail

REPO_ROOT=${REPO_ROOT:-$(git -C "$(dirname "${BASH_SOURCE[0]}")" rev-parse --show-toplevel)}
VERL_DIR=${VERL_DIR:-${REPO_ROOT}/verl}

# ---- user-adjustable ----
MODEL_PATH=${MODEL_PATH:-/scratch/rarya124/models/Qwen3-4B-Base}
TRAIN_FILE=${TRAIN_FILE:-${REPO_ROOT}/gaudi/data/dapo_math/train.parquet}
TEST_FILE=${TEST_FILE:-${REPO_ROOT}/gaudi/data/dapo_math/val.parquet}

NNODES=${NNODES:-1}
# Habana permits ONE process per module, so training and rollout cannot share a
# module. Modules 0..N_HPUS_PER_NODE-1 train; the rollout engines get the modules
# starting at HPU_ROLLOUT_MODULE_OFFSET.
N_HPUS_PER_NODE=${N_HPUS_PER_NODE:-2}
HPU_ROLLOUT_MODULE_OFFSET=${HPU_ROLLOUT_MODULE_OFFSET:-2}

TRAIN_BATCH_SIZE=${TRAIN_BATCH_SIZE:-32}
PPO_MINI_BATCH_SIZE=${PPO_MINI_BATCH_SIZE:-16}
PPO_MICRO_BATCH_SIZE_PER_GPU=${PPO_MICRO_BATCH_SIZE_PER_GPU:-1}
LOG_PROB_MICRO_BATCH_SIZE_PER_GPU=${LOG_PROB_MICRO_BATCH_SIZE_PER_GPU:-1}

# PREFILL_BUCKET_MAX must be >= MAX_PROMPT_LENGTH, else a short prompt pads up to
# an oversized FusedSDPA graph and overruns the KV cache. Real DAPO prompts measure
# 93-251 tokens. Keep these two in step if you change either.
MAX_PROMPT_LENGTH=${MAX_PROMPT_LENGTH:-1024}
MAX_RESPONSE_LENGTH=${MAX_RESPONSE_LENGTH:-1024}

ACTOR_LR=${ACTOR_LR:-1e-6}
KL_LOSS_COEF=${KL_LOSS_COEF:-0.001}
ENTROPY_COEFF=${ENTROPY_COEFF:-0}

ROLLOUT_TP=${ROLLOUT_TP:-1}
ROLLOUT_N=${ROLLOUT_N:-8}

PROJECT_NAME=${PROJECT_NAME:-sglang-verl-hpu}
EXPERIMENT_NAME=${EXPERIMENT_NAME:-grpo_qwen3_4b_sglang_gaudi}
SAVE_FREQ=${SAVE_FREQ:-20}
TEST_FREQ=${TEST_FREQ:-5}
TOTAL_EPOCHS=${TOTAL_EPOCHS:-1}

RAY_NUM_CPUS=${RAY_NUM_CPUS:-72}
# ---- end user-adjustable ----

########################### Gaudi runtime ###########################

# Load the HPU platform + FSDP engine registration into every verl process
# (driver and Ray workers) without patching the verl source tree.
export PYTHONPATH=${REPO_ROOT}/gaudi:${VERL_DIR}${PYTHONPATH:+:${PYTHONPATH}}
export VERL_USE_EXTERNAL_MODULES=verl_hpu_plugin
export VERL_PLATFORM=hpu

# Eager mode. Lazy mode replays a recorded graph and is far more brittle here.
export PT_HPU_LAZY_MODE=${PT_HPU_LAZY_MODE:-0}
# Ray must not rewrite HABANA_VISIBLE_MODULES; the plugin owns module placement.
export RAY_EXPERIMENTAL_NOSET_HABANA_VISIBLE_MODULES=1
export HPU_ROLLOUT_MODULE_OFFSET

# SGLang HPU prefill bucketing. Every prompt pads up to the next bucket.
export SGLANG_HPU_PREFILL_BUCKET_MIN=${SGLANG_HPU_PREFILL_BUCKET_MIN:-256}
export SGLANG_HPU_PREFILL_BUCKET_STEP=${SGLANG_HPU_PREFILL_BUCKET_STEP:-256}
export SGLANG_HPU_PREFILL_BUCKET_MAX=${SGLANG_HPU_PREFILL_BUCKET_MAX:-${MAX_PROMPT_LENGTH}}

if [[ "${SGLANG_HPU_PREFILL_BUCKET_MAX}" -lt "${MAX_PROMPT_LENGTH}" ]]; then
    echo "SGLANG_HPU_PREFILL_BUCKET_MAX (${SGLANG_HPU_PREFILL_BUCKET_MAX}) < MAX_PROMPT_LENGTH (${MAX_PROMPT_LENGTH})" >&2
    exit 1
fi

########################### parameter arrays ###########################

DATA=(
    algorithm.adv_estimator=grpo
    data.train_files=${TRAIN_FILE}
    data.val_files=${TEST_FILE}
    # The parquet carries a raw-text `prompt` and a chat-formatted `source_prompt`.
    # Only `source_prompt` contains the "Answer: $Answer" instruction that the dapo
    # reward manager requires, and verl feeds prompt_key to apply_chat_template.
    data.prompt_key=source_prompt
    data.train_batch_size=${TRAIN_BATCH_SIZE}
    data.max_prompt_length=${MAX_PROMPT_LENGTH}
    data.max_response_length=${MAX_RESPONSE_LENGTH}
    data.filter_overlong_prompts=True
    data.truncation='error'
    algorithm.use_kl_in_reward=False
)

MODEL=(
    actor_rollout_ref.model.path=${MODEL_PATH}
    actor_rollout_ref.model.use_remove_padding=True
    actor_rollout_ref.model.enable_gradient_checkpointing=True
)

ACTOR=(
    actor_rollout_ref.actor.optim.lr=${ACTOR_LR}
    actor_rollout_ref.actor.ppo_mini_batch_size=${PPO_MINI_BATCH_SIZE}
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=${PPO_MICRO_BATCH_SIZE_PER_GPU}
    actor_rollout_ref.actor.use_kl_loss=True
    actor_rollout_ref.actor.kl_loss_coef=${KL_LOSS_COEF}
    actor_rollout_ref.actor.kl_loss_type=low_var_kl
    actor_rollout_ref.actor.entropy_coeff=${ENTROPY_COEFF}
    actor_rollout_ref.actor.fsdp_config.param_offload=False
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=False
    # HPU compiles a new Habana graph per distinct tensor shape. Dynamic batching
    # packs a different token count into every micro-batch, so it eventually dies
    # with "Graph duplication failed. synStatus=26". Fixed shapes only.
    actor_rollout_ref.actor.use_dynamic_bsz=False
)

ROLLOUT=(
    actor_rollout_ref.rollout.name=sglang
    actor_rollout_ref.rollout.tensor_model_parallel_size=${ROLLOUT_TP}
    actor_rollout_ref.rollout.n=${ROLLOUT_N}
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=${LOG_PROB_MICRO_BATCH_SIZE_PER_GPU}
    actor_rollout_ref.rollout.log_prob_use_dynamic_bsz=False
    actor_rollout_ref.rollout.enable_chunked_prefill=False
    actor_rollout_ref.rollout.free_cache_engine=True
)

REF=(
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=${LOG_PROB_MICRO_BATCH_SIZE_PER_GPU}
    actor_rollout_ref.ref.log_prob_use_dynamic_bsz=False
    actor_rollout_ref.ref.fsdp_config.param_offload=True
)

REWARD=(
    # Strict: requires a literal "Answer: <value>" line. Verified on CPU against
    # dataset ground truth — a correct, correctly-formatted answer scores +1.0.
    reward_model.reward_manager=dapo
)

TRAINER=(
    trainer.device=hpu
    trainer.critic_warmup=0
    trainer.logger='["console","wandb"]'
    trainer.project_name=${PROJECT_NAME}
    trainer.experiment_name=${EXPERIMENT_NAME}
    trainer.n_gpus_per_node=${N_HPUS_PER_NODE}
    trainer.nnodes=${NNODES}
    trainer.save_freq=${SAVE_FREQ}
    trainer.test_freq=${TEST_FREQ}
    trainer.total_epochs=${TOTAL_EPOCHS}
    # Slurm grants 72 CPUs via cgroup but the container sees the host's 152 and
    # mis-sizes its pool, hanging at bootstrap. Pin it.
    ray_kwargs.ray_init.num_cpus=${RAY_NUM_CPUS}
)

########################### launch ###########################

python3 -m verl.trainer.main_ppo \
    "${DATA[@]}" \
    "${MODEL[@]}" \
    "${ACTOR[@]}" \
    "${ROLLOUT[@]}" \
    "${REF[@]}" \
    "${REWARD[@]}" \
    "${TRAINER[@]}" \
    "$@"
