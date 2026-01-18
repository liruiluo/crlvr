#!/usr/bin/env bash
set -euo pipefail

# Based on experiments/lora-rl.sh, but:
# - rollout uses HuggingFace/PEFT (`rollout.name=hf`) so token-wise MoE-LoRA forward is honored.
# - token-wise routing: topk -> softmax(within topk).

ROOT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

# Ascend / NNAL env
set +e
set +u
source /usr/local/Ascend/ascend-toolkit/set_env.sh >/dev/null 2>&1 || true
export LD_LIBRARY_PATH=/usr/local/Ascend/driver/lib64/driver:${LD_LIBRARY_PATH:-}
source /usr/local/Ascend/nnal/atb/set_env.sh --cxx_abi=0 >/dev/null 2>&1 || true
source /usr/local/Ascend/nnal/asdsip/set_env.sh >/dev/null 2>&1 || true
set -u
set -e

PYTHON_BIN="${PYTHON_BIN:-/usr/local/python3.11.13/bin/python}"

export PYTHONUNBUFFERED=1
export HYDRA_FULL_ERROR=1
export VLLM_ASCEND_ENABLE_NZ=0

# Reduce CPU oversubscription (Ray heartbeats)
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export TOKENIZERS_PARALLELISM=true
export RAY_DISABLE_DOCKER_CPU_WARNING=1
export RAY_DISABLE_DASHBOARD=1

# Optional overrides for quick smoke / long runs.
TOTAL_TRAINING_STEPS="${TOTAL_TRAINING_STEPS:-500}"
TOTAL_EPOCHS="${TOTAL_EPOCHS:-1}"
SAVE_FREQ="${SAVE_FREQ:-100}"
TEST_FREQ="${TEST_FREQ:-100}"
TRAINER_PROJECT_NAME="${TRAINER_PROJECT_NAME:-}"
TRAINER_EXPERIMENT_NAME="${TRAINER_EXPERIMENT_NAME:-}"
TRAINER_LOGGER_OVERRIDE="${TRAINER_LOGGER_OVERRIDE:-}"
TRAINER_BALANCE_BATCH="${TRAINER_BALANCE_BATCH:-}"
LOG_FILE="${LOG_FILE:-algorithmic_qwen_3b_moe_lora_hf_rollout.log}"

# Optional training/rollout overrides (keep empty to use config/script defaults).
N_GPUS_PER_NODE="${N_GPUS_PER_NODE:-}"
DATASET_SIZE="${DATASET_SIZE:-}"
MAX_PROMPT_LENGTH="${MAX_PROMPT_LENGTH:-}"
MAX_RESPONSE_LENGTH="${MAX_RESPONSE_LENGTH:-}"
TRAIN_BATCH_SIZE="${TRAIN_BATCH_SIZE:-}"
ROLLOUT_N="${ROLLOUT_N:-}"
PPO_MINI_BATCH_SIZE="${PPO_MINI_BATCH_SIZE:-}"
PPO_MICRO_BATCH_SIZE_PER_GPU="${PPO_MICRO_BATCH_SIZE_PER_GPU:-}"
ROLLOUT_LOGPROB_MICRO_BATCH_SIZE_PER_GPU="${ROLLOUT_LOGPROB_MICRO_BATCH_SIZE_PER_GPU:-}"
REF_LOGPROB_MICRO_BATCH_SIZE_PER_GPU="${REF_LOGPROB_MICRO_BATCH_SIZE_PER_GPU:-}"
ROLLOUT_CALCULATE_LOG_PROBS="${ROLLOUT_CALCULATE_LOG_PROBS:-}"
ACTOR_USE_KL_LOSS="${ACTOR_USE_KL_LOSS:-}"

# Torch compile (can be unstable with multi-rank + NPU). Leave empty to use defaults.
ACTOR_USE_TORCH_COMPILE="${ACTOR_USE_TORCH_COMPILE:-}"
REF_USE_TORCH_COMPILE="${REF_USE_TORCH_COMPILE:-}"

# Avoid HCCL/dist port collisions from stale or concurrent jobs on the same node.
MASTER_ADDR="${MASTER_ADDR:-$(hostname -I 2>/dev/null | awk '{print $1}')}"
if [[ -z "${MASTER_ADDR}" ]]; then
  MASTER_ADDR="127.0.0.1"
fi
if [[ -z "${MASTER_PORT:-}" ]]; then
  MASTER_PORT="$("$PYTHON_BIN" - <<'PY'
import socket
s = socket.socket()
s.bind(("", 0))
print(s.getsockname()[1])
s.close()
PY
)"
fi
export MASTER_ADDR MASTER_PORT

if [[ -z "${HCCL_HOST_SOCKET_PORT_RANGE:-}" || -z "${HCCL_NPU_SOCKET_PORT_RANGE:-}" ]]; then
  HCCL_PORT_BASE="$("$PYTHON_BIN" - <<'PY'
import random
print(random.randrange(20000, 52000, 4000))
PY
)"
  export HCCL_HOST_SOCKET_PORT_RANGE="${HCCL_PORT_BASE}-$((HCCL_PORT_BASE + 1999))"
  export HCCL_NPU_SOCKET_PORT_RANGE="$((HCCL_PORT_BASE + 2000))-$((HCCL_PORT_BASE + 3999))"
fi

EXTRA_ARGS=(
  "trainer.total_training_steps=${TOTAL_TRAINING_STEPS}"
  "trainer.total_epochs=${TOTAL_EPOCHS}"
  "trainer.save_freq=${SAVE_FREQ}"
  "trainer.test_freq=${TEST_FREQ}"
)
if [[ -n "${TRAINER_PROJECT_NAME}" ]]; then
  EXTRA_ARGS+=("trainer.project_name=${TRAINER_PROJECT_NAME}")
fi
if [[ -n "${TRAINER_EXPERIMENT_NAME}" ]]; then
  EXTRA_ARGS+=("trainer.experiment_name=${TRAINER_EXPERIMENT_NAME}")
fi
if [[ -n "${TRAINER_LOGGER_OVERRIDE}" ]]; then
  EXTRA_ARGS+=("trainer.logger=${TRAINER_LOGGER_OVERRIDE}")
fi
if [[ -n "${TRAINER_BALANCE_BATCH}" ]]; then
  EXTRA_ARGS+=("trainer.balance_batch=${TRAINER_BALANCE_BATCH}")
fi
if [[ -n "${N_GPUS_PER_NODE}" ]]; then
  EXTRA_ARGS+=("trainer.n_gpus_per_node=${N_GPUS_PER_NODE}")
fi
if [[ -n "${DATASET_SIZE}" ]]; then
  EXTRA_ARGS+=("reasoning_gym.dataset_size=${DATASET_SIZE}")
fi
if [[ -n "${MAX_PROMPT_LENGTH}" ]]; then
  EXTRA_ARGS+=("data.max_prompt_length=${MAX_PROMPT_LENGTH}")
fi
if [[ -n "${MAX_RESPONSE_LENGTH}" ]]; then
  EXTRA_ARGS+=("data.max_response_length=${MAX_RESPONSE_LENGTH}")
fi
if [[ -n "${TRAIN_BATCH_SIZE}" ]]; then
  EXTRA_ARGS+=("data.train_batch_size=${TRAIN_BATCH_SIZE}")
fi
if [[ -n "${ROLLOUT_N}" ]]; then
  EXTRA_ARGS+=("actor_rollout_ref.rollout.n=${ROLLOUT_N}")
fi
if [[ -n "${PPO_MINI_BATCH_SIZE}" ]]; then
  EXTRA_ARGS+=("actor_rollout_ref.actor.ppo_mini_batch_size=${PPO_MINI_BATCH_SIZE}")
fi
if [[ -n "${PPO_MICRO_BATCH_SIZE_PER_GPU}" ]]; then
  EXTRA_ARGS+=("actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=${PPO_MICRO_BATCH_SIZE_PER_GPU}")
fi
if [[ -n "${ROLLOUT_LOGPROB_MICRO_BATCH_SIZE_PER_GPU}" ]]; then
  EXTRA_ARGS+=("actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=${ROLLOUT_LOGPROB_MICRO_BATCH_SIZE_PER_GPU}")
fi
if [[ -n "${REF_LOGPROB_MICRO_BATCH_SIZE_PER_GPU}" ]]; then
  EXTRA_ARGS+=("actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=${REF_LOGPROB_MICRO_BATCH_SIZE_PER_GPU}")
fi
if [[ -n "${ROLLOUT_CALCULATE_LOG_PROBS}" ]]; then
  EXTRA_ARGS+=("actor_rollout_ref.rollout.calculate_log_probs=${ROLLOUT_CALCULATE_LOG_PROBS}")
fi
if [[ -n "${ACTOR_USE_KL_LOSS}" ]]; then
  EXTRA_ARGS+=("actor_rollout_ref.actor.use_kl_loss=${ACTOR_USE_KL_LOSS}")
fi
if [[ -n "${ACTOR_USE_TORCH_COMPILE}" ]]; then
  EXTRA_ARGS+=("actor_rollout_ref.actor.use_torch_compile=${ACTOR_USE_TORCH_COMPILE}")
fi
if [[ -n "${REF_USE_TORCH_COMPILE}" ]]; then
  EXTRA_ARGS+=("actor_rollout_ref.ref.use_torch_compile=${REF_USE_TORCH_COMPILE}")
fi

"$PYTHON_BIN" reasoning-gym/training/train_grpo.py \
  --config-dir reasoning-gym/training/configs/inter_generalisation \
  --config-name algorithmic_qwen_3b \
  -- \
  "hydra.searchpath=[file:///verl/verl/trainer/config]" \
  actor_rollout_ref.model.external_lib=crlvr_external_lib \
  actor_rollout_ref.rollout.name=hf \
  actor_rollout_ref.actor.ppo_mini_batch_size=16 \
  actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=8 \
  actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=8 \
  actor_rollout_ref.rollout.gpu_memory_utilization=0.6 \
  actor_rollout_ref.rollout.free_cache_engine=True \
  actor_rollout_ref.rollout.layered_summon=true \
  actor_rollout_ref.model.path=models/Qwen2.5-3B-Instruct \
  actor_rollout_ref.model.lora_rank=64 \
  actor_rollout_ref.model.lora_alpha=128 \
  actor_rollout_ref.model.override_config.moe_lora_enable=true \
  actor_rollout_ref.model.override_config.moe_lora_num_experts=4 \
  actor_rollout_ref.model.override_config.moe_lora_top_k=2 \
  actor_rollout_ref.model.override_config.moe_lora_router_init_std=0.02 \
  actor_rollout_ref.actor.optim.lr=1e-4 \
  actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
  actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=16 \
  trainer.n_gpus_per_node=4 \
  trainer.project_name=llm_cl \
  trainer.device=npu \
  trainer.experiment_name=composite_algorithmic_qwen2.5_3b_moe_lora_hf_rollout \
  "${EXTRA_ARGS[@]}" \
  2>&1 | tee "$LOG_FILE"
