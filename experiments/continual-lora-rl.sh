#!/usr/bin/env bash
set -euo pipefail

# Continual GRPO training over RG tasks:
# - Uses the dataset list in the base YAML as the task order
# - Trains each task sequentially (no mixing)
# - Splits `trainer.total_training_steps` evenly across tasks
# - Saves a checkpoint at the end of each task
#
# You can control per-task steps by setting:
#   continual.steps_per_task=50 continual.override_total_training_steps=true

export PATH="/usr/local/python3.11.13/bin:$PATH"

echo '/usr/local/Ascend/driver/lib64/driver' > /etc/ld.so.conf.d/ascend-driver.conf
ldconfig

# Ensures HCCL / Ascend runtime libs are visible for torch_npu & vllm_ascend.
# Shellcheck disable=SC1091
set +u
source /usr/local/Ascend/ascend-toolkit/set_env.sh
source /usr/local/Ascend/nnal/asdsip/set_env.sh
source /usr/local/Ascend/nnal/atb/set_env.sh
set -u

PYTHONUNBUFFERED=1 VLLM_ASCEND_ENABLE_NZ=0 HYDRA_FULL_ERROR=1 \
python reasoning-gym/training/train_grpo.py \
  --config-dir reasoning-gym/training/configs/inter_generalisation \
  --config-name algorithmic_qwen_3b \
  -- \
  "hydra.searchpath=[file:///verl/verl/trainer/config]" \
  actor_rollout_ref.actor.use_torch_compile=false \
  actor_rollout_ref.actor.fsdp_config.use_torch_compile=false \
  actor_rollout_ref.actor.ppo_mini_batch_size=16 \
  actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=8 \
  actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=8 \
  actor_rollout_ref.rollout.gpu_memory_utilization=0.6 \
  actor_rollout_ref.rollout.free_cache_engine=True \
  actor_rollout_ref.model.path=models/Qwen2.5-3B-Instruct \
  actor_rollout_ref.model.lora_rank=64 \
  actor_rollout_ref.model.lora_alpha=128 \
  actor_rollout_ref.actor.optim.lr=1e-4 \
  actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
  actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=16 \
  trainer.n_gpus_per_node=4 \
  trainer.project_name=llm_cl \
  trainer.device=npu \
  trainer.experiment_name=composite_algorithmic_qwen2.5_3b_lora_continual \
  continual.enabled=true \
  continual.save_ckpt_on_task_end=true \
  continual.eval_on_task_start=true \
  continual.eval_on_task_end=true \
  "$@" \
  2>&1 | tee algorithmic_qwen_3b_lora_continual.log
