# verl 0.7.0.dev0, vlllm 0.11.0+empty, vllm_ascend 0.11.0rc1, torch 2.7.1, torch_npu 2.7.1
# Need to modify some config terms. Ref to changes in algirthmic_qwen_3b.yaml

PYTHONUNBUFFERED=1 VLLM_ASCEND_ENABLE_NZ=0 HYDRA_FULL_ERROR=1 \
python reasoning-gym/training/train_grpo.py \
  --config-dir reasoning-gym/training/configs/inter_generalisation \
  --config-name algorithmic_qwen_3b \
  -- \
  "hydra.searchpath=[file:///verl/verl/trainer/config]" \
  actor_rollout_ref.actor.ppo_mini_batch_size=16 \
  actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=8 \
  actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=8 \
  actor_rollout_ref.rollout.gpu_memory_utilization=0.6 \
  actor_rollout_ref.rollout.free_cache_engine=True \
  actor_rollout_ref.model.path=models/Qwen2.5-3B-Instruct \
  actor_rollout_ref.model.lora_rank=64 \
  actor_rollout_ref.model.lora_alpha=16 \
  actor_rollout_ref.actor.optim.lr=3e-6 \
  actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
  actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=16 \
  trainer.n_gpus_per_node=4 \
  trainer.project_name=llm_cl \
  trainer.device=npu \
  trainer.experiment_name=composite_algorithmic_qwen2.5_3b_lora \
  2>&1 | tee algorithmic_qwen_3b_lora.log