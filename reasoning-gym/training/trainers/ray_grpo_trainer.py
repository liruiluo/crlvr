# Adapted version of Bytedance code:
# https://github.com/volcengine/verl/blob/a65c9157bc0b85b64cd753de19f94e80a11bd871/verl/trainer/main_ppo.py

import gc
import json
import math
import uuid
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

import numpy as np
import torch
from omegaconf import OmegaConf, open_dict
from rewards import reward_registry
from torchdata.stateful_dataloader import StatefulDataLoader
from utils import ReasoningGymDataset
from verl import DataProto
from verl.trainer.ppo.ray_trainer import (
    AdvantageEstimator,
    RayPPOTrainer,
    # _timer,
    apply_kl_penalty,
    compute_advantage,
    compute_data_metrics,
    compute_timing_metrics,
    reduce_metrics,
)
from verl.trainer.ppo.utils import Role, WorkerType, need_critic, need_reference_policy, need_reward_model
from verl.utils.debug import marked_timer
_timer = marked_timer
from verl.utils.dataset.rl_dataset import collate_fn as verl_rl_collate_fn
from verl.single_controller.ray.base import create_colocated_worker_cls
from verl.single_controller.ray import RayClassWithInitArgs, RayResourcePool, RayWorkerGroup
from reasoning_gym.utils import extract_answer
from verl.protocol import pad_dataproto_to_divisor, unpad_dataproto

import reasoning_gym
from reasoning_gym.composite import DatasetSpec


@dataclass(frozen=True)
class ContinualTaskSpec:
    name: str
    datasets: dict[str, dict[str, Any]]  # dataset_name -> {"weight": float, "config": dict}
    start_step: int  # inclusive, 1-based
    end_step: int  # inclusive, 1-based

class RayGRPOTrainer(RayPPOTrainer):
    def __init__(
        self,
        config,
        tokenizer,
        train_dataset: ReasoningGymDataset,
        val_dataset: ReasoningGymDataset,
        role_worker_mapping: dict,
        resource_pool_manager,
        ray_worker_group_cls,
        max_output_length: int = 1024,
    ):
        # Keep a reference early (before base class init) so continual mode helpers can use it.
        self.config = config
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        self.max_output_length = max_output_length

        if config.curriculum.enabled:
            self.last_k = config.curriculum.last_k
        else:
            self.last_k = None

        self.continual_enabled = False
        self.continual_tasks: list[ContinualTaskSpec] = []
        self.continual_current_task_idx: Optional[int] = None
        self.checkpoint_saved_steps: set[int] = set()

        continual_cfg = getattr(config, "continual", None)
        if continual_cfg is not None and bool(continual_cfg.get("enabled", False)):
            if config.curriculum.enabled:
                raise ValueError("continual.enabled=True is not supported with curriculum.enabled=True")
            self.continual_enabled = True
            self.continual_tasks = self.build_continual_schedule(config, continual_cfg)

            if self.continual_tasks:
                # Apply first task before base trainer initialization (dataloaders/samplers).
                self.apply_continual_task(task_idx=0)
                self.continual_current_task_idx = 0
                self.dump_continual_schedule()

        self.reward_functions = []
        if hasattr(config, "reward") and hasattr(config.reward, "secondary_rewards"):
            for func_config in config.reward.secondary_rewards:
                func_name = func_config.name
                scaling_factor = func_config.get("scaling_factor", 1.0)
                func = reward_registry.get(func_name)
                if func:
                    # Store both function and its arguments
                    self.reward_functions.append(
                        {
                            "function": func,
                            "name": func_name,
                            "scaling_factor": scaling_factor,
                            "kwargs": func_config.get("kwargs", {}),
                        }
                    )

        train_reward_fn = lambda data: self._score_output(data, num_examine=0)
        val_reward_fn = lambda data: self._score_output(data, num_examine=1)

        super().__init__(
            config,
            tokenizer,
            role_worker_mapping,
            resource_pool_manager,
            ray_worker_group_cls,
            reward_fn=train_reward_fn,
            val_reward_fn=val_reward_fn,
        )

    def dataset_cfg_to_dict(self, dataset_cfg: Any) -> dict[str, Any]:
        weight = float(dataset_cfg.get("weight", 1.0))
        cfg = {}
        if "config" in dataset_cfg and dataset_cfg.config is not None:
            cfg = OmegaConf.to_container(dataset_cfg.config, resolve=True)
            if cfg is None:
                cfg = {}
        return {"weight": weight, "config": dict(cfg)}

    def build_continual_schedule(self, config, continual_cfg) -> list[ContinualTaskSpec]:
        base_datasets_cfg = config.reasoning_gym.datasets
        base_dataset_defs: dict[str, dict[str, Any]] = {
            ds_name: self.dataset_cfg_to_dict(ds_cfg) for ds_name, ds_cfg in base_datasets_cfg.items()
        }

        tasks_cfg = continual_cfg.get("tasks", None)
        if tasks_cfg:
            raw_tasks = tasks_cfg
        else:
            # Default: one task per dataset in config order.
            raw_tasks = list(base_dataset_defs.keys())

        normalized_tasks: list[dict[str, Any]] = []
        for item in raw_tasks:
            if isinstance(item, str):
                dataset_name = item
                if dataset_name not in base_dataset_defs:
                    raise ValueError(
                        f"continual.tasks references unknown dataset '{dataset_name}' "
                        f"(not in reasoning_gym.datasets)"
                    )
                normalized_tasks.append(
                    {"name": dataset_name, "datasets": {dataset_name: base_dataset_defs[dataset_name]}}
                )
                continue

            # DictConfig behaves like a dict for .get()/iteration.
            task_name = item.get("name", None)
            datasets = item.get("datasets", None)
            dataset_single = item.get("dataset", None)

            if datasets is None and dataset_single is not None:
                datasets = {dataset_single: item.get("dataset_config", {})}

            if datasets is None:
                raise ValueError("continual.tasks items must be string, or have 'dataset' or 'datasets' fields")

            task_datasets: dict[str, dict[str, Any]] = {}
            for ds_name, ds_override in dict(datasets).items():
                if ds_name not in base_dataset_defs:
                    raise ValueError(
                        f"continual.tasks references unknown dataset '{ds_name}' "
                        f"(not in reasoning_gym.datasets)"
                    )
                base_def = deepcopy(base_dataset_defs[ds_name])

                # Optional overrides.
                if ds_override is None:
                    task_datasets[ds_name] = base_def
                    continue

                if isinstance(ds_override, dict) or hasattr(ds_override, "get"):
                    override_weight = ds_override.get("weight", None)
                    override_config = ds_override.get("config", None)
                    if override_weight is not None:
                        base_def["weight"] = float(override_weight)
                    if override_config is not None:
                        base_def["config"].update(dict(override_config))
                task_datasets[ds_name] = base_def

            if not task_name:
                if len(task_datasets) == 1:
                    task_name = next(iter(task_datasets.keys()))
                else:
                    task_name = "phase_" + "_".join(task_datasets.keys())

            normalized_tasks.append({"name": str(task_name), "datasets": task_datasets, "steps": item.get("steps")})

        if not normalized_tasks:
            return []

        steps_per_task = continual_cfg.get("steps_per_task", None)
        override_total = bool(continual_cfg.get("override_total_training_steps", False))

        if steps_per_task is not None:
            steps_per_task = int(steps_per_task)
            if steps_per_task <= 0:
                raise ValueError("continual.steps_per_task must be positive")
            total_steps = steps_per_task * len(normalized_tasks)
            if override_total:
                with open_dict(config):
                    config.trainer.total_training_steps = total_steps
        else:
            total_steps = int(config.trainer.total_training_steps)

        # Allocate per-task steps.
        steps_list: list[int] = []
        if steps_per_task is not None:
            steps_list = [steps_per_task] * len(normalized_tasks)
        else:
            explicit_steps = [t.get("steps") for t in normalized_tasks]
            has_any_explicit = any(s is not None for s in explicit_steps)

            if has_any_explicit:
                fixed = [int(s) if s is not None else None for s in explicit_steps]
                if any(s is not None and s <= 0 for s in fixed):
                    raise ValueError("continual.tasks[*].steps must be positive when provided")

                sum_fixed = sum(s for s in fixed if s is not None)
                if sum_fixed > total_steps:
                    raise ValueError(
                        f"Sum of continual.tasks[*].steps ({sum_fixed}) exceeds trainer.total_training_steps ({total_steps})"
                    )

                missing = [i for i, s in enumerate(fixed) if s is None]
                remaining = total_steps - sum_fixed
                if missing and remaining <= 0:
                    raise ValueError(
                        "Some continual.tasks[*].steps are missing, but no remaining steps to allocate "
                        f"(sum_fixed={sum_fixed}, total={total_steps})"
                    )

                if missing:
                    base = remaining // len(missing)
                    remainder = remaining % len(missing)
                    if base <= 0:
                        raise ValueError(
                            f"Remaining steps ({remaining}) is too small for {len(missing)} tasks without explicit steps"
                        )
                    for j, idx in enumerate(missing):
                        fixed[idx] = base + (1 if j < remainder else 0)

                steps_list = [int(s) for s in fixed]  # now fully filled
            else:
                base = total_steps // len(normalized_tasks)
                remainder = total_steps % len(normalized_tasks)
                if base <= 0:
                    raise ValueError(
                        f"trainer.total_training_steps ({total_steps}) is too small for {len(normalized_tasks)} tasks"
                    )
                for i in range(len(normalized_tasks)):
                    steps_list.append(base + (1 if i < remainder else 0))

        schedule: list[ContinualTaskSpec] = []
        current = 1
        for task, n_steps in zip(normalized_tasks, steps_list, strict=True):
            end = current + int(n_steps) - 1
            schedule.append(
                ContinualTaskSpec(
                    name=str(task["name"]),
                    datasets=task["datasets"],
                    start_step=current,
                    end_step=end,
                )
            )
            current = end + 1

        return schedule

    def continual_task_index_for_step(self, global_step: int) -> Optional[int]:
        if not self.continual_tasks:
            return None
        for i, task in enumerate(self.continual_tasks):
            if task.start_step <= global_step <= task.end_step:
                return i
        return None

    def make_task_data_source(self, task: ContinualTaskSpec, seed: int):
        dataset_size = int(self.config.reasoning_gym.dataset_size)
        dataset_specs = [
            DatasetSpec(name=ds_name, weight=float(ds_def["weight"]), config=dict(ds_def["config"]))
            for ds_name, ds_def in task.datasets.items()
        ]
        return reasoning_gym.create_dataset("composite", seed=seed, size=dataset_size, datasets=dataset_specs)

    def apply_continual_task(self, task_idx: int) -> None:
        task = self.continual_tasks[task_idx]
        print(
            f"[continual] switch task={task.name} idx={task_idx} steps={task.start_step}-{task.end_step} "
            f"datasets={list(task.datasets.keys())}"
        )
        # Match train_grpo.py seeding convention, but keep seeds task-specific for diversity.
        seed_offset = 1000 * task_idx
        train_source = self.make_task_data_source(task, seed=1 + seed_offset)
        val_source = self.make_task_data_source(task, seed=2 + seed_offset)
        self.train_dataset.data = train_source
        self.val_dataset.data = val_source

    def dump_continual_schedule(self) -> None:
        base_dir = Path(str(self.config.trainer.default_local_dir))
        base_dir.mkdir(parents=True, exist_ok=True)
        payload = {
            "total_training_steps": int(self.config.trainer.total_training_steps),
            "tasks": [
                {
                    "name": t.name,
                    "start_step": t.start_step,
                    "end_step": t.end_step,
                    "datasets": t.datasets,
                }
                for t in self.continual_tasks
            ],
        }
        (base_dir / "continual_schedule.json").write_text(json.dumps(payload, indent=2, ensure_ascii=False))

    def maybe_save_checkpoint(self, reason: str) -> bool:
        step = int(self.global_steps)
        did_save = step not in self.checkpoint_saved_steps
        if did_save:
            self.checkpoint_saved_steps.add(step)
            print(f"[continual] save checkpoint step={step} reason={reason}")
            self._save_checkpoint()
        else:
            print(f"[continual] checkpoint already saved step={step}; record reason={reason}")

        if self.continual_enabled:
            base_dir = Path(str(self.config.trainer.default_local_dir))
            base_dir.mkdir(parents=True, exist_ok=True)
            path = base_dir / "continual_task_checkpoints.jsonl"
            record = {"global_step": step, "reason": reason, "task": None, "saved": did_save}
            if self.continual_current_task_idx is not None and 0 <= self.continual_current_task_idx < len(
                self.continual_tasks
            ):
                record["task"] = self.continual_tasks[self.continual_current_task_idx].name
            with path.open("a", encoding="utf-8") as f:
                f.write(json.dumps(record, ensure_ascii=False) + "\n")

        return did_save

    def rollout_pad_divisor(self) -> int:
        divisor = int(getattr(self.actor_rollout_wg, "world_size", 1))
        if self.async_rollout_mode:
            workers = getattr(self.async_rollout_manager, "agent_loop_workers", None)
            if workers is not None:
                divisor = math.lcm(divisor, len(workers))
        return max(divisor, 1)

    def init_workers(self):
        """Initialize distributed training workers using Ray backend.

        Creates:
        1. Ray resource pools from configuration
        2. Worker groups for each role (actor, critic, etc.)
        """
        self.resource_pool_manager.create_resource_pool()

        self.resource_pool_to_cls = {pool: {} for pool in self.resource_pool_manager.resource_pool_dict.values()}

        # create actor and rollout
        actor_role = Role.ActorRolloutRef if Role.ActorRolloutRef in self.role_worker_mapping else Role.ActorRollout
        if self.hybrid_engine:
            resource_pool = self.resource_pool_manager.get_resource_pool(actor_role)
            actor_rollout_cls = RayClassWithInitArgs(
                cls=self.role_worker_mapping[actor_role],
                config=self.config.actor_rollout_ref,
                role=str(actor_role),
            )
            self.resource_pool_to_cls[resource_pool][str(actor_role)] = actor_rollout_cls
        else:
            raise NotImplementedError

        # create critic
        if self.use_critic:
            resource_pool = self.resource_pool_manager.get_resource_pool(Role.Critic)
            critic_cfg = omega_conf_to_dataclass(self.config.critic)
            critic_cls = RayClassWithInitArgs(cls=self.role_worker_mapping[Role.Critic], config=critic_cfg)
            self.resource_pool_to_cls[resource_pool][str(Role.Critic)] = critic_cls

        # create reference policy if needed
        if self.use_reference_policy and Role.RefPolicy in self.role_worker_mapping:
            resource_pool = self.resource_pool_manager.get_resource_pool(Role.RefPolicy)
            ref_policy_cls = RayClassWithInitArgs(
                self.role_worker_mapping[Role.RefPolicy],
                config=self.config.actor_rollout_ref,
                role=str(Role.RefPolicy),
            )
            self.resource_pool_to_cls[resource_pool][str(Role.RefPolicy)] = ref_policy_cls

        # create a reward model if reward_fn is None
        if self.use_rm:
            # we create a RM here
            resource_pool = self.resource_pool_manager.get_resource_pool(Role.RewardModel)
            rm_cls = RayClassWithInitArgs(self.role_worker_mapping[Role.RewardModel], config=self.config.reward_model)
            self.resource_pool_to_cls[resource_pool][str(Role.RewardModel)] = rm_cls

        # initialize WorkerGroup
        # NOTE: if you want to use a different resource pool for each role, which can support different parallel size,
        # you should not use `create_colocated_worker_cls`.
        # Instead, directly pass different resource pool to different worker groups.
        # See https://github.com/volcengine/verl/blob/master/examples/ray/tutorial.ipynb for more information.
        all_wg = {}
        wg_kwargs = {}  # Setting up kwargs for RayWorkerGroup
        if OmegaConf.select(self.config.trainer, "ray_wait_register_center_timeout") is not None:
            wg_kwargs["ray_wait_register_center_timeout"] = self.config.trainer.ray_wait_register_center_timeout
        if OmegaConf.select(self.config.global_profiler, "steps") is not None:
            wg_kwargs["profile_steps"] = OmegaConf.select(self.config.global_profiler, "steps")
            # Only require nsight worker options when tool is nsys
            if OmegaConf.select(self.config.global_profiler, "tool") == "nsys":
                assert (
                    OmegaConf.select(self.config.global_profiler.global_tool_config.nsys, "worker_nsight_options")
                    is not None
                ), "worker_nsight_options must be set when using nsys with profile_steps"
                wg_kwargs["worker_nsight_options"] = OmegaConf.to_container(
                    OmegaConf.select(self.config.global_profiler.global_tool_config.nsys, "worker_nsight_options")
                )
        wg_kwargs["device_name"] = self.device_name

        for resource_pool, class_dict in self.resource_pool_to_cls.items():
            worker_dict_cls = create_colocated_worker_cls(class_dict=class_dict)
            wg_dict = self.ray_worker_group_cls(
                resource_pool=resource_pool,
                ray_cls_with_init=worker_dict_cls,
                **wg_kwargs,
            )
            spawn_wg = wg_dict.spawn(prefix_set=class_dict.keys())
            all_wg.update(spawn_wg)

        if self.use_critic:
            self.critic_wg = all_wg[str(Role.Critic)]
            self.critic_wg.init_model()

        if self.use_reference_policy and not self.ref_in_actor:
            if str(Role.RefPolicy) in all_wg:
                self.ref_policy_wg = all_wg[str(Role.RefPolicy)]
                self.ref_policy_wg.init_model()
            else:
                # Model engine: ActorRolloutRefWorker
                assert str(Role.ActorRolloutRef) in all_wg, f"{all_wg.keys()=}"
                self.ref_policy_wg = all_wg[str(Role.ActorRolloutRef)]

        self.rm_wg = None
        # initalization of rm_wg will be deprecated in the future
        if self.use_rm:
            self.rm_wg = all_wg[str(Role.RewardModel)]
            self.rm_wg.init_model()

        # we should create rollout at the end so that vllm can have a better estimation of kv cache memory
        self.actor_rollout_wg = all_wg[str(actor_role)]
        self.actor_rollout_wg.init_model()

        # create async rollout manager and request scheduler
        self.async_rollout_mode = False
        if self.config.actor_rollout_ref.rollout.mode == "async":
            from .agent_loop import AgentLoopManagerWithoutReward

            self.async_rollout_mode = True
            if self.config.reward_model.enable and self.config.reward_model.enable_resource_pool:
                rm_resource_pool = self.resource_pool_manager.get_resource_pool(Role.RewardModel)
            else:
                rm_resource_pool = None

            self.async_rollout_manager = AgentLoopManagerWithoutReward(
                config=self.config,
                worker_group=self.actor_rollout_wg,
                rm_resource_pool=rm_resource_pool,
            )

    def _validate(self):
        reward_tensor_lst = []
        data_source_lst = []

        # Lists to collect samples for the table
        sample_inputs = []
        sample_outputs = []
        sample_scores = []

        for test_data in self.val_dataloader:
            test_batch = DataProto.from_single_dict(test_data)

            # we only do validation on rule-based rm
            if self.config.reward_model.enable and test_batch[0].non_tensor_batch['reward_model']['style'] == 'model':
                return {}

            # Store original inputs
            input_ids = test_batch.batch['input_ids']
            input_texts = [self.tokenizer.decode(ids, skip_special_tokens=True) for ids in input_ids]
            sample_inputs.extend(input_texts)

            # if 'multi_modal_inputs' in test_batch.non_tensor_batch.keys():
            #     test_gen_batch = test_batch.pop(
            #         batch_keys=['input_ids', 'attention_mask', 'position_ids'],
            #         non_tensor_batch_keys=['raw_prompt_ids', 'multi_modal_data', 'multi_modal_inputs'],
            #     )
            # else:
            #     test_gen_batch = test_batch.pop(
            #         batch_keys=['input_ids', 'attention_mask', 'position_ids'],
            #         non_tensor_batch_keys=['raw_prompt_ids'],
            #     )
            test_gen_batch = self._get_gen_batch(test_batch)
            test_gen_batch.meta_info = {
                'eos_token_id': self.tokenizer.eos_token_id,
                'pad_token_id': self.tokenizer.pad_token_id,
                'recompute_log_prob': False,
                'do_sample': self.config.actor_rollout_ref.rollout.val_kwargs.do_sample,
                'validate': True,
            }
            print(f'test_gen_batch meta info: {test_gen_batch.meta_info}')

            # pad to be divisible by dp_size
            test_gen_batch_padded, pad_size = pad_dataproto_to_divisor(test_gen_batch, self.rollout_pad_divisor())
            if not self.async_rollout_mode:
                test_output_gen_batch_padded = self.actor_rollout_wg.generate_sequences(test_gen_batch_padded)
            else:
                test_output_gen_batch_padded = self.async_rollout_manager.generate_sequences(test_gen_batch_padded)
            # unpad
            test_output_gen_batch = unpad_dataproto(test_output_gen_batch_padded, pad_size=pad_size)
            print('validation generation end')

            # Store generated outputs
            output_ids = test_output_gen_batch.batch['responses']
            output_texts = [self.tokenizer.decode(ids, skip_special_tokens=True) for ids in output_ids]
            sample_outputs.extend(output_texts)

            test_batch = test_batch.union(test_output_gen_batch)

            # evaluate using reward_function
            reward_tensor = self.val_reward_fn(test_batch)

            # Store scores
            scores = reward_tensor.sum(-1).cpu().tolist()
            sample_scores.extend(scores)

            reward_tensor_lst.append(reward_tensor)
            data_source_lst.append(test_batch.non_tensor_batch.get('data_source', ['unknown'] * reward_tensor.shape[0]))

        self._maybe_log_val_generations(inputs=sample_inputs, outputs=sample_outputs, scores=sample_scores)

        reward_tensor = torch.cat(reward_tensor_lst, dim=0).sum(-1).cpu()  # (batch_size,)
        data_sources = np.concatenate(data_source_lst, axis=0)

            # evaluate test_score based on data source
        data_source_reward = {}
        for i in range(reward_tensor.shape[0]):
            data_source = data_sources[i]
            if data_source not in data_source_reward:
                data_source_reward[data_source] = []
            data_source_reward[data_source].append(reward_tensor[i].item())

        metric_dict = {}
        for data_source, rewards in data_source_reward.items():
            metric_dict[f'val/test_score/{data_source}'] = np.mean(rewards)

        return metric_dict

    def _score_output(self, data: DataProto, num_examine: int = 0) -> torch.Tensor:
        reward_tensor = torch.zeros_like(data.batch["responses"], dtype=torch.float32)

        num_printed = 0
        for i in range(len(data)):
            data_item = data[i]  # DataProtoItem

            prompt_ids = data_item.batch["prompts"]  # tokenized prompts
            prompt_length = prompt_ids.shape[-1]

            valid_prompt_length = data_item.batch["attention_mask"][:prompt_length].sum()
            valid_prompt_ids = prompt_ids[-valid_prompt_length:]

            response_ids = data_item.batch["responses"]
            valid_response_length = data_item.batch["attention_mask"][prompt_length:].sum()
            valid_response_ids = response_ids[:valid_response_length]

            # decode
            prompt_str = self.tokenizer.decode(valid_prompt_ids)
            response_str = self.tokenizer.decode(valid_response_ids)
            sequences_str = prompt_str + response_str

            index = data_item.non_tensor_batch["index"]
            correctness_score = self._compute_correctness_score(
                solution_str=response_str,
                index=index,
            )
            if self.config.reward.use_accuracy:
                reward_components = {"correctness": correctness_score}
                total_reward = correctness_score
            else:
                reward_components = {}
                total_reward = 0

            for reward_fn in self.reward_functions:
                func = reward_fn["function"]
                name = reward_fn["name"]
                scaling_factor = reward_fn["scaling_factor"]
                kwargs = reward_fn["kwargs"]
                if name == "cosine":
                    is_correct = correctness_score == 1.0
                    reward = func(response_str, scaling_factor, is_correct=is_correct, **kwargs)
                elif name == "length":
                    reward = func(response_str, scaling_factor, correctness_score=correctness_score, **kwargs)
                else:
                    reward = func(response_str, scaling_factor, **kwargs)
                reward_components[name] = reward
                total_reward += reward

            reward_tensor[i, valid_response_length - 1] = total_reward

            if num_printed < num_examine:
                components = ", ".join([f"{k}={v:.2f}" for k, v in reward_components.items()])
                print(f"(score={total_reward}, seq={sequences_str}, response={response_str})")
                print(f"reward={total_reward:.2f} ({components})")
                num_printed += 1

        return reward_tensor

    def _compute_correctness_score(self, solution_str: str, index: int) -> float:
        found_answer = extract_answer(solution_str, tag_name="answer")
        data = self.train_dataset.data

        entry = data[index]
        if self.train_dataset.experiment:
            experiment = self.train_dataset.experiment
            return experiment.score_answer_with_id(found_answer, entry["metadata"]["entry_id"])
        else:
            return data.score_answer(found_answer, entry=entry)

    def _create_dataloader(self, train_dataset, val_dataset, collate_fn, train_sampler):
        self.train_dataloader = StatefulDataLoader(
            dataset=self.train_dataset,
            batch_size=self.config.data.train_batch_size,
            shuffle=True,
            drop_last=True,
            collate_fn=verl_rl_collate_fn,
        )

        self.val_dataloader = StatefulDataLoader(
            dataset=self.val_dataset,
            batch_size=len(self.val_dataset),
            shuffle=True,
            drop_last=True,
            collate_fn=verl_rl_collate_fn,
        )

        assert len(self.train_dataloader) >= 1
        assert len(self.val_dataloader) >= 1

        print(f"Size of train dataloader: {len(self.train_dataloader)}")
        print(f"Size of val dataloader: {len(self.val_dataloader)}")

        # inject total_training_steps to actor/critic optim_config. This is hacky.
        total_training_steps = len(self.train_dataloader) * self.config.trainer.total_epochs

        if self.config.trainer.total_training_steps is not None:
            total_training_steps = self.config.trainer.total_training_steps

        self.total_training_steps = total_training_steps
        print(f"Total training steps: {self.total_training_steps}")

        OmegaConf.set_struct(self.config, True)
        with open_dict(self.config):
            self.config.actor_rollout_ref.actor.optim.total_training_steps = total_training_steps
            self.config.critic.optim.total_training_steps = total_training_steps

    def _get_gen_batch(self, batch: DataProto) -> DataProto:
        reward_model_keys = set({"data_source", "reward_model", "extra_info", "uid", "index"}) & batch.non_tensor_batch.keys()

        # pop those keys for generation
        batch_keys_to_pop = ["input_ids", "attention_mask", "position_ids"]
        non_tensor_batch_keys_to_pop = set(batch.non_tensor_batch.keys()) - reward_model_keys
        gen_batch = batch.pop(
            batch_keys=batch_keys_to_pop,
            non_tensor_batch_keys=list(non_tensor_batch_keys_to_pop),
        )

        # For agent loop, we need reward model keys to compute score.
        if self.async_rollout_mode:
            gen_batch.non_tensor_batch.update(batch.non_tensor_batch)

        return gen_batch

    def fit(self):
        """
        The training loop of PPO.
        The driver process only need to call the compute functions of the worker group through RPC to construct the PPO dataflow.
        The light-weight advantage computation is done on the driver process.
        """
        from omegaconf import OmegaConf
        
        from verl.utils.tracking import Tracking

        logger = Tracking(
            project_name=self.config.trainer.project_name,
            experiment_name=self.config.trainer.experiment_name,
            default_backend=self.config.trainer.logger,
            config=OmegaConf.to_container(self.config, resolve=True),
        )

        self.global_steps = 0

        # load checkpoint before doing anything
        self._load_checkpoint()
        
        # perform validation before training
        # currently, we only support validation using the reward_function.
        # if self.val_reward_fn is not None and self.config.trainer.get("val_before_train", True):
        #     val_metrics = self._validate()
        #     print(f"Initial validation metrics: {val_metrics}")
        #     logger.log(data=val_metrics, step=self.global_steps)
        #     if self.config.trainer.get("val_only", False):
        #         return

        # we start from step 1
        self.global_steps += 1
        last_val_metrics = None

        if self.continual_enabled:
            task_idx = self.continual_task_index_for_step(int(self.global_steps))
            if task_idx is not None and task_idx != self.continual_current_task_idx:
                self.apply_continual_task(task_idx=task_idx)
                self.continual_current_task_idx = task_idx
            if bool(getattr(self.config.continual, "eval_on_task_start", False)):
                with _timer("continual_eval_start", timing_raw := {}):
                    val_metrics = self._validate()
                prefixed = {f"continual/task_start_eval/{k}": v for k, v in val_metrics.items()}
                logger.log(data=prefixed, step=self.global_steps)
        
        for epoch in range(self.config.trainer.total_epochs):
            for batch_dict in self.train_dataloader:
                if self.continual_enabled:
                    task_idx = self.continual_task_index_for_step(int(self.global_steps))
                    if task_idx is not None and task_idx != self.continual_current_task_idx:
                        self.apply_continual_task(task_idx=task_idx)
                        self.continual_current_task_idx = task_idx
                        if bool(getattr(self.config.continual, "eval_on_task_start", False)):
                            with _timer("continual_eval_start", timing_raw := {}):
                                val_metrics = self._validate()
                            prefixed = {f"continual/task_start_eval/{k}": v for k, v in val_metrics.items()}
                            logger.log(data=prefixed, step=self.global_steps)

                if hasattr(self.actor_rollout_wg, "async_calls_finalize_fn_exec"):
                    self.actor_rollout_wg.async_calls_finalize_fn_exec(blocking=False)
                metrics = {}
                timing_raw = {}

                batch: DataProto = DataProto.from_single_dict(batch_dict)
                batch.meta_info["temperature"] = self.config.actor_rollout_ref.rollout.temperature

                # add uid to batch
                batch.non_tensor_batch["uid"] = np.array(
                    [str(uuid.uuid4()) for _ in range(len(batch.batch))], dtype=object
                )

                gen_batch = self._get_gen_batch(batch)

                # pass global_steps to trace
                gen_batch.meta_info["global_steps"] = self.global_steps
                gen_batch_output = gen_batch.repeat(
                    repeat_times=self.config.actor_rollout_ref.rollout.n, interleave=True
                )

                is_last_step = self.global_steps >= self.total_training_steps
                with _timer("step", timing_raw):
                    # generate a batch
                    with _timer("gen", timing_raw):
                        if not self.async_rollout_mode:
                            gen_batch_output = self.actor_rollout_wg.generate_sequences(gen_batch_output)
                        else:
                            gen_batch_output_padded, pad_size = pad_dataproto_to_divisor(
                                gen_batch_output, self.rollout_pad_divisor()
                            )
                            gen_batch_output_padded = self.async_rollout_manager.generate_sequences(gen_batch_output_padded)
                            gen_batch_output = unpad_dataproto(gen_batch_output_padded, pad_size=pad_size)

                        timing_raw.update(gen_batch_output.meta_info["timing"])
                        gen_batch_output.meta_info.pop("timing", None)

                    if self.config.algorithm.adv_estimator == AdvantageEstimator.REMAX:
                        with _timer("gen_max", timing_raw):
                            gen_baseline_batch = deepcopy(gen_batch)
                            gen_baseline_batch.meta_info["do_sample"] = False
                            if not self.async_rollout_mode:
                                gen_baseline_output = self.actor_rollout_wg.generate_sequences(gen_baseline_batch)
                            else:
                                gen_baseline_batch_padded, pad_size = pad_dataproto_to_divisor(
                                    gen_baseline_batch, self.rollout_pad_divisor()
                                )
                                gen_baseline_output_padded = self.async_rollout_manager.generate_sequences(
                                    gen_baseline_batch_padded
                                )
                                gen_baseline_output = unpad_dataproto(gen_baseline_output_padded, pad_size=pad_size)
                            batch = batch.union(gen_baseline_output)

                            reward_baseline_tensor = self.reward_fn(batch)
                            reward_baseline_tensor = reward_baseline_tensor.sum(dim=-1)

                            batch.pop(batch_keys=list(gen_baseline_output.batch.keys()))

                            batch.batch["reward_baselines"] = reward_baseline_tensor

                            del gen_baseline_batch, gen_baseline_output
                    # repeat to align with repeated responses in rollout
                    batch = batch.repeat(repeat_times=self.config.actor_rollout_ref.rollout.n, interleave=True)
                    batch = batch.union(gen_batch_output)

                    # balance the number of valid tokens on each dp rank.
                    # Note that this breaks the order of data inside the batch.
                    # Please take care when you implement group based adv computation such as GRPO and rloo
                    if self.config.trainer.balance_batch:
                        self._balance_batch(batch, metrics=metrics)

                    # compute global_valid tokens
                    batch.meta_info["global_token_num"] = torch.sum(batch.batch["attention_mask"], dim=-1).tolist()

                    # recompute old_log_probs
                    with _timer("old_log_prob", timing_raw):
                        #TODO: entropy etc
                        old_log_prob = self.actor_rollout_wg.compute_log_prob(batch)
                        batch = batch.union(old_log_prob)

                    if self.use_reference_policy:
                        # compute reference log_prob
                        with _timer("ref", timing_raw):
                            if not self.ref_in_actor:
                                ref_log_prob = self.ref_policy_wg.compute_ref_log_prob(batch)
                            else:
                                ref_log_prob = self.actor_rollout_wg.compute_ref_log_prob(batch)
                            batch = batch.union(ref_log_prob)

                    # compute values
                    if self.use_critic:
                        with _timer("values", timing_raw):
                            values = self.critic_wg.compute_values(batch)
                            batch = batch.union(values)

                    with _timer("adv", timing_raw):
                        # # we combine with rule-based rm
                        reward_tensor = self.reward_fn(batch)
                        batch.batch["token_level_scores"] = reward_tensor
                            
                        # compute rewards. apply_kl_penalty if available
                        if self.config.algorithm.use_kl_in_reward:
                            batch, kl_metrics = apply_kl_penalty(
                                batch, kl_ctrl=self.kl_ctrl_in_reward, kl_penalty=self.config.algorithm.kl_penalty
                            )
                            metrics.update(kl_metrics)
                        else:
                            batch.batch["token_level_rewards"] = batch.batch["token_level_scores"]
                            
                        # compute advantages, executed on the driver process
                        batch = compute_advantage(
                            batch,
                            adv_estimator=self.config.algorithm.adv_estimator,
                            gamma=self.config.algorithm.gamma,
                            lam=self.config.algorithm.lam,
                            num_repeat=self.config.actor_rollout_ref.rollout.n,
                        )
                        
                    # update critic
                    if self.use_critic:
                        with _timer("update_critic", timing_raw):
                            critic_output = self.critic_wg.update_critic(batch)
                        critic_output_metrics = reduce_metrics(critic_output.meta_info["metrics"])
                        metrics.update(critic_output_metrics)

                    # implement critic warmup
                    if self.config.trainer.critic_warmup <= self.global_steps:
                        # update actor
                        with _timer("update_actor", timing_raw):
                            actor_output = self.actor_rollout_wg.update_actor(batch)
                        actor_output_metrics = reduce_metrics(actor_output.meta_info["metrics"])
                        metrics.update(actor_output_metrics)

                    # validate
                    if (
                        self.val_reward_fn is not None
                        and self.config.trainer.test_freq > 0
                        and (is_last_step or self.global_steps % self.config.trainer.test_freq == 0)
                    ):
                        with _timer("testing", timing_raw):
                            val_metrics: dict = self._validate()
                            if is_last_step:
                                last_val_metrics = val_metrics
                        metrics.update(val_metrics)

                    if self.config.trainer.save_freq > 0 and (
                        is_last_step or self.global_steps % self.config.trainer.save_freq == 0
                    ):
                        with _timer("save_checkpoint", timing_raw):
                            self.maybe_save_checkpoint(reason="periodic")

                # collect metrics
                if self.config.curriculum.enabled:
                    grouped_scores = self.train_dataset.aggregate(last_n=self.config.curriculum.last_k)

                    if self.config.curriculum.schedule.automatic:
                        for dataset_name in grouped_scores.keys():
                            if self.global_steps % self.config.curriculum.schedule.update_steps == 0:
                                self.train_dataset.update_experiment_difficulty(dataset_name, method="increment")
                    else:
                        for dataset_name in grouped_scores.keys():
                            if (
                                grouped_scores[dataset_name]["results"] > self.config.curriculum.success_threshold
                            ) and (grouped_scores[dataset_name]["total_samples"] >= self.config.curriculum.last_k):
                                print(
                                    f"Increasing difficulty for dataset: {dataset_name} (success rate: {grouped_scores[dataset_name]['results']:.2f}, samples: {grouped_scores[dataset_name]['total_samples']})"
                                )
                                self.train_dataset.update_experiment_difficulty(dataset_name, method="increment")
                            elif (
                                grouped_scores[dataset_name]["results"] < self.config.curriculum.failure_threshold
                            ) and (grouped_scores[dataset_name]["total_samples"] >= self.config.curriculum.last_k):
                                print(
                                    f"Decreasing difficulty for dataset: {dataset_name} (success rate: {grouped_scores[dataset_name]['results']:.2f}, samples: {grouped_scores[dataset_name]['total_samples']})"
                                )
                                self.train_dataset.update_experiment_difficulty(dataset_name, method="decrement")

                metrics.update(compute_data_metrics(batch=batch, use_critic=self.use_critic))
                metrics.update(compute_timing_metrics(batch=batch, timing_raw=timing_raw))
                # TODO: implement actual tflpo and theoretical tflpo

                # TODO: make a canonical logger that supports various backend
                if self.continual_enabled and self.continual_current_task_idx is not None:
                    metrics["continual/task_index"] = float(self.continual_current_task_idx)
                logger.log(data=metrics, step=self.global_steps)

                if self.continual_enabled and self.continual_current_task_idx is not None:
                    task = self.continual_tasks[self.continual_current_task_idx]
                    if (
                        bool(getattr(self.config.continual, "eval_on_task_end", False))
                        and int(self.global_steps) == int(task.end_step)
                    ):
                        with _timer("continual_eval_end", timing_raw := {}):
                            val_metrics = self._validate()
                        prefixed = {f"continual/task_end_eval/{k}": v for k, v in val_metrics.items()}
                        logger.log(data=prefixed, step=self.global_steps)

                    if (
                        bool(getattr(self.config.continual, "save_ckpt_on_task_end", True))
                        and int(self.global_steps) == int(task.end_step)
                    ):
                        self.maybe_save_checkpoint(reason="task_end")

                if is_last_step:
                    print(f"Final validation metrics: {last_val_metrics}")
                    return

                self.global_steps += 1
                gc.collect()
