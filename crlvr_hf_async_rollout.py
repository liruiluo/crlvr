"""crlvr_hf_async_rollout

Adds an async rollout backend `hf` for veRL/verl that runs generation through
HuggingFace/PEFT directly, so custom PEFT forwards (e.g. MoE-LoRA) are honored.

How it works (hybrid/async):
- Rollout workers (AsyncActorRolloutRefWorker) still do the usual wake_up/sleep
  context switch. During wake_up, verl will call `rollout.update_weights(...)`.
- `HFAsyncRollout.update_weights` reads LoRA (+ MoE router) weights from the
  training FSDP model via the `source_model=...` kwarg and updates a local
  HF inference model.
- A tiny Ray server actor forwards token-in-token-out `generate(...)` calls to
  rank-0 worker, which runs `HFAsyncRollout.generate(...)`.

Enable via Hydra overrides:
  actor_rollout_ref.rollout.name=hf
  actor_rollout_ref.model.external_lib=crlvr_external_lib

Notes:
- This backend is intended for correctness (token-wise MoE-LoRA) and is slower
  than vLLM.
"""

from __future__ import annotations

import asyncio
import os
from typing import Any, Generator, Optional

import ray
import torch
from peft import LoraConfig, PeftModel, TaskType, get_peft_model
from peft.utils.save_and_load import get_peft_model_state_dict, set_peft_model_state_dict
from torch.distributed.device_mesh import DeviceMesh
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP, StateDictType
from torch.distributed.fsdp.api import ShardedStateDictConfig
from transformers import AutoModelForCausalLM, GenerationConfig

from verl.single_controller.ray import RayClassWithInitArgs
from verl.utils.config import omega_conf_to_dataclass
from verl.utils.device import get_device_id, get_device_name, get_torch_device
from verl.utils.fsdp_utils import fsdp_version
from verl.workers.config import HFModelConfig, RolloutConfig
from verl.workers.rollout import base as rollout_base
from verl.workers.rollout.base import BaseRollout
from verl.workers.rollout.replica import RolloutMode, RolloutReplica, RolloutReplicaRegistry, TokenOutput
from verl.workers.rollout.utils import is_valid_ipv6_address

import crlvr_moe_lora_patch as moe_patch


def iter_prefix_submodules(module: torch.nn.Module, prefix: str):
    for name, submodule in module.named_modules():
        if name.startswith(prefix) and "." not in name[len(prefix) :]:
            yield name, submodule


class HFAsyncRollout(BaseRollout):
    def __init__(self, config: RolloutConfig, model_config: HFModelConfig, device_mesh: DeviceMesh):
        super().__init__(config, model_config, device_mesh)
        self.rank = int(os.getenv("RANK", "0"))
        self.device = torch.device(get_device_name(), get_device_id())
        self.infer_model: Optional[torch.nn.Module] = None
        self.pending_adapter_state_dicts: Optional[dict[str, dict[str, torch.Tensor]]] = None
        self.pending_router_state_dict: Optional[dict[str, torch.Tensor]] = None

    async def resume(self, tags: list[str]):
        return

    async def release(self):
        get_torch_device().empty_cache()

    def build_infer_model(self) -> torch.nn.Module:
        get_torch_device().set_per_process_memory_fraction(0.95)
        get_torch_device().empty_cache()
        dtype = getattr(torch, self.config.dtype)
        base = AutoModelForCausalLM.from_pretrained(
            self.model_config.local_path,
            torch_dtype=dtype,
            config=self.model_config.hf_config,
            trust_remote_code=self.model_config.trust_remote_code,
        ).to(self.device)

        if self.model_config.lora_rank > 0:
            lora_cfg = LoraConfig(
                task_type=TaskType.CAUSAL_LM,
                r=self.model_config.lora_rank,
                lora_alpha=self.model_config.lora_alpha,
                target_modules=self.model_config.target_modules,
                exclude_modules=self.model_config.exclude_modules,
                bias="none",
            )
            base = get_peft_model(base, lora_cfg)

        override_cfg = getattr(self.model_config, "override_config", {}) or {}
        if bool(override_cfg.get("moe_lora_enable", False)):
            settings = moe_patch.MoeLoraSettings(
                enabled=True,
                num_experts=int(override_cfg.get("moe_lora_num_experts", 4)),
                top_k=int(override_cfg.get("moe_lora_top_k", 2)),
                router_init_std=float(override_cfg.get("moe_lora_router_init_std", 0.02)),
            )
            moe_patch.enable_moe_lora(base, settings=settings)

        base.eval()
        return base

    async def update_weights(self, weights: Generator[tuple[str, torch.Tensor], None, None], **kwargs):
        del weights

        source_model = kwargs.get("source_model", None)
        if source_model is None:
            raise RuntimeError("HFAsyncRollout.update_weights requires source_model=...")

        layered_summon = bool(kwargs.get("layered_summon", False))

        peft_model = getattr(source_model, "_fsdp_wrapped_module", source_model)
        if not isinstance(peft_model, PeftModel):
            raise TypeError(f"Expected source_model to wrap a PeftModel, got: {type(peft_model)}")

        adapter_names = sorted(peft_model.peft_config.keys())

        adapter_state_dicts: dict[str, dict[str, torch.Tensor]] = {name: {} for name in adapter_names}
        router_state_dict: dict[str, torch.Tensor] = {}

        def to_plain_tensor(t: torch.Tensor) -> torch.Tensor:
            if hasattr(t, "full_tensor"):
                t = t.full_tensor()
            if hasattr(t, "to_local"):
                t = t.to_local()
            return t.detach().cpu()

        if fsdp_version(source_model) > 0 and layered_summon:
            prefix_list = [
                "_fsdp_wrapped_module.base_model.model.model.layers.",
                "_fsdp_wrapped_module.base_model.model.model.language_model.layers.",
                "base_model.model.model.layers.",
                "base_model.model.model.language_model.layers.",
            ]

            visited: set[str] = set()
            gathered_any = False
            for prefix in prefix_list:
                for module_name, submodule in iter_prefix_submodules(source_model, prefix):
                    if module_name in visited:
                        continue
                    visited.add(module_name)
                    if fsdp_version(submodule) <= 0:
                        continue

                    module_prefix = module_name.replace("_fsdp_wrapped_module.", "")
                    with FSDP.summon_full_params(submodule, writeback=False):
                        inner = getattr(submodule, "_fsdp_wrapped_module", submodule)
                        layer_sd_full = inner.state_dict()
                        layer_sd = {
                            k: v for k, v in layer_sd_full.items() if ("lora_" in k) or (".moe_router." in k)
                        }
                        for adapter_name in adapter_names:
                            sub_sd = get_peft_model_state_dict(
                                peft_model, adapter_name=adapter_name, state_dict=layer_sd
                            )
                            for k in sorted(sub_sd.keys()):
                                adapter_state_dicts[adapter_name][f"{module_prefix}.{k}"] = to_plain_tensor(sub_sd[k])

                        for k in sorted(layer_sd.keys()):
                            if ".moe_router." not in k:
                                continue
                            router_state_dict[f"{module_prefix}.{k}"] = to_plain_tensor(layer_sd[k])

                    # summon_full_params may leave non-root modules marked as root; reset to
                    # avoid FSDP lazy-init assertions on later forwards.
                    submodule._is_root = False
                    get_torch_device().empty_cache()
                    gathered_any = True

            if not gathered_any:
                raise RuntimeError(
                    "layered_summon=True but did not find any FSDP-wrapped transformer layers to summon. "
                    "Disable layered_summon or adjust FSDP wrapping."
                )
        else:
            model_state_dict: dict[str, torch.Tensor]
            if fsdp_version(source_model) > 0:
                with FSDP.state_dict_type(
                    source_model,
                    StateDictType.SHARDED_STATE_DICT,
                    ShardedStateDictConfig(offload_to_cpu=False),
                ):
                    model_state_dict = source_model.state_dict()
            else:
                model_state_dict = peft_model.state_dict()

            for adapter_name in adapter_names:
                sd = get_peft_model_state_dict(peft_model, adapter_name=adapter_name, state_dict=model_state_dict)
                for k in sorted(sd.keys()):
                    key = k.replace("_fsdp_wrapped_module.", "")
                    adapter_state_dicts[adapter_name][key] = to_plain_tensor(sd[k])

            for k in sorted(model_state_dict.keys()):
                if ".moe_router." not in k:
                    continue
                key = k.replace("_fsdp_wrapped_module.", "")
                router_state_dict[key] = to_plain_tensor(model_state_dict[k])

        self.pending_adapter_state_dicts = adapter_state_dicts
        self.pending_router_state_dict = router_state_dict

        if self.infer_model is not None:
            for adapter_name, adapter_sd in self.pending_adapter_state_dicts.items():
                set_peft_model_state_dict(self.infer_model, adapter_sd, adapter_name=adapter_name)

            if self.pending_router_state_dict:
                self.infer_model.load_state_dict(self.pending_router_state_dict, strict=False)

            self.pending_adapter_state_dicts = None
            self.pending_router_state_dict = None

    async def generate(
        self,
        prompt_ids: list[int],
        sampling_params: dict[str, Any],
        request_id: str,
        image_data: Optional[list[Any]] = None,
        video_data: Optional[list[Any]] = None,
    ) -> TokenOutput:
        del request_id

        if image_data is not None or video_data is not None:
            raise NotImplementedError("HF async rollout currently supports text-only prompts")

        if self.infer_model is None:
            self.infer_model = self.build_infer_model()
        if self.pending_adapter_state_dicts is not None:
            for adapter_name, adapter_sd in self.pending_adapter_state_dicts.items():
                set_peft_model_state_dict(self.infer_model, adapter_sd, adapter_name=adapter_name)
            if self.pending_router_state_dict:
                self.infer_model.load_state_dict(self.pending_router_state_dict, strict=False)
            self.pending_adapter_state_dicts = None
            self.pending_router_state_dict = None

        max_possible_tokens = int(self.config.max_model_len) - len(prompt_ids)
        if max_possible_tokens < 0:
            raise ValueError(
                f"Prompt length ({len(prompt_ids)}) exceeds max_model_len ({self.config.max_model_len})."
            )

        if "max_tokens" in sampling_params:
            max_tokens = int(sampling_params.pop("max_tokens"))
        elif "max_new_tokens" in sampling_params:
            max_tokens = int(sampling_params.pop("max_new_tokens"))
        else:
            max_tokens = int(self.config.response_length)
        max_tokens = min(max_tokens, int(self.config.response_length))
        max_tokens = max(0, min(max_tokens, max_possible_tokens))

        return_logprobs = bool(sampling_params.pop("logprobs", False))
        if max_tokens <= 0:
            return TokenOutput(token_ids=[], log_probs=[] if return_logprobs else None, stop_reason="length")

        do_sample = bool(sampling_params.pop("do_sample", self.config.do_sample))
        temperature = float(sampling_params.pop("temperature", self.config.temperature))
        top_p = float(sampling_params.pop("top_p", self.config.top_p))
        top_k = int(sampling_params.pop("top_k", self.config.top_k))
        if top_k < 0:
            top_k = 0

        generation_config = GenerationConfig(
            do_sample=do_sample,
            num_beams=1,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
        )

        input_ids = torch.tensor([prompt_ids], device=self.device, dtype=torch.long)
        attention_mask = torch.ones_like(input_ids, dtype=torch.long)

        with torch.no_grad(), torch.autocast(device_type=get_device_name(), dtype=torch.bfloat16):
            out = self.infer_model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_new_tokens=max_tokens,
                generation_config=generation_config,
                return_dict_in_generate=True,
                output_scores=return_logprobs,
                use_cache=True,
            )

        seq = out.sequences[0].tolist()
        token_ids = seq[len(prompt_ids) :]

        log_probs = None
        if return_logprobs:
            scores = torch.stack(list(out.scores), dim=0)[:, 0, :]
            chosen = scores.gather(-1, torch.tensor(token_ids, device=scores.device).unsqueeze(-1)).squeeze(-1)
            log_probs = (chosen - scores.logsumexp(dim=-1)).float().cpu().tolist()

        return TokenOutput(token_ids=token_ids, log_probs=log_probs, stop_reason="completed")


@ray.remote(num_cpus=1)
class HFRolloutTokenServer:
    def __init__(
        self,
        config: RolloutConfig,
        model_config: HFModelConfig,
        rollout_mode: RolloutMode,
        workers,
        replica_rank: int,
        node_rank: int,
    ):
        self.config: RolloutConfig = omega_conf_to_dataclass(config)
        self.model_config: HFModelConfig = omega_conf_to_dataclass(model_config, dataclass_type=HFModelConfig)
        self.rollout_mode = rollout_mode
        self.workers = workers
        self.replica_rank = replica_rank
        self.node_rank = node_rank

        self._server_address = ray.util.get_node_ip_address().strip("[]")
        self._server_port = 0

        if len(self.workers) == 0:
            raise RuntimeError("HFRolloutTokenServer requires non-empty workers")
        self.rank0_worker = self.workers[0]

    async def get_server_address(self):
        return self._server_address, self._server_port

    async def wake_up(self):
        if self.rollout_mode == RolloutMode.HYBRID:
            await asyncio.gather(*[w.wake_up.remote() for w in self.workers])
        return True

    async def sleep(self):
        if self.rollout_mode == RolloutMode.HYBRID:
            await asyncio.gather(*[w.sleep.remote() for w in self.workers])
        return True

    async def clear_kv_cache(self):
        get_torch_device().empty_cache()
        return True

    async def generate(
        self,
        request_id: str,
        prompt_ids: list[int],
        sampling_params: dict[str, Any],
        image_data: Optional[list[Any]] = None,
        video_data: Optional[list[Any]] = None,
    ) -> TokenOutput:
        worker = self.workers[(hash(request_id) & 0x7FFFFFFF) % len(self.workers)]
        if video_data is not None:
            raise NotImplementedError("HF async rollout currently supports text-only prompts")

        return await worker.generate.remote(
            prompt_ids=prompt_ids,
            sampling_params=sampling_params,
            request_id=request_id,
            image_data=image_data,
        )


class HFReplica(RolloutReplica):
    def get_ray_class_with_init_args(self) -> RayClassWithInitArgs:
        raise NotImplementedError("HFReplica only supports hybrid mode")

    async def launch_servers(self):
        if len(self.workers) != self.world_size:
            raise RuntimeError(f"worker number {len(self.workers)} not equal to world size {self.world_size}")

        node_id = await self.workers[0].__ray_call__.remote(lambda self: ray.get_runtime_context().get_node_id())
        name = f"hf_server_{self.replica_rank}_0" if not self.is_reward_model else f"hf_server_reward_{self.replica_rank}_0"

        server = HFRolloutTokenServer.options(
            scheduling_strategy=ray.util.scheduling_strategies.NodeAffinitySchedulingStrategy(
                node_id=node_id,
                soft=False,
            ),
            name=name,
        ).remote(
            config=self.config,
            model_config=self.model_config,
            rollout_mode=self.rollout_mode,
            workers=self.workers,
            replica_rank=self.replica_rank,
            node_rank=0,
        )

        self.servers = [server]
        self._server_handle = server

        server_address, server_port = await server.get_server_address.remote()
        self._server_address = f"[{server_address}]:{server_port}" if is_valid_ipv6_address(server_address) else f"{server_address}:{server_port}"


def apply_hf_async_rollout_patch() -> None:
    rollout_base._ROLLOUT_REGISTRY[("hf", "async")] = "crlvr_hf_async_rollout.HFAsyncRollout"
    RolloutReplicaRegistry.register("hf", lambda: HFReplica)


apply_hf_async_rollout_patch()
