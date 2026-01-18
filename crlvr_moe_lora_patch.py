"""crlvr_moe_lora_patch

Enable MoE-LoRA (Mixture-of-Experts LoRA) on top of PEFT LoRA inside veRL/verl.

Key requirement implemented:
  - gating is **topk first, then softmax within topk**.

Usage (Hydra overrides):
  actor_rollout_ref.model.external_lib=crlvr_external_lib
  actor_rollout_ref.model.override_config.moe_lora_enable=true
  actor_rollout_ref.model.override_config.moe_lora_num_experts=4
  actor_rollout_ref.model.override_config.moe_lora_top_k=2
  actor_rollout_ref.model.override_config.moe_lora_router_init_std=0.02

Implementation strategy:
  - Keep PEFT adapters as experts: [default, moe_1, ..., moe_{E-1}]
  - Add per-LoRA-Linear router: Linear(in_features -> E)
  - Patch PEFT LoRA Linear forward:
      logits -> topk -> softmax(within topk) -> weighted sum of expert LoRA deltas

"""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
from typing import Iterable, Sequence

import torch
import torch.nn.functional as F
import verl.workers.fsdp_workers as fsdp_workers
from peft import PeftModel
from peft.tuners.lora.layer import Linear as PeftLoraLinear


@dataclass(frozen=True)
class MoeLoraSettings:
    enabled: bool = False
    num_experts: int = 4
    top_k: int = 2
    router_init_std: float = 0.02
    expert_name_prefix: str = "moe"
    router_adapter_name: str = "default"

    @classmethod
    def from_peft_config(cls, peft_config: object) -> "MoeLoraSettings":
        enabled = bool(getattr(peft_config, "moe_lora_enable", False))
        if not enabled:
            return cls(enabled=False)

        return cls(
            enabled=True,
            num_experts=int(getattr(peft_config, "moe_lora_num_experts", cls.num_experts)),
            top_k=int(getattr(peft_config, "moe_lora_top_k", cls.top_k)),
            router_init_std=float(getattr(peft_config, "moe_lora_router_init_std", cls.router_init_std)),
        )


def iter_peft_lora_linears(model: torch.nn.Module) -> Iterable[PeftLoraLinear]:
    for module in model.modules():
        if isinstance(module, PeftLoraLinear):
            yield module


def ensure_router(module: PeftLoraLinear, settings: MoeLoraSettings, in_features: int) -> None:
    if not hasattr(module, "moe_router"):
        module.moe_router = torch.nn.ModuleDict()

    if settings.router_adapter_name in module.moe_router:
        return

    base_weight = getattr(getattr(module, "base_layer", None), "weight", None)
    device = base_weight.device if base_weight is not None else None
    dtype = base_weight.dtype if base_weight is not None else None

    router = torch.nn.Linear(in_features, settings.num_experts, bias=True, device=device, dtype=dtype)
    torch.nn.init.normal_(router.weight, mean=0.0, std=settings.router_init_std)
    torch.nn.init.zeros_(router.bias)

    module.moe_router[settings.router_adapter_name] = router


def patch_peft_lora_linear_forward() -> None:
    """Patch PEFT LoRA Linear forward to support MoE-LoRA."""

    if getattr(PeftLoraLinear.forward, "crlvr_moe_lora_patched", False):
        return

    orig_forward = PeftLoraLinear.forward

    def cast_if_needed(tensor: torch.Tensor, dtype: torch.dtype) -> torch.Tensor:
        return tensor if tensor.dtype == dtype else tensor.to(dtype)

    def moe_forward(self: PeftLoraLinear, x: torch.Tensor, *args, **kwargs) -> torch.Tensor:  # type: ignore[misc]
        if not getattr(self, "crlvr_moe_lora_enabled", False):
            return orig_forward(self, x, *args, **kwargs)

        # If PEFT wants mixed-batch forward or adapters are disabled/merged, fall back.
        if self.disable_adapters or self.merged or ("adapter_names" in kwargs and kwargs["adapter_names"] is not None):
            return orig_forward(self, x, *args, **kwargs)

        # Base output.
        result = self.base_layer(x, *args, **kwargs)
        result_dtype = result.dtype

        expert_adapters: Sequence[str] = getattr(self, "crlvr_moe_expert_adapters", ())
        if not expert_adapters:
            raise RuntimeError("MoE-LoRA enabled but no expert adapters are configured on this module")

        router_adapter = getattr(self, "crlvr_moe_router_adapter", "default")
        top_k = int(getattr(self, "crlvr_moe_top_k", 1))
        top_k = max(1, min(top_k, len(expert_adapters)))

        if not hasattr(self, "moe_router") or router_adapter not in self.moe_router:
            raise RuntimeError(f"MoE-LoRA enabled but router adapter '{router_adapter}' is missing on this module")

        router = self.moe_router[router_adapter]
        router_in = cast_if_needed(x, router.weight.dtype)
        logits = router(router_in)  # (..., num_experts)
        if logits.shape[-1] != len(expert_adapters):
            raise RuntimeError(
                "MoE-LoRA router output dim mismatch: "
                f"logits.shape[-1]={logits.shape[-1]} vs num_experts={len(expert_adapters)}"
            )

        # IMPORTANT: topk first, then softmax within topk.
        topk_vals, topk_idx = torch.topk(logits, k=top_k, dim=-1)
        topk_probs = F.softmax(topk_vals, dim=-1).to(logits.dtype)
        gate = torch.zeros_like(logits)
        gate.scatter_(-1, topk_idx, topk_probs)

        for expert_idx, adapter_name in enumerate(expert_adapters):
            if adapter_name not in self.lora_A or adapter_name not in self.lora_B:
                raise RuntimeError(f"MoE-LoRA expert adapter '{adapter_name}' is missing LoRA weights on this module")

            w = gate[..., expert_idx].unsqueeze(-1)  # (..., 1)
            lora_A = self.lora_A[adapter_name]
            lora_B = self.lora_B[adapter_name]
            dropout = self.lora_dropout[adapter_name]
            scaling = self.scaling[adapter_name]

            x_cast = cast_if_needed(x, lora_A.weight.dtype)
            delta = lora_B(lora_A(dropout(x_cast))) * scaling

            result = result + delta.to(result_dtype) * w.to(result_dtype)

        return result

    moe_forward.crlvr_moe_lora_patched = True  # type: ignore[attr-defined]
    PeftLoraLinear.forward = moe_forward  # type: ignore[assignment]


def enable_moe_lora(peft_model: PeftModel, settings: MoeLoraSettings) -> None:
    """Turn a single-adapter LoRA PeftModel into a multi-expert MoE-LoRA PeftModel."""

    if not settings.enabled:
        return

    if not isinstance(peft_model, PeftModel):
        raise TypeError(f"enable_moe_lora expects peft.PeftModel, got {type(peft_model)}")

    if settings.num_experts < 1:
        raise ValueError(f"moe_lora_num_experts must be >= 1, got {settings.num_experts}")
    if settings.top_k < 1 or settings.top_k > settings.num_experts:
        raise ValueError(f"moe_lora_top_k must be in [1, num_experts], got {settings.top_k}")

    base_cfg = peft_model.peft_config.get("default", None)
    if base_cfg is None:
        raise RuntimeError("Expected a 'default' PEFT LoRA adapter config, but peft_model.peft_config has no 'default'")

    # Ensure expert adapters exist.
    expert_adapters = ["default"]
    for i in range(1, settings.num_experts):
        name = f"{settings.expert_name_prefix}_{i}"
        expert_adapters.append(name)
        if name in peft_model.peft_config:
            continue
        cfg_i = deepcopy(base_cfg)
        cfg_i.inference_mode = False
        peft_model.add_adapter(name, cfg_i, low_cpu_mem_usage=False)

    # Make sure all expert LoRA params are trainable.
    for n, p in peft_model.named_parameters():
        if "lora_" in n:
            p.requires_grad = True

    # Patch PEFT forward once (global).
    patch_peft_lora_linear_forward()

    # Enable MoE per-module, attach routers, and initialize new experts by copying "default".
    for module in iter_peft_lora_linears(peft_model):
        base_layer = getattr(module, "base_layer", None)
        if base_layer is None:
            raise RuntimeError(f"Unexpected PEFT LoRA Linear without base_layer: {module!r}")

        in_features = getattr(base_layer, "in_features", None)
        if in_features is None:
            w = getattr(base_layer, "weight", None)
            if w is None or w.ndim != 2:
                raise RuntimeError(f"Cannot infer in_features for base layer: {base_layer!r}")
            in_features = int(w.shape[1])

        ensure_router(module, settings=settings, in_features=int(in_features))

        # Copy default LoRA weights into each expert for stable start.
        if "default" not in module.lora_A or "default" not in module.lora_B:
            continue
        a0 = module.lora_A["default"].weight
        b0 = module.lora_B["default"].weight
        for adapter_name in expert_adapters:
            if adapter_name == "default":
                continue
            if adapter_name not in module.lora_A or adapter_name not in module.lora_B:
                continue
            module.lora_A[adapter_name].weight.data.copy_(a0.data)
            module.lora_B[adapter_name].weight.data.copy_(b0.data)

        module.crlvr_moe_lora_enabled = True
        module.crlvr_moe_expert_adapters = tuple(expert_adapters)
        module.crlvr_moe_top_k = settings.top_k
        module.crlvr_moe_router_adapter = settings.router_adapter_name

        # Ensure router params are trainable.
        for p in module.moe_router[settings.router_adapter_name].parameters():
            p.requires_grad = True

        # Tiny bias nudge towards expert 0.
        with torch.no_grad():
            bias = module.moe_router[settings.router_adapter_name].bias
            if bias is not None and bias.numel() >= 1:
                bias[0] += 1e-3


def patch_verl_get_peft_model() -> None:
    """Wrap verl's get_peft_model to inject MoE-LoRA after LoRA model creation."""
    fn = getattr(fsdp_workers, "get_peft_model", None)
    if fn is None:
        raise RuntimeError("verl.workers.fsdp_workers has no get_peft_model")
    if getattr(fn, "crlvr_moe_lora_patched", False):
        return

    def wrapped_get_peft_model(model, peft_config, *args, **kwargs):
        peft_model = fn(model, peft_config, *args, **kwargs)

        settings = MoeLoraSettings.from_peft_config(peft_config)
        if settings.enabled:
            print(
                "[crlvr_moe_lora_patch] MoE-LoRA enabled: "
                f"num_experts={settings.num_experts}, top_k={settings.top_k} (topk->softmax), router_init_std={settings.router_init_std}"
            )
            enable_moe_lora(peft_model, settings=settings)

        return peft_model

    wrapped_get_peft_model.crlvr_moe_lora_patched = True  # type: ignore[attr-defined]
    fsdp_workers.get_peft_model = wrapped_get_peft_model


def apply_moe_lora_patch() -> None:
    patch_peft_lora_linear_forward()
    patch_verl_get_peft_model()


apply_moe_lora_patch()
