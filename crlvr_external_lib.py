"""crlvr_external_lib

Single entry-point for verl `model.external_lib` (HFModelConfig expects a string).

Imports:
- `crlvr_moe_lora_patch`: enable MoE-LoRA inside PEFT LoRA.
- `crlvr_hf_async_rollout`: register HF async rollout backend (`rollout.name=hf`).
"""

def patch_transformers_gradient_checkpointing() -> None:
    """Force Transformers to use re-entrant checkpointing.

    Some configs/environments default to non-reentrant checkpointing, which can raise
    `torch.utils.checkpoint.CheckpointError` when the recomputation differs in the
    number of saved tensors. Re-entrant checkpointing avoids this strict check.
    """

    from transformers.modeling_utils import PreTrainedModel

    orig = getattr(PreTrainedModel, "gradient_checkpointing_enable", None)
    if orig is None:
        raise AttributeError("transformers.PreTrainedModel.gradient_checkpointing_enable is missing")
    if getattr(orig, "crlvr_patched", False):
        return

    def wrapped(self, gradient_checkpointing_kwargs=None):
        kwargs = {} if gradient_checkpointing_kwargs is None else dict(gradient_checkpointing_kwargs)
        # verl currently passes {"use_reentrant": False}; override to avoid non-reentrant
        # checkpoint tensor-count mismatch errors seen with PEFT/MoE-LoRA on NPU.
        kwargs["use_reentrant"] = True
        return orig(self, gradient_checkpointing_kwargs=kwargs)

    wrapped.crlvr_patched = True  # type: ignore[attr-defined]
    PreTrainedModel.gradient_checkpointing_enable = wrapped  # type: ignore[assignment]


patch_transformers_gradient_checkpointing()

import crlvr_hf_async_rollout  # noqa: F401
import crlvr_moe_lora_patch  # noqa: F401
