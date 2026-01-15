# Adapted version of Bytedance code.
"""
Agent loop that skips reward computation.
"""

import logging
import os
from typing import Any

import numpy as np
import ray
import torch
from omegaconf import DictConfig

from verl.experimental.agent_loop.agent_loop import AgentLoopWorkerBase, _InternalAgentLoopOutput, AgentLoopManager
from verl.protocol import DataProto
from verl.utils.model import compute_position_id_with_mask


logger = logging.getLogger(__file__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))



@ray.remote
class AgentLoopWorker(AgentLoopWorkerBase):
    """Agent loop worker that skips reward computation."""

    def __init__(
        self, config: DictConfig, server_handles: list[ray.actor.ActorHandle], reward_router_address: str = None
    ):
        """Initialize agent loop manager.
        Args:
            config (DictConfig): YAML config.
            server_handles (List[ray.actor.ActorHandle]): OpenAI compatible LLM server actor handles.
            reward_router_address (str): reward router address.
        """
        super().__init__(config, server_handles, reward_router_address)


    async def _agent_loop_postprocess(self, output, **kwargs) -> _InternalAgentLoopOutput:
        """Perform post-processing operations on the output of each individual agent loop."""
        output.extra_fields["raw_prompt"] = kwargs["raw_prompt"]

        # Some AgentLoop may have already computed the reward score, e.g SWE-agent.

        # NOTE: consistent with the legacy batch version of generate_sequences that existed in the
        # deprecated vLLM SPMD rollout implementation.
        # prompt_ids: left padded with zeros (e.g., [0,0,0,0,1,2,3,4])
        # response_ids: right padded with zeros (e.g., [5,6,7,8,0,0,0,0])
        # input_ids: concatenation of prompt + response
        # Mask:
        # For example, if the prompt is [1,2,3,4] and the response is [5,6,7,(tool start)8,9(tool end),10,11,12]
        # - prompt_attention_mask: 0s for padding, 1s for tokens
        #   e.g., [0,0,0,0,1,1,1,1]
        # - response_attention_mask: 0s for padding, 1s for tokens
        #   e.g., [1,1,1,1,1,1,1,1,1,1,1,0,0,0,0]
        # attention_mask: concatenation of prompt_attention_mask and response_attention_mask
        #   e.g., [0,0,0,0,1,1,1,1(prompt),1,1,1,1,1,1,1,1,1,1,1,0,0,0,0(response)]
        # - response_mask: 1s for LLM generated tokens, 0 for tool response/padding tokens
        #   e.g., [1,1,1,1,1,1,1,(tool start),0,0(tool end),1,1,0,0,0,0]
        # - position_ids: sequential positions for tokens, starting at 0
        #   e.g., [0,0,0,0,0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,0,0,0,0]

        self.tokenizer.padding_side = "left"
        prompt_output = self.tokenizer.pad(
            {"input_ids": output.prompt_ids},
            padding="max_length",
            max_length=self.config.actor_rollout_ref.rollout.prompt_length,
            return_tensors="pt",
            return_attention_mask=True,
        )
        if prompt_output["input_ids"].dim() == 1:
            prompt_output["input_ids"] = prompt_output["input_ids"].unsqueeze(0)
            prompt_output["attention_mask"] = prompt_output["attention_mask"].unsqueeze(0)

        self.tokenizer.padding_side = "right"
        response_output = self.tokenizer.pad(
            {"input_ids": output.response_ids},
            padding="max_length",
            max_length=self.config.actor_rollout_ref.rollout.response_length,
            return_tensors="pt",
            return_attention_mask=True,
        )
        if response_output["input_ids"].dim() == 1:
            response_output["input_ids"] = response_output["input_ids"].unsqueeze(0)
            response_output["attention_mask"] = response_output["attention_mask"].unsqueeze(0)

        response_mask_output = self.tokenizer.pad(
            {"input_ids": output.response_mask},
            padding="max_length",
            max_length=self.config.actor_rollout_ref.rollout.response_length,
            return_tensors="pt",
            return_attention_mask=False,
        )
        if response_mask_output["input_ids"].dim() == 1:
            response_mask_output["input_ids"] = response_mask_output["input_ids"].unsqueeze(0)

        response_logprobs = None
        if output.response_logprobs is not None:
            pad_size = self.config.actor_rollout_ref.rollout.response_length - len(output.response_logprobs)
            response_logprobs = torch.tensor(output.response_logprobs + [0.0] * pad_size).unsqueeze(0)

        response_mask = response_mask_output["input_ids"] * response_output["attention_mask"]
        attention_mask = torch.cat([prompt_output["attention_mask"], response_output["attention_mask"]], dim=1)
        input_ids = torch.cat([prompt_output["input_ids"], response_output["input_ids"]], dim=1)

        routed_experts = None
        if output.routed_experts is not None:
            total_length = input_ids.shape[1]
            length, layer_num, topk_num = output.routed_experts.shape
            experts_tensor = torch.from_numpy(output.routed_experts)
            routed_experts = torch.zeros(1, total_length, layer_num, topk_num, dtype=experts_tensor.dtype)

            # Calculate start position: left padding means original prompt starts at the end
            start_pos = prompt_output["input_ids"].shape[1] - len(output.prompt_ids)
            end_pos = min(start_pos + length, total_length)

            # Add boundary checks for robustness
            if start_pos < 0 or end_pos > total_length:
                raise ValueError(
                    f"Invalid position range: start_pos={start_pos}, end_pos={end_pos}, total_length={total_length}"
                )

            routed_experts[:, start_pos:end_pos] = experts_tensor.unsqueeze(0)

        # Handle multi-modal inputs and position_ids calculation
        # Only support Qwen2VLImageProcessor for multi-modal processing currently
        # TODO: support other multi-modal inputs
        multi_modal_inputs = None
        if self.processor is not None:
            images = getattr(output, "multi_modal_data", {}).get("image", None)
            current_text = self.tokenizer.decode(input_ids.squeeze(0), skip_special_tokens=True)
            multi_modal_inputs = self.processor(text=[current_text], images=images, return_tensors="pt")
            multi_modal_inputs.pop("input_ids", None)
            multi_modal_inputs.pop("attention_mask", None)

            # We must use dict(multi_modal_inputs) to convert BatchFeature values to a new dict
            # because np.array() only keeps the keys for BatchFeature.
            multi_modal_inputs = dict(multi_modal_inputs.convert_to_tensors("pt"))
        if self.processor is not None and "Qwen2VLImageProcessor" in self.processor.image_processor.__class__.__name__:
            from verl.models.transformers.qwen2_vl import get_rope_index

            image_grid_thw = multi_modal_inputs.get("image_grid_thw")
            video_grid_thw = multi_modal_inputs.get("video_grid_thw")
            second_per_grid_ts = multi_modal_inputs.get("second_per_grid_ts")

            vision_position_ids = get_rope_index(
                self.processor,
                input_ids=input_ids.squeeze(0),
                image_grid_thw=image_grid_thw,
                video_grid_thw=video_grid_thw,
                second_per_grid_ts=second_per_grid_ts,
                attention_mask=attention_mask.squeeze(0),
            ).unsqueeze(0)  # (1, 3, seq_len)

            valid_mask = attention_mask[0].bool()
            text_position_ids = torch.ones((1, len(input_ids[0])), dtype=torch.long)
            text_position_ids[0, valid_mask] = torch.arange(valid_mask.sum().item())
            text_position_ids = text_position_ids.unsqueeze(0)
            position_ids = torch.cat((text_position_ids, vision_position_ids), dim=1)  # (1, 4, seq_length)
        else:
            position_ids = compute_position_id_with_mask(attention_mask)  # (1, seq_len)

        return _InternalAgentLoopOutput(
            prompt_ids=prompt_output["input_ids"],
            response_ids=response_output["input_ids"],
            input_ids=input_ids,
            position_ids=position_ids,
            response_mask=response_mask,
            attention_mask=attention_mask,
            response_logprobs=response_logprobs,
            routed_experts=routed_experts,
            multi_modal_inputs=multi_modal_inputs,
            multi_modal_data=output.multi_modal_data,
            # reward_score=output.reward_score,
            num_turns=output.num_turns,
            metrics=output.metrics,
            extra_fields=output.extra_fields,
        )

class AgentLoopManagerWithoutReward(AgentLoopManager):
    agent_loop_workers_class = AgentLoopWorker
