# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from dataclasses import dataclass, field

from torch import nn

from torchtitan.config import JobConfig
from torchtitan.models.moe import MoEArgs
from torchtitan.models.utils import get_moe_model_nparams_and_flops
from torchtitan.protocols.train_spec import BaseModelArgs
from torchtitan.tools.logging import logger


@dataclass
class Qwen35MoEModelArgs(BaseModelArgs):
    """Flat model args for Qwen3.5 MoE hybrid decoder.

    Flattened from the upstream PR's nested Config classes (Model.Config,
    TransformerBlock.Config, Attention.Config, GatedDeltaNet.Config).
    """

    # --- Standard transformer ---
    dim: int = 2048
    n_layers: int = 40
    vocab_size: int = 248320
    norm_eps: float = 1e-6
    max_seq_len: int = 262144

    # --- Full attention ---
    n_heads: int = 16
    n_kv_heads: int = 2
    head_dim: int = 256
    rotary_dim: int = 64  # partial RoPE: only first rotary_dim dims get RoPE
    qk_norm: bool = True

    # --- GatedDeltaNet (linear attention) ---
    gdn_n_key_heads: int = 16
    gdn_n_value_heads: int = 32
    gdn_key_head_dim: int = 128
    gdn_value_head_dim: int = 128
    gdn_conv_kernel_size: int = 4
    gdn_norm_eps: float = 1e-6
    gdn_fla_backend: str = "fla_chunked"

    # --- RoPE ---
    rope_theta: float = 10_000_000.0

    # --- Hybrid layer arrangement ---
    full_attention_interval: int = 4  # every Nth layer is full attention

    # --- MoE ---
    moe_inter_dim: int = 512  # per-expert FFN hidden dim
    moe_args: MoEArgs = field(default_factory=MoEArgs)

    # --- Shared expert ---
    shared_ffn_hidden_dim: int = 512

    # --- Attention backend ---
    attn_backend: str = "sdpa"  # sdpa, flex
    attn_mask_type: str = "causal"

    # --- Misc ---
    depth_init: bool = True
    eos_id: int = 151645

    def update_from_config(self, job_config: JobConfig, **kwargs) -> None:
        seq_len = job_config.training.seq_len
        if seq_len > self.max_seq_len:
            logger.warning(
                f"Sequence length {seq_len} exceeds original maximum {self.max_seq_len}."
            )
        self.max_seq_len = seq_len

        self.moe_args._debug_force_load_balance = (
            job_config.training.debug_moe_force_load_balance
        )

    def get_nparams_and_flops(
        self, model: nn.Module, seq_len: int
    ) -> tuple[int, float]:
        return get_moe_model_nparams_and_flops(self, model, seq_len)
