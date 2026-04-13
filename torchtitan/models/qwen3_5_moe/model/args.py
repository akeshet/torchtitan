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
class AttentionConfig:
    """Configuration for full attention layers in Qwen3.5 MoE."""

    n_heads: int = 16
    n_kv_heads: int | None = None
    head_dim: int | None = None
    rotary_dim: int | None = None
    qk_norm: bool = True
    norm_eps: float = 1e-6
    bias: bool = False
    attn_backend: str = "sdpa"
    attn_mask_type: str = "causal"
    # TODO: audit whether this field is actually used in model code
    rope_backend: str = "cos_sin"


@dataclass
class GatedDeltaNetConfig:
    """Configuration for GatedDeltaNet linear attention layers."""

    n_key_heads: int = 16
    n_value_heads: int = 32
    key_head_dim: int = 128
    value_head_dim: int = 128
    conv_kernel_size: int = 4
    norm_eps: float = 1e-6
    fla_backend: str = "fla_chunked"


@dataclass
class RopeConfig:
    """Configuration for Rotary Position Embeddings."""

    dim: int = 64
    max_seq_len: int = 262144
    theta: float = 10_000_000.0
    # TODO: audit whether these fields are actually used in model code
    backend: str = "cos_sin"
    scaling: str | None = None
    rope_factor: float | None = None
    original_seq_len: int | None = None


@dataclass
class MoELayerConfig:
    """MoE configuration wrapping MoEArgs with per-expert hidden dim."""

    hidden_dim: int = 512
    num_experts: int = 8
    num_shared_experts: int = 0
    top_k: int = 1
    score_func: str = "sigmoid"
    route_norm: bool = False
    score_before_experts: bool = True
    use_grouped_mm: bool = True
    _debug_force_load_balance: bool = False

    def to_moe_args(self) -> MoEArgs:
        args = MoEArgs(
            num_experts=self.num_experts,
            num_shared_experts=self.num_shared_experts,
            top_k=self.top_k,
            score_func=self.score_func,
            route_norm=self.route_norm,
            score_before_experts=self.score_before_experts,
            use_grouped_mm=self.use_grouped_mm,
        )
        args._debug_force_load_balance = self._debug_force_load_balance
        return args


@dataclass
class FeedForwardConfig:
    """Configuration for the shared expert FeedForward."""

    hidden_dim: int = 512


@dataclass
class LayerConfig:
    """Per-layer configuration grouping attention, deltanet, MoE, and FFN."""

    norm_eps: float = 1e-6
    attention: AttentionConfig = field(default_factory=AttentionConfig)
    deltanet: GatedDeltaNetConfig = field(default_factory=GatedDeltaNetConfig)
    moe: MoELayerConfig = field(default_factory=MoELayerConfig)
    feed_forward: FeedForwardConfig = field(default_factory=FeedForwardConfig)


@dataclass
class Qwen35MoEModelArgs(BaseModelArgs):
    """Model args for Qwen3.5 MoE hybrid decoder."""

    dim: int = 2048
    n_layers: int = 40
    vocab_size: int = 248320
    norm_eps: float = 1e-6

    rope: RopeConfig = field(default_factory=RopeConfig)
    layer: LayerConfig = field(default_factory=LayerConfig)

    full_attention_interval: int = 4

    depth_init: bool = True
    eos_id: int = 151645

    @property
    def n_heads(self) -> int:
        """Exposed for get_moe_model_nparams_and_flops compatibility."""
        return self.layer.attention.n_heads

    @property
    def moe_args(self) -> MoEArgs:
        """Exposed for get_moe_model_nparams_and_flops compatibility."""
        return self.layer.moe.to_moe_args()

    def update_from_config(self, job_config: JobConfig, **kwargs) -> None:
        seq_len = job_config.training.seq_len
        if seq_len > self.rope.max_seq_len:
            logger.warning(
                f"Sequence length {seq_len} exceeds original maximum {self.rope.max_seq_len}."
            )
        self.rope.max_seq_len = seq_len

        self.layer.moe._debug_force_load_balance = (
            job_config.training.debug_moe_force_load_balance
        )

    def get_nparams_and_flops(
        self, model: nn.Module, seq_len: int
    ) -> tuple[int, float]:
        return get_moe_model_nparams_and_flops(self, model, seq_len)
