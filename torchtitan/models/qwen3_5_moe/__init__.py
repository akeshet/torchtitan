# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from torchtitan.components.loss import build_cross_entropy_loss
from torchtitan.components.lr_scheduler import build_lr_schedulers
from torchtitan.components.optimizer import build_optimizers_with_moe_load_balancing
from torchtitan.components.tokenizer import build_hf_tokenizer
from torchtitan.components.validate import build_validator
from torchtitan.datasets.hf_datasets import build_hf_dataloader
from torchtitan.protocols.train_spec import TrainSpec

from .infra.parallelize import parallelize_qwen35_moe
from .model.args import (
    AttentionConfig,
    FeedForwardConfig,
    GatedDeltaNetConfig,
    LayerConfig,
    MoELayerConfig,
    Qwen35MoEModelArgs,
    RopeConfig,
)
from .model.model import Qwen35MoEModel
from .model.state_dict_adapter import Qwen35MoEStateDictAdapter

__all__ = [
    "parallelize_qwen35_moe",
    "Qwen35MoEModelArgs",
    "Qwen35MoEModel",
    "qwen3_5_moe_args",
]

qwen3_5_moe_args = {
    "debugmodel": Qwen35MoEModelArgs(
        dim=256,
        n_layers=8,
        vocab_size=2048,
        norm_eps=1e-6,
        rope=RopeConfig(
            dim=16,
            max_seq_len=1024,
            theta=10000.0,
            backend="cos_sin",
        ),
        layer=LayerConfig(
            attention=AttentionConfig(
                n_heads=4,
                n_kv_heads=2,
                head_dim=64,
                rotary_dim=16,
                qk_norm=True,
                norm_eps=1e-6,
                attn_backend="sdpa",
                attn_mask_type="causal",
                rope_backend="cos_sin",
            ),
            deltanet=GatedDeltaNetConfig(
                n_key_heads=2,
                n_value_heads=4,
                key_head_dim=64,
                value_head_dim=64,
            ),
            moe=MoELayerConfig(
                num_experts=8,
                num_shared_experts=0,
                top_k=2,
                hidden_dim=256,
                score_func="softmax",
                route_norm=True,
                score_before_experts=False,
                use_grouped_mm=False,
            ),
            feed_forward=FeedForwardConfig(hidden_dim=256),
        ),
        full_attention_interval=4,
    ),
    "35b-a3b": Qwen35MoEModelArgs(
        dim=2048,
        n_layers=40,
        vocab_size=248320,
        norm_eps=1e-6,
        rope=RopeConfig(
            dim=64,
            max_seq_len=262144,
            theta=10_000_000.0,
            backend="cos_sin",
        ),
        layer=LayerConfig(
            attention=AttentionConfig(
                n_heads=16,
                n_kv_heads=2,
                head_dim=256,
                rotary_dim=64,
                qk_norm=True,
                norm_eps=1e-6,
                attn_backend="sdpa",
                attn_mask_type="causal",
                rope_backend="cos_sin",
            ),
            deltanet=GatedDeltaNetConfig(
                n_key_heads=16,
                n_value_heads=32,
                key_head_dim=128,
                value_head_dim=128,
            ),
            moe=MoELayerConfig(
                num_experts=256,
                num_shared_experts=0,
                top_k=8,
                hidden_dim=512,
                score_func="softmax",
                route_norm=True,
                score_before_experts=False,
            ),
            feed_forward=FeedForwardConfig(hidden_dim=512),
        ),
        full_attention_interval=4,
    ),
    "35b-a3b-varlen": Qwen35MoEModelArgs(
        dim=2048,
        n_layers=40,
        vocab_size=248320,
        norm_eps=1e-6,
        rope=RopeConfig(
            dim=64,
            max_seq_len=262144,
            theta=10_000_000.0,
            backend="cos_sin",
        ),
        layer=LayerConfig(
            attention=AttentionConfig(
                n_heads=16,
                n_kv_heads=2,
                head_dim=256,
                rotary_dim=64,
                qk_norm=True,
                norm_eps=1e-6,
                attn_backend="varlen",
                attn_mask_type="block_causal",
                rope_backend="cos_sin",
            ),
            deltanet=GatedDeltaNetConfig(
                n_key_heads=16,
                n_value_heads=32,
                key_head_dim=128,
                value_head_dim=128,
            ),
            moe=MoELayerConfig(
                num_experts=256,
                num_shared_experts=0,
                top_k=8,
                hidden_dim=512,
                score_func="softmax",
                route_norm=True,
                score_before_experts=False,
            ),
            feed_forward=FeedForwardConfig(hidden_dim=512),
        ),
        full_attention_interval=4,
    ),
    "122b-a10b": Qwen35MoEModelArgs(
        dim=3072,
        n_layers=48,
        vocab_size=248320,
        norm_eps=1e-6,
        rope=RopeConfig(
            dim=64,
            max_seq_len=262144,
            theta=10_000_000.0,
            backend="cos_sin",
        ),
        layer=LayerConfig(
            attention=AttentionConfig(
                n_heads=32,
                n_kv_heads=2,
                head_dim=256,
                rotary_dim=64,
                qk_norm=True,
                norm_eps=1e-6,
                attn_backend="sdpa",
                attn_mask_type="causal",
                rope_backend="cos_sin",
            ),
            deltanet=GatedDeltaNetConfig(
                n_key_heads=16,
                n_value_heads=64,
                key_head_dim=128,
                value_head_dim=128,
            ),
            moe=MoELayerConfig(
                num_experts=256,
                num_shared_experts=0,
                top_k=8,
                hidden_dim=1024,
                score_func="softmax",
                route_norm=True,
                score_before_experts=False,
            ),
            feed_forward=FeedForwardConfig(hidden_dim=1024),
        ),
        full_attention_interval=4,
    ),
    "397b-a17b": Qwen35MoEModelArgs(
        dim=4096,
        n_layers=60,
        vocab_size=248320,
        norm_eps=1e-6,
        rope=RopeConfig(
            dim=64,
            max_seq_len=262144,
            theta=10_000_000.0,
            backend="cos_sin",
        ),
        layer=LayerConfig(
            attention=AttentionConfig(
                n_heads=32,
                n_kv_heads=2,
                head_dim=256,
                rotary_dim=64,
                qk_norm=True,
                norm_eps=1e-6,
                attn_backend="sdpa",
                attn_mask_type="causal",
                rope_backend="cos_sin",
            ),
            deltanet=GatedDeltaNetConfig(
                n_key_heads=16,
                n_value_heads=64,
                key_head_dim=128,
                value_head_dim=128,
            ),
            moe=MoELayerConfig(
                num_experts=512,
                num_shared_experts=0,
                top_k=10,
                hidden_dim=1024,
                score_func="softmax",
                route_norm=True,
                score_before_experts=False,
            ),
            feed_forward=FeedForwardConfig(hidden_dim=1024),
        ),
        full_attention_interval=4,
    ),
    "397B_A19B": Qwen35MoEModelArgs(
        dim=4096,
        n_layers=60,
        vocab_size=248320,
        norm_eps=1e-6,
        rope=RopeConfig(
            dim=64,
            max_seq_len=1_000_000,
            theta=10_000_000.0,
            backend="cos_sin",
            scaling="yarn",
            rope_factor=3.0,
            original_seq_len=262144,
        ),
        layer=LayerConfig(
            attention=AttentionConfig(
                n_heads=32,
                n_kv_heads=2,
                head_dim=256,
                rotary_dim=64,
                qk_norm=True,
                norm_eps=1e-6,
                attn_backend="sdpa",
                attn_mask_type="causal",
                rope_backend="cos_sin",
            ),
            deltanet=GatedDeltaNetConfig(
                n_key_heads=16,
                n_value_heads=64,
                key_head_dim=128,
                value_head_dim=128,
            ),
            moe=MoELayerConfig(
                num_experts=512,
                num_shared_experts=0,
                top_k=10,
                hidden_dim=1024,
                score_func="softmax",
                route_norm=True,
                score_before_experts=False,
            ),
            feed_forward=FeedForwardConfig(hidden_dim=1024),
        ),
        full_attention_interval=4,
    ),
}


def get_train_spec() -> TrainSpec:
    return TrainSpec(
        model_cls=Qwen35MoEModel,
        model_args=qwen3_5_moe_args,
        parallelize_fn=parallelize_qwen35_moe,
        pipelining_fn=None,
        build_optimizers_fn=build_optimizers_with_moe_load_balancing,
        build_lr_schedulers_fn=build_lr_schedulers,
        build_dataloader_fn=build_hf_dataloader,
        build_tokenizer_fn=build_hf_tokenizer,
        build_loss_fn=build_cross_entropy_loss,
        build_validator_fn=build_validator,
        state_dict_adapter=Qwen35MoEStateDictAdapter,
    )
