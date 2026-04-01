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
from torchtitan.models.moe import MoEArgs
from torchtitan.protocols.train_spec import TrainSpec

from .infra.parallelize import parallelize_qwen35_moe
from .model.args import Qwen35MoEModelArgs
from .model.model import Qwen35MoEModel
from .model.state_dict_adapter import Qwen35MoEStateDictAdapter

__all__ = [
    "parallelize_qwen35_moe",
    "Qwen35MoEModelArgs",
    "Qwen35MoEModel",
    "qwen3_5_moe_args",
]

# Adding different variants of the model
# Translated from PR #2545's nested Config definitions to flat dataclass args.

qwen3_5_moe_args = {
    "debugmodel": Qwen35MoEModelArgs(
        dim=256,
        n_layers=8,
        vocab_size=2048,
        norm_eps=1e-6,
        max_seq_len=1024,
        # Full attention
        n_heads=4,
        n_kv_heads=2,
        head_dim=64,
        rotary_dim=16,
        qk_norm=True,
        # GatedDeltaNet
        gdn_n_key_heads=2,
        gdn_n_value_heads=4,
        gdn_key_head_dim=64,
        gdn_value_head_dim=64,
        gdn_conv_kernel_size=4,
        gdn_norm_eps=1e-6,
        gdn_fla_backend="fla_chunked",
        # RoPE
        rope_theta=10000.0,
        # Hybrid
        full_attention_interval=4,
        # MoE
        moe_inter_dim=256,
        moe_args=MoEArgs(
            num_experts=8,
            num_shared_experts=0,
            top_k=2,
            score_func="softmax",
            route_norm=True,
            score_before_experts=False,
            use_grouped_mm=False,
        ),
        # Shared expert
        shared_ffn_hidden_dim=256,
    ),
    "35b-a3b": Qwen35MoEModelArgs(
        dim=2048,
        n_layers=40,
        vocab_size=248320,
        norm_eps=1e-6,
        max_seq_len=262144,
        # Full attention
        n_heads=16,
        n_kv_heads=2,
        head_dim=256,
        rotary_dim=64,
        qk_norm=True,
        # GatedDeltaNet
        gdn_n_key_heads=16,
        gdn_n_value_heads=32,
        gdn_key_head_dim=128,
        gdn_value_head_dim=128,
        gdn_conv_kernel_size=4,
        gdn_norm_eps=1e-6,
        gdn_fla_backend="fla_chunked",
        # RoPE
        rope_theta=10_000_000.0,
        # Hybrid
        full_attention_interval=4,
        # MoE
        moe_inter_dim=512,
        moe_args=MoEArgs(
            num_experts=256,
            num_shared_experts=0,
            top_k=8,
            score_func="softmax",
            route_norm=True,
            score_before_experts=False,
        ),
        # Shared expert
        shared_ffn_hidden_dim=512,
    ),
    "122b-a10b": Qwen35MoEModelArgs(
        dim=3072,
        n_layers=48,
        vocab_size=248320,
        norm_eps=1e-6,
        max_seq_len=262144,
        # Full attention
        n_heads=32,
        n_kv_heads=2,
        head_dim=256,
        rotary_dim=64,
        qk_norm=True,
        # GatedDeltaNet
        gdn_n_key_heads=16,
        gdn_n_value_heads=64,
        gdn_key_head_dim=128,
        gdn_value_head_dim=128,
        gdn_conv_kernel_size=4,
        gdn_norm_eps=1e-6,
        gdn_fla_backend="fla_chunked",
        # RoPE
        rope_theta=10_000_000.0,
        # Hybrid
        full_attention_interval=4,
        # MoE
        moe_inter_dim=1024,
        moe_args=MoEArgs(
            num_experts=256,
            num_shared_experts=0,
            top_k=8,
            score_func="softmax",
            route_norm=True,
            score_before_experts=False,
        ),
        # Shared expert
        shared_ffn_hidden_dim=1024,
    ),
    "397b-a17b": Qwen35MoEModelArgs(
        dim=4096,
        n_layers=60,
        vocab_size=248320,
        norm_eps=1e-6,
        max_seq_len=262144,
        # Full attention
        n_heads=32,
        n_kv_heads=2,
        head_dim=256,
        rotary_dim=64,
        qk_norm=True,
        # GatedDeltaNet
        gdn_n_key_heads=16,
        gdn_n_value_heads=64,
        gdn_key_head_dim=128,
        gdn_value_head_dim=128,
        gdn_conv_kernel_size=4,
        gdn_norm_eps=1e-6,
        gdn_fla_backend="fla_chunked",
        # RoPE
        rope_theta=10_000_000.0,
        # Hybrid
        full_attention_interval=4,
        # MoE
        moe_inter_dim=1024,
        moe_args=MoEArgs(
            num_experts=512,
            num_shared_experts=0,
            top_k=10,
            score_func="softmax",
            route_norm=True,
            score_before_experts=False,
        ),
        # Shared expert
        shared_ffn_hidden_dim=1024,
    ),
    "397B_A19B": Qwen35MoEModelArgs(
        dim=4096,
        n_layers=60,
        vocab_size=248320,
        norm_eps=1e-6,
        max_seq_len=1_000_000,
        # Full attention
        n_heads=32,
        n_kv_heads=2,
        head_dim=256,
        rotary_dim=64,
        qk_norm=True,
        # GatedDeltaNet
        gdn_n_key_heads=16,
        gdn_n_value_heads=64,
        gdn_key_head_dim=128,
        gdn_value_head_dim=128,
        gdn_conv_kernel_size=4,
        gdn_norm_eps=1e-6,
        gdn_fla_backend="fla_chunked",
        # RoPE
        rope_theta=10_000_000.0,
        # Hybrid
        full_attention_interval=4,
        # MoE
        moe_inter_dim=1024,
        moe_args=MoEArgs(
            num_experts=512,
            num_shared_experts=0,
            top_k=10,
            score_func="softmax",
            route_norm=True,
            score_before_experts=False,
        ),
        # Shared expert
        shared_ffn_hidden_dim=1024,
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
