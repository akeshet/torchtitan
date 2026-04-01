# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Parallelization for Qwen3.5 MoE hybrid decoder.

Handles the hybrid architecture with both full attention (Attention) and
linear attention (GatedDeltaNet) layers. Key design: maintain Shard(1)
residual stream throughout (SequenceParallel).

- Full attention layers: standard TP on wq/wk/wv/wo with SP on norms
- GatedDeltaNet layers: allgather input, Replicate DTensors internally,
  with DTensor-safe wrappers for conv1d (depthwise) and FLA kernel
- MoE: reuses apply_moe_ep_tp from llama4
- Shared expert: TP on w1/w3/w2 with Shard(1) residual
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributed.device_mesh import DeviceMesh
from torch.distributed.tensor import DTensor, Replicate, Shard
from torch.distributed.tensor.parallel import (
    ColwiseParallel,
    parallelize_module,
    PrepareModuleInput,
    PrepareModuleInputOutput,
    RowwiseParallel,
    SequenceParallel,
)

import torchtitan.models.qwen3_5_moe.model.model as _qwen3_model
from torchtitan.config import CompileConfig, JobConfig, TORCH_DTYPE_MAP
from torchtitan.distributed import NoParallel, ParallelDims
from torchtitan.distributed.activation_checkpoint import apply_ac
from torchtitan.models.llama3.infra.parallelize import apply_ddp
from torchtitan.models.llama4.infra.parallelize import (
    apply_compile,
    apply_fsdp,
    apply_moe_ep_tp,
)
from torchtitan.tools.logging import logger


# for selective op activation checkpointing
_op_sac_save_list = {
    torch.ops.aten.mm.default,
    torch.ops.aten._scaled_dot_product_efficient_attention.default,
    torch.ops.aten._scaled_dot_product_flash_attention.default,
    torch.ops._c10d_functional.reduce_scatter_tensor.default,
    torch.ops.aten.max.default,
    torch._higher_order_ops.flex_attention,
}


# ---------------------------------------------------------------------------
# DTensor-safe wrappers
# ---------------------------------------------------------------------------


class _DTensorSafeInnerAttention(nn.Module):
    """Wrapper that strips DTensor from Q/K/V before inner_attention and
    wraps the output back as DTensor."""

    def __init__(self, inner: nn.Module):
        super().__init__()
        self.inner = inner

    def forward(
        self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, *args, **kwargs
    ) -> torch.Tensor:
        is_dtensor = isinstance(q, DTensor)
        if is_dtensor:
            mesh, placements = q.device_mesh, q.placements
            q, k, v = q.to_local(), k.to_local(), v.to_local()
        out = self.inner(q, k, v, *args, **kwargs)
        if is_dtensor:
            out = DTensor.from_local(out, mesh, placements, run_check=False)
        return out


class _DTensorSafeConv1d(nn.Module):
    """Conv1d wrapper that bypasses DTensor dispatch for depthwise conv.

    DTensor's _tp_conv handler doesn't support depthwise conv (groups > 1).
    This wrapper stores weight as a Replicate DTensor (for mesh consistency
    needed by gradient norm clipping) but runs F.conv1d on local tensors.
    """

    def __init__(self, original: nn.Conv1d, tp_mesh: DeviceMesh):
        super().__init__()
        self.weight = nn.Parameter(
            DTensor.from_local(
                original.weight.data, tp_mesh, [Replicate()], run_check=False
            ),
            requires_grad=original.weight.requires_grad,
        )
        self.bias: nn.Parameter | None = None
        if original.bias is not None:
            self.bias = nn.Parameter(
                DTensor.from_local(
                    original.bias.data, tp_mesh, [Replicate()], run_check=False
                ),
                requires_grad=original.bias.requires_grad,
            )
        self.stride = original.stride
        self.padding = original.padding
        self.dilation = original.dilation
        self.groups = original.groups

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        is_dtensor = isinstance(x, DTensor)
        x_local = x.to_local() if is_dtensor else x
        w_local = (
            self.weight.to_local() if isinstance(self.weight, DTensor) else self.weight
        )
        b_local = None
        if self.bias is not None:
            b_local = (
                self.bias.to_local() if isinstance(self.bias, DTensor) else self.bias
            )
        out = F.conv1d(
            x_local,
            w_local,
            b_local,
            self.stride,
            self.padding,
            self.dilation,
            self.groups,
        )
        if is_dtensor:
            out = DTensor.from_local(out, x.device_mesh, x.placements, run_check=False)
        return out


_dispatch_patched = False
_softplus_registered = False


def _register_dtensor_softplus() -> None:
    """Register aten.softplus.default (and backward) as DTensor pointwise ops."""
    global _softplus_registered
    if _softplus_registered:
        return
    _softplus_registered = True

    from torch.distributed.tensor._op_schema import RuntimeSchemaInfo
    from torch.distributed.tensor._ops._pointwise_ops import pointwise_strategy
    from torch.distributed.tensor._ops.registration import register_op_strategy

    register_op_strategy(
        torch.ops.aten.softplus.default,
        schema_info=RuntimeSchemaInfo(static_kwargkey=["out"]),
    )(pointwise_strategy)

    register_op_strategy(
        torch.ops.aten.softplus_backward.default,
        schema_info=RuntimeSchemaInfo(static_kwargkey=["out"]),
    )(pointwise_strategy)


def _install_dtensor_safe_dispatch() -> None:
    """Monkey-patch _gated_delta_rule_dispatch to handle DTensor inputs."""
    global _dispatch_patched
    if _dispatch_patched:
        return
    _dispatch_patched = True

    original_dispatch = _qwen3_model._gated_delta_rule_dispatch

    def _dtensor_safe_dispatch(
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        g: torch.Tensor,
        beta: torch.Tensor,
        backend: str,
    ) -> torch.Tensor:
        if isinstance(q, DTensor):
            mesh, placements = q.device_mesh, q.placements
            out = original_dispatch(
                q.to_local(),
                k.to_local(),
                v.to_local(),
                g.to_local(),
                beta.to_local(),
                backend,
            )
            return DTensor.from_local(out, mesh, placements, run_check=False)
        return original_dispatch(q, k, v, g, beta, backend)

    _qwen3_model._gated_delta_rule_dispatch = _dtensor_safe_dispatch


# ---------------------------------------------------------------------------
# Main parallelization function
# ---------------------------------------------------------------------------


def parallelize_qwen35_moe(
    model: nn.Module,
    parallel_dims: ParallelDims,
    job_config: JobConfig,
):
    world_mesh = parallel_dims.world_mesh
    assert (
        job_config.training.seq_len % parallel_dims.seq_len_divisor == 0
    ), f"""
        Sequence length {job_config.training.seq_len} must be divisible by the product of TP degree
        ({parallel_dims.tp}) and 2 * CP degree ({parallel_dims.cp}).
        """

    attn_backend = model.model_args.attn_backend
    if job_config.parallelism.context_parallel_degree > 1 and attn_backend not in (
        "sdpa",
    ):
        raise NotImplementedError(
            f"Context Parallel only supports SDPA attention for Qwen3.5 MoE on v0.2.0. "
            f"Got attn_backend='{attn_backend}'."
        )

    model_compile_enabled = (
        job_config.compile.enable and "model" in job_config.compile.components
    )

    if parallel_dims.tp_enabled:
        if (
            job_config.parallelism.enable_async_tensor_parallel
            and not model_compile_enabled
        ):
            raise RuntimeError("Async TP requires torch.compile")

        enable_float8_linear = "float8" in job_config.model.converters
        float8_is_rowwise = job_config.quantize.linear.float8.recipe_name in (
            "rowwise",
            "rowwise_with_gw_hp",
        )
        enable_float8_tensorwise_tp = enable_float8_linear and not float8_is_rowwise

        apply_non_moe_tp(
            model,
            world_mesh["tp"],
            loss_parallel=not job_config.parallelism.disable_loss_parallel,
            enable_float8_tensorwise_tp=enable_float8_tensorwise_tp,
            enable_async_tp=job_config.parallelism.enable_async_tensor_parallel,
        )

    if parallel_dims.tp_enabled or parallel_dims.ep_enabled:
        apply_moe_ep_tp(
            model,
            tp_mesh=world_mesh["tp"] if parallel_dims.tp_enabled else None,
            ep_mesh=world_mesh["ep"] if parallel_dims.ep_enabled else None,
            ep_tp_mesh=(
                world_mesh["ep", "tp"]
                if parallel_dims.tp_enabled
                and parallel_dims.ep_enabled
                and parallel_dims.etp_enabled
                else None
            ),
            etp_enabled=parallel_dims.etp_enabled,
        )

    if job_config.activation_checkpoint.mode != "none":
        use_flex_attn = attn_backend == "flex"
        apply_ac(
            model,
            job_config.activation_checkpoint,
            model_compile_enabled=model_compile_enabled,
            use_flex_attn=use_flex_attn,
            op_sac_save_list=_op_sac_save_list,
            base_folder=job_config.job.dump_folder,
        )

    # turn on per-TransformerBlock compile after AC wrapping and before FSDP
    if model_compile_enabled:
        apply_compile(model, job_config.compile)

    if parallel_dims.fsdp_enabled:
        if parallel_dims.dp_replicate_enabled:
            dp_mesh_dim_names = ("dp_replicate", "dp_shard_cp")
        else:
            dp_mesh_dim_names = ("dp_shard_cp",)
        dp_mesh = world_mesh[tuple(dp_mesh_dim_names)]

        dp_mod_ep_mesh_dim_names = []
        if parallel_dims.ep_enabled:
            if parallel_dims.dp_replicate_enabled:
                dp_mod_ep_mesh_dim_names.append("dp_replicate")
            dp_mod_ep_mesh_dim_names.append("dp_shard_mod_ep")

        apply_fsdp(
            model,
            dp_mesh,
            param_dtype=TORCH_DTYPE_MAP[job_config.training.mixed_precision_param],
            reduce_dtype=TORCH_DTYPE_MAP[job_config.training.mixed_precision_reduce],
            pp_enabled=parallel_dims.pp_enabled,
            cpu_offload=job_config.training.enable_cpu_offload,
            reshard_after_forward_policy=job_config.parallelism.fsdp_reshard_after_forward,
            ep_degree=parallel_dims.ep,
            dp_mod_ep_mesh=(
                world_mesh[tuple(dp_mod_ep_mesh_dim_names)]
                if parallel_dims.ep_enabled
                else None
            ),
            gradient_divide_factor=parallel_dims.fsdp_gradient_divide_factor,
        )

        if parallel_dims.dp_replicate_enabled:
            logger.info("Applied HSDP to the model")
        else:
            logger.info("Applied FSDP to the model")

        if parallel_dims.cp_enabled:
            logger.info("Applied Context Parallel to the model")

        if job_config.training.enable_cpu_offload:
            logger.info("Applied CPU Offloading to the model")
    elif parallel_dims.dp_replicate_enabled:
        if world_mesh.ndim > 1:
            raise RuntimeError("DDP has not supported > 1D parallelism")
        apply_ddp(
            model,
            world_mesh,
            enable_compile=model_compile_enabled,
            enable_compiled_autograd=job_config.parallelism.enable_compiled_autograd,
        )

    return model


# ---------------------------------------------------------------------------
# Non-MoE tensor parallelism
# ---------------------------------------------------------------------------


def apply_non_moe_tp(
    model: nn.Module,
    tp_mesh: DeviceMesh,
    loss_parallel: bool,
    enable_float8_tensorwise_tp: bool,
    enable_async_tp: bool,
):
    """Apply tensor parallelism to non-MoE components.

    Handles the hybrid architecture:
    - Full attention layers: standard TP on Q/K/V/O projections
    - GatedDeltaNet layers: NoParallel on all submodules (Replicate DTensors)
      with DTensor-safe wrappers for conv1d and FLA kernel dispatch
    - Shared expert: TP on w1/w3/w2
    """
    # Patch FLA kernel dispatch to handle Replicate DTensor inputs (idempotent).
    _install_dtensor_safe_dispatch()
    # Register softplus as a DTensor pointwise op.
    _register_dtensor_softplus()

    # Parallel styles for float8 vs standard
    if enable_float8_tensorwise_tp:
        from torchao.float8.float8_tensor_parallel import (
            Float8ColwiseParallel,
            Float8RowwiseParallel,
            PrepareFloat8ModuleInput,
        )

        rowwise_parallel, colwise_parallel, prepare_module_input = (
            Float8RowwiseParallel,
            Float8ColwiseParallel,
            PrepareFloat8ModuleInput,
        )
    else:
        rowwise_parallel, colwise_parallel, prepare_module_input = (
            RowwiseParallel,
            ColwiseParallel,
            PrepareModuleInput,
        )

    # Global: embedding, final norm, output head
    parallelize_module(
        model,
        tp_mesh,
        {
            "tok_embeddings": RowwiseParallel(
                input_layouts=Replicate(),
                output_layouts=Shard(1),
            ),
            "norm": SequenceParallel(),
            "output": ColwiseParallel(
                input_layouts=Shard(1),
                output_layouts=Shard(-1) if loss_parallel else Replicate(),
                use_local_output=not loss_parallel,
            ),
        },
    )

    # Per-layer plans
    for transformer_block in model.layers.values():
        layer_plan = {
            "attention_norm": SequenceParallel(),
            "ffn_norm": SequenceParallel(),
        }

        if transformer_block.layer_type == "full_attention":
            # Full attention: standard TP on Q/K/V/O projections
            layer_plan.update(
                {
                    "attn": prepare_module_input(
                        input_layouts=(Shard(1), Replicate(), None, None),
                        desired_input_layouts=(Replicate(), Replicate(), None, None),
                    ),
                    "attn.wq": colwise_parallel(use_local_output=False),
                    "attn.wk": colwise_parallel(use_local_output=False),
                    "attn.wv": colwise_parallel(use_local_output=False),
                    "attn.q_norm": SequenceParallel(sequence_dim=2),
                    "attn.k_norm": SequenceParallel(sequence_dim=2),
                    "attn.wo": rowwise_parallel(output_layouts=Shard(1)),
                }
            )
        else:
            # GatedDeltaNet: conv1d needs full sequence, FLA kernel needs
            # plain tensors. Keep intermediates as Replicate DTensors.

            # Replace depthwise conv1d with DTensor-safe wrapper
            transformer_block.attn.conv1d = _DTensorSafeConv1d(
                transformer_block.attn.conv1d, tp_mesh
            )

            layer_plan.update(
                {
                    "attn": PrepareModuleInputOutput(
                        input_layouts=(Shard(1),),
                        desired_input_layouts=(Replicate(),),
                        output_layouts=(Replicate(),),
                        desired_output_layouts=(Shard(1),),
                    ),
                    "attn.in_proj_qkv": NoParallel(use_local_output=False),
                    "attn.in_proj_z": NoParallel(use_local_output=False),
                    "attn.in_proj_a": NoParallel(use_local_output=False),
                    "attn.in_proj_b": NoParallel(use_local_output=False),
                    "attn.out_proj": NoParallel(use_local_output=False),
                    "attn.norm": NoParallel(use_local_output=False),
                }
            )

        # Shared expert gate + shared expert FFN
        layer_plan.update(
            {
                "shared_gate": NoParallel(
                    input_layout=Shard(1),
                    output_layout=Shard(1),
                    use_local_output=True,
                ),
                "shared_ffn": prepare_module_input(
                    input_layouts=(Shard(1),),
                    desired_input_layouts=(Replicate(),),
                ),
                "shared_ffn.w1": colwise_parallel(),
                "shared_ffn.w2": rowwise_parallel(output_layouts=Shard(1)),
                "shared_ffn.w3": colwise_parallel(),
            }
        )

        parallelize_module(
            module=transformer_block,
            device_mesh=tp_mesh,
            parallelize_plan=layer_plan,
        )

        # Distribute standalone GatedDeltaNet parameters (A_log, dt_bias)
        # as Replicate DTensors on the TP mesh.
        if transformer_block.layer_type != "full_attention":
            attn = transformer_block.attn
            attn.A_log = nn.Parameter(
                DTensor.from_local(
                    attn.A_log.data, tp_mesh, [Replicate()], run_check=False
                ),
                requires_grad=attn.A_log.requires_grad,
            )
            attn.dt_bias = nn.Parameter(
                DTensor.from_local(
                    attn.dt_bias.data, tp_mesh, [Replicate()], run_check=False
                ),
                requires_grad=attn.dt_bias.requires_grad,
            )

    if enable_async_tp:
        from torch.distributed._symmetric_memory import enable_symm_mem_for_group

        torch._inductor.config._micro_pipeline_tp = True
        enable_symm_mem_for_group(tp_mesh.get_group().group_name)

    logger.info(
        f"Applied {'Float8 tensorwise ' if enable_float8_tensorwise_tp else ''}{'Async ' if enable_async_tp else ''}"
        "Tensor Parallelism to the model"
    )
