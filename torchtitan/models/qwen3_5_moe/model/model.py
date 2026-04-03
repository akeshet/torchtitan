# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import torch
import torch.nn.functional as F
from torch import nn
from torch.nn.attention.flex_attention import and_masks, BlockMask

from torchtitan.components.tokenizer import BaseTokenizer
from torchtitan.models.attention import (
    create_attention_mask,
    FlexAttentionWrapper,
    get_causal_mask_mod,
    get_document_mask_mod,
    ScaledDotProductAttentionWrapper,
)
from torchtitan.models.moe import FeedForward, MoE
from torchtitan.protocols.model import AttentionMasksType
from torchtitan.protocols.train_spec import ModelProtocol

from .args import AttentionConfig, GatedDeltaNetConfig, Qwen35MoEModelArgs

try:
    from fla.ops.gated_delta_rule import (
        chunk_gated_delta_rule as _fla_chunk_gated_delta_rule,
        fused_recurrent_gated_delta_rule as _fla_fused_recurrent_gated_delta_rule,
    )

    _HAS_FLA = True
except ImportError:
    _HAS_FLA = False


# ---------------------------------------------------------------------------
# Utility modules
# ---------------------------------------------------------------------------


class OffsetRMSNorm(nn.Module):
    """RMSNorm with offset: ``(1 + weight) * norm(x)``, weight init to zeros."""

    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.zeros(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        input_dtype = x.dtype
        x = x.float()
        variance = x.pow(2).mean(-1, keepdim=True)
        x = x * torch.rsqrt(variance + self.eps)
        return ((1.0 + self.weight.float()) * x).to(input_dtype)

    def reset_parameters(self):
        nn.init.zeros_(self.weight)


class RMSNormGated(nn.Module):
    """Gated RMSNorm: ``silu(gate) * weight * norm(x)``, weight init to ones.

    Used inside GatedDeltaNet. Takes ``(hidden_states, gate)`` separately.
    """

    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, hidden_states: torch.Tensor, gate: torch.Tensor) -> torch.Tensor:
        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.float()
        # Norm before gate (matching transformers Qwen3_5MoeRMSNormGated)
        variance = hidden_states.pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.eps)
        hidden_states = (self.weight * hidden_states).to(input_dtype)
        hidden_states = hidden_states * F.silu(gate.float())
        return hidden_states.to(input_dtype)

    def reset_parameters(self):
        nn.init.ones_(self.weight)


# ---------------------------------------------------------------------------
# RoPE — adapted from qwen3/model/model.py with partial RoPE support
# ---------------------------------------------------------------------------


def precompute_freqs_cis(
    dim: int, max_seq_len: int, base: float = 1_000_000.0
) -> torch.Tensor:
    freqs = 1.0 / (base ** (torch.arange(0, dim, 2)[: (dim // 2)].float() / dim))
    t = torch.arange(max_seq_len, dtype=freqs.dtype, device=freqs.device)
    idx_theta = torch.outer(t, freqs).float()
    freqs = torch.cat([idx_theta, idx_theta], dim=-1)
    freqs_cis = torch.cat([freqs.cos(), freqs.sin()], dim=-1)
    return freqs_cis


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def reshape_for_broadcast(
    freqs_cis: torch.Tensor, x: torch.Tensor, positions: torch.Tensor | None = None
) -> torch.Tensor:
    ndim = x.ndim
    assert ndim > 1
    bz, seqlen, _, head_dim = x.shape
    if positions is None:
        freqs_cis = freqs_cis[0:seqlen]
        assert freqs_cis.shape == (seqlen, head_dim * 2)
        shape = [-1, seqlen, 1, head_dim * 2]
        return freqs_cis.view(*shape)
    elif positions.size(0) == 1:
        assert positions.shape == (1, seqlen)
        freqs_cis = freqs_cis[positions.squeeze(0)]
        assert freqs_cis.shape == (seqlen, head_dim * 2)
        shape = [-1, seqlen, 1, head_dim * 2]
        return freqs_cis.view(*shape)
    else:
        assert positions.shape == (bz, seqlen)
        freqs_cis_expanded = freqs_cis[None, :, None, :].expand(bz, -1, -1, -1)
        freqs_cis = torch.gather(
            freqs_cis_expanded,
            dim=1,
            index=positions.view(bz, seqlen, 1, 1).expand(bz, seqlen, 1, head_dim * 2),
        )
        assert freqs_cis.shape == (bz, seqlen, 1, head_dim * 2)
        return freqs_cis


def apply_rotary_emb(
    xq: torch.Tensor,
    xk: torch.Tensor,
    freqs_cis: torch.Tensor,
    positions: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    head_dim = xq.shape[-1]
    freqs_cis = reshape_for_broadcast(freqs_cis, xq, positions)
    cos = freqs_cis[..., :head_dim].to(dtype=xq.dtype, device=xq.device)
    sin = freqs_cis[..., head_dim:].to(dtype=xq.dtype, device=xq.device)
    xq_out = (xq * cos) + (rotate_half(xq) * sin)
    xk_out = (xk * cos) + (rotate_half(xk) * sin)
    return xq_out.type_as(xq), xk_out.type_as(xk)


def apply_partial_rotary_emb(
    xq: torch.Tensor,
    xk: torch.Tensor,
    freqs_cis: torch.Tensor,
    rotary_dim: int,
    positions: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Apply RoPE only to the first ``rotary_dim`` elements of Q and K."""
    if rotary_dim >= xq.shape[-1]:
        return apply_rotary_emb(xq, xk, freqs_cis, positions)
    xq_rot, xq_pass = xq[..., :rotary_dim], xq[..., rotary_dim:]
    xk_rot, xk_pass = xk[..., :rotary_dim], xk[..., rotary_dim:]
    xq_rot, xk_rot = apply_rotary_emb(xq_rot, xk_rot, freqs_cis, positions)
    xq = torch.cat([xq_rot, xq_pass], dim=-1)
    xk = torch.cat([xk_rot, xk_pass], dim=-1)
    return xq, xk


def repeat_kv(x: torch.Tensor, n_rep: int) -> torch.Tensor:
    """torch.repeat_interleave(x, dim=2, repeats=n_rep)"""
    bs, slen, n_kv_heads, head_dim = x.shape
    if n_rep == 1:
        return x
    return (
        torch.unsqueeze(x, dim=3)
        .expand(bs, slen, n_kv_heads, n_rep, head_dim)
        .reshape(bs, slen, n_kv_heads * n_rep, head_dim)
    )


# ---------------------------------------------------------------------------
# Gated Delta Rule — pure-torch fallback
# ---------------------------------------------------------------------------


def _l2norm(x: torch.Tensor, dim: int = -1, eps: float = 1e-6) -> torch.Tensor:
    """L2 normalization matching the FLA library implementation."""
    return x * torch.rsqrt((x * x).sum(dim=dim, keepdim=True) + eps)


def _torch_chunk_gated_delta_rule(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
) -> torch.Tensor:
    """Pure-torch reference implementation of the gated delta rule.

    Matches the transformers ``torch_recurrent_gated_delta_rule`` semantics:
    L2-normalizes Q/K, scales Q by ``1/sqrt(d_k)``, uses ``exp(g)`` as decay,
    and applies the delta-rule error-correction update.

    Uses ``(B, L, H, D)`` layout matching the FLA kernel convention.

    Args:
        q: (B, L, H, D_k)
        k: (B, L, H, D_k)
        v: (B, L, H, D_v)
        g: (B, L, H) — log-space decay (negative values)
        beta: (B, L, H) — update weight

    Returns:
        output: (B, L, H, D_v)
    """
    B, L, H, D_k = q.shape
    D_v = v.shape[-1]
    dtype = q.dtype

    # L2 normalize Q and K (matching use_qk_l2norm_in_kernel=True)
    q = _l2norm(q.float(), dim=-1)
    k = _l2norm(k.float(), dim=-1)

    # Scale query
    scale = D_k**-0.5
    q = q * scale

    v = v.float()
    g, beta = g.float(), beta.float()

    output = torch.zeros(B, L, H, D_v, dtype=torch.float32, device=q.device)
    state = torch.zeros(B, H, D_k, D_v, dtype=torch.float32, device=q.device)

    for t in range(L):
        q_t = q[:, t, :, :]  # (B, H, D_k)
        k_t = k[:, t, :, :]  # (B, H, D_k)
        v_t = v[:, t, :, :]  # (B, H, D_v)
        g_t = (
            g[:, t, :].exp().unsqueeze(-1).unsqueeze(-1)
        )  # (B, H, 1, 1) — exp(log-decay)
        b_t = beta[:, t, :].unsqueeze(-1)  # (B, H, 1)

        # Decay state
        state = state * g_t
        # Delta-rule error correction: retrieve, compute delta, update
        kv_mem = (state * k_t.unsqueeze(-1)).sum(dim=-2)  # (B, H, D_v)
        delta = (v_t - kv_mem) * b_t  # (B, H, D_v)
        state = state + k_t.unsqueeze(-1) * delta.unsqueeze(-2)  # (B, H, D_k, D_v)
        # Query against state
        output[:, t, :, :] = (state * q_t.unsqueeze(-1)).sum(dim=-2)

    return output.to(dtype)


def _gated_delta_rule_dispatch(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
    backend: str,
) -> torch.Tensor:
    """Dispatch gated delta rule to the selected backend.

    Args:
        q: (B, L, H, D_k)
        k: (B, L, H, D_k)
        v: (B, L, H, D_v)
        g: (B, L, H) — log-space decay
        beta: (B, L, H) — update weight
        backend: One of ``"fla_chunked"``, ``"fla_fused_recurrent"``,
            ``"torch_naive"``.

    Returns:
        output: (B, L, H, D_v)
    """
    _VALID_BACKENDS = {"fla_chunked", "fla_fused_recurrent", "torch_naive"}
    if backend not in _VALID_BACKENDS:
        raise ValueError(
            f"Unknown fla_backend '{backend}'. Valid options: "
            "'fla_chunked', 'fla_fused_recurrent', 'torch_naive'."
        )

    if backend == "torch_naive":
        return _torch_chunk_gated_delta_rule(q, k, v, g, beta)

    if not _HAS_FLA:
        raise RuntimeError(
            f"Backend '{backend}' requires the `fla` package, but it is not installed."
        )

    if backend == "fla_chunked":
        result = _fla_chunk_gated_delta_rule(
            q, k, v, g, beta, use_qk_l2norm_in_kernel=True
        )
    elif backend == "fla_fused_recurrent":
        result = _fla_fused_recurrent_gated_delta_rule(
            q, k, v, g, beta=beta, use_qk_l2norm_in_kernel=True
        )

    if isinstance(result, tuple):
        return result[0]
    return result


# ---------------------------------------------------------------------------
# GatedDeltaNet — linear attention module
# ---------------------------------------------------------------------------


class GatedDeltaNet(nn.Module):
    """Gated DeltaNet linear attention.

    Completely different from standard attention: no RoPE, no attention masks,
    different head structure. Uses recurrent state + gated delta rule.
    """

    def __init__(self, config: GatedDeltaNetConfig, *, dim: int):
        super().__init__()
        self.n_key_heads = config.n_key_heads
        self.n_value_heads = config.n_value_heads
        self.key_head_dim = config.key_head_dim
        self.value_head_dim = config.value_head_dim
        self.conv_kernel_size = config.conv_kernel_size
        self.fla_backend = config.fla_backend

        key_dim = config.n_key_heads * config.key_head_dim
        value_dim = config.n_value_heads * config.value_head_dim
        conv_dim = key_dim * 2 + value_dim

        self.in_proj_qkv = nn.Linear(dim, conv_dim, bias=False)
        self.in_proj_z = nn.Linear(dim, value_dim, bias=False)
        self.in_proj_a = nn.Linear(dim, config.n_value_heads, bias=False)
        self.in_proj_b = nn.Linear(dim, config.n_value_heads, bias=False)

        self.conv1d = nn.Conv1d(
            in_channels=conv_dim,
            out_channels=conv_dim,
            bias=False,
            kernel_size=config.conv_kernel_size,
            groups=conv_dim,  # depthwise
            padding=0,  # causal padding applied manually in forward
        )

        self.A_log = nn.Parameter(torch.zeros(config.n_value_heads))
        self.dt_bias = nn.Parameter(torch.ones(config.n_value_heads))

        self.norm = RMSNormGated(config.value_head_dim, eps=config.norm_eps)
        self.out_proj = nn.Linear(value_dim, dim, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, L, D = x.shape

        # Projections
        qkv = self.in_proj_qkv(x)  # (B, L, conv_dim)
        z = self.in_proj_z(x)  # (B, L, value_dim)
        a = self.in_proj_a(x)  # (B, L, n_value_heads)
        b = self.in_proj_b(x)  # (B, L, n_value_heads)

        # Causal Conv1d + SiLU
        qkv = F.pad(qkv.transpose(1, 2), (self.conv_kernel_size - 1, 0))
        qkv = F.silu(self.conv1d(qkv).transpose(1, 2))  # (B, L, conv_dim)

        # Split into q, k, v
        key_dim = self.n_key_heads * self.key_head_dim
        value_dim = self.n_value_heads * self.value_head_dim
        q, k, v = qkv.split([key_dim, key_dim, value_dim], dim=-1)

        # Reshape to heads — stay in (B, L, H, D) layout for FLA kernel
        q = q.view(B, L, self.n_key_heads, self.key_head_dim)
        k = k.view(B, L, self.n_key_heads, self.key_head_dim)
        v = v.view(B, L, self.n_value_heads, self.value_head_dim)

        # Repeat q, k if n_value_heads > n_key_heads (grouped heads)
        if self.n_value_heads > self.n_key_heads:
            repeat = self.n_value_heads // self.n_key_heads
            q = q.repeat_interleave(repeat, dim=2)
            k = k.repeat_interleave(repeat, dim=2)

        # Compute log-decay (g) and update weight (beta) — (B, L, H_v) layout
        # g is in log-space: always negative, exp(g) ∈ (0, 1) is the actual decay
        g = -torch.exp(self.A_log.float()) * F.softplus(
            a.float() + self.dt_bias
        )  # (B, L, H_v)
        beta = torch.sigmoid(b)  # (B, L, H_v)

        # Gated delta rule — all tensors in (B, L, H, D) layout
        output = _gated_delta_rule_dispatch(
            q, k, v, g, beta, self.fla_backend
        )  # (B, L, H_v, D_v)

        # Apply gated norm (output already in (B, L, H_v, D_v))
        z = z.view(B, L, self.n_value_heads, self.value_head_dim)
        output = self.norm(output, z)

        # Project output
        output = output.reshape(B, L, -1)
        return self.out_proj(output)

    def init_weights(self, init_std: float):
        for linear in (
            self.in_proj_qkv,
            self.in_proj_z,
            self.in_proj_a,
            self.in_proj_b,
        ):
            nn.init.trunc_normal_(linear.weight, mean=0.0, std=0.02)
        nn.init.trunc_normal_(self.out_proj.weight, mean=0.0, std=init_std)
        # A_log: log-uniform init for decay values
        with torch.no_grad():
            self.A_log.copy_(
                torch.log(torch.empty_like(self.A_log).uniform_(1e-6, 16.0))
            )
        nn.init.ones_(self.dt_bias)
        self.norm.reset_parameters()


# ---------------------------------------------------------------------------
# Attention — full attention with output gating + partial RoPE
# ---------------------------------------------------------------------------


class Attention(nn.Module):
    """Full attention with output gating and partial RoPE for Qwen3.5 MoE.

    Key differences from GQAttention:
    - ``wq`` is 2x wider: produces both query and sigmoid gate
    - Partial RoPE: only first ``rotary_dim`` elements get RoPE
    - Output gating: ``attn_output * sigmoid(gate)`` before ``wo``
    - QK norm uses ``OffsetRMSNorm``
    """

    def __init__(self, config: AttentionConfig, *, dim: int):
        super().__init__()
        self.n_heads = config.n_heads
        self.n_kv_heads = (
            config.n_heads if config.n_kv_heads is None else config.n_kv_heads
        )
        self.head_dim = (
            config.head_dim if config.head_dim is not None else dim // config.n_heads
        )
        self.rotary_dim = config.rotary_dim
        self.n_rep = self.n_heads // self.n_kv_heads
        self.use_flex_attn = config.attn_backend == "flex"

        # QK norm uses OffsetRMSNorm (not nn.RMSNorm)
        self.q_norm: OffsetRMSNorm | None = None
        self.k_norm: OffsetRMSNorm | None = None
        if config.qk_norm:
            self.q_norm = OffsetRMSNorm(self.head_dim, eps=config.norm_eps)
            self.k_norm = OffsetRMSNorm(self.head_dim, eps=config.norm_eps)

        # Scaling factor (explicit when head_dim differs from dim // n_heads)
        self.scaling = self.head_dim**-0.5 if config.head_dim is not None else None

        # wq is 2x wider: produces query + gate
        self.wq = nn.Linear(dim, self.n_heads * self.head_dim * 2, bias=config.bias)
        self.wk = nn.Linear(dim, self.n_kv_heads * self.head_dim, bias=config.bias)
        self.wv = nn.Linear(dim, self.n_kv_heads * self.head_dim, bias=config.bias)
        self.wo = nn.Linear(self.n_heads * self.head_dim, dim, bias=config.bias)

        if self.use_flex_attn:
            self.inner_attention = FlexAttentionWrapper()
        else:
            self.inner_attention = ScaledDotProductAttentionWrapper()

    def init_weights(self, init_std: float):
        nn.init.trunc_normal_(self.wq.weight, mean=0.0, std=0.02)
        if self.wq.bias is not None:
            nn.init.trunc_normal_(self.wq.bias, mean=0.0, std=0.02)
        for linear in (self.wk, self.wv):
            nn.init.trunc_normal_(linear.weight, mean=0.0, std=0.02)
            if linear.bias is not None:
                nn.init.trunc_normal_(linear.bias, mean=0.0, std=0.02)
        nn.init.trunc_normal_(self.wo.weight, mean=0.0, std=init_std)
        if self.wo.bias is not None:
            nn.init.trunc_normal_(self.wo.bias, mean=0.0, std=init_std)
        if self.q_norm is not None:
            self.q_norm.reset_parameters()
        if self.k_norm is not None:
            self.k_norm.reset_parameters()

    def forward(
        self,
        x: torch.Tensor,
        rope_cache: torch.Tensor,
        attention_masks: AttentionMasksType | None,
        positions: torch.Tensor | None = None,
    ) -> torch.Tensor:
        bs, seqlen, _ = x.shape

        # Project Q (2x wider for query + gate), K, V
        xq_gate = self.wq(x).view(bs, seqlen, -1, self.head_dim * 2)
        xq, gate = xq_gate.chunk(2, dim=-1)  # each (bs, seqlen, n_heads, head_dim)
        xk = self.wk(x).view(bs, seqlen, -1, self.head_dim)
        xv = self.wv(x).view(bs, seqlen, -1, self.head_dim)

        # QK norm (before RoPE)
        if self.q_norm is not None:
            xq = self.q_norm(xq)
        if self.k_norm is not None:
            xk = self.k_norm(xk)

        # Partial RoPE
        xq, xk = apply_partial_rotary_emb(
            xq, xk, rope_cache, self.rotary_dim, positions
        )

        # Repeat k/v heads for GQA
        keys = repeat_kv(xk, self.n_rep)
        values = repeat_kv(xv, self.n_rep)

        xq = xq.transpose(1, 2)  # (bs, n_heads, seqlen, head_dim)
        xk = keys.transpose(1, 2)
        xv = values.transpose(1, 2)

        if self.use_flex_attn:
            assert isinstance(attention_masks, BlockMask), attention_masks
            output = self.inner_attention(
                xq, xk, xv, block_mask=attention_masks, scale=self.scaling
            )
        else:
            assert attention_masks is None
            output = self.inner_attention(xq, xk, xv, scale=self.scaling)

        output = output.transpose(1, 2).contiguous()  # (bs, seqlen, n_heads, head_dim)

        # Output gating: attn_output * sigmoid(gate) before wo
        output = output * torch.sigmoid(gate)
        output = output.view(bs, seqlen, -1)
        return self.wo(output)


# ---------------------------------------------------------------------------
# TransformerBlock — hybrid decoder layer
# ---------------------------------------------------------------------------


class TransformerBlock(nn.Module):
    """Transformer block for Qwen3.5 MoE hybrid decoder.

    Each layer uses either full attention (``Attention``) or linear attention
    (``GatedDeltaNet``), determined by ``full_attention_interval``. Both types
    share the same MoE + gated shared expert FFN structure.
    """

    def __init__(self, layer_id: int, model_args: Qwen35MoEModelArgs):
        super().__init__()
        dim = model_args.dim
        layer_config = model_args.layer
        self.layer_id = layer_id

        # Determine layer type
        is_full_attn = (layer_id + 1) % model_args.full_attention_interval == 0
        self.layer_type = "full_attention" if is_full_attn else "linear_attention"

        # Attention: full or DeltaNet
        if self.layer_type == "full_attention":
            self.attn = Attention(layer_config.attention, dim=dim)
        else:
            self.attn = GatedDeltaNet(layer_config.deltanet, dim=dim)

        # MoE (routed experts only, num_shared_experts=0)
        # NOTE: Weight layout difference vs transformers —
        # transformers fuses gate_proj and up_proj into a single gate_up_proj
        # tensor of shape (num_experts, 2*intermediate_size, hidden_size),
        # while we keep them as separate w1 (gate) and w3 (up) in GroupedExperts.
        # Checkpoint conversion must split gate_up_proj along dim=1 into w1/w3.
        self.moe_enabled = True  # always True for Qwen3.5 MoE
        self.moe = MoE(
            layer_config.moe.to_moe_args(),
            dim=dim,
            hidden_dim=layer_config.moe.hidden_dim,
        )

        # Shared expert: FeedForward + sigmoid gate
        self.shared_ffn = FeedForward(
            dim=dim, hidden_dim=layer_config.feed_forward.hidden_dim
        )
        self.shared_gate = nn.Linear(dim, 1, bias=False)

        # Norms (OffsetRMSNorm)
        self.attention_norm = OffsetRMSNorm(dim, eps=layer_config.norm_eps)
        self.ffn_norm = OffsetRMSNorm(dim, eps=layer_config.norm_eps)

        if model_args.depth_init:
            self.weight_init_std = 0.02 / (2 * (layer_id + 1)) ** 0.5
        else:
            self.weight_init_std = 0.02 / (2 * model_args.n_layers) ** 0.5

    def forward(
        self,
        x: torch.Tensor,
        rope_cache: torch.Tensor,
        attention_masks: AttentionMasksType | None,
        positions: torch.Tensor | None = None,
    ) -> torch.Tensor:
        # Attention block
        h = self.attention_norm(x)
        if self.layer_type == "full_attention":
            h = self.attn(h, rope_cache, attention_masks, positions)
        else:
            h = self.attn(h)  # DeltaNet ignores rope/masks
        x = x + h

        # FFN block: MoE + gated shared expert
        h = self.ffn_norm(x)
        moe_out = self.moe(h)
        shared_out = torch.sigmoid(self.shared_gate(h)) * self.shared_ffn(h)
        x = x + moe_out + shared_out
        return x

    def init_weights(self, buffer_device: torch.device):
        self.attn.init_weights(self.weight_init_std)
        self.moe.init_weights(self.weight_init_std, buffer_device)
        self.shared_ffn.init_weights(self.weight_init_std)
        nn.init.trunc_normal_(self.shared_gate.weight, mean=0.0, std=0.02)
        self.attention_norm.reset_parameters()
        self.ffn_norm.reset_parameters()


# ---------------------------------------------------------------------------
# Model — top-level hybrid decoder
# ---------------------------------------------------------------------------


class Qwen35MoEModel(nn.Module, ModelProtocol):
    """Qwen3.5 MoE hybrid decoder model.

    Alternates between GatedDeltaNet (linear attention) and full attention
    layers, controlled by ``full_attention_interval``. Every Nth layer uses
    full attention; the rest use GatedDeltaNet.
    """

    def __init__(self, model_args: Qwen35MoEModelArgs):
        super().__init__()
        self.model_args = model_args
        self.vocab_size = model_args.vocab_size
        self.n_layers = model_args.n_layers

        self.tok_embeddings = nn.Embedding(model_args.vocab_size, model_args.dim)

        self.register_buffer(
            "freqs_cis", self._precompute_freqs_cis(), persistent=False
        )

        self.layers = torch.nn.ModuleDict()
        for layer_id in range(model_args.n_layers):
            self.layers[str(layer_id)] = TransformerBlock(layer_id, model_args)

        self.norm = OffsetRMSNorm(model_args.dim, eps=model_args.norm_eps)
        self.output = nn.Linear(model_args.dim, model_args.vocab_size, bias=False)

    def _precompute_freqs_cis(self) -> torch.Tensor:
        return precompute_freqs_cis(
            self.model_args.rope.dim,
            self.model_args.rope.max_seq_len,
            self.model_args.rope.theta,
        )

    def init_weights(
        self,
        buffer_device: torch.device | None = None,
    ):
        buffer_device = buffer_device or self.freqs_cis.device
        with torch.device(buffer_device):
            self.freqs_cis = self._precompute_freqs_cis()
        if self.tok_embeddings is not None:
            nn.init.normal_(self.tok_embeddings.weight)
        for layer in self.layers.values():
            if layer is not None:
                layer.init_weights(buffer_device)
        if self.norm is not None:
            self.norm.reset_parameters()
        final_out_std = self.model_args.dim**-0.5
        cutoff_factor = 3
        if self.output is not None:
            nn.init.trunc_normal_(
                self.output.weight,
                mean=0.0,
                std=final_out_std,
                a=-cutoff_factor * final_out_std,
                b=cutoff_factor * final_out_std,
            )

    def get_attention_masks(
        self,
        input_batch: torch.Tensor,
        tokenizer: BaseTokenizer,
        extra_inputs: dict[str, torch.Tensor] | None = None,
    ) -> AttentionMasksType:
        mask_mods = [get_causal_mask_mod()]
        match self.model_args.layer.attention.attn_mask_type:
            case "causal":
                B = 1
            case "block_causal":
                B = input_batch.shape[0]
                mask_mods.append(
                    get_document_mask_mod(input_batch, tokenizer.eos_id)
                )
            case _:
                raise ValueError(
                    f"Unknown attention mask type: {self.model_args.layer.attention.attn_mask_type}"
                )
        return create_attention_mask(
            and_masks(*mask_mods), B, None, input_batch.shape[1], input_batch.shape[1]
        )

    def forward(
        self,
        tokens: torch.Tensor,
        attention_masks: AttentionMasksType | None = None,
        positions: torch.Tensor | None = None,
    ) -> torch.Tensor:
        h = self.tok_embeddings(tokens) if self.tok_embeddings is not None else tokens

        for layer in self.layers.values():
            h = layer(h, self.freqs_cis, attention_masks, positions)

        h = self.norm(h) if self.norm is not None else h
        output = self.output(h) if self.output is not None else h
        return output
