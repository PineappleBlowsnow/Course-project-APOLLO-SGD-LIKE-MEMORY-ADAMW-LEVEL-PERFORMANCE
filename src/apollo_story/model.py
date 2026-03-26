from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Literal

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class GPTConfig:
    vocab_size: int = 50257
    block_size: int = 256
    n_layer: int = 12
    n_head: int = 12
    n_embd: int = 768
    dropout: float = 0.1
    bias: bool = True


@dataclass
class LLaMAConfig:
    vocab_size: int = 32000
    block_size: int = 256
    n_layer: int = 12
    n_head: int = 12
    n_embd: int = 768
    intermediate_size: int = 2048
    dropout: float = 0.0
    bias: bool = False
    rope_theta: float = 10000.0
    rms_norm_eps: float = 1.0e-6


class LayerNorm(nn.Module):
    def __init__(self, ndim: int, bias: bool) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(ndim))
        self.bias = nn.Parameter(torch.zeros(ndim)) if bias else None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.layer_norm(x, self.weight.shape, self.weight, self.bias, 1e-5)


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        variance = x.pow(2).mean(dim=-1, keepdim=True)
        x = x * torch.rsqrt(variance + self.eps)
        return self.weight * x


def _rotate_half(x: torch.Tensor) -> torch.Tensor:
    x1 = x[..., ::2]
    x2 = x[..., 1::2]
    return torch.stack((-x2, x1), dim=-1).flatten(start_dim=-2)


def _apply_rope(
    q: torch.Tensor,
    k: torch.Tensor,
    positions: torch.Tensor,
    theta: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    head_dim = q.size(-1)
    device = q.device
    dtype = q.dtype
    inv_freq = 1.0 / (theta ** (torch.arange(0, head_dim, 2, device=device, dtype=torch.float32) / head_dim))
    freqs = torch.outer(positions.to(torch.float32), inv_freq)
    emb = torch.repeat_interleave(freqs, repeats=2, dim=-1)
    cos = emb.cos().to(dtype=dtype)[None, None, :, :]
    sin = emb.sin().to(dtype=dtype)[None, None, :, :]
    q = (q * cos) + (_rotate_half(q) * sin)
    k = (k * cos) + (_rotate_half(k) * sin)
    return q, k


class CausalSelfAttention(nn.Module):
    def __init__(self, config: GPTConfig) -> None:
        super().__init__()
        assert config.n_embd % config.n_head == 0
        self.n_head = config.n_head
        self.n_embd = config.n_embd
        self.dropout = config.dropout
        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd, bias=config.bias)
        self.c_proj = nn.Linear(config.n_embd, config.n_embd, bias=config.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch, seq_len, channels = x.size()
        q, k, v = self.c_attn(x).split(self.n_embd, dim=2)
        head_dim = channels // self.n_head
        q = q.view(batch, seq_len, self.n_head, head_dim).transpose(1, 2)
        k = k.view(batch, seq_len, self.n_head, head_dim).transpose(1, 2)
        v = v.view(batch, seq_len, self.n_head, head_dim).transpose(1, 2)
        y = F.scaled_dot_product_attention(
            q,
            k,
            v,
            attn_mask=None,
            dropout_p=self.dropout if self.training else 0.0,
            is_causal=True,
        )
        y = y.transpose(1, 2).contiguous().view(batch, seq_len, channels)
        return self.c_proj(y)


class MLP(nn.Module):
    def __init__(self, config: GPTConfig) -> None:
        super().__init__()
        self.c_fc = nn.Linear(config.n_embd, 4 * config.n_embd, bias=config.bias)
        self.c_proj = nn.Linear(4 * config.n_embd, config.n_embd, bias=config.bias)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.c_fc(x)
        x = F.gelu(x)
        x = self.c_proj(x)
        return self.dropout(x)


class Block(nn.Module):
    def __init__(self, config: GPTConfig) -> None:
        super().__init__()
        self.ln_1 = LayerNorm(config.n_embd, bias=config.bias)
        self.attn = CausalSelfAttention(config)
        self.ln_2 = LayerNorm(config.n_embd, bias=config.bias)
        self.mlp = MLP(config)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x


class GPT(nn.Module):
    def __init__(self, config: GPTConfig) -> None:
        super().__init__()
        self.config = config
        self.transformer = nn.ModuleDict(
            dict(
                wte=nn.Embedding(config.vocab_size, config.n_embd),
                wpe=nn.Embedding(config.block_size, config.n_embd),
                drop=nn.Dropout(config.dropout),
                h=nn.ModuleList([Block(config) for _ in range(config.n_layer)]),
                ln_f=LayerNorm(config.n_embd, bias=config.bias),
            )
        )
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        self.transformer.wte.weight = self.lm_head.weight
        self.apply(self._init_weights)

    def _init_weights(self, module: nn.Module) -> None:
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, input_ids: torch.Tensor, labels: torch.Tensor | None = None) -> dict[str, torch.Tensor]:
        _, seq_len = input_ids.size()
        if seq_len > self.config.block_size:
            raise ValueError(f"Sequence length {seq_len} exceeds block size {self.config.block_size}")
        positions = torch.arange(0, seq_len, device=input_ids.device, dtype=torch.long)
        tok_emb = self.transformer.wte(input_ids)
        pos_emb = self.transformer.wpe(positions)[None, :, :]
        x = self.transformer.drop(tok_emb + pos_emb)
        for block in self.transformer.h:
            x = block(x)
        x = self.transformer.ln_f(x)
        logits = self.lm_head(x)
        output = {"logits": logits}
        if labels is not None:
            shift_logits = logits[:, :-1, :].contiguous()
            shift_labels = labels[:, 1:].contiguous()
            output["loss"] = F.cross_entropy(
                shift_logits.view(-1, shift_logits.size(-1)),
                shift_labels.view(-1),
            )
        return output


class LLaMAAttention(nn.Module):
    def __init__(self, config: LLaMAConfig) -> None:
        super().__init__()
        assert config.n_embd % config.n_head == 0
        self.n_head = config.n_head
        self.head_dim = config.n_embd // config.n_head
        self.dropout = config.dropout
        self.rope_theta = config.rope_theta
        self.force_eager_attention = False
        self.q_proj = nn.Linear(config.n_embd, config.n_embd, bias=config.bias)
        self.k_proj = nn.Linear(config.n_embd, config.n_embd, bias=config.bias)
        self.v_proj = nn.Linear(config.n_embd, config.n_embd, bias=config.bias)
        self.o_proj = nn.Linear(config.n_embd, config.n_embd, bias=config.bias)

    def forward(self, x: torch.Tensor, positions: torch.Tensor) -> torch.Tensor:
        batch, seq_len, channels = x.shape
        q = self.q_proj(x).view(batch, seq_len, self.n_head, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(batch, seq_len, self.n_head, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(batch, seq_len, self.n_head, self.head_dim).transpose(1, 2)
        q, k = _apply_rope(q, k, positions, self.rope_theta)
        if getattr(self, "force_eager_attention", False):
            scale = 1.0 / math.sqrt(self.head_dim)
            scores = torch.matmul(q, k.transpose(-2, -1)) * scale
            causal_mask = torch.triu(
                torch.ones(seq_len, seq_len, device=x.device, dtype=torch.bool),
                diagonal=1,
            )
            scores = scores.masked_fill(causal_mask, float("-inf"))
            attn = torch.softmax(scores, dim=-1)
            if self.training and self.dropout > 0.0:
                attn = F.dropout(attn, p=self.dropout)
            y = torch.matmul(attn, v)
        else:
            y = F.scaled_dot_product_attention(
                q,
                k,
                v,
                attn_mask=None,
                dropout_p=self.dropout if self.training else 0.0,
                is_causal=True,
            )
        y = y.transpose(1, 2).contiguous().view(batch, seq_len, channels)
        return self.o_proj(y)


class LLaMAMLP(nn.Module):
    def __init__(self, config: LLaMAConfig) -> None:
        super().__init__()
        self.gate_proj = nn.Linear(config.n_embd, config.intermediate_size, bias=config.bias)
        self.up_proj = nn.Linear(config.n_embd, config.intermediate_size, bias=config.bias)
        self.down_proj = nn.Linear(config.intermediate_size, config.n_embd, bias=config.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))


class LLaMADecoderLayer(nn.Module):
    def __init__(self, config: LLaMAConfig) -> None:
        super().__init__()
        self.input_layernorm = RMSNorm(config.n_embd, config.rms_norm_eps)
        self.self_attn = LLaMAAttention(config)
        self.post_attention_layernorm = RMSNorm(config.n_embd, config.rms_norm_eps)
        self.mlp = LLaMAMLP(config)

    def forward(self, x: torch.Tensor, positions: torch.Tensor) -> torch.Tensor:
        x = x + self.self_attn(self.input_layernorm(x), positions)
        x = x + self.mlp(self.post_attention_layernorm(x))
        return x


class LLaMA(nn.Module):
    def __init__(self, config: LLaMAConfig) -> None:
        super().__init__()
        self.config = config
        self.model = nn.ModuleDict(
            dict(
                embed_tokens=nn.Embedding(config.vocab_size, config.n_embd),
                layers=nn.ModuleList([LLaMADecoderLayer(config) for _ in range(config.n_layer)]),
                norm=RMSNorm(config.n_embd, config.rms_norm_eps),
            )
        )
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        self.apply(self._init_weights)

    def _init_weights(self, module: nn.Module) -> None:
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, input_ids: torch.Tensor, labels: torch.Tensor | None = None) -> dict[str, torch.Tensor]:
        _, seq_len = input_ids.shape
        if seq_len > self.config.block_size:
            raise ValueError(f"Sequence length {seq_len} exceeds block size {self.config.block_size}")
        positions = torch.arange(seq_len, device=input_ids.device, dtype=torch.long)
        x = self.model.embed_tokens(input_ids)
        for layer in self.model.layers:
            x = layer(x, positions)
        x = self.model.norm(x)
        logits = self.lm_head(x)
        output = {"logits": logits}
        if labels is not None:
            shift_logits = logits[:, :-1, :].contiguous()
            shift_labels = labels[:, 1:].contiguous()
            output["loss"] = F.cross_entropy(
                shift_logits.view(-1, shift_logits.size(-1)),
                shift_labels.view(-1),
            )
        return output


ModelKind = Literal["gpt", "llama"]


def build_model_from_config(model_cfg: dict[str, object]) -> nn.Module:
    model_type = str(model_cfg.get("type", "gpt")).lower()
    if model_type == "gpt":
        return GPT(
            GPTConfig(
                vocab_size=int(model_cfg["vocab_size"]),
                block_size=int(model_cfg["block_size"]),
                n_layer=int(model_cfg["n_layer"]),
                n_head=int(model_cfg["n_head"]),
                n_embd=int(model_cfg["n_embd"]),
                dropout=float(model_cfg.get("dropout", 0.1)),
                bias=bool(model_cfg.get("bias", True)),
            )
        )
    if model_type == "llama":
        return LLaMA(
            LLaMAConfig(
                vocab_size=int(model_cfg["vocab_size"]),
                block_size=int(model_cfg["block_size"]),
                n_layer=int(model_cfg["n_layer"]),
                n_head=int(model_cfg["n_head"]),
                n_embd=int(model_cfg["n_embd"]),
                intermediate_size=int(model_cfg["intermediate_size"]),
                dropout=float(model_cfg.get("dropout", 0.0)),
                bias=bool(model_cfg.get("bias", False)),
                rope_theta=float(model_cfg.get("rope_theta", 10000.0)),
                rms_norm_eps=float(model_cfg.get("rms_norm_eps", 1.0e-6)),
            )
        )
    raise ValueError(f"Unsupported model type: {model_cfg.get('type')}")
