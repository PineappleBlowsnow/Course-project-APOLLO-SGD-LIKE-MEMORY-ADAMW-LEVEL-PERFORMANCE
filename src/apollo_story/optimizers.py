from __future__ import annotations

import math
import random
from dataclasses import dataclass
from typing import Any, Iterable

import torch
from torch.optim import Optimizer


@dataclass(frozen=True)
class MatrixMeta:
    original_shape: tuple[int, ...]
    transposed: bool


def _flatten_and_orient(tensor: torch.Tensor) -> tuple[torch.Tensor, MatrixMeta]:
    flat = tensor.reshape(tensor.shape[0], -1)
    transposed = flat.shape[0] > flat.shape[1]
    oriented = flat.transpose(0, 1) if transposed else flat
    return oriented, MatrixMeta(tuple(tensor.shape), transposed)


def _restore_orientation(matrix: torch.Tensor, meta: MatrixMeta) -> torch.Tensor:
    flat = matrix.transpose(0, 1) if meta.transposed else matrix
    return flat.reshape(meta.original_shape)


def _resolve_rank(rows: int, rank: int | None, rank_ratio: float | None) -> int:
    if rank is not None:
        return max(1, min(rank, rows))
    if rank_ratio is None:
        return max(1, rows // 4)
    return max(1, min(rows, int(rows * rank_ratio)))


def _channel_scale(update_like: torch.Tensor, reference: torch.Tensor, eps: float) -> torch.Tensor:
    num = torch.linalg.vector_norm(update_like, dim=0)
    den = torch.linalg.vector_norm(reference, dim=0).clamp_min(eps)
    return num / den


def _tensor_scale(update_like: torch.Tensor, reference: torch.Tensor, eps: float) -> torch.Tensor:
    num = torch.linalg.vector_norm(update_like)
    den = torch.linalg.vector_norm(reference).clamp_min(eps)
    return num / den


def _grouped_rowwise_scale(
    update_like: torch.Tensor,
    reference: torch.Tensor,
    eps: float,
    num_groups: int,
) -> torch.Tensor:
    rows, cols = update_like.shape
    num_groups = max(1, min(num_groups, rows))
    row_indices = torch.arange(rows, device=update_like.device)
    row_groups = torch.div(row_indices * num_groups, rows, rounding_mode="floor")
    scales = torch.empty((num_groups, cols), device=update_like.device, dtype=update_like.dtype)
    for group_idx in range(num_groups):
        mask = row_groups == group_idx
        group_update = update_like[mask]
        group_reference = reference[mask]
        num = torch.linalg.vector_norm(group_update, dim=0)
        den = torch.linalg.vector_norm(group_reference, dim=0).clamp_min(eps)
        scales[group_idx] = num / den
    return scales


def _random_projection(rows: int, rank: int, device: torch.device, seed: int) -> torch.Tensor:
    generator = torch.Generator(device=device.type if device.type != "mps" else "cpu")
    generator.manual_seed(seed)
    projection = torch.randn(rank, rows, device=device, dtype=torch.float32, generator=generator)
    return projection / math.sqrt(rank)


def _svd_projection(matrix: torch.Tensor, rank: int) -> torch.Tensor:
    left, _, _ = torch.linalg.svd(matrix, full_matrices=False)
    return left[:, :rank].transpose(0, 1).contiguous()


class TraceableOptimizer(Optimizer):
    def __init__(self, params, defaults) -> None:
        super().__init__(params, defaults)
        self._pending_traces: list[dict[str, Any]] = []

    def pop_traces(self) -> list[dict[str, Any]]:
        traces = self._pending_traces
        self._pending_traces = []
        return traces

    def predicted_update_tensors(
        self,
        params: list[torch.Tensor],
        grads: list[torch.Tensor],
    ) -> list[torch.Tensor]:
        grad_by_param = {id(param): grad.detach().float() for param, grad in zip(params, grads)}
        updates: dict[int, torch.Tensor] = {}
        for group in self.param_groups:
            for param in group["params"]:
                grad = grad_by_param.get(id(param))
                if grad is None:
                    continue
                updates[id(param)] = self._predicted_update_for_param(group, param, grad)
        return [updates.get(id(param), torch.zeros_like(param, dtype=torch.float32)) for param in params]

    def _predicted_update_for_param(
        self,
        group: dict[str, Any],
        param: torch.Tensor,
        grad: torch.Tensor,
    ) -> torch.Tensor:
        raise NotImplementedError

    def _record_scaling(
        self,
        group: dict[str, Any],
        step: int,
        scale: torch.Tensor | float,
        tag: str,
        rank: int | None = None,
    ) -> None:
        tracked_steps = set(group.get("scaling_log_steps", []))
        tracked_params = set(group.get("track_param_names", []))
        if step not in tracked_steps or group.get("name") not in tracked_params:
            return
        if isinstance(scale, torch.Tensor):
            values = scale.detach().float().cpu().tolist()
            mean = float(scale.mean().item())
            std = float(scale.std(unbiased=False).item()) if scale.numel() > 1 else 0.0
        else:
            values = [float(scale)]
            mean = float(scale)
            std = 0.0
        self._pending_traces.append(
            {
                "step": step,
                "param_name": group.get("name"),
                "optimizer": self.__class__.__name__,
                "tag": tag,
                "rank": rank,
                "mean": mean,
                "std": std,
                "values": values,
            }
        )


def build_named_param_groups(model: torch.nn.Module, config: dict[str, Any]) -> list[dict[str, Any]]:
    train_cfg = config["training"]
    analysis_cfg = config["analysis"]
    groups: list[dict[str, Any]] = []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        groups.append(
            {
                "params": [param],
                "name": name,
                "weight_decay": train_cfg["weight_decay"] if param.dim() >= 2 else 0.0,
                "track_param_names": analysis_cfg.get("track_param_names", []),
                "scaling_log_steps": analysis_cfg.get("scaling_log_steps", []),
            }
        )
    return groups


def _init_like(state: dict[str, Any], key: str, ref: torch.Tensor) -> torch.Tensor:
    tensor = state.get(key)
    if tensor is None or tensor.shape != ref.shape:
        tensor = torch.zeros_like(ref, dtype=torch.float32)
        state[key] = tensor
    return tensor


def _apply_norm_limiter(update: torch.Tensor, state: dict[str, Any], enabled: bool, gamma: float) -> torch.Tensor:
    if not enabled:
        return update
    norm = torch.linalg.vector_norm(update).item()
    prev = state.get("prev_update_norm")
    if prev is not None and prev > 0.0 and norm > gamma * prev:
        update = update * (gamma * prev / max(norm, 1e-12))
        norm = torch.linalg.vector_norm(update).item()
    state["prev_update_norm"] = norm
    return update


def _predict_norm_limited_update(update: torch.Tensor, state: dict[str, Any], enabled: bool, gamma: float) -> torch.Tensor:
    if not enabled:
        return update
    norm = torch.linalg.vector_norm(update).item()
    prev = state.get("prev_update_norm")
    if prev is not None and prev > 0.0 and norm > gamma * prev:
        update = update * (gamma * prev / max(norm, 1e-12))
    return update


def _predict_adam_update(
    grad: torch.Tensor,
    state: dict[str, Any],
    beta1: float,
    beta2: float,
    eps: float,
) -> tuple[torch.Tensor, int]:
    next_step = int(state.get("step", 0)) + 1
    exp_avg = state.get("exp_avg")
    exp_avg_sq = state.get("exp_avg_sq")
    if exp_avg is None or exp_avg.shape != grad.shape:
        exp_avg = torch.zeros_like(grad, dtype=torch.float32)
    else:
        exp_avg = exp_avg.detach().float()
    if exp_avg_sq is None or exp_avg_sq.shape != grad.shape:
        exp_avg_sq = torch.zeros_like(grad, dtype=torch.float32)
    else:
        exp_avg_sq = exp_avg_sq.detach().float()
    exp_avg = exp_avg * beta1 + grad * (1.0 - beta1)
    exp_avg_sq = exp_avg_sq * beta2 + grad * grad * (1.0 - beta2)
    m_hat = exp_avg / (1.0 - beta1**next_step)
    v_hat = exp_avg_sq / (1.0 - beta2**next_step)
    return m_hat / (torch.sqrt(v_hat) + eps), next_step


class SGDMomentum(TraceableOptimizer):
    def __init__(self, params, lr: float, momentum: float, weight_decay: float) -> None:
        defaults = dict(lr=lr, momentum=momentum, weight_decay=weight_decay)
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        for group in self.param_groups:
            for param in group["params"]:
                if param.grad is None:
                    continue
                grad = param.grad.detach().float()
                state = self.state[param]
                velocity = _init_like(state, "velocity", grad)
                if group["weight_decay"] != 0.0:
                    grad = grad.add(param.detach().float(), alpha=group["weight_decay"])
                velocity.mul_(group["momentum"]).add_(grad)
                param.add_(velocity.to(param.dtype), alpha=-group["lr"])
        return loss

    def _predicted_update_for_param(
        self,
        group: dict[str, Any],
        param: torch.Tensor,
        grad: torch.Tensor,
    ) -> torch.Tensor:
        state = self.state[param]
        velocity = state.get("velocity")
        if velocity is None or velocity.shape != grad.shape:
            velocity = torch.zeros_like(grad, dtype=torch.float32)
        else:
            velocity = velocity.detach().float()
        if group["weight_decay"] != 0.0:
            grad = grad.add(param.detach().float(), alpha=group["weight_decay"])
        return velocity * group["momentum"] + grad


class AdamWTracked(TraceableOptimizer):
    def __init__(
        self,
        params,
        lr: float,
        betas: tuple[float, float],
        eps: float,
        norm_limiter: bool,
        norm_limiter_gamma: float,
    ) -> None:
        defaults = dict(
            lr=lr,
            betas=betas,
            eps=eps,
            norm_limiter=norm_limiter,
            norm_limiter_gamma=norm_limiter_gamma,
        )
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        for group in self.param_groups:
            beta1, beta2 = group["betas"]
            for param in group["params"]:
                if param.grad is None:
                    continue
                grad = param.grad.detach().float()
                state = self.state[param]
                state["step"] = state.get("step", 0) + 1
                step = state["step"]
                if group["weight_decay"] != 0.0:
                    param.mul_(1.0 - group["lr"] * group["weight_decay"])
                exp_avg = _init_like(state, "exp_avg", grad)
                exp_avg_sq = _init_like(state, "exp_avg_sq", grad)
                exp_avg.mul_(beta1).add_(grad, alpha=1.0 - beta1)
                exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1.0 - beta2)
                m_hat = exp_avg / (1.0 - beta1**step)
                v_hat = exp_avg_sq / (1.0 - beta2**step)
                update = m_hat / (torch.sqrt(v_hat) + group["eps"])

                if grad.ndim >= 2:
                    grad_matrix, meta = _flatten_and_orient(grad)
                    update_matrix, _ = _flatten_and_orient(update)
                    scale = _channel_scale(update_matrix, grad_matrix, group["eps"])
                    self._record_scaling(group, step, scale, tag="adamw_channel_scale", rank=None)

                update = _apply_norm_limiter(
                    update,
                    state,
                    group["norm_limiter"],
                    group["norm_limiter_gamma"],
                )
                param.add_(update.to(param.dtype), alpha=-group["lr"])
        return loss

    def _predicted_update_for_param(
        self,
        group: dict[str, Any],
        param: torch.Tensor,
        grad: torch.Tensor,
    ) -> torch.Tensor:
        beta1, beta2 = group["betas"]
        state = self.state[param]
        update, _ = _predict_adam_update(grad, state, beta1, beta2, group["eps"])
        return _predict_norm_limited_update(
            update,
            state,
            group["norm_limiter"],
            group["norm_limiter_gamma"],
        )


class StructuredAdam(TraceableOptimizer):
    def __init__(
        self,
        params,
        lr: float,
        betas: tuple[float, float],
        eps: float,
        granularity: str,
        scale: float,
        num_groups: int,
        norm_limiter: bool,
        norm_limiter_gamma: float,
    ) -> None:
        defaults = dict(
            lr=lr,
            betas=betas,
            eps=eps,
            granularity=granularity,
            scale=scale,
            num_groups=num_groups,
            norm_limiter=norm_limiter,
            norm_limiter_gamma=norm_limiter_gamma,
        )
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        for group in self.param_groups:
            beta1, beta2 = group["betas"]
            for param in group["params"]:
                if param.grad is None:
                    continue
                grad = param.grad.detach().float()
                state = self.state[param]
                state["step"] = state.get("step", 0) + 1
                step = state["step"]
                if group["weight_decay"] != 0.0:
                    param.mul_(1.0 - group["lr"] * group["weight_decay"])

                exp_avg = _init_like(state, "exp_avg", grad)
                exp_avg_sq = _init_like(state, "exp_avg_sq", grad)
                exp_avg.mul_(beta1).add_(grad, alpha=1.0 - beta1)
                exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1.0 - beta2)
                m_hat = exp_avg / (1.0 - beta1**step)
                v_hat = exp_avg_sq / (1.0 - beta2**step)
                elementwise_update = m_hat / (torch.sqrt(v_hat) + group["eps"])

                if grad.ndim < 2:
                    update = elementwise_update
                else:
                    grad_matrix, meta = _flatten_and_orient(grad)
                    update_matrix, _ = _flatten_and_orient(elementwise_update)
                    if group["granularity"] == "tensor":
                        scale = _tensor_scale(update_matrix, grad_matrix, group["eps"]) * group["scale"]
                        structured_update = grad_matrix * scale
                    elif group["granularity"] == "grouped_rowwise":
                        scale = _grouped_rowwise_scale(
                            update_matrix,
                            grad_matrix,
                            group["eps"],
                            group["num_groups"],
                        ) * group["scale"]
                        rows, _ = grad_matrix.shape
                        row_indices = torch.arange(rows, device=grad_matrix.device)
                        row_groups = torch.div(
                            row_indices * group["num_groups"],
                            rows,
                            rounding_mode="floor",
                        )
                        structured_update = grad_matrix * scale[row_groups]
                    else:
                        scale = _channel_scale(update_matrix, grad_matrix, group["eps"]) * group["scale"]
                        structured_update = grad_matrix * scale.unsqueeze(0)
                    self._record_scaling(
                        group,
                        step,
                        scale,
                        tag=f"structured_{group['granularity']}_scale",
                        rank=None,
                    )
                    update = _restore_orientation(structured_update, meta)

                update = _apply_norm_limiter(
                    update,
                    state,
                    group["norm_limiter"],
                    group["norm_limiter_gamma"],
                )
                param.add_(update.to(param.dtype), alpha=-group["lr"])
        return loss

    def _predicted_update_for_param(
        self,
        group: dict[str, Any],
        param: torch.Tensor,
        grad: torch.Tensor,
    ) -> torch.Tensor:
        beta1, beta2 = group["betas"]
        state = self.state[param]
        elementwise_update, _ = _predict_adam_update(grad, state, beta1, beta2, group["eps"])

        if grad.ndim < 2:
            update = elementwise_update
        else:
            grad_matrix, meta = _flatten_and_orient(grad)
            update_matrix, _ = _flatten_and_orient(elementwise_update)
            if group["granularity"] == "tensor":
                scale = _tensor_scale(update_matrix, grad_matrix, group["eps"]) * group["scale"]
                structured_update = grad_matrix * scale
            elif group["granularity"] == "grouped_rowwise":
                scale = _grouped_rowwise_scale(
                    update_matrix,
                    grad_matrix,
                    group["eps"],
                    group["num_groups"],
                ) * group["scale"]
                rows, _ = grad_matrix.shape
                row_indices = torch.arange(rows, device=grad_matrix.device)
                row_groups = torch.div(
                    row_indices * group["num_groups"],
                    rows,
                    rounding_mode="floor",
                )
                structured_update = grad_matrix * scale[row_groups]
            else:
                scale = _channel_scale(update_matrix, grad_matrix, group["eps"]) * group["scale"]
                structured_update = grad_matrix * scale.unsqueeze(0)
            update = _restore_orientation(structured_update, meta)
        return _predict_norm_limited_update(
            update,
            state,
            group["norm_limiter"],
            group["norm_limiter_gamma"],
        )


class GaLore(TraceableOptimizer):
    def __init__(
        self,
        params,
        lr: float,
        betas: tuple[float, float],
        eps: float,
        rank: int | None,
        rank_ratio: float | None,
        projection_update_gap: int,
    ) -> None:
        defaults = dict(
            lr=lr,
            betas=betas,
            eps=eps,
            rank=rank,
            rank_ratio=rank_ratio,
            projection_update_gap=projection_update_gap,
        )
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        for group in self.param_groups:
            beta1, beta2 = group["betas"]
            for param in group["params"]:
                if param.grad is None:
                    continue
                grad = param.grad.detach().float()
                state = self.state[param]
                state["step"] = state.get("step", 0) + 1
                step = state["step"]
                if group["weight_decay"] != 0.0:
                    param.mul_(1.0 - group["lr"] * group["weight_decay"])

                if grad.ndim < 2:
                    exp_avg = _init_like(state, "exp_avg", grad)
                    exp_avg_sq = _init_like(state, "exp_avg_sq", grad)
                    exp_avg.mul_(beta1).add_(grad, alpha=1.0 - beta1)
                    exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1.0 - beta2)
                    m_hat = exp_avg / (1.0 - beta1**step)
                    v_hat = exp_avg_sq / (1.0 - beta2**step)
                    update = m_hat / (torch.sqrt(v_hat) + group["eps"])
                else:
                    grad_matrix, meta = _flatten_and_orient(grad)
                    rows, cols = grad_matrix.shape
                    rank = _resolve_rank(rows, group["rank"], group["rank_ratio"])
                    if "projection" not in state or (step - 1) % group["projection_update_gap"] == 0:
                        state["projection"] = _svd_projection(grad_matrix, rank)
                    projection = state["projection"]
                    low_rank_grad = projection @ grad_matrix

                    exp_avg_r = state.get("exp_avg_r")
                    exp_avg_sq_r = state.get("exp_avg_sq_r")
                    if exp_avg_r is None or exp_avg_r.shape != low_rank_grad.shape:
                        exp_avg_r = torch.zeros_like(low_rank_grad, dtype=torch.float32)
                        exp_avg_sq_r = torch.zeros_like(low_rank_grad, dtype=torch.float32)
                        state["exp_avg_r"] = exp_avg_r
                        state["exp_avg_sq_r"] = exp_avg_sq_r
                    exp_avg_r.mul_(beta1).add_(low_rank_grad, alpha=1.0 - beta1)
                    exp_avg_sq_r.mul_(beta2).addcmul_(low_rank_grad, low_rank_grad, value=1.0 - beta2)
                    m_hat_r = exp_avg_r / (1.0 - beta1**step)
                    v_hat_r = exp_avg_sq_r / (1.0 - beta2**step)
                    update_low = m_hat_r / (torch.sqrt(v_hat_r) + group["eps"])
                    update = _restore_orientation(projection.transpose(0, 1) @ update_low, meta)

                param.add_(update.to(param.dtype), alpha=-group["lr"])
        return loss

    def _predicted_update_for_param(
        self,
        group: dict[str, Any],
        param: torch.Tensor,
        grad: torch.Tensor,
    ) -> torch.Tensor:
        beta1, beta2 = group["betas"]
        state = self.state[param]
        next_step = int(state.get("step", 0)) + 1
        if grad.ndim < 2:
            update, _ = _predict_adam_update(grad, state, beta1, beta2, group["eps"])
            return update

        grad_matrix, meta = _flatten_and_orient(grad)
        rows, _ = grad_matrix.shape
        rank = _resolve_rank(rows, group["rank"], group["rank_ratio"])
        projection = state.get("projection")
        if projection is None or projection.shape != (rank, rows) or (next_step - 1) % group["projection_update_gap"] == 0:
            projection = _svd_projection(grad_matrix, rank)
        else:
            projection = projection.detach().float()
        low_rank_grad = projection @ grad_matrix
        exp_avg_r = state.get("exp_avg_r")
        exp_avg_sq_r = state.get("exp_avg_sq_r")
        if exp_avg_r is None or exp_avg_r.shape != low_rank_grad.shape:
            exp_avg_r = torch.zeros_like(low_rank_grad, dtype=torch.float32)
            exp_avg_sq_r = torch.zeros_like(low_rank_grad, dtype=torch.float32)
        else:
            exp_avg_r = exp_avg_r.detach().float()
            exp_avg_sq_r = exp_avg_sq_r.detach().float()
        exp_avg_r = exp_avg_r * beta1 + low_rank_grad * (1.0 - beta1)
        exp_avg_sq_r = exp_avg_sq_r * beta2 + low_rank_grad * low_rank_grad * (1.0 - beta2)
        m_hat_r = exp_avg_r / (1.0 - beta1**next_step)
        v_hat_r = exp_avg_sq_r / (1.0 - beta2**next_step)
        update_low = m_hat_r / (torch.sqrt(v_hat_r) + group["eps"])
        return _restore_orientation(projection.transpose(0, 1) @ update_low, meta)


class Apollo(TraceableOptimizer):
    def __init__(
        self,
        params,
        lr: float,
        betas: tuple[float, float],
        eps: float,
        rank: int | None,
        rank_ratio: float | None,
        projection: str,
        granularity: str,
        scale: float,
        projection_update_gap: int,
        norm_limiter: bool,
        norm_limiter_gamma: float,
    ) -> None:
        defaults = dict(
            lr=lr,
            betas=betas,
            eps=eps,
            rank=rank,
            rank_ratio=rank_ratio,
            projection=projection,
            granularity=granularity,
            scale=scale,
            projection_update_gap=projection_update_gap,
            norm_limiter=norm_limiter,
            norm_limiter_gamma=norm_limiter_gamma,
        )
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        for group in self.param_groups:
            beta1, beta2 = group["betas"]
            for param in group["params"]:
                if param.grad is None:
                    continue
                grad = param.grad.detach().float()
                state = self.state[param]
                state["step"] = state.get("step", 0) + 1
                step = state["step"]
                if group["weight_decay"] != 0.0:
                    param.mul_(1.0 - group["lr"] * group["weight_decay"])

                if grad.ndim < 2:
                    exp_avg = _init_like(state, "exp_avg", grad)
                    exp_avg_sq = _init_like(state, "exp_avg_sq", grad)
                    exp_avg.mul_(beta1).add_(grad, alpha=1.0 - beta1)
                    exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1.0 - beta2)
                    m_hat = exp_avg / (1.0 - beta1**step)
                    v_hat = exp_avg_sq / (1.0 - beta2**step)
                    update = m_hat / (torch.sqrt(v_hat) + group["eps"])
                else:
                    grad_matrix, meta = _flatten_and_orient(grad)
                    rows, cols = grad_matrix.shape
                    rank = _resolve_rank(rows, group["rank"], group["rank_ratio"])
                    if group["projection"] == "random":
                        if "proj_seed" not in state or (step - 1) % group["projection_update_gap"] == 0:
                            state["proj_seed"] = random.randint(0, 2**31 - 1)
                        projection = _random_projection(rows, rank, grad_matrix.device, state["proj_seed"])
                    elif group["projection"] == "svd":
                        projection = _svd_projection(grad_matrix, rank)
                    else:
                        raise ValueError(f"Unsupported projection type: {group['projection']}")

                    low_rank_grad = projection @ grad_matrix
                    exp_avg_r = state.get("exp_avg_r")
                    exp_avg_sq_r = state.get("exp_avg_sq_r")
                    if exp_avg_r is None or exp_avg_r.shape != low_rank_grad.shape:
                        exp_avg_r = torch.zeros_like(low_rank_grad, dtype=torch.float32)
                        exp_avg_sq_r = torch.zeros_like(low_rank_grad, dtype=torch.float32)
                        state["exp_avg_r"] = exp_avg_r
                        state["exp_avg_sq_r"] = exp_avg_sq_r
                    exp_avg_r.mul_(beta1).add_(low_rank_grad, alpha=1.0 - beta1)
                    exp_avg_sq_r.mul_(beta2).addcmul_(low_rank_grad, low_rank_grad, value=1.0 - beta2)
                    m_hat_r = exp_avg_r / (1.0 - beta1**step)
                    v_hat_r = exp_avg_sq_r / (1.0 - beta2**step)
                    adaptive_r = m_hat_r / (torch.sqrt(v_hat_r) + group["eps"])

                    if group["granularity"] == "tensor":
                        scale = _tensor_scale(adaptive_r, low_rank_grad, group["eps"]) * group["scale"]
                        update_matrix = grad_matrix * scale
                    else:
                        scale = _channel_scale(adaptive_r, low_rank_grad, group["eps"]) * group["scale"]
                        update_matrix = grad_matrix * scale.unsqueeze(0)

                    self._record_scaling(group, step, scale, tag=f"apollo_{group['granularity']}_scale", rank=rank)
                    update_matrix = _apply_norm_limiter(
                        update_matrix,
                        state,
                        group["norm_limiter"],
                        group["norm_limiter_gamma"],
                    )
                    update = _restore_orientation(update_matrix, meta)

                param.add_(update.to(param.dtype), alpha=-group["lr"])
        return loss

    def _predicted_update_for_param(
        self,
        group: dict[str, Any],
        param: torch.Tensor,
        grad: torch.Tensor,
    ) -> torch.Tensor:
        beta1, beta2 = group["betas"]
        state = self.state[param]
        next_step = int(state.get("step", 0)) + 1
        if grad.ndim < 2:
            update, _ = _predict_adam_update(grad, state, beta1, beta2, group["eps"])
            return update

        grad_matrix, meta = _flatten_and_orient(grad)
        rows, _ = grad_matrix.shape
        rank = _resolve_rank(rows, group["rank"], group["rank_ratio"])
        if group["projection"] == "random":
            stored_seed = state.get("proj_seed")
            if stored_seed is None:
                stored_seed = 0
            projection = _random_projection(rows, rank, grad_matrix.device, int(stored_seed))
        elif group["projection"] == "svd":
            projection = _svd_projection(grad_matrix, rank)
        else:
            raise ValueError(f"Unsupported projection type: {group['projection']}")

        low_rank_grad = projection @ grad_matrix
        exp_avg_r = state.get("exp_avg_r")
        exp_avg_sq_r = state.get("exp_avg_sq_r")
        if exp_avg_r is None or exp_avg_r.shape != low_rank_grad.shape:
            exp_avg_r = torch.zeros_like(low_rank_grad, dtype=torch.float32)
            exp_avg_sq_r = torch.zeros_like(low_rank_grad, dtype=torch.float32)
        else:
            exp_avg_r = exp_avg_r.detach().float()
            exp_avg_sq_r = exp_avg_sq_r.detach().float()
        exp_avg_r = exp_avg_r * beta1 + low_rank_grad * (1.0 - beta1)
        exp_avg_sq_r = exp_avg_sq_r * beta2 + low_rank_grad * low_rank_grad * (1.0 - beta2)
        m_hat_r = exp_avg_r / (1.0 - beta1**next_step)
        v_hat_r = exp_avg_sq_r / (1.0 - beta2**next_step)
        adaptive_r = m_hat_r / (torch.sqrt(v_hat_r) + group["eps"])

        if group["granularity"] == "tensor":
            scale = _tensor_scale(adaptive_r, low_rank_grad, group["eps"]) * group["scale"]
            update_matrix = grad_matrix * scale
        else:
            scale = _channel_scale(adaptive_r, low_rank_grad, group["eps"]) * group["scale"]
            update_matrix = grad_matrix * scale.unsqueeze(0)

        update_matrix = _predict_norm_limited_update(
            update_matrix,
            state,
            group["norm_limiter"],
            group["norm_limiter_gamma"],
        )
        return _restore_orientation(update_matrix, meta)


class ApolloMini(Apollo):
    def __init__(self, params, **kwargs) -> None:
        kwargs.setdefault("rank", 1)
        kwargs.setdefault("rank_ratio", None)
        kwargs.setdefault("granularity", "tensor")
        kwargs.setdefault("scale", 64.0)
        super().__init__(params, **kwargs)


def build_optimizer(model: torch.nn.Module, config: dict[str, Any]) -> TraceableOptimizer:
    groups = build_named_param_groups(model, config)
    opt_cfg = config["optimizer"]
    train_cfg = config["training"]
    name = opt_cfg["name"].lower()

    if name == "sgd":
        return SGDMomentum(
            groups,
            lr=train_cfg["learning_rate"],
            momentum=train_cfg["momentum"],
            weight_decay=train_cfg["weight_decay"],
        )
    if name in {"adamw", "elementwise"}:
        return AdamWTracked(
            groups,
            lr=train_cfg["learning_rate"],
            betas=tuple(train_cfg["betas"]),
            eps=train_cfg["eps"],
            norm_limiter=opt_cfg.get("norm_limiter", False),
            norm_limiter_gamma=opt_cfg.get("norm_limiter_gamma", 1.01),
        )
    if name == "galore":
        return GaLore(
            groups,
            lr=train_cfg["learning_rate"],
            betas=tuple(train_cfg["betas"]),
            eps=train_cfg["eps"],
            rank=opt_cfg.get("rank"),
            rank_ratio=opt_cfg.get("rank_ratio"),
            projection_update_gap=opt_cfg.get("projection_update_gap", 200),
        )
    if name in {"structured", "channelwise", "tensorwise", "channelwise_nl", "grouped_rowwise_k2", "grouped_rowwise_k4"}:
        granularity = opt_cfg.get("granularity", "channel")
        norm_limiter = opt_cfg.get("norm_limiter", False)
        num_groups = int(opt_cfg.get("num_groups", 1))
        if name == "channelwise":
            granularity = "channel"
            norm_limiter = False
        elif name == "tensorwise":
            granularity = "tensor"
            norm_limiter = False
        elif name == "channelwise_nl":
            granularity = "channel"
            norm_limiter = True
        elif name == "grouped_rowwise_k2":
            granularity = "grouped_rowwise"
            num_groups = 2
            norm_limiter = False
        elif name == "grouped_rowwise_k4":
            granularity = "grouped_rowwise"
            num_groups = 4
            norm_limiter = False
        return StructuredAdam(
            groups,
            lr=train_cfg["learning_rate"],
            betas=tuple(train_cfg["betas"]),
            eps=train_cfg["eps"],
            granularity=granularity,
            scale=opt_cfg.get("scale", 1.0),
            num_groups=num_groups,
            norm_limiter=norm_limiter,
            norm_limiter_gamma=opt_cfg.get("norm_limiter_gamma", 1.01),
        )
    if name == "apollo":
        return Apollo(
            groups,
            lr=train_cfg["learning_rate"],
            betas=tuple(train_cfg["betas"]),
            eps=train_cfg["eps"],
            rank=opt_cfg.get("rank"),
            rank_ratio=opt_cfg.get("rank_ratio"),
            projection=opt_cfg.get("projection", "random"),
            granularity=opt_cfg.get("granularity", "channel"),
            scale=opt_cfg.get("scale", 1.0),
            projection_update_gap=opt_cfg.get("projection_update_gap", 200),
            norm_limiter=opt_cfg.get("norm_limiter", False),
            norm_limiter_gamma=opt_cfg.get("norm_limiter_gamma", 1.01),
        )
    if name == "apollo_mini":
        return ApolloMini(
            groups,
            lr=train_cfg["learning_rate"],
            betas=tuple(train_cfg["betas"]),
            eps=train_cfg["eps"],
            rank=opt_cfg.get("rank", 1),
            projection=opt_cfg.get("projection", "random"),
            scale=opt_cfg.get("scale", 64.0),
            projection_update_gap=opt_cfg.get("projection_update_gap", 200),
            norm_limiter=opt_cfg.get("norm_limiter", False),
            norm_limiter_gamma=opt_cfg.get("norm_limiter_gamma", 1.01),
        )
    raise ValueError(f"Unsupported optimizer: {opt_cfg['name']}")
