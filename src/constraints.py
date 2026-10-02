"""Normalized control bounds and per-step rate limits for inverse optimization."""

import math

import torch
from torch import nn
from torch.nn import functional as F


CONTROL_NAMES = (
    "u1_feed_rate",
    "u2_water_rate",
    "u3_reagent_rate",
    "u4_speed",
    "u5_temperature_setpoint",
)


class ControlConstraints(nn.Module):
    """Shared bounds for one control vector or an (optional batched) MPC horizon.

    Controls are dimensionless [0, 1] actuator commands, NOT the physical-unit
    features used by the trained surrogate. Convert to physical units before
    passing constrained controls to that model or InverseProcessObjective.
    A one-dimensional tensor is one step; (..., horizon, 5) is a trajectory.
    """

    def __init__(
        self,
        lower: torch.Tensor | None = None,
        upper: torch.Tensor | None = None,
        max_move: float = 0.05,
    ) -> None:
        super().__init__()
        lower = torch.tensor([0.2, 0.1, 0.0, 0.2, 0.1]) if lower is None else torch.as_tensor(lower)
        upper = torch.tensor([1.0, 0.9, 1.0, 1.0, 0.9]) if upper is None else torch.as_tensor(upper)
        if lower.shape != upper.shape or lower.ndim != 1 or lower.numel() == 0:
            raise ValueError("Bounds must be nonempty vectors of equal length")
        if not lower.is_floating_point() or not upper.is_floating_point():
            raise ValueError("Bounds must use floating point tensors")
        if not torch.isfinite(lower).all() or not torch.isfinite(upper).all() or (lower >= upper).any():
            raise ValueError("Bounds must be finite and strictly ordered")
        if not math.isfinite(max_move) or max_move <= 0:
            raise ValueError("max_move must be finite and positive")
        self.register_buffer("lower", lower.detach().clone())
        self.register_buffer("upper", upper.detach().clone().to(lower))
        self.max_move = max_move

    def _check(self, controls: torch.Tensor, previous: torch.Tensor) -> None:
        if controls.ndim < 1 or controls.shape[-1] != self.lower.numel():
            raise ValueError(f"Controls must end with {self.lower.numel()} components")
        expected_previous_shape = (
            controls.shape[:-2] + (self.lower.numel(),)
            if controls.ndim > 1 else controls.shape
        )
        if previous.shape != expected_previous_shape:
            raise ValueError("Previous controls must match the trajectory's batch shape")
        if controls.ndim > 1 and controls.shape[-2] == 0:
            raise ValueError("Control trajectories must contain at least one step")
        if controls.dtype != self.lower.dtype or previous.dtype != self.lower.dtype:
            raise ValueError("Controls and previous controls must match the bounds dtype")
        if controls.device != self.lower.device or previous.device != self.lower.device:
            raise ValueError("Controls and previous controls must match the bounds device")
        if not torch.isfinite(controls).all() or not torch.isfinite(previous).all():
            raise ValueError("Controls and previous controls must be finite")
        if (previous < self.lower).any() or (previous > self.upper).any():
            raise ValueError("Previous controls must satisfy actuator bounds")

    def _trajectory(self, controls: torch.Tensor) -> torch.Tensor:
        return controls.unsqueeze(-2) if controls.ndim == 1 else controls

    def _step_bounds(self, previous: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        return torch.maximum(self.lower, previous - self.max_move), torch.minimum(
            self.upper, previous + self.max_move
        )


class ControlProjection(ControlConstraints):
    """Project controls onto actuator and sequential rate constraints exactly."""

    def forward(self, controls: torch.Tensor, previous: torch.Tensor) -> torch.Tensor:
        self._check(controls, previous)
        trajectory = self._trajectory(controls)
        projected = []
        last = previous
        for step in trajectory.unbind(dim=-2):
            lower, upper = self._step_bounds(last)
            last = torch.minimum(torch.maximum(step, lower), upper)
            projected.append(last)
        result = torch.stack(projected, dim=-2)
        return result.squeeze(-2) if controls.ndim == 1 else result


class SoftplusConstraintPenalty(ControlConstraints):
    """Smooth scalar penalty for bounds and consecutive control moves.

    Penalties are squared softplus violations in units of the bound width and
    max_move respectively. They are positive even on the feasible side of a
    boundary, unlike exact projection.
    """

    def __init__(self, *args, sharpness: float = 20.0, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        if not math.isfinite(sharpness) or sharpness <= 0:
            raise ValueError("sharpness must be finite and positive")
        self.sharpness = sharpness

    def forward(self, controls: torch.Tensor, previous: torch.Tensor) -> torch.Tensor:
        self._check(controls, previous)
        trajectory = self._trajectory(controls)
        widths = self.upper - self.lower
        violations = (
            (self.lower - trajectory) / widths,
            (trajectory - self.upper) / widths,
        )
        steps = torch.cat((previous.unsqueeze(-2), trajectory[..., :-1, :]), dim=-2)
        changes = (trajectory - steps) / self.max_move
        violations += (changes - 1, -changes - 1)
        return sum(
            (F.softplus(value * self.sharpness) / self.sharpness).square().sum()
            for value in violations
        )


class SigmoidControlParameterization(ControlConstraints):
    """Map unconstrained logits into bounds and rate-feasible control commands."""

    def forward(self, logits: torch.Tensor, previous: torch.Tensor) -> torch.Tensor:
        self._check(logits, previous)
        trajectory = self._trajectory(logits)
        commands = []
        last = previous
        for step in trajectory.unbind(dim=-2):
            lower, upper = self._step_bounds(last)
            last = lower + (upper - lower) * torch.sigmoid(step)
            commands.append(last)
        result = torch.stack(commands, dim=-2)
        return result.squeeze(-2) if logits.ndim == 1 else result
