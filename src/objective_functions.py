"""Differentiable inverse-optimization objective for a frozen process surrogate."""

import math

import torch
from torch import nn


class InverseProcessObjective(nn.Module):
    """Minimize energy and control moves while rewarding predicted quality.

    Bounds and support limits are in physical units and follow the checkpoint's
    input order (five disturbances, then five controls). The surrogate predicts
    standardized outputs, as in notebooks/02_train_model.ipynb. The objective
    accepts one fixed disturbance vector and one candidate control vector.
    """

    def __init__(
        self,
        model: nn.Module,
        input_mean: torch.Tensor,
        input_scale: torch.Tensor,
        output_mean: torch.Tensor,
        output_scale: torch.Tensor,
        control_lower: torch.Tensor,
        control_upper: torch.Tensor,
        support_lower: torch.Tensor,
        support_upper: torch.Tensor,
        *,
        quality_min: float | None = None,
        energy_max: float | None = None,
        quality_weight: float = 1.0,
        energy_weight: float = 1.0,
        movement_weight: float = 0.1,
        constraint_weight: float = 10.0,
        support_weight: float = 10.0,
    ) -> None:
        super().__init__()
        parameters = tuple(model.parameters())
        if not parameters:
            raise ValueError("The surrogate must have parameters")
        device, dtype = parameters[0].device, parameters[0].dtype

        def vector(value: torch.Tensor, length: int, name: str) -> torch.Tensor:
            tensor = torch.as_tensor(value, device=device, dtype=dtype).detach().clone()
            if tensor.shape != (length,) or not torch.isfinite(tensor).all():
                raise ValueError(f"{name} must be a finite vector of length {length}")
            return tensor

        self.register_buffer("input_mean", vector(input_mean, 10, "input_mean"))
        self.register_buffer("input_scale", vector(input_scale, 10, "input_scale"))
        self.register_buffer("output_mean", vector(output_mean, 2, "output_mean"))
        self.register_buffer("output_scale", vector(output_scale, 2, "output_scale"))
        self.register_buffer("control_lower", vector(control_lower, 5, "control_lower"))
        self.register_buffer("control_upper", vector(control_upper, 5, "control_upper"))
        self.register_buffer("support_lower", vector(support_lower, 10, "support_lower"))
        self.register_buffer("support_upper", vector(support_upper, 10, "support_upper"))
        if (self.input_scale <= 0).any() or (self.output_scale <= 0).any():
            raise ValueError("Scaler scales must be positive")
        if (self.control_lower >= self.control_upper).any():
            raise ValueError("Control lower bounds must be below upper bounds")
        if (self.support_lower >= self.support_upper).any():
            raise ValueError("Support lower bounds must be below upper bounds")

        weights = (quality_weight, energy_weight, movement_weight,
                   constraint_weight, support_weight)
        if any(not math.isfinite(weight) or weight < 0 for weight in weights):
            raise ValueError("Objective weights must be finite and nonnegative")
        for name, limit in (("quality_min", quality_min), ("energy_max", energy_max)):
            if limit is not None and not math.isfinite(limit):
                raise ValueError(f"{name} must be finite")
        self.quality_min = quality_min
        self.energy_max = energy_max
        self.quality_weight = quality_weight
        self.energy_weight = energy_weight
        self.movement_weight = movement_weight
        self.constraint_weight = constraint_weight
        self.support_weight = support_weight

        self.model = model.eval()
        self.model.requires_grad_(False)

    def forward(
        self,
        disturbances: torch.Tensor,
        controls: torch.Tensor,
        previous_controls: torch.Tensor,
    ) -> torch.Tensor:
        """Return a scalar loss differentiable with respect to ``controls``."""
        for name, value in (("disturbances", disturbances), ("controls", controls),
                            ("previous_controls", previous_controls)):
            if value.shape != (5,) or not torch.isfinite(value).all():
                raise ValueError(f"{name} must be a finite vector of length 5")
            if value.device != self.input_mean.device or value.dtype != self.input_mean.dtype:
                raise ValueError(f"{name} must match the model's device and dtype")

        fixed_disturbances = disturbances.detach()
        fixed_previous = previous_controls.detach()
        physical_inputs = torch.cat((fixed_disturbances, controls))
        scaled_inputs = (physical_inputs - self.input_mean) / self.input_scale
        prediction = self.model(scaled_inputs.unsqueeze(0))
        if prediction.shape != (1, 2) or not torch.isfinite(prediction).all():
            raise ValueError("Surrogate must return one finite, two-output prediction")
        predicted = prediction.squeeze(0) * self.output_scale + self.output_mean

        quality_term = -self.quality_weight * prediction[0, 0]
        energy_term = self.energy_weight * prediction[0, 1]
        movement_penalty = self.movement_weight * torch.sum(
            ((controls - fixed_previous) / self.input_scale[5:]) ** 2
        )
        control_violation = (
            torch.relu((self.control_lower - controls) / self.input_scale[5:]) ** 2
            + torch.relu((controls - self.control_upper) / self.input_scale[5:]) ** 2
        ).sum()
        output_violation = predicted.new_zeros(())
        if self.quality_min is not None:
            output_violation = output_violation + (
                torch.relu((self.quality_min - predicted[0]) / self.output_scale[0]) ** 2
            )
        if self.energy_max is not None:
            output_violation = output_violation + (
                torch.relu((predicted[1] - self.energy_max) / self.output_scale[1]) ** 2
            )
        constraint_penalty = self.constraint_weight * (control_violation + output_violation)
        support_penalty = self.support_weight * (
            torch.relu((self.support_lower - physical_inputs) / self.input_scale) ** 2
            + torch.relu((physical_inputs - self.support_upper) / self.input_scale) ** 2
        ).sum()
        return quality_term + energy_term + movement_penalty + constraint_penalty + support_penalty
