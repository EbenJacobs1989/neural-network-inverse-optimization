"""Inspectable native CasADi reconstruction and prediction/Jacobian acceptance."""

from dataclasses import dataclass
from typing import Any

import casadi as ca
import numpy as np
import torch
from torch import nn

from .model import ForwardModel


@dataclass(frozen=True)
class SymbolicModel:
    forward: ca.Function
    jacobian: ca.Function


def convert_model(predictor: ForwardModel) -> SymbolicModel:
    """Reconstruct supported Sequential operations; Linear uses W @ x + b."""
    contract = predictor.contract
    features = ca.MX.sym("engineering_inputs", contract.n_input)
    values = (features - ca.DM(contract.input_mean)) / ca.DM(contract.input_scale)
    network = predictor.network
    if not isinstance(network, nn.Sequential):
        raise ValueError("Conversion requires an inspectable nn.Sequential, not an opaque forward callback")

    def apply_layer(layer: nn.Module, expression: Any, path: str) -> Any:
        if isinstance(layer, nn.Sequential):
            for name, child in layer.named_children():
                expression = apply_layer(child, expression, f"{path}.{name}")
            return expression
        if isinstance(layer, nn.Linear):
            result = ca.mtimes(ca.DM(layer.weight.detach().cpu().numpy().astype(float)), expression)
            return result if layer.bias is None else result + ca.DM(layer.bias.detach().cpu().numpy().astype(float))
        if isinstance(layer, nn.SiLU):
            return expression * (0.5 * (1 + ca.tanh(expression / 2)))
        if isinstance(layer, nn.Tanh):
            return ca.tanh(expression)
        if isinstance(layer, nn.Sigmoid):
            return 0.5 * (1 + ca.tanh(expression / 2))
        if isinstance(layer, nn.Softplus):
            scaled = layer.beta * expression
            stable = ca.fmax(scaled, 0) + ca.log(1 + ca.exp(-ca.fabs(scaled)))
            return ca.if_else(scaled > layer.threshold, expression, stable / layer.beta)
        if isinstance(layer, nn.ReLU):
            return ca.fmax(expression, 0)
        if isinstance(layer, nn.Identity):
            return expression
        raise ValueError(f"Unsupported layer {path}: {type(layer).__name__}; export explicit operations or identify a supported model")

    values = apply_layer(network, values, "network")
    if values.shape != (contract.n_output, 1):
        raise ValueError("Converted output dimensions differ from contract")
    outputs = values * ca.DM(contract.output_scale) + ca.DM(contract.output_mean)
    return SymbolicModel(ca.Function("forward_engineering", [features], [outputs]),
                         ca.Function("jacobian_engineering", [features], [ca.jacobian(outputs, features)]))


def equivalence_report(predictor: ForwardModel, symbolic: SymbolicModel, samples: np.ndarray,
                       *, prediction_atol: float = 1e-9, jacobian_atol: float = 1e-8) -> dict[str, Any]:
    """Compare engineering outputs and full dy/dx, including scaler derivatives."""
    samples = np.asarray(samples, dtype=float)
    if samples.ndim != 2 or samples.shape[1] != predictor.contract.n_input or not np.isfinite(samples).all():
        raise ValueError("Equivalence samples must have shape (samples, inputs) and be finite")
    if len(samples) == 0 or min(prediction_atol, jacobian_atol) <= 0:
        raise ValueError("Equivalence requires samples and positive tolerances")
    prediction_errors, jacobian_errors, relative_errors, failures = [], [], [], []
    for index, sample in enumerate(samples):
        features = predictor.tensor(sample).requires_grad_(True)
        expected = predictor.predict_tensor(features).detach().cpu().numpy()
        derivative = torch.autograd.functional.jacobian(predictor.predict_tensor, features).detach().cpu().numpy()
        actual = np.asarray(symbolic.forward(sample)).ravel()
        actual_derivative = np.asarray(symbolic.jacobian(sample))
        prediction_error = np.abs(actual - expected)
        jacobian_error = np.abs(actual_derivative - derivative)
        prediction_errors.extend(prediction_error.tolist())
        jacobian_errors.extend(jacobian_error.ravel().tolist())
        relative_errors.extend((prediction_error / np.maximum(np.abs(expected), 1e-12)).tolist())
        if (not np.isfinite(actual).all() or not np.isfinite(actual_derivative).all()
                or np.max(prediction_error) > prediction_atol or np.max(jacobian_error) > jacobian_atol):
            failures.append({"index": index, "input": sample.tolist(), "torch": expected.tolist(),
                             "casadi": actual.tolist(), "prediction_error": prediction_error.tolist(),
                             "jacobian_max_error": float(np.max(jacobian_error))})
    return {"accepted": not failures, "samples": len(samples), "dtype": str(predictor.dtype),
            "prediction_atol": prediction_atol, "jacobian_atol": jacobian_atol,
            "max_absolute_prediction_error": float(np.max(prediction_errors)),
            "mean_absolute_prediction_error": float(np.mean(prediction_errors)),
            "max_relative_prediction_error": float(np.max(relative_errors)),
            "max_absolute_jacobian_error": float(np.max(jacobian_errors)),
            "mean_absolute_jacobian_error": float(np.mean(jacobian_errors)), "failures": failures}