"""Authoritative feature/scaler contract and trusted local checkpoint loading."""

from dataclasses import dataclass
from pathlib import Path
from typing import Any
import pickle

import numpy as np
import pandas as pd
import torch
from torch import nn
import yaml


def load_config(path: Path) -> dict[str, Any]:
    """Load a YAML mapping without executing constructors."""
    with Path(path).open(encoding="utf-8") as stream:
        value = yaml.safe_load(stream)
    if not isinstance(value, dict):
        raise ValueError(f"Configuration must be a mapping: {path}")
    return value


def require_fields(config: dict[str, Any], names: tuple[str, ...], context: str) -> None:
    missing = [name for name in names if name not in config]
    if missing:
        raise ValueError(f"{context} configuration is missing: {', '.join(missing)}")


def vector(value: Any, size: int, name: str) -> np.ndarray:
    """Return a finite engineering-unit vector with an exact shape."""
    array = np.asarray(value, dtype=np.float64)
    if array.shape != (size,) or not np.isfinite(array).all():
        raise ValueError(f"{name} must be a finite vector of shape ({size},)")
    return array.copy()


@dataclass(frozen=True)
class ModelContract:
    """Immutable metadata; input and output arrays follow the named feature order."""

    metadata: dict[str, Any]
    input_mean: np.ndarray
    input_scale: np.ndarray
    output_mean: np.ndarray
    output_scale: np.ndarray
    train_min: np.ndarray
    train_max: np.ndarray
    validation_min: np.ndarray
    validation_max: np.ndarray
    output_min: np.ndarray
    output_max: np.ndarray

    def __post_init__(self) -> None:
        for key, roles in (("inputs", {"MV", "DV"}), ("outputs", {"CV", "KPI"})):
            signals = self.metadata.get(key, [])
            names = [signal.get("name") for signal in signals]
            if not names or any(not name for name in names) or len(set(names)) != len(names):
                raise ValueError(f"{key} must have unique nonempty feature names")
            if any(signal.get("role") not in roles or not signal.get("unit") for signal in signals):
                raise ValueError(f"{key} require supported roles and explicit units")
        if not self.mv_indices or not self.dv_indices:
            raise ValueError("Static contract requires MV and DV signals")
        for name in ("input_mean", "input_scale", "train_min", "train_max", "validation_min", "validation_max"):
            array = vector(getattr(self, name), self.n_input, name)
            array.setflags(write=False)
            object.__setattr__(self, name, array)
        for name in ("output_mean", "output_scale", "output_min", "output_max"):
            array = vector(getattr(self, name), self.n_output, name)
            array.setflags(write=False)
            object.__setattr__(self, name, array)
        if np.any(self.input_scale <= 0) or np.any(self.output_scale <= 0):
            raise ValueError("Scaler standard deviations must be positive")
        for lower, upper in ((self.train_min, self.train_max), (self.validation_min, self.validation_max),
                             (self.output_min, self.output_max)):
            if np.any(lower >= upper):
                raise ValueError("Envelope bounds must be strictly ordered")
        if self.metadata.get("dtype") not in {"float32", "float64"}:
            raise ValueError("dtype must be float32 or float64")
        if self.metadata.get("framework") != "pytorch" or self.metadata.get("model_type") != "static_current_output":
            raise ValueError("Loader supports the verified static PyTorch contract only")

    @property
    def input_names(self) -> list[str]:
        return [signal["name"] for signal in self.metadata["inputs"]]

    @property
    def output_names(self) -> list[str]:
        return [signal["name"] for signal in self.metadata["outputs"]]

    @property
    def mv_indices(self) -> list[int]:
        return [index for index, signal in enumerate(self.metadata["inputs"]) if signal["role"] == "MV"]

    @property
    def dv_indices(self) -> list[int]:
        return [index for index, signal in enumerate(self.metadata["inputs"]) if signal["role"] == "DV"]

    @property
    def n_input(self) -> int:
        return len(self.input_names)

    @property
    def n_output(self) -> int:
        return len(self.output_names)

    def assemble(self, controls: Any, disturbances: Any) -> np.ndarray:
        features = np.empty(self.n_input)
        features[self.mv_indices] = vector(controls, len(self.mv_indices), "MVs")
        features[self.dv_indices] = vector(disturbances, len(self.dv_indices), "DVs")
        return features

    def validate_input(self, features: Any, *, allow_outside: bool = False) -> np.ndarray:
        values = vector(features, self.n_input, "model input")
        if not allow_outside and (np.any(values < self.train_min) or np.any(values > self.train_max)):
            raise ValueError("Model input is outside the training envelope; explicit advisory override required")
        return values


class ForwardModel:
    """Engineering-unit predictor; the frozen network itself consumes standardised data."""

    def __init__(self, network: nn.Module, contract: ModelContract) -> None:
        self.contract = contract
        self.network = network.eval().requires_grad_(False)
        self.dtype = getattr(torch, contract.metadata["dtype"])
        self.device = torch.device(contract.metadata.get("device", "cpu"))
        self.network.to(device=self.device, dtype=self.dtype)
        with torch.no_grad():
            prediction = self.network(torch.zeros(contract.n_input, dtype=self.dtype, device=self.device))
        if prediction.shape != (contract.n_output,):
            raise ValueError("Configured dimensions do not match neural network")

    def tensor(self, value: Any) -> torch.Tensor:
        return torch.as_tensor(np.array(value, copy=True), dtype=self.dtype, device=self.device)

    def predict_tensor(self, features: torch.Tensor) -> torch.Tensor:
        if features.shape != (self.contract.n_input,) or not torch.isfinite(features).all():
            raise ValueError("Invalid model input shape or non-finite values")
        if features.dtype != self.dtype or features.device != self.device:
            raise ValueError("Input must match contract dtype/device")
        contract = self.contract
        result = self.network((features - self.tensor(contract.input_mean)) / self.tensor(contract.input_scale))
        result = result * self.tensor(contract.output_scale) + self.tensor(contract.output_mean)
        if result.shape != (contract.n_output,) or not torch.isfinite(result).all():
            raise ValueError("Network returned invalid outputs")
        return result

    def predict(self, features: Any, *, allow_outside: bool = False) -> np.ndarray:
        values = self.contract.validate_input(features, allow_outside=allow_outside)
        with torch.no_grad():
            return self.predict_tensor(self.tensor(values)).cpu().numpy()


def load_model(root: Path, config_path: Path | None = None) -> tuple[ForwardModel, pd.DataFrame, np.ndarray, np.ndarray]:
    """Load trusted local artifacts and derive envelopes using the chronological split only."""
    root = Path(root).resolve()
    config = load_config(config_path or root / "configs/model.yaml")
    require_fields(config, ("checkpoint", "input_scaler", "output_scaler", "inputs", "outputs", "hidden_sizes",
                           "activation", "checkpoint_dtype", "data_path", "train_fraction", "validation_fraction", "support_quantiles"), "Model")
    checkpoint_path = root / config["checkpoint"]
    try:
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    except (OSError, RuntimeError, ValueError) as error:
        raise ValueError(f"Checkpoint could not be loaded: {checkpoint_path}") from error
    input_names = [signal["name"] for signal in config["inputs"]]
    output_names = [signal["name"] for signal in config["outputs"]]
    if checkpoint["input_columns"] != input_names or checkpoint["output_columns"] != output_names:
        raise ValueError("Checkpoint and configured feature order differ")
    if list(checkpoint["hidden_sizes"]) != config["hidden_sizes"] or config["activation"] != "SiLU":
        raise ValueError("Unsupported or mismatched checkpoint architecture")
    scalers = []
    for key, names in (("input_scaler", input_names), ("output_scaler", output_names)):
        with (root / config[key]).open("rb") as stream:
            scaler = pickle.load(stream)
        if list(scaler.feature_names_in_) != names:
            raise ValueError(f"{key} feature order differs from checkpoint")
        scalers.append(scaler)
    input_scaler, output_scaler = scalers
    data = pd.read_csv(root / config["data_path"])
    if not np.isfinite(data[input_names + output_names].to_numpy(dtype=float)).all():
        raise ValueError("Data contains missing or non-finite model signals")
    split = int(len(data) * config["train_fraction"])
    validation_end = split + int(len(data) * config["validation_fraction"])
    if not 0 < split < validation_end < len(data):
        raise ValueError("Invalid chronological split configuration")
    train, validation = data.iloc[:split], data.iloc[split:validation_end]
    for scaler, names in ((input_scaler, input_names), (output_scaler, output_names)):
        if (int(scaler.n_samples_seen_) != split or not np.allclose(scaler.mean_, train[names].mean(), rtol=1e-5)
            or not np.allclose(scaler.scale_, train[names].std(ddof=0), rtol=1e-5)):
            raise ValueError("Scalers are not fitted to the configured chronological training split")
    contract = ModelContract(config, input_scaler.mean_, input_scaler.scale_, output_scaler.mean_, output_scaler.scale_,
                             train[input_names].min().to_numpy(), train[input_names].max().to_numpy(),
                             validation[input_names].min().to_numpy(), validation[input_names].max().to_numpy(),
                             train[output_names].min().to_numpy(), train[output_names].max().to_numpy())
    sizes = [len(input_names), *config["hidden_sizes"], len(output_names)]
    layers: list[nn.Module] = []
    for index in range(len(sizes) - 1):
        layers.append(nn.Linear(sizes[index], sizes[index + 1]))
        if index < len(sizes) - 2:
            layers.append(nn.SiLU())
    network = nn.Sequential(*layers)
    state = checkpoint["model_state_dict"]
    if any(str(value.dtype).split(".")[-1] != config["checkpoint_dtype"] for value in state.values()):
        raise ValueError("Checkpoint dtype differs from contract")
    network.load_state_dict({key.removeprefix("network."): value for key, value in state.items()}, strict=True)
    quantiles = config["support_quantiles"]
    if len(quantiles) != 2 or not 0 <= quantiles[0] < quantiles[1] <= 1:
        raise ValueError("support_quantiles must be ordered inside [0,1]")
    support = train[input_names].quantile(quantiles).to_numpy()
    return ForwardModel(network, contract), data, support[0], support[1]