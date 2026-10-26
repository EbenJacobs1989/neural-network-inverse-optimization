"""Independent synthetic-plant replay with governed first moves and deliberate mismatch."""

import ast
from collections import deque
import json
import logging
from pathlib import Path
from time import perf_counter
from typing import Any, Callable

import numpy as np
import pandas as pd
from scipy.special import expit

from .benchmarking import run_method
from .governance import ValidationGate
from .model import vector
from .mpc import ForecastProvider, MPCController
from .optimization import OptimizationProblem

LOGGER = logging.getLogger(__name__)


def load_original_plant(root: Path) -> Callable:
    """Reuse exactly one trusted notebook function, without running notebook cells."""
    notebook = root / "notebooks/01_generate_data.ipynb"
    document = json.loads(notebook.read_text(encoding="utf-8"))
    functions = [node for cell in document["cells"] if cell["cell_type"] == "code"
                 for node in ast.parse("".join(cell["source"])).body
                 if isinstance(node, ast.FunctionDef) and node.name == "steady_state_response"]
    if len(functions) != 1:
        raise ValueError("Original plant notebook must define exactly one steady_state_response")
    scope: dict[str, Any] = {"np": np, "expit": expit}
    exec(compile(ast.Module(body=functions, type_ignores=[]), str(notebook), "exec"), scope)
    return scope["steady_state_response"]


def simulate(problem: OptimizationProblem, config: dict[str, Any], inverse_config: dict[str, Any],
             disturbance: np.ndarray, previous: np.ndarray, initial_cv: np.ndarray,
             plant: Callable, steps: int, *, policy: str = "NMPC",
             disturbance_provider: ForecastProvider | None = None,
             target_provider: ForecastProvider | None = None) -> pd.DataFrame:
    """Synthetic-only mismatch experiment. This function never writes to a plant."""
    if steps < 1:
        raise ValueError("Simulation steps must be positive")
    contract = problem.predictor.contract
    size = contract.n_output
    plant_alpha = vector(config["plant_alpha"], size, "plant_alpha")
    if np.any(plant_alpha <= 0) or np.any(plant_alpha > 1):
        raise ValueError("Plant alpha must be in (0,1]")
    gain, bias = vector(config["plant_gain"], size, "plant gain"), vector(config["plant_bias"], size, "plant bias")
    noise = vector(config["measurement_noise"], size, "measurement noise")
    unmeasured = vector(config["unmeasured_disturbance"], size, "unmeasured disturbance")
    if np.any(noise < 0):
        raise ValueError("Noise standard deviations cannot be negative")
    delay = config["plant_delay_steps"]
    if isinstance(delay, bool) or not isinstance(delay, int) or delay < 0:
        raise ValueError("Plant delay must be a nonnegative integer")
    interval = config["t_step"]
    if disturbance_provider is None and config.get("forecast_strategy", "constant_current_value") != "constant_current_value":
        raise ValueError("Non-persistence forecast strategy requires an explicit ForecastProvider")
    disturbances = disturbance_provider or ForecastProvider.constant(disturbance, interval)
    target_after = problem.target + contract.output_scale * np.asarray(config.get("target_step_standard_deviations", np.zeros(size)))
    targets = target_provider or ForecastProvider.step(problem.target, target_after, config.get("target_step_sample", steps // 2), interval)
    controller = MPCController(problem, config, disturbances, targets) if policy == "NMPC" else None
    gate = ValidationGate(problem, config)
    gate.seed_approved(previous, 0.)
    state = vector(initial_cv, size, "initial plant CVs")
    measured = state.copy()
    rng = np.random.default_rng(config["seed"])
    initial_target = np.asarray(plant(np.asarray(disturbance)[None, :], np.asarray(previous)[None, :]))[0]
    queue = deque([initial_target.copy() for _ in range(delay)])
    rows = []
    original_target = problem.target.copy()
    try:
        for step in range(steps):
            now = step * interval
            disturbance = disturbances.horizon(step, config["n_horizon"])[0]
            reference = targets.horizon(step, config["n_horizon"])[0]
            problem.target = reference
            try:
                candidate = controller.step(measured, previous) if controller is not None else run_method(policy, problem, disturbance, previous, inverse_config)
            except RuntimeError as error:
                LOGGER.exception("Numerical solver failed at sample %s", step)
                candidate = problem.result(previous, disturbance, previous, iterations=0, status="Solver_Failure", success=False,
                    converged=False, started=perf_counter(), initialization=previous, diagnostics={"error": str(error)})
            decision = gate.approve(candidate, previous, disturbance, measurement_time=now, result_time=now, now=now)
            controls = previous.copy() if decision.final_command is None else decision.final_command.copy()
            static_target = np.asarray(plant(np.asarray(disturbance)[None, :], controls[None, :]))[0]
            queue.append(static_target)
            delayed_target = queue.popleft()
            independent_target = contract.output_mean + gain * (delayed_target - contract.output_mean) + bias + unmeasured
            state += plant_alpha * (independent_target - state)
            measured = state + rng.normal(0., noise)
            hard_violation = float(np.maximum(np.maximum(problem.hard_lower - measured, measured - problem.hard_upper), 0).max())
            rows.append({"policy": policy, "sample": step, "time": now, "controls": controls.tolist(),
                "measured_cv": measured.tolist(), "target": reference.tolist(), "disturbance": disturbance.tolist(),
                "predicted_cv": candidate.outputs.tolist(), "mv_rate": ((controls - previous) / interval).tolist(),
                "objective": candidate.objective, **{f"cost_{key}": value for key, value in candidate.components.items()},
                "runtime": candidate.runtime, "iterations": candidate.iterations, "solver_status": candidate.status,
                "feasible": candidate.feasible, "accepted": decision.accepted, "fallback_tier": decision.tier,
                "rejection_reason": decision.rejection_reason, "envelope_ok": candidate.envelope_ok,
                "slack_max": float(np.max(candidate.diagnostics.get("soft_slacks", [0.]))),
                "hard_cv_violation": hard_violation, "mv_travel": float(np.abs(controls - previous).sum()),
                "absolute_error": float(np.abs(measured - reference).sum())})
            previous = controls
    finally:
        problem.target = original_target
    return pd.DataFrame(rows)