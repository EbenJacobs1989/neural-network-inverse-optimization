"""Temporary dynamic wrapper, validated forecasts, measured feedback and do-mpc 5.1.2."""

from dataclasses import dataclass
from importlib.metadata import version
from time import perf_counter
from typing import Any, Callable

import casadi as ca
import do_mpc
import numpy as np

from .conversion import SymbolicModel, convert_model
from .model import require_fields, vector
from .optimization import OptimizationProblem, OptimizationResult


@dataclass(frozen=True)
class ForecastProvider:
    """Return (N+1, signals), explicitly aligned to sample k through k+N."""

    values: Any
    width: int
    strategy: str = "supplied_array"
    unit: str = "engineering_units_from_contract"
    sample_interval: float = 1.0

    @classmethod
    def constant(cls, values: Any, sample_interval: float = 1.0) -> "ForecastProvider":
        array = np.asarray(values)
        if array.ndim != 1:
            raise ValueError("Constant forecast must be a vector")
        return cls(array, len(array), "constant_current_value", sample_interval=sample_interval)

    @classmethod
    def callback(cls, function: Callable[[int, int], Any], width: int, sample_interval: float = 1.0) -> "ForecastProvider":
        return cls(function, width, "callback", sample_interval=sample_interval)

    @classmethod
    def step(cls, before: Any, after: Any, at_sample: int, sample_interval: float = 1.0) -> "ForecastProvider":
        before = np.asarray(before, dtype=float)
        after = vector(after, len(before), "step target")
        return cls.callback(lambda start, horizon: np.asarray([before if index < at_sample else after
                            for index in range(start, start + horizon + 1)]), len(before), sample_interval)

    def horizon(self, sample: int, length: int) -> np.ndarray:
        if sample < 0 or length < 1 or self.width < 1 or self.sample_interval <= 0 or not np.isfinite(self.sample_interval):
            raise ValueError("Forecast sample, horizon, dimension or interval invalid")
        if callable(self.values):
            values = np.asarray(self.values(sample, length), dtype=float)
        else:
            array = np.asarray(self.values, dtype=float)
            if self.strategy == "constant_current_value":
                values = np.tile(vector(array, self.width, "constant forecast"), (length + 1, 1))
            else:
                values = array[sample:sample + length + 1]
        if values.shape != (length + 1, self.width) or not np.isfinite(values).all():
            raise ValueError(f"Forecast must supply finite shape ({length + 1}, {self.width}); no implicit padding")
        return values


def build_dynamic_model(problem: OptimizationProblem, alpha: Any, symbolic: SymbolicModel | None = None) -> do_mpc.model.Model:
    """Static NN + explicit first-order CV lag, not an identified transition model."""
    contract = problem.predictor.contract
    alpha = vector(alpha, contract.n_output, "alpha")
    if np.any(alpha <= 0) or np.any(alpha > 1):
        raise ValueError("alpha must be in (0, 1] per CV; identify plant time constants before production")
    symbolic = symbolic or convert_model(problem.predictor)
    model = do_mpc.model.Model("discrete", "MX")
    cv = model.set_variable("_x", "cv", shape=(contract.n_output, 1))
    memory = model.set_variable("_x", "mv_memory", shape=(len(contract.mv_indices), 1))
    controls = model.set_variable("_u", "mv", shape=(len(contract.mv_indices), 1))
    disturbance = model.set_variable("_tvp", "disturbance", shape=(len(contract.dv_indices), 1))
    model.set_variable("_tvp", "target", shape=(contract.n_output, 1))
    features = ca.vertcat(*[controls[contract.mv_indices.index(index)] if index in contract.mv_indices
                            else disturbance[contract.dv_indices.index(index)] for index in range(contract.n_input)])
    steady = symbolic.forward(features)
    next_cv = cv + ca.DM(alpha) * (steady - cv)
    model.set_rhs("cv", next_cv)
    model.set_rhs("mv_memory", controls)
    model.set_expression("next_cv", next_cv)
    model.set_expression("steady_output", steady)
    model.set_expression("physical_input", features)
    model.setup()
    return model


class MeasuredStateFeedback:
    """CVs are measured; MV memory records the actual approved/applied command."""

    def __init__(self, n_cv: int, n_mv: int) -> None:
        self.n_cv, self.n_mv = n_cv, n_mv

    def state(self, measured_cv: Any, applied_mv: Any) -> np.ndarray:
        return np.concatenate([vector(measured_cv, self.n_cv, "measured CVs"),
                               vector(applied_mv, self.n_mv, "actually applied MVs")]).reshape(-1, 1)


class MPCController:
    """Receding horizon with explicit memory-state move constraints and retained warm start."""

    def __init__(self, problem: OptimizationProblem, config: dict[str, Any],
                 disturbance_provider: ForecastProvider, target_provider: ForecastProvider) -> None:
        if version("do_mpc") != "5.1.2":
            raise RuntimeError("Validated controller requires do-mpc==5.1.2")
        require_fields(config, ("model_kind", "alpha", "n_horizon", "t_step", "ipopt_max_iterations", "ipopt_max_cpu_time", "ipopt_tolerance"), "MPC")
        for name in ("ipopt_max_iterations", "ipopt_max_cpu_time", "ipopt_tolerance"):
            if not isinstance(config[name], (int, float)) or not np.isfinite(config[name]) or config[name] <= 0:
                raise ValueError(f"{name} must be finite and positive")
        if config["model_kind"] != "temporary_static_lag_wrapper":
            raise ValueError("Only the explicitly labelled temporary wrapper is implemented")
        horizon = config["n_horizon"]
        if isinstance(horizon, bool) or not isinstance(horizon, int) or horizon < 1:
            raise ValueError("n_horizon must be a positive integer")
        if not np.isfinite(config["t_step"]) or config["t_step"] <= 0:
            raise ValueError("t_step must be positive")
        if any(provider.sample_interval != config["t_step"] for provider in (disturbance_provider, target_provider)):
            raise ValueError("Forecast and controller sample alignment differs")
        self.problem, self.config = problem, config
        self.disturbance_provider, self.target_provider = disturbance_provider, target_provider
        self.sample, self.initialized = 0, False
        contract = problem.predictor.contract
        if "sampling_interval" in contract.metadata and contract.metadata["sampling_interval"] != config["t_step"]:
            raise ValueError("Controller interval differs from model contract sampling interval")
        if disturbance_provider.width != len(contract.dv_indices) or target_provider.width != contract.n_output:
            raise ValueError("Forecast feature dimension differs from model contract")
        self.feedback = MeasuredStateFeedback(contract.n_output, len(contract.mv_indices))
        self.model = build_dynamic_model(problem, config["alpha"])
        self.controller = controller = do_mpc.controller.MPC(self.model)
        controller.settings.n_horizon = horizon
        controller.settings.t_step = config["t_step"]
        controller.settings.store_full_solution = True
        controller.settings.nlpsol_opts = {"ipopt.print_level": 0, "print_time": False, "ipopt.sb": "yes",
            "ipopt.max_iter": config["ipopt_max_iterations"], "ipopt.max_cpu_time": config["ipopt_max_cpu_time"],
            "ipopt.tol": config["ipopt_tolerance"], "ipopt.bound_relax_factor": 0.0}
        cv, memory, controls = self.model.x["cv"], self.model.x["mv_memory"], self.model.u["mv"]
        target = self.model.tvp["target"]
        components = problem.components(cv, controls, memory, target, backend="casadi", soft_slack=ca.MX(0))
        terminal = problem.components(cv, controls, memory, target, backend="casadi", terminal=True)["tracking"]
        controller.set_objective(mterm=terminal, lterm=sum(components.values()))
        controller.set_rterm(mv=np.zeros((len(problem.lower), 1)))
        controller.bounds["lower", "_u", "mv"] = np.maximum(problem.lower, problem.input_lower[contract.mv_indices])
        controller.bounds["upper", "_u", "mv"] = np.minimum(problem.upper, problem.input_upper[contract.mv_indices])
        controller.bounds["lower", "_x", "cv"] = problem.hard_lower
        controller.bounds["upper", "_x", "cv"] = problem.hard_upper
        controller.scaling["_x", "cv"] = contract.output_scale
        controller.scaling["_x", "mv_memory"] = contract.input_scale[contract.mv_indices]
        controller.scaling["_u", "mv"] = contract.input_scale[contract.mv_indices]
        controller.set_nl_cons("move_up", (controls - memory) / ca.DM(problem.max_move), ub=1.)
        controller.set_nl_cons("move_down", (memory - controls) / ca.DM(problem.max_move), ub=1.)
        controller.set_nl_cons("next_cv_low", (ca.DM(problem.hard_lower) - self.model.aux["next_cv"]) / ca.DM(contract.output_scale), ub=0.)
        controller.set_nl_cons("next_cv_high", (self.model.aux["next_cv"] - ca.DM(problem.hard_upper)) / ca.DM(contract.output_scale), ub=0.)
        preferred_low = (ca.DM(problem.soft_lower) - cv) / ca.DM(contract.output_scale)
        preferred_high = (cv - ca.DM(problem.soft_upper)) / ca.DM(contract.output_scale)
        controller.set_nl_cons("preferred_low", preferred_low, ub=0., soft_constraint=True, penalty_term_cons=0.)
        controller.set_nl_cons("preferred_high", preferred_high, ub=0., soft_constraint=True, penalty_term_cons=0.)
        self._template = controller.get_tvp_template()

        def tvp_function(time: Any):
            disturbances, targets = self._forecasts()
            for index in range(horizon + 1):
                self._template["_tvp", index, "disturbance"] = disturbances[index]
                self._template["_tvp", index, "target"] = targets[index]
            return self._template

        controller.set_tvp_fun(tvp_function)
        controller.prepare_nlp()
        for stage in range(horizon):
            slack_low = controller.opt_x["_eps", stage, 0, "preferred_low"]
            slack_high = controller.opt_x["_eps", stage, 0, "preferred_high"]
            controller.nlp_obj += ca.sum1(ca.DM(problem.soft_weights) * (slack_low**2 + slack_high**2))
        controller.create_nlp()
        self._objective_function = ca.Function("horizon_objective", [controller.opt_x, controller.opt_p], [controller.nlp_obj])

    def _forecasts(self) -> tuple[np.ndarray, np.ndarray]:
        horizon = self.config["n_horizon"]
        disturbances = self.disturbance_provider.horizon(self.sample, horizon)
        targets = self.target_provider.horizon(self.sample, horizon)
        contract = self.problem.predictor.contract
        for disturbance in disturbances:
            if np.any(disturbance < self.problem.input_lower[contract.dv_indices]) or np.any(disturbance > self.problem.input_upper[contract.dv_indices]):
                raise ValueError("Disturbance forecast outside model envelope")
        return disturbances, targets

    def step(self, measured_cv: Any, applied_mv: Any) -> OptimizationResult:
        started = perf_counter()
        state = self.feedback.state(measured_cv, applied_mv)
        disturbance, targets = self._forecasts()
        previous = np.asarray(applied_mv)
        self.controller.x0 = state
        if not self.initialized:
            self.controller.u0 = previous.reshape(-1, 1)
            self.controller.set_initial_guess()
            self.initialized = True
        controls = np.asarray(self.controller.make_step(state)).ravel()
        stats = self.controller.solver_stats
        steady = self.problem.predictor.predict(self.problem.predictor.contract.assemble(controls, disturbance[0]), allow_outside=True)
        outputs = np.asarray(measured_cv) + np.asarray(self.config["alpha"]) * (steady - np.asarray(measured_cv))
        components = {key: float(value) for key, value in self.problem.components(outputs, controls, previous, targets[1]).items()}
        _, _, violations = self.problem.evaluate(controls, disturbance[0], previous)
        violations["hard_cv"] = float(np.max(np.maximum.reduce([self.problem.hard_lower - outputs, outputs - self.problem.hard_upper, np.zeros_like(outputs)])))
        contract = self.problem.predictor.contract
        horizon_cv = np.asarray(self.controller.data.prediction(("_x", "cv")))[:, :, 0].T
        horizon_mv = np.asarray(self.controller.data.prediction(("_u", "mv")))[:, :, 0].T
        horizon_rate = np.diff(np.vstack([previous, horizon_mv]), axis=0)
        violations["horizon_hard_cv"] = float(np.maximum(np.maximum(self.problem.hard_lower - horizon_cv, horizon_cv - self.problem.hard_upper), 0).max())
        violations["horizon_rate"] = float(np.maximum(np.abs(horizon_rate) - self.problem.max_move, 0).max())
        violations["horizon_mv"] = float(np.maximum(np.maximum(self.problem.lower - horizon_mv, horizon_mv - self.problem.upper), 0).max())
        for stage_controls, stage_disturbance in zip(horizon_mv, disturbance):
            features = contract.assemble(stage_controls, stage_disturbance)
            violation = np.maximum(np.maximum(self.problem.input_lower - features, features - self.problem.input_upper), 0) / contract.input_scale
            violations["envelope"] = max(violations["envelope"], float(violation.max()))
        success = bool(stats.get("success", False))
        result = OptimizationResult(controls, outputs, sum(components.values()), components, int(stats.get("iter_count", 0)),
            str(stats.get("return_status", "Unknown")), success, success, perf_counter() - started,
            max(violations.values()) <= self.problem.tolerance, violations, previous.tolist(), None,
            violations["envelope"] <= self.problem.tolerance,
            {"prediction_horizon": self.config["n_horizon"], "temporary_dynamic_wrapper": True,
             "horizon_objective": float(self._objective_function(self.controller.opt_x_num, self.controller.opt_p_num)),
             "horizon_cv": horizon_cv.tolist(), "horizon_mv": horizon_mv.tolist(),
             "soft_slacks": np.asarray(self.controller.opt_x_num["_eps"]).ravel().tolist()})
        self.sample += 1
        return result