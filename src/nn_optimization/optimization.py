"""One engineering-unit objective, constraints and result contract for all solvers."""

from dataclasses import asdict, dataclass, field
from time import perf_counter
from typing import Any

import casadi as ca
import numpy as np
import torch

from .conversion import SymbolicModel, convert_model
from .model import ForwardModel, require_fields, vector


def validate_solver_settings(config: dict[str, Any]) -> None:
    require_fields(config, ("multistart", "seed", "torch_method", "torch_iterations", "torch_learning_rate",
                           "gradient_tolerance", "ipopt_max_iterations", "ipopt_max_cpu_time", "ipopt_tolerance"), "Inverse solver")
    for name in ("multistart", "torch_iterations", "ipopt_max_iterations"):
        if isinstance(config[name], bool) or not isinstance(config[name], int) or config[name] < 1:
            raise ValueError(f"{name} must be a positive integer")
    if isinstance(config["seed"], bool) or not isinstance(config["seed"], int) or config["seed"] < 0:
        raise ValueError("seed must be a nonnegative integer")
    for name in ("torch_learning_rate", "gradient_tolerance", "ipopt_max_cpu_time", "ipopt_tolerance"):
        if not isinstance(config[name], (int, float)) or not np.isfinite(config[name]) or config[name] <= 0:
            raise ValueError(f"{name} must be finite and positive")


@dataclass
class OptimizationResult:
    controls: np.ndarray
    outputs: np.ndarray
    objective: float
    components: dict[str, float]
    iterations: int
    status: str
    success: bool
    converged: bool
    runtime: float
    feasible: bool
    violations: dict[str, float]
    initialization: list[float]
    gradient_norm: float | None
    envelope_ok: bool
    diagnostics: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        values = asdict(self)
        values["controls"] = self.controls.tolist()
        values["outputs"] = self.outputs.tolist()
        return values


@dataclass
class OptimizationProblem:
    predictor: ForwardModel
    lower: np.ndarray
    upper: np.ndarray
    max_move: np.ndarray
    hard_lower: np.ndarray
    hard_upper: np.ndarray
    soft_lower: np.ndarray
    soft_upper: np.ndarray
    target: np.ndarray
    tracking: np.ndarray
    terminal: np.ndarray
    movement: np.ndarray
    economic_output: np.ndarray
    economic_mv: np.ndarray
    soft_weights: np.ndarray
    regularisation: float = 0.0
    tolerance: float = 1e-6
    envelope_margin: float = 0.0

    def __post_init__(self) -> None:
        contract = self.predictor.contract
        for name in ("lower", "upper", "max_move", "movement", "economic_mv"):
            setattr(self, name, vector(getattr(self, name), len(contract.mv_indices), name))
        for name in ("hard_lower", "hard_upper", "soft_lower", "soft_upper", "target", "tracking", "terminal", "economic_output", "soft_weights"):
            setattr(self, name, vector(getattr(self, name), contract.n_output, name))
        if np.any(self.lower >= self.upper) or np.any(self.max_move <= 0):
            raise ValueError("MV bounds must be ordered and rate limits positive")
        if np.any(self.hard_lower >= self.hard_upper) or np.any(self.soft_lower > self.soft_upper):
            raise ValueError("Output limits must be ordered")
        for name in ("tracking", "terminal", "movement", "soft_weights"):
            if np.any(getattr(self, name) < 0):
                raise ValueError(f"{name} weights cannot be negative")
        if not np.isfinite([self.regularisation, self.tolerance, self.envelope_margin]).all():
            raise ValueError("Problem scalars must be finite")
        if self.regularisation < 0 or self.tolerance <= 0 or not 0 <= self.envelope_margin < .5:
            raise ValueError("Invalid regularisation, tolerance or envelope margin")

    @property
    def input_lower(self) -> np.ndarray:
        contract = self.predictor.contract
        return contract.train_min + self.envelope_margin * (contract.train_max - contract.train_min)

    @property
    def input_upper(self) -> np.ndarray:
        contract = self.predictor.contract
        return contract.train_max - self.envelope_margin * (contract.train_max - contract.train_min)

    def effective_bounds(self, previous: Any, disturbances: Any) -> tuple[np.ndarray, np.ndarray]:
        contract = self.predictor.contract
        previous = vector(previous, len(contract.mv_indices), "previous MVs")
        disturbances = vector(disturbances, len(contract.dv_indices), "DVs")
        if np.any(disturbances < self.input_lower[contract.dv_indices]) or np.any(disturbances > self.input_upper[contract.dv_indices]):
            raise ValueError("Disturbance is outside the approved model envelope")
        lower = np.maximum.reduce([self.lower, previous - self.max_move, self.input_lower[contract.mv_indices]])
        upper = np.minimum.reduce([self.upper, previous + self.max_move, self.input_upper[contract.mv_indices]])
        if np.any(lower > upper):
            raise ValueError("Infeasible intersection of physical, rate and envelope bounds")
        return lower, upper

    def components(self, outputs: Any, controls: Any, previous: Any, target: Any = None,
                   *, backend: str = "numpy", soft_slack: Any = None, terminal: bool = False) -> dict[str, Any]:
        """Same scaled objective expression for NumPy, Torch and CasADi."""
        contract = self.predictor.contract
        if backend == "torch":
            constant, sum_values, positive = self.predictor.tensor, torch.sum, torch.relu
        elif backend == "casadi":
            constant, sum_values = ca.DM, ca.sum1
            positive = lambda value: ca.fmax(value, 0)
        elif backend == "numpy":
            constant, sum_values = np.asarray, np.sum
            positive = lambda value: np.maximum(value, 0)
        else:
            raise ValueError(f"Unknown objective backend {backend}")
        reference = constant(self.target) if target is None else target
        error = (outputs - reference) / constant(contract.output_scale)
        scaled_output = (outputs - constant(contract.output_mean)) / constant(contract.output_scale)
        control_scale = constant(contract.input_scale[contract.mv_indices])
        scaled_control = (controls - constant(contract.input_mean[contract.mv_indices])) / control_scale
        change = (controls - previous) / control_scale
        tracking = sum_values(constant(self.terminal if terminal else self.tracking) * error**2)
        if terminal:
            return {"tracking": tracking}
        low = positive((constant(self.soft_lower) - outputs) / constant(contract.output_scale))
        high = positive((outputs - constant(self.soft_upper)) / constant(contract.output_scale))
        soft = sum_values(constant(self.soft_weights) * (low**2 + high**2)) if soft_slack is None else soft_slack
        return {"tracking": tracking,
                "economic": sum_values(constant(self.economic_output) * scaled_output) + sum_values(constant(self.economic_mv) * scaled_control),
                "movement": sum_values(constant(self.movement) * change**2), "soft_constraint": soft,
                "regularisation": self.regularisation * sum_values(scaled_control**2)}

    def evaluate(self, controls: Any, disturbances: Any, previous: Any) -> tuple[np.ndarray, dict[str, float], dict[str, float]]:
        contract = self.predictor.contract
        controls = vector(controls, len(contract.mv_indices), "MVs")
        previous = vector(previous, len(contract.mv_indices), "previous MVs")
        features = contract.assemble(controls, disturbances)
        outputs = self.predictor.predict(features, allow_outside=True)
        components = {key: float(value) for key, value in self.components(outputs, controls, previous).items()}
        violations = {
            "mv": float(np.max(np.maximum.reduce([self.lower - controls, controls - self.upper, np.zeros_like(controls)]))),
            "rate": float(np.max(np.maximum(np.abs(controls - previous) - self.max_move, 0))),
            "hard_cv": float(np.max(np.maximum.reduce([self.hard_lower - outputs, outputs - self.hard_upper, np.zeros_like(outputs)]))),
            "envelope": float(np.max(np.maximum.reduce([(self.input_lower - features) / contract.input_scale,
                                                         (features - self.input_upper) / contract.input_scale, np.zeros_like(features)]))),
        }
        return outputs, components, violations

    def result(self, controls: Any, disturbances: Any, previous: Any, *, iterations: int, status: str,
               success: bool, converged: bool, started: float, initialization: Any,
               gradient_norm: float | None = None, diagnostics: dict[str, Any] | None = None) -> OptimizationResult:
        outputs, components, violations = self.evaluate(controls, disturbances, previous)
        return OptimizationResult(np.asarray(controls).copy(), outputs, sum(components.values()), components,
                                  iterations, status, success, converged, perf_counter() - started,
                                  max(violations.values()) <= self.tolerance, violations,
                                  np.asarray(initialization).tolist(), gradient_norm,
                                  violations["envelope"] <= self.tolerance, diagnostics or {})


def problem_from_config(predictor: ForwardModel, config: dict[str, Any], support_lower: np.ndarray,
                        support_upper: np.ndarray) -> OptimizationProblem:
    """Map legacy normalised commands to physical MVs exactly once."""
    contract = predictor.contract
    require_fields(config, ("command_lower", "command_upper", "command_max_move", "tracking_weights", "terminal_weights",
        "movement_weights", "economic_output_coefficients", "economic_mv_coefficients", "soft_weights", "regularisation_weight",
        "feasibility_tolerance", "envelope_margin", "hard_lower_standard_deviations", "hard_upper_standard_deviations",
        "soft_lower_standard_deviations", "soft_upper_standard_deviations", "target_standard_deviations"), "Inverse objective")
    validate_solver_settings(config)
    origin = support_lower[contract.mv_indices]
    span = (support_upper - support_lower)[contract.mv_indices]
    mv_size = len(contract.mv_indices)
    def output_limit(key: str) -> np.ndarray:
        return contract.output_mean + contract.output_scale * vector(config[key], contract.n_output, key)
    return OptimizationProblem(predictor,
        origin + span * vector(config["command_lower"], mv_size, "command_lower"),
        origin + span * vector(config["command_upper"], mv_size, "command_upper"),
        span * vector(config["command_max_move"], mv_size, "command_max_move"),
        output_limit("hard_lower_standard_deviations"), output_limit("hard_upper_standard_deviations"),
        output_limit("soft_lower_standard_deviations"), output_limit("soft_upper_standard_deviations"),
        output_limit("target_standard_deviations"), config["tracking_weights"], config["terminal_weights"],
        config["movement_weights"], config["economic_output_coefficients"], config["economic_mv_coefficients"],
        config["soft_weights"], config["regularisation_weight"], config["feasibility_tolerance"], config["envelope_margin"])


def starting_points(lower: np.ndarray, upper: np.ndarray, previous: np.ndarray, count: int, seed: int,
                    warm_start: np.ndarray | None = None) -> list[np.ndarray]:
    if isinstance(count, bool) or not isinstance(count, int) or count < 1:
        raise ValueError("multistart must be a positive integer")
    rng = np.random.default_rng(seed)
    starts = [np.clip(previous, lower, upper)]
    if warm_start is not None and count > 1:
        starts.append(np.clip(vector(warm_start, len(lower), "warm start"), lower, upper))
    starts.extend(rng.uniform(lower, upper) for _ in range(count - len(starts)))
    return starts


def select_result(results: list[OptimizationResult]) -> OptimizationResult:
    feasible = [result for result in results if result.feasible and np.isfinite(result.objective)]
    successful = [result for result in feasible if result.success]
    best = min(successful or feasible or results, key=lambda result: result.objective)
    best.diagnostics["starts"] = [{"status": result.status, "objective": result.objective, "feasible": result.feasible} for result in results]
    best.diagnostics["no_feasible_solution"] = not bool(feasible)
    best.runtime = sum(result.runtime for result in results)
    return best


def solve_torch(problem: OptimizationProblem, disturbances: np.ndarray, previous: np.ndarray,
                config: dict[str, Any], *, warm_start: np.ndarray | None = None) -> OptimizationResult:
    """Optimise bounded input logits. Hard CV feasibility is checked, not promised by penalties."""
    validate_solver_settings(config)
    predictor, contract = problem.predictor, problem.predictor.contract
    lower, upper = problem.effective_bounds(previous, disturbances)
    previous_tensor = predictor.tensor(previous)
    fixed = predictor.tensor(disturbances)
    low, width = predictor.tensor(lower), predictor.tensor(upper - lower)
    results = []
    for start in starting_points(lower, upper, previous, config["multistart"], config["seed"], warm_start):
        started = perf_counter()
        ratio = np.clip(np.divide(start - lower, upper - lower, out=np.full_like(start, .5), where=upper > lower), 1e-5, 1 - 1e-5)
        logits = torch.nn.Parameter(predictor.tensor(np.log(ratio / (1 - ratio))))
        method = config["torch_method"]
        if method == "Adam":
            optimizer = torch.optim.Adam([logits], lr=config["torch_learning_rate"])
        elif method == "LBFGS":
            optimizer = torch.optim.LBFGS([logits], lr=config["torch_learning_rate"], max_iter=1, line_search_fn="strong_wolfe")
        else:
            raise ValueError("torch_method must be Adam or LBFGS")
        best = problem.result(start, disturbances, previous, iterations=0, status="Iteration_Limit", success=True,
                              converged=False, started=started, initialization=start)
        gradient = None
        iterations = config["torch_iterations"]
        if not isinstance(iterations, int) or iterations < 1:
            raise ValueError("torch_iterations must be positive")
        for iteration in range(iterations):
            def closure() -> torch.Tensor:
                optimizer.zero_grad(set_to_none=True)
                controls = low + width * torch.sigmoid(logits)
                features = torch.stack([controls[contract.mv_indices.index(index)] if index in contract.mv_indices
                                        else fixed[contract.dv_indices.index(index)] for index in range(contract.n_input)])
                outputs = predictor.predict_tensor(features)
                loss = sum(problem.components(outputs, controls, previous_tensor, backend="torch").values())
                loss.backward()
                if not torch.isfinite(loss) or logits.grad is None or not torch.isfinite(logits.grad).all():
                    raise ValueError("Non-finite inverse objective or gradient")
                return loss
            if method == "LBFGS":
                optimizer.step(closure)
            else:
                closure()
                optimizer.step()
            closure()
            gradient = float(torch.linalg.vector_norm(logits.grad))
            converged = gradient <= config["gradient_tolerance"]
            controls = (low + width * torch.sigmoid(logits)).detach().cpu().numpy()
            candidate = problem.result(controls, disturbances, previous, iterations=iteration + 1,
                status="Converged" if converged else "Iteration_Limit", success=True, converged=converged,
                started=started, initialization=start, gradient_norm=gradient)
            if (candidate.feasible and not best.feasible) or (candidate.feasible == best.feasible and candidate.objective < best.objective):
                best = candidate
            if converged:
                break
        best.iterations = iteration + 1
        best.runtime = perf_counter() - started
        best.gradient_norm = gradient
        best.converged = converged
        best.status = "Converged" if converged else "Iteration_Limit"
        results.append(best)
    return select_result(results)


def solve_casadi(problem: OptimizationProblem, disturbances: np.ndarray, previous: np.ndarray,
                 config: dict[str, Any], *, symbolic: SymbolicModel | None = None,
                 warm_start: np.ndarray | None = None) -> OptimizationResult:
    """Scaled IPOPT NLP with hard CV limits and explicit normalised soft slacks."""
    validate_solver_settings(config)
    symbolic = symbolic or convert_model(problem.predictor)
    contract = problem.predictor.contract
    lower, upper = problem.effective_bounds(previous, disturbances)
    scale, mean = contract.input_scale[contract.mv_indices], contract.input_mean[contract.mv_indices]
    results = []
    for start in starting_points(lower, upper, previous, config["multistart"], config["seed"], warm_start):
        started = perf_counter()
        opti = ca.Opti()
        scaled = opti.variable(len(lower))
        controls = ca.DM(mean) + ca.DM(scale) * scaled
        features = ca.vertcat(*[controls[contract.mv_indices.index(index)] if index in contract.mv_indices
                                else disturbances[contract.dv_indices.index(index)] for index in range(contract.n_input)])
        outputs = symbolic.forward(features)
        slack_low, slack_high = opti.variable(contract.n_output), opti.variable(contract.n_output)
        soft_cost = ca.sum1(ca.DM(problem.soft_weights) * (slack_low**2 + slack_high**2))
        objective = sum(problem.components(outputs, controls, ca.DM(previous), backend="casadi", soft_slack=soft_cost).values())
        opti.minimize(objective)
        opti.subject_to(opti.bounded((lower - mean) / scale, scaled, (upper - mean) / scale))
        normalized_outputs = (outputs - ca.DM(contract.output_mean)) / ca.DM(contract.output_scale)
        opti.subject_to(opti.bounded((problem.hard_lower - contract.output_mean) / contract.output_scale,
                                    normalized_outputs, (problem.hard_upper - contract.output_mean) / contract.output_scale))
        opti.subject_to(slack_low >= 0)
        opti.subject_to(slack_high >= 0)
        opti.subject_to((ca.DM(problem.soft_lower) - outputs) / ca.DM(contract.output_scale) <= slack_low)
        opti.subject_to((outputs - ca.DM(problem.soft_upper)) / ca.DM(contract.output_scale) <= slack_high)
        opti.set_initial(scaled, (start - mean) / scale)
        initial_outputs = problem.predictor.predict(contract.assemble(start, disturbances))
        opti.set_initial(slack_low, np.maximum((problem.soft_lower - initial_outputs) / contract.output_scale, 0))
        opti.set_initial(slack_high, np.maximum((initial_outputs - problem.soft_upper) / contract.output_scale, 0))
        opti.solver("ipopt", {"print_time": False}, {"print_level": 0, "sb": "yes", "bound_relax_factor": 0,
            "max_iter": config["ipopt_max_iterations"], "max_cpu_time": config["ipopt_max_cpu_time"], "tol": config["ipopt_tolerance"]})
        failure = None
        try:
            solution = opti.solve()
            candidate = np.asarray(solution.value(controls)).reshape(-1)
            slacks = [np.asarray(solution.value(slack_low)).reshape(-1).tolist(), np.asarray(solution.value(slack_high)).reshape(-1).tolist()]
        except RuntimeError as error:
            failure = str(error).splitlines()[-1]
            candidate, slacks = start, None
        stats = opti.stats()
        success = bool(stats.get("success", False)) and failure is None
        results.append(problem.result(candidate, disturbances, previous, iterations=int(stats.get("iter_count", 0)),
            status=str(stats.get("return_status", "Solver_Failure")), success=success, converged=success,
            started=started, initialization=start, diagnostics={"slacks_normalised": slacks, "failure": failure}))
    return select_result(results)