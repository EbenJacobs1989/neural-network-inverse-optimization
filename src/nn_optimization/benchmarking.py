"""Seeded global-search adapters and honest, common-objective benchmark summaries."""

from time import perf_counter
from typing import Any

import numpy as np
import pandas as pd
from scipy.optimize import NonlinearConstraint, differential_evolution

from .optimization import OptimizationProblem, OptimizationResult, solve_casadi, solve_torch


def solve_global(problem: OptimizationProblem, disturbances: np.ndarray, previous: np.ndarray,
                 config: dict[str, Any], method: str) -> OptimizationResult:
    """Global exploration; feasibility-first selection does not turn penalties into hard constraints."""
    started = perf_counter()
    lower, upper = problem.effective_bounds(previous, disturbances)
    def score(controls: np.ndarray) -> tuple[float, float]:
        _, components, violations = problem.evaluate(controls, disturbances, previous)
        excess = sum(max(value - problem.tolerance, 0) for value in violations.values())
        return excess, sum(components.values())
    iterations = config["pso_iterations"]
    if method == "DE":
        hard_constraint = NonlinearConstraint(lambda controls: problem.evaluate(controls, disturbances, previous)[0],
                                             problem.hard_lower, problem.hard_upper)
        result = differential_evolution(lambda controls: score(controls)[1], list(zip(lower, upper)), seed=config["seed"],
                                        maxiter=iterations, popsize=max(2, config["pso_particles"] // len(lower)), polish=False,
                                        x0=np.clip(previous, lower, upper), tol=1e-8, constraints=(hard_constraint,))
        best, status, converged = result.x, str(result.message), bool(result.success)
        iterations = int(result.nit)
    elif method == "PSO":
        rng = np.random.default_rng(config["seed"])
        count = config["pso_particles"]
        if count < 2 or iterations < 1:
            raise ValueError("PSO requires at least two particles and one iteration")
        positions = rng.uniform(lower, upper, (count, len(lower)))
        positions[0] = np.clip(previous, lower, upper)
        velocity = np.zeros_like(positions)
        personal = positions.copy()
        scores = [score(controls) for controls in personal]
        best = personal[min(range(count), key=lambda index: scores[index])].copy()
        for _ in range(iterations):
            velocity = (config["pso_inertia"] * velocity
                        + config["pso_cognitive"] * rng.random(positions.shape) * (personal - positions)
                        + config["pso_social"] * rng.random(positions.shape) * (best - positions))
            positions = np.clip(positions + velocity, lower, upper)
            for index, controls in enumerate(positions):
                candidate_score = score(controls)
                if candidate_score < scores[index]:
                    personal[index], scores[index] = controls, candidate_score
            best = personal[min(range(count), key=lambda index: scores[index])].copy()
        status, converged = "PSO_Budget_Complete", False
    else:
        raise ValueError("Global method must be PSO or DE")
    return problem.result(best, disturbances, previous, iterations=iterations, status=status, success=True,
        converged=converged, started=started, initialization=previous,
        diagnostics={"algorithm": method, "hard_constraints": "postvalidated, not guaranteed by search"})


def run_method(method: str, problem: OptimizationProblem, disturbance: np.ndarray, previous: np.ndarray,
               config: dict[str, Any]) -> OptimizationResult:
    started = perf_counter()
    if method in {"Adam", "LBFGS"}:
        result = solve_torch(problem, disturbance, previous, dict(config, torch_method=method))
    elif method == "IPOPT":
        result = solve_casadi(problem, disturbance, previous, config)
    elif method in {"PSO", "DE"}:
        result = solve_global(problem, disturbance, previous, config, method)
    elif method == "PSO+LBFGS":
        exploration = solve_global(problem, disturbance, previous, config, "PSO")
        local = solve_torch(problem, disturbance, previous, dict(config, torch_method="LBFGS"), warm_start=exploration.controls)
        result = min([exploration, local], key=lambda candidate: (not candidate.feasible, candidate.objective))
        result.diagnostics["hybrid"] = {"PSO": exploration.objective, "LBFGS": local.objective}
    else:
        raise ValueError(f"Unknown optimizer {method}")
    result.runtime = perf_counter() - started
    return result


def benchmark_static(problem: OptimizationProblem, scenarios: list[tuple[np.ndarray, np.ndarray]],
                     config: dict[str, Any], repeats: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Repeat identical scenario/seed pairs; report runtime and initialization sensitivity separately."""
    if repeats < 1:
        raise ValueError("benchmark repeats must be positive")
    rows = []
    for scenario_index, (disturbance, previous) in enumerate(scenarios):
        for method in ("Adam", "LBFGS", "PSO", "DE", "PSO+LBFGS", "IPOPT"):
            for repeat in range(repeats):
                result = run_method(method, problem, disturbance, previous, config)
                rows.append({"scenario": scenario_index, "method": method, "repeat": repeat, "seed": config["seed"],
                    **result.to_dict(), "hard_constraint_violation": max(result.violations.values()),
                    "mv_travel": float(np.abs(result.controls - previous).sum()),
                    "initialisation_objective_spread": float(np.ptp([entry["objective"] for entry in result.diagnostics.get("starts", [{"objective": result.objective}])]))})
    frame = pd.DataFrame(rows)
    summaries = []
    for (scenario, method), group in frame.groupby(["scenario", "method"], sort=False):
        summaries.append({"scenario": scenario, "method": method, "objective_mean": float(group.objective.mean()),
            "objective_repeatability_std": float(group.objective.std(ddof=0)),
            "runtime_p50": float(group.runtime.quantile(.5)), "runtime_p95": float(group.runtime.quantile(.95)),
            "feasible_fraction": float(group.feasible.mean()), "runs": len(group)})
    return frame, pd.DataFrame(summaries)


def closed_loop_metrics(frame: pd.DataFrame, t_step: float) -> dict[str, Any]:
    return {"steps": len(frame), "integrated_absolute_error": float(frame.absolute_error.sum() * t_step),
            "total_mv_travel": float(frame.mv_travel.sum()), "hard_cv_violation_steps": int((frame.hard_cv_violation > 1e-6).sum()),
            "governance_rejections": int((~frame.accepted).sum()), "runtime_p50": float(frame.runtime.quantile(.5)),
            "runtime_p95": float(frame.runtime.quantile(.95)), "final_predicted_cv": frame.iloc[-1].predicted_cv,
            "mean_objective": float(frame.objective.mean())}