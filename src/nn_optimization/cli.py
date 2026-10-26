"""Root-independent workflows for the existing synthetic checkpoint."""

import argparse
import json
import logging
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .benchmarking import benchmark_static, closed_loop_metrics, run_method
from .model import load_config, load_model
from .optimization import OptimizationProblem, problem_from_config, solve_casadi, solve_torch


def select_scenarios(problem: OptimizationProblem, data: pd.DataFrame, count: int = 2) -> list[tuple[np.ndarray, np.ndarray]]:
    """Use temporal holdout points that actually satisfy the input and actuator envelope."""
    contract = problem.predictor.contract
    start = int(len(data) * (contract.metadata["train_fraction"] + contract.metadata["validation_fraction"]))
    scenarios = []
    for features in data.iloc[start:][contract.input_names].to_numpy():
        controls, disturbances = features[contract.mv_indices], features[contract.dv_indices]
        if (np.all(features >= problem.input_lower) and np.all(features <= problem.input_upper)
                and np.all(controls >= problem.lower) and np.all(controls <= problem.upper)):
            scenarios.append((disturbances.copy(), controls.copy()))
            if len(scenarios) == count:
                return scenarios
    raise ValueError(f"No {count} held-out scenarios inside all configured MV/input limits")


def write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, indent=2, allow_nan=False), encoding="utf-8")


def main(command: str, default_root: Path) -> int:
    parser = argparse.ArgumentParser(description=f"Synthetic advisory AiPC: {command}")
    parser.add_argument("--root", type=Path, default=default_root)
    parser.add_argument("--model-config", type=Path, default=Path("configs/model.yaml"))
    parser.add_argument("--inverse-config", type=Path, default=Path("configs/inverse_optimization.yaml"))
    parser.add_argument("--mpc-config", type=Path, default=Path("configs/mpc.yaml"))
    parser.add_argument("--steps", type=int, default=None)
    parser.add_argument("--repeats", type=int, default=None)
    parser.add_argument("--method", choices=["both", "Adam", "LBFGS", "IPOPT", "PSO", "DE", "PSO+LBFGS"], default="both")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s %(message)s")
    root = args.root.resolve()
    predictor, data, support_lower, support_upper = load_model(root, root / args.model_config)
    inverse = load_config(root / args.inverse_config)
    problem = problem_from_config(predictor, inverse, support_lower, support_upper)
    scenarios = select_scenarios(problem, data)
    disturbance, previous = scenarios[0]
    output = root / inverse["output_path"]
    output.mkdir(parents=True, exist_ok=True)
    if command == "inverse":
        baseline = sum(problem.evaluate(previous, disturbance, previous)[1].values())
        if args.method == "both":
            torch_result = solve_torch(problem, disturbance, previous, inverse)
            nlp_result = solve_casadi(problem, disturbance, previous, inverse, warm_start=torch_result.controls)
            report = {"label": "existing synthetic checkpoint, held-out scenario", "baseline_objective": baseline,
                      "Adam": torch_result.to_dict(), "IPOPT": nlp_result.to_dict()}
            accepted = torch_result.feasible and nlp_result.feasible and nlp_result.success
        else:
            result = run_method(args.method, problem, disturbance, previous, inverse)
            report = {"baseline_objective": baseline, args.method: result.to_dict()}
            accepted = result.feasible and result.success
        write_json(output / "inverse_results.json", report)
        print(json.dumps(report, indent=2))
        return 0 if accepted else 1
    from .simulation import load_original_plant, simulate
    mpc_config = load_config(root / args.mpc_config)
    output = root / mpc_config["output_path"]
    output.mkdir(parents=True, exist_ok=True)
    plant = load_original_plant(root)
    initial_cv = np.asarray(plant(disturbance[None, :], previous[None, :]))[0]
    steps = args.steps if args.steps is not None else mpc_config["simulation_steps"]
    if command == "simulation":
        from .reporting import plot_simulation
        frame = simulate(problem, mpc_config, inverse, disturbance, previous, initial_cv, plant, steps)
        frame.to_csv(output / "mpc_simulation.csv", index=False)
        metrics = closed_loop_metrics(frame, mpc_config["t_step"])
        metrics["label"] = "synthetic original plant with deliberate mismatch; not production validation"
        write_json(output / "mpc_summary.json", metrics)
        plot_simulation(frame, problem, output / "mpc_simulation.png")
        print(json.dumps(metrics, indent=2))
        return 0 if frame.feasible.all() else 1
    if command == "benchmark":
        repeats = args.repeats if args.repeats is not None else inverse["benchmark_repeats"]
        runs, summary = benchmark_static(problem, scenarios, inverse, repeats)
        runs.to_csv(output / "benchmark_static_runs.csv", index=False)
        runs.to_json(output / "benchmark_static_runs.json", orient="records", indent=2)
        summary.to_csv(output / "benchmark_static_summary.csv", index=False)
        summary.to_json(output / "benchmark_static_summary.json", orient="records", indent=2)
        closed = []
        for policy in ("Adam", "LBFGS", "IPOPT", "NMPC"):
            frame = simulate(problem, mpc_config, inverse, disturbance, previous, initial_cv, plant, steps, policy=policy)
            frame.to_csv(output / f"benchmark_closed_loop_{policy}.csv", index=False)
            closed.append({"policy": policy, **closed_loop_metrics(frame, mpc_config["t_step"])})
        write_json(output / "benchmark_closed_loop.json", closed)
        pd.DataFrame(closed).to_csv(output / "benchmark_closed_loop_summary.csv", index=False)
        print(summary.to_string(index=False))
        print(json.dumps(closed, indent=2))
        return 0
    raise ValueError(f"Unknown command {command}")