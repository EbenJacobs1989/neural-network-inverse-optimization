"""Plot simulation artifacts separately from optimisation and plant logic."""

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .optimization import OptimizationProblem


def plot_simulation(frame: pd.DataFrame, problem: OptimizationProblem, path: Path) -> None:
    contract = problem.predictor.contract
    cv, controls, targets = np.stack(frame.measured_cv), np.stack(frame.controls), np.stack(frame.target)
    disturbances, rates = np.stack(frame.disturbance), np.stack(frame.mv_rate)
    count = contract.n_output + 2 * len(contract.mv_indices) + len(contract.dv_indices) + 4
    figure, axes = plt.subplots(count, 1, figsize=(12, 2 * count), sharex=True, constrained_layout=True)
    row = 0
    for index, name in enumerate(contract.output_names):
        axes[row].plot(frame.time, cv[:, index], label=name)
        axes[row].plot(frame.time, targets[:, index], "--", label="target")
        for limit, style, label in ((problem.hard_lower[index], ":", "hard limits"), (problem.hard_upper[index], ":", None),
                                   (problem.soft_lower[index], "--", "preferred limits"), (problem.soft_upper[index], "--", None)):
            axes[row].axhline(limit, color="grey", linestyle=style, label=label)
        axes[row].legend(fontsize=7)
        row += 1
    for index, feature in enumerate(contract.mv_indices):
        axes[row].step(frame.time, controls[:, index], label=contract.input_names[feature])
        axes[row].axhline(problem.lower[index], linestyle=":", color="grey")
        axes[row].axhline(problem.upper[index], linestyle=":", color="grey")
        axes[row].legend(fontsize=7)
        row += 1
    interval = float(frame.time.iloc[1] - frame.time.iloc[0]) if len(frame) > 1 else 1.
    for index in range(len(problem.lower)):
        axes[row].step(frame.time, rates[:, index], label=f"MV {index + 1} rate")
        axes[row].axhline(problem.max_move[index] / interval, linestyle=":", color="grey")
        axes[row].axhline(-problem.max_move[index] / interval, linestyle=":", color="grey")
        axes[row].legend(fontsize=7)
        row += 1
    for index, feature in enumerate(contract.dv_indices):
        axes[row].plot(frame.time, disturbances[:, index], label=contract.input_names[feature])
        axes[row].legend(fontsize=7)
        row += 1
    axes[row].plot(frame.time, frame.objective, label="objective")
    axes[row + 1].step(frame.time, frame.accepted.astype(int), label="governance accepted")
    axes[row + 1].step(frame.time, frame.envelope_ok.astype(int), linestyle="--", label="input envelope")
    codes, labels = pd.factorize(frame.solver_status)
    axes[row + 2].step(frame.time, codes, label="solver status")
    axes[row + 2].set_yticks(range(len(labels)), labels, fontsize=7)
    axes[row + 3].plot(frame.time, frame.slack_max, label="maximum normalised soft slack")
    for axis in axes[row:]:
        axis.legend(fontsize=7)
    axes[-1].set_xlabel("Synthetic sample time")
    figure.suptitle("Synthetic mismatched-plant experiment, not plant validation")
    figure.savefig(path, dpi=110)
    plt.close(figure)