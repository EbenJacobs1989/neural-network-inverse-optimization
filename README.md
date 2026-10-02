# Neural Network Inverse Optimization

**A reproducible, end-to-end research playground for data-driven process control.**

Generate a simulated mineral-processing plant, learn its input/output behavior,
optimize manipulated variables with a frozen neural network, test the proposed
actions against the *original* plant equation, and finally replay them in a
lagged closed loop. The important question is not simply whether an optimizer
reduces its model's loss: **does the real simulated process improve?**

> [!IMPORTANT]
> This is a synthetic research example, **not a deployable controller**. The
> learned surrogate is static, whereas the original plant has first-order
> dynamics and measurement noise. Output and training-support constraints are
> soft penalties. Review original-plant and closed-loop results before drawing
> process-control conclusions.

## The experiment at a glance

```text
8,000 sequential synthetic observations
       │
       ├── 5 correlated, slow disturbances (d1–d5)
       ├── 5 smooth operator-controlled inputs (u1–u5)
       └── 2 lagged, noisy measurements (quality and energy)
       │
       ▼
Chronological train / validation / test split (70% / 15% / 15%)
       │
       ▼
10 → 32 → 32 → 16 → 2 SiLU neural surrogate
       │
       ├── Hold disturbances fixed; optimize bounded control moves
       ├── Check recommendations against the original plant equation
       └── Replay 1,000 steps with original plant dynamics
```

| Objective | Meaning |
|---|---|
| **Increase quality** | Reward `y1_quality` in the optimization loss. |
| **Reduce energy** | Penalize `y2_energy_load` in the optimization loss. |
| **Avoid unsafe moves** | Enforce actuator bounds and a maximum normalized command move of `0.05` per step. |
| **Stay credible** | Penalize predicted output violations and departures from a training-derived support envelope. |
| **Check reality** | Measure actual static-plant errors and lagged closed-loop outcomes, not just neural predictions. |

### Inputs, outputs and units

| Role | Columns |
|---|---|
| Observed disturbances, held fixed during each decision | `d1_feed_grade`, `d2_hardness`, `d3_moisture`, `d4_impurity`, `d5_ambient` |
| Manipulated variables, in the generator's physical units | `u1_feed_rate`, `u2_water_rate`, `u3_reagent_rate`, `u4_speed`, `u5_temperature_setpoint` |
| Outputs | `y1_quality`, `y2_energy_load` |

The actuator **commands** in the constraints module are dimensionless. A
training-derived 1st/99th-percentile transformation maps them to the
**physical-unit controls** expected by the network and generating function.
Do not apply `[0, 1]` command limits directly to the CSV's physical controls.

## Quick start

From the project root, use a Python environment with Jupyter, NumPy, pandas,
SciPy, Matplotlib, seaborn, scikit-learn, PyTorch, `nbformat` and `nbclient`.
For example:

```powershell
python -m pip install jupyter numpy pandas scipy matplotlib seaborn scikit-learn torch nbformat nbclient ipykernel
Set-Location notebooks
jupyter lab
```

Open and **run all cells in order** in notebooks `01` through `06`. Notebook
paths such as `../data/`, `../models/` and `../results/` assume that the kernel
works from `notebooks/`. Notebook 05's all-row screening pass prints tens of
thousands of lines and takes substantially longer than the others; the complete
printout is also saved as a searchable text file.

### Notebook roadmap

| # | Notebook | Why run it | Key artifacts |
|---:|---|---|---|
| 01 | [Generate data](notebooks/01_generate_data.ipynb) | Build a smooth original plant with correlated AR disturbances, gradual operator actions, nonlinear steady-state responses, output lags and measurement noise; plot all variables. | `data/synthetic_process_data.csv` |
| 02 | [Train model](notebooks/02_train_model.ipynb) | Fit a 10–32–32–16–2 SiLU PyTorch model; split in time *before* fitting scalers; use AdamW, validation-based stopping, a scheduler and gradient clipping. | `models/process_model.pt`, `models/*_scaler.pkl`, `results/training_metrics.csv`, training/parity/residual PNGs |
| 03 | [Validate model](notebooks/03_validate_model.ipynb) | Measure held-out R², MAE and RMSE; inspect prediction and residual structure; calculate local automatic-differentiation sensitivities and conditional response surfaces. | `results/validation_*.csv`, `results/validation_*.png` |
| 04 | [Optimize one point](notebooks/04_inverse_optimization.ipynb) | Freeze a seeded held-out disturbance scenario and optimize one bounded, rate-feasible control change using Adam. Compare predicted versus original **steady-state** improvement. | `results/inverse_optimization_*.csv`, convergence PNG |
| 05 | [Compare optimizers](notebooks/05_compare_optimizers.ipynb) | Benchmark gradient solvers; compare Adam, LBFGS, PSO, DE and PSO+LBFGS with an approximate original-plant reference; screen all 8,000 rows. | `results/optimizer_benchmark_*`, `results/optimizer_control_*`, `results/optimizer_full_dataset_*` |
| 06 | [Simulate closed loop](notebooks/06_closed_loop_simulation.ipynb) | Replay 1,000 identical disturbances and noise samples for historical operator, Adam and LBFGS policies against the **lagged original plant**. | `results/closed_loop_trajectories.csv`, `closed_loop_summary.csv`, trend PNGs |

### Reusable constraint and objective code

- [`src/objective_functions.py`](src/objective_functions.py) defines
  `InverseProcessObjective`, a scalar differentiable loss combining
  `-standardized_quality + standardized_energy + movement_penalty +
  constraint_penalty + support_penalty`. Give it the *frozen* surrogate,
  training-fitted scaler statistics, physical control bounds, physical
  training-support bounds, five fixed disturbances, five candidate controls
  and five previous controls. Optional output limits are in physical units.
  Gradients flow to candidate controls, not model weights.
- [`src/constraints.py`](src/constraints.py) provides **projection**,
  **differentiable softplus penalties**, and **sigmoid reparameterization**
  for single steps or batched MPC horizons. Normalized `u1`–`u5` command
  ranges are `[0.2,1.0]`, `[0.1,0.9]`, `[0,1]`, `[0.2,1.0]`, `[0.1,0.9]`;
  maximum change is `0.05` per control per step. Projection and sigmoid
  enforce actuator and rate limits, whereas the softplus penalty only
  *discourages* violations.

The support envelope is **per variable**, not proof that a joint operating
point was observed during training. The saved scaler pickles should only be
loaded from trusted local artifacts; pickle is not safe for untrusted files.

## How to read the optimization checks

Notebook 05 deliberately separates three experiments:

1. **Four-solver neural benchmark:** Adam, SGD, RMSprop and LBFGS each get 60
   steps on the **same 100 feasible, seeded held-out test scenarios**.
   `optimizer_benchmark_*.csv` and plots contain run-level results, stepwise
   convergence, solver evaluations and a *neural-prediction-based* ranking.
2. **Five-solver original-plant check:** Adam, LBFGS, particle swarm
   optimization (**PSO**), differential evolution (**DE**) and hybrid
   **PSO+LBFGS** use those same 100 scenarios. Independent, seeded DE searches
   plus bounded local polishing establish an **approximate**, not certified,
   true steady-state optimum on each scenario. See
   [the 100-row report](results/optimizer_control_report.md) and
   `optimizer_control_*.csv` for predicted/true output, error, feasibility
   and tail-risk comparisons.
3. **Full-data quality screening:** The same five methods run over **every
   one of 8,000 rows** with a smaller screening budget: eight
   steps/generations (hybrid: five PSO + three LBFGS), and a separate
   12-generation DE plant reference with local polish. Starting controls
   outside bounds are **projected and recorded**, not silently dropped;
   disturbances are unchanged. The
   [full-data report](results/optimizer_full_dataset_report.md) ranks the
   held-out 1,200 test rows; the summary also separates 5,600 training and
   1,200 validation rows. The **complete printed quality, violation, summary
   and original-plant matrices** live in
   [optimizer_full_dataset_quality_printout.txt](results/optimizer_full_dataset_quality_printout.txt)
   and corresponding `optimizer_full_dataset_*.csv` files.

| Quality check | Definition | What to watch |
|---|---|---|
| **Optimization error** | Candidate neural objective minus the best neural objective *found* for the same scenario. | Gap to a discovered solution, not a certified global minimum. |
| **True regret** | Original-plant objective minus an independently searched original-plant reference. | Higher values mean missed true-process opportunity; the reference is approximate. |
| **Model exploitation** | *Predicted objective improvement* minus *original-plant improvement* under the same penalties. | Positive values mean the surrogate overpromised. |
| **Generalization error** | Absolute prediction error divided by training output scales, at starting and optimized controls. | Optimized controls can expose errors hidden at historical operating points. |
| **Constraint satisfaction** | Actual quality/energy, marginal support, actuator and rate violations below the logged tolerance. | Hard command bounds do not imply hard output feasibility. |

**Do not compare the deep and screening rankings as if they used equal budgets.**
Likewise, training-row screening results are in-sample diagnostics; use the
held-out split to discuss generalization. Solve time depends on hardware and
does not include the cost of building the numerical true-process reference.

## Closed-loop reality check

Notebook 06 reconstructs the original plant's dynamic state from historical
inputs, then subjects three policies to the same final 1,000 disturbances
and the same seeded measurement noise. Operator setpoints are replayed
**unchanged**, and their constraint breaches are logged rather than erased.
Adam and LBFGS choose a new bounded control move at each step with the
frozen surrogate; the **original** plant applies the move with quality and
energy time constants of 8 and 15 samples.

The economic score is a **unitless research proxy, not a currency estimate**:

```text
score = measured_quality_in_training_standard_deviations
      - measured_energy_in_training_standard_deviations
      - 0.1 × squared_control_moves_in_input_standard_deviations

cumulative benefit = Σ(optimizer_score[t] - operator_score[t])
```

In the included simulation, both optimized policies **increase average
quality but worsen the realized score** relative to the historical operator:
Adam's cumulative benefit is approximately `-539.7` score points and LBFGS's
is approximately `-608.9`. Their neural objectives improve, but the lagged
true plant uses more energy. This mismatch is a *finding*, not an error to
conceal: examine `closed_loop_summary.csv`, prediction-bias diagnostics,
output-limit violations and trend plots before claiming a control benefit.

## Repository layout

```text
notebooks/  01–06: executable experiment, narratives and plots
src/        differentiable objective, reusable constraints and module stubs
data/       generated sequential CSV
models/     saved checkpoint and training-fitted scalers
results/    diagnostics, optimization matrices, reports and figures
tests/      focused constraint and objective tests
```

To run the focused module tests from the project root:

```powershell
python -m unittest discover -s tests -p "test_*.py" -v
```

**Next research steps:** evaluate uncertainty or ensembles under off-policy
actions, account explicitly for state and lag during optimization, define
plant-calibrated economics and hard safety constraints, and validate any
proposed policy independently before attempting deployment.
