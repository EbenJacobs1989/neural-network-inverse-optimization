# Neural Network Inverse Optimization

Project scaffold for generating process data, training and validating a neural
network model, performing inverse optimization, comparing optimizers, and
running closed-loop simulations.

Run `notebooks/01_generate_data.ipynb` with a Python kernel from the `notebooks/`
directory to generate `data/synthetic_process_data.csv` and inspect time-series,
output-distribution, and correlation plots. The notebook requires NumPy, pandas,
SciPy, Matplotlib, and seaborn; its `generate_process_data` function returns a
reproducible DataFrame with 8,000 sequential observations by default.

Run `notebooks/02_train_model.ipynb` next from `notebooks/` with PyTorch and
scikit-learn installed. It trains a 10-32-32-16-2 SiLU network with chronological
70/15/15 train/validation/test splits and train-only input/output scaling.
The notebook saves `process_model.pt`, `input_scaler.pkl`, and `output_scaler.pkl`
in `models/`, plus `training_metrics.csv`, `parity_plot.png`,
`residual_plots.png`, and `training_curves.png` in `results/`. The model
checkpoint contains the `state_dict`, column ordering, hidden sizes, and best
epoch; only load pickle scalers from trusted sources.

Run `notebooks/03_validate_model.ipynb` from `notebooks/` to evaluate the saved
model on the final 15% of the time series. It exports held-out R², MAE and RMSE,
actual/predicted and residual diagnostics, physical-unit automatic-differentiation
gradients and unit-comparable sensitivity rankings, one-at-a-time responses and
control-interaction contours. CSV tables and `validation_*.png` figures are saved
in `results/`. The neural model is static; its gradients and sweeps are not
dynamic or causal plant responses.

`src/objective_functions.py` provides `InverseProcessObjective` for differentiable
inverse optimization. Supply the frozen surrogate from the training checkpoint,
its train-fitted scaler `mean_` and `scale_` vectors, control bounds, and
training-only support bounds (for example, per-input 1st/99th percentiles of
the first 70% of the data). All bounds and optional quality/energy limits are
in physical units. Call it with tensors of five fixed disturbances, five
candidate controls (with `requires_grad=True`), and five previous controls,
all in checkpoint input order and the model's dtype/device. The scalar loss
rewards standardized quality, penalizes standardized energy, scaled control
moves, squared control/output limit violations, and excursions beyond the
training support envelope. Weights are configurable and nonnegative; the
model parameters are frozen, while gradients flow to candidate controls.
The support envelope is per-variable, not a guarantee that joint operating
conditions are represented in the training data.

`src/constraints.py` offers `ControlProjection`, `SoftplusConstraintPenalty`,
and `SigmoidControlParameterization` with default normalized actuator ranges
`[0.2, 1.0]`, `[0.1, 0.9]`, `[0.0, 1.0]`, `[0.2, 1.0]`, `[0.1, 0.9]` and a
per-control move limit of `0.05` per step. Pass either one five-element control
vector or a tensor shaped `(..., horizon, 5)` plus the preceding control vector
shaped `(..., 5)`. Projection and sigmoid mapping enforce both constraints;
the differentiable softplus penalty discourages violations without enforcing
feasibility. These bounds are **dimensionless actuator commands**, not the
physical-unit model features. Convert commands to physical units before
evaluating the surrogate or inverse objective; use matched physical bounds
in the objective to avoid mixing units.

Run `notebooks/04_inverse_optimization.ipynb` from `notebooks/` after training.
It reproducibly selects a feasible held-out test row, holds its five disturbances
fixed, and minimizes `InverseProcessObjective` with Adam over five logits.
`SigmoidControlParameterization` keeps normalized commands within their actuator
bounds and within 0.05 of the starting commands. The conversion from normalized
commands to physical controls uses the first 70% of the data's 1st/99th
percentile limits; those same train-only limits define the support envelope.
Quality/energy limits are set relative to the starting prediction and penalized
(not guaranteed). The notebook saves convergence, stepwise objective/predictions/
violations/moves, starting and final controls, and model-versus-original-plant
comparisons to `results/inverse_optimization_*`. Original-plant improvements are
noise-free steady-state counterfactuals with identical disturbances, not
instantaneous measured changes from the lagged process.

Run `notebooks/05_compare_optimizers.ipynb` from `notebooks/` to benchmark
Adam, SGD, RMSprop and LBFGS on the same 100 seeded, feasible held-out test
points. Each gets 60 steps with the same frozen model, objective, output
limits, training-support penalty and sigmoid-enforced actuator/rate limits.
The notebook saves per-scenario runs, per-step convergence and a summary
ranking as `results/optimizer_benchmark_*.csv`, plus box and convergence
plots as PNGs. It selects a best-balanced optimizer by equally ranking
feasibility rate, median and 10th-percentile objective gain, improvement
rate and median wall-clock time. Times depend on hardware; LBFGS closure
evaluations are recorded separately from steps. Physical control distances
mix units, so compare normalized command distances across controls instead.

The second half of notebook 05 extends the same 100-scenario benchmark to
Adam, LBFGS, particle swarm optimization (PSO), differential evolution (DE)
and hybrid PSO+LBFGS. For each scenario it records the neural-network
recommendation and a separately computed, approximate optimum of the
original noise-free steady-state generating function. Both objectives use the
same training-only scaling, rate-feasible control domain, starting controls,
output-limit penalties and support penalty. `results/optimizer_control_runs.csv`
contains predicted and true outputs/objectives, true regret, optimization
error relative to the best neural solution found, model exploitation (predicted
minus true objective gain), off-policy generalization error, violation metrics,
solve time and recommended controls. `results/optimizer_true_optima.csv` records
each numerical plant reference; `results/optimizer_control_summary.csv` and
`results/optimizer_control_report.md` rank the five methods by actual
feasibility, typical/tail true regret, model exploitation and speed. A figure
and per-method convergence CSV are also exported. The true reference is a
two-seed global search with local polishing, **not** a certified global
optimum. Read the report's feasibility and tail-risk warnings: a higher
predicted gain does not imply a safe or profitable closed-loop controller.

The final section of notebook 05 runs a separate **full-dataset quality check**
across all 8,000 observations and all five methods. Unlike the deeper
100-scenario benchmark, this screening sweep uses eight optimizer
steps/generations per method and a twelve-generation true-plant reference
search with local polishing. It projects starting controls that fall outside
normalized actuator bounds, records the original and projected setpoints,
and never removes rows or modifies disturbances. The notebook prints the
complete per-row quality, constraint-violation and true-plant reference
matrices; they are also saved in
`results/optimizer_full_dataset_quality_printout.txt` and
`results/optimizer_full_dataset_*.csv` for searching and analysis. The
generated `optimizer_full_dataset_report.md` ranks methods on **held-out
test** rows; the full summary includes separate train, validation and test
matrices. Projection and in-sample results must not be confused with
historical operating performance or a deployment-ready controller.

Run `notebooks/06_closed_loop_simulation.ipynb` from `notebooks/` to replay the
last 1,000 test disturbances with historical operator, Adam, and LBFGS
controllers. All three start from the same reconstructed true-process state
and use the same seeded measurement noise. At every step the optimizers
minimize the frozen surrogate objective with fixed observed disturbances,
apply one rate-feasible setpoint, and advance the original synthetic plant
with its quality/energy lags. Output limits use fixed training-set quantiles
and soft penalties; violations are logged both for predicted and measured
outputs. Historical operator setpoints are not altered to obey limits.
The notebook exports per-step trajectories, policy summaries, and disturbance,
setpoint, output and objective/benefit figures to `results/closed_loop_*`.
Economic benefit is **unitless**, not monetary: cumulative differences from
the operator in measured quality and energy standardized using training
scales, less 0.1 times squared normalized physical control moves. It excludes
unmodeled operating costs and the surrogate's support/constraint penalties.
Compare the exported model-prediction errors with the realized economic score:
a lower surrogate objective does not imply better true-plant operation.
