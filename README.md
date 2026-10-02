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
