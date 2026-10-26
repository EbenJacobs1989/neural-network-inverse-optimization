# Current State Audit

Audit date: 2026-10-23. Scope: this repository, not the other workspace projects.

## Source-control and environment baseline

- Branch: `main`, tracking `origin/main`.
- Pre-existing modifications: `notebooks/04_inverse_optimization.ipynb`,
  `src/optimizers.py`; untracked `tests/test_optimizers.py` and Python caches.
  All are preserved. Git is available through GitHub Desktop, not PATH.
- VS Code selects Python 3.14 at the user-local Python installation. No local
  virtual environment or dependency manifest exists. PyTorch 2.14.0, NumPy
  2.5.2, CasADi 3.8.1, pandas 3.0.5, SciPy 1.18.1 and Matplotlib 3.11.1 are
  installed. A corrected distribution-name check verifies do_mpc 5.1.2 already
  installed; only PyYAML and pytest require installation. Do not upgrade Torch.

## Current directory structure

```text
README.md
notebooks/
  01_generate_data.ipynb
  02_train_model.ipynb
  03_validate_model.ipynb
  04_inverse_optimization.ipynb
  05_compare_optimizers.ipynb
  06_closed_loop_simulation.ipynb
src/
  constraints.py
  objective_functions.py
  optimizers.py
  neural_model.py          (docstring stub)
  process_generator.py     (stub)
  simulator.py             (stub)
  plotting.py              (stub)
tests/
  test_constraints.py
  test_objective_functions.py
  test_optimizers.py
data/                     (synthetic sequential CSV; contents omitted)
models/                   (checkpoint and two scaler pickles; contents omitted)
results/                  (CSV, PNG, markdown and text artifacts; omitted)
```

## Execution flow and neural-model contract

Notebooks are the existing entry points and assume a `notebooks/` working
directory. Notebook 01 generates 8,000 sequential synthetic samples with
correlated disturbances, smooth controls, lagged outputs and measurement noise.
Notebook 02 splits chronologically 70/15/15 before fitting StandardScalers,
trains with AdamW, validation stopping, scheduling and gradient clipping, and
saves a checkpoint. Notebook 03 evaluates held-out prediction and sensitivities.
Notebook 04 chooses one bounded move. Notebook 05 benchmarks Adam, SGD,
RMSprop, LBFGS, PSO, DE and PSO+LBFGS, including original-plant comparisons.
Notebook 06 replays a lagged original plant using common disturbances and noise.

Verified checkpoint: PyTorch, float32, CPU-compatible, `network` Sequential
10 -> 32 -> 32 -> 16 -> 2, Linear layers with SiLU between hidden layers.
Model weights are `network.{0,2,4,6}.{weight,bias}`. No history or state is an
input: this is a **static current-output regression of lagged measurements**,
not an identified steady-state or transition model.

Exact input order: d1_feed_grade, d2_hardness, d3_moisture, d4_impurity,
d5_ambient, u1_feed_rate, u2_water_rate, u3_reagent_rate, u4_speed,
u5_temperature_setpoint. Outputs: y1_quality, y2_energy_load.
Both trusted local scaler pickles contain matching feature names and
`n_samples_seen_=5600`. Transform is `(input - mean) / scale`; inverse output
transform is `scaled_output * scale + mean`. Physical unit symbols and a real
sampling interval are not certified in the checkpoint and must not be invented.

## Existing inverse formulation

Frozen forward predictor; optimise controls, never train an inverse network.
`InverseProcessObjective` minimises negative standardised quality plus
standardised energy, quadratic scaled moves, squared ReLU output violations,
control violations and departures from marginal training support.
Command bounds are [0.2,0.1,0,0.2,0.1] to [1,0.9,1,1,0.9]; command move limit
is 0.05. Physical controls use training 1/99 percentiles as origin and span.
Projection and sigmoid parameterisation enforce MV and rate bounds. CV and
support penalties are **soft**, not guaranteed safety constraints.
The pre-existing do-mpc adapter has horizon one and a previous-MV state only;
it is a static optimisation adapter, not a dynamic plant controller.

## Strengths to preserve

- Chronological scaler fitting; saved feature-order metadata; frozen networks.
- Deterministic seeds, bounds/rate tests, derivative checks in existing adapter.
- Independent original-plant comparisons, held-out reporting and candid negative
  closed-loop outcomes. No claim of optimizer superiority is justified a priori.
- Existing notebooks, checkpoint, scalers, result artifacts and experiments.

## Risks and compatibility issues

- Architecture/loading/scaling duplicated across notebooks; package modules are
  mostly stubs; no installable package or external configuration.
- Training on dynamic measurements confounds a putative steady-state mapping.
- Per-feature quantiles do not certify joint support. Fixed disturbances can
  make a scenario infeasible regardless of the selected controls.
- Pickles require trusted provenance. No model confidence calibration, real
  timestamps, analyser delay specification, certified engineering units, plant
  safety limits or plant-identified controller dynamics are provided.
- Float32 notebook inference differs from float64 symbolic evaluation; report
  native-float32 and promoted-float64 comparisons separately.
- Notebook-relative paths and embedded constants complicate root execution.
- Existing source has no TODO/FIXME or bare-except markers; notebooks contain
  clipping and fixed research budgets, seeded RNG and repeated scaler setup.
  Clipping must not hide governance rejection. Historical data/results remain
  unchanged. No temporal leak was found in the documented split/scaler flow;
  off-policy validation and model drift remain open research risks.
- Full-data optimizer screening includes training rows; only temporal holdout
  supports generalisation claims. Module-wide seeding/thread settings in
  notebooks are global side effects; new library code will avoid these.

## Implementation plan and file impact

Retain all existing source APIs and notebooks. Add a compact installable
`src/nn_optimization/` package rather than duplicate the proposed empty module
tree. Working modules will own contract/loading, symbolic conversion,
objective/constraints/results, inverse solvers, forecasts/dynamic MPC,
governance, simulation and benchmarking. Create root-executable scripts for
conversion validation, inverse solves, MPC simulation and benchmarks.
Create `configs/{model,inverse_optimization,mpc,example.local}.yaml`, focused
deterministic tests, `pyproject.toml`, `.gitignore`, and architecture, contract,
formulation, MPC, validation and migration documents. Modify README only for
new root commands and safety distinctions. Preserve earlier adapter behavior.

## Unverified assumptions and production boundaries

Example envelope/bounds are derived only from the synthetic training split.
Example hard limits are demonstration limits, not approved plant limits.
Wrapper alpha/time constants, delay, freshness and confidence thresholds need
plant engineering approval. Forecast persistence is explicitly an assumption.
Measured CV feedback plus memory of the actually approved MV is sufficient
only for the temporary wrapper. No EKF/MHE or plant-write interface is claimed.
The cheapest first behavioral check is prediction **and Jacobian** equivalence
on deterministic interior/boundary points with the frozen checkpoint.