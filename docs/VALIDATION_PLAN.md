# Validation Plan

## Executed Numerical and Software Gates

- Contract: dimensions, order, dtype, positive scales, non-finite inputs and
  feature metadata. Loader checks original checkpoint and scaler split.
- Conversion: engineering predictions **and full Jacobians**, supported layers,
  scalers, unsupported-layer failures; 128-point real-checkpoint validation.
- Objective: identical NumPy/Torch/CasADi cost at identical engineering points.
- Inverse: MV/rate feasibility, empty intersections, multistart repeatability,
  improvement, structured diagnostics, hard infeasibility and explicit slacks.
- Governance: success/status, NaN/inf, input/output support, limits, freshness,
  bounded fallback, unavailable required confidence and independent horizon
  validation despite a candidate's false feasibility claim.
- MPC: discrete model/controller construction, complete forecasts, interval and
  dimension checks, two receding-horizon solves and terminal hard constraints.
- Workflow: deterministic small synthetic plant with different dynamics, DV and
  target steps; real-checkpoint independent-plant simulation and benchmarks.
- Compatibility: run the existing constraint, objective and static adapter tests
  alongside the package tests. Do not rerun/rewrite expensive legacy notebooks
  as a side effect of an ordinary unit test.

```powershell
python -m pytest -q
python scripts/validate_conversion.py
python scripts/run_inverse_optimization.py
python scripts/run_mpc_simulation.py --steps 12
python scripts/benchmark_optimizers.py --repeats 2 --steps 8
```

Results are synthetic demonstration results, never plant results. Small repeat
counts make runtime percentiles descriptive only. Static solver comparisons and
closed-loop policy outcomes must not be merged into one misleading ranking.

## Required Before Advisory Plant Use

| Test | Required evidence and release decision |
|---|---|
| Historical replay | Aligned held-out plant signals and proposed moves; no plant writes |
| Temporal holdout | Chronological split before transforms; no random leakage across time |
| Off-policy validation | Prediction and sensitivity checks around proposed, not only historical, MVs |
| Target steps | Identified settling behaviour; feasible transients and terminal states |
| Disturbance steps/forecasts | Persistence baseline, forecast bias and missing-horizon rejection |
| Noise and analyser delays | Identified alignment, filtering and estimator confidence |
| Missing/stale data | Immediate rejection plus bounded approved fallback, no NaN propagation |
| Infeasible targets | Honest infeasibility, no hidden relaxation of hard safety bounds |
| MV saturation/rate limits | Physical-unit tests at all corners and after actuator interventions |
| Plant/model mismatch | Validated lag, gain, bias, delay and unmeasured-disturbance sweeps |
| Solver failure/timeout | Rejected result, captured status and fallback expiry |
| Envelope excursion | Per-feature report and no silent extrapolation or clipping |
| Model drift | Monitored residuals with engineering-approved disable/revalidation criteria |
| Governance intervention | Last accepted versus applied MV reconciliation and external authority |

## Shadow and Advisory Acceptance

Shadow mode requires approved signal contracts, certified hard limits, replay
coverage, engineering economics, identified dynamics, confidence policy,
performance bounds, fallback ownership and independently reviewed logs. No
commands are released to a plant during shadow collection.

Advisory mode requires a signed acceptance of target tracking, constraint
violations, feasibility fraction, stale data, solver time tails, analyser timing,
operator handling and external DCS/SIS protection. The framework intentionally
does not set invented numerical plant acceptance thresholds. Initial operation
must retain existing APC/manual strategy and independently auditable approvals.

## Not Yet Validated

No real plant deployment, long-horizon stability proof, robust invariant set,
delay identification, EKF/MHE, confidence calibration, real engineering units,
plant-economic calibration or exhaustive scenario campaign is claimed. Standard
tests use small deterministic non-confidential fixtures. Checkpoint-dependent
scripts require the existing trusted local synthetic artifacts.