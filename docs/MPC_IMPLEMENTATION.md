# MPC Implementation

## Static Is Not Dynamic

The saved network predicts current lagged/noisy measurements from current
inputs. Repeated static predictions erase state and cannot represent settling
time, transport delay or analyser dynamics. The temporary first-order wrapper
is an explicit modelling assumption, not a discovery from that checkpoint.

For a first-order time constant tau and interval dt:

$$
\alpha_i=1-\exp(-\Delta t/\tau_i).
$$

The example alpha values correspond to the original synthetic generator's
8/15-sample constants, but are still **placeholders for any real plant**;
`alpha_is_identified: false` records this. They do not repair the bias from
training the static network on dynamic outputs. Identify time constants and
validate step responses before interpreting a real MPC trajectory.

## do-mpc 5.1.2 Model

- Discrete model, native CasADi MX symbols.
- `_x.cv`: measured CV vector; `_x.mv_memory`: prior actually applied MVs.
- `_u.mv`: engineering-unit manipulated variables.
- `_tvp.disturbance` and `_tvp.target`: aligned forecasts including terminal row.
- No artificial uncertain parameters or unobserved process states are added.
- Bounds/rates and terminal hard limits are explicit inspectable constraints.
- Preferred CV limits use separate nonnegative, quadratically penalised slacks.
- Predictions, objective, solver status, iterations, slacks and whole-horizon
  constraint checks are retained. Normal repeated calls keep solver warm starts.

Only the first move is offered to `ValidationGate`. A rejected move is never
used as MV memory. The next state includes the command actually chosen by the
gate or by the external strategy. Solver success alone is not release approval.

## Forecast Interface

`ForecastProvider.horizon(k,N)` must return finite shape `(N+1, feature_count)`.
It supports fixed vectors, supplied time-aligned arrays, callbacks and step or
schedule trajectories. Supplied arrays are indexed from the scenario origin;
short arrays fail rather than repeating their last row. Callbacks receive
`(sample_index, horizon)` and must return all N+1 rows.

```python
disturbance = ForecastProvider.constant(current_dv, sample_interval=t_step)
schedule = ForecastProvider(target_rows, width=n_cv, sample_interval=t_step)
step_target = ForecastProvider.step(before, after, at_sample=20, sample_interval=t_step)
callback = ForecastProvider.callback(forecast_function, width=n_dv, sample_interval=t_step)
```

Persistence explicitly means `d[k+i|k]=d[k]`. There is no fabricated knowledge
of future disturbances. Controller/provider/model intervals must align. Values
are in the model contract's engineering units; a dimensional array cannot
prove unit semantics, so its provider must supply correctly converted signals.
The APIs do not hide conversions or guess timestamps from array position.

## State Feedback and Timing

CVs are directly measured in the demonstration; MV memory is controller-owned
history of applied commands. `MeasuredStateFeedback.state` validates the exact
shape and rejects missing or non-finite readings. The governance interface
requires measurement, solve and current timestamps in one consistent clock,
checks freshness and rejects future-dated data.

Real CV analyser delays, asynchronous sample alignment and filtering are
unknown. External ingestion must align them before this interface. No EKF or
moving-horizon estimator is implemented or claimed: a future estimator must
produce a validated state with the same model-state contract and expose its
confidence, timestamp and rejection policy. MV memory must never be estimated
from an unapproved recommendation.

## Independent Simulation

`simulation.load_original_plant` extracts exactly the trusted generator's
`steady_state_response` AST function without executing notebook cells. It
therefore preserves the original equation without copying constants into the
controller. The simulation supports different plant alpha, gain, output bias,
seeded noise, unmeasured disturbances and a nonnegative integer transport delay.
The controller does **not** assume it knows that plant delay. Delay is deliberately
plant/controller mismatch here, not an identified controller delay model.

The simulation logs candidate, acceptance, fallback tier, applied moves,
measurements, targets, disturbances, rates, costs, slacks and violations. If a
bounded last-good hold expires, the gate returns no command; the **synthetic
external strategy** keeps the previous setpoint. This is logged, not represented
as renewed governance approval. Plant safety and DCS/SIS remain external.

## Transition-Model Upgrade

The controller depends on a discrete model exposing `cv`, `mv_memory`, `mv`,
`disturbance`, `target` and `next_cv`, not on symbolic copies of neural layers
inside its solve loop. `build_dynamic_model` owns the temporary transition.
Replace that construction with a validated one-step transition using the same
state contract; retain forecasts, objective, constraints, governance and replay.
Additional latent/history states require explicit estimator/state-contract
extension, not pretending they are measured CVs. Do not train an inverse NN.