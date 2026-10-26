# AiPC Architecture

This is an installable research framework around the existing frozen forward
predictor, not an inverse network, plant controller or certified safety system.
The compact package separates responsibilities in working modules rather than
creating a mostly empty nested module tree.

## Steady-state inverse optimisation

```mermaid
flowchart LR
    A[Model contract and frozen checkpoint] --> B[Forward engineering-unit predictor]
    C[Disturbance and prior MV] --> D[Shared objective and constraints]
    B --> D
    D --> E[Bounded Torch input optimisation]
    D --> F[CasADi IPOPT with hard constraints]
    D --> G[PSO DE and hybrid exploration]
    E --> H[Structured candidate and diagnostics]
    F --> H
    G --> H
    H --> I[Independent validation gate]
    I --> J[Approved operating target for existing control]
```

The inverse layer answers **where to operate**. Its steady-state interpretation
is provisional because the existing static network was fitted to lagged noisy
outputs. All solvers use the same engineering predictor and objective module.
The legacy notebook objective remains available unchanged for compatibility.

## Dynamic NMPC

```mermaid
flowchart LR
    A[Measured CV and applied MV memory] --> B[Measured-state feedback]
    C[Aligned DV forecasts and target schedule] --> D[do-mpc discrete MX model]
    E[Frozen NN plus explicit lag wrapper] --> D
    B --> D
    D --> F[Horizon constrained IPOPT solve]
    F --> G[First candidate move and full horizon checks]
    G --> H[Independent governance]
    H --> I[Simulated plant or external execution owner]
    I --> A
```

NMPC answers **how to move**. The current states are measured CVs plus applied
MV memory. MV memory is updated from the actual approved move, not the solver's
previous suggestion. Forecasts supply N+1 rows. Native CasADi operations provide
exact model derivatives; no opaque callbacks or ONNX approximation are used.

## Combined Hybrid Architecture

```mermaid
flowchart TD
    A[Slow economic operating-point calculation] --> B[Steady-state candidate]
    B --> C[Operating-target governance]
    C --> D[CV target provider or existing APC target filter]
    D --> E[Fast NMPC trajectory planning]
    E --> F[Move governance]
    F --> G[External control or independent simulation]
    G --> H[Measured feedback and disturbances]
    H --> A
    H --> E
```

An integration owner can supply the accepted steady-state predicted CVs through
`ForecastProvider.constant(steady_state_result.outputs, t_step)`, or a schedule,
to the controller. A raw steady-state MV recommendation is **not** a validated
dynamic trajectory. The default demonstration uses a fixed CV target followed
by a target step; it does not claim an online plant economic coordinator.

## Runtime Governance

```mermaid
flowchart TD
    A[Solver candidate] --> B{Success status finite fresh supported and feasible?}
    B -->|yes| C[Approved recommendation]
    B -->|no| D{Valid constrained steady-state target?}
    D -->|yes| E[Target only through existing control filter]
    D -->|no| F{Approved last-good command still valid in time?}
    F -->|yes| G[Bounded hold last good]
    F -->|no| H[Return no command: external APC DCS or manual strategy]
    C --> I[External execution owner]
    E --> I
    G --> I
```

No module contains an OPC-UA client, DCS write, network upload, SIS bypass or
plant-write operation. `GateDecision.tier` distinguishes a target-only release
from an NMPC move. External control retains authority at all times.

## Ownership

| Module | Responsibility |
|---|---|
| model | YAML mapping, signal/scaler contract, trusted checkpoint, feature assembly |
| conversion | Native CasADi expressions and full prediction/Jacobian parity |
| optimization | Shared cost, bounds, feasibility, results, Torch/IPOPT multistart |
| mpc | Forecasts, temporary dynamics, state feedback, discrete do-mpc controller |
| governance | Independent finite/freshness/status/envelope/constraint gate and fallback |
| benchmarking | PSO/DE/hybrid adapters, repeated static metrics, closed-loop metrics |
| simulation | Original synthetic plant extraction and deliberately mismatched replay |
| reporting | Offline Matplotlib artifact generation only |
| cli | Root-independent commands and holdout scenario selection |

Library code does not reseed global RNGs or alter global Torch thread settings.
PSO, multistart and simulation use local seeded NumPy generators. Plotting's Agg
backend is imported only by the simulation reporting workflow.