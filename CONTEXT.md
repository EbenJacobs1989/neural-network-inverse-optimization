# Project Context

## Purpose and Scope

This project provides research workflows for a frozen process-output predictor, inverse optimisation, and advisory nonlinear model-predictive control (NMPC). The examples use synthetic process data. The project does not control a plant or provide a certified safety system.

## Domain Definitions

| Term | Definition |
| --- | --- |
| Manipulated variable (MV) | A process command or setpoint selected by an optimisation method. |
| Disturbance variable (DV) | An input that affects predicted process outputs but is not selected by the controller. |
| Controlled variable (CV) | A process output measured or predicted against a target or limit. |
| Static predictor | The frozen forward neural network that maps current MVs and DVs to outputs. It does not identify process dynamics. |
| Dynamic NMPC | A receding-horizon controller that uses measured CVs, applied-MV memory, and aligned forecasts. |
| Governance gate | An independent check that can approve or reject a solver candidate before a simulated or external execution owner acts. |

## Domain Rules and Relationships

- The model contract owns signal names, order, units, scaler values, and input/output roles.
- The dynamic demonstration wraps the static predictor in a temporary first-order CV model. Its alpha values are not identified plant dynamics.
- Forecast providers return N+1 rows for an N-step horizon. MV memory records the command that was actually approved or applied.
- Solver success does not establish governance approval. Hard constraints and preferred soft limits remain distinct.
- Simulation results are synthetic research evidence, not plant validation or deployment approval.

## Architecture

The `src/nn_optimization` package owns model loading, symbolic conversion, shared optimisation, do-mpc construction, governance, simulation, reporting, and workflow orchestration. YAML files under `configs/` define model, optimisation, and MPC settings. Scripts expose supported command-line workflows, and notebooks provide analysis and demonstrations. See [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) and [docs/MPC_IMPLEMENTATION.md](docs/MPC_IMPLEMENTATION.md) for detail.

## Operating Constraints

- Python 3.10 or later is required.
- The dynamic MPC implementation pins do-mpc 5.1.2 and uses CasADi/IPOPT.
- Model checkpoints and scaler artifacts are local project inputs. The loader validates them against the configured model contract.
- The examples include no plant-write interface. External control systems retain execution authority.

## References

- [README.md](README.md)
- [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md)
- [docs/MODEL_CONTRACT.md](docs/MODEL_CONTRACT.md)
- [docs/MPC_IMPLEMENTATION.md](docs/MPC_IMPLEMENTATION.md)
- [docs/VALIDATION_PLAN.md](docs/VALIDATION_PLAN.md)