# Migration Guide

## Compatibility

No existing notebooks, source APIs, model/scaler artifacts or result files are
deleted or replaced. The pre-existing notebook 04, optimizer adapter and its
test were already dirty and remain untouched by the modular extension.
Notebook execution still uses the original working-directory convention.
There is no inverse-network training, checkpoint rewrite or retraining step.

| Existing workflow | Modular path |
|---|---|
| Notebook 01 plant/data generation | Retained; simulation reuses its exact plant function without rerunning cells |
| Notebook 02 model/scaler training | Retained; `model.load_model` loads and checks its saved artifacts |
| Notebook 03 validation/sensitivities | Retained; conversion script additionally checks symbolic derivatives |
| Notebook 04 bounded Adam inverse step | Retained; new inverse command adds shared-cost Torch/IPOPT multistart |
| Notebook 05 PSO/DE/LBFGS comparisons | Retained; new benchmark adapters apply one shared objective and contract |
| Notebook 06 original-plant replay | Retained; new simulation adds dynamic NMPC, forecasts and release governance |
| `constraints.ControlProjection` and sigmoid classes | Public APIs unchanged; new solvers generalise physical-unit bounds by contract |
| `objective_functions.InverseProcessObjective` | Exact legacy loss preserved; new configurable tracking/economics are opt-in |
| `optimizers.casadi_objective`, `optimize_do_mpc` | Preserved static horizon-one adapter; not renamed or advertised as dynamic NMPC |

The new demonstration adds tracking to the legacy economic proxy. Its numeric
objectives are therefore **not** directly comparable with old CSV loss values.
To reproduce a legacy economic-only research objective in the new formulation,
set tracking weights to zero and match economic, movement and preferred-limit
weights and limits explicitly. Hard-envelope rejection in the new framework is
intentionally stricter than the old support penalty.

## Environment

No project-local environment existed during inspection. VS Code selected a
user-local Python 3.14 interpreter. do-mpc 5.1.2 and CasADi were already present;
only PyYAML/pytest were installed, and Torch was not upgraded. Reuse a suitable
existing environment or create **one** project environment if needed:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -e ".[test,notebooks]"
```

Do not create another environment if `.venv` already exists. Select the same
interpreter in VS Code. Existing trusted checkpoint/scaler/data files are local
inputs, not test fixtures or package payloads. Install commands above are for
new setup; do not blindly reinstall/upgrade an existing production Torch stack.

## Root Commands

```powershell
python scripts/validate_conversion.py
python scripts/run_inverse_optimization.py
python scripts/run_inverse_optimization.py --method IPOPT
python scripts/run_inverse_optimization.py --method PSO+LBFGS
python scripts/run_mpc_simulation.py --steps 12
python scripts/benchmark_optimizers.py --repeats 2 --steps 8
python -m pytest -q
python -m unittest discover -s tests -p "test_*.py" -v
```

`unittest` discovers retained class-based tests; use pytest for the complete
suite including the new function/fixture tests. Scripts resolve their root
through pathlib and support OneDrive paths. Inverse/simulation/benchmark scripts
also accept `--root`, `--model-config`, `--inverse-config` and `--mpc-config`.
Defaults live in three YAML files; `example.local.yaml` is a non-confidential
integration template, not an automatically merged override or live mode switch.

## Artifacts and Governance

New generated files go to ignored `results/aipc/`. They include conversion JSON,
inverse JSON, simulation CSV/JSON/PNG, static run/summary CSV and JSON, and
closed-loop policy CSV plus summary CSV/JSON. Existing result files are untouched.
Use `OptimizationResult` for a **candidate**, and `ValidationGate.approve` for
an independent decision. A target-only fallback still needs existing control's
filter; it must not be treated as an NMPC move or a plant-write instruction.

The highest-value migration after software validation is identifying a dynamic
forward model or time constants on properly aligned temporal holdout data.