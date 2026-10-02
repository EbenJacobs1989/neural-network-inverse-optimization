# Inverse optimization: process-control comparison

Rank 1: **DE** under the five equally ranked criteria below.
These scores are research diagnostics, **not a deployment recommendation**.

Numerical oracle output feasibility: 87.0%; the remaining cases also have output violations at the minimum-penalty reference. This is not proof that no feasible point exists.
The top method is truly feasible in 70.0% and improves the original plant objective in 71.0% of scenarios. No method should be used for control without independent safety and closed-loop validation.

## Ranking rule
Rank 1 is best for true-plant feasibility rate, median and 90th-percentile true regret, median absolute model exploitation, and median solve time. Sum the five ranks; break ties by feasibility, regret, then time. Lower regret and error are better.

## Results (100 paired held-out scenarios)

| Rank | Method | Feasible | True improvement | Median true regret | P90 regret | Median optimization error | Median exploitation | P90 exploitation | Median optimized error | Median seconds |
|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | DE | 70.0% | 71.0% | 0.2197 | 1.7664 | 0.0000 | 0.0071 | 0.7491 | 0.2603 | 0.0678 |
| 2 | Adam | 71.0% | 73.0% | 0.2306 | 1.6634 | 0.0152 | 0.0082 | 0.5772 | 0.2627 | 0.1715 |
| 3 | PSO | 70.0% | 71.0% | 0.2294 | 1.7675 | 0.0000 | 0.0083 | 0.7531 | 0.2608 | 0.0328 |
| 4 | LBFGS | 70.0% | 71.0% | 0.2227 | 1.7677 | 0.0000 | 0.0080 | 0.7532 | 0.2607 | 0.8922 |
| 5 | PSO+LBFGS | 70.0% | 71.0% | 0.2359 | 1.7684 | 0.0000 | 0.0237 | 0.7531 | 0.2633 | 0.0808 |

## Interpretation
- The oracle is the best of two seeded DE runs, bounded local polishing, and the starting point on the original noise-free steady-state plant; global optimality is not proven.
- Optimization error compares predicted losses to the best found by the five methods, not a certified neural global optimum.
- Exploitation = neural predicted gain minus actual steady-state gain under matched penalties; positive means the model overstated improvement.
- Generalization error is mean absolute output prediction error scaled by training output standard deviations; the optimized error captures off-policy shift.
- True feasibility includes actual output thresholds, support, actuator and rate limits. Output limits are soft in the objective; feasibility cannot be assumed.
- Solve times exclude oracle computation and depend on the machine. Steps are not equal-cost: evaluate the number of model calls and tail regret as well.
- The true oracle and comparisons are static steady-state counterfactuals. Confirm controller performance using the lagged closed-loop plant in notebook 06 before control use.

See optimizer_control_runs.csv, optimizer_true_optima.csv, optimizer_control_summary.csv and optimizer_control_convergence.csv for full data.
