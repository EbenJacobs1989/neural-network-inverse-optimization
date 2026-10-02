# Full-dataset process-control optimization check

All 8,000 generated rows were checked: 5,600 training, 1,200 validation and 1,200 test.
Starting controls outside actuator bounds were projected; their adjustment is reported per row.
The full sweep uses a screening search budget (8 optimizer steps/generations and a 12-generation plant-reference DE search plus local polish), not the deeper 100-scenario budget.
True optima are **approximate noise-free steady-state references**, not proven global optima.

## Held-out test ranking (do not rank deployment decisions on training rows)

Equal-rank sum: true constraint satisfaction (higher), median and 90th percentile true regret (lower), 90th percentile model exploitation (lower), and solve time (lower). Ties: feasibility, median regret, then time.

| Rank | Method | Feasible | Median true regret | P90 true regret | P90 exploitation | Median seconds |
|---:|---|---:|---:|---:|---:|---:|
| 1 | PSO | 67.4% | 0.1559 | 1.7784 | 0.4302 | 0.0028 |
| 2 | DE | 67.2% | 0.1647 | 1.5591 | 0.2703 | 0.0055 |
| 3 | Adam | 59.7% | 0.3117 | 1.7094 | 0.2570 | 0.0094 |
| 4 | PSO+LBFGS | 66.7% | 0.1578 | 1.8124 | 0.4097 | 0.0130 |
| 5 | LBFGS | 67.1% | 0.1556 | 1.8538 | 0.4543 | 0.0199 |

## Quality checks

The complete printed quality, violation, summary and original-plant reference matrices are in optimizer_full_dataset_quality_printout.txt; separate CSVs hold all rows for analysis.
Positive exploitation means predicted gains exceeded actual steady-state gains. Generalization errors compare frozen-network outputs with the generating function, not the lagged and noisy historical measurements.
Constraints include true quality/energy thresholds, disturbance and control training-support limits, normalized actuator bounds and the 0.05 move limit. Fixed out-of-support disturbances may make scenarios infeasible regardless of the optimizer.
This is a static screening result; validate the ranking in the lagged closed-loop simulation before any process-control use. Training-row results are in-sample diagnostics.
