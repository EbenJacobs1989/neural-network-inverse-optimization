# Optimisation Formulations

All values passed to the forward function, constraints, forecasts and results
are engineering-unit values in contract order. Costs are dimensionless research
proxies using training standard deviations. Configure engineering economics
and approved hard limits before any application outside synthetic simulation.

## Shared Objective

Let S_y and S_u denote diagonal training standard-deviation matrices, r the
engineering target, c_y and c_u configurable economic coefficients and u_p the
previous actually applied control:

$$
\begin{aligned}
e &= S_y^{-1}(\hat y-r), & \Delta v &= S_u^{-1}(u-u_p),\\
J &= e^TQe+c_y^TS_y^{-1}(\hat y-\mu_y)
 +c_u^TS_u^{-1}(u-\mu_u)+\Delta v^TR\Delta v\\
&\quad+\sum_i w_i[(s_i^-)^2+(s_i^+)^2]
 +\lambda\|S_u^{-1}(u-\mu_u)\|^2.
\end{aligned}
$$

Tracking, economics, movement, soft constraints and regularisation are reported
separately. Negative quality coefficient rewards quality; positive energy or
reagent coefficients discourage consumption. These example coefficients are
not metal prices or currency. `OptimizationProblem.components` is the single
expression implementation shared by NumPy, Torch and CasADi.

## Direct PyTorch Inverse Optimisation

Freeze all network parameters. Optimise unconstrained logits v only:

$$
\begin{aligned}
\ell &= \max(u_\min,u_p-\Delta u_\max,u_\text{envelope,min}),\\
h &= \min(u_\max,u_p+\Delta u_\max,u_\text{envelope,max}),\\
u(v)&=\ell+(h-\ell)\odot\operatorname{sigmoid}(v),\\
v^*&=\arg\min_v J(f_\theta(u(v),d),u(v)).
\end{aligned}
$$

Empty intersections fail clearly. Zero-width feasible components are allowed.
Starts include the current MV, an optional supplied warm start and seeded
uniform additional starts. Adam and LBFGS are supported. The best feasible
iterate is retained; gradient norm, convergence, budget termination, all start
outcomes and infeasibility are explicit.

For Torch evaluation, normalised slacks are exact ReLU violations. **No hard CV
guarantee is attributed to these penalties.** Predicted hard limits are checked
when selecting a candidate and again at the independent gate. The default gate
rejects `Iteration_Limit`, even when a bounded candidate improves the objective.

## CasADi/IPOPT Steady-State NLP

$$
\begin{aligned}
\min_{u,s^-,s^+}\quad &J(f_\theta(u,d),u,s^-,s^+)\\
\text{s.t.}\quad &\ell\le u\le h,\\
&y_\text{hard,min}\le f_\theta(u,d)\le y_\text{hard,max},\\
&(y_\text{soft,min}-f_\theta(u,d))\oslash\sigma_y\le s^-,\\
&(f_\theta(u,d)-y_\text{soft,max})\oslash\sigma_y\le s^+,\\
&s^-,s^+\ge0.
\end{aligned}
$$

Decision controls and constraints are numerically scaled. Slacks are in output
standard-deviation units; multiply by sigma_y for engineering excursions.
IPOPT has finite iteration/CPU budgets and disables bound relaxation. Failed
solver iterates are not released; a failed result contains a diagnostic start
point and failure status instead of pretending that point solved the NLP.
Barrier solutions can contain tiny positive interior slacks even at inactive
limits; the exact evaluated objective breakdown uses minimal violations.

## Dynamic NMPC

The temporary model is:

$$
y^\text{ss}_k=f_\theta(u_k,d_k),\qquad
y_{k+1}=y_k+\alpha\odot(y^\text{ss}_k-y_k),\qquad
m_{k+1}=u_k.
$$

The state is `(y,m)`, with m the previous applied MV. The controller solves:

$$
\min_{U,S}\sum_{k=0}^{N-1}\ell(y_k,u_k,m_k,r_k,S_k)
 +(S_y^{-1}(y_N-r_N))^TP(S_y^{-1}(y_N-r_N))
$$

subject to the dynamics, input support, MV bounds, hard CV bounds (including
the terminal prediction), nonnegative preferred-limit slacks and:

$$
-\Delta u_\max\le u_k-m_k\le\Delta u_\max.
$$

Q, P, R and quadratic soft penalties are configurable. do-mpc's automatic
quadratic rterm is disabled to avoid double-counting; the common movement cost
uses explicit memory. Quadratic slack terms are added to stage-indexed decision
variables between `prepare_nlp` and `create_nlp`. The horizon objective and
first-move diagnostic objective are distinct and both are labelled.

## Global Search and Fair Comparison

PSO uses the same model/cost and feasibility-first ranking, with hard MV/rate
bounds built into its search box. It does not promise nonlinear feasibility.
DE uses SciPy's nonlinear constraint interface for hard CV limits and is still
post-validated. PSO+LBFGS passes PSO controls as a warm start. All legacy notebook
algorithms remain unchanged. Benchmark runtime covers method construction and
all starts; NMPC per-step runtime excludes one-time controller construction.
Different algorithms have different budgets; small demonstration rankings do
not establish general superiority or a certified global optimum.