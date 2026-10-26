# Model Contract

The single source of signal metadata is `configs/model.yaml`; fitted numbers
come only from the trusted saved scalers through `model.load_model`. Scripts do
not maintain their own feature-name or scaler-array copies.

## Verified Current Model

- Name/version: `synthetic_process_forward`, `legacy-checkpoint-epoch-20`.
- Checkpoint: `models/process_model.pt`, PyTorch, Linear 10-32-32-16-2 with
  SiLU hidden activations. Saved parameters are float32; notebook inference is
  float32. New numerical work promotes the exact saved weights to float64 on
  CPU, without training or rounding them again.
- Input roles/order: five DVs, then five MVs, in the exact YAML/checkpoint order.
- Output order: quality CV, energy KPI. Both can be tracked and constrained.
- No measured-state, lagged-input or context signals exist in this network.
- Physical unit symbols are not certified by the checkpoint. The YAML labels
  them `synthetic_generator_unit`, not a guessed plant unit.
- Sampling interval is one **synthetic sample**, not a certified number of
  seconds. The demonstration uses `t_step=1` in that same clock.
- Missing/non-finite values: reject. Filtering: none. Alignment: inputs and
  outputs from the same row. Real analyser delays are unknown.
- Confidence: unavailable; no confidence score is fabricated.

## Classification

**Static current-output model**: `y[k] = f(u[k], d[k])`. Training responses are
lagged, noisy original-plant measurements. The absence of history makes this
neither a true identified steady-state model nor a transition model. Repeated
evaluations cannot represent process state, accumulated energy or delay.

## Scaling Equations

For engineering input vector z and network output g:

$$
z_s=(z-\mu_z)\oslash\sigma_z,\qquad
\hat y=g_\theta(z_s)\odot\sigma_y+\mu_y.
$$

Saved StandardScalers have 5,600 fitted rows. The loader checks feature names,
sample count, means **and population standard deviations** against the first
70% of the supplied synthetic CSV. Validation and test rows never fit scalers.
Feature-order mismatches, zero/negative scales, wrong dimensions, non-finite
values, wrong checkpoint architecture/dtype and unsupported conversions fail.
Only trusted local pickle artifacts may be loaded.

## Envelope and Commands

Train minima/maxima and validation minima/maxima are separately retained in the
contract. Output bounds come from training output extrema. A configured inward
safety margin can shrink input support. Public engineering-unit inference
rejects out-of-training-envelope points unless an explicit advisory override is
provided. Tensor inference is an internal differentiable operation; optimisers
constrain its inputs and conversion diagnostics deliberately sample boundaries.

The legacy physical-command map remains:

$$
u_\text{physical}=q_{.01,\text{train}}+
 (q_{.99,\text{train}}-q_{.01,\text{train}})\odot u_\text{command}.
$$

MV command bounds and rate limits are external YAML values. This transformation
is distinct from StandardScaler standardisation. It is never applied to DVs.

Envelope reports contain names, per-feature excursions normalised by training
range and rejection disposition. They are marginal checks, **not joint-density
or calibrated uncertainty estimates**. Validation extrema are descriptive, not
automatically an extension of the approved training envelope.

## Conversion Acceptance

Supported native operations: Sequential, Linear, SiLU, Tanh, Sigmoid, Softplus,
ReLU and Identity. Unknown operations report the precise layer path. ReLU and
Softplus threshold derivatives have their normal nonsmooth caveats; avoid exact
kinks when comparing classical derivatives. ONNX is not required or enabled.

The real 128-point acceptance set includes nominal, train/validation extrema,
chronological historical samples and seeded random interior points. Extrema
vectors are synthetic boundary combinations, not evidence of observed joint
support. The output report saves failing sample values and all tolerances.

Measured float64 maximum errors: prediction `1.4210854715202004e-14`, full
engineering-unit Jacobian `3.552713678800501e-15`. Strict limits are `1e-9` and
`1e-8`. Native float32 errors are `9.796434980557933e-6` and
`3.3459321659989882e-6`; `2e-5` limits accommodate observed input subtraction,
activation and output rounding. These are numerical conversion results, not
plant-model accuracy or closed-loop guarantees.