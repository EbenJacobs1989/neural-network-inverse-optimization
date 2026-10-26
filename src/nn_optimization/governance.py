"""Independent advisory release gate. No plant communication or plant writes."""

from dataclasses import dataclass
from typing import Any

import numpy as np

from .model import vector
from .optimization import OptimizationProblem, OptimizationResult


@dataclass(frozen=True)
class EnvelopeReport:
    within: bool
    outside_features: list[str]
    normalised_distance: list[float]
    disposition: str


def check_envelope(values: Any, names: list[str], lower: np.ndarray, upper: np.ndarray,
                   *, margin: float = 0.0) -> EnvelopeReport:
    """Marginal distance divided by envelope width, not a joint-support guarantee."""
    values = vector(values, len(names), "envelope values")
    lower, upper = vector(lower, len(names), "envelope lower"), vector(upper, len(names), "envelope upper")
    if np.any(lower >= upper) or not np.isfinite(margin) or not 0 <= margin < .5:
        raise ValueError("Envelope bounds/margin invalid")
    width = upper - lower
    distance = np.maximum.reduce([(lower + margin * width - values) / width,
                                   (values - upper + margin * width) / width, np.zeros_like(values)])
    outside = [name for name, excursion in zip(names, distance) if excursion > 1e-12]
    return EnvelopeReport(not outside, outside, distance.tolist(), "reject" if outside else "eligible")


@dataclass
class GateDecision:
    accepted: bool
    final_command: np.ndarray | None
    rejection_reason: str | None
    original_candidate: np.ndarray
    last_known_good: np.ndarray | None
    details: dict[str, Any]
    tier: str


class ValidationGate:
    """Validate recommendations independently; held commands expire into external control."""

    def __init__(self, problem: OptimizationProblem, config: dict[str, Any]) -> None:
        self.problem, self.config = problem, config
        self.last_good: np.ndarray | None = None
        self.last_good_time: float | None = None
        for name in ("measurement_max_age", "result_max_age", "hold_last_good_seconds", "max_recommendation_move_factor"):
            if not np.isfinite(config[name]) or config[name] < 0:
                raise ValueError(f"{name} must be finite and nonnegative")

    def seed_approved(self, controls: Any, timestamp: float) -> None:
        """Record an externally approved initial command; never infer approval from a solver."""
        controls = vector(controls, len(self.problem.lower), "approved initial MVs")
        if not np.isfinite(timestamp) or np.any(controls < self.problem.lower) or np.any(controls > self.problem.upper):
            raise ValueError("Approved initial command or timestamp invalid")
        self.last_good, self.last_good_time = controls, timestamp

    def _check(self, result: OptimizationResult, previous: np.ndarray, disturbances: np.ndarray,
               measurement_time: float, result_time: float, now: float, confidence: float | None) -> tuple[list[str], dict[str, Any]]:
        problem, contract = self.problem, self.problem.predictor.contract
        reasons, details = [], {}
        if not np.isfinite([measurement_time, result_time, now]).all():
            reasons.append("invalid timestamps")
        else:
            if not 0 <= now - measurement_time <= self.config["measurement_max_age"]:
                reasons.append("measurement stale or future-dated")
            if not 0 <= now - result_time <= self.config["result_max_age"]:
                reasons.append("recommendation stale or future-dated")
        if not result.success or result.status not in self.config["acceptable_statuses"]:
            reasons.append("unacceptable solver termination")
        minimum = self.config.get("minimum_confidence")
        if minimum is not None and (confidence is None or not np.isfinite(confidence) or confidence < minimum):
            reasons.append("required model confidence unavailable or low")
        try:
            controls = vector(result.controls, len(problem.lower), "candidate MVs")
            outputs = vector(result.outputs, contract.n_output, "candidate outputs")
            previous = vector(previous, len(problem.lower), "applied prior MVs")
            features = contract.assemble(controls, disturbances)
            if (not np.isfinite([result.objective, result.runtime]).all()
                    or not all(np.isfinite(value) for value in result.components.values())
                    or result.gradient_norm is not None and not np.isfinite(result.gradient_norm)):
                reasons.append("non-finite objective or diagnostics")
            if not result.feasible or not result.envelope_ok or any(not np.isfinite(value) or value > problem.tolerance for value in result.violations.values()):
                reasons.append("candidate/horizon feasibility failed")
            if np.any(controls < problem.lower - problem.tolerance) or np.any(controls > problem.upper + problem.tolerance):
                reasons.append("MV bounds violated")
            if np.any(np.abs(controls - previous) > problem.max_move + problem.tolerance):
                reasons.append("MV rate limit violated")
            if np.any(outputs < problem.hard_lower - problem.tolerance) or np.any(outputs > problem.hard_upper + problem.tolerance):
                reasons.append("hard CV limit violated")
            if self.last_good is not None and np.any(np.abs(controls - self.last_good) > self.config["max_recommendation_move_factor"] * problem.max_move + problem.tolerance):
                reasons.append("change versus last accepted recommendation too large")
            input_report = check_envelope(features, contract.input_names, contract.train_min, contract.train_max, margin=problem.envelope_margin)
            output_report = check_envelope(outputs, contract.output_names, contract.output_min, contract.output_max)
            details.update(input_envelope=vars(input_report), output_envelope=vars(output_report))
            if not input_report.within or not output_report.within:
                reasons.append("model envelope violated")
            if "horizon_cv" in result.diagnostics:
                horizon_cv = np.asarray(result.diagnostics["horizon_cv"], dtype=float)
                horizon_mv = np.asarray(result.diagnostics["horizon_mv"], dtype=float)
                if (horizon_cv.ndim != 2 or horizon_mv.ndim != 2
                        or horizon_cv.shape != (len(horizon_mv) + 1, contract.n_output)
                        or horizon_mv.shape[1] != len(problem.lower)
                        or not np.isfinite(horizon_cv).all() or not np.isfinite(horizon_mv).all()):
                    reasons.append("invalid horizon diagnostics")
                elif (np.any(horizon_cv < problem.hard_lower - problem.tolerance)
                      or np.any(horizon_cv > problem.hard_upper + problem.tolerance)
                      or np.any(horizon_mv < problem.lower - problem.tolerance)
                      or np.any(horizon_mv > problem.upper + problem.tolerance)
                      or np.any(np.abs(np.diff(np.vstack([previous, horizon_mv]), axis=0)) > problem.max_move + problem.tolerance)):
                    reasons.append("independent horizon constraint check failed")
        except (ValueError, TypeError) as error:
            reasons.append(str(error))
        return reasons, details

    def approve(self, result: OptimizationResult, previous: np.ndarray, disturbances: np.ndarray, *,
                measurement_time: float, result_time: float, now: float,
                confidence: float | None = None,
                steady_state: OptimizationResult | None = None,
                steady_state_time: float | None = None) -> GateDecision:
        """A secondary steady-state result is a target for existing control, never a direct move."""
        reasons, details = self._check(result, previous, disturbances, measurement_time, result_time, now, confidence)
        original = np.asarray(result.controls).copy()
        last = None if self.last_good is None else self.last_good.copy()
        if not reasons:
            self.last_good, self.last_good_time = result.controls.copy(), now
            return GateDecision(True, result.controls.copy(), None, original, last, details, "optimizer_recommendation")
        if steady_state is not None and steady_state_time is not None:
            secondary_reasons, secondary_details = self._check(steady_state, previous, disturbances, measurement_time, steady_state_time, now, confidence)
            details["steady_state"] = secondary_details
            if not secondary_reasons:
                details["release_type"] = "target_only_requires_existing_control_filter"
                return GateDecision(False, steady_state.controls.copy(), "; ".join(reasons), original, last,
                                    details, "steady_state_target_for_existing_control")
        if (self.last_good is not None and self.last_good_time is not None and np.isfinite(now)
                and 0 <= now - self.last_good_time <= self.config["hold_last_good_seconds"]
                and np.all(np.abs(self.last_good - np.asarray(previous)) <= self.problem.max_move + self.problem.tolerance)):
            return GateDecision(False, self.last_good.copy(), "; ".join(reasons), original, last, details, "hold_last_good")
        return GateDecision(False, None, "; ".join(reasons), original, last, details, "external_APC_DCS_or_manual")