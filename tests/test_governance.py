"""Independent approval, rejection and bounded fallback."""

from dataclasses import replace
from time import perf_counter

import numpy as np
import pytest

from nn_optimization.governance import ValidationGate
from test_inverse_optimization import problem
from test_model_contract import predictor


@pytest.fixture
def gate(problem):
    config = {"measurement_max_age": 2., "result_max_age": 1., "hold_last_good_seconds": 5.,
              "max_recommendation_move_factor": 1., "minimum_confidence": None, "acceptable_statuses": ["Solve_Succeeded"]}
    gate = ValidationGate(problem, config)
    gate.seed_approved([0.], 0.)
    return gate


def candidate(problem):
    return problem.result([.1], [0.], [0.], iterations=1, status="Solve_Succeeded", success=True,
                          converged=True, started=perf_counter(), initialization=[0.])


def test_accepts_valid_command(gate, problem):
    result = gate.approve(candidate(problem), np.array([0.]), np.array([0.]), measurement_time=0., result_time=0., now=0.)
    assert result.accepted


@pytest.mark.parametrize("changes", [{"controls": np.array([np.nan])}, {"outputs": np.array([np.inf])},
    {"success": False}, {"controls": np.array([.3])}, {"controls": np.array([2.])}])
def test_rejections_hold_last_good(gate, problem, changes):
    decision = gate.approve(replace(candidate(problem), **changes), np.array([0.]), np.array([0.]),
                            measurement_time=0., result_time=0., now=0.)
    assert not decision.accepted and decision.rejection_reason
    assert decision.tier == "hold_last_good"
    np.testing.assert_array_equal(decision.final_command, [0.])


def test_stale_result_and_expired_hold(gate, problem):
    decision = gate.approve(candidate(problem), np.array([0.]), np.array([0.]), measurement_time=0., result_time=0., now=10.)
    assert not decision.accepted and decision.final_command is None
    assert decision.tier == "external_APC_DCS_or_manual"


def test_output_envelope_and_confidence(gate, problem):
    outside = replace(candidate(problem), outputs=np.array([6.]))
    decision = gate.approve(outside, np.array([0.]), np.array([0.]), measurement_time=0., result_time=0., now=0.)
    assert not decision.accepted and "envelope" in decision.rejection_reason
    gate.config["minimum_confidence"] = .9
    decision = gate.approve(candidate(problem), np.array([0.]), np.array([0.]), measurement_time=0., result_time=0., now=0.)
    assert not decision.accepted and "confidence" in decision.rejection_reason


def test_gate_does_not_trust_false_horizon_feasibility(gate, problem):
    result = candidate(problem)
    result.diagnostics.update(horizon_cv=[[.3], [.3]], horizon_mv=[[.9]])
    decision = gate.approve(result, np.array([0.]), np.array([0.]), measurement_time=0., result_time=0., now=0.)
    assert not decision.accepted and "horizon" in decision.rejection_reason