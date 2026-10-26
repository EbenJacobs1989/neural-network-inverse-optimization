"""Shared-objective global search and governed mismatched-plant workflows."""

import numpy as np
import pytest

from nn_optimization.benchmarking import run_method
from nn_optimization.mpc import ForecastProvider
from nn_optimization.simulation import simulate
from test_inverse_optimization import problem, solver_config
from test_model_contract import predictor


@pytest.mark.parametrize("method", ["PSO", "DE", "PSO+LBFGS"])
def test_global_methods_share_bounds_and_objective(problem, solver_config, method):
    config = dict(solver_config, pso_iterations=3, pso_particles=6, pso_inertia=.7, pso_cognitive=1.4, pso_social=1.4)
    result = run_method(method, problem, np.array([0.]), np.array([0.]), config)
    assert result.feasible
    assert abs(result.controls[0]) <= .2 + 1e-6
    assert result.objective == pytest.approx(sum(problem.evaluate(result.controls, [0.], [0.])[1].values()))


def test_mismatched_plant_with_forecast_and_target_steps(problem, solver_config):
    config = {"model_kind": "temporary_static_lag_wrapper", "alpha": [.2], "n_horizon": 2, "t_step": 1.,
        "ipopt_max_iterations": 100, "ipopt_max_cpu_time": 10., "ipopt_tolerance": 1e-8,
        "plant_alpha": [.1], "plant_gain": [1.1], "plant_bias": [.02], "measurement_noise": [0.],
        "unmeasured_disturbance": [0.], "plant_delay_steps": 1, "seed": 42, "measurement_max_age": 2.,
        "result_max_age": 1., "hold_last_good_seconds": 5., "max_recommendation_move_factor": 1.,
        "minimum_confidence": None, "acceptable_statuses": ["Solve_Succeeded", "Solved_To_Acceptable_Level"]}
    plant = lambda disturbances, controls: .3 + .2 * controls + .1 * disturbances
    forecast = ForecastProvider.step([0.], [.1], 1)
    target = ForecastProvider.step([.4], [.45], 1)
    frame = simulate(problem, config, solver_config, np.array([0.]), np.array([0.]), np.array([.3]), plant, 3,
                     disturbance_provider=forecast, target_provider=target)
    assert len(frame) == 3 and frame.feasible.all() and frame.accepted.all()
    assert frame.iloc[-1].target == [.45] and frame.iloc[-1].disturbance == [.1]
    assert frame.mv_rate.map(lambda value: abs(value[0]) <= .2 + 1e-6).all()