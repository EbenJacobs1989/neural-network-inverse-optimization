"""Actual do-mpc construction, forecast validation and receding-horizon solve."""

import numpy as np
import pytest

from nn_optimization.mpc import ForecastProvider, MPCController, build_dynamic_model
from test_inverse_optimization import problem
from test_model_contract import predictor


def test_forecast_validation():
    provider = ForecastProvider.constant([.2, .3])
    assert provider.horizon(3, 4).shape == (5, 2)
    with pytest.raises(ValueError, match="shape"):
        ForecastProvider(np.zeros((2, 2)), 2).horizon(0, 4)
    values = ForecastProvider.step([0.], [1.], 2).horizon(0, 3)
    np.testing.assert_array_equal(values.ravel(), [0., 0., 1., 1.])


def test_mpc_smoke_step(problem):
    config = {"model_kind": "temporary_static_lag_wrapper", "alpha": [.2], "n_horizon": 3, "t_step": 1.,
              "ipopt_max_iterations": 100, "ipopt_max_cpu_time": 10., "ipopt_tolerance": 1e-8}
    controller = MPCController(problem, config, ForecastProvider.constant([0.]), ForecastProvider.constant(problem.target))
    state = problem.predictor.predict([0., 0.])
    result = controller.step(state, np.array([0.]))
    assert result.success and result.feasible
    assert abs(result.controls[0]) <= .2 + 1e-6
    second = controller.step(result.outputs, result.controls)
    assert second.success and second.feasible