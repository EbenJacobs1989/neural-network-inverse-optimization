"""Small deterministic shared-objective and constrained-solver tests."""

from dataclasses import replace

import casadi as ca
import numpy as np
import pytest

from nn_optimization.conversion import convert_model
from nn_optimization.optimization import OptimizationProblem, solve_casadi, solve_torch
from test_model_contract import predictor


@pytest.fixture
def problem(predictor):
    return OptimizationProblem(predictor, np.array([-.9]), np.array([.9]), np.array([.2]),
        np.array([-5.]), np.array([5.]), np.array([-.5]), np.array([.5]), np.array([.4]),
        np.ones(1), np.ones(1)*2, np.ones(1)*.1, -np.ones(1), np.zeros(1), np.ones(1)*10)


@pytest.fixture
def solver_config():
    return {"multistart": 2, "seed": 42, "torch_method": "Adam", "torch_learning_rate": .1,
            "torch_iterations": 30, "gradient_tolerance": 1e-6, "ipopt_max_iterations": 100,
            "ipopt_max_cpu_time": 10, "ipopt_tolerance": 1e-9}


def test_objective_all_backends_agree(problem):
    model, contract = problem.predictor, problem.predictor.contract
    controls, previous, disturbance = np.array([.1]), np.array([0.]), np.array([0.])
    features = contract.assemble(controls, disturbance)
    outputs = model.predict(features)
    expected = sum(problem.components(outputs, controls, previous).values())
    torch_cost = sum(problem.components(model.tensor(outputs), model.tensor(controls), model.tensor(previous), backend="torch").values())
    symbolic_cost = sum(problem.components(ca.DM(outputs), ca.DM(controls), ca.DM(previous), backend="casadi").values())
    assert float(torch_cost) == pytest.approx(expected, abs=1e-12)
    assert float(symbolic_cost) == pytest.approx(expected, abs=1e-12)


@pytest.mark.parametrize("solver", [solve_torch, solve_casadi])
def test_inverse_bounds_improvement_reproducibility(problem, solver_config, solver):
    previous, disturbance = np.array([0.]), np.array([0.])
    first = solver(problem, disturbance, previous, solver_config)
    second = solver(problem, disturbance, previous, solver_config)
    assert first.feasible and first.success
    assert abs(first.controls[0]) <= .2 + problem.tolerance
    assert first.objective <= sum(problem.evaluate(previous, disturbance, previous)[1].values()) + 1e-8
    np.testing.assert_allclose(first.controls, second.controls, atol=1e-10)
    assert first.iterations > 0 and first.runtime > 0 and first.initialization
    assert len(first.diagnostics["starts"]) == 2


def test_infeasible_intersection(problem):
    with pytest.raises(ValueError, match="Infeasible intersection"):
        problem.effective_bounds(np.array([3.]), np.array([0.]))


def test_hard_limits_and_soft_penalties(problem, solver_config):
    baseline = problem.predictor.predict([0., 0.])[0]
    limited = replace(problem, hard_upper=np.array([baseline + .001]), soft_upper=np.array([baseline - .05]))
    result = solve_casadi(limited, np.array([0.]), np.array([0.]), solver_config)
    assert result.feasible
    assert result.outputs[0] <= limited.hard_upper[0] + limited.tolerance
    assert result.components["soft_constraint"] > 0
    assert np.max(result.diagnostics["slacks_normalised"]) > 0
    assert problem.components(np.array([0.]), np.array([0.]), np.array([0.]))["soft_constraint"] == 0


def test_infeasible_hard_cv_is_not_success(problem, solver_config):
    limited = replace(problem, hard_lower=np.array([4.]), hard_upper=np.array([5.]))
    result = solve_casadi(limited, np.array([0.]), np.array([0.]), solver_config)
    assert not result.success and not result.feasible
    assert result.diagnostics["no_feasible_solution"]


@pytest.mark.parametrize("field,value", [("multistart", 0), ("torch_iterations", -1), ("seed", -1),
                                         ("torch_learning_rate", float("nan")), ("ipopt_max_cpu_time", 0)])
def test_invalid_settings_fail_clearly(problem, solver_config, field, value):
    with pytest.raises(ValueError, match=field):
        solve_torch(problem, np.array([0.]), np.array([0.]), dict(solver_config, **{field: value}))