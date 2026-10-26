"""Prediction and derivative parity, including nontrivial scaling."""

from dataclasses import replace

import numpy as np
import pytest
from torch import nn

from nn_optimization.conversion import convert_model, equivalence_report
from nn_optimization.model import ForwardModel
from test_model_contract import predictor


def test_predictions_and_jacobians(predictor):
    contract = replace(predictor.contract, input_mean=np.array([0.3, -0.2]), input_scale=np.array([2., 3.]),
                       output_mean=np.array([4.]), output_scale=np.array([2.]))
    model = ForwardModel(predictor.network, contract)
    samples = np.vstack([[-1, -1], [0, 0], [1, 1], np.random.default_rng(4).uniform(-1, 1, (20, 2))])
    report = equivalence_report(model, convert_model(model), samples)
    assert report["accepted"], report


@pytest.mark.parametrize("activation", [nn.Tanh(), nn.Sigmoid(), nn.Softplus(), nn.ReLU(), nn.Identity()])
def test_supported_activations(predictor, activation):
    model = ForwardModel(nn.Sequential(nn.Linear(2, 2), activation, nn.Linear(2, 1)), predictor.contract)
    assert equivalence_report(model, convert_model(model), np.array([[.4, .2], [-.5, .7]]))["accepted"]


def test_unsupported_operation_fails(predictor):
    predictor.network = nn.Sequential(nn.Linear(2, 1), nn.Dropout())
    with pytest.raises(ValueError, match="network.1: Dropout"):
        convert_model(predictor)