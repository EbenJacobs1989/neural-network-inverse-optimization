"""Small non-confidential fixtures for contract validation."""

from dataclasses import replace

import numpy as np
import pytest
import torch
from torch import nn

from nn_optimization.model import ForwardModel, ModelContract


@pytest.fixture
def predictor():
    metadata = {"inputs": [{"name": "feed", "role": "MV", "unit": "demo"},
                           {"name": "ore", "role": "DV", "unit": "demo"}],
                "outputs": [{"name": "quality", "role": "CV", "unit": "demo"}],
                "framework": "pytorch", "model_type": "static_current_output", "dtype": "float64", "device": "cpu"}
    contract = ModelContract(metadata, np.zeros(2), np.ones(2), np.zeros(1), np.ones(1),
                             -np.ones(2), np.ones(2), -np.ones(2), np.ones(2), -np.ones(1)*5, np.ones(1)*5)
    network = nn.Sequential(nn.Linear(2, 3), nn.SiLU(), nn.Linear(3, 1)).double()
    with torch.no_grad():
        for parameter in network.parameters():
            parameter.fill_(0.2)
    return ForwardModel(network, contract)


def test_contract_shape_order_and_scaler(predictor):
    contract = predictor.contract
    np.testing.assert_array_equal(contract.assemble([0.3], [-0.2]), [0.3, -0.2])
    assert predictor.predict([0, 0]).dtype == np.float64
    assert all(not parameter.requires_grad for parameter in predictor.network.parameters())
    with pytest.raises(ValueError, match="standard deviations"):
        replace(contract, input_scale=np.zeros(2))
    with pytest.raises(ValueError, match="shape"):
        replace(contract, input_mean=np.zeros(3))


@pytest.mark.parametrize("values", [[np.nan, 0], [np.inf, 0], [0], [2, 0]])
def test_invalid_inputs_rejected(predictor, values):
    with pytest.raises(ValueError):
        predictor.predict(values)


def test_missing_feature_rejected(predictor):
    metadata = dict(predictor.contract.metadata)
    metadata["inputs"] = [{"role": "MV", "unit": "demo"}]
    with pytest.raises(ValueError, match="feature names"):
        replace(predictor.contract, metadata=metadata)