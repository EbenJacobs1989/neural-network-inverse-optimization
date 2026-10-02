"""Tests for the differentiable inverse-optimization objective."""

import sys
import unittest
from pathlib import Path

import torch
from torch import nn

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from objective_functions import InverseProcessObjective


class InverseProcessObjectiveTests(unittest.TestCase):
    def setUp(self):
        self.model = nn.Sequential(nn.Linear(10, 2, bias=False))
        with torch.no_grad():
            self.model[0].weight.zero_()
            self.model[0].weight[0, 5] = 1.0
            self.model[0].weight[1, 6] = 1.0
        self.kwargs = dict(
            input_mean=torch.zeros(10),
            input_scale=torch.ones(10),
            output_mean=torch.zeros(2),
            output_scale=torch.ones(2),
            control_lower=-torch.ones(5),
            control_upper=torch.ones(5),
            support_lower=-2 * torch.ones(10),
            support_upper=2 * torch.ones(10),
        )
        self.disturbances = torch.zeros(5)
        self.previous = torch.zeros(5)

    def test_loss_is_scalar_and_gradient_reaches_only_controls(self):
        objective = InverseProcessObjective(self.model, **self.kwargs)
        disturbances = self.disturbances.clone().requires_grad_()
        previous = self.previous.clone().requires_grad_()
        controls = torch.zeros(5, requires_grad=True)
        loss = objective(disturbances, controls, previous)
        self.assertEqual(loss.shape, ())
        loss.backward()
        torch.testing.assert_close(controls.grad, torch.tensor([-1., 1., 0., 0., 0.]))
        self.assertIsNone(disturbances.grad)
        self.assertIsNone(previous.grad)
        self.assertTrue(all(not parameter.requires_grad for parameter in self.model.parameters()))

    def test_move_and_constraint_penalties(self):
        objective = InverseProcessObjective(
            self.model, **self.kwargs,
            quality_weight=0, energy_weight=0, movement_weight=2,
            constraint_weight=3, support_weight=0,
        )
        controls = torch.tensor([2., 0., 0., 0., 0.], requires_grad=True)
        loss = objective(self.disturbances, controls, self.previous)
        self.assertAlmostEqual(loss.item(), 11.)  # 2 * 2^2 + 3 * (2 - 1)^2
        loss.backward()
        self.assertGreater(controls.grad[0].item(), 0)

    def test_output_and_training_support_penalties(self):
        objective = InverseProcessObjective(
            self.model, **self.kwargs,
            quality_min=1, energy_max=-1,
            quality_weight=0, energy_weight=0, movement_weight=0,
            constraint_weight=3, support_weight=2,
        )
        controls = torch.zeros(5, requires_grad=True)
        self.assertAlmostEqual(
            objective(self.disturbances, controls, self.previous).item(), 6.
        )
        controls = torch.tensor([0., 0., 0., 0., 3.], requires_grad=True)
        loss = objective(self.disturbances, controls, self.previous)
        self.assertAlmostEqual(loss.item(), 20.)  # 6 output + 12 bound + 2 support
        loss.backward()
        self.assertGreater(controls.grad[4].item(), 0)

    def test_validates_bounds_and_input_shape(self):
        with self.assertRaisesRegex(ValueError, "Control lower bounds"):
            InverseProcessObjective(
                self.model, **{**self.kwargs, "control_upper": -torch.ones(5)}
            )
        with self.assertRaisesRegex(ValueError, "weights"):
            InverseProcessObjective(self.model, **self.kwargs, support_weight=-1)
        objective = InverseProcessObjective(self.model, **self.kwargs)
        with self.assertRaisesRegex(ValueError, "controls"):
            objective(self.disturbances, torch.zeros(4), self.previous)


if __name__ == "__main__":
    unittest.main()
