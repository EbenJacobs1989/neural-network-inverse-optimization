"""Tests for normalized control constraints."""

import sys
import unittest
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from constraints import ControlProjection, SigmoidControlParameterization, SoftplusConstraintPenalty


class ControlConstraintTests(unittest.TestCase):
    def setUp(self):
        self.previous = torch.tensor([0.5, 0.5, 0.5, 0.5, 0.5])

    def assert_feasible(self, values, previous, constraints):
        trajectory = values.unsqueeze(0) if values.ndim == 1 else values
        self.assertTrue(torch.all(trajectory >= constraints.lower))
        self.assertTrue(torch.all(trajectory <= constraints.upper))
        moves = trajectory - torch.cat((previous.unsqueeze(0), trajectory[:-1]))
        self.assertTrue(torch.all(moves.abs() <= constraints.max_move + 1e-6))

    def test_projection_exactly_respects_bounds_and_sequential_moves(self):
        constraints = ControlProjection()
        requested = torch.tensor([[2., -2., 3., -4., 2.]] * 3, requires_grad=True)
        actual = constraints(requested, self.previous)
        expected = self.previous + torch.tensor([0.05, -0.05, 0.05, -0.05, 0.05])
        torch.testing.assert_close(actual[-1], expected + 0.1 * torch.tensor([1., -1., 1., -1., 1.]))
        self.assert_feasible(actual, self.previous, constraints)
        actual.sum().backward()
        self.assertIsNotNone(requested.grad)

    def test_projection_clamps_to_actuator_bound(self):
        constraints = ControlProjection()
        previous = torch.tensor([0.98, 0.12, 0.98, 0.21, 0.89])
        actual = constraints(torch.tensor([5., -5., 5., -5., 5.]), previous)
        torch.testing.assert_close(actual, torch.tensor([1., 0.1, 1., 0.2, 0.9]))

    def test_softplus_penalty_is_smooth_and_grows_with_violations(self):
        penalty = SoftplusConstraintPenalty()
        feasible = self.previous.clone().requires_grad_()
        violated = (self.previous + 0.3).detach().requires_grad_()
        feasible_loss = penalty(feasible, self.previous)
        violated_loss = penalty(violated, self.previous)
        self.assertEqual(feasible_loss.shape, ())
        self.assertGreater(violated_loss.item(), feasible_loss.item())
        violated_loss.backward()
        self.assertTrue(torch.isfinite(violated.grad).all())
        self.assertTrue(torch.all(violated.grad > 0))

    def test_mpc_penalty_includes_between_step_changes(self):
        penalty = SoftplusConstraintPenalty()
        steady = self.previous.expand(3, -1).clone()
        alternating = torch.stack((self.previous, self.previous + 0.3, self.previous))
        self.assertGreater(penalty(alternating, self.previous), penalty(steady, self.previous))

    def test_sigmoid_feasible_and_differentiable_across_horizon(self):
        constraints = SigmoidControlParameterization()
        logits = torch.zeros(4, 5, requires_grad=True)
        controls = constraints(logits, self.previous)
        self.assert_feasible(controls, self.previous, constraints)
        self.assertEqual(controls.shape, logits.shape)
        controls[-1].sum().backward()
        self.assertTrue(torch.all(logits.grad.abs() > 0))

    def test_batch_horizons_and_invalid_previous(self):
        previous = self.previous.expand(2, -1).clone()
        logits = torch.zeros(2, 3, 5)
        controls = SigmoidControlParameterization()(logits, previous)
        self.assertEqual(controls.shape, logits.shape)
        self.assertTrue(torch.all((controls[:, 0] - previous).abs() <= 0.05 + 1e-6))
        with self.assertRaisesRegex(ValueError, "Previous controls"):
            ControlProjection()(torch.zeros(5), torch.ones(5) * 2)
        with self.assertRaisesRegex(ValueError, "at least one step"):
            ControlProjection()(torch.empty(0, 5), self.previous)


if __name__ == "__main__":
    unittest.main()
