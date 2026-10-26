"""Parity and feasibility checks for the optional do-mpc optimizer."""

import importlib.util
import sys
import unittest
from pathlib import Path

import numpy as np
import torch
from torch import nn

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from objective_functions import InverseProcessObjective
from optimizers import casadi_objective, optimize_do_mpc


@unittest.skipUnless(importlib.util.find_spec("casadi"), "Requires optional CasADi")
class DoMPCOptimizerTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(42)
        self.model = nn.Sequential(nn.Linear(10, 8), nn.SiLU(), nn.Linear(8, 2)).double()
        self.objective = InverseProcessObjective(
            self.model, torch.arange(10).double() / 10, torch.linspace(1, 2, 10).double(),
            torch.tensor([3., 4.]).double(), torch.tensor([2., 3.]).double(),
            -torch.ones(5).double(), torch.ones(5).double(),
            -torch.ones(10).double(), torch.ones(10).double(),
            quality_min=4., energy_max=2.,
        )

    def test_symbolic_predictions_loss_and_gradient_match_torch(self):
        import casadi as ca

        function = casadi_objective(self.objective)
        disturbances = torch.linspace(-2, 2, 5).double()
        previous = torch.zeros(5).double()
        for values in ([0., 0., 0., 0., 0.], [-2., -.3, .2, .4, 2.]):
            with self.subTest(controls=values):
                controls = torch.tensor(values, dtype=torch.float64, requires_grad=True)
                loss = self.objective(disturbances, controls, previous)
                loss.backward()
                symbolic_loss, outputs = function(disturbances.numpy(), controls.detach().numpy(), previous.numpy())
                self.assertAlmostEqual(float(symbolic_loss), loss.item(), places=10)
                inputs = torch.cat((disturbances, controls))
                expected = self.model((inputs - self.objective.input_mean) / self.objective.input_scale)
                expected = expected * self.objective.output_scale + self.objective.output_mean
                np.testing.assert_allclose(np.asarray(outputs).ravel(), expected.detach().numpy(), atol=1e-10)
                symbol = ca.SX.sym("candidate", 5)
                gradient = ca.Function("gradient", [symbol], [ca.gradient(function(disturbances.numpy(), symbol, previous.numpy())[0], symbol)])
                np.testing.assert_allclose(np.asarray(gradient(controls.detach().numpy())).ravel(), controls.grad.numpy(), atol=1e-10)

    @unittest.skipUnless(importlib.util.find_spec("do_mpc"), "Requires optional do-mpc")
    def test_solve_improves_loss_and_respects_bounds_and_moves(self):
        disturbances = torch.zeros(5).double()
        previous = torch.tensor([.98, -.98, 0., 0., 0.]).double()
        move = torch.full((5,), .05).double()
        result = optimize_do_mpc(self.objective, disturbances, previous, move)
        controls = result["controls"]
        self.assertTrue(torch.all(controls >= -1. - 1e-7))
        self.assertTrue(torch.all(controls <= 1. + 1e-7))
        self.assertTrue(torch.all((controls - previous).abs() <= move + 1e-7))
        self.assertLessEqual(self.objective(disturbances, controls, previous).item(),
                             self.objective(disturbances, previous, previous).item() + 1e-7)

    def test_rejects_unsupported_network_layers(self):
        self.objective.model = nn.Sequential(nn.Linear(10, 2).double(), nn.ReLU())
        with self.assertRaisesRegex(ValueError, "Unsupported surrogate layer"):
            casadi_objective(self.objective)


if __name__ == "__main__":
    unittest.main()