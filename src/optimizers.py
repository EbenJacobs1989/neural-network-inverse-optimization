"""Optimization algorithms for inverse process optimization."""

from time import perf_counter

import numpy as np
import torch
from torch import nn

from objective_functions import InverseProcessObjective


def casadi_objective(objective: InverseProcessObjective):
	"""Translate the supported frozen Linear/SiLU network and loss to CasADi."""
	import casadi as ca

	def array(tensor):
		return tensor.detach().cpu().numpy().astype(float)

	disturbances = ca.SX.sym("disturbances", 5)
	controls = ca.SX.sym("controls", 5)
	previous = ca.SX.sym("previous", 5)
	physical = ca.vertcat(disturbances, controls)
	input_scale = ca.DM(array(objective.input_scale))
	output_scale = ca.DM(array(objective.output_scale))
	prediction = (physical - ca.DM(array(objective.input_mean))) / input_scale
	network = objective.model
	if not isinstance(network, nn.Sequential):
		network = getattr(network, "network", None)
	if not isinstance(network, nn.Sequential):
		raise ValueError("Expected an nn.Sequential model or model.network")
	for layer in network:
		if isinstance(layer, nn.Linear):
			prediction = ca.mtimes(ca.DM(array(layer.weight)), prediction)
			if layer.bias is not None:
				prediction = prediction + ca.DM(array(layer.bias))
		elif isinstance(layer, nn.SiLU):
			prediction = prediction * (0.5 * (1 + ca.tanh(prediction / 2)))
		else:
			raise ValueError(f"Unsupported surrogate layer: {type(layer).__name__}")
	if prediction.shape != (2, 1):
		raise ValueError("Surrogate must produce two outputs")
	outputs = prediction * output_scale + ca.DM(array(objective.output_mean))
	control_scale = input_scale[5:]
	violation = ca.sumsqr(ca.fmax(
		(ca.DM(array(objective.control_lower)) - controls) / control_scale, 0
	)) + ca.sumsqr(ca.fmax(
		(controls - ca.DM(array(objective.control_upper))) / control_scale, 0
	))
	if objective.quality_min is not None:
		violation += ca.fmax((objective.quality_min - outputs[0]) / output_scale[0], 0) ** 2
	if objective.energy_max is not None:
		violation += ca.fmax((outputs[1] - objective.energy_max) / output_scale[1], 0) ** 2
	support = ca.sumsqr(ca.fmax(
		(ca.DM(array(objective.support_lower)) - physical) / input_scale, 0
	)) + ca.sumsqr(ca.fmax(
		(physical - ca.DM(array(objective.support_upper))) / input_scale, 0
	))
	loss = (
		-objective.quality_weight * prediction[0]
		+ objective.energy_weight * prediction[1]
		+ objective.movement_weight * ca.sumsqr((controls - previous) / control_scale)
		+ objective.constraint_weight * violation
		+ objective.support_weight * support
	)
	return ca.Function("inverse_objective", [disturbances, controls, previous], [loss, outputs])


def optimize_do_mpc(
	objective: InverseProcessObjective,
	disturbances: torch.Tensor,
	previous_controls: torch.Tensor,
	max_physical_move: torch.Tensor,
	*,
	max_iterations: int = 250,
) -> dict:
	"""Solve one static-surrogate move using do-mpc/IPOPT with horizon one.

	The state stores the previous physical controls, not dynamic plant outputs.
	Output/support limits remain squared penalties, as in the PyTorch loss.
	Returned controls have the objective's device and dtype. Failure raises;
	unsuccessful solver iterates are never presented as recommendations.
	"""
	try:
		import casadi as ca
		import do_mpc
	except ImportError as error:
		raise ImportError("Install the optional optimizer with: pip install 'do-mpc>=5.1,<6'") from error

	def vector(value, name):
		result = np.asarray(torch.as_tensor(value).detach().cpu(), dtype=float)
		if result.shape != (5,) or not np.isfinite(result).all():
			raise ValueError(f"{name} must be a finite vector of length 5")
		return result

	fixed = vector(disturbances, "disturbances")
	previous = vector(previous_controls, "previous_controls")
	move = vector(max_physical_move, "max_physical_move")
	if np.any(move <= 0):
		raise ValueError("max_physical_move must be positive")
	if isinstance(max_iterations, bool) or not isinstance(max_iterations, int) or max_iterations < 1:
		raise ValueError("max_iterations must be a positive integer")
	lower = vector(objective.control_lower, "control_lower")
	upper = vector(objective.control_upper, "control_upper")
	if np.any(previous < lower) or np.any(previous > upper):
		raise ValueError("previous_controls must satisfy actuator bounds")

	started = perf_counter()
	symbolic_loss = casadi_objective(objective)
	mpc_model = do_mpc.model.Model("discrete")
	state = mpc_model.set_variable("_x", "previous_controls", shape=(5, 1))
	controls = mpc_model.set_variable("_u", "controls", shape=(5, 1))
	mpc_model.set_rhs("previous_controls", controls)
	loss, outputs = symbolic_loss(ca.DM(fixed), controls, state)
	mpc_model.set_expression("predicted_outputs", outputs)
	mpc_model.setup()
	state = mpc_model.x["previous_controls"]
	controls = mpc_model.u["controls"]
	loss, outputs = symbolic_loss(ca.DM(fixed), controls, state)
	controller = do_mpc.controller.MPC(mpc_model)
	controller.settings.n_horizon = 1
	controller.settings.t_step = 1.0
	controller.settings.nlpsol_opts = {
		"ipopt.print_level": 0, "print_time": False,
		"ipopt.sb": "yes", "ipopt.max_iter": max_iterations,
		"ipopt.tol": 1e-8, "ipopt.bound_relax_factor": 0.0,
	}
	controller.set_objective(mterm=ca.SX(0), lterm=loss)
	controller.set_rterm(controls=np.zeros((5, 1)))
	controller.bounds["lower", "_u", "controls"] = lower
	controller.bounds["upper", "_u", "controls"] = upper
	controller.scaling["_u", "controls"] = objective.input_scale[5:].detach().cpu().numpy()
	controller.scaling["_x", "previous_controls"] = objective.input_scale[5:].detach().cpu().numpy()
	controller.set_nl_cons("move_up", (controls - state) / ca.DM(move), ub=1.0)
	controller.set_nl_cons("move_down", (state - controls) / ca.DM(move), ub=1.0)
	controller.setup()
	controller.x0 = previous.reshape(5, 1)
	controller.u0 = previous.reshape(5, 1)
	controller.set_initial_guess()
	solution = controller.make_step(previous.reshape(5, 1)).reshape(5)
	stats = controller.solver_stats
	if not stats["success"]:
		raise RuntimeError(f"do-mpc/IPOPT failed: {stats['return_status']}")
	result = torch.as_tensor(solution, device=objective.input_mean.device,
							 dtype=objective.input_mean.dtype)
	rounded = vector(result, "solution")
	tolerance = 1e-6 * np.maximum(1.0, upper - lower)
	if (np.any(rounded < lower - tolerance) or np.any(rounded > upper + tolerance)
			or np.any(np.abs(rounded - previous) > move + tolerance)):
		raise RuntimeError("do-mpc returned controls outside actuator or rate limits")
	return {
		"controls": result,
		"status": stats["return_status"],
		"iterations": int(stats["iter_count"]),
		"elapsed_seconds": perf_counter() - started,
	}
