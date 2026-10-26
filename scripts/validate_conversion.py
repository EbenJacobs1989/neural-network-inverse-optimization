"""Validate the real checkpoint from any working directory; failure exits nonzero."""

import copy
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import numpy as np
from nn_optimization.conversion import convert_model, equivalence_report
from nn_optimization.model import ForwardModel, ModelContract, load_model


def main() -> int:
    predictor, data, _, _ = load_model(ROOT)
    contract = predictor.contract
    rng = np.random.default_rng(contract.metadata["conversion_seed"])
    historical = data[contract.input_names].to_numpy()[::137]
    interior = rng.uniform(contract.train_min, contract.train_max, (contract.metadata["conversion_random_samples"], contract.n_input))
    samples = np.vstack([contract.input_mean, contract.train_min, contract.train_max,
                         contract.validation_min, contract.validation_max, historical, interior])
    symbolic = convert_model(predictor)
    promoted = equivalence_report(predictor, symbolic, samples,
        prediction_atol=contract.metadata["conversion_prediction_atol"], jacobian_atol=contract.metadata["conversion_jacobian_atol"])
    metadata = dict(contract.metadata, dtype="float32")
    native_contract = ModelContract(metadata, contract.input_mean, contract.input_scale, contract.output_mean,
                                    contract.output_scale, contract.train_min, contract.train_max,
                                    contract.validation_min, contract.validation_max, contract.output_min, contract.output_max)
    native = ForwardModel(copy.deepcopy(predictor.network), native_contract)
    native_report = equivalence_report(native, symbolic, samples,
        prediction_atol=contract.metadata["native_prediction_atol"], jacobian_atol=contract.metadata["native_jacobian_atol"])
    report = {"float64": promoted, "native_float32": native_report,
              "note": "Same saved float32 weights; float64 evaluation isolates symbolic correctness. Native tolerance includes input/output rounding."}
    output = ROOT / "results/aipc"
    output.mkdir(parents=True, exist_ok=True)
    (output / "conversion_validation.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    return 0 if promoted["accepted"] and native_report["accepted"] else 1


if __name__ == "__main__":
    raise SystemExit(main())