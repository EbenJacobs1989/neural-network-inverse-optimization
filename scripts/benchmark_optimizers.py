"""Benchmark identical static scenarios and separate closed-loop experiments."""
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from nn_optimization.cli import main

if __name__ == "__main__":
    raise SystemExit(main("benchmark", ROOT))