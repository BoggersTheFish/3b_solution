#!/usr/bin/env python3
"""
General planar 3-body demo: user masses, triangular initial configuration.

Run from repo root:
  PYTHONPATH=src python examples/general_3body_demo.py

Or after pip install -e .:
  python examples/general_3body_demo.py
"""

from __future__ import annotations

import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(_ROOT / "src"))

import numpy as np

from threebody.core import SimulationConfig, run_simulation
from threebody.visualize import plot_energy_tension, plot_trajectories


def main() -> None:
    rng = np.random.default_rng(42)
    g = 1.0
    masses = np.array([1.0, 1.2, 0.9], dtype=np.float64)

    # Wide triangle, small velocities — rich chaotic-ish interaction without immediate collision
    pos0 = np.array(
        [
            [-1.0, -0.2],
            [1.0, 0.1],
            [0.0, 1.0],
        ],
        dtype=np.float64,
    )
    vel0 = np.array(
        [
            [0.05, 0.12],
            [-0.08, -0.05],
            [0.02, -0.04],
        ],
        dtype=np.float64,
    )
    # Tiny jitter for reproducible non-symmetric break of degeneracy
    vel0 += rng.normal(0.0, 0.002, size=vel0.shape)

    config = SimulationConfig(
        t_end=25.0,
        dt_initial=0.015,
        dt_min=1e-6,
        dt_max=0.018,
        gravitational_constant=g,
        softening=1e-4,
        tension_high=0.008,
        tension_low=0.001,
        shrink_factor=0.55,
        grow_factor=1.04,
    )

    result = run_simulation(pos0, vel0, masses, config, store_stride=4)

    print(f"Steps taken: {result.step_count}")
    print(f"Termination: {result.termination}")
    e0 = float(result.energies[0])
    e_rel = (result.energies[-1] - e0) / abs(e0)
    print(f"Relative energy change over run: {e_rel:.3e}")

    plot_trajectories(result, title="General 3-body (triangular IC + jitter)")
    plot_energy_tension(result, e0=e0, title="General 3-body: energy & tension")


if __name__ == "__main__":
    main()
