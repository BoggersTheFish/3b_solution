#!/usr/bin/env python3
"""
Pythagorean three-body problem to t = 50 — the strength case for tension-adaptive Δt.

Masses 3 : 4 : 5 on a right triangle (COM frame, initially at rest): repeated close
approaches and chaotic scattering make fixed-Δt leapfrog painful; the tension policy
allocates resolution where stress spikes.

Run:
  PYTHONPATH=src python examples/run_pythagorean_demo.py
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(_ROOT / "src"))

from threebody.core import DEFAULT_TENSION_EMA_ALPHA, SimulationConfig, pythagorean_three_body, run_simulation
from threebody.visualize import plot_energy_tension, plot_trajectories

# --- one-line tunable: EMA blend on tension (0 = raw τ each step; ~0.35 = smooth policy) ---
TENSION_EMA_ALPHA = DEFAULT_TENSION_EMA_ALPHA


def main() -> None:
    g = 1.0
    pos0, vel0, masses = pythagorean_three_body(gravitational_constant=g)

    config = SimulationConfig(
        t_end=50.0,
        dt_initial=0.018,
        dt_min=1e-6,
        dt_max=0.028,
        gravitational_constant=g,
        softening=1e-4,
        tension_high=0.014,
        tension_low=0.0018,
        shrink_factor=0.56,
        grow_factor=1.035,
        tension_ema_alpha=TENSION_EMA_ALPHA,
    )

    t0 = time.perf_counter()
    result = run_simulation(pos0, vel0, masses, config, store_stride=8)
    wall = time.perf_counter() - t0

    print(f"Wall time: {wall:.2f} s")
    print(f"Steps taken: {result.step_count}")
    print(f"Termination: {result.termination}")
    e0 = float(result.energies[0])
    e_rel = (result.energies[-1] - e0) / abs(e0)
    print(f"Relative energy change over run: {e_rel:.3e}")

    plot_trajectories(
        result,
        title="Pythagorean 3-body (t = 50): masses 3:4:5, tension-adaptive leapfrog",
    )
    plot_energy_tension(
        result,
        e0=e0,
        title="Pythagorean (t=50): live energy drift, smoothed τ, adaptive Δt",
    )


if __name__ == "__main__":
    main()
