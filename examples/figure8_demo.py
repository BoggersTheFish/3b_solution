#!/usr/bin/env python3
"""
One-click figure-8 (Chenciner–Montgomery) demo: tension-adaptive symplectic leapfrog,
live energy + tension plots, and figure-8 animation.

Run from repo root:
  PYTHONPATH=src python examples/figure8_demo.py

Optional animation file (GIF needs Pillow; MP4 needs ffmpeg):
  PYTHONPATH=src python examples/figure8_demo.py --save figure8_orbit.gif

Or after pip install -e .:
  python examples/figure8_demo.py
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(_ROOT / "src"))

from threebody.core import (
    benchmark_figure8_ts_config,
    chenciner_montgomery_figure8,
    run_simulation,
)
from threebody.visualize import animate_figure8_trajectory, plot_energy_tension, plot_trajectories


def main() -> None:
    parser = argparse.ArgumentParser(description="Figure-8 demo: trajectories, energy/tension, animation.")
    parser.add_argument(
        "--save",
        type=str,
        default=None,
        metavar="PATH",
        help="Write animation to this path (.gif with Pillow, .mp4 with ffmpeg).",
    )
    args = parser.parse_args()

    g = 1.0
    config = benchmark_figure8_ts_config(gravitational_constant=g)
    pos0, vel0, masses = chenciner_montgomery_figure8(gravitational_constant=g, mass=1.0)

    result = run_simulation(pos0, vel0, masses, config, store_stride=2)

    print(f"Steps taken: {result.step_count}")
    print(f"Termination: {result.termination}")
    e0 = float(result.energies[0])
    e_rel = (result.energies[-1] - e0) / abs(e0)
    print(f"Relative energy change over run: {e_rel:.3e}")

    plot_trajectories(result, title="Figure-8 orbit (Chenciner–Montgomery IC)")
    plot_energy_tension(
        result,
        e0=e0,
        title="Figure-8: live energy drift & smoothed tension (EMA τ)",
    )

    save_path = Path(args.save) if args.save else None
    show_anim = save_path is None
    animate_figure8_trajectory(
        result,
        title="Figure-8 animation (tension-adaptive symplectic leapfrog)",
        save_path=save_path,
        show=show_anim,
    )


if __name__ == "__main__":
    main()
