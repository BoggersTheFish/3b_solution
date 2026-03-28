"""
Three-body dynamics with a tension-adaptive symplectic leapfrog integrator.

Core symbols import without matplotlib. Plotting and animation load matplotlib lazily.

Public initial-condition helpers: ``chenciner_montgomery_figure8``, ``pythagorean_three_body``.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from threebody.core import (
    DEFAULT_TENSION_EMA_ALPHA,
    SimulationConfig,
    SimulationResult,
    StepCallback,
    benchmark_figure8_ts_config,
    chenciner_montgomery_figure8,
    gravitational_acceleration,
    pythagorean_three_body,
    run_simulation,
    total_energy,
)

__all__ = [
    "DEFAULT_TENSION_EMA_ALPHA",
    "SimulationConfig",
    "SimulationResult",
    "StepCallback",
    "benchmark_figure8_ts_config",
    "chenciner_montgomery_figure8",
    "pythagorean_three_body",
    "gravitational_acceleration",
    "run_simulation",
    "total_energy",
    "plot_energy_tension",
    "plot_trajectories",
    "animate_figure8_trajectory",
]

__version__ = "1.0.0"


def __getattr__(name: str) -> Any:
    if name in ("plot_energy_tension", "plot_trajectories", "animate_figure8_trajectory"):
        from threebody import visualize

        return getattr(visualize, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


if TYPE_CHECKING:
    from threebody.visualize import animate_figure8_trajectory, plot_energy_tension, plot_trajectories
