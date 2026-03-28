"""Matplotlib plots for trajectories, energy, tension, adaptive Δt, and figure-8 animation."""

from __future__ import annotations

from pathlib import Path

import matplotlib
import matplotlib.animation as mplanim
import matplotlib.pyplot as plt
import numpy as np

from threebody.core import SimulationResult


def _maybe_show(fig: plt.Figure, show: bool) -> None:
    if not show:
        plt.close(fig)
        return
    if matplotlib.get_backend().lower() == "agg":
        plt.close(fig)
        return
    plt.show()


def plot_trajectories(
    result: SimulationResult,
    *,
    title: str = "Three-body trajectories",
    figsize: tuple[float, float] = (7.0, 7.0),
    equal_aspect: bool = True,
    show: bool = True,
    save_path: str | Path | None = None,
) -> plt.Figure:
    """
    Plot planar trajectories of all three bodies in the xy plane.
    """
    pos = result.positions
    fig, ax = plt.subplots(figsize=figsize)
    colors = ["#e74c3c", "#3498db", "#2ecc71"]
    labels = ["Body 1", "Body 2", "Body 3"]
    for i in range(3):
        ax.plot(pos[:, i, 0], pos[:, i, 1], color=colors[i], lw=1.2, alpha=0.9, label=labels[i])
        ax.scatter(pos[0, i, 0], pos[0, i, 1], color=colors[i], s=36, zorder=5, edgecolors="k", lw=0.5)
        ax.scatter(pos[-1, i, 0], pos[-1, i, 1], color=colors[i], s=48, marker="*", zorder=5, edgecolors="k", lw=0.5)
    ax.set_title(title)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.legend(loc="best", framealpha=0.92)
    if equal_aspect:
        ax.set_aspect("equal", adjustable="datalim")
    fig.tight_layout()
    if save_path is not None:
        fig.savefig(save_path, dpi=160, bbox_inches="tight")
    _maybe_show(fig, show)
    return fig


def animate_figure8_trajectory(
    result: SimulationResult,
    *,
    title: str = "Figure-8 orbit (live)",
    interval_ms: int = 28,
    trail_length: int = 140,
    max_frames: int = 480,
    save_path: str | Path | None = None,
    save_fps: int = 22,
    show: bool = True,
    figsize: tuple[float, float] = (7.0, 7.0),
) -> mplanim.FuncAnimation | None:
    """
    Animate the three bodies along stored trajectory (figure-8 friendly).

    Faint full paths are drawn; moving markers trace the orbit. For ``save_path``:
    ``.gif`` uses Pillow if installed; ``.mp4`` / ``.mov`` use ffmpeg. If the
    backend is Agg and ``show`` is True with no ``save_path``, a hint is printed.
    """
    pos = result.positions
    n = pos.shape[0]
    if n < 2:
        raise ValueError("Need at least two stored frames to animate.")
    frame_idx = np.unique(
        np.linspace(0, n - 1, num=min(n, max_frames), dtype=float).astype(np.intp)
    )

    fig, ax = plt.subplots(figsize=figsize)
    colors = ["#e74c3c", "#3498db", "#2ecc71"]
    for i in range(3):
        ax.plot(pos[:, i, 0], pos[:, i, 1], color=colors[i], lw=0.9, alpha=0.22, zorder=1)

    trails: list[list[tuple[float, float]]] = [[] for _ in range(3)]
    trail_lines = [
        ax.plot([], [], color=colors[i], lw=1.05, alpha=0.55, zorder=2)[0] for i in range(3)
    ]
    scatters = [
        ax.scatter([], [], color=colors[i], s=56, zorder=6, edgecolors="k", linewidths=0.6)
        for i in range(3)
    ]

    ax.set_title(title)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_aspect("equal", adjustable="datalim")
    pad = 0.08 * max(
        float(np.ptp(pos[:, :, 0])),
        float(np.ptp(pos[:, :, 1])),
        1e-6,
    )
    ax.set_xlim(float(np.min(pos[:, :, 0])) - pad, float(np.max(pos[:, :, 0])) + pad)
    ax.set_ylim(float(np.min(pos[:, :, 1])) - pad, float(np.max(pos[:, :, 1])) + pad)

    def _update(frame: int) -> tuple:
        fi = int(frame_idx[frame])
        for i in range(3):
            xy = (float(pos[fi, i, 0]), float(pos[fi, i, 1]))
            trails[i].append(xy)
            if len(trails[i]) > trail_length:
                trails[i].pop(0)
            scatters[i].set_offsets(np.array([[xy[0], xy[1]]]))
            if trails[i]:
                tx, ty = zip(*trails[i])
                trail_lines[i].set_data(tx, ty)
            else:
                trail_lines[i].set_data([], [])
        return (*trail_lines, *scatters)

    anim = mplanim.FuncAnimation(
        fig,
        _update,
        frames=len(frame_idx),
        interval=interval_ms,
        blit=False,
    )

    if save_path is not None:
        out = Path(save_path)
        out.parent.mkdir(parents=True, exist_ok=True)
        suffix = out.suffix.lower()
        if suffix == ".gif":
            try:
                anim.save(out, writer="pillow", fps=save_fps)
            except Exception as exc:
                raise RuntimeError(
                    "GIF export requires Pillow (pip install pillow)."
                ) from exc
        elif suffix in (".mp4", ".mov", ".avi"):
            anim.save(out, writer="ffmpeg", fps=save_fps)
        else:
            anim.save(out, fps=save_fps)

    if show:
        if matplotlib.get_backend().lower() == "agg" and save_path is None:
            plt.close(fig)
            print(
                "Animation: non-interactive backend (Agg). "
                "Pass save_path=... (e.g. figure8.gif with Pillow, or .mp4 with ffmpeg) "
                "or use a GUI backend."
            )
            return anim
        plt.show()
    else:
        plt.close(fig)

    return anim


def plot_energy_tension(
    result: SimulationResult,
    *,
    e0: float | None = None,
    title: str = "Energy & tension (tension-adaptive leapfrog)",
    figsize: tuple[float, float] = (10.0, 8.0),
    show: bool = True,
    save_path: str | Path | None = None,
) -> plt.Figure:
    """
    Three stacked panels: relative energy drift, live tension τ(t), and Δt series.
    """
    t = result.times
    e = result.energies
    if e0 is None:
        e0 = float(e[0])
    rel = (e - e0) / (abs(e0) + 1e-15)
    tau = result.tensions
    dts = result.dt_series

    fig, axes = plt.subplots(3, 1, sharex=True, figsize=figsize)
    axes[0].plot(t, rel, color="#8e44ad", lw=1.0)
    axes[0].set_ylabel(r"$(E - E_0) / |E_0|$")
    axes[0].set_title(title)
    axes[0].grid(True, alpha=0.35)

    axes[1].plot(t, tau, color="#c0392b", lw=1.0)
    axes[1].set_ylabel(r"Tension $\tau$")
    axes[1].grid(True, alpha=0.35)

    axes[2].step(t, dts, where="post", color="#16a085", lw=1.0)
    axes[2].set_ylabel(r"$\Delta t$")
    axes[2].set_xlabel("time")
    axes[2].grid(True, alpha=0.35)

    fig.tight_layout()
    if save_path is not None:
        fig.savefig(save_path, dpi=160, bbox_inches="tight")
    _maybe_show(fig, show)
    return fig
