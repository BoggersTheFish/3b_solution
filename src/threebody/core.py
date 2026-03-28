"""
Tension-adaptive symplectic leapfrog for the planar three-body problem.

The spine is **Störmer–Verlet / kick–drift–kick** leapfrog. For fixed Δt and a
separable Hamiltonian H = T(p) + V(q), the map is symplectic. When Δt varies
step-to-step (tension-driven adaptation), the map is only approximately
symplectic; polish from double/triple-pendulum work keeps adaptation **smooth**
so Δt does not chatter:

- **EMA-smoothed tension** damps single-step noise before shrink/grow decisions.
- **Tight bounds** on Δt keep energy drift small while still exposing where the
  orbit is stiff.

Tension τ blends relative energy drift across a step with a force-imbalance
proxy on **a**. High τ shrinks the next step; low τ allows growth within
[dt_min, dt_max].
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np
from numpy.typing import NDArray

Float64 = np.float64
Vec2 = NDArray[np.float64]  # shape (3, 2)


def gravitational_acceleration(
    positions: Vec2,
    masses: NDArray[np.float64],
    gravitational_constant: float,
    softening: float = 0.0,
) -> Vec2:
    """
    Pairwise Newtonian gravity in 2D with optional softening ε² in the denominator.

    a_i = G * Σ_{j≠i} m_j * (r_j - r_i) / (|r_ij|² + ε²)^(3/2)
    """
    n = positions.shape[0]
    acc = np.zeros_like(positions, dtype=np.float64)
    g = float(gravitational_constant)
    eps2 = float(softening) ** 2

    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            rij = positions[j] - positions[i]
            dist2 = float(np.dot(rij, rij)) + eps2
            inv_r3 = dist2 ** (-1.5)
            acc[i] += g * masses[j] * rij * inv_r3
    return acc


def total_energy(
    positions: Vec2,
    velocities: Vec2,
    masses: NDArray[np.float64],
    gravitational_constant: float,
    softening: float = 0.0,
) -> float:
    """Total mechanical energy E = T + U for pairwise gravity in 2D."""
    ke = 0.5 * float(np.sum(masses[:, np.newaxis] * velocities**2))
    pe = 0.0
    n = positions.shape[0]
    g = float(gravitational_constant)
    eps2 = float(softening) ** 2
    for i in range(n):
        for j in range(i + 1, n):
            rij = positions[j] - positions[i]
            dist = np.sqrt(float(np.dot(rij, rij)) + eps2)
            pe -= g * masses[i] * masses[j] / dist
    return ke + pe


def _force_imbalance_metric(accelerations: Vec2, masses: NDArray[np.float64]) -> float:
    """
    Internal momentum-flux proxy: ||Σ m_i a_i|| / (Σ m_i ||a_i|| + ε).

    For exact pairwise gravity, Σ m_i a_i = 0. Numerical asymmetry shows up here.
    """
    total = np.sum(masses[:, np.newaxis] * accelerations, axis=0)
    num = float(np.linalg.norm(total))
    den = float(np.sum(masses * np.linalg.norm(accelerations, axis=1))) + 1e-15
    return num / den


def compute_tension(
    energy_before: float,
    energy_after: float,
    acc: Vec2,
    masses: NDArray[np.float64],
    w_energy: float = 1.0,
    w_force: float = 0.5,
    eps_energy: float = 1e-12,
) -> float:
    """
    Scalar tension τ ∈ [0, ∞): larger means the step was more stressful.

    τ = w_e * |ΔE| / (|E| + ε) + w_f * (force imbalance metric).
    """
    e_mid = 0.5 * (energy_before + energy_after)
    rel = abs(energy_after - energy_before) / (abs(e_mid) + eps_energy)
    f_imb = _force_imbalance_metric(acc, masses)
    return float(w_energy * rel + w_force * f_imb)


def chenciner_montgomery_figure8(
    gravitational_constant: float = 1.0,
    mass: float = 1.0,
) -> tuple[Vec2, Vec2, NDArray[np.float64]]:
    """
    Exact Chenciner–Montgomery figure-8 initial conditions (equal masses, G=1).

    Positions and velocities are dimensionally consistent with the given G and mass;
    velocities are scaled so that the same orbit shape is obtained when G and m
    are changed (standard Hamiltonian scaling).
    """
    # Classical normalized values (G=1, m=1)
    q1 = np.array([0.97000436, -0.24308753], dtype=np.float64)
    q2 = np.array([-0.97000436, 0.24308753], dtype=np.float64)
    q3 = np.array([0.0, 0.0], dtype=np.float64)
    positions = np.stack([q1, q2, q3], axis=0)

    v1 = np.array([0.46620368, 0.43236573], dtype=np.float64)
    v2 = np.array([0.46620368, 0.43236573], dtype=np.float64)
    v3 = np.array([-0.93240736, -0.86473146], dtype=np.float64)
    velocities = np.stack([v1, v2, v3], axis=0)

    masses = np.full(3, float(mass), dtype=np.float64)
    gm = float(gravitational_constant) * float(mass)
    # Scale from (G=1,m=1) to arbitrary G,m: time/velocity scale as sqrt/Gm, length fixed
    if gravitational_constant != 1.0 or mass != 1.0:
        scale_v = np.sqrt(gm)
        velocities = velocities * scale_v
    return positions, velocities, masses


def benchmark_figure8_ts_config(
    gravitational_constant: float = 1.0,
) -> SimulationConfig:
    """
    Preset used for the README **TS** benchmark row (figure-8, one period, G=m=1).

    Reported on the reference run: wall time **13.49 s**, relative energy drift
    **5.95e-8** (see README). Tune ``dt_*`` / tension fields if your machine or
    stride differs; physics and policy are unchanged.
    """
    return SimulationConfig(
        t_end=6.326,
        dt_initial=0.009,
        dt_min=1e-5,
        dt_max=0.01032,
        gravitational_constant=gravitational_constant,
        softening=0.0,
        tension_high=0.0038,
        tension_low=0.0005,
        shrink_factor=0.62,
        grow_factor=1.03,
        tension_ema_alpha=0.35,
        w_energy=1.0,
        w_force=0.5,
    )


@dataclass
class SimulationConfig:
    """Hyperparameters for tension-adaptive symplectic leapfrog."""

    t_end: float
    dt_initial: float
    dt_min: float = 1e-6
    # Variable-Δt maps are not globally symplectic — keep max step modest.
    dt_max: float = 0.01032
    gravitational_constant: float = 1.0
    softening: float = 0.0
    # Tension feedback uses EMA-smoothed τ (pendulum-style polish) before shrink/grow.
    tension_high: float = 0.0038
    tension_low: float = 0.0005
    shrink_factor: float = 0.62
    grow_factor: float = 1.03
    # τ_smooth ← α·τ_raw + (1−α)·τ_smooth_prev ; α=1 disables smoothing.
    tension_ema_alpha: float = 0.35
    w_energy: float = 1.0
    w_force: float = 0.5
    max_steps: int = 5_000_000


@dataclass
class SimulationResult:
    """Time series from a completed run."""

    times: NDArray[np.float64]
    positions: NDArray[np.float64]  # (T, 3, 2)
    energies: NDArray[np.float64]
    tensions: NDArray[np.float64]
    dt_series: NDArray[np.float64]
    step_count: int
    termination: Literal["time", "max_steps"]


def run_simulation(
    positions0: Vec2,
    velocities0: Vec2,
    masses: NDArray[np.float64],
    config: SimulationConfig,
    *,
    store_stride: int = 1,
) -> SimulationResult:
    """
    Integrate the 3-body problem with tension-adaptive kick–drift–kick leapfrog.

    At each accepted step, tension is computed from the energy jump across the step
    and the acceleration field; Δt is adjusted before the next step.
    """
    pos = np.array(positions0, dtype=np.float64, copy=True)
    vel = np.array(velocities0, dtype=np.float64, copy=True)
    masses = np.asarray(masses, dtype=np.float64)

    g = config.gravitational_constant
    eps = config.softening

    dt = float(np.clip(config.dt_initial, config.dt_min, config.dt_max))

    times_list: list[float] = [0.0]
    pos_snapshots: list[Vec2] = [pos.copy()]
    energy_list: list[float] = []
    tension_list: list[float] = []
    dt_list: list[float] = []

    acc = gravitational_acceleration(pos, masses, g, eps)
    e0 = total_energy(pos, vel, masses, g, eps)
    energy_list.append(e0)
    tension_list.append(0.0)
    dt_list.append(dt)

    t = 0.0
    step = 0
    stride_counter = 0
    tau_smooth = 0.0
    ema = float(config.tension_ema_alpha)
    ema_clip = float(np.clip(ema, 0.0, 1.0)) if ema > 0.0 else 0.0

    while t < config.t_end - 1e-15:
        if step >= config.max_steps:
            return _finalize_result(
                times_list,
                pos_snapshots,
                energy_list,
                tension_list,
                dt_list,
                step,
                "max_steps",
                store_stride,
            )

        dt_step = float(np.clip(dt, config.dt_min, config.dt_max))
        remaining = config.t_end - t
        if dt_step > remaining:
            dt_step = remaining

        e_before = total_energy(pos, vel, masses, g, eps)

        # Kick half
        vel_half = vel + 0.5 * dt_step * acc
        # Drift
        pos = pos + dt_step * vel_half
        # New acceleration and kick half
        acc_new = gravitational_acceleration(pos, masses, g, eps)
        vel = vel_half + 0.5 * dt_step * acc_new
        acc = acc_new

        t += dt_step
        step += 1

        e_after = total_energy(pos, vel, masses, g, eps)
        tau_raw = compute_tension(
            e_before,
            e_after,
            acc,
            masses,
            w_energy=config.w_energy,
            w_force=config.w_force,
        )
        if ema <= 0.0:
            tau_smooth = tau_raw
        else:
            tau_smooth = ema_clip * tau_raw + (1.0 - ema_clip) * tau_smooth

        # Adapt next Δt from smoothed tension (reduces Δt chatter vs raw τ).
        if tau_smooth > config.tension_high:
            dt *= config.shrink_factor
        elif tau_smooth < config.tension_low:
            dt *= config.grow_factor
        dt = float(np.clip(dt, config.dt_min, config.dt_max))

        stride_counter += 1
        if stride_counter % store_stride == 0:
            times_list.append(t)
            pos_snapshots.append(pos.copy())
            energy_list.append(e_after)
            tension_list.append(tau_smooth)
            dt_list.append(dt)

    # Ensure the final state at t_end is recorded for plotting when stride skips it
    if abs(times_list[-1] - t) > 1e-14 and step > 0:
        times_list.append(t)
        pos_snapshots.append(pos.copy())
        energy_list.append(float(total_energy(pos, vel, masses, g, eps)))
        tension_list.append(tau_smooth)
        dt_list.append(dt)

    return _finalize_result(
        times_list,
        pos_snapshots,
        energy_list,
        tension_list,
        dt_list,
        step,
        "time",
        store_stride,
    )


def _finalize_result(
    times_list: list[float],
    pos_snapshots: list[Vec2],
    energy_list: list[float],
    tension_list: list[float],
    dt_list: list[float],
    step_count: int,
    termination: Literal["time", "max_steps"],
    store_stride: int,
) -> SimulationResult:
    times = np.array(times_list, dtype=np.float64)
    positions = np.stack(pos_snapshots, axis=0)
    energies = np.array(energy_list, dtype=np.float64)
    tensions = np.array(tension_list, dtype=np.float64)
    dt_series = np.array(dt_list, dtype=np.float64)
    _ = store_stride  # reserved for future subsampling metadata
    return SimulationResult(
        times=times,
        positions=positions,
        energies=energies,
        tensions=tensions,
        dt_series=dt_series,
        step_count=step_count,
        termination=termination,
    )
