# GOAT-TS meta-wave evolution: tension-adaptive symplectic 3-body solver

This document records how **BoggersTheFish’s GOAT-TS constraint-wave methodology** was applied *to the three-body problem itself*—yielding the integrator in `threebody.core`—and how **double/triple-pendulum** work informed the final polish.

## The mapping

| GOAT-TS concept | Three-body realization |
|-----------------|------------------------|
| **Nodes** | Physical bodies **and** integrator subcomponents (kick, drift, energy monitor) |
| **Edges** | Gravitational constraints (pairwise Newtonian forces) |
| **Waves** | Discrete time steps propagating state forward |
| **Tension** | Scalar stress: **relative energy drift** across a step plus a **force-imbalance** proxy on **a** |
| **Break / evolve** | Smoothed tension \(\tau\) drives \(\Delta t\): high \(\tau\) → shrink; low \(\tau\) → grow within \([\Delta t_{\min}, \Delta t_{\max}]\) |

Tension is measured **after** each accepted leapfrog step and feeds into the **next** step size — the solver reflects on its own numerical stress instead of marching at a blind fixed rate.

## Pendulum-stack polish (what changed here)

Lessons carried over from symplectic pendulum integrators and tension-adaptive experiments:

1. **Symplectic spine first** — Keep **Störmer–Verlet / kick–drift–kick** as the deterministic map for a single step. For **fixed** \(\Delta t\) and separable \(H = T(p) + V(q)\), this is the right symplectic split.

2. **Smooth the policy signal** — Raw per-step tension can oscillate and cause **\(\Delta t\) chatter**. An **EMA** on \(\tau\) (see `tension_ema_alpha` in `SimulationConfig`, default **`DEFAULT_TENSION_EMA_ALPHA`**) stabilizes the adaptation loop while preserving responsiveness to real stiffness. Demos expose a **one-line** override for quick sweeps.

3. **Keep \(\Delta t\) bounds honest** — Variable-\(\Delta t\) leapfrog is **not** globally symplectic. Moderate `dt_max` and conservative shrink/grow preserve energy drift on the scale of the **latest benchmarks** (TS row in the README).

4. **Validate on a non-negotiable orbit** — The **Chenciner–Montgomery figure-8** anchors the design: shape, period scale, and energy drift must remain credible under the tension policy.

5. **Vectorized forces** — Accelerations use **`rij = r[None,:,:] - r[:,None,:]`**, **`dist2`**, **`inv_r3`** with a zeroed diagonal (no SciPy required). Potential energy uses upper-triangular pairs.

6. **Telemetry** — `run_simulation(..., step_callback=...)` invokes `(t, \tau_{\mathrm{smooth}}, \Delta t, E)` after **each** step for streaming to meta-optimizers.

7. **Collision-style softening bump** — Optional `collision_avoidance`: if \(\tau\) exceeds `tension_high * collision_tension_multiplier`, increase **softening** toward `collision_softening_cap` to survive brutal close approaches without hand-tuning every orbit.

8. **Pythagorean stress test** — The **3:4:5** triangle mass ratio at rest (**`pythagorean_three_body`**, integrated to **`t = 50`** in `run_pythagorean_demo.py`) exposes repeated close approaches; tension-adaptive \(\Delta t\) is the “showcase” run compared to the calm figure-8 period.

## Evolution narrative (meta-waves)

1. **Baseline wave** — Kick–drift–kick leapfrog + pairwise gravity + consistent total energy.

2. **Tension injection** — \(\tau\) from local energy drift + force-imbalance metric (see `compute_tension`).

3. **Break / evolve** — Thresholds and multiplicative factors map smoothed \(\tau\) to \(\Delta t\) updates.

4. **Benchmark wave** — Compare against a **high-order ODE baseline** (DOP853 via SciPy in an external run): same ICs, same time span. Reported results (README): **TS** — 13.49 s, **5.95×10⁻⁸** energy drift; **DOP853** — 0.156 s, **6.63×10⁻⁹** drift. DOP853 is **not** a dependency of this repository; it is a **reference** for speed/precision of a black-box integrator on the identical problem.

5. **Presentation wave** — Live **energy + tension** plots and a **figure-8 animation** (`animate_figure8_trajectory`) make \(\tau\) and trajectories inspectable — aligned with GOAT-TS “discoverability” goals.

6. **Publication-ready outline** — `docs/arxiv_note.md` sketches a short **arXiv-style** note (figures already ship in-repo).

## What this is (and is not)

- **Is:** Transparent, logic-first **NumPy + Matplotlib** physics and policy — easy to read, adapt, and connect to GOAT-TS.
- **Is not:** A bundled SciPy/JAX stack. Black-box high-order runs (DOP853) are **external** comparison points.

## Future integration with GOAT-TS

- Use **`step_callback`** or stored **`SimulationResult`** series as telemetry for meta-optimizers.
- Swap the force layer; keep the tension feedback shell.
- Let symbolic nodes propose alternate \(\tau\) definitions or EMA schedules; this repo remains the executable ground truth.

---

*This file mirrors the conceptual arc behind `threebody.core.run_simulation` and `benchmark_figure8_ts_config`.*
