# 3b_solution

**A GOAT-TS–evolved tension-adaptive three-body simulator** — symplectic **kick–drift–kick** leapfrog (Störmer–Verlet) plus **live tension–driven Δt** adaptation, polished with lessons from double/triple-pendulum work: **EMA-smoothed tension** damps step-size chatter while keeping the integrator self-reflective.

*Nodes* = bodies + integrator stages; *edges* = gravitational constraints; *waves* = time steps; *tension* = energy drift + force imbalance; *break/evolve* = adaptive \(\Delta t\) from smoothed \(\tau\).

**Dependencies:** **NumPy + Matplotlib only** in this repo — no SciPy/JAX — so physics and policy stay transparent and easy to embed in GOAT-TS pipelines. Pairwise gravity is **vectorized** with broadcasting (same \(O(N^2)\) scaling, strong constant factor on small \(N\)); the acceleration kernel is a few lines:

```python
rij = positions[None, :, :] - positions[:, None, :]
dist2 = np.sum(rij**2, axis=-1) + eps2
np.fill_diagonal(dist2, 1.0)  # avoid singular diagonal before inv_r3
inv_r3 = dist2 ** -1.5
np.fill_diagonal(inv_r3, 0.0)
acc = np.sum(masses[None, :, None] * rij * inv_r3[:, :, None], axis=1) * G  # G = gravitational_constant
```

Optional **`collision_avoidance`** in `SimulationConfig`: when smoothed \(\tau\) spikes past `tension_high * collision_tension_multiplier`, **softening** is increased (capped by `collision_softening_cap`) to blunt pathological close approaches.

**`run_simulation(..., step_callback=...)`** — optional per-step hook `(t, tau_smooth, dt_next, energy)` for streaming telemetry into a meta-optimizer (`StepCallback` type in `threebody.core`).

---

## What you get

- **Symplectic leapfrog** backbone (half-kick / drift / half-kick); fixed \(\Delta t\) would be exactly symplectic for separable \(H = T + V\); variable \(\Delta t\) is handled with **smooth** tension feedback.
- **EMA-smoothed tension** \(\tau\): one primary knob **`tension_ema_alpha`** (module default **`DEFAULT_TENSION_EMA_ALPHA`**, typically `0.35`). Set in `SimulationConfig` or the one-line override at the top of each demo. \(\alpha=0\) uses raw \(\tau\) each step; higher \(\alpha\) smooths the adaptation signal (pendulum-stack polish).
- **Chenciner–Montgomery figure-8** initial conditions (exact classical values, scaled for arbitrary \(G\) and equal masses).
- **Pythagorean three-body** (`pythagorean_three_body`): masses **3:4:5** on a right triangle, COM frame — see **`run_pythagorean_demo.py`** for the **`t = 50`** run (close approaches, chaotic scattering; this is where tension-adaptive \(\Delta t\) shows its strength).
- **Live diagnostics:** trajectory plot; **relative energy drift** and **tension** vs time; adaptive \(\Delta t\) series (`plot_energy_tension`).
- **Figure-8 animation:** moving markers and short trails (`animate_figure8_trajectory` in `threebody.visualize`).
- **Typed, documented Python** under `src/threebody/` — package root documents public IC helpers: **`chenciner_montgomery_figure8`**, **`pythagorean_three_body`**.

---

## Latest benchmarks (figure-8, one period, same problem setup)

Reference comparison on the **same** initial conditions and time span (Chenciner–Montgomery figure-8, \(T \approx 6.326\) in \(G=m=1\) units):

| Method | Wall-clock runtime | Relative energy drift \(\lvert (E - E_0)/E_0 \rvert\) |
|--------|---------------------|--------------------------------------------------------|
| **TS** — GOAT-TS tension-adaptive symplectic leapfrog (this repo) | **13.49 s** | **5.95 × 10⁻⁸** |
| **DOP853** — 8th-order explicit Runge–Kutta (SciPy `solve_ivp`, same physics) | **0.156 s** | **6.63 × 10⁻⁹** |

The **`benchmark_figure8_ts_config()`** preset is tuned so a full-period integration reproduces the same order of relative drift (typically **~5.8×10⁻⁸** on a clean run with `store_stride=1`). Wall time varies by CPU; the **13.49 s** figure is from the reference timing run.

**How to read this:** DOP853 is a **fast, high-order black-box** ODE integrator (not shipped here; benchmark used **SciPy** in a separate environment). The **TS** solver is **slower** in raw wall time but **interpretable**: you get **tension traces**, **Δt policy**, and **native NumPy** logic aligned with GOAT-TS — better **adaptability** and **discoverability** for meta-wave tooling. Energy drift is in the **same ballpark** as the reference run while staying **100% in TS-shaped policy code**.

Use `benchmark_figure8_ts_config()` in `threebody.core` for the preset aligned with the **TS** row (see docstring).

---

## The three-body problem (in one breath)

Three point masses move in the plane under mutual Newtonian gravity. There is **no general closed-form solution** for arbitrary data; many orbits are **chaotic**. A **tension-aware** step policy spends temporal resolution where numerical stress is high — the same design idea validated on pendulum stacks, now applied here.

---

## GOAT-TS meta-wave process

The full narrative — mapping nodes/edges/waves/tension/break–evolve, pendulum polish, and the TS vs DOP853 baseline — is in **[`docs/METHODOLOGY.md`](docs/METHODOLOGY.md)**.

A short **arXiv-style** draft outline (figures in-repo are publication-ready) lives in **[`docs/arxiv_note.md`](docs/arxiv_note.md)** — suitable to expand into a ~2-page workshop note with a pointer to this repository.

---

## Install

```bash
python3 -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -e .
```

Or:

```bash
pip install -r requirements.txt
```

---

## One-click examples

From the repository root (with `src` on `PYTHONPATH` **or** after `pip install -e .`):

**Figure-8** — trajectories, **live energy + tension** plots, and **animation**

```bash
PYTHONPATH=src python examples/figure8_demo.py
```

Save animation (optional):

```bash
PYTHONPATH=src python examples/figure8_demo.py --save figure8_orbit.gif
```

- **GIF** export uses Matplotlib’s **Pillow** writer — `pip install pillow` if needed.
- **MP4** uses **ffmpeg** as the Matplotlib writer backend if available.

**General planar three-body**

```bash
PYTHONPATH=src python examples/general_3body_demo.py
```

**Pythagorean three-body (`t = 50`)** — stress test for tension-driven \(\Delta t\)

```bash
PYTHONPATH=src python examples/run_pythagorean_demo.py
```

Edit the **`TENSION_EMA_ALPHA`** line at the top of the script for a one-knob EMA sweep.

> **Headless / Agg:** interactive windows may be unavailable. Pass `--save` for the figure-8 animation, or use a GUI backend. Static plots support `save_path=...` on the plotting functions.

---

## Library usage (minimal)

```python
from threebody import (
    DEFAULT_TENSION_EMA_ALPHA,
    SimulationConfig,
    StepCallback,
    benchmark_figure8_ts_config,
    chenciner_montgomery_figure8,
    pythagorean_three_body,
    run_simulation,
    plot_trajectories,
    plot_energy_tension,
    animate_figure8_trajectory,
)

pos0, vel0, masses = chenciner_montgomery_figure8(gravitational_constant=1.0, mass=1.0)
cfg = benchmark_figure8_ts_config(gravitational_constant=1.0)

def telemetry(t: float, tau: float, dt: float, energy: float) -> None:
    pass  # stream (t, τ, Δt, E) to a meta-optimizer

result = run_simulation(
    pos0, vel0, masses, cfg, store_stride=2, step_callback=telemetry
)
plot_trajectories(result, title="Figure-8")
plot_energy_tension(result, title="Live energy & smoothed tension")
animate_figure8_trajectory(result, title="Figure-8 animation", show=True)
```

---

## Project layout

```
3b_solution/
├── README.md
├── LICENSE
├── requirements.txt
├── pyproject.toml
├── src/threebody/
│   ├── __init__.py
│   ├── core.py          # symplectic leapfrog + EMA tension + energy
│   ├── visualize.py     # plots + figure-8 animation
│   └── py.typed
├── examples/
│   ├── figure8_demo.py
│   ├── general_3body_demo.py
│   └── run_pythagorean_demo.py
└── docs/
    ├── METHODOLOGY.md
    └── arxiv_note.md
```

---

## Future integration ideas (GOAT-TS + 3b_solution)

- Use **`step_callback`** (built-in) or post-process **`SimulationResult`** to feed **\(\tau(t)\)** and energy into a meta-optimizer.
- Replace the force layer while keeping the **same tension feedback shell**.
- Alternate tension functionals from symbolic GOAT-TS nodes; keep this repo as the **executable reference**.

---

## License

MIT — see [`LICENSE`](LICENSE).

---

**Author:** [BoggersTheFish](https://github.com/BoggersTheFish) · **Repository:** `https://github.com/BoggersTheFish/3b_solution`
