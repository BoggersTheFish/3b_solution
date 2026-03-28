# 3b_solution

**A GOAT-TS–evolved tension-adaptive three-body simulator** — symplectic **kick–drift–kick** leapfrog (Störmer–Verlet) plus **live tension–driven Δt** adaptation, polished with lessons from double/triple-pendulum work: **EMA-smoothed tension** damps step-size chatter while keeping the integrator self-reflective.

*Nodes* = bodies + integrator stages; *edges* = gravitational constraints; *waves* = time steps; *tension* = energy drift + force imbalance; *break/evolve* = adaptive \(\Delta t\) from smoothed \(\tau\).

**Dependencies:** **NumPy + Matplotlib only** in this repo — no SciPy/JAX — so physics and policy stay transparent and easy to embed in GOAT-TS pipelines.

---

## What you get

- **Symplectic leapfrog** backbone (half-kick / drift / half-kick); fixed \(\Delta t\) would be exactly symplectic for separable \(H = T + V\); variable \(\Delta t\) is handled with **smooth** tension feedback.
- **EMA-smoothed tension** \(\tau\) (configurable `tension_ema_alpha`): shrink/grow decisions use smoothed \(\tau\), not raw single-step noise.
- **Chenciner–Montgomery figure-8** initial conditions (exact classical values, scaled for arbitrary \(G\) and equal masses).
- **Live diagnostics:** trajectory plot; **relative energy drift** and **tension** vs time; adaptive \(\Delta t\) series (`plot_energy_tension`).
- **Figure-8 animation:** moving markers and short trails (`animate_figure8_trajectory` in `threebody.visualize`).
- **Typed, documented Python** under `src/threebody/`.

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

> **Headless / Agg:** interactive windows may be unavailable. Pass `--save` for the figure-8 animation, or use a GUI backend. Static plots support `save_path=...` on the plotting functions.

---

## Library usage (minimal)

```python
from threebody import (
    SimulationConfig,
    benchmark_figure8_ts_config,
    chenciner_montgomery_figure8,
    run_simulation,
    plot_trajectories,
    plot_energy_tension,
    animate_figure8_trajectory,
)

pos0, vel0, masses = chenciner_montgomery_figure8(gravitational_constant=1.0, mass=1.0)
cfg = benchmark_figure8_ts_config(gravitational_constant=1.0)
result = run_simulation(pos0, vel0, masses, cfg, store_stride=2)
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
│   └── general_3body_demo.py
└── docs/
    └── METHODOLOGY.md
```

---

## Future integration ideas (GOAT-TS + 3b_solution)

- Stream **\(\tau(t)\)** and energy residuals into a meta-optimizer as **telemetry**.
- Replace the force layer while keeping the **same tension feedback shell**.
- Alternate tension functionals from symbolic GOAT-TS nodes; keep this repo as the **executable reference**.

---

## License

MIT — see [`LICENSE`](LICENSE).

---

**Author:** [BoggersTheFish](https://github.com/BoggersTheFish) · **Repository:** `https://github.com/BoggersTheFish/3b_solution`
