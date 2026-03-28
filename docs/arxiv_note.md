# Tension-driven adaptive symplectic leapfrog for the planar N-body problem

**Abstract.** We describe a minimal, transparent integrator for gravitational N-body dynamics in the plane: **Störmer–Verlet (kick–drift–kick) leapfrog** as the symplectic backbone, with **step size** adapted from a scalar **tension** measured after each step. Tension blends **local relative energy drift** with a **force-imbalance** probe on the acceleration field; an optional **exponential moving average (EMA)** damps oscillations in the adaptation signal—an idea validated on simpler Hamiltonian chains (double/triple pendula) before porting here. The implementation is **NumPy-only** (no SciPy/JAX dependency), suited to **GOAT-TS**-style workflows that prize interpretable telemetry (\(\tau(t)\), \(\Delta t(t)\)) alongside trajectories. We benchmark the **Chenciner–Montgomery figure-8** orbit against a high-order Runge–Kutta baseline (DOP853, external SciPy run) and highlight the **Pythagorean three-body** problem at **\(t=50\)** as a stress test where tension-adaptive time stepping shows its strength.

**1. Symplectic core.** For separable \(H = T(\mathbf{p}) + V(\mathbf{q})\), fixed-step leapfrog is symplectic. Variable \(\Delta t\) breaks exact symplecticity across steps; we therefore keep \(\Delta t\) within modest bounds and smooth the tension signal so the policy does not chatter.

**2. Tension.** After each accepted step, we form \(\tau\) from \(|E_{\mathrm{after}}-E_{\mathrm{before}}|/(|E|+\varepsilon)\) and \(\|\sum_i m_i \mathbf{a}_i\|/(\sum_i m_i\|\mathbf{a}_i\|+\varepsilon)\). Smoothed tension \(\tau_{\mathrm{smooth}} \leftarrow \alpha\tau + (1-\alpha)\tau_{\mathrm{smooth}}\) with **one primary knob** \(\alpha =\) `tension_ema_alpha` \(\in [0,1]\).

**3. Forces.** Pairwise Newtonian gravity uses a **vectorized** \(O(N^2)\) kernel (broadcast reductions) for clarity and speed as \(N\) grows beyond three.

**4. Figures.** This repository ships **trajectory plots**, **live energy and tension panels**, and a **figure-8 animation**—sufficient for a short arXiv-style note or blog post. **Suggested citation** (adapt as needed): point to the GitHub repository [BoggersTheFish/3b_solution](https://github.com/BoggersTheFish/3b_solution) and the `docs/METHODOLOGY.md` narrative for the GOAT-TS mapping.

**5. Outlook.** The same shell accepts alternate potentials or learned corrections; tension traces remain first-class outputs for meta-optimization.

---

*Draft for a ~2-page workshop note; not submitted to arXiv by this repository.*
