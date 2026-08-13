# CR3BP: Earth-Moon System

A numerical toolkit for the **3D Circular Restricted Three-Body Problem (CR3BP)**,
applied to spacecraft trajectories in the Earth-Moon system. Models a spacecraft
of negligible mass under the combined gravitational attraction of Earth and Moon,
integrated with a validated 4th-order Runge-Kutta scheme.

Originally developed for ME2 Maths & Computing coursework at Imperial College
London (2025-2026), refactored here into a tested, reusable Python package.

![3D CR3BP trajectories](figures/01_trajectories_3d.png)

## What this models

The CR3BP treats Earth and Moon as two primaries orbiting their common centre
of mass in circular orbits, in a rotating, non-dimensional reference frame. A
third body (the spacecraft) has negligible mass and does not perturb the
primaries — hence *restricted* three-body problem, as opposed to the general
three-body problem.

The equations of motion in the rotating frame:

```
x'' - 2y' = ∂Ω/∂x
y'' + 2x' = ∂Ω/∂y
z''       = ∂Ω/∂z
```

where the effective potential Ω combines gravitational attraction from both
primaries with the centrifugal effect of the rotating frame:

```
Ω(x, y, z) = 0.5*(x² + y²) + (1-μ)/r1 + μ/r2
```

Note the z-equation contains only gravitational terms — Coriolis and
centrifugal effects act purely in the rotating x-y plane, since the rotation
axis is parallel to z. This is what allows bounded out-of-plane oscillation in
the 3D cases while the planar orbits remain exactly at z = 0.

μ is the Earth-Moon mass ratio:

```
μ = M_Moon / (M_Earth + M_Moon) ≈ 0.01215058
```

## Method

**Integrator:** classical 4th-order Runge-Kutta (RK4). The CR3BP is an initial
value problem advanced over many time steps, so accumulated numerical error
matters — RK4 gives accurate trajectories at acceptable computational cost.
The second-order equations are rewritten as a first-order system in
`[x, y, z, vx, vy, vz]` so RK4 applies directly.

**Validation:** stability and accuracy are assessed via the **Jacobi
constant**, `C_J = 2Ω - v²`, which is analytically conserved in the CR3BP.
For the chosen step size `Δt = 0.0002`, relative drift stays small across all
four tested trajectories (max ~1.7×10⁻⁸), confirming the dominant error source
is time discretisation, not RK4 instability.

**Post-processing:** Poincaré sections (z = 0 crossings for 3D orbits, y = 0
for planar orbits), zero-velocity curves derived from the Jacobi constant
field, and a windowed FFT (Hanning window, to reduce spectral leakage) of each
trajectory component to identify dominant in-plane vs out-of-plane
frequencies.

## Four tested orbits

| Orbit | Description | Initial state `[x, y, z, vx, vy, vz]` |
|---|---|---|
| A | Planar retrograde around Earth | `[0.4, 0, 0, 0, -1.2, 0]` |
| B | Planar transfer | `[0.4, 0, 0, 0, 1.05, 0]` |
| C | 3D halo-like orbit near L1 | `[L1-0.02, 0, 0.04, 0, -0.15, 0]` |
| D | 3D orbit near the Moon | `[0.92, 0, 0.03, 0, 0.5, 0.05]` |

Orbits A and B remain exactly planar (z ≡ 0), confirming the 3D code
correctly reduces to the planar case. Orbits C and D show genuine bounded
out-of-plane motion.

## Repo structure

```
src/cr3bp/
  dynamics.py           effective potential, equations of motion, Jacobi constant, Lagrange points
  integrator.py          RK4 step + full trajectory integration
  initial_conditions.py  the four orbit definitions (A-D)
  postprocessing.py      Poincaré sections, zero-velocity field, FFT analysis
scripts/
  run_analysis.py        end-to-end script: integrates all orbits, validates, saves all figures
notebooks/
  CR3BP_earth_moon_analysis.ipynb   original exploratory notebook
tests/
  test_dynamics.py        sanity tests: Jacobi conservation, planar reduction, Lagrange point
figures/                  generated output figures
```

## Usage

```bash
git clone <this-repo>
cd cr3bp-earth-moon
pip install -r requirements.txt

# Run the full analysis (integrates all 4 orbits, saves figures to ./figures/)
python scripts/run_analysis.py

# Run the test suite
pytest tests/ -v
```

Or use the package directly:

```python
from cr3bp import integrate, jacobi
from cr3bp.initial_conditions import STATE_C, N_STEPS_C

traj, time = integrate(STATE_C, dt=0.0002, n_steps=N_STEPS_C)
print(jacobi(traj[0]))  # Jacobi constant at t=0
```

## Results

Zero-velocity curves show the energetically forbidden regions associated with
the Jacobi constant — all trajectories remain confined within their allowed
zones, with L1 sitting between Earth and Moon as expected from energy
conservation.

![Zero-velocity curves](figures/02_zero_velocity_curves.png)

Phase portraits confirm coupled in-plane dynamics `(x, vx)`, `(y, vy)` and
bounded, oscillatory out-of-plane motion `(z, vz)`. Orbit C (near L1) shows
complex diamond-shaped structures reflecting its higher energy; Orbit D (near
Moon) shows tighter, more regular loops.

![Phase portraits](figures/03_phase_portraits.png)

Fourier analysis of orbit C's z-component shows dominant frequencies near
0.52, 0.46, and 1.04, indicating structured but multi-frequency vertical
motion.

![Fourier analysis](figures/05_fourier_analysis.png)

## Reference

Szebehely, V. (1967) *Theory of Orbits: The Restricted Problem of Three
Bodies*. Academic Press.

## License

MIT — see [LICENSE](LICENSE).
