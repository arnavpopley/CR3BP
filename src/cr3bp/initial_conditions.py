"""
Initial conditions for the four orbit types analysed in this study,
spanning planar and fully 3D (out-of-plane) motion.

State vector convention: [x, y, z, vx, vy, vz]
"""

import numpy as np

from .dynamics import MU, find_lagrange

L1 = find_lagrange(MU, 0.8)
L2 = find_lagrange(MU, 1.15)

# Orbit A: planar retrograde orbit around Earth (z = 0 throughout)
STATE_A = np.array([0.4, 0.0, 0.0, 0.0, -1.2, 0.0])
N_STEPS_A = 300_000

# Orbit B: planar transfer orbit (z = 0 throughout)
STATE_B = np.array([0.4, 0.0, 0.0, 0.0, 1.05, 0.0])
N_STEPS_B = 200_000

# Orbit C: 3D halo-like orbit near L1
STATE_C = np.array([L1 - 0.02, 0.0, 0.04, 0.0, -0.15, 0.0])
N_STEPS_C = 250_000

# Orbit D: 3D orbit near the Moon with z-oscillation
STATE_D = np.array([0.92, 0.0, 0.03, 0.0, 0.5, 0.05])
N_STEPS_D = 250_000

ORBITS = {
    "A": {"label": "Planar retrograde", "state0": STATE_A, "n_steps": N_STEPS_A},
    "B": {"label": "Planar transfer", "state0": STATE_B, "n_steps": N_STEPS_B},
    "C": {"label": "3D near L1", "state0": STATE_C, "n_steps": N_STEPS_C},
    "D": {"label": "3D near Moon", "state0": STATE_D, "n_steps": N_STEPS_D},
}
