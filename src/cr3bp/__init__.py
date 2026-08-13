"""
cr3bp: A numerical toolkit for the 3D Circular Restricted Three-Body
Problem (CR3BP), applied to the Earth-Moon system.

Originally developed for ME2 Maths & Computing coursework at
Imperial College London (2025-2026), refactored into a reusable
package.
"""

from .dynamics import MU, X_EARTH, X_MOON, eff_potential, eom, jacobi, find_lagrange
from .integrator import rk4_step, integrate
from .initial_conditions import ORBITS, L1, L2
from .postprocessing import (
    zero_velocity_field,
    poincare_y,
    poincare_z,
    fft_analysis,
    dominant_peaks,
)

__all__ = [
    "MU",
    "X_EARTH",
    "X_MOON",
    "eff_potential",
    "eom",
    "jacobi",
    "find_lagrange",
    "rk4_step",
    "integrate",
    "ORBITS",
    "L1",
    "L2",
    "zero_velocity_field",
    "poincare_y",
    "poincare_z",
    "fft_analysis",
    "dominant_peaks",
]

__version__ = "1.0.0"
