import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from cr3bp import MU, find_lagrange, integrate, jacobi
from cr3bp.initial_conditions import STATE_A, STATE_C


def test_lagrange_point_l1_between_earth_and_moon():
    l1 = find_lagrange(MU, 0.8)
    x_earth = -MU
    x_moon = 1.0 - MU
    assert x_earth < l1 < x_moon


def test_planar_orbit_stays_in_plane():
    """Orbit A starts with z=0 and vz=0, so it should remain exactly
    planar for the whole integration (validates the 3D code correctly
    reduces to the planar case)."""
    traj, _ = integrate(STATE_A, dt=0.0002, n_steps=2000)
    assert np.max(np.abs(traj[:, 2])) < 1e-10


def test_jacobi_constant_is_conserved():
    """The Jacobi constant is analytically conserved in the CR3BP;
    relative drift under RK4 should stay small over a short window."""
    traj, _ = integrate(STATE_C, dt=0.0002, n_steps=5000)
    cj = np.array([jacobi(traj[i]) for i in range(0, len(traj), 50)])
    rel_err = np.max(np.abs(cj - cj[0]) / np.abs(cj[0]))
    assert rel_err < 1e-6


def test_3d_orbit_has_out_of_plane_motion():
    """Orbit C has nonzero initial z, so it should show real
    out-of-plane (z != 0) motion, unlike the planar orbits."""
    traj, _ = integrate(STATE_C, dt=0.0002, n_steps=2000)
    assert np.max(np.abs(traj[:, 2])) > 1e-3
