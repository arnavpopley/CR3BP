"""
Core dynamics for the 3D Circular Restricted Three-Body Problem (CR3BP).

Models a spacecraft of negligible mass moving under the combined
gravitational attraction of two primaries (Earth and Moon) that orbit
their common centre of mass in circular orbits, in a rotating,
non-dimensional frame.

Reference:
    Szebehely, V. (1967) Theory of Orbits: The Restricted Problem of
    Three Bodies. Academic Press.
"""

import numpy as np

# Earth-Moon mass ratio: mu = M_Moon / (M_Earth + M_Moon)
MU = 0.01215058

X_EARTH = -MU
X_MOON = 1.0 - MU


def eff_potential(x, y, z, mu=MU):
    """
    Effective potential in the rotating frame.

        Omega(x, y, z) = 0.5*(x^2 + y^2) + (1 - mu)/r1 + mu/r2

    Note: z does NOT appear in the centrifugal term (only x, y),
    since the rotation axis is parallel to z.
    """
    r1 = np.sqrt((x - X_EARTH) ** 2 + y ** 2 + z ** 2)
    r2 = np.sqrt((x - X_MOON) ** 2 + y ** 2 + z ** 2)
    return 0.5 * (x ** 2 + y ** 2) + (1 - mu) / r1 + mu / r2


def eom(state, mu=MU):
    """
    3D CR3BP equations of motion in the rotating frame.

    Input:  state = [x, y, z, vx, vy, vz]
    Output: [dx/dt, dy/dt, dz/dt, dvx/dt, dvy/dt, dvz/dt]

        x'' =  2*vy + x - (1-mu)*(x+mu)/r1^3 - mu*(x-1+mu)/r2^3
        y'' = -2*vx + y - (1-mu)*y/r1^3       - mu*y/r2^3
        z'' =           - (1-mu)*z/r1^3        - mu*z/r2^3

    The z-equation has no Coriolis (2v) or centrifugal (+z) terms,
    since the rotation axis is parallel to z -- only gravity acts
    out of plane.
    """
    x, y, z, vx, vy, vz = state

    r1 = np.sqrt((x - X_EARTH) ** 2 + y ** 2 + z ** 2)
    r2 = np.sqrt((x - X_MOON) ** 2 + y ** 2 + z ** 2)
    r1 = max(r1, 1e-12)
    r2 = max(r2, 1e-12)

    ax = (2.0 * vy + x
          - (1 - mu) * (x - X_EARTH) / r1 ** 3
          - mu * (x - X_MOON) / r2 ** 3)

    ay = (-2.0 * vx + y
          - (1 - mu) * y / r1 ** 3
          - mu * y / r2 ** 3)

    az = (-(1 - mu) * z / r1 ** 3
          - mu * z / r2 ** 3)

    return np.array([vx, vy, vz, ax, ay, az])


def jacobi(state, mu=MU):
    """
    Jacobi constant: C_J = 2*Omega - v^2

    Analytically conserved for the CR3BP -- used here to validate
    numerical integration accuracy and stability. Describes total
    mechanical energy in the rotating frame; regions where v^2 < 0
    are energetically forbidden to the spacecraft.
    """
    x, y, z, vx, vy, vz = state
    omega = eff_potential(x, y, z, mu=mu)
    return 2.0 * omega - (vx ** 2 + vy ** 2 + vz ** 2)


def find_lagrange(mu, x0, tol=1e-14, max_iter=500):
    """
    Newton-Raphson solver for a collinear Lagrange point (L1 or L2)
    near a starting guess x0, along the Earth-Moon line.
    """
    x = x0
    for _ in range(max_iter):
        r1 = abs(x - X_EARTH)
        r2 = abs(x - X_MOON)
        f = x - (1 - mu) * (x - X_EARTH) / r1 ** 3 - mu * (x - X_MOON) / r2 ** 3
        df = 1 + 2 * (1 - mu) / r1 ** 3 + 2 * mu / r2 ** 3
        x_next = x - f / df
        if abs(x_next - x) < tol:
            return x_next
        x = x_next
    return x
