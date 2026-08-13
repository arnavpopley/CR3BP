"""
Classical fourth-order Runge-Kutta (RK4) integrator for the CR3BP.

RK4 was chosen because the CR3BP is an initial value problem advanced
over many time steps, so accumulated numerical error matters. Stability
and accuracy are validated separately via Jacobi constant conservation
(see cr3bp.dynamics.jacobi and the stability analysis in the notebook).
"""

import numpy as np

from .dynamics import eom


def rk4_step(state, dt, mu=None):
    """Advance one CR3BP state vector by a single RK4 step of size dt."""
    kwargs = {} if mu is None else {"mu": mu}
    k1 = eom(state, **kwargs)
    k2 = eom(state + 0.5 * dt * k1, **kwargs)
    k3 = eom(state + 0.5 * dt * k2, **kwargs)
    k4 = eom(state + dt * k3, **kwargs)
    return state + (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)


def integrate(state0, dt, n_steps, mu=None):
    """
    Integrate a CR3BP trajectory forward from state0.

    Returns:
        traj: (n_steps + 1, 6) array of state vectors
        time: (n_steps + 1,) array of time values
    """
    traj = np.zeros((n_steps + 1, 6))
    traj[0] = state0
    for i in range(n_steps):
        traj[i + 1] = rk4_step(traj[i], dt, mu=mu)
    time = np.arange(n_steps + 1) * dt
    return traj, time
