"""
Post-processing utilities: Poincare sections, zero-velocity fields,
and Fourier analysis of trajectory components.

The Fourier analysis links this project to the discrete Fourier
transform topic covered separately in ME2 Computing: a Hanning window
is applied before the FFT to reduce spectral leakage, and the spectra
are used to identify dominant in-plane vs out-of-plane frequencies.
"""

import numpy as np

from .dynamics import X_EARTH, X_MOON, MU


def zero_velocity_field(x_range=(-1.5, 1.5), y_range=(-1.5, 1.5), n=800, mu=MU):
    """Zero-velocity (Jacobi constant) field over the z=0 plane."""
    xg = np.linspace(*x_range, n)
    yg = np.linspace(*y_range, n)
    X, Y = np.meshgrid(xg, yg)
    R1 = np.sqrt((X - X_EARTH) ** 2 + Y ** 2)
    R2 = np.sqrt((X - X_MOON) ** 2 + Y ** 2)
    CJ_field = (X ** 2 + Y ** 2) + 2 * (1 - mu) / R1 + 2 * mu / R2
    return X, Y, CJ_field


def poincare_z(traj):
    """z = 0 crossings with vz > 0 -- used for 3D (out-of-plane) orbits."""
    cx, cy = [], []
    for i in range(len(traj) - 1):
        if traj[i, 2] * traj[i + 1, 2] < 0 and traj[i + 1, 5] > 0:
            f = -traj[i, 2] / (traj[i + 1, 2] - traj[i, 2])
            cx.append(traj[i, 0] + f * (traj[i + 1, 0] - traj[i, 0]))
            cy.append(traj[i, 1] + f * (traj[i + 1, 1] - traj[i, 1]))
    return np.array(cx), np.array(cy)


def poincare_y(traj):
    """y = 0 crossings with vy > 0 -- used for planar orbits."""
    cx, cvx = [], []
    for i in range(len(traj) - 1):
        if traj[i, 1] * traj[i + 1, 1] < 0 and traj[i + 1, 4] > 0:
            f = -traj[i, 1] / (traj[i + 1, 1] - traj[i, 1])
            cx.append(traj[i, 0] + f * (traj[i + 1, 0] - traj[i, 0]))
            cvx.append(traj[i, 3] + f * (traj[i + 1, 3] - traj[i, 3]))
    return np.array(cx), np.array(cvx)


def fft_analysis(signal, dt):
    """Windowed FFT (Hanning) of a single trajectory component."""
    sig = signal - np.mean(signal)
    n = len(sig)
    window = np.hanning(n)
    fv = np.fft.fft(sig * window)
    fr = np.fft.fftfreq(n, d=dt)
    positive = fr > 0
    return fr[positive], np.abs(fv[positive]) ** 2


def dominant_peaks(freqs, power, max_freq=15, n_peaks=5):
    """Top local-maximum frequency peaks below max_freq, by power."""
    mask = freqs < max_freq
    f, p = freqs[mask], power[mask]
    peaks = [
        (f[i], p[i])
        for i in range(1, len(p) - 1)
        if p[i] > p[i - 1] and p[i] > p[i + 1]
    ]
    peaks.sort(key=lambda pair: pair[1], reverse=True)
    return peaks[:n_peaks]
