"""
Run the full CR3BP analysis: integrate all four orbits, validate
Jacobi constant conservation, run Poincare + Fourier post-processing,
and save all figures to ./figures/.

Usage:
    python scripts/run_analysis.py
"""

import os
import sys

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from cr3bp import (
    MU, X_EARTH, X_MOON, L1,
    jacobi, integrate, ORBITS,
    zero_velocity_field, poincare_y, poincare_z,
    fft_analysis, dominant_peaks,
)

DT = 0.0002
FIG_DIR = os.path.join(os.path.dirname(__file__), "..", "figures")

CYAN, GREEN, MAGENTA, GOLD = "#00E5FF", "#00FF88", "#FF00FF", "#FFD700"
BLUE, GREY = "#4488FF", "#AAAAAA"


def main():
    os.makedirs(FIG_DIR, exist_ok=True)
    plt.style.use("dark_background")

    print("3D CIRCULAR RESTRICTED THREE-BODY PROBLEM: Earth-Moon System")
    print(f"mu = {MU}")
    print(f"L1 = {L1:.8f}")

    trajectories, times = {}, {}
    for key, orbit in ORBITS.items():
        print(f"Integrating Orbit {key} ({orbit['label']})...")
        traj, t = integrate(orbit["state0"], DT, orbit["n_steps"])
        trajectories[key] = traj
        times[key] = t

    print("\nJacobi constant validation:")
    for key, orbit in ORBITS.items():
        traj = trajectories[key]
        cj = np.array([jacobi(traj[i]) for i in range(0, len(traj), 100)])
        err = np.max(np.abs(cj - cj[0]) / np.abs(cj[0]))
        print(f"  {key} ({orbit['label']}): CJ_0 = {cj[0]:.6f}, max rel error = {err:.2e}")

    # --- Poincare sections ---
    poinc_A = poincare_y(trajectories["A"])
    poinc_B = poincare_y(trajectories["B"])
    poinc_C = poincare_z(trajectories["C"])
    poinc_D = poincare_z(trajectories["D"])
    print(
        f"\nPoincare crossings: "
        f"A={len(poinc_A[0])}, B={len(poinc_B[0])}, "
        f"C={len(poinc_C[0])}, D={len(poinc_D[0])}"
    )

    # --- Jacobi constant time series ---
    step = 200
    cj_series, t_series = {}, {}
    for key in ORBITS:
        traj, t = trajectories[key], times[key]
        cj_series[key] = np.array([jacobi(traj[i]) for i in range(0, len(traj), step)])
        t_series[key] = t[::step]

    # --- Fourier analysis (Orbits C and D only, out-of-plane motion) ---
    fft_data = {}
    for key in ("C", "D"):
        traj = trajectories[key]
        fft_data[key] = {}
        for comp, name in [(0, "x"), (1, "y"), (2, "z")]:
            fr, pw = fft_analysis(traj[:, comp], DT)
            fft_data[key][name] = (fr, pw, dominant_peaks(fr, pw))

    print("\nDominant z-frequencies (Orbit C, 3D L1):")
    for f, _p in fft_data["C"]["z"][2][:3]:
        print(f"  f = {f:.3f}, T = {1/f:.3f}")

    _plot_trajectories_3d(trajectories)
    _plot_zero_velocity(trajectories)
    _plot_phase_portraits(trajectories)
    _plot_jacobi_conservation(t_series, cj_series)
    _plot_fourier(times, trajectories, fft_data)

    print(f"\nDone. Figures saved to {os.path.abspath(FIG_DIR)}")


def _plot_trajectories_3d(trajectories):
    fig = plt.figure(figsize=(18, 7))
    configs = [
        (131, trajectories["C"], "cool", "3D Orbit Near L1", CYAN, True),
        (132, trajectories["D"], "plasma", "3D Orbit Near Moon", MAGENTA, False),
        (133, trajectories["A"], "viridis", "Planar Retrograde (z=0)", GREEN, False),
    ]
    for pos, traj, cmap_name, title, color, show_legend in configs:
        ax = fig.add_subplot(pos, projection="3d")
        colors = getattr(cm, cmap_name)(np.linspace(0, 1, len(traj)))
        step = 30 if pos == 133 else 20
        for i in range(0, len(traj) - 1, step):
            ax.plot(traj[i:i + step + 1, 0], traj[i:i + step + 1, 1], traj[i:i + step + 1, 2],
                    color=colors[i], linewidth=0.3, alpha=0.8)
        ax.scatter([X_EARTH], [0], [0], color=BLUE, s=80, zorder=10, label="Earth" if show_legend else None)
        ax.scatter([X_MOON], [0], [0], color=GREY, s=40, zorder=10, label="Moon" if show_legend else None)
        if show_legend:
            ax.scatter([L1], [0], [0], color=MAGENTA, s=60, marker="x", zorder=10, label="L1")
            ax.legend(fontsize=8)
        ax.set_xlabel("x"); ax.set_ylabel("y"); ax.set_zlabel("z")
        ax.set_title(title, fontsize=13, fontweight="bold", color=color)
    plt.suptitle("3D CR3BP Trajectories: Earth-Moon System", fontsize=16, fontweight="bold", y=1.02)
    plt.tight_layout()
    plt.savefig(os.path.join(FIG_DIR, "01_trajectories_3d.png"), dpi=150, bbox_inches="tight")
    plt.close()


def _plot_zero_velocity(trajectories):
    X, Y, cj_field = zero_velocity_field()
    fig, ax = plt.subplots(figsize=(10, 9))
    levels = np.linspace(2.5, 4.0, 60)
    cf = ax.contourf(X, Y, cj_field, levels=levels, cmap="inferno", alpha=0.9)

    ax.plot(trajectories["A"][:, 0], trajectories["A"][:, 1], color=GREEN, alpha=0.2, linewidth=0.1, label="Retrograde")
    ax.plot(trajectories["B"][:, 0], trajectories["B"][:, 1], color=GOLD, alpha=0.2, linewidth=0.1, label="Transfer")
    ax.plot(trajectories["C"][:, 0], trajectories["C"][:, 1], color=CYAN, alpha=0.3, linewidth=0.15, label="3D L1 (projected)")

    ax.plot(X_EARTH, 0, "o", color=BLUE, markersize=12, zorder=10)
    ax.plot(X_MOON, 0, "o", color=GREY, markersize=7, zorder=10)
    ax.plot(L1, 0, "x", color=GOLD, markersize=10, markeredgewidth=2, zorder=10)
    ax.annotate("L1", (L1, 0), textcoords="offset points", xytext=(0, 10), color=GOLD, fontsize=10)

    cbar = plt.colorbar(cf, ax=ax, shrink=0.8)
    cbar.set_label("Jacobi Constant", fontsize=12)
    ax.set_xlabel("x", fontsize=13); ax.set_ylabel("y", fontsize=13)
    ax.set_title("Zero-Velocity Curves & Trajectory Projections (z=0 plane)", fontsize=14, fontweight="bold")
    ax.set_aspect("equal"); ax.set_xlim([-1.5, 1.5]); ax.set_ylim([-1.5, 1.5])
    leg = ax.legend(fontsize=10, loc="upper left")
    for line in leg.get_lines():
        line.set_alpha(1.0)
        line.set_linewidth(2.0)
    plt.tight_layout()
    plt.savefig(os.path.join(FIG_DIR, "02_zero_velocity_curves.png"), dpi=150, bbox_inches="tight")
    plt.close()


def _plot_phase_portraits(trajectories):
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    for row, (key, label, color, cmap_name) in enumerate([
        ("C", "Orbit C (3D L1)", CYAN, "cool"),
        ("D", "Orbit D (3D Moon)", MAGENTA, "plasma"),
    ]):
        traj = trajectories[key]
        s = 20
        colors = getattr(cm, cmap_name)(np.linspace(0, 1, len(traj[::s])))
        for col_idx, (i1, i2, n1, n2) in enumerate([(0, 3, "x", "vx"), (1, 4, "y", "vy"), (2, 5, "z", "vz")]):
            ax = axes[row, col_idx]
            ax.scatter(traj[::s, i1], traj[::s, i2], c=colors, s=0.3, alpha=0.7)
            ax.set_xlabel(n1); ax.set_ylabel(n2)
            ax.set_title(f"({n1}, {n2}): {label}", fontweight="bold", color=color)
            ax.grid(True, alpha=0.1)
    plt.suptitle("Phase Portraits: Including Out-of-Plane (z, vz)", fontsize=15, fontweight="bold")
    plt.tight_layout()
    plt.savefig(os.path.join(FIG_DIR, "03_phase_portraits.png"), dpi=150, bbox_inches="tight")
    plt.close()


def _plot_jacobi_conservation(t_series, cj_series):
    fig, axes = plt.subplots(2, 1, figsize=(13, 8))
    plot_data = [("A", GREEN, "Planar retro"), ("B", GOLD, "Planar trans"),
                 ("C", CYAN, "3D L1"), ("D", MAGENTA, "3D Moon")]

    for key, color, label in plot_data:
        axes[0].plot(t_series[key], cj_series[key], color=color, linewidth=0.8, alpha=0.8, label=label)
    axes[0].set_xlabel("Time"); axes[0].set_ylabel("CJ")
    axes[0].set_title("Jacobi Constant vs Time", fontweight="bold")
    axes[0].legend(fontsize=9); axes[0].grid(True, alpha=0.15)

    for key, color, label in plot_data:
        err = np.abs(cj_series[key] - cj_series[key][0]) / np.abs(cj_series[key][0])
        axes[1].semilogy(t_series[key], err + 1e-16, color=color, linewidth=0.8, alpha=0.8, label=label)
    axes[1].axhline(y=1e-12, color=GOLD, linestyle="--", alpha=0.4, label="Machine prec.")
    axes[1].set_xlabel("Time"); axes[1].set_ylabel("Relative Error")
    axes[1].set_title("Jacobi Constant Error (Validating RK4)", fontweight="bold")
    axes[1].legend(fontsize=9); axes[1].grid(True, alpha=0.15)
    axes[1].set_ylim([1e-16, 1e-1])

    plt.tight_layout()
    plt.savefig(os.path.join(FIG_DIR, "04_jacobi_conservation.png"), dpi=150, bbox_inches="tight")
    plt.close()


def _plot_fourier(times, trajectories, fft_data):
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    for row, (key, color) in enumerate([("C", CYAN), ("D", MAGENTA)]):
        orbit_name = "3D L1" if key == "C" else "3D Moon"
        for col_idx, comp_name in enumerate(["x", "y", "z"]):
            ax = axes[row, col_idx]
            fr, pw, peaks = fft_data[key][comp_name]
            mask = fr < 15
            ax.semilogy(fr[mask], pw[mask], color=GOLD, linewidth=0.5)
            for f_pk, p_pk in peaks[:3]:
                ax.plot(f_pk, p_pk, "v", color=MAGENTA, markersize=7)
                ax.annotate(f"f={f_pk:.2f}", (f_pk, p_pk), textcoords="offset points",
                            xytext=(8, 8), color=MAGENTA, fontsize=8)
            ax.set_xlabel("Frequency", fontsize=10)
            ax.set_ylabel("PSD", fontsize=10)
            ax.set_title(f"{comp_name}(t): {orbit_name}", fontsize=11, fontweight="bold", color=color)
            ax.grid(True, alpha=0.1)
    plt.suptitle("Fourier Analysis: In-Plane (x,y) vs Out-of-Plane (z) Frequencies", fontsize=15, fontweight="bold")
    plt.tight_layout()
    plt.savefig(os.path.join(FIG_DIR, "05_fourier_analysis.png"), dpi=150, bbox_inches="tight")
    plt.close()


if __name__ == "__main__":
    main()
