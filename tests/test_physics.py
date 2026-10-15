"""Sanity checks for the billiard core: run with `python tests/test_physics.py`."""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from sinai_billiard import BilliardTable, Disk, simulate
from sinai_billiard.analysis import (
    diffusion_coefficients_xy,
    lyapunov_exponent,
    mean_square_displacement_xy,
)
from sinai_billiard.experiments import build_table_equal_area, run_ensemble


def test_speed_conserved():
    table = BilliardTable(1.0, 1.0, [Disk((0.5, 0.5), 0.25)], periodic=False)
    traj = simulate(table, [0.1, 0.1], [0.6, 0.8], n_events=5000)
    speeds = np.linalg.norm(traj.velocities, axis=1)
    assert np.allclose(speeds, speeds[0], atol=1e-9), speeds.max() - speeds.min()


def test_never_inside_disk():
    table = BilliardTable(1.0, 1.0, [Disk((0.5, 0.5), 0.25)], periodic=False)
    traj = simulate(table, [0.05, 0.05], [1.0, 0.3], n_events=5000)
    times = np.linspace(0, traj.times[-1], 20000)
    pos, _ = traj.sample(times)
    d = np.linalg.norm(pos - np.array([0.5, 0.5]), axis=1)
    assert d.min() > 0.25 - 1e-6, d.min()


def test_never_outside_walls():
    table = BilliardTable(1.0, 1.0, [Disk((0.5, 0.5), 0.25)], periodic=False)
    traj = simulate(table, [0.05, 0.05], [1.0, 0.3], n_events=5000)
    times = np.linspace(0, traj.times[-1], 20000)
    pos, _ = traj.sample(times)
    assert pos[:, 0].min() > -1e-6 and pos[:, 0].max() < 1 + 1e-6
    assert pos[:, 1].min() > -1e-6 and pos[:, 1].max() < 1 + 1e-6


def test_torus_wraps_and_conserves_speed():
    table = BilliardTable(1.0, 1.0, [Disk((0.5, 0.5), 0.2)], periodic=True)
    traj = simulate(table, [0.05, 0.05], [0.9, 0.4], n_events=5000)
    speeds = np.linalg.norm(traj.velocities, axis=1)
    assert np.allclose(speeds, speeds[0], atol=1e-9)
    assert any(e.kind == "wrap" for e in traj.events)


def test_lyapunov_positive_for_sinai():
    table = BilliardTable(1.0, 1.0, [Disk((0.5, 0.5), 0.25)], periodic=False)
    lam, _, _ = lyapunov_exponent(table, [0.1, 0.15], [0.37, 0.93], t_max=60.0, dt_rescale=1.0)
    assert lam > 0, lam


def test_lyapunov_near_zero_for_integrable_square():
    table = BilliardTable(1.0, 1.0, [], periodic=False)
    lam, _, _ = lyapunov_exponent(table, [0.1, 0.15], [0.37, 0.93], t_max=60.0, dt_rescale=1.0)
    assert abs(lam) < 0.05, lam


def test_rectangular_table_is_valid():
    table = BilliardTable(2.5, 1.0, [Disk((1.25, 0.5), 0.15)], periodic=False)
    traj = simulate(table, [0.1, 0.1], [0.6, 0.8], n_events=5000)
    speeds = np.linalg.norm(traj.velocities, axis=1)
    assert np.allclose(speeds, speeds[0], atol=1e-9)
    assert traj.positions[:, 0].max() < 2.5 + 1e-6


def test_elongated_torus_diffuses_anisotropically():
    # a non-square cell should transport particles at different rates along each axis
    # (which axis wins depends on the unobstructed-channel widths, not just the side
    # lengths -- see the aspect-ratio notebook section, this is not a simple "longer
    # side = faster" relationship)
    table = BilliardTable(2.5, 1.0, [Disk((1.25, 0.5), 0.15)], periodic=True)
    ensemble = run_ensemble(table, n_particles=30, n_events=4000, t_max=200.0, seed=5)
    times = np.linspace(0, 160.0, 100)
    msd_x, msd_y = mean_square_displacement_xy(table, ensemble, times)
    D_x, D_y = diffusion_coefficients_xy(times, msd_x, msd_y)
    assert D_x > 0 and D_y > 0
    assert abs(D_x - D_y) / max(D_x, D_y) > 0.1, (D_x, D_y)


def test_width_height_transpose_is_an_exact_symmetry():
    # swapping width<->height and x<->y, vx<->vy must give the literally same dynamics
    # (this is the symmetry that a naive "fix width, vary height" aspect-ratio sweep breaks,
    # because it silently also changes the table's area and the disk's area fraction)
    table1 = BilliardTable(2.0, 1.0, [Disk((1.0, 0.5), 0.2)], periodic=False)
    table2 = BilliardTable(1.0, 2.0, [Disk((0.5, 1.0), 0.2)], periodic=False)
    pos0, vel0 = np.array([0.3, 0.2]), np.array([0.6, 0.8])
    traj1 = simulate(table1, pos0, vel0, n_events=3000)
    traj2 = simulate(table2, pos0[::-1], vel0[::-1], n_events=3000)
    assert np.allclose(traj1.times, traj2.times)
    assert np.allclose(traj1.positions, traj2.positions[:, ::-1], atol=1e-9)
    assert np.allclose(traj1.velocities, traj2.velocities[:, ::-1], atol=1e-9)


def test_equal_area_aspect_helper_is_symmetric():
    t1 = build_table_equal_area("sinai_square", aspect=2.0, area=1.0, radius_fraction=0.15)
    t2 = build_table_equal_area("sinai_square", aspect=0.5, area=1.0, radius_fraction=0.15)
    assert np.isclose(t1.width, t2.height)
    assert np.isclose(t1.height, t2.width)
    assert np.isclose(t1.disks[0].radius, t2.disks[0].radius)
    assert np.isclose(t1.width * t1.height, 1.0)
    assert np.isclose(t2.width * t2.height, 1.0)


def test_lyapunov_matches_under_equal_area_aspect_mirror():
    # aspect=r and aspect=1/r are mirror images of the *same* system, so lambda must match
    t1 = build_table_equal_area("sinai_square", aspect=2.0)
    t2 = build_table_equal_area("sinai_square", aspect=0.5)
    lam1, _, _ = lyapunov_exponent(t1, pos0=[0.05, 0.05], vel0=[0.53, 0.85], t_max=80.0)
    lam2, _, _ = lyapunov_exponent(t2, pos0=[0.05, 0.05], vel0=[0.85, 0.53], t_max=80.0)
    assert abs(lam1 - lam2) / max(lam1, lam2) < 0.15, (lam1, lam2)


if __name__ == "__main__":
    tests = [v for k, v in list(globals().items()) if k.startswith("test_")]
    for t in tests:
        t()
        print(f"OK  {t.__name__}")
    print(f"\n{len(tests)} physics checks passed.")
