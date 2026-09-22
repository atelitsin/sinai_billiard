"""Sanity checks for the billiard core: run with `python tests/test_physics.py`."""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from sinai_billiard import BilliardTable, Disk, simulate
from sinai_billiard.analysis import lyapunov_exponent


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


if __name__ == "__main__":
    tests = [v for k, v in list(globals().items()) if k.startswith("test_")]
    for t in tests:
        t()
        print(f"OK  {t.__name__}")
    print(f"\n{len(tests)} physics checks passed.")
