"""Helpers for building tables, running parameter sweeps, and exporting data to CSV."""
import numpy as np
import pandas as pd

from .geometry import BilliardTable, Disk
from .simulation import EventTrajectory, simulate


def build_table(kind: str, width=1.0, height=1.0, disk_radius=0.25, disk_center=None, disks=None):
    """kind in {'sinai_square', 'sinai_torus', 'multi_disk', 'integrable_square', 'integrable_torus'}."""
    if disk_center is None:
        disk_center = (width / 2, height / 2)

    if kind == "sinai_square":
        return BilliardTable(width, height, [Disk(disk_center, disk_radius)], periodic=False)
    if kind == "sinai_torus":
        return BilliardTable(width, height, [Disk(disk_center, disk_radius)], periodic=True)
    if kind == "multi_disk":
        return BilliardTable(width, height, disks or [], periodic=False)
    if kind == "multi_disk_torus":
        return BilliardTable(width, height, disks or [], periodic=True)
    if kind == "integrable_square":
        return BilliardTable(width, height, [], periodic=False)
    if kind == "integrable_torus":
        return BilliardTable(width, height, [], periodic=True)
    raise ValueError(f"unknown table kind: {kind}")


def build_table_equal_area(kind: str, aspect: float = 1.0, area: float = 1.0, radius_fraction: float = 0.15):
    """Rectangle parametrized by aspect = height/width, with the table's *area* and the
    disk's *area fraction* held fixed as aspect varies.

    A plain build_table(width=1, height=h, disk_radius=R) sweep changes the table's total
    area AND the disk's relative size at the same time as h varies, so aspect ratios aren't
    directly comparable to each other (e.g. aspect=2 and aspect=0.5 are not mirror images of
    the same system -- they're differently-sized tables with a differently-sized obstacle).
    Fixing width*height = area and radius = radius_fraction * sqrt(area) removes both
    confounds: aspect=r and aspect=1/r are then exact mirror images (transpose symmetry),
    so e.g. lambda(r) == lambda(1/r) and D_x(r) == D_y(1/r) up to sampling noise.
    """
    width = np.sqrt(area / aspect)
    height = np.sqrt(area * aspect)
    radius = radius_fraction * np.sqrt(area)
    return build_table(kind, width=width, height=height, disk_radius=radius)


def random_initial_conditions(table: BilliardTable, n, rng=None, margin=1e-3):
    """n random starting positions (outside every disk) with random unit velocities."""
    rng = rng or np.random.default_rng()
    positions, velocities = [], []
    while len(positions) < n:
        p = rng.uniform([margin, margin], [table.width - margin, table.height - margin])
        if any(np.linalg.norm(p - d.center) < d.radius + margin for d in table.disks):
            continue
        angle = rng.uniform(0, 2 * np.pi)
        positions.append(p)
        velocities.append([np.cos(angle), np.sin(angle)])
    return np.array(positions), np.array(velocities)


def run_ensemble(table: BilliardTable, n_particles, n_events=3000, t_max=None, seed=0):
    rng = np.random.default_rng(seed)
    positions, velocities = random_initial_conditions(table, n_particles, rng)
    return [simulate(table, p, v, n_events=n_events, t_max=t_max) for p, v in zip(positions, velocities)]


def sweep_parameter(param_name, values, table_kwargs, metric_fn, table_builder=build_table):
    """Vary one table_builder kwarg over `values`, call metric_fn(table) -> dict, collect a DataFrame."""
    rows = []
    for v in values:
        table = table_builder(**{**table_kwargs, param_name: v})
        row = {param_name: v, **metric_fn(table)}
        rows.append(row)
    return pd.DataFrame(rows)


def trajectory_to_dataframe(traj: EventTrajectory) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "t": traj.times,
            "x": traj.positions[:, 0],
            "y": traj.positions[:, 1],
            "vx": traj.velocities[:, 0],
            "vy": traj.velocities[:, 1],
            "event": [e.kind for e in traj.events],
            "event_ref": [str(e.ref) for e in traj.events],
        }
    )


def export_trajectory_csv(traj: EventTrajectory, path):
    df = trajectory_to_dataframe(traj)
    df.to_csv(path, index=False)
    return df
