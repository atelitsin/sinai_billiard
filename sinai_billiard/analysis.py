"""Chaos diagnostics: Lyapunov exponent, diffusion/MSD, collision statistics, ergodicity."""
import numpy as np

from .geometry import BilliardTable
from .simulation import EventTrajectory, simulate


def lyapunov_exponent(
    table: BilliardTable,
    pos0,
    vel0,
    eps: float = 1e-8,
    t_max: float = 200.0,
    dt_rescale: float = 1.0,
    n_events_per_segment: int = 4000,
    seed: int = 0,
):
    """Benettin-style estimate: track a shadow trajectory eps away, periodically
    rescale its separation back to eps (to avoid saturation at the table size),
    and average ln(separation / eps) over time. Returns (lambda, times, running_estimate).
    """
    rng = np.random.default_rng(seed)
    pos_ref = np.asarray(pos0, dtype=float).copy()
    vel_ref = np.asarray(vel0, dtype=float)
    vel_ref = vel_ref / np.linalg.norm(vel_ref)

    direction = rng.normal(size=2)
    direction /= np.linalg.norm(direction)
    pos_pert = pos_ref + eps * direction
    vel_pert = vel_ref.copy()

    n_segments = max(1, int(t_max / dt_rescale))
    log_sum = 0.0
    t_elapsed = 0.0
    times, running = [], []

    for _ in range(n_segments):
        traj_ref = simulate(table, pos_ref, vel_ref, n_events=n_events_per_segment, t_max=dt_rescale)
        traj_pert = simulate(table, pos_pert, vel_pert, n_events=n_events_per_segment, t_max=dt_rescale)
        pos_ref, vel_ref = traj_ref.state_at(dt_rescale)
        pos_pert, vel_pert = traj_pert.state_at(dt_rescale)

        sep_vec = pos_pert - pos_ref
        d = np.linalg.norm(sep_vec)
        if d == 0:
            d = eps
        log_sum += np.log(d / eps)
        t_elapsed += dt_rescale
        times.append(t_elapsed)
        running.append(log_sum / t_elapsed)

        pos_pert = pos_ref + (sep_vec / d) * eps
        vel_pert = vel_ref.copy()

    return log_sum / t_elapsed, np.array(times), np.array(running)


def free_path_lengths(traj: EventTrajectory):
    """Length of every straight flight segment between consecutive events."""
    speed = np.linalg.norm(traj.velocities[0])
    return np.diff(traj.times) * speed


def collision_angles(traj: EventTrajectory, table: BilliardTable, disk_idx=None):
    """Angle between the incoming velocity and the inward normal, at each disk collision."""
    angles = []
    for i, e in enumerate(traj.events):
        if e.kind != "disk" or (disk_idx is not None and e.ref != disk_idx):
            continue
        disk = table.disks[e.ref]
        n = (e.pos - disk.center) / disk.radius
        vel_in = traj.events[i - 1].vel
        vel_in_dir = vel_in / np.linalg.norm(vel_in)
        cos_theta = np.clip(-vel_in_dir @ n, -1.0, 1.0)
        angles.append(np.arccos(cos_theta))
    return np.array(angles)


def position_histogram(table: BilliardTable, traj: EventTrajectory, bins=50, n_samples=200000):
    times = np.linspace(0, traj.times[-1], min(n_samples, max(len(traj.events) * 5, 2000)))
    pos, _ = traj.sample(times)
    H, xedges, yedges = np.histogram2d(
        pos[:, 0], pos[:, 1], bins=bins, range=[[0, table.width], [0, table.height]]
    )
    return H, xedges, yedges


def uniformity_deviation(H):
    """Coefficient of variation of the visitation histogram; 0 = perfectly uniform (ergodic)."""
    mean = H.mean()
    return H.std() / mean if mean > 0 else np.nan


def velocity_direction_autocorrelation(traj: EventTrajectory, dt=0.05, max_lag=20.0):
    duration = traj.times[-1]
    times = np.arange(0, duration, dt)
    _, vel = traj.sample(times)
    theta = np.arctan2(vel[:, 1], vel[:, 0])
    u = np.stack([np.cos(theta), np.sin(theta)], axis=1)
    n = len(u)
    max_lag_steps = min(int(max_lag / dt), n - 1)
    lags = np.arange(0, max_lag_steps)
    corr = np.array([1.0 if l == 0 else np.mean(np.sum(u[: n - l] * u[l:], axis=1)) for l in lags])
    return lags * dt, corr


def _unwrapped_displacements(table: BilliardTable, trajectories, times):
    """Per-trajectory (x, y) displacement from t=0, one (len(times), 2) array per trajectory."""
    out = []
    for traj in trajectories:
        unwrapped = traj.unwrapped_positions(table)
        idx = np.clip(np.searchsorted(traj.times, times, side="right") - 1, 0, len(traj.times) - 1)
        seg_dt = times - traj.times[idx]
        pos = unwrapped[idx] + traj.velocities[idx] * seg_dt[:, None]
        out.append(pos - unwrapped[0])
    return out


def mean_square_displacement(table: BilliardTable, trajectories, times):
    """Ensemble-averaged MSD(t) = <dx^2 + dy^2> using unwrapped positions (meaningful on the torus)."""
    displacements = _unwrapped_displacements(table, trajectories, times)
    msd = sum(np.sum(disp ** 2, axis=1) for disp in displacements)
    return msd / len(trajectories)


def mean_square_displacement_xy(table: BilliardTable, trajectories, times):
    """Component-wise ensemble MSD: (<dx^2>(t), <dy^2>(t)) -- reveals anisotropic diffusion
    on non-square domains, where transport along the two axes need not be equal."""
    displacements = _unwrapped_displacements(table, trajectories, times)
    msd_x = sum(disp[:, 0] ** 2 for disp in displacements) / len(trajectories)
    msd_y = sum(disp[:, 1] ** 2 for disp in displacements) / len(trajectories)
    return msd_x, msd_y


def diffusion_coefficient(times, msd, fit_start_frac=0.3):
    """Linear fit of MSD(t) = 4*D*t over the (later) linear regime; returns D and the fit."""
    start = int(len(times) * fit_start_frac)
    slope, intercept = np.polyfit(times[start:], msd[start:], 1)
    return slope / 4.0, (slope, intercept)


def diffusion_coefficients_xy(times, msd_x, msd_y, fit_start_frac=0.3):
    """Per-axis diffusion coefficients from MSD_x(t) = 2*D_x*t, MSD_y(t) = 2*D_y*t."""
    start = int(len(times) * fit_start_frac)
    slope_x, _ = np.polyfit(times[start:], msd_x[start:], 1)
    slope_y, _ = np.polyfit(times[start:], msd_y[start:], 1)
    return slope_x / 2.0, slope_y / 2.0
