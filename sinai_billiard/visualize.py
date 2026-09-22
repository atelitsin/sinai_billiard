"""Static and animated views of billiard trajectories."""
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation

from .geometry import BilliardTable
from .simulation import EventTrajectory

_COLORS = plt.rcParams["axes.prop_cycle"].by_key()["color"]


def plot_trajectory(table: BilliardTable, traj: EventTrajectory, ax=None, color=None, n_points=4000):
    if ax is None:
        _, ax = plt.subplots(figsize=(5, 5))
    table.draw(ax)
    t_max = traj.times[-1]
    if table.periodic:
        # draw wrap-broken segments so lines don't jump across the torus
        pos = traj.positions
        seg_start = 0
        for i in range(1, len(traj.events)):
            if traj.events[i].kind == "wrap":
                ax.plot(pos[seg_start:i + 1, 0], pos[seg_start:i + 1, 1], color=color or _COLORS[0], linewidth=0.8)
                seg_start = i
        ax.plot(pos[seg_start:, 0], pos[seg_start:, 1], color=color or _COLORS[0], linewidth=0.8)
    else:
        times = np.linspace(0, t_max, n_points)
        pos, _ = traj.sample(times)
        ax.plot(pos[:, 0], pos[:, 1], color=color or _COLORS[0], linewidth=0.8)
    ax.plot(*traj.positions[0], "go", markersize=6, zorder=4, label="старт")
    return ax


def animate_trajectories(table: BilliardTable, trajectories, duration=20.0, fps=30, trail=200, figsize=(5, 5)):
    """Animate one or more EventTrajectory objects moving on the same table."""
    if isinstance(trajectories, EventTrajectory):
        trajectories = [trajectories]

    fig, ax = plt.subplots(figsize=figsize)
    table.draw(ax)

    n_frames = int(duration * fps)
    t_max = min(traj.times[-1] for traj in trajectories)
    frame_times = np.linspace(0, min(duration, t_max), n_frames)

    dots = []
    trails = []
    for i, _ in enumerate(trajectories):
        color = _COLORS[i % len(_COLORS)]
        (dot,) = ax.plot([], [], "o", color=color, markersize=7, zorder=5)
        (trail_line,) = ax.plot([], [], "-", color=color, linewidth=1.0, alpha=0.6)
        dots.append(dot)
        trails.append(trail_line)

    def update(frame_idx):
        t = frame_times[frame_idx]
        for traj, dot, trail_line in zip(trajectories, dots, trails):
            pos, _ = traj.state_at(t)
            dot.set_data([pos[0]], [pos[1]])
            t0 = max(0.0, t - trail * (frame_times[1] - frame_times[0]))
            tail_times = np.linspace(t0, t, min(trail, max(2, int((t - t0) / (frame_times[1] - frame_times[0]) + 1))))
            tail_pos, _ = traj.sample(tail_times)
            trail_line.set_data(tail_pos[:, 0], tail_pos[:, 1])
        return dots + trails

    anim = FuncAnimation(fig, update, frames=n_frames, interval=1000 / fps, blit=True)
    return anim


def plot_poincare_disk(table: BilliardTable, traj: EventTrajectory, disk_idx=0, ax=None):
    """Birkhoff-style section on a disk boundary: arclength s vs sin(reflection angle)."""
    if ax is None:
        _, ax = plt.subplots(figsize=(5, 4))
    disk = table.disks[disk_idx]
    s_vals, p_vals = [], []
    for i, e in enumerate(traj.events):
        if e.kind != "disk" or e.ref != disk_idx:
            continue
        n = (e.pos - disk.center) / disk.radius
        theta = np.arctan2(n[1], n[0])
        s = (theta % (2 * np.pi)) * disk.radius
        vel_in = traj.events[i - 1].vel
        vel_in_dir = vel_in / np.linalg.norm(vel_in)
        tangent = np.array([-n[1], n[0]])
        p = vel_in_dir @ tangent
        s_vals.append(s)
        p_vals.append(p)
    ax.scatter(s_vals, p_vals, s=4, alpha=0.6)
    ax.set_xlabel("s (длина дуги на диске)")
    ax.set_ylabel("p = sin(угол падения)")
    ax.set_title("Сечение Пуанкаре (Биркгофовы координаты) на диске")
    return ax


def plot_wall_section(table: BilliardTable, traj: EventTrajectory, wall="bottom", ax=None):
    """Poincare section on a flat wall: x-position vs vx at each bounce off `wall`."""
    if ax is None:
        _, ax = plt.subplots(figsize=(5, 4))
    xs, vxs = [], []
    for i, e in enumerate(traj.events):
        if e.kind != "wall" or e.ref != wall:
            continue
        vel_in = traj.events[i - 1].vel
        xs.append(e.pos[0])
        vxs.append(vel_in[0])
    ax.scatter(xs, vxs, s=4, alpha=0.6)
    ax.set_xlabel("x")
    ax.set_ylabel("v_x (перед столкновением)")
    ax.set_title(f"Сечение Пуанкаре на стенке '{wall}'")
    return ax
