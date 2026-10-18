"""
Billiard table geometry: walls (reflecting or periodic) and circular scatterers.

Collisions are resolved analytically (exact quadratic/linear roots), so trajectories
are exact straight-line segments between events -- no integration error accumulates,
which matters for chaos diagnostics like Lyapunov exponents.
"""
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import numpy as np

_EPS = 1e-9


@dataclass
class Disk:
    center: np.ndarray
    radius: float

    def __post_init__(self):
        self.center = np.asarray(self.center, dtype=float)


@dataclass
class BilliardTable:
    """
    Rectangular domain [0, width] x [0, height] with optional circular obstacles.

    periodic=False -> classic Sinai billiard: walls reflect specularly.
    periodic=True  -> torus (original Sinai construction): walls wrap the particle.

    Disks must lie fully inside the domain, away from the boundary, so that no
    disk touches a wall or its periodic image; this keeps collision detection to
    a single fundamental domain.
    """

    width: float
    height: float
    disks: List[Disk] = field(default_factory=list)
    periodic: bool = False

    def __post_init__(self):
        for d in self.disks:
            cx, cy = d.center
            if not (d.radius < cx < self.width - d.radius and d.radius < cy < self.height - d.radius):
                raise ValueError("disk must lie strictly inside the domain, clear of the walls")

    # ---- wall events -------------------------------------------------
    def wall_event(self, pos: np.ndarray, vel: np.ndarray) -> Tuple[float, Optional[str]]:
        best_t, best_wall = np.inf, None
        x, y = pos
        vx, vy = vel
        if vx > 0:
            t = (self.width - x) / vx
            if t > _EPS and t < best_t:
                best_t, best_wall = t, "right"
        elif vx < 0:
            t = (0.0 - x) / vx
            if t > _EPS and t < best_t:
                best_t, best_wall = t, "left"
        if vy > 0:
            t = (self.height - y) / vy
            if t > _EPS and t < best_t:
                best_t, best_wall = t, "top"
        elif vy < 0:
            t = (0.0 - y) / vy
            if t > _EPS and t < best_t:
                best_t, best_wall = t, "bottom"
        return best_t, best_wall

    def apply_wall(self, pos: np.ndarray, vel: np.ndarray, wall_id: str) -> Tuple[np.ndarray, np.ndarray]:
        pos = pos.copy()
        vel = vel.copy()
        if self.periodic:
            if wall_id == "right":
                pos[0] = 0.0
            elif wall_id == "left":
                pos[0] = self.width
            elif wall_id == "top":
                pos[1] = 0.0
            elif wall_id == "bottom":
                pos[1] = self.height
        else:
            if wall_id in ("right", "left"):
                pos[0] = self.width if wall_id == "right" else 0.0
                vel[0] = -vel[0]
            else:
                pos[1] = self.height if wall_id == "top" else 0.0
                vel[1] = -vel[1]
        return pos, vel

    # ---- disk events ---------------------------------------------------
    def disk_event(self, pos: np.ndarray, vel: np.ndarray) -> Tuple[float, Optional[int]]:
        best_t, best_idx = np.inf, None
        for i, disk in enumerate(self.disks):
            d = pos - disk.center
            a = vel @ vel
            b = 2.0 * (d @ vel)
            c = d @ d - disk.radius ** 2
            disc = b * b - 4 * a * c
            if disc <= 0:
                continue
            sq = np.sqrt(disc)
            t1 = (-b - sq) / (2 * a)
            t2 = (-b + sq) / (2 * a)
            for t in (t1, t2):
                if t > _EPS and t < best_t:
                    best_t, best_idx = t, i
        return best_t, best_idx

    def apply_disk(self, pos: np.ndarray, vel: np.ndarray, disk_idx: int) -> np.ndarray:
        disk = self.disks[disk_idx]
        n = (pos - disk.center)
        n = n / np.linalg.norm(n)
        return vel - 2.0 * (vel @ n) * n

    def draw(self, ax, disk_color="#c0392b", wall_color="black"):
        ax.plot(
            [0, self.width, self.width, 0, 0],
            [0, 0, self.height, self.height, 0],
            color=wall_color,
            linestyle="--" if self.periodic else "-",
            linewidth=1.5,
        )
        for disk in self.disks:
            circle = plt_circle(disk.center, disk.radius, disk_color)
            ax.add_patch(circle)
        ax.set_xlim(-0.05 * self.width, 1.05 * self.width)
        ax.set_ylim(-0.05 * self.height, 1.05 * self.height)
        ax.set_aspect("equal")


def plt_circle(center, radius, color):
    from matplotlib.patches import Circle

    return Circle(tuple(center), radius, color=color, zorder=3)
