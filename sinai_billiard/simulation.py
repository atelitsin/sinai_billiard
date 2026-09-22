"""Event-driven billiard dynamics: exact straight-line flight between collisions."""
from dataclasses import dataclass
from typing import List, Optional

import numpy as np

from .geometry import BilliardTable


@dataclass
class Event:
    t: float
    pos: np.ndarray
    vel: np.ndarray
    kind: str  # "start" | "wall" | "wrap" | "disk"
    ref: Optional[object]  # wall name or disk index


class EventTrajectory:
    """A sequence of events plus O(log n) state lookup at arbitrary time t."""

    def __init__(self, events: List[Event]):
        self.events = events
        self.times = np.array([e.t for e in events])
        self.positions = np.array([e.pos for e in events])
        self.velocities = np.array([e.vel for e in events])

    def state_at(self, t):
        """Position and velocity at time t (0 <= t <= self.times[-1]), by linear flight."""
        idx = np.searchsorted(self.times, t, side="right") - 1
        idx = np.clip(idx, 0, len(self.events) - 1)
        dt = t - self.times[idx]
        pos = self.positions[idx] + self.velocities[idx] * dt
        return pos, self.velocities[idx]

    def sample(self, times):
        pos = np.empty((len(times), 2))
        vel = np.empty((len(times), 2))
        for i, t in enumerate(times):
            pos[i], vel[i] = self.state_at(t)
        return pos, vel

    def unwrapped_positions(self, table: BilliardTable):
        """Cumulative displacement ignoring periodic wraps (for diffusion/MSD)."""
        pos = self.positions.copy()
        offset = np.zeros(2)
        out = np.empty_like(pos)
        out[0] = pos[0]
        for i in range(1, len(pos)):
            e = self.events[i]
            step = pos[i] - pos[i - 1]
            if e.kind == "wrap":
                if e.ref in ("left", "right"):
                    step[0] += table.width if e.ref == "right" else -table.width
                else:
                    step[1] += table.height if e.ref == "top" else -table.height
            out[i] = out[i - 1] + step
        return out

    @property
    def collisions(self):
        """Events that are physical collisions (wall bounce or disk hit), not wraps/start."""
        return [e for e in self.events if e.kind in ("wall", "disk")]


def simulate(table: BilliardTable, pos0, vel0, n_events: int = 2000, t_max: float = None) -> EventTrajectory:
    pos = np.asarray(pos0, dtype=float).copy()
    vel = np.asarray(vel0, dtype=float).copy()
    t = 0.0
    events = [Event(t, pos.copy(), vel.copy(), "start", None)]

    for _ in range(n_events):
        t_wall, wall_id = table.wall_event(pos, vel)
        t_disk, disk_id = table.disk_event(pos, vel)

        if t_wall == np.inf and t_disk == np.inf:
            break

        if t_wall <= t_disk:
            dt = t_wall
            pos = pos + vel * dt
            t += dt
            pos, vel = table.apply_wall(pos, vel, wall_id)
            kind, ref = ("wrap" if table.periodic else "wall"), wall_id
        else:
            dt = t_disk
            pos = pos + vel * dt
            t += dt
            vel = table.apply_disk(pos, vel, disk_id)
            kind, ref = "disk", disk_id

        events.append(Event(t, pos.copy(), vel.copy(), kind, ref))
        if t_max is not None and t >= t_max:
            break

    return EventTrajectory(events)
