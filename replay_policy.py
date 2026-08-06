"""
replay_policy.py

Open-loop playback of a recorded joint trajectory (see record_trajectory.py).
Same act()/reset() interface as Policy/ZeroPolicy, so main.py can swap it in
via --replay-csv without touching the control loop. Ignores the observation
argument -- this is pure time-indexed playback of what was demonstrated, not
feedback control.
"""

from __future__ import annotations

import bisect
import csv
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from robot_leg import JOINT_NAMES


class ReplayPolicy:
    def __init__(self, trajectory: List[Tuple[float, Dict[str, float]]], *, loop: bool = False) -> None:
        if not trajectory:
            raise ValueError("trajectory is empty")
        self._t = [t for t, _ in trajectory]  # sorted, starts at 0
        self._q = [q for _, q in trajectory]
        self._loop = loop
        self._duration = self._t[-1]
        self._t0: Optional[float] = None

    @classmethod
    def from_csv(cls, path: Path, *, loop: bool = False) -> "ReplayPolicy":
        """Load a trajectory recorded by record_trajectory.py (columns: t_s, JOINT_NAMES...)."""
        path = Path(path)
        trajectory: List[Tuple[float, Dict[str, float]]] = []
        with open(path, newline="") as f:
            reader = csv.DictReader(f)
            missing = set(JOINT_NAMES) - set(reader.fieldnames or [])
            if missing:
                raise ValueError(f"{path}: missing column(s) {sorted(missing)}")
            for row in reader:
                t = float(row["t_s"])
                q = {name: float(row[name]) for name in JOINT_NAMES}
                trajectory.append((t, q))
        if not trajectory:
            raise ValueError(f"{path}: no rows")

        trajectory.sort(key=lambda item: item[0])
        t0 = trajectory[0][0]
        trajectory = [(t - t0, q) for t, q in trajectory]  # re-base so playback starts at t=0
        return cls(trajectory, loop=loop)

    def reset(self) -> None:
        """Re-arm playback -- the next act() call becomes t=0. Call this right
        before the control loop starts, not at construction time, so replay
        timing lines up with when torque is actually enabled."""
        self._t0 = None

    def _sample(self, elapsed: float) -> Dict[str, float]:
        if self._loop and self._duration > 0:
            elapsed = elapsed % self._duration
        if elapsed <= self._t[0]:
            return dict(self._q[0])
        if elapsed >= self._t[-1]:
            return dict(self._q[-1])

        i = bisect.bisect_right(self._t, elapsed) - 1
        t0, t1 = self._t[i], self._t[i + 1]
        q0, q1 = self._q[i], self._q[i + 1]
        frac = (elapsed - t0) / (t1 - t0) if t1 > t0 else 0.0
        return {name: q0[name] + frac * (q1[name] - q0[name]) for name in JOINT_NAMES}

    def act(self, observation: Dict[str, float]) -> Dict[str, float]:
        now = time.monotonic()
        if self._t0 is None:
            self._t0 = now
        return self._sample(now - self._t0)

    @property
    def finished(self) -> bool:
        return self._t0 is not None and not self._loop and (time.monotonic() - self._t0) >= self._duration
