"""
zero_policy.py

Trivial policy: always commands all joints to 0deg. Same act()/reset()
interface as Policy (robot_leg.Leg.get_observation() dict in, kwargs for
Leg.set_action() out), so main.py can swap between this and a real,
model-backed Policy without changing the control loop. Useful for
exercising the full stack (bus, safety, control loop) before a trained
policy is wired in.
"""

from __future__ import annotations

from typing import Dict

from robot_leg import JOINT_NAMES


class ZeroPolicy:
    def reset(self) -> None:
        """No internal state to clear."""

    def act(self, observation: Dict[str, float]) -> Dict[str, float]:
        return {name: 0.0 for name in JOINT_NAMES}
