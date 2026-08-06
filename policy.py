"""
policy.py

Turns one observation into one joint-space action. Knows nothing about CAN,
motors, or the control loop: Leg.get_observation() dict in, kwargs for
Leg.set_action() out. Model loading/inference is a placeholder until a
trained checkpoint is dropped in.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Optional

from robot_leg import JOINT_NAMES


class Policy:
    def __init__(self, *, action_scale: float = 1.0) -> None:
        self.action_scale = float(action_scale)
        self._model = None  # TODO: onnxruntime.InferenceSession or similar

    @classmethod
    def from_files(cls, policy_dir: Path) -> "Policy":
        """Load config + checkpoint (e.g. policy.onnx) from policy_dir.

        TODO:
          1. find config file (config.yaml/json) and checkpoint in policy_dir
          2. parse config (obs/action ordering, normalization, action_scale)
          3. load the checkpoint into self._model
        """
        policy_dir = Path(policy_dir)
        raise NotImplementedError(f"policy loading not implemented yet ({policy_dir})")

    def reset(self) -> None:
        """Clear any internal state (e.g. observation history) between runs."""

    def act(self, observation: Dict[str, float]) -> Dict[str, float]:
        """observation: Leg.get_observation() 형식 (JOINT_NAMES -> deg).
        반환값: Leg.set_action(**action)에 그대로 넣을 수 있는 관절공간 dict.

        TODO:
          1. observation dict -> model input vector, fixed JOINT_NAMES order
          2. self._model 추론 -> raw action vector
          3. raw action * self.action_scale -> {joint_name: target_deg}
        """
        if self._model is None:
            raise RuntimeError("policy has no model loaded -- call Policy.from_files() first")
        missing = set(JOINT_NAMES) - set(observation)
        if missing:
            raise ValueError(f"observation missing joint(s): {sorted(missing)}")
        # TODO: 실제 추론 로직
        #   1. observation dict -> model input vector, fixed JOINT_NAMES order
        #   2. self._model 추론 -> raw action vector
        #   3. raw action * self.action_scale -> {joint_name: target_deg}
        raise NotImplementedError
