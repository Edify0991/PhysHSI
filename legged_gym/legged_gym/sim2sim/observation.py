from collections import deque
from typing import List

import numpy as np


class HistoryObservationBuilder:
    def __init__(
        self,
        num_joints: int,
        history: int,
        default_qpos: np.ndarray,
        command: np.ndarray,
        obs_scales: dict,
        end_effector_body_ids: List[int],
        base_body_id: int,
    ):
        self.num_joints = num_joints
        self.history = history
        self.default_qpos = default_qpos
        self.command = command
        self.obs_scales = obs_scales
        self.end_effector_body_ids = end_effector_body_ids
        self.base_body_id = base_body_id
        self._obs_history = deque(maxlen=history)

    def reset(self, one_step_obs: np.ndarray):
        self._obs_history.clear()
        for _ in range(self.history):
            self._obs_history.append(one_step_obs.copy())

    def build(
        self,
        qpos: np.ndarray,
        qvel: np.ndarray,
        xmat: np.ndarray,
        xpos: np.ndarray,
        last_action: np.ndarray,
    ) -> np.ndarray:
        rot = xmat[self.base_body_id].reshape(3, 3)
        world_to_base = rot.T

        base_ang_vel = world_to_base @ qvel[3:6]
        gravity_proj = world_to_base @ np.array([0.0, 0.0, -1.0], dtype=np.float64)
        base_lin_vel = world_to_base @ qvel[0:3]

        ee_local = []
        base_pos = xpos[self.base_body_id]
        for body_id in self.end_effector_body_ids:
            rel = xpos[body_id] - base_pos
            ee_local.append(world_to_base @ rel)
        end_effector_pos = np.concatenate(ee_local, axis=0) if ee_local else np.zeros(15)

        current_obs = np.concatenate(
            [
                self.command,
                base_ang_vel * self.obs_scales["ang_vel"],
                gravity_proj,
                (qpos - self.default_qpos) * self.obs_scales["dof_pos"],
                qvel * self.obs_scales["dof_vel"],
                end_effector_pos,
                last_action,
                base_lin_vel * self.obs_scales["lin_vel"],
            ],
            axis=0,
        ).astype(np.float32)

        actor_step_obs = current_obs[:-3]
        if len(self._obs_history) == 0:
            self.reset(actor_step_obs)
        self._obs_history.append(actor_step_obs)
        return np.concatenate(list(self._obs_history), axis=0)
