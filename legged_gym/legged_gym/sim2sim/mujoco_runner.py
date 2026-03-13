from dataclasses import dataclass
from typing import List

import numpy as np

from .config import Sim2SimConfig
from .observation import HistoryObservationBuilder
from .policy import load_policy


@dataclass
class Sim2SimMetrics:
    steps: int
    duration_s: float
    mean_abs_torque: float
    max_abs_torque: float


class MujocoSim2SimRunner:
    def __init__(self, cfg: Sim2SimConfig):
        try:
            import mujoco
            import mujoco.viewer
        except ImportError as exc:
            raise ImportError("Please install mujoco first: pip install mujoco") from exc

        self.mujoco = mujoco
        self.viewer_mod = mujoco.viewer
        self.cfg = cfg

        self.model = mujoco.MjModel.from_xml_path(cfg.sim.model_path)
        self.data = mujoco.MjData(self.model)

        self.policy = load_policy(cfg.policy.policy_type, cfg.policy.policy_path, cfg.policy.device)

        self.joint_ids = self._resolve_joint_ids(cfg.robot.joint_names)
        self.actuator_ids = self._resolve_actuator_ids(cfg.robot.actuator_names)
        self.base_body_id = self._resolve_body_id(cfg.robot.body_name)
        self.ee_body_ids = self._resolve_body_ids(cfg.robot.end_effector_body_names)

        n = len(self.joint_ids)
        self.default_qpos = np.asarray(cfg.robot.default_joint_pos, dtype=np.float64)
        self.kp = np.asarray(cfg.robot.kp, dtype=np.float64)
        self.kd = np.asarray(cfg.robot.kd, dtype=np.float64)
        self.torque_limit = np.asarray(cfg.robot.torque_limit, dtype=np.float64)
        if self.torque_limit.size == 0:
            self.torque_limit = np.full(n, np.inf)

        self.last_action = np.zeros(n, dtype=np.float32)
        self.command = np.asarray(cfg.obs.command, dtype=np.float32)

        self.obs_builder = HistoryObservationBuilder(
            num_joints=n,
            history=cfg.obs.actor_history,
            default_qpos=self.default_qpos,
            command=self.command,
            obs_scales=cfg.obs.obs_scales,
            end_effector_body_ids=self.ee_body_ids,
            base_body_id=self.base_body_id,
        )

    def _resolve_joint_ids(self, names: List[str]) -> List[int]:
        ids = []
        for name in names:
            jid = self.mujoco.mj_name2id(self.model, self.mujoco.mjtObj.mjOBJ_JOINT, name)
            if jid < 0:
                raise ValueError(f"Joint not found in MuJoCo model: {name}")
            ids.append(jid)
        return ids

    def _resolve_actuator_ids(self, names: List[str]) -> List[int]:
        ids = []
        for name in names:
            aid = self.mujoco.mj_name2id(self.model, self.mujoco.mjtObj.mjOBJ_ACTUATOR, name)
            if aid < 0:
                raise ValueError(f"Actuator not found in MuJoCo model: {name}")
            ids.append(aid)
        return ids

    def _resolve_body_id(self, name: str) -> int:
        bid = self.mujoco.mj_name2id(self.model, self.mujoco.mjtObj.mjOBJ_BODY, name)
        if bid < 0:
            raise ValueError(f"Body not found in MuJoCo model: {name}")
        return bid

    def _resolve_body_ids(self, names: List[str]) -> List[int]:
        ids = []
        for name in names:
            bid = self.mujoco.mj_name2id(self.model, self.mujoco.mjtObj.mjOBJ_BODY, name)
            if bid < 0:
                raise ValueError(f"End-effector body not found in MuJoCo model: {name}")
            ids.append(bid)
        return ids

    def _read_joint_state(self):
        qpos = np.zeros(len(self.joint_ids), dtype=np.float64)
        qvel = np.zeros(len(self.joint_ids), dtype=np.float64)
        for i, jid in enumerate(self.joint_ids):
            qpos_adr = self.model.jnt_qposadr[jid]
            qvel_adr = self.model.jnt_dofadr[jid]
            qpos[i] = self.data.qpos[qpos_adr]
            qvel[i] = self.data.qvel[qvel_adr]
        return qpos, qvel

    def _compute_torque(self, action: np.ndarray, qpos: np.ndarray, qvel: np.ndarray) -> np.ndarray:
        joint_target = self.default_qpos + self.cfg.robot.action_scale * action
        torque = self.kp * (joint_target - qpos) - self.kd * qvel
        return np.clip(torque, -self.torque_limit, self.torque_limit)

    def reset(self):
        self.mujoco.mj_resetData(self.model, self.data)
        for i, jid in enumerate(self.joint_ids):
            qpos_adr = self.model.jnt_qposadr[jid]
            qvel_adr = self.model.jnt_dofadr[jid]
            self.data.qpos[qpos_adr] = self.default_qpos[i]
            self.data.qvel[qvel_adr] = 0.0
        self.mujoco.mj_forward(self.model, self.data)
        qpos, qvel = self._read_joint_state()
        self.obs_builder.build(
            qpos=qpos,
            qvel=qvel,
            xmat=self.data.xmat,
            xpos=self.data.xpos,
            last_action=self.last_action,
        )

    def run(self) -> Sim2SimMetrics:
        self.model.opt.timestep = self.cfg.sim.timestep
        self.reset()
        steps = int(self.cfg.sim.duration_s / self.cfg.sim.timestep)
        torque_values = []

        viewer = None
        if self.cfg.sim.render:
            viewer = self.viewer_mod.launch_passive(self.model, self.data)

        for step in range(steps):
            qpos, qvel = self._read_joint_state()
            obs = self.obs_builder.build(
                qpos=qpos,
                qvel=qvel,
                xmat=self.data.xmat,
                xpos=self.data.xpos,
                last_action=self.last_action,
            )
            action = self.policy.act(obs)
            self.last_action = action.astype(np.float32)

            torque = self._compute_torque(action, qpos, qvel)
            torque_values.append(np.abs(torque))

            for _ in range(self.cfg.sim.decimation):
                self.data.ctrl[:] = 0.0
                self.data.ctrl[self.actuator_ids] = torque
                self.mujoco.mj_step(self.model, self.data)
                if viewer is not None:
                    viewer.sync()

        if viewer is not None:
            viewer.close()

        all_torque = np.concatenate(torque_values, axis=0)
        return Sim2SimMetrics(
            steps=steps,
            duration_s=self.cfg.sim.duration_s,
            mean_abs_torque=float(all_torque.mean()),
            max_abs_torque=float(all_torque.max()),
        )
