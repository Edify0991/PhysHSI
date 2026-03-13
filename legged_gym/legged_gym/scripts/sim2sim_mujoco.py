import argparse
import time
from collections import deque
from dataclasses import dataclass
from typing import Dict, List, Optional

import numpy as np
import torch

import mujoco
from mujoco import viewer as mj_viewer


TASK_CFG_MODULES = {
    "carrybox": "legged_gym.envs.g1.carrybox_config",
    "sitdown": "legged_gym.envs.g1.sitdown_config",
    "liedown": "legged_gym.envs.g1.liedown_config",
    "standup": "legged_gym.envs.g1.standup_config",
    "styleloco_dinosaur": "legged_gym.envs.g1.styleloco_dinosaur_config",
    "styleloco_highknee": "legged_gym.envs.g1.styleloco_highknee_config",
}


@dataclass
class ObjectPose:
    pos: np.ndarray
    quat_xyzw: np.ndarray


def _import_cfg(task: str):
    if task not in TASK_CFG_MODULES:
        raise ValueError(f"Unsupported task {task}. Available: {list(TASK_CFG_MODULES.keys())}")
    module = __import__(TASK_CFG_MODULES[task], fromlist=["G1Cfg"])
    return module.G1Cfg()


def quat_conjugate_xyzw(q: np.ndarray) -> np.ndarray:
    return np.array([-q[0], -q[1], -q[2], q[3]], dtype=np.float64)


def quat_mul_xyzw(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    ax, ay, az, aw = a
    bx, by, bz, bw = b
    return np.array(
        [
            aw * bx + ax * bw + ay * bz - az * by,
            aw * by - ax * bz + ay * bw + az * bx,
            aw * bz + ax * by - ay * bx + az * bw,
            aw * bw - ax * bx - ay * by - az * bz,
        ],
        dtype=np.float64,
    )


def quat_rotate_inverse_xyzw(q: np.ndarray, v: np.ndarray) -> np.ndarray:
    q_inv = quat_conjugate_xyzw(q)
    vq = np.array([v[0], v[1], v[2], 0.0], dtype=np.float64)
    return quat_mul_xyzw(quat_mul_xyzw(q_inv, vq), q)[:3]


def quat_to_tan_norm_xyzw(q: np.ndarray) -> np.ndarray:
    ref_tan = np.array([1.0, 0.0, 0.0], dtype=np.float64)
    ref_norm = np.array([0.0, 0.0, 1.0], dtype=np.float64)

    def quat_rotate(qin: np.ndarray, v: np.ndarray) -> np.ndarray:
        vq = np.array([v[0], v[1], v[2], 0.0], dtype=np.float64)
        return quat_mul_xyzw(quat_mul_xyzw(qin, vq), quat_conjugate_xyzw(qin))[:3]

    return np.concatenate([quat_rotate(q, ref_tan), quat_rotate(q, ref_norm)])


def mj_quat_wxyz_to_xyzw(q_wxyz: np.ndarray) -> np.ndarray:
    return np.array([q_wxyz[1], q_wxyz[2], q_wxyz[3], q_wxyz[0]], dtype=np.float64)


class Sim2SimMujocoRunner:
    def __init__(self, args: argparse.Namespace):
        self.args = args
        self.cfg = _import_cfg(args.task)
        self.model = mujoco.MjModel.from_xml_path(args.mjcf)
        self.data = mujoco.MjData(self.model)
        self.policy = torch.jit.load(args.policy, map_location=args.device).eval()

        self.sim_dt = self.model.opt.timestep
        self.control_decimation = self.cfg.control.decimation
        self.control_dt = self.control_decimation * self.sim_dt
        self.history_len = self.cfg.env.num_actor_history

        self.num_actions = self.cfg.env.num_actions
        self.default_joint_angles = self.cfg.init_state.default_joint_angles
        self.action_scale = self.cfg.control.action_scale
        self.obs_scales = self.cfg.normalization.obs_scales

        self.actuated_joint_ids = list(self.model.actuator_trnid[:, 0])
        self.actuated_joint_names = [mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_JOINT, j) for j in self.actuated_joint_ids]
        if len(self.actuated_joint_ids) != self.num_actions:
            raise ValueError(f"Actuator count {len(self.actuated_joint_ids)} != policy action dim {self.num_actions}")

        self.qpos_adr = [self.model.jnt_qposadr[jid] for jid in self.actuated_joint_ids]
        self.qvel_adr = [self.model.jnt_dofadr[jid] for jid in self.actuated_joint_ids]

        self.upper_body_name = self.cfg.asset.upper_body_link
        self.upper_body_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, self.upper_body_name)
        self.ee_body_ids = self._resolve_end_effector_bodies()

        self.kp = np.array([self._find_pd_gain(n, self.cfg.control.stiffness, default=100.0) for n in self.actuated_joint_names])
        self.kd = np.array([self._find_pd_gain(n, self.cfg.control.damping, default=2.0) for n in self.actuated_joint_names])

        self.command = np.array([args.command_x, args.command_y, args.command_yaw], dtype=np.float64)
        self.action_hist = np.zeros(self.num_actions, dtype=np.float64)
        self.obs_hist = deque(maxlen=self.history_len)
        for _ in range(self.history_len):
            self.obs_hist.append(np.zeros(self.cfg.env.num_one_step_proprio_obs, dtype=np.float32))

    def _find_pd_gain(self, joint_name: str, gain_table: Dict[str, float], default: float) -> float:
        for key, val in gain_table.items():
            if key in joint_name:
                return float(val)
        return float(default)

    def _resolve_end_effector_bodies(self) -> List[int]:
        body_names = [mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_BODY, i) for i in range(self.model.nbody)]

        def pick_two(keyword: str) -> List[int]:
            ids = [i for i, n in enumerate(body_names) if n and keyword in n]
            ids.sort(key=lambda x: body_names[x])
            return ids[:2]

        hand_ids = pick_two(self.cfg.asset.hand_pos_name)
        foot_ids = pick_two(self.cfg.asset.foot_name)
        head_ids = [i for i, n in enumerate(body_names) if n == self.cfg.asset.head_name]
        if len(hand_ids) != 2 or len(foot_ids) != 2 or not head_ids:
            raise ValueError("Failed to resolve end-effector bodies from MJCF names.")
        return [hand_ids[0], hand_ids[1], foot_ids[0], foot_ids[1], head_ids[0]]

    def reset(self):
        mujoco.mj_resetData(self.model, self.data)
        self.data.qpos[:3] = np.array(self.cfg.init_state.pos, dtype=np.float64)
        self.data.qpos[3:7] = np.array([self.cfg.init_state.rot[3], *self.cfg.init_state.rot[:3]], dtype=np.float64)

        for idx, jname in enumerate(self.actuated_joint_names):
            self.data.qpos[self.qpos_adr[idx]] = self.default_joint_angles[jname]
            self.data.qvel[self.qvel_adr[idx]] = 0.0

        mujoco.mj_forward(self.model, self.data)

    def _base_state(self):
        base_q_xyzw = mj_quat_wxyz_to_xyzw(self.data.xquat[self.upper_body_id])
        base_lin_vel = quat_rotate_inverse_xyzw(base_q_xyzw, self.data.cvel[self.upper_body_id, 3:6])
        base_ang_vel = quat_rotate_inverse_xyzw(base_q_xyzw, self.data.cvel[self.upper_body_id, 0:3])
        projected_gravity = quat_rotate_inverse_xyzw(base_q_xyzw, np.array([0.0, 0.0, -1.0]))
        return base_q_xyzw, base_lin_vel, base_ang_vel, projected_gravity

    def _task_obs(self, base_q_xyzw: np.ndarray) -> np.ndarray:
        if self.args.task.startswith("styleloco"):
            return np.zeros((0,), dtype=np.float32)

        if self.args.task == "carrybox":
            box = self._get_object_pose(self.args.box_body)
            goal = np.array(self.args.goal_pos, dtype=np.float64)
            box_local = quat_rotate_inverse_xyzw(base_q_xyzw, box.pos - self.data.qpos[:3])
            box_q_local = quat_mul_xyzw(quat_conjugate_xyzw(base_q_xyzw), box.quat_xyzw)
            goal_local = quat_rotate_inverse_xyzw(base_q_xyzw, goal - self.data.qpos[:3])
            return np.concatenate([box_local, quat_to_tan_norm_xyzw(box_q_local), np.array(self.args.box_size), goal_local]).astype(np.float32)

        chair = self._get_object_pose(self.args.chair_body)
        chair_local = quat_rotate_inverse_xyzw(base_q_xyzw, chair.pos - self.data.qpos[:3])
        chair_q_local = quat_mul_xyzw(quat_conjugate_xyzw(base_q_xyzw), chair.quat_xyzw)

        if self.args.task == "standup":
            marker = np.array(self.args.marker_pos, dtype=np.float64)
            marker_local = quat_rotate_inverse_xyzw(base_q_xyzw, marker - self.data.qpos[:3])
            return np.concatenate([chair_local, quat_to_tan_norm_xyzw(chair_q_local), marker_local]).astype(np.float32)

        return np.concatenate([chair_local, quat_to_tan_norm_xyzw(chair_q_local)]).astype(np.float32)

    def _get_object_pose(self, body_name: Optional[str]) -> ObjectPose:
        if not body_name:
            return ObjectPose(pos=np.zeros(3), quat_xyzw=np.array([0.0, 0.0, 0.0, 1.0]))
        bid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, body_name)
        if bid < 0:
            raise ValueError(f"Body {body_name} not found in MJCF")
        return ObjectPose(pos=self.data.xpos[bid].copy(), quat_xyzw=mj_quat_wxyz_to_xyzw(self.data.xquat[bid]))

    def _build_obs(self) -> np.ndarray:
        base_q_xyzw, base_lin_vel, base_ang_vel, projected_gravity = self._base_state()

        dof_pos = np.array([self.data.qpos[adr] for adr in self.qpos_adr], dtype=np.float64)
        dof_vel = np.array([self.data.qvel[adr] for adr in self.qvel_adr], dtype=np.float64)
        default_pos = np.array([self.default_joint_angles[j] for j in self.actuated_joint_names], dtype=np.float64)

        root_pos = self.data.qpos[:3].copy()
        ee_local = []
        for bid in self.ee_body_ids:
            ee_local.append(quat_rotate_inverse_xyzw(base_q_xyzw, self.data.xpos[bid] - root_pos))
        ee_local = np.concatenate(ee_local)

        if self.args.task.startswith("styleloco"):
            one_step = np.concatenate([
                self.command,
                base_ang_vel * self.obs_scales.ang_vel,
                projected_gravity,
                (dof_pos - default_pos) * self.obs_scales.dof_pos,
                dof_vel * self.obs_scales.dof_vel,
                ee_local,
                self.action_hist,
            ])
            task_obs = np.zeros((0,), dtype=np.float64)
        else:
            one_step = np.concatenate([
                base_ang_vel * self.obs_scales.ang_vel,
                projected_gravity,
                (dof_pos - default_pos) * self.obs_scales.dof_pos,
                dof_vel * self.obs_scales.dof_vel,
                ee_local,
                self.action_hist,
            ])
            task_obs = self._task_obs(base_q_xyzw)

        self.obs_hist.append(one_step.astype(np.float32))
        history = np.concatenate(list(self.obs_hist), axis=0)
        return np.concatenate([history, task_obs.astype(np.float32)]).astype(np.float32)

    def step(self):
        obs = self._build_obs()
        obs_tensor = torch.from_numpy(obs).unsqueeze(0).to(self.args.device)
        with torch.no_grad():
            action = self.policy(obs_tensor).detach().cpu().numpy()[0]
        action = np.clip(action, -self.cfg.normalization.clip_actions, self.cfg.normalization.clip_actions)
        self.action_hist = action.copy()

        q = np.array([self.data.qpos[adr] for adr in self.qpos_adr], dtype=np.float64)
        qd = np.array([self.data.qvel[adr] for adr in self.qvel_adr], dtype=np.float64)
        default_pos = np.array([self.default_joint_angles[j] for j in self.actuated_joint_names], dtype=np.float64)
        target = default_pos + self.action_scale * action
        torques = self.kp * (target - q) - self.kd * qd

        if self.model.nu == len(torques):
            self.data.ctrl[:] = torques
        else:
            self.data.ctrl[: len(torques)] = torques

        for _ in range(self.control_decimation):
            mujoco.mj_step(self.model, self.data)

    def run(self):
        self.reset()
        n_steps = int(self.args.duration_s / self.control_dt)
        if self.args.headless:
            for _ in range(n_steps):
                self.step()
            return

        with mj_viewer.launch_passive(self.model, self.data) as viewer:
            for _ in range(n_steps):
                tic = time.time()
                self.step()
                viewer.sync()
                remain = self.control_dt - (time.time() - tic)
                if remain > 0:
                    time.sleep(remain)


def parse_args():
    parser = argparse.ArgumentParser("PhysHSI sim2sim evaluator in MuJoCo")
    parser.add_argument("--task", type=str, required=True, choices=list(TASK_CFG_MODULES.keys()))
    parser.add_argument("--policy", type=str, required=True, help="TorchScript policy exported from IsaacGym")
    parser.add_argument("--mjcf", type=str, required=True, help="MuJoCo XML path for the robot + scene")
    parser.add_argument("--duration_s", type=float, default=10.0)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--headless", action="store_true")

    parser.add_argument("--command_x", type=float, default=0.8)
    parser.add_argument("--command_y", type=float, default=0.0)
    parser.add_argument("--command_yaw", type=float, default=0.0)

    parser.add_argument("--chair_body", type=str, default="chair")
    parser.add_argument("--marker_pos", type=float, nargs=3, default=[0.0, 0.0, 0.0])
    parser.add_argument("--box_body", type=str, default="box")
    parser.add_argument("--box_size", type=float, nargs=3, default=[0.25, 0.18, 0.12])
    parser.add_argument("--goal_pos", type=float, nargs=3, default=[2.0, 0.0, 0.8])
    return parser.parse_args()


def main():
    args = parse_args()
    runner = Sim2SimMujocoRunner(args)
    runner.run()


if __name__ == "__main__":
    main()
