from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List

import yaml


@dataclass
class SimConfig:
    model_path: str
    timestep: float = 0.002
    decimation: int = 4
    duration_s: float = 15.0
    render: bool = False
    backend: str = "mujoco"


@dataclass
class RobotConfig:
    joint_names: List[str]
    actuator_names: List[str]
    default_joint_pos: List[float]
    kp: List[float]
    kd: List[float]
    action_scale: float = 0.25
    torque_limit: List[float] = field(default_factory=list)
    body_name: str = "pelvis"
    end_effector_body_names: List[str] = field(default_factory=list)


@dataclass
class ObsConfig:
    actor_history: int = 6
    obs_scales: Dict[str, float] = field(default_factory=lambda: {
        "lin_vel": 2.0,
        "ang_vel": 0.25,
        "dof_pos": 1.0,
        "dof_vel": 0.05,
    })
    command: List[float] = field(default_factory=lambda: [0.8, 0.0, 0.0])


@dataclass
class PolicyConfig:
    policy_path: str
    policy_type: str = "jit"
    device: str = "cpu"


@dataclass
class Sim2SimConfig:
    sim: SimConfig
    robot: RobotConfig
    obs: ObsConfig
    policy: PolicyConfig


def load_sim2sim_config(path: str) -> Sim2SimConfig:
    cfg_path = Path(path)
    with cfg_path.open("r", encoding="utf-8") as f:
        raw = yaml.safe_load(f)

    sim = SimConfig(**raw["sim"])
    robot = RobotConfig(**raw["robot"])
    obs = ObsConfig(**raw.get("obs", {}))
    policy = PolicyConfig(**raw["policy"])

    root = cfg_path.parent
    sim.model_path = str((root / sim.model_path).resolve())
    policy.policy_path = str((root / policy.policy_path).resolve())
    return Sim2SimConfig(sim=sim, robot=robot, obs=obs, policy=policy)
