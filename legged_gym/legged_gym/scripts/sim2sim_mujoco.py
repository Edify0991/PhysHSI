import os
import sys

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, "..", ".."))
if REPO_ROOT not in sys.path:
    sys.path.append(REPO_ROOT)

import argparse

from legged_gym.sim2sim import MujocoSim2SimRunner, load_sim2sim_config


def get_args():
    parser = argparse.ArgumentParser(description="Sim2Sim evaluation in MuJoCo / MuJoCo Playground")
    parser.add_argument("--config", type=str, required=True, help="Path to sim2sim yaml config")
    parser.add_argument("--render", action="store_true", help="Enable MuJoCo viewer")
    parser.add_argument(
        "--backend",
        type=str,
        default=None,
        choices=["mujoco", "mujoco_playground"],
        help="Backend selector. 'mujoco_playground' currently uses MuJoCo stepping with playground-compatible config.",
    )
    return parser.parse_args()


def main():
    args = get_args()
    cfg = load_sim2sim_config(args.config)
    if args.render:
        cfg.sim.render = True
    if args.backend is not None:
        cfg.sim.backend = args.backend

    if cfg.sim.backend not in {"mujoco", "mujoco_playground"}:
        raise ValueError(f"Unsupported backend: {cfg.sim.backend}")

    runner = MujocoSim2SimRunner(cfg)
    metrics = runner.run()
    print("[sim2sim] done")
    print(f"  steps: {metrics.steps}")
    print(f"  duration_s: {metrics.duration_s:.2f}")
    print(f"  mean_abs_torque: {metrics.mean_abs_torque:.4f}")
    print(f"  max_abs_torque: {metrics.max_abs_torque:.4f}")


if __name__ == "__main__":
    main()
