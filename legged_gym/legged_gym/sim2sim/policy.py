from typing import Protocol

import numpy as np
import torch


class Policy(Protocol):
    def act(self, obs: np.ndarray) -> np.ndarray:
        ...


class JitPolicy:
    def __init__(self, path: str, device: str = "cpu"):
        self.device = torch.device(device)
        self.model = torch.jit.load(path, map_location=self.device)
        self.model.eval()

    @torch.no_grad()
    def act(self, obs: np.ndarray) -> np.ndarray:
        obs_tensor = torch.from_numpy(obs).to(self.device).float().unsqueeze(0)
        action = self.model(obs_tensor)
        return action.squeeze(0).detach().cpu().numpy()


class OnnxPolicy:
    def __init__(self, path: str):
        import onnxruntime as ort

        self.session = ort.InferenceSession(path)
        self.input_name = self.session.get_inputs()[0].name

    def act(self, obs: np.ndarray) -> np.ndarray:
        outputs = self.session.run(None, {self.input_name: obs[np.newaxis, :].astype(np.float32)})
        return outputs[0].squeeze(0)


def load_policy(policy_type: str, path: str, device: str = "cpu") -> Policy:
    if policy_type == "jit":
        return JitPolicy(path=path, device=device)
    if policy_type == "onnx":
        return OnnxPolicy(path=path)
    raise ValueError(f"Unsupported policy_type: {policy_type}")
