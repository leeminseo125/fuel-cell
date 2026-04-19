"""
SAC 에이전트 — stable-baselines3 + CUDA GPU 지원
행동: a_norm in [-1, 1] -> P_fc* = 5 + 77*(a+1)/2 [kWe]
NVIDIA A4000 GPU 활용: 대규모 네트워크 + 대용량 리플레이 버퍼
"""

import os
import numpy as np

try:
    import torch
    HAS_TORCH = True
    CUDA_AVAILABLE = torch.cuda.is_available()
except ImportError:
    HAS_TORCH = False
    CUDA_AVAILABLE = False

try:
    from stable_baselines3 import SAC
    from stable_baselines3.common.callbacks import BaseCallback
    HAS_SB3 = True
except ImportError:
    HAS_SB3 = False


def get_device():
    """최적 디바이스 선택 (A4000 GPU 우선)"""
    if not HAS_TORCH:
        return "cpu"
    if CUDA_AVAILABLE:
        gpu_name = torch.cuda.get_device_name(0)
        vram_gb = torch.cuda.get_device_properties(0).total_mem / 1e9
        print(f"  [GPU] {gpu_name} ({vram_gb:.1f} GB VRAM)")
        return "cuda"
    return "cpu"


class SACAgent:
    """SAC EMS 에이전트 래퍼 — GPU 가속"""

    def __init__(self, env=None, model_path: str = None):
        self.env = env
        self.model = None
        self.P_fc_prev = 0.0
        self.ramp_limit = 4.0     # kW/s
        self.device = get_device()

        if model_path and os.path.exists(model_path) and HAS_SB3:
            self.model = SAC.load(model_path, device=self.device)

    def build_model(self, env):
        """SAC 모델 생성 — A4000 GPU 최적화 설정"""
        if not HAS_SB3:
            raise RuntimeError("stable-baselines3 not installed")

        self.env = env

        # A4000 (16GB VRAM): 대형 네트워크 + 대용량 버퍼
        if CUDA_AVAILABLE:
            buffer_size = 2_000_000
            batch_size = 512
            net_arch = [512, 512, 256]
            learning_starts = 10_000
        else:
            buffer_size = 1_000_000
            batch_size = 256
            net_arch = [256, 256, 128]
            learning_starts = 5_000

        self.model = SAC(
            "MlpPolicy",
            env,
            learning_rate=3e-4,
            gamma=0.99,
            buffer_size=buffer_size,
            batch_size=batch_size,
            tau=0.005,
            ent_coef="auto",
            target_entropy=-1.0,
            learning_starts=learning_starts,
            train_freq=(1, "step"),
            gradient_steps=1,
            policy_kwargs={
                "net_arch": net_arch,
            },
            device=self.device,
            verbose=1,
        )
        print(f"  [SAC] Device: {self.device}, Buffer: {buffer_size:,}, "
              f"Batch: {batch_size}, Net: {net_arch}")
        return self.model

    def train(self, total_timesteps: int = 500_000, log_dir: str = None):
        if self.model is None:
            raise RuntimeError("Call build_model() first")

        callbacks = []
        if log_dir:
            os.makedirs(log_dir, exist_ok=True)

        self.model.learn(
            total_timesteps=total_timesteps,
            callback=callbacks if callbacks else None,
            progress_bar=True,
        )

    def save(self, path: str):
        if self.model:
            self.model.save(path)

    def load(self, path: str):
        if HAS_SB3:
            self.model = SAC.load(path, device=self.device)

    def predict(self, obs: np.ndarray, deterministic: bool = True) -> float:
        """obs -> a_norm -> P_fc_star [kWe]"""
        if self.model is None:
            a_norm = np.random.uniform(-1, 1)
        else:
            action, _ = self.model.predict(obs, deterministic=deterministic)
            a_norm = float(action[0]) if hasattr(action, '__len__') else float(action)

        P_fc_star = 5.0 + 77.0 * (a_norm + 1.0) / 2.0
        return P_fc_star, a_norm

    def apply_ramp_limit(self, P_fc_star: float, dt: float) -> float:
        """경사 제한 (6.2)"""
        if dt > 0:
            max_ramp = self.ramp_limit * dt
            P_fc_star = np.clip(P_fc_star,
                                self.P_fc_prev - max_ramp,
                                self.P_fc_prev + max_ramp)
        P_fc_star = np.clip(P_fc_star, 5.0, 82.0)
        self.P_fc_prev = P_fc_star
        return float(P_fc_star)

    def reset(self):
        self.P_fc_prev = 0.0
