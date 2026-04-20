"""
Multi-Algorithm RL Agent Manager
8개 강화학습 알고리즘을 통합 관리하는 에이전트 시스템

지원 알고리즘:
  1. SAC  (Soft Actor-Critic) — off-policy, 연속 행동, 엔트로피 정규화
  2. PPO  (Proximal Policy Optimization) — on-policy, 클리핑 기반
  3. TD3  (Twin Delayed DDPG) — off-policy, 쌍둥이 Q + 지연 정책 갱신
  4. DDPG (Deep Deterministic Policy Gradient) — off-policy, 결정적 정책
  5. A2C  (Advantage Actor-Critic) — on-policy, 동기식
  6. TQC  (Truncated Quantile Critics) — off-policy, 분포적 RL (sb3-contrib)
  7. TRPO (Trust Region Policy Optimization) — on-policy, 자연 경사 (sb3-contrib)
  8. ARS  (Augmented Random Search) — 무경사 최적화 (sb3-contrib)
"""

import os
import time
import numpy as np

try:
    import torch
    HAS_TORCH = True
    CUDA_AVAILABLE = torch.cuda.is_available()
except ImportError:
    HAS_TORCH = False
    CUDA_AVAILABLE = False

try:
    from stable_baselines3 import SAC, PPO, TD3, DDPG, A2C
    from stable_baselines3.common.callbacks import BaseCallback, EvalCallback
    from stable_baselines3.common.noise import NormalActionNoise
    HAS_SB3 = True
except ImportError:
    HAS_SB3 = False

try:
    from sb3_contrib import TQC, TRPO, ARS
    HAS_SB3_CONTRIB = True
except ImportError:
    HAS_SB3_CONTRIB = False


def get_device():
    if not HAS_TORCH:
        return "cpu"
    if CUDA_AVAILABLE:
        gpu_name = torch.cuda.get_device_name(0)
        vram_gb = torch.cuda.get_device_properties(0).total_mem / 1e9
        print(f"  [GPU] {gpu_name} ({vram_gb:.1f} GB VRAM)")
        return "cuda"
    if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        print("  [GPU] Apple MPS (Metal Performance Shaders)")
        return "mps"
    return "cpu"


class RewardLoggingCallback(BaseCallback):
    """학습 중 보상 추이 기록"""

    def __init__(self, verbose=0):
        super().__init__(verbose)
        self.episode_rewards = []
        self.episode_lengths = []
        self._current_rewards = []

    def _on_step(self):
        # infos에서 에피소드 보상 추출
        if self.locals.get("infos"):
            for info in self.locals["infos"]:
                if "episode" in info:
                    self.episode_rewards.append(info["episode"]["r"])
                    self.episode_lengths.append(info["episode"]["l"])
        return True

    def get_stats(self):
        if not self.episode_rewards:
            return {"mean_reward": 0, "std_reward": 0, "n_episodes": 0}
        return {
            "mean_reward": np.mean(self.episode_rewards[-50:]),
            "std_reward": np.std(self.episode_rewards[-50:]),
            "n_episodes": len(self.episode_rewards),
            "best_reward": max(self.episode_rewards) if self.episode_rewards else 0,
        }


# ─────────────────────────────────────────────
# 알고리즘별 하이퍼파라미터 설정
# ─────────────────────────────────────────────

def _get_hyperparams(algo_name, device, env):
    """알고리즘별 최적 하이퍼파라미터"""
    is_gpu = device in ("cuda", "mps")

    # 공통 버퍼/배치 크기
    large_buffer = 2_000_000 if is_gpu else 500_000
    med_buffer = 1_000_000 if is_gpu else 300_000
    large_batch = 512 if is_gpu else 256
    med_batch = 256 if is_gpu else 128
    large_net = [512, 512, 256] if is_gpu else [256, 256, 128]
    med_net = [256, 256] if is_gpu else [128, 128]

    configs = {
        "SAC": {
            "cls": SAC,
            "kwargs": {
                "learning_rate": 1e-4,
                "gamma": 0.995,
                "buffer_size": large_buffer,
                "batch_size": large_batch,
                "tau": 0.005,
                "ent_coef": "auto",
                "target_entropy": -0.5,
                "learning_starts": 5000,
                "train_freq": (1, "step"),
                "gradient_steps": 2,
                "policy_kwargs": {"net_arch": large_net},
            },
        },
        "PPO": {
            "cls": PPO,
            "kwargs": {
                "learning_rate": 3e-4,
                "gamma": 0.995,
                "n_steps": 4096,
                "batch_size": med_batch,
                "n_epochs": 10,
                "clip_range": 0.2,
                "ent_coef": 0.01,
                "vf_coef": 0.5,
                "max_grad_norm": 0.5,
                "gae_lambda": 0.95,
                "policy_kwargs": {"net_arch": dict(pi=med_net, vf=med_net)},
            },
        },
        "TD3": {
            "cls": TD3,
            "kwargs": {
                "learning_rate": 1e-4,
                "gamma": 0.995,
                "buffer_size": large_buffer,
                "batch_size": large_batch,
                "tau": 0.005,
                "learning_starts": 5000,
                "train_freq": (1, "step"),
                "gradient_steps": 2,
                "policy_delay": 2,
                "target_policy_noise": 0.2,
                "target_noise_clip": 0.5,
                "policy_kwargs": {"net_arch": large_net},
                "action_noise": NormalActionNoise(
                    mean=np.zeros(1), sigma=0.1 * np.ones(1)
                ),
            },
        },
        "DDPG": {
            "cls": DDPG,
            "kwargs": {
                "learning_rate": 1e-4,
                "gamma": 0.995,
                "buffer_size": med_buffer,
                "batch_size": med_batch,
                "tau": 0.005,
                "learning_starts": 5000,
                "train_freq": (1, "step"),
                "gradient_steps": 1,
                "policy_kwargs": {"net_arch": med_net},
                "action_noise": NormalActionNoise(
                    mean=np.zeros(1), sigma=0.15 * np.ones(1)
                ),
            },
        },
        "A2C": {
            "cls": A2C,
            "kwargs": {
                "learning_rate": 7e-4,
                "gamma": 0.995,
                "n_steps": 2048,
                "ent_coef": 0.01,
                "vf_coef": 0.5,
                "max_grad_norm": 0.5,
                "gae_lambda": 0.95,
                "policy_kwargs": {"net_arch": dict(pi=med_net, vf=med_net)},
            },
        },
    }

    # sb3-contrib 알고리즘
    if HAS_SB3_CONTRIB:
        configs["TQC"] = {
            "cls": TQC,
            "kwargs": {
                "learning_rate": 1e-4,
                "gamma": 0.995,
                "buffer_size": large_buffer,
                "batch_size": large_batch,
                "tau": 0.005,
                "ent_coef": "auto",
                "target_entropy": -0.5,
                "learning_starts": 5000,
                "train_freq": (1, "step"),
                "gradient_steps": 2,
                "top_quantiles_to_drop_per_net": 2,
                "policy_kwargs": {
                    "net_arch": large_net,
                    "n_quantiles": 25,
                    "n_critics": 2,
                },
            },
        }
        configs["TRPO"] = {
            "cls": TRPO,
            "kwargs": {
                "learning_rate": 1e-4,
                "gamma": 0.995,
                "n_steps": 4096,
                "batch_size": med_batch,
                "gae_lambda": 0.95,
                "cg_max_steps": 15,
                "cg_damping": 0.1,
                "target_kl": 0.01,
                "policy_kwargs": {"net_arch": dict(pi=med_net, vf=med_net)},
            },
        }
        configs["ARS"] = {
            "cls": ARS,
            "kwargs": {
                "learning_rate": 0.02,
                "n_delta": 8,
                "n_top": 4,
                "delta_std": 0.05,
                "policy_kwargs": {"net_arch": [64, 64]},
            },
        }

    return configs.get(algo_name)


class MultiRLAgent:
    """
    다중 RL 알고리즘 통합 에이전트

    사용법:
      agent = MultiRLAgent("SAC", env)
      agent.build()
      agent.train(100000)
      P_fc, a_norm = agent.predict(obs)
    """

    AVAILABLE_ALGOS = ["SAC", "PPO", "TD3", "DDPG", "A2C"]
    CONTRIB_ALGOS = ["TQC", "TRPO", "ARS"]

    def __init__(self, algo_name, env=None):
        if not HAS_SB3:
            raise RuntimeError("stable-baselines3 not installed")

        self.algo_name = algo_name
        self.env = env
        self.model = None
        self.device = get_device()
        self.P_fc_prev = 0.0
        self.ramp_limit = 4.0
        self.reward_callback = None

        if algo_name in self.CONTRIB_ALGOS and not HAS_SB3_CONTRIB:
            raise RuntimeError(f"{algo_name} requires sb3-contrib: pip install sb3-contrib")

    @classmethod
    def get_all_algo_names(cls):
        """사용 가능한 모든 알고리즘 이름 반환"""
        algos = list(cls.AVAILABLE_ALGOS)
        if HAS_SB3_CONTRIB:
            algos.extend(cls.CONTRIB_ALGOS)
        return algos

    def build(self, env=None):
        """모델 생성"""
        if env is not None:
            self.env = env

        config = _get_hyperparams(self.algo_name, self.device, self.env)
        if config is None:
            raise ValueError(f"Unknown algorithm: {self.algo_name}")

        cls = config["cls"]
        kwargs = config["kwargs"].copy()

        # On-policy 알고리즘은 CPU가 더 효율적 (SB3 권장)
        on_policy_algos = {"PPO", "A2C", "TRPO", "ARS"}
        effective_device = "cpu" if self.algo_name in on_policy_algos else self.device

        self.model = cls(
            "MlpPolicy",
            self.env,
            device=effective_device,
            verbose=0,
            **kwargs,
        )

        param_count = sum(p.numel() for p in self.model.policy.parameters())
        print(f"  [{self.algo_name}] Device: {effective_device}, "
              f"Params: {param_count:,}")
        return self.model

    def train(self, total_timesteps, log_interval=None, progress_bar=True):
        """학습 실행"""
        if self.model is None:
            raise RuntimeError("Call build() first")

        self.reward_callback = RewardLoggingCallback()

        self.model.learn(
            total_timesteps=total_timesteps,
            callback=self.reward_callback,
            progress_bar=progress_bar,
            log_interval=log_interval,
        )
        return self.reward_callback.get_stats()

    def predict(self, obs, deterministic=True):
        """관측 → (P_fc_star [kWe], a_norm)"""
        if self.model is None:
            a_norm = np.random.uniform(-1, 1)
        else:
            action, _ = self.model.predict(obs, deterministic=deterministic)
            a_norm = float(action[0]) if hasattr(action, '__len__') else float(action)

        P_fc_star = 5.0 + 77.0 * (a_norm + 1.0) / 2.0
        return P_fc_star, a_norm

    def apply_ramp_limit(self, P_fc_star, dt):
        if dt > 0:
            max_ramp = self.ramp_limit * dt
            P_fc_star = np.clip(P_fc_star,
                                self.P_fc_prev - max_ramp,
                                self.P_fc_prev + max_ramp)
        P_fc_star = np.clip(P_fc_star, 5.0, 82.0)
        self.P_fc_prev = P_fc_star
        return float(P_fc_star)

    def save(self, path):
        if self.model:
            os.makedirs(os.path.dirname(path), exist_ok=True)
            self.model.save(path)

    def load(self, path):
        config = _get_hyperparams(self.algo_name, self.device, None)
        if config:
            self.model = config["cls"].load(path, device=self.device)

    def reset(self):
        self.P_fc_prev = 0.0

    def get_training_stats(self):
        if self.reward_callback:
            return self.reward_callback.get_stats()
        return {}


def evaluate_agent(agent, env, n_episodes=1):
    """
    에이전트 평가 — km/kg H2 효율 측정

    Returns:
        dict: 평가 결과 (km_per_kg, distance_km, h2_kg, etc.)
    """
    results = []

    for ep in range(n_episodes):
        obs, _ = env.reset()
        agent.reset()
        total_reward = 0.0

        while True:
            P_fc_star, a_norm = agent.predict(obs, deterministic=True)
            P_fc_star = agent.apply_ramp_limit(P_fc_star, 1.0)
            a_norm_clipped = 2.0 * (P_fc_star - 5.0) / 77.0 - 1.0
            a_norm_clipped = np.clip(a_norm_clipped, -1.0, 1.0)

            obs, reward, terminated, truncated, info = env.step(
                np.array([a_norm_clipped])
            )
            total_reward += reward

            if terminated or truncated:
                break

        dist_km = env.distance_m / 1e3
        h2_kg = env.total_H2_kg
        km_per_kg = dist_km / max(h2_kg, 1e-9)
        fc_per_100km = h2_kg / max(dist_km, 1e-9) * 100
        completed = not env.terminated  # truncated = normal end

        results.append({
            "distance_km": dist_km,
            "h2_kg": h2_kg,
            "km_per_kg": km_per_kg,
            "fc_per_100km": fc_per_100km,
            "total_reward": total_reward,
            "final_SOC": env.battery.SOC,
            "final_SOH_fc": env.pemfc.SOH_fc,
            "final_SOH_bat": env.battery.SOH,
            "completed": completed,
            "steps": env.ctrl_step_idx,
            "terminated_early": env.terminated,
        })

    if n_episodes == 1:
        return results[0]

    avg = {k: np.mean([r[k] for r in results])
           for k in results[0] if isinstance(results[0][k], (int, float))}
    avg["completed"] = all(r["completed"] for r in results)
    avg["all_results"] = results
    return avg
