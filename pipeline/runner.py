"""
12 인스턴스 실행 러너 (6 시나리오 x 2 EMS)
규칙 기반 vs SAC x (A,B,C,D,E,F) = 12 인스턴스
고정밀 서브스테핑: dt_sim=0.01s, dt_ctrl=1.0s
"""

import os
import time
import numpy as np
import pandas as pd

import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from scenarios.scenario_configs import SCENARIOS, SCENARIO_IDS
from pipeline.cycle_loader import load_cycle
from env.env import FCHEVEnv
from control.rule_ems import RuleBasedEMS
from control.sac_agent import SACAgent
from models.bop_model import BoPModel


# 고정 시뮬레이션 파라미터
DT_SIM = 0.01     # 내부 물리 스텝 (0.01s = 10ms)
DT_CTRL = 1.0     # 제어/로그 주기 (1.0s)


def _make_env(scenario_id: str):
    """시나리오별 환경 생성 (공통 조건)"""
    cfg = SCENARIOS[scenario_id]
    v, a = load_cycle(cfg["cycle"], cfg["duration_s"], dt=DT_SIM)

    env = FCHEVEnv(
        cycle_v=v, cycle_a=a,
        duration_s=cfg["duration_s"],
        dt_sim=DT_SIM,
        dt_ctrl=DT_CTRL,
        T_amb_C=cfg["temperature_C"],
        SOC_init=cfg["SOC_init"],
        scenario_id=scenario_id,
    )
    return env


def _extract_result(env, scenario_id: str, ems_type: str) -> dict:
    return {
        "scenario_id": scenario_id,
        "ems_type": ems_type,
        "log": env.get_log_dict(),
        "total_H2_kg": env.total_H2_kg,
        "distance_km": env.distance_m / 1e3,
        "final_SOC": env.battery.SOC,
        "final_SOH_fc": env.pemfc.SOH_fc,
        "final_SOH_bat": env.battery.SOH,
        "terminated_early": env.terminated,
        "steps": env.ctrl_step_idx,
    }


def run_rule_instance(scenario_id: str) -> dict:
    """규칙 기반 EMS로 시나리오 실행 (서브스테핑)"""
    cfg = SCENARIOS[scenario_id]
    env = _make_env(scenario_id)
    obs, _ = env.reset()

    rule_ems = RuleBasedEMS()
    total_ctrl_steps = int(cfg["duration_s"] / DT_CTRL)

    t0 = time.time()
    for step_i in range(total_ctrl_steps):
        # 현재 상태에서 P_low 추출 (관측 역정규화)
        P_low_kw = obs[0] * 82.0
        v_ms = obs[7] * 36.0
        SOC = obs[1]

        # 규칙 기반 결정 (dt_ctrl=1.0s 기준 경사 제한)
        rule_res = rule_ems.compute(P_low_kw, SOC, v_ms, DT_CTRL)
        P_fc_star = rule_res["P_fc_star"]

        # 행동 정규화
        a_norm = 2.0 * (P_fc_star - 5.0) / 77.0 - 1.0
        a_norm = np.clip(a_norm, -1.0, 1.0)

        obs, reward, terminated, truncated, info = env.step(np.array([a_norm]))

        # zone 정보 추가
        if env.log:
            env.log[-1]["ems_zone"] = rule_res["zone"]

        if terminated or truncated:
            break

        # 진행률 (10% 단위)
        if (step_i + 1) % (total_ctrl_steps // 10) == 0:
            pct = (step_i + 1) / total_ctrl_steps * 100
            elapsed = time.time() - t0
            print(f"    [{scenario_id}-Rule] {pct:.0f}% ({elapsed:.0f}s)")

    return _extract_result(env, scenario_id, "rule")


def run_sac_instance(scenario_id: str, model_path: str = None,
                     train: bool = True,
                     train_timesteps: int = 500_000) -> dict:
    """SAC EMS로 시나리오 실행 (학습 + 평가)"""
    cfg = SCENARIOS[scenario_id]
    env = _make_env(scenario_id)

    sac = SACAgent()

    # 학습
    if train and model_path is None:
        try:
            sac.build_model(env)
            print(f"  [SAC-{scenario_id}] Training {train_timesteps:,} steps on {sac.device}...")
            sac.train(total_timesteps=train_timesteps)
            save_path = os.path.join(
                os.path.dirname(os.path.dirname(__file__)),
                "dataset", f"sac_model_{scenario_id}"
            )
            sac.save(save_path)
            print(f"  [SAC-{scenario_id}] Model saved to {save_path}")
        except Exception as e:
            print(f"  [SAC-{scenario_id}] Training failed: {e}")
            print(f"  [SAC-{scenario_id}] Running with random policy")
    elif model_path:
        sac.load(model_path)

    # 평가
    obs, _ = env.reset()
    sac.reset()
    total_ctrl_steps = int(cfg["duration_s"] / DT_CTRL)

    t0 = time.time()
    for step_i in range(total_ctrl_steps):
        P_fc_star, a_norm = sac.predict(obs, deterministic=True)
        P_fc_star = sac.apply_ramp_limit(P_fc_star, DT_CTRL)

        a_norm_clipped = 2.0 * (P_fc_star - 5.0) / 77.0 - 1.0
        a_norm_clipped = np.clip(a_norm_clipped, -1.0, 1.0)

        obs, reward, terminated, truncated, info = env.step(
            np.array([a_norm_clipped])
        )

        if terminated or truncated:
            break

        if (step_i + 1) % (total_ctrl_steps // 10) == 0:
            pct = (step_i + 1) / total_ctrl_steps * 100
            elapsed = time.time() - t0
            print(f"    [{scenario_id}-SAC] {pct:.0f}% ({elapsed:.0f}s)")

    return _extract_result(env, scenario_id, "sac")


def run_all_instances(train_sac: bool = True,
                      train_timesteps: int = 500_000) -> list:
    """
    12 인스턴스 순차 실행 (규칙 6 + SAC 6)
    """
    results = []
    dataset_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "dataset")
    os.makedirs(dataset_dir, exist_ok=True)

    # Rule-based 전체 시나리오
    for sid in SCENARIO_IDS:
        cfg = SCENARIOS[sid]
        duration_h = cfg["duration_s"] // 3600
        print(f"\n{'='*60}")
        print(f"  [{sid}] {cfg['name']} — Rule-based EMS")
        print(f"  dt_sim={DT_SIM}s, dt_ctrl={DT_CTRL}s, "
              f"substeps={int(DT_CTRL/DT_SIM)}")
        print(f"{'='*60}")

        t0 = time.time()
        result = run_rule_instance(sid)
        elapsed = time.time() - t0
        results.append(result)

        # CSV 저장
        df = pd.DataFrame(result["log"])
        csv_name = f"{sid}_rule_{cfg['cycle']}_{duration_h}h.csv"
        df.to_csv(os.path.join(dataset_dir, csv_name), index=False)
        print(f"  -> Distance: {result['distance_km']:.1f} km, "
              f"H2: {result['total_H2_kg']:.3f} kg, "
              f"SOC: {result['final_SOC']:.3f}, "
              f"Time: {elapsed:.0f}s")

    # SAC 전체 시나리오
    for sid in SCENARIO_IDS:
        cfg = SCENARIOS[sid]
        duration_h = cfg["duration_s"] // 3600
        print(f"\n{'='*60}")
        print(f"  [{sid}] {cfg['name']} — SAC EMS")
        print(f"  dt_sim={DT_SIM}s, dt_ctrl={DT_CTRL}s, "
              f"train_steps={train_timesteps:,}")
        print(f"{'='*60}")

        t0 = time.time()
        result = run_sac_instance(sid, train=train_sac,
                                   train_timesteps=train_timesteps)
        elapsed = time.time() - t0
        results.append(result)

        # CSV 저장
        df = pd.DataFrame(result["log"])
        csv_name = f"{sid}_sac_{cfg['cycle']}_{duration_h}h.csv"
        df.to_csv(os.path.join(dataset_dir, csv_name), index=False)
        print(f"  -> Distance: {result['distance_km']:.1f} km, "
              f"H2: {result['total_H2_kg']:.3f} kg, "
              f"SOC: {result['final_SOC']:.3f}, "
              f"Time: {elapsed:.0f}s")

    return results
