#!/usr/bin/env python3
"""
FCHEV Multi-Algorithm RL Tournament Training Pipeline

4단계 학습 파이프라인:
  Phase 1 — Screening: 모든 RL 알고리즘을 짧은 스텝으로 학습
  Phase 2 — Tournament: 시나리오 A에서 평가, km/kg H2로 순위 매김
  Phase 3 — Focus Training: 상위 3개 알고리즘 집중 학습 (커리큘럼 + 긴 스텝)
  Phase 4 — Final Evaluation: 최고 에이전트를 전체 6개 시나리오에서 평가

사용법:
    python rl_train.py                             # 전체 파이프라인
    python rl_train.py --phase 1                   # 스크리닝만
    python rl_train.py --phase 3 --algos SAC TQC   # 특정 알고리즘 집중 학습
    python rl_train.py --screening-steps 200000    # 스크리닝 스텝 수 조정
    python rl_train.py --focus-steps 2000000       # 집중 학습 스텝 수 조정
    python rl_train.py --reward v3                 # 보상 함수 v3 사용
    python rl_train.py --scenarios A B             # 특정 시나리오만 최종 평가
"""

import os
import sys
import time
import json
import argparse
import traceback

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, PROJECT_ROOT)

import numpy as np
import pandas as pd

from scenarios.scenario_configs import SCENARIOS, SCENARIO_IDS, SHARED_ENV
from pipeline.cycle_loader import load_cycle
from env.env_rl import FCHEVRLEnv, FCHEVCurriculumWrapper
from control.multi_rl import MultiRLAgent, evaluate_agent, get_device
from control.rule_ems import RuleBasedEMS
from pipeline.validator import validate_result

DT_SIM = 0.01
DT_CTRL = 1.0

RESULTS_DIR = os.path.join(PROJECT_ROOT, "rl_results")
MODELS_DIR = os.path.join(RESULTS_DIR, "models")
DATASET_DIR = os.path.join(PROJECT_ROOT, "dataset")


def _make_rl_env(scenario_id, reward_mode="v2", curriculum_duration_s=None):
    """RL 학습용 환경 생성"""
    cfg = SCENARIOS[scenario_id]
    v, a = load_cycle(cfg["cycle"], cfg["duration_s"], dt=DT_SIM)
    env = FCHEVRLEnv(
        cycle_v=v, cycle_a=a,
        duration_s=cfg["duration_s"],
        dt_sim=DT_SIM, dt_ctrl=DT_CTRL,
        T_amb_C=cfg["temperature_C"],
        SOC_init=cfg["SOC_init"],
        scenario_id=scenario_id,
        reward_mode=reward_mode,
        curriculum_duration_s=curriculum_duration_s,
    )
    return env


def _make_eval_env(scenario_id, reward_mode="v2"):
    """평가용 환경 (full duration)"""
    return _make_rl_env(scenario_id, reward_mode=reward_mode)


def _run_rule_baseline(scenario_id):
    """규칙 기반 베이스라인 실행 (비교 기준)"""
    from env.env import FCHEVEnv
    cfg = SCENARIOS[scenario_id]
    v, a = load_cycle(cfg["cycle"], cfg["duration_s"], dt=DT_SIM)
    env = FCHEVEnv(
        cycle_v=v, cycle_a=a,
        duration_s=cfg["duration_s"],
        dt_sim=DT_SIM, dt_ctrl=DT_CTRL,
        T_amb_C=cfg["temperature_C"],
        SOC_init=cfg["SOC_init"],
        scenario_id=scenario_id,
    )
    obs, _ = env.reset()
    rule = RuleBasedEMS()
    total_steps = int(cfg["duration_s"] / DT_CTRL)

    for _ in range(total_steps):
        P_low_kw = obs[0] * 82.0
        v_ms = obs[7] * 36.0
        SOC = obs[1]
        res = rule.compute(P_low_kw, SOC, v_ms, DT_CTRL)
        a_norm = 2.0 * (res["P_fc_star"] - 5.0) / 77.0 - 1.0
        a_norm = np.clip(a_norm, -1.0, 1.0)
        obs, _, terminated, truncated, _ = env.step(np.array([a_norm]))
        if terminated or truncated:
            break

    dist_km = env.distance_m / 1e3
    h2_kg = env.total_H2_kg
    return {
        "distance_km": dist_km,
        "h2_kg": h2_kg,
        "km_per_kg": dist_km / max(h2_kg, 1e-9),
        "fc_per_100km": h2_kg / max(dist_km, 1e-9) * 100,
        "final_SOC": env.battery.SOC,
        "final_SOH_fc": env.pemfc.SOH_fc,
        "completed": not env.terminated,
        "log": env.get_log_dict(),
    }


# ═══════════════════════════════════════════════
# Phase 1: Screening — 모든 알고리즘 짧은 학습
# ═══════════════════════════════════════════════
def phase1_screening(scenario_id="A", reward_mode="v2",
                     screening_steps=150_000, reward_modes=None):
    """
    Phase 1: 모든 사용 가능한 RL 알고리즘을 짧게 학습

    Returns:
        list[dict]: 각 알고리즘의 학습 결과
    """
    print("\n" + "=" * 70)
    print("  PHASE 1: SCREENING — 모든 알고리즘 스크리닝 학습")
    print("=" * 70)

    all_algos = MultiRLAgent.get_all_algo_names()
    if reward_modes is None:
        reward_modes = [reward_mode]

    results = []
    total_combos = len(all_algos) * len(reward_modes)
    combo_idx = 0

    for rm in reward_modes:
        for algo_name in all_algos:
            combo_idx += 1
            print(f"\n{'─' * 60}")
            print(f"  [{combo_idx}/{total_combos}] {algo_name} (reward={rm})")
            print(f"  Screening: {screening_steps:,} steps on scenario {scenario_id}")
            print(f"{'─' * 60}")

            try:
                # 학습 환경 (커리큘럼 적용)
                train_env = _make_rl_env(scenario_id, reward_mode=rm,
                                         curriculum_duration_s=3600)
                curriculum_env = FCHEVCurriculumWrapper(train_env, stages=[
                    (0, 1800),
                    (30000, 3600),
                    (60000, 7200),
                    (100000, 14400),
                    (120000, 18000),
                ])

                agent = MultiRLAgent(algo_name, curriculum_env)
                agent.build()

                t0 = time.time()
                train_stats = agent.train(screening_steps, progress_bar=True)
                train_time = time.time() - t0

                # 평가
                eval_env = _make_eval_env(scenario_id, reward_mode=rm)
                eval_result = evaluate_agent(agent, eval_env)

                # 모델 저장
                model_path = os.path.join(
                    MODELS_DIR, f"screen_{algo_name}_{rm}_{scenario_id}")
                agent.save(model_path)

                result = {
                    "algo": algo_name,
                    "reward_mode": rm,
                    "scenario": scenario_id,
                    "phase": "screening",
                    "steps": screening_steps,
                    "train_time_s": train_time,
                    "model_path": model_path,
                    **eval_result,
                    **{f"train_{k}": v for k, v in train_stats.items()},
                }
                results.append(result)

                print(f"  Result: {eval_result['km_per_kg']:.1f} km/kg H2, "
                      f"dist={eval_result['distance_km']:.1f} km, "
                      f"H2={eval_result['h2_kg']:.3f} kg, "
                      f"completed={eval_result['completed']}, "
                      f"time={train_time:.0f}s")

            except Exception as e:
                print(f"  [ERROR] {algo_name}: {e}")
                traceback.print_exc()
                results.append({
                    "algo": algo_name,
                    "reward_mode": rm,
                    "scenario": scenario_id,
                    "phase": "screening",
                    "steps": screening_steps,
                    "km_per_kg": 0.0,
                    "error": str(e),
                })

    return results


# ═══════════════════════════════════════════════
# Phase 2: Tournament — 성능 순위 결정
# ═══════════════════════════════════════════════
def phase2_tournament(screening_results, rule_baseline):
    """
    Phase 2: 스크리닝 결과를 km/kg H2 기준으로 순위 매김

    Returns:
        list[dict]: 순위별 정렬된 결과 (상위 → 하위)
    """
    print("\n" + "=" * 70)
    print("  PHASE 2: TOURNAMENT — 성능 순위 결정")
    print("=" * 70)

    valid_results = [r for r in screening_results if "error" not in r]
    valid_results.sort(key=lambda x: x.get("km_per_kg", 0), reverse=True)

    rule_km_kg = rule_baseline["km_per_kg"]
    print(f"\n  Rule Baseline: {rule_km_kg:.1f} km/kg H2")
    print(f"{'─' * 70}")
    print(f"  {'Rank':<6} {'Algorithm':<10} {'Reward':<8} {'km/kg H2':>10} "
          f"{'vs Rule':>10} {'Distance':>10} {'H2 (kg)':>10} {'Done':>6}")
    print(f"{'─' * 70}")

    for i, r in enumerate(valid_results):
        km_kg = r.get("km_per_kg", 0)
        vs_rule = ((km_kg - rule_km_kg) / rule_km_kg * 100) if rule_km_kg > 0 else 0
        print(f"  {i+1:<6} {r['algo']:<10} {r['reward_mode']:<8} "
              f"{km_kg:>10.1f} {vs_rule:>+9.1f}% "
              f"{r.get('distance_km', 0):>10.1f} "
              f"{r.get('h2_kg', 0):>10.3f} "
              f"{'Y' if r.get('completed', False) else 'N':>6}")

    print(f"{'─' * 70}")

    # 상위 선정 (최소 3개, 최대 5개)
    n_select = min(max(3, len(valid_results) // 2), 5, len(valid_results))
    selected = valid_results[:n_select]

    print(f"\n  Top {n_select} selected for Phase 3 focus training:")
    for r in selected:
        print(f"    -> {r['algo']} (reward={r['reward_mode']}, "
              f"{r.get('km_per_kg', 0):.1f} km/kg)")

    return valid_results, selected


# ═══════════════════════════════════════════════
# Phase 3: Focus Training — 상위 알고리즘 집중 학습
# ═══════════════════════════════════════════════
def phase3_focus_training(selected_combos, scenario_id="A",
                          focus_steps=800_000):
    """
    Phase 3: 상위 알고리즘들을 긴 스텝 + 커리큘럼으로 집중 학습

    Returns:
        list[dict]: 집중 학습 결과
    """
    print("\n" + "=" * 70)
    print("  PHASE 3: FOCUS TRAINING — 상위 알고리즘 집중 학습")
    print(f"  Steps: {focus_steps:,} per algorithm")
    print("=" * 70)

    results = []

    for idx, combo in enumerate(selected_combos):
        algo_name = combo["algo"]
        rm = combo["reward_mode"]
        print(f"\n{'─' * 60}")
        print(f"  [{idx+1}/{len(selected_combos)}] {algo_name} (reward={rm})")
        print(f"  Focus: {focus_steps:,} steps with curriculum")
        print(f"{'─' * 60}")

        try:
            # 커리큘럼 학습 환경 (단계적 에피소드 확장)
            cfg = SCENARIOS[scenario_id]
            full_dur = cfg["duration_s"]
            train_env = _make_rl_env(scenario_id, reward_mode=rm,
                                     curriculum_duration_s=1800)

            # 더 세밀한 커리큘럼 스테이지
            frac = focus_steps
            curriculum_env = FCHEVCurriculumWrapper(train_env, stages=[
                (0, 1800),                          # 30분
                (int(frac * 0.05), 3600),            # 1시간
                (int(frac * 0.10), 5400),            # 1.5시간
                (int(frac * 0.15), 7200),            # 2시간
                (int(frac * 0.25), 10800),           # 3시간
                (int(frac * 0.35), 14400),           # 4시간
                (int(frac * 0.50), min(18000, full_dur)),  # 5시간
                (int(frac * 0.70), min(36000, full_dur)),  # 10시간 (if applicable)
                (int(frac * 0.85), full_dur),        # full
            ])

            agent = MultiRLAgent(algo_name, curriculum_env)
            agent.build()

            # 기존 스크리닝 모델 로드 시도
            screen_path = combo.get("model_path", "")
            if screen_path and os.path.exists(screen_path + ".zip"):
                try:
                    agent.load(screen_path)
                    agent.model.set_env(curriculum_env)
                    print(f"  Loaded screening model, continuing training...")
                except Exception:
                    print(f"  Could not load screening model, training from scratch")

            t0 = time.time()
            train_stats = agent.train(focus_steps, progress_bar=True)
            train_time = time.time() - t0

            # 평가 (full scenario)
            eval_env = _make_eval_env(scenario_id, reward_mode=rm)
            eval_result = evaluate_agent(agent, eval_env)

            # 모델 저장
            model_path = os.path.join(
                MODELS_DIR, f"focus_{algo_name}_{rm}_{scenario_id}")
            agent.save(model_path)

            result = {
                "algo": algo_name,
                "reward_mode": rm,
                "scenario": scenario_id,
                "phase": "focus",
                "steps": focus_steps,
                "train_time_s": train_time,
                "model_path": model_path,
                **eval_result,
                **{f"train_{k}": v for k, v in train_stats.items()},
            }
            results.append(result)

            print(f"  Result: {eval_result['km_per_kg']:.1f} km/kg H2, "
                  f"dist={eval_result['distance_km']:.1f} km, "
                  f"H2={eval_result['h2_kg']:.3f} kg, "
                  f"completed={eval_result['completed']}, "
                  f"time={train_time:.0f}s")

        except Exception as e:
            print(f"  [ERROR] {algo_name}: {e}")
            traceback.print_exc()
            results.append({
                "algo": algo_name,
                "reward_mode": rm,
                "scenario": scenario_id,
                "phase": "focus",
                "steps": focus_steps,
                "km_per_kg": 0.0,
                "error": str(e),
            })

    # 최종 순위
    valid = [r for r in results if "error" not in r]
    valid.sort(key=lambda x: x.get("km_per_kg", 0), reverse=True)

    print(f"\n{'─' * 60}")
    print("  Focus Training Results:")
    for i, r in enumerate(valid):
        print(f"  {i+1}. {r['algo']} (reward={r['reward_mode']}): "
              f"{r['km_per_kg']:.1f} km/kg H2")
    print(f"{'─' * 60}")

    return valid


# ═══════════════════════════════════════════════
# Phase 4: Final Evaluation — 전체 시나리오 평가
# ═══════════════════════════════════════════════
def phase4_final_evaluation(best_results, scenario_ids=None):
    """
    Phase 4: 최고 에이전트를 전체 시나리오에서 평가, 규칙 기반과 비교

    Returns:
        pd.DataFrame: 전체 비교 보고서
    """
    if scenario_ids is None:
        scenario_ids = SCENARIO_IDS

    print("\n" + "=" * 70)
    print("  PHASE 4: FINAL EVALUATION — 전체 시나리오 평가")
    print("=" * 70)

    # 최고 에이전트 선택 (최대 3개)
    top_agents = best_results[:min(3, len(best_results))]

    all_results = []

    # 규칙 기반 베이스라인
    print("\n  Running Rule-based baseline on all scenarios...")
    for sid in scenario_ids:
        cfg = SCENARIOS[sid]
        print(f"    [{sid}] {cfg['name']}...", end=" ", flush=True)
        t0 = time.time()
        rule_result = _run_rule_baseline(sid)
        elapsed = time.time() - t0
        print(f"{rule_result['km_per_kg']:.1f} km/kg ({elapsed:.0f}s)")

        all_results.append({
            "scenario": sid,
            "ems": "rule",
            "algo": "Rule-based",
            "reward_mode": "-",
            "distance_km": rule_result["distance_km"],
            "h2_kg": rule_result["h2_kg"],
            "km_per_kg": rule_result["km_per_kg"],
            "fc_per_100km": rule_result["fc_per_100km"],
            "final_SOC": rule_result["final_SOC"],
            "completed": rule_result["completed"],
            "log": rule_result["log"],
        })

    # RL 에이전트 평가
    for agent_info in top_agents:
        algo = agent_info["algo"]
        rm = agent_info["reward_mode"]
        model_path = agent_info.get("model_path", "")

        print(f"\n  Evaluating {algo} (reward={rm}) on all scenarios...")

        for sid in scenario_ids:
            cfg = SCENARIOS[sid]
            print(f"    [{sid}] {cfg['name']}...", end=" ", flush=True)

            try:
                eval_env = _make_eval_env(sid, reward_mode=rm)
                agent = MultiRLAgent(algo)

                # 학습된 모델 로드
                if model_path and os.path.exists(model_path + ".zip"):
                    agent.load(model_path)
                else:
                    print(f"[no model, skip]")
                    continue

                t0 = time.time()
                eval_result = evaluate_agent(agent, eval_env)
                elapsed = time.time() - t0

                print(f"{eval_result['km_per_kg']:.1f} km/kg ({elapsed:.0f}s)")

                all_results.append({
                    "scenario": sid,
                    "ems": f"rl_{algo}",
                    "algo": algo,
                    "reward_mode": rm,
                    "distance_km": eval_result["distance_km"],
                    "h2_kg": eval_result["h2_kg"],
                    "km_per_kg": eval_result["km_per_kg"],
                    "fc_per_100km": eval_result["fc_per_100km"],
                    "final_SOC": eval_result["final_SOC"],
                    "completed": eval_result["completed"],
                    "log": eval_env.get_log_dict(),
                })

            except Exception as e:
                print(f"[ERROR: {e}]")

    # 비교 보고서 생성
    df = pd.DataFrame([{k: v for k, v in r.items() if k != "log"}
                        for r in all_results])

    # 개선율 계산
    for sid in scenario_ids:
        rule_rows = df[(df["scenario"] == sid) & (df["ems"] == "rule")]
        if rule_rows.empty:
            continue
        rule_km_kg = rule_rows.iloc[0]["km_per_kg"]
        rl_rows = df[(df["scenario"] == sid) & (df["ems"] != "rule")]
        for idx in rl_rows.index:
            rl_km_kg = df.loc[idx, "km_per_kg"]
            if rule_km_kg > 0:
                df.loc[idx, "improvement_pct"] = round(
                    (rl_km_kg - rule_km_kg) / rule_km_kg * 100, 2)

    return df, all_results


# ═══════════════════════════════════════════════
# CSV 로그 저장
# ═══════════════════════════════════════════════
def save_logs(all_results):
    """시뮬레이션 로그 CSV 저장"""
    os.makedirs(DATASET_DIR, exist_ok=True)
    for r in all_results:
        if "log" not in r or not r["log"]:
            continue
        sid = r["scenario"]
        ems = r["ems"]
        algo = r.get("algo", "unknown")
        cfg = SCENARIOS[sid]
        duration_h = cfg["duration_s"] // 3600

        if ems == "rule":
            csv_name = f"{sid}_rule_{cfg['cycle']}_{duration_h}h.csv"
        else:
            csv_name = f"{sid}_{algo}_{cfg['cycle']}_{duration_h}h.csv"

        df = pd.DataFrame(r["log"])
        df.to_csv(os.path.join(DATASET_DIR, csv_name), index=False)
        print(f"  Saved: {csv_name}")


# ═══════════════════════════════════════════════
# 메인 파이프라인
# ═══════════════════════════════════════════════
def print_banner():
    print("""
+══════════════════════════════════════════════════════════════════+
|  FCHEV Multi-Algorithm RL Tournament Training Pipeline          |
|  8 RL Algorithms × 2 Reward Functions × Curriculum Learning     |
|  SAC · PPO · TD3 · DDPG · A2C · TQC · TRPO · ARS              |
|  Goal: Beat Rule-based EMS (130-175 km/kg H2)                  |
+══════════════════════════════════════════════════════════════════+
    """)


def main():
    parser = argparse.ArgumentParser(
        description="FCHEV Multi-Algorithm RL Tournament")
    parser.add_argument("--phase", type=int, default=0,
                        help="Run specific phase (1-4), 0=all")
    parser.add_argument("--screening-steps", type=int, default=150_000,
                        help="Phase 1 screening steps per algo")
    parser.add_argument("--focus-steps", type=int, default=800_000,
                        help="Phase 3 focus training steps")
    parser.add_argument("--reward", type=str, default="v2",
                        choices=["v2", "v3", "both"],
                        help="Reward function version")
    parser.add_argument("--scenario", type=str, default="A",
                        help="Training scenario (default: A)")
    parser.add_argument("--scenarios", nargs="+", default=None,
                        help="Final eval scenarios (default: all)")
    parser.add_argument("--algos", nargs="+", default=None,
                        help="Specific algorithms for phase 3")
    parser.add_argument("--skip-rule", action="store_true",
                        help="Skip rule baseline (use cached)")
    args = parser.parse_args()

    print_banner()

    # 디렉토리 생성
    os.makedirs(RESULTS_DIR, exist_ok=True)
    os.makedirs(MODELS_DIR, exist_ok=True)
    os.makedirs(DATASET_DIR, exist_ok=True)

    device = get_device()
    print(f"  Device: {device}")
    print(f"  Training scenario: {args.scenario}")

    reward_modes = ["v2", "v3"] if args.reward == "both" else [args.reward]
    print(f"  Reward modes: {reward_modes}")
    print(f"  Available algorithms: {MultiRLAgent.get_all_algo_names()}")

    t_start = time.time()

    # ── 규칙 기반 베이스라인 ──
    print(f"\n  Computing rule baseline for scenario {args.scenario}...")
    rule_baseline = _run_rule_baseline(args.scenario)
    print(f"  Rule baseline: {rule_baseline['km_per_kg']:.1f} km/kg H2, "
          f"dist={rule_baseline['distance_km']:.1f} km, "
          f"H2={rule_baseline['h2_kg']:.3f} kg")

    # ── Phase 1: Screening ──
    if args.phase in (0, 1):
        screening_results = phase1_screening(
            scenario_id=args.scenario,
            reward_mode=args.reward,
            screening_steps=args.screening_steps,
            reward_modes=reward_modes,
        )

        # 결과 저장
        screen_df = pd.DataFrame([
            {k: v for k, v in r.items() if k != "log"}
            for r in screening_results
        ])
        screen_df.to_csv(
            os.path.join(RESULTS_DIR, "phase1_screening.csv"), index=False)

        if args.phase == 1:
            print("\n  Phase 1 complete. Results saved to rl_results/")
            return

    # ── Phase 2: Tournament ──
    if args.phase in (0, 2):
        if args.phase == 2:
            screen_path = os.path.join(RESULTS_DIR, "phase1_screening.csv")
            if os.path.exists(screen_path):
                screen_df = pd.read_csv(screen_path)
                screening_results = screen_df.to_dict("records")
            else:
                print("  [ERROR] No screening results. Run phase 1 first.")
                return

        ranked, selected = phase2_tournament(screening_results, rule_baseline)

        # 결과 저장
        ranked_df = pd.DataFrame([
            {k: v for k, v in r.items() if k != "log"}
            for r in ranked
        ])
        ranked_df.to_csv(
            os.path.join(RESULTS_DIR, "phase2_tournament.csv"), index=False)

        if args.phase == 2:
            print("\n  Phase 2 complete. Results saved to rl_results/")
            return

    # ── Phase 3: Focus Training ──
    if args.phase in (0, 3):
        if args.phase == 3:
            # 특정 알고리즘 지정 또는 tournament 결과 로드
            if args.algos:
                selected = [{"algo": a, "reward_mode": args.reward,
                             "model_path": ""}
                            for a in args.algos]
            else:
                tour_path = os.path.join(RESULTS_DIR, "phase2_tournament.csv")
                if os.path.exists(tour_path):
                    tour_df = pd.read_csv(tour_path)
                    selected = tour_df.head(3).to_dict("records")
                else:
                    print("  [ERROR] No tournament results. Run phase 2 first.")
                    return

        focus_results = phase3_focus_training(
            selected, scenario_id=args.scenario,
            focus_steps=args.focus_steps,
        )

        # 결과 저장
        focus_df = pd.DataFrame([
            {k: v for k, v in r.items() if k != "log"}
            for r in focus_results
        ])
        focus_df.to_csv(
            os.path.join(RESULTS_DIR, "phase3_focus.csv"), index=False)

        if args.phase == 3:
            print("\n  Phase 3 complete. Results saved to rl_results/")
            return

    # ── Phase 4: Final Evaluation ──
    if args.phase in (0, 4):
        if args.phase == 4:
            focus_path = os.path.join(RESULTS_DIR, "phase3_focus.csv")
            if os.path.exists(focus_path):
                focus_df = pd.read_csv(focus_path)
                focus_results = focus_df.to_dict("records")
            else:
                print("  [ERROR] No focus results. Run phase 3 first.")
                return

        eval_scenarios = args.scenarios or SCENARIO_IDS
        report_df, all_results = phase4_final_evaluation(
            focus_results, eval_scenarios)

        # 보고서 출력
        print(f"\n{'=' * 80}")
        print("  FINAL COMPARISON REPORT")
        print(f"{'=' * 80}")
        print(report_df.to_string(index=False))

        # 핵심 지표 요약
        print(f"\n{'─' * 80}")
        print("  Key Metrics Summary:")
        print(f"{'─' * 80}")
        for sid in eval_scenarios:
            rule_rows = report_df[(report_df["scenario"] == sid) &
                                  (report_df["ems"] == "rule")]
            rl_rows = report_df[(report_df["scenario"] == sid) &
                                (report_df["ems"] != "rule")]

            if not rule_rows.empty:
                r = rule_rows.iloc[0]
                print(f"  [{sid}-Rule] {r['km_per_kg']:.1f} km/kg H2 "
                      f"({r['fc_per_100km']:.3f} kg/100km)")

            for _, r in rl_rows.iterrows():
                imp = r.get("improvement_pct", 0)
                print(f"  [{sid}-{r['algo']:>4s}] {r['km_per_kg']:.1f} km/kg H2 "
                      f"({r['fc_per_100km']:.3f} kg/100km) "
                      f"Improvement: {imp:+.2f}%")

        # 저장
        report_df.to_csv(
            os.path.join(RESULTS_DIR, "final_comparison.csv"), index=False)
        save_logs(all_results)

        # 시각화 실행
        try:
            from visualize_rl import main as viz_main
            viz_main(report_df, all_results)
        except Exception as e:
            print(f"\n  [WARN] Visualization skipped: {e}")

    # 전체 소요 시간
    elapsed = time.time() - t_start
    hours = int(elapsed // 3600)
    mins = int((elapsed % 3600) // 60)
    secs = int(elapsed % 60)
    print(f"\n{'=' * 70}")
    print(f"  Total elapsed: {hours}h {mins}m {secs}s")
    print(f"  Results: {RESULTS_DIR}/")
    print(f"  Models:  {MODELS_DIR}/")
    print(f"  Dataset: {DATASET_DIR}/")
    print(f"{'=' * 70}")


if __name__ == "__main__":
    main()
