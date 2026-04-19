#!/usr/bin/env python3
"""
FCHEV 완전 물리 시뮬레이터 & SAC EMS 비교 플랫폼 v4.0
고정밀 서브스테핑 + GPU 가속 + 24시간 시나리오

사용법:
    python fchev_pipeline.py                          # 전체 실행 (6시나리오 x 2 EMS)
    python fchev_pipeline.py --rule-only              # 규칙 기반만
    python fchev_pipeline.py --no-train               # SAC 학습 없이
    python fchev_pipeline.py --timesteps 1000000      # SAC 학습 스텝 조정
    python fchev_pipeline.py --scenarios A B           # 특정 시나리오만
"""

import os
import sys
import argparse
import time

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, PROJECT_ROOT)

import numpy as np
import pandas as pd

from scenarios.scenario_configs import SCENARIOS, SCENARIO_IDS, SHARED_ENV
from pipeline.runner import (
    run_rule_instance, run_sac_instance, run_all_instances,
    DT_SIM, DT_CTRL
)
from pipeline.validator import validate_result, generate_comparison_report


def _check_gpu():
    """GPU 상태 확인"""
    try:
        import torch
        if torch.cuda.is_available():
            name = torch.cuda.get_device_name(0)
            vram = torch.cuda.get_device_properties(0).total_mem / 1e9
            print(f"  GPU: {name} ({vram:.1f} GB VRAM)")
            return True
        else:
            print("  GPU: Not available (CPU mode)")
            return False
    except ImportError:
        print("  GPU: PyTorch not installed (CPU mode)")
        return False


def print_banner():
    print("""
+==============================================================+
|  FCHEV 완전 물리 시뮬레이터 & SAC EMS 비교 플랫폼  v4.0       |
|  고정밀 서브스테핑: dt_sim=0.01s, dt_ctrl=1.0s (100x)         |
|  H2 Tank(700bar) -> PEMFC -> BoP -> DC Bus -> Battery -> Wheel |
|  참조: Sery & Leduc (2022), Int. J. Engine Research            |
+==============================================================+
    """)


def print_config():
    print(f"  [Config] dt_sim = {DT_SIM}s (내부 물리), "
          f"dt_ctrl = {DT_CTRL}s (제어/로그)")
    print(f"  [Config] substeps = {int(DT_CTRL / DT_SIM)} per ctrl step")
    print(f"  [Config] 공통 환경: T={SHARED_ENV['temperature_C']}C, "
          f"SOC_init={SHARED_ENV['SOC_init']}, "
          f"H2={SHARED_ENV['m_H2_init_kg']}kg")
    _check_gpu()


def print_scenario_table(scenario_ids=None):
    if scenario_ids is None:
        scenario_ids = SCENARIO_IDS
    print("\n+------+------------------+-------+--------+----------+--------+")
    print("|  ID  |      환경        | 사이클 | 시간   | 예상거리  | 예상H2  |")
    print("+------+------------------+-------+--------+----------+--------+")
    for sid in scenario_ids:
        cfg = SCENARIOS[sid]
        crit = " !" if cfg.get("critical") else "  "
        print(f"|  {sid}{crit} | {cfg['name']:<16s} | {cfg['cycle']:<5s} | "
              f"{cfg['duration_s']//3600:>4d}h  | {cfg['expected_distance_km']:>6d} km | "
              f"{cfg['expected_h2_kg']:>5.2f} kg|")
    print("+------+------------------+-------+--------+----------+--------+\n")


def run_selected(scenario_ids: list, rule_only: bool = False,
                 train_sac: bool = True, train_timesteps: int = 500_000):
    """선택된 시나리오 실행"""
    results = []
    dataset_dir = os.path.join(PROJECT_ROOT, "dataset")
    os.makedirs(dataset_dir, exist_ok=True)

    # Rule-based
    for sid in scenario_ids:
        cfg = SCENARIOS[sid]
        duration_h = cfg["duration_s"] // 3600
        print(f"\n{'='*60}")
        print(f"  [{sid}] {cfg['name']} -- Rule-based EMS")
        print(f"{'='*60}")

        t0 = time.time()
        result = run_rule_instance(sid)
        elapsed = time.time() - t0
        results.append(result)

        df = pd.DataFrame(result["log"])
        csv_name = f"{sid}_rule_{cfg['cycle']}_{duration_h}h.csv"
        df.to_csv(os.path.join(dataset_dir, csv_name), index=False)

        v = validate_result(result)
        print(f"  Distance   : {result['distance_km']:.1f} km")
        print(f"  H2 consumed: {result['total_H2_kg']:.3f} kg")
        print(f"  FC/100km   : {v['fc_per_100km']:.4f} kg/100km")
        print(f"  km/kg H2   : {v['km_per_kg_H2']:.1f}")
        print(f"  Final SOC  : {result['final_SOC']:.3f}")
        print(f"  SOH_fc     : {result['final_SOH_fc']:.5f}")
        print(f"  Terminated : {result['terminated_early']}")
        print(f"  Elapsed    : {elapsed:.1f}s")

    # SAC
    if not rule_only:
        for sid in scenario_ids:
            cfg = SCENARIOS[sid]
            duration_h = cfg["duration_s"] // 3600
            print(f"\n{'='*60}")
            print(f"  [{sid}] {cfg['name']} -- SAC EMS")
            print(f"{'='*60}")

            t0 = time.time()
            result = run_sac_instance(
                sid, train=train_sac,
                train_timesteps=train_timesteps
            )
            elapsed = time.time() - t0
            results.append(result)

            df = pd.DataFrame(result["log"])
            csv_name = f"{sid}_sac_{cfg['cycle']}_{duration_h}h.csv"
            df.to_csv(os.path.join(dataset_dir, csv_name), index=False)

            v = validate_result(result)
            print(f"  Distance   : {result['distance_km']:.1f} km")
            print(f"  H2 consumed: {result['total_H2_kg']:.3f} kg")
            print(f"  FC/100km   : {v['fc_per_100km']:.4f} kg/100km")
            print(f"  km/kg H2   : {v['km_per_kg_H2']:.1f}")
            print(f"  Final SOC  : {result['final_SOC']:.3f}")
            print(f"  SOH_fc     : {result['final_SOH_fc']:.5f}")
            print(f"  Terminated : {result['terminated_early']}")
            print(f"  Elapsed    : {elapsed:.1f}s")

    return results


def main():
    parser = argparse.ArgumentParser(description="FCHEV Simulator v4.0 & SAC EMS")
    parser.add_argument("--rule-only", action="store_true",
                        help="규칙 기반 EMS만 실행")
    parser.add_argument("--no-train", action="store_true",
                        help="SAC 학습 없이 실행 (랜덤 정책)")
    parser.add_argument("--timesteps", type=int, default=500_000,
                        help="SAC 학습 스텝 수 (기본: 500,000)")
    parser.add_argument("--scenarios", nargs="+", default=None,
                        help="실행할 시나리오 (예: A B D F)")
    args = parser.parse_args()

    print_banner()
    print_config()

    scenario_ids = args.scenarios if args.scenarios else SCENARIO_IDS
    # 유효성 검사
    for sid in scenario_ids:
        if sid not in SCENARIOS:
            print(f"  [ERROR] Unknown scenario: {sid}")
            print(f"  Available: {SCENARIO_IDS}")
            sys.exit(1)

    print_scenario_table(scenario_ids)

    dataset_dir = os.path.join(PROJECT_ROOT, "dataset")
    os.makedirs(dataset_dir, exist_ok=True)

    t_start = time.time()

    results = run_selected(
        scenario_ids=scenario_ids,
        rule_only=args.rule_only,
        train_sac=not args.no_train,
        train_timesteps=args.timesteps,
    )

    # 비교 보고서
    print(f"\n{'='*60}")
    print("  Comparison Report")
    print(f"{'='*60}\n")

    report = generate_comparison_report(results)
    report_path = os.path.join(dataset_dir, "comparison_report.csv")
    report.to_csv(report_path, index=False)
    print(report.to_string(index=False))

    # 핵심 지표
    print(f"\n{'-'*60}")
    print("  Key Metrics")
    print(f"{'-'*60}")

    for sid in scenario_ids:
        rule_r = [r for r in results if r["scenario_id"] == sid and r["ems_type"] == "rule"]
        sac_r = [r for r in results if r["scenario_id"] == sid and r["ems_type"] == "sac"]

        if rule_r:
            r = rule_r[0]
            v = validate_result(r)
            print(f"  [{sid}-Rule] {v['km_per_kg_H2']:.1f} km/kg H2  "
                  f"({v['fc_per_100km']:.3f} kg/100km)")
        if sac_r:
            r = sac_r[0]
            v = validate_result(r)
            imp_row = report[(report["scenario"] == sid) & (report["ems"] == "sac")]
            imp = imp_row["improvement_pct"].values[0] if "improvement_pct" in imp_row.columns and not imp_row.empty else 0
            print(f"  [{sid}-SAC]  {v['km_per_kg_H2']:.1f} km/kg H2  "
                  f"({v['fc_per_100km']:.3f} kg/100km)  "
                  f"Improvement: {imp:+.2f}%")

    # 24시간 시나리오 완주 체크
    long_results = [r for r in results if r["scenario_id"] in ["E", "F"]]
    if long_results:
        print(f"\n{'-'*60}")
        print("  24h Scenarios -- Completion Check")
        print(f"{'-'*60}")
        for r in long_results:
            status = "COMPLETED" if not r["terminated_early"] else "TERMINATED (H2 depleted)"
            print(f"  [{r['scenario_id']}-{r['ems_type'].upper():>4s}] {status}  "
                  f"(H2: {r['total_H2_kg']:.3f}/{SHARED_ENV['m_H2_init_kg']} kg, "
                  f"Dist: {r['distance_km']:.1f} km)")

    elapsed_total = time.time() - t_start
    hours = int(elapsed_total // 3600)
    mins = int((elapsed_total % 3600) // 60)
    secs = int(elapsed_total % 60)
    print(f"\n  Total elapsed: {hours}h {mins}m {secs}s")
    print(f"  Report saved: {report_path}")
    print(f"  CSV files: {dataset_dir}/")


if __name__ == "__main__":
    main()
