"""
Sery & Leduc 기준선 검증기
- UDDS: 0.67 +/- 0.05 kg/100km
- HWFET: 0.72 +/- 0.05 kg/100km
- 공전: 12.4 +/- 2.0 g/h
- FC 효율: eta@10kWe = 66.8 +/- 2%
6 시나리오 (A~F) 지원
"""

import numpy as np
import pandas as pd

from models.pemfc_model import eta_fcs
from scenarios.scenario_configs import SCENARIO_IDS


def validate_result(result: dict) -> dict:
    """단일 결과 검증"""
    sid = result["scenario_id"]
    ems = result["ems_type"]
    dist_km = result["distance_km"]
    h2_kg = result["total_H2_kg"]

    checks = {}

    # 수소 소비율
    if dist_km > 0:
        fc_per_100km = h2_kg / dist_km * 100
    else:
        fc_per_100km = float("inf")
    checks["fc_per_100km"] = fc_per_100km

    # km/kg H2
    if h2_kg > 0:
        km_per_kg = dist_km / h2_kg
    else:
        km_per_kg = 0.0
    checks["km_per_kg_H2"] = km_per_kg

    # UDDS 기준선 검증 (A, C, E)
    if sid in ["A", "C", "E"]:
        ref = 0.67
        tol = 0.05
        checks["fc_ref"] = ref
        checks["fc_tol"] = tol
        checks["fc_pass"] = abs(fc_per_100km - ref) <= tol if ems == "rule" else True
    # HWFET 기준선 검증 (B, D, F)
    elif sid in ["B", "D", "F"]:
        ref = 0.72
        tol = 0.05
        checks["fc_ref"] = ref
        checks["fc_tol"] = tol
        checks["fc_pass"] = abs(fc_per_100km - ref) <= tol if ems == "rule" else True

    # FC 효율 검증
    eta_at_10 = eta_fcs(10.0)
    checks["eta_at_10kwe"] = eta_at_10
    checks["eta_pass"] = abs(eta_at_10 - 0.668) <= 0.02

    # 장시간 시나리오 완주 검증
    if sid in ["D", "F"]:
        duration_h = {"D": 10, "F": 24}[sid]
        checks[f"completed_{duration_h}h"] = not result["terminated_early"]

    checks["terminated_early"] = result["terminated_early"]

    return checks


def generate_comparison_report(results: list) -> pd.DataFrame:
    """12 인스턴스 비교 보고서 생성"""
    rows = []

    for r in results:
        v = validate_result(r)
        row = {
            "scenario": r["scenario_id"],
            "ems": r["ems_type"],
            "distance_km": round(r["distance_km"], 1),
            "H2_consumed_kg": round(r["total_H2_kg"], 3),
            "fc_per_100km": round(v["fc_per_100km"], 4),
            "km_per_kg_H2": round(v["km_per_kg_H2"], 1),
            "final_SOC": round(r["final_SOC"], 3),
            "final_SOH_fc": round(r["final_SOH_fc"], 5),
            "final_SOH_bat": round(r["final_SOH_bat"], 5),
            "eta_at_10kwe": round(v["eta_at_10kwe"], 4),
            "terminated_early": r["terminated_early"],
        }

        # 장시간 완주 체크
        if r["scenario_id"] == "D":
            row["completed_10h"] = v.get("completed_10h", None)
        if r["scenario_id"] == "F":
            row["completed_24h"] = v.get("completed_24h", None)

        rows.append(row)

    df = pd.DataFrame(rows)

    # SAC 개선율 계산 (모든 시나리오)
    for sid in SCENARIO_IDS:
        rule_rows = df[(df["scenario"] == sid) & (df["ems"] == "rule")]
        sac_rows = df[(df["scenario"] == sid) & (df["ems"] == "sac")]
        if not rule_rows.empty and not sac_rows.empty:
            rule_km_kg = rule_rows.iloc[0]["km_per_kg_H2"]
            sac_km_kg = sac_rows.iloc[0]["km_per_kg_H2"]
            if rule_km_kg > 0:
                improvement = (sac_km_kg - rule_km_kg) / rule_km_kg * 100
                idx = sac_rows.index[0]
                df.loc[idx, "improvement_pct"] = round(improvement, 2)

    return df
