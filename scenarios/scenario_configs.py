"""
6개 시나리오 설정 — 동일 환경 조건 공유
UDDS/HWFET x 5h/10h/24h = 6 시나리오
모든 시나리오: T_amb=23C, SOC_init=0.60, H2=6.33kg, 700bar x 3탱크
"""

# 공통 환경 파라미터 (모든 시나리오가 반드시 동일 조건 공유)
SHARED_ENV = {
    "temperature_C": 23.0,
    "T_amb_K": 296.15,
    "SOC_init": 0.60,
    "m_H2_init_kg": 6.33,
    "P_tank_init_bar": 700.0,
    "humidity_pct": 50.0,
    "altitude_m": 0.0,
    "grade_rad": 0.0,
    "P_aux_kW": 1.5,
}

SCENARIOS = {
    # ── 5시간 ──
    "A": {
        "name": "시내 UDDS 5시간",
        "cycle": "UDDS",
        "duration_s": 18_000,
        "include_idle": True,
        "fc_per_100km_ref": 0.67,
        "expected_distance_km": 322,
        "expected_h2_kg": 2.16,
        "tank_fraction": 0.34,
        **SHARED_ENV,
    },
    "B": {
        "name": "고속 HWFET 5시간",
        "cycle": "HWFET",
        "duration_s": 18_000,
        "include_idle": False,
        "fc_per_100km_ref": 0.72,
        "expected_distance_km": 380,
        "expected_h2_kg": 2.74,
        "tank_fraction": 0.43,
        **SHARED_ENV,
    },
    # ── 10시간 ──
    "C": {
        "name": "시내 UDDS 10시간",
        "cycle": "UDDS",
        "duration_s": 36_000,
        "include_idle": True,
        "fc_per_100km_ref": 0.67,
        "expected_distance_km": 640,
        "expected_h2_kg": 4.29,
        "tank_fraction": 0.68,
        **SHARED_ENV,
    },
    "D": {
        "name": "고속 HWFET 10시간",
        "cycle": "HWFET",
        "duration_s": 36_000,
        "include_idle": False,
        "fc_per_100km_ref": 0.72,
        "expected_distance_km": 776,
        "expected_h2_kg": 5.59,
        "tank_fraction": 0.88,
        "critical": True,
        **SHARED_ENV,
    },
    # ── 24시간 ──
    "E": {
        "name": "시내 UDDS 24시간",
        "cycle": "UDDS",
        "duration_s": 86_400,
        "include_idle": True,
        "fc_per_100km_ref": 0.67,
        "expected_distance_km": 1546,
        "expected_h2_kg": 6.33,
        "tank_fraction": 1.00,
        **SHARED_ENV,
    },
    "F": {
        "name": "고속 HWFET 24시간",
        "cycle": "HWFET",
        "duration_s": 86_400,
        "include_idle": False,
        "fc_per_100km_ref": 0.72,
        "expected_distance_km": 1862,
        "expected_h2_kg": 6.33,
        "tank_fraction": 1.00,
        "critical": True,
        **SHARED_ENV,
    },
}

# 시나리오 ID 목록 (실행 순서)
SCENARIO_IDS = list(SCENARIOS.keys())
