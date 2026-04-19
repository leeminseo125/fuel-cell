"""
기어박스 + 휠 동역학 — Sery & Leduc Table 2 로드로드 파라미터
수식 (3.33)~(3.35)
Backward 방식: v(t) 고정 → P_wheel 역산
"""

import numpy as np

# Sery & Leduc Table 2
VEHICLE_PARAMS = {
    "m_v":    2057.0,      # kg
    "a_rl":   178.7,       # N
    "b_rl":   0.919,       # N/(km/h)
    "c_rl":   0.04037,     # N/(km/h)²
    "eta_dt": 0.93,        # 기어박스+디퍼렌셜
    "r_t":    0.346,       # m (235/55R19)
}

# 수소 탱크 파라미터
TANK_PARAMS = {
    "n_tanks":    3,
    "V_per_tank": 0.0522,
    "V_tank":     0.1566,
    "m_H2_0":     6.33,
    "GS_tank":    70.0,
    "P_min_bar":  10.0,
}

# 기어비
GEAR_RATIO = 7.981     # 감속비 (넥쏘)


def compute_P_req(v_ms: float, a: float, grade: float = 0.0) -> float:
    """
    Backward 계산: v(t), a(t) → P_wheel [W]
    v_ms  : 차량 속도 [m/s]
    a     : 가속도 [m/s²]
    grade : 경사각 [rad]
    """
    m_v = VEHICLE_PARAMS["m_v"]
    v_kmh = v_ms * 3.6

    # 로드로드 저항 (3.33)
    F_rl = (VEHICLE_PARAMS["a_rl"]
            + VEHICLE_PARAMS["b_rl"] * v_kmh
            + VEHICLE_PARAMS["c_rl"] * v_kmh ** 2)

    # 경사 저항
    F_grade = m_v * 9.81 * np.sin(grade)

    # 총 요구력 (3.34)
    F_trac = m_v * a + F_rl + F_grade

    # 요구 동력 (3.35)
    eta_dt = VEHICLE_PARAMS["eta_dt"]
    if F_trac >= 0:
        P_wheel = F_trac * v_ms / eta_dt
    else:
        # 제동 (회생 가능)
        P_wheel = F_trac * v_ms * eta_dt

    return P_wheel   # [W]


def v_to_omega_motor(v_ms: float) -> float:
    """차량 속도 → 모터 각속도 [rad/s]"""
    r_t = VEHICLE_PARAMS["r_t"]
    omega_wheel = v_ms / r_t
    omega_motor = omega_wheel * GEAR_RATIO
    return omega_motor
