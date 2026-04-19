"""
회생 제동 에너지 회수 모델
수식 (3.36): P_regen = min(f·|F|·v·η_gen·η_inv, 35 kWe)
"""

import numpy as np

P_REGEN_MAX_KWE = 35.0   # Sery & Leduc BAB130


def compute_regen(F_brake: float,
                  v_ms: float,
                  f_regen: float = 0.70,
                  eta_gen: float = 0.97,
                  eta_inv: float = 0.95) -> float:
    """
    F_brake : 제동력 [N] (음수)
    v_ms    : 차량 속도 [m/s]
    반환    : P_regen [W] (양수 = 배터리/SC에 충전 가능한 전력)
    """
    if F_brake >= 0 or v_ms < 0.5:
        return 0.0

    P_raw = f_regen * abs(F_brake) * v_ms * eta_gen * eta_inv
    P_regen = min(P_raw, P_REGEN_MAX_KWE * 1e3)
    return P_regen
