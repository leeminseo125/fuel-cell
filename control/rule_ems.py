"""
규칙 기반 EMS — Z0~Z6 구역
FC를 효율 최적 구간으로 운전하면서 배터리 SOC를 유지하는 전략
"""

import numpy as np


class RuleBasedEMS:
    """
    Zone 정의:
      Z0: 정차 (v < 0.5 m/s) → FC idle 또는 off
      Z1: 저부하 (P_low < 10 kW) → FC 최대효율(8~10 kW) 고정, 잉여 → 배터리 충전
      Z2: 중부하 (10 ≤ P_low < 40) → FC 추종
      Z3: 고부하 (40 ≤ P_low < 60) → FC 최대 + 배터리 보조
      Z4: 피크 (P_low ≥ 60) → FC 정격(82) + 배터리 최대 보조
      Z5: 회생 (P_low < 0) → FC off, 배터리/SC 충전
      Z6: SOC 강제 충전 (SOC < 0.30) → FC 출력 증가
      Z7: SOC 과충전 (SOC > 0.90) → FC 출력 감소
    """

    def __init__(self):
        self.P_fc_prev = 0.0
        self.ramp_limit = 4.0     # kW/s

    def reset(self):
        self.P_fc_prev = 0.0

    def compute(self, P_low_kw: float, SOC_bat: float,
                v_ms: float, dt: float) -> dict:
        """
        반환: P_fc_star [kWe] — FC 출력 기준값
        """
        zone = "Z2"

        # FC 필요 출력 ≈ 요구 전력 (로드로드가 보조 부하 포함)
        P_fc_need = P_low_kw

        # 회생
        if P_low_kw < 0:
            P_fc_star = 0.0
            zone = "Z5"
        # 정차
        elif v_ms < 0.5:
            P_fc_star = 2.0   # idle 최소 운전
            zone = "Z0"
        # 저부하 — FC 최대 효율점
        elif P_fc_need < 10.0:
            P_fc_star = 9.0   # η=66.8%
            zone = "Z1"
        # 중부하 — FC가 요구 추종, 배터리가 과도 보조
        elif P_fc_need < 40.0:
            if SOC_bat > 0.55:
                P_fc_star = P_fc_need
            else:
                P_fc_star = P_fc_need + 3.0  # SOC 회복 마진
            zone = "Z2"
        # 고부하
        elif P_fc_need < 60.0:
            P_fc_star = min(P_fc_need, 60.0)
            zone = "Z3"
        # 피크
        else:
            P_fc_star = 82.0
            zone = "Z4"

        # SOC 강제 충전 (Z6)
        if SOC_bat < 0.30:
            P_fc_star = max(P_fc_star, P_fc_need + 15.0)
            zone = "Z6"

        # SOC 과충전 방지 (Z7)
        if SOC_bat > 0.85:
            P_fc_star = min(P_fc_star, max(P_fc_need - 3.0, 0.0))
            zone = "Z7"

        # 범위 제한
        P_fc_star = np.clip(P_fc_star, 0.0, 82.0)

        # 경사 제한
        if dt > 0:
            max_ramp = self.ramp_limit * dt
            P_fc_star = np.clip(P_fc_star,
                                self.P_fc_prev - max_ramp,
                                self.P_fc_prev + max_ramp)

        P_fc_star = np.clip(P_fc_star, 0.0, 82.0)
        self.P_fc_prev = P_fc_star

        return {
            "P_fc_star": float(P_fc_star),
            "zone": zone,
        }
