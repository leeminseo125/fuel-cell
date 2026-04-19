"""
슈퍼커패시터 모델
수식 (3.26), (3.27)
C_sc=22F, V_max=370V, V_min=185V, R_ESR=0.3mΩ
"""

import numpy as np


class SuperCapModel:

    def __init__(self):
        self.C_sc = 22.0       # F
        self.V_max = 370.0     # V
        self.V_min = 185.0     # V
        self.R_ESR = 0.3e-3    # Ω

        # State
        self.Q_sc = self.C_sc * 0.5 * (self.V_max + self.V_min)  # initial charge
        self.V_sc = self.Q_sc / self.C_sc

    def reset(self):
        self.Q_sc = self.C_sc * 0.5 * (self.V_max + self.V_min)
        self.V_sc = self.Q_sc / self.C_sc

    @property
    def SOC_sc(self) -> float:
        """수식 (3.27)"""
        v2 = self.V_sc ** 2
        return np.clip((v2 - self.V_min ** 2) / (self.V_max ** 2 - self.V_min ** 2),
                        0.0, 1.0)

    def step(self, P_sc_kw: float, dt: float) -> dict:
        """P_sc_kw: 양수=방전, 음수=충전"""
        if abs(P_sc_kw) < 1e-6:
            I_sc = 0.0
        else:
            V_eff = max(self.V_sc, self.V_min * 0.5)
            I_sc = (P_sc_kw * 1e3) / V_eff

        # 전하 업데이트
        dQ = -I_sc * dt
        self.Q_sc += dQ
        self.Q_sc = np.clip(self.Q_sc,
                            self.C_sc * self.V_min,
                            self.C_sc * self.V_max)

        # 전압 (3.26)
        self.V_sc = self.Q_sc / self.C_sc - I_sc * self.R_ESR
        self.V_sc = np.clip(self.V_sc, self.V_min * 0.9, self.V_max * 1.05)

        return {
            "V_sc": self.V_sc,
            "I_sc": I_sc,
            "SOC_sc": self.SOC_sc,
            "P_sc_actual_kw": P_sc_kw,
        }
