"""
리튬이온 배터리 모델 — 2RC 등가 회로 + SOH 열화
수식 (3.19)~(3.24)
"""

import numpy as np


def _voc_lookup(soc: float) -> float:
    """SOC → 개방 전압 (근사 다항식)"""
    soc = np.clip(soc, 0.0, 1.0)
    # 240V 공칭, SOC 기반 근사
    return 220.0 + 40.0 * soc - 10.0 * (soc - 0.5) ** 2


class BatteryModel:
    """
    입력:  P_bat [kW] (양수=방전, 음수=충전), dt [s]
    출력:  SOC, SOH, V_bat, I_bat, T_bat
    """

    def __init__(self):
        # 사양 — Sery & Leduc Figure 1
        self.Q_nom_Ah = 6.5         # Ah
        self.V_nom = 240.0          # V
        self.E_nom_kWh = 1.56       # kWh
        self.P_dis_max = 39.0       # kWe
        self.P_chg_max = 35.0       # kWe

        # 2RC 파라미터
        self.R_0 = 0.10             # Ω
        self.R_1 = 0.05             # Ω
        self.C_1 = 1000.0           # F
        self.R_2 = 0.03             # Ω
        self.C_2 = 5000.0           # F

        # SOH 열화 — 고출력 Li-ion (수소차 보조 배터리 특성 반영)
        self.k_cyc = 2.0e-8          # A⁻¹ (고출력 배터리 보정)
        self.k_cal = 1.0e-10         # baseline calendar
        self.E_a = 20000.0           # J/mol
        self.gamma_DOD = 2.0
        self.gamma_C = 0.5           # 고출력 배터리는 고C율 내성

        # State
        self.SOC = 0.60
        self.SOH = 1.0
        self.V_R1 = 0.0
        self.V_R2 = 0.0
        self.T_bat = 298.15         # K (25°C)
        self.I_bat = 0.0

    def reset(self, SOC_init: float = 0.60):
        self.SOC = SOC_init
        self.SOH = 1.0
        self.V_R1 = 0.0
        self.V_R2 = 0.0
        self.T_bat = 298.15
        self.I_bat = 0.0

    @property
    def T_bat_C(self) -> float:
        return self.T_bat - 273.15

    def step(self, P_bat_kw: float, dt: float, T_amb: float = 296.15) -> dict:
        # 전력 제한
        if P_bat_kw > 0:
            P_bat_kw = min(P_bat_kw, self.P_dis_max)
        else:
            P_bat_kw = max(P_bat_kw, -self.P_chg_max)

        # SOC 경계 보호
        if self.SOC <= 0.05 and P_bat_kw > 0:
            P_bat_kw = 0.0
        if self.SOC >= 0.99 and P_bat_kw < 0:
            P_bat_kw = 0.0

        V_oc = _voc_lookup(self.SOC)

        # 전류 계산: P = V * I ≈ V_oc * I (1차 근사)
        if abs(P_bat_kw) < 0.001:
            I_bat = 0.0
        else:
            I_bat = (P_bat_kw * 1e3) / V_oc  # A (양수=방전)

        self.I_bat = I_bat

        # 2RC 동역학 (3.20~3.21)
        dVR1 = (-self.V_R1 / (self.R_1 * self.C_1) + I_bat / self.C_1) * dt
        dVR2 = (-self.V_R2 / (self.R_2 * self.C_2) + I_bat / self.C_2) * dt
        self.V_R1 += dVR1
        self.V_R2 += dVR2

        # 단자 전압 (3.19)
        V_bat = V_oc - I_bat * self.R_0 - self.V_R1 - self.V_R2

        # SOC 업데이트 (3.22)
        Q_bat_As = self.Q_nom_Ah * 3600.0
        eta_coulomb = 0.995 if I_bat > 0 else 1.0
        dSOC = -eta_coulomb * I_bat * dt / Q_bat_As
        self.SOC = np.clip(self.SOC + dSOC, 0.0, 1.0)

        # 열 모델 (단순)
        Q_gen = I_bat ** 2 * self.R_0
        Q_cool_bat = 5.0 * (self.T_bat - T_amb)
        dT = (Q_gen - Q_cool_bat) / 2000.0 * dt
        self.T_bat += dT
        self.T_bat = np.clip(self.T_bat, T_amb - 10, 333.15)

        # SOH 열화 (3.23~3.24)
        C_rate = abs(I_bat) / self.Q_nom_Ah if self.Q_nom_Ah > 0 else 0.0
        C_rate_eff = min(C_rate, 5.0)  # 고출력 배터리 — 정격 C-rate 내 스트레스 캡
        DOD = 1.0 - self.SOC
        f_stress = ((1 + self.gamma_DOD * (DOD - 0.30) ** 2)
                    * (1 + self.gamma_C * max(0, C_rate_eff - 1.0) ** 2))
        # 온도 효과
        k_cal_T = self.k_cal * np.exp(-self.E_a / (8.314 * self.T_bat))
        # 사이클 + 캘린더
        k_cyc_eff = self.k_cyc
        if self.T_bat_C > 45.0:
            k_cyc_eff *= 3.0

        dSOH = -(k_cyc_eff * abs(I_bat) * f_stress / self.Q_nom_Ah
                 + k_cal_T) * dt
        self.SOH = max(self.SOH + dSOH, 0.0)

        return {
            "SOC": self.SOC,
            "SOH_bat": self.SOH,
            "V_bat": V_bat,
            "I_bat": I_bat,
            "T_bat_K": self.T_bat,
            "T_bat_C": self.T_bat_C,
            "P_bat_actual_kw": P_bat_kw,
        }
