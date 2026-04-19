"""
PMSM 모터 모델 — d-q 방정식 (간이)
수식 (3.29)~(3.32)
"""

import numpy as np


class PMSMModel:
    """120 kW PMSM (넥쏘)"""

    def __init__(self):
        self.P_rated = 120.0     # kW
        self.T_max = 395.0       # N·m
        self.rpm_max = 16000.0
        self.p = 4               # 극쌍수
        self.R_s = 0.012         # Ω
        self.L_d = 0.18e-3       # H
        self.L_q = 0.34e-3       # H
        self.psi_pm = 0.082      # Wb
        self.J = 0.05            # kg·m²
        self.B = 0.01            # N·m·s/rad

    def compute_torque_from_power(self, P_mech_kw: float, omega_m: float) -> dict:
        """
        역방향(backward) 계산: 요구 기계 출력 → 토크
        omega_m : 모터 각속도 [rad/s]
        """
        if omega_m < 0.1:
            T_em = 0.0
            P_elec_kw = 0.0
            eta_motor = 0.0
        else:
            T_em = (P_mech_kw * 1e3) / omega_m
            T_em = np.clip(T_em, -self.T_max, self.T_max)
            # 전기 손실 근사
            I_q = T_em / (1.5 * self.p * self.psi_pm) if self.psi_pm > 0 else 0.0
            P_loss = 3.0 / 2.0 * self.R_s * I_q ** 2
            P_elec_kw = P_mech_kw + P_loss / 1e3
            eta_motor = P_mech_kw / P_elec_kw if P_elec_kw > 0.01 else 0.0
            eta_motor = np.clip(eta_motor, 0.0, 1.0)

        return {
            "T_em": T_em,
            "P_elec_kw": P_elec_kw,
            "eta_motor": eta_motor,
        }
