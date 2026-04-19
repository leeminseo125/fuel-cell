"""
PEMFC 스택 전기화학 모델 — Sery & Leduc 효율 맵 + 열화 모델
수식 (3.5)~(3.14)
"""

import numpy as np

# ---------- 2D 룩업: 시스템 효율 (Sery & Leduc 그림 8) ----------
ETA_FCS_TABLE_KW = [0, 5, 8, 10, 20, 30, 40, 50, 60, 70, 82]
ETA_FCS_TABLE_VAL = [0.00, 0.64, 0.668, 0.668, 0.630, 0.610,
                     0.590, 0.560, 0.540, 0.515, 0.480]

# ---------- 상수 ----------
_F = 96485.33        # C/mol  Faraday
_R = 8.314           # J/(mol·K)
_M_H2 = 2.016e-3    # kg/mol
_N_CELL = 440
_A_CELL = 312e-4     # m² (312 cm²)
_LHV_H2 = 120.0e6   # J/kg  수소 저위발열량


def eta_fcs(P_kwe: float) -> float:
    return float(np.interp(P_kwe, ETA_FCS_TABLE_KW, ETA_FCS_TABLE_VAL))


class PEMFCModel:
    """
    입력:  P_fc_demand [kWe], dt [s]
    출력:  P_stack [kWe], m_dot_H2 [kg/s], eta, T_st [°C], SOH_fc 등
    """

    def __init__(self):
        # 전기화학
        self.N_cell = _N_CELL
        self.A_cell = _A_CELL
        self.alpha = 0.5
        self.I_0 = 0.001 * _A_CELL * 1e4   # A  (0.001 A/cm² × 312 cm²)
        self.R_mem_base = 0.05              # Ω (base)
        self.R_contact = 0.01               # Ω
        self.I_lim_base = 2.0 * _A_CELL * 1e4  # A

        # 막 수분화
        self.lambda_mem = 10.0   # [6, 14]

        # 열 모델
        self.T_st = 338.15       # K (65 °C)
        self.C_th_st = 20000.0   # J/K
        self.Q_cool_coeff = 500.0  # W/K

        # SOH
        self.SOH_fc = 1.0
        self.sigma1 = 1.8e-16
        self.sigma2 = 8.4e-9
        self.sigma3 = 1.1e-7
        self.sigma4 = 2.0e-8
        self.overheat_sigma4 = False
        self.N_ss = 0             # start-stop count
        self._prev_on = False

        # 과도 응답 (760 ms, 0→90 kWe)
        self.P_fc_actual = 0.0
        self.tau_fc = 0.76        # s

        # 이전 출력
        self.P_fc_prev = 0.0

    def reset(self):
        self.lambda_mem = 10.0
        self.T_st = 338.15
        self.SOH_fc = 1.0
        self.P_fc_actual = 0.0
        self.P_fc_prev = 0.0
        self.N_ss = 0
        self._prev_on = False
        self.overheat_sigma4 = False

    @property
    def T_st_C(self) -> float:
        return self.T_st - 273.15

    def step(self, P_fc_demand_kwe: float, dt: float, T_amb: float = 296.15) -> dict:
        # 과도 응답 1차 필터
        alpha_f = 1.0 - np.exp(-dt / self.tau_fc)
        self.P_fc_actual += alpha_f * (P_fc_demand_kwe - self.P_fc_actual)
        P_fc = np.clip(self.P_fc_actual, 0.0, 82.0)

        # 시스템 효율
        eta = eta_fcs(P_fc)

        # 수소 소비 (3.10)
        # Sery & Leduc: 공전 12.4 g/h = 3.44e-6 kg/s
        if P_fc > 0.1 and eta > 0.01:
            P_fc_w = P_fc * 1e3
            m_dot_H2 = P_fc_w / (eta * _LHV_H2)
        elif P_fc > 0.0:
            # 공전 수소 소비
            m_dot_H2 = 12.4e-3 / 3600.0  # 12.4 g/h → kg/s
        else:
            m_dot_H2 = 0.0

        # 스택 전류 (근사)
        V_stack_approx = 0.65 * self.N_cell if P_fc > 0.1 else 1.0
        I_st = (P_fc * 1e3) / V_stack_approx if P_fc > 0.1 else 0.0

        # 막 수분화 업데이트 (3.11) — 간이
        prod_rate = I_st * self.N_cell / (2 * _F) if I_st > 0 else 0.0
        evap_rate = 0.001 * (self.T_st - 333.15) if self.T_st > 333.15 else 0.0
        dlambda = (prod_rate * 0.01 - evap_rate * 0.005) * dt
        self.lambda_mem = np.clip(self.lambda_mem + dlambda, 4.0, 16.0)

        # 막 상태 효과
        R_mem = self.R_mem_base
        I_lim = self.I_lim_base
        if self.lambda_mem < 6.0:
            R_mem *= 3.0
        if self.lambda_mem > 14.0:
            I_lim *= 0.3

        # 열 모델 (3.12~3.13)
        if P_fc > 0.1:
            V_cell_approx = 0.65
            Q_gen = self.N_cell * I_st * (1.482 - V_cell_approx)
        else:
            Q_gen = 0.0
        Q_cool = self.Q_cool_coeff * (self.T_st - T_amb)
        Q_loss = 50.0  # W 방사 손실
        dT = (Q_gen - Q_cool - Q_loss) / self.C_th_st * dt
        self.T_st += dT
        self.T_st = np.clip(self.T_st, T_amb, 373.15)

        # SOH 열화 (3.14)
        dP_dt = (P_fc - self.P_fc_prev) / dt if dt > 0 else 0.0
        is_idle = P_fc < 1.0

        # start-stop count
        is_on = P_fc > 1.0
        if is_on and not self._prev_on:
            self.N_ss += 1
        self._prev_on = is_on

        self.overheat_sigma4 = self.T_st_C > 85.0
        s4_mult = 5.0 if self.overheat_sigma4 else 1.0

        dSOH = -(self.sigma1 * dP_dt ** 2
                 + self.sigma2 * float(is_idle)
                 + self.sigma4 * s4_mult) * dt
        self.SOH_fc = max(self.SOH_fc + dSOH, 0.0)

        self.P_fc_prev = P_fc

        return {
            "P_fc_kwe": P_fc,
            "P_stack_kwe": P_fc / 0.97 if P_fc > 0.1 else 0.0,  # before DC/DC
            "m_dot_H2": m_dot_H2,           # kg/s
            "eta_fcs": eta,
            "I_st": I_st,
            "T_st_K": self.T_st,
            "T_st_C": self.T_st_C,
            "SOH_fc": self.SOH_fc,
            "lambda_mem": self.lambda_mem,
            "dP_dt": dP_dt,
        }
