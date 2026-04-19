"""
H2 고압 탱크 모델 — Abel-Noble 실기체 + Sery & Leduc 열역학
수식 (3.1), (3.3), (3.4)
"""

import numpy as np

try:
    import CoolProp.CoolProp as CP
    HAS_COOLPROP = True
except ImportError:
    HAS_COOLPROP = False

# Abel-Noble 상수
_B_AN = 2.661e-5       # m³/mol
_R_H2 = 4124.18        # J/(kg·K)  specific gas constant for H2
_M_H2 = 2.016e-3       # kg/mol


class H2TankModel:
    """
    입력:  ṁ_H₂_out [kg/s], T_amb [K], dt [s]
    출력:  P_tank [Pa], P_tank_bar [bar], m_H₂_rem [kg],
           T_tank [K], shutdown_flag [bool]
    """

    def __init__(self,
                 V_tank: float = 0.1566,
                 m_H2_0: float = 6.33,
                 GS_tank: float = 70.0,
                 P_min_bar: float = 10.0,
                 T_init: float = 293.15):
        self.V_tank = V_tank          # m³ (3 × 52.2 L)
        self.m_H2_0 = m_H2_0          # kg
        self.GS_tank = GS_tank         # W/K
        self.P_min_bar = P_min_bar     # bar

        # State
        self.m_H2 = m_H2_0
        self.T_tank = T_init           # K
        self.shutdown = False

    def reset(self):
        self.m_H2 = self.m_H2_0
        self.T_tank = 293.15
        self.shutdown = False

    # ------------------------------------------------------------------
    def _get_cp_cv(self, T: float, rho: float):
        """CoolProp 또는 이상기체 근사로 c_p, c_v 계산"""
        if HAS_COOLPROP:
            try:
                cp = CP.PropsSI('Cpmass', 'T', T, 'Dmass', rho, 'Hydrogen')
                cv = CP.PropsSI('Cvmass', 'T', T, 'Dmass', rho, 'Hydrogen')
                return cp, cv
            except Exception:
                pass
        # fallback: ideal-gas
        cv = 10160.0   # J/(kg·K) approx for H2
        cp = 14300.0
        return cp, cv

    def _abel_noble_pressure(self, rho: float, T: float) -> float:
        """수식 (3.1): P = ρ·R·T / (1 − b·ρ/M)"""
        rho_mol = rho / _M_H2          # mol/m³
        denom = 1.0 - _B_AN * rho_mol
        if denom <= 0.01:
            denom = 0.01
        P = rho * _R_H2 * T / denom
        return P

    def step(self, m_dot_out: float, T_amb: float, dt: float) -> dict:
        if self.shutdown:
            return self._make_output()

        # 질량 감소 (3.4)
        dm = m_dot_out * dt
        self.m_H2 = max(self.m_H2 - dm, 0.0)

        if self.m_H2 <= 0.01:
            self.shutdown = True
            return self._make_output()

        rho = self.m_H2 / self.V_tank

        cp, cv = self._get_cp_cv(self.T_tank, rho)

        # 열역학 (3.3): m·cv·dT/dt = ṁ·(cp-cv)·T + GS·(Tamb - T)
        if self.m_H2 > 0.01:
            dTdt = (m_dot_out * (cp - cv) * self.T_tank
                    + self.GS_tank * (T_amb - self.T_tank)) / (self.m_H2 * cv)
            self.T_tank += dTdt * dt
            self.T_tank = np.clip(self.T_tank, 200.0, 450.0)

        P_tank = self._abel_noble_pressure(rho, self.T_tank)
        P_tank_bar = P_tank / 1e5

        if P_tank_bar < self.P_min_bar:
            self.shutdown = True

        return self._make_output()

    def _make_output(self) -> dict:
        rho = self.m_H2 / self.V_tank if self.m_H2 > 0 else 0.0
        P = self._abel_noble_pressure(rho, self.T_tank) if rho > 0 else 0.0
        return {
            "P_tank_Pa": P,
            "P_tank_bar": P / 1e5 if P > 0 else 0.0,
            "m_H2_rem": self.m_H2,
            "T_tank_K": self.T_tank,
            "shutdown_flag": self.shutdown,
        }
