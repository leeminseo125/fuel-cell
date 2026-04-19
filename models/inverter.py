"""
3상 IGBT 인버터 모델
수식 (3.28)
"""

import numpy as np


class InverterModel:
    """DC → 3상 AC 변환 (구동) / AC → DC 정류 (회생)"""

    def __init__(self):
        self.f_sw = 10_000      # Hz  switching frequency
        self.E_sw = 5e-4        # J   switching energy per pulse
        self.R_DS_on = 0.01     # Ω   on-state resistance
        self.eta_inv_drive = 0.97
        self.eta_inv_regen = 0.95

    def compute(self, P_dc_kw: float, is_regen: bool = False) -> dict:
        """
        P_dc_kw: DC 버스에서 인버터로 들어가는 전력 [kW]
        """
        if is_regen:
            eta = self.eta_inv_regen
        else:
            eta = self.eta_inv_drive

        P_ac_kw = P_dc_kw * eta
        P_loss_kw = P_dc_kw * (1.0 - eta)

        return {
            "P_ac_kw": P_ac_kw,
            "P_loss_inv_kw": P_loss_kw,
            "eta_inv": eta,
        }
