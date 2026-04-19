"""
BoP(Balance of Plant) 모델 — Sery & Leduc 그림 9 룩업 테이블
압축기 + 냉각팬 + 냉각수펌프 + 12V LDC
"""

import numpy as np


# Sery & Leduc 그림 9 정상 상태 실험 측정값
_P_COMP_KW = [0, 10, 20, 30, 40, 50, 60, 70, 82]
_P_COMP_V  = [0, 0.5, 1.5, 2.5, 3.5, 4.5, 6.0, 7.5, 9.1]

_P_FANS_KW = [0, 40, 50, 60, 70, 82]
_P_FANS_V  = [0.2, 0.2, 0.2, 0.5, 1.2, 1.8]

_P_PUMP_KW = [0, 50, 60, 70, 82]
_P_PUMP_V  = [0.05, 0.15, 0.3, 0.5, 0.6]

P_12V_KWE = 0.273   # 고정 (Sery & Leduc 그림 14 평균값)


class BoPModel:

    def compute(self, P_fcs_kwe: float) -> dict:
        P_comp = float(np.interp(P_fcs_kwe, _P_COMP_KW, _P_COMP_V))
        P_fans = float(np.interp(P_fcs_kwe, _P_FANS_KW, _P_FANS_V))
        P_pump = float(np.interp(P_fcs_kwe, _P_PUMP_KW, _P_PUMP_V))
        P_12V = P_12V_KWE
        P_BoP = P_comp + P_fans + P_pump + P_12V
        return {
            "P_comp": P_comp,
            "P_fans": P_fans,
            "P_pump": P_pump,
            "P_12V": P_12V,
            "P_BoP": P_BoP,
            "P_fcs_net": P_fcs_kwe - P_BoP,
        }
