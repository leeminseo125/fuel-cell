"""
DC 버스 에너지 균형 모델
수식 (3.18): P_fcs_net + P_bat + P_sc = P_inv + P_aux
"""

import numpy as np


class DCBusModel:
    """DC 버스 에너지 균형 + DC/DC 컨버터 효율"""

    # DC/DC 컨버터 효율
    ETA_FC_DCDC = 0.97      # FC 단방향 BHDC
    ETA_BAT_DIS = 0.96      # 배터리 방전
    ETA_BAT_CHG = 0.97      # 배터리 충전
    ETA_SC = 0.98            # SC 양방향

    P_AUX_KW = 1.5          # 공조·조명 고정 [kWe]

    V_BUS_NOM = 350.0       # V (공칭)

    def compute(self,
                P_fcs_net_kw: float,
                P_req_low_kw: float,
                P_high_kw: float,
                SOC_bat: float,
                SOC_sc: float) -> dict:
        """
        P_fcs_net_kw : FC 순출력 (BoP 차감 후) [kWe]
        P_req_low_kw : DWT 저주파 요구 [kWe]
        P_high_kw    : DWT 고주파 요구 [kWe]
        """
        # SC가 고주파 담당
        P_sc = P_high_kw / self.ETA_SC if P_high_kw >= 0 else P_high_kw * self.ETA_SC

        # FC 순출력 (DC/DC 적용)
        P_fc_bus = P_fcs_net_kw * self.ETA_FC_DCDC

        # 배터리 = 저주파 요구 + 보조 부하 - FC 공급
        P_bat_need = P_req_low_kw + self.P_AUX_KW - P_fc_bus

        # 배터리 DC/DC 효율 적용
        if P_bat_need > 0:
            # 방전
            P_bat = P_bat_need / self.ETA_BAT_DIS
        else:
            # 충전 (음수)
            P_bat = P_bat_need * self.ETA_BAT_CHG

        return {
            "P_fc_bus_kw": P_fc_bus,
            "P_bat_kw": P_bat,           # 양수=방전, 음수=충전
            "P_sc_kw": P_sc,
            "P_aux_kw": self.P_AUX_KW,
            "V_bus": self.V_BUS_NOM,
        }
