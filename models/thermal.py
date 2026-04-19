"""
열관리 시스템 (TMS) — 규칙 기반 피드백 제어 (고정)
수식 (3.37)~(3.38)
"""

import numpy as np


class ThermalManagementSystem:
    """
    FC 스택 + 배터리 독립 냉각 루프
    RL 에이전트 제어 대상 아님
    """

    def __init__(self):
        # 스택 냉각
        self.m_dot_cool = 0.5      # kg/s (냉각수 유량)
        self.cp_cool = 3800.0      # J/(kg·K)
        self.UA_rad = 200.0        # W/K (라디에이터)

        # 목표 온도
        self.T_st_target = 343.15  # K (70°C)
        self.T_bat_target = 303.15 # K (30°C)

    def compute_stack_cooling(self, T_st: float, T_amb: float) -> float:
        """스택 냉각 방열 [W]"""
        # 간이 제어: 목표 온도 이상이면 냉각 활성
        if T_st > self.T_st_target:
            Q_cool = self.m_dot_cool * self.cp_cool * (T_st - self.T_st_target)
            Q_rad = self.UA_rad * (T_st - T_amb)
            return Q_cool + Q_rad
        return 50.0  # 최소 방열

    def compute_bat_cooling(self, T_bat: float, T_amb: float) -> float:
        """배터리 냉각 [W]"""
        if T_bat > self.T_bat_target:
            return 100.0 * (T_bat - self.T_bat_target)
        return 0.0

    def get_penalties(self, T_st_C: float, T_bat_C: float) -> dict:
        """열 위반 패널티"""
        penalties = {
            "overheat_fc": T_st_C > 85.0,
            "underheat_fc": T_st_C < 40.0,
            "overheat_bat": T_bat_C > 45.0,
        }
        return penalties
