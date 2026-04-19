"""
FCHEVEnv — Gymnasium 환경 (고정밀 서브스테핑)
내부 시뮬레이션: dt_sim (기본 0.01s) — 수치 안정성
외부 제어/로그: dt_ctrl (고정 1.0s) — 1초 간격 저장

에너지 흐름:
  H2 Tank(700bar) -> Regulator -> PEMFC Stack -> BoP -> DC Bus
  -> Battery/SC -> Inverter -> PMSM -> Gearbox -> Wheel
"""

import numpy as np
import gymnasium as gym
from gymnasium import spaces

from models.h2_tank import H2TankModel
from models.pemfc_model import PEMFCModel, eta_fcs
from models.bop_model import BoPModel
from models.dc_bus import DCBusModel
from models.battery import BatteryModel
from models.sc_model import SuperCapModel
from models.inverter import InverterModel
from models.motor import PMSMModel
from models.drivetrain import compute_P_req, v_to_omega_motor, VEHICLE_PARAMS
from models.regen_brake import compute_regen
from models.thermal import ThermalManagementSystem
from control.dwt import DWTDecomposer

# LHV of hydrogen [J/kg]
_LHV_H2 = 120.0e6
# 공전 수소 소비 12.4 g/h -> kg/s
_IDLE_H2_KGS = 12.4e-3 / 3600.0
# DC/DC 효율
_ETA_FC_DCDC = 0.97
# 보조 부하 [kW]
_P_AUX_KW = 1.5


def _compute_reward(m_dot_H2, P_fc_kwe, eta_fcs_val, P_eComp_kwe,
                    delta_SOH_fc, delta_SOH_bat, SOC_bat,
                    dPfc_dt, T_st_C, dt_ctrl, m_H2_0=6.33):
    """보상 함수 (6.6~6.12) — dt_ctrl(1초) 단위"""
    R_H2 = -100.0 * m_dot_H2 * dt_ctrl / m_H2_0
    R_fc = 8.0 * max(0.0, eta_fcs_val - 0.60)
    R_BoP = -3.0 * max(0.0, P_eComp_kwe - 0.5)
    R_deg = -350.0 * abs(delta_SOH_fc) - 200.0 * abs(delta_SOH_bat)
    R_SOC = -50.0 * (SOC_bat - 0.60) ** 2
    R_con = (-1000.0 * max(0.0, 0.30 - SOC_bat) ** 2
             - 1000.0 * max(0.0, SOC_bat - 0.90) ** 2
             - 500.0 * max(0.0, abs(dPfc_dt) - 4.0) ** 2
             - 200.0 * max(0.0, T_st_C - 85.0) ** 2)
    return R_H2 + R_fc + R_BoP + R_deg + R_SOC + R_con


class FCHEVEnv(gym.Env):
    """
    FCHEV 고정밀 시뮬레이터 Gymnasium 환경

    핵심 설계:
      - dt_sim : 내부 물리 스텝 (기본 0.01s) — ODE 수치 안정성
      - dt_ctrl: 에이전트 제어 주기 (고정 1.0s) — 로그 저장/행동 결정
      - 매 step() 호출 시 dt_ctrl/dt_sim (=100)회 서브스테핑 수행
      - cycle_v/cycle_a는 dt_sim 해상도로 사전 생성됨
    """

    metadata = {"render_modes": []}

    def __init__(self, cycle_v: np.ndarray, cycle_a: np.ndarray,
                 duration_s: int = 18000,
                 dt_sim: float = 0.01,
                 dt_ctrl: float = 1.0,
                 T_amb_C: float = 23.0,
                 SOC_init: float = 0.60,
                 scenario_id: str = "A"):
        super().__init__()

        self.cycle_v = cycle_v
        self.cycle_a = cycle_a
        self.duration_s = duration_s
        self.dt_sim = dt_sim
        self.dt_ctrl = dt_ctrl
        self.substeps = max(1, int(round(dt_ctrl / dt_sim)))
        self.T_amb = T_amb_C + 273.15
        self.T_amb_C = T_amb_C
        self.SOC_init = SOC_init
        self.scenario_id = scenario_id
        self.total_ctrl_steps = int(duration_s / dt_ctrl)
        self.cycle_len = len(cycle_v)

        self.action_space = spaces.Box(low=-1.0, high=1.0,
                                       shape=(1,), dtype=np.float32)
        self.observation_space = spaces.Box(low=-2.0, high=2.0,
                                            shape=(12,), dtype=np.float32)

        # 모델 인스턴스
        self.tank = H2TankModel()
        self.pemfc = PEMFCModel()
        self.bop = BoPModel()
        self.battery = BatteryModel()
        self.sc = SuperCapModel()
        self.tms = ThermalManagementSystem()
        self.dwt = DWTDecomposer()

        self.ctrl_step_idx = 0
        self.sim_step_idx = 0
        self.distance_m = 0.0
        self.total_H2_kg = 0.0
        self.terminated = False
        self.truncated = False
        self.log = []

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.tank.reset()
        self.pemfc.reset()
        self.battery.reset(SOC_init=self.SOC_init)
        self.sc.reset()
        self.dwt.reset()
        self.ctrl_step_idx = 0
        self.sim_step_idx = 0
        self.distance_m = 0.0
        self.total_H2_kg = 0.0
        self.terminated = False
        self.truncated = False
        self.log = []
        return self._get_obs(), {}

    def _get_cycle_values(self, sim_idx: int):
        idx = sim_idx % self.cycle_len
        return float(self.cycle_v[idx]), float(self.cycle_a[idx])

    def _get_obs(self) -> np.ndarray:
        v, a = self._get_cycle_values(self.sim_step_idx)
        P_req = compute_P_req(v, a)
        P_low, _ = self.dwt.decompose(P_req)

        obs = np.array([
            np.clip(P_low / 1e3 / 82.0, -1.0, 1.5),
            self.battery.SOC,
            self.pemfc.SOH_fc,
            self.battery.SOH,
            self.tank.m_H2 / self.tank.m_H2_0,
            (self.pemfc.T_st_C - 65.0) / 25.0,
            (self.battery.T_bat_C - 30.0) / 20.0,
            v / 36.0,
            np.clip(a / 4.0, -1.0, 1.0),
            (self.pemfc.lambda_mem - 10.0) / 6.0,
            (eta_fcs(self.pemfc.P_fc_actual) - 0.56) / 0.11,
            self.ctrl_step_idx / max(self.total_ctrl_steps, 1),
        ], dtype=np.float32)
        return obs

    def step(self, action):
        """
        에이전트: 1초(dt_ctrl)마다 action 선택
        내부: dt_sim(0.01s) 간격 서브스텝 100회 수행
        """
        if self.terminated or self.truncated:
            return self._get_obs(), 0.0, True, False, {}

        # 행동 파싱
        a_norm = float(action[0]) if hasattr(action, '__len__') else float(action)
        a_norm = np.clip(a_norm, -1.0, 1.0)
        P_fc_target = 5.0 + 77.0 * (a_norm + 1.0) / 2.0

        # 서브스테핑 누적
        acc_H2_kg = 0.0
        acc_distance_m = 0.0
        acc_m_dot_H2 = 0.0
        acc_P_fc = 0.0
        acc_eta = 0.0
        acc_P_bat = 0.0
        acc_P_sc = 0.0
        acc_P_comp = 0.0
        acc_P_fans = 0.0
        acc_P_pump = 0.0
        acc_P_regen = 0.0
        acc_P_req = 0.0
        acc_P_low = 0.0
        acc_P_high = 0.0
        acc_v = 0.0
        acc_a = 0.0

        SOH_fc_before = self.pemfc.SOH_fc
        SOH_bat_before = self.battery.SOH

        sub_terminated = False
        n_actual = 0
        last_fc_res = None
        last_bop_res = None
        last_tank_res = None

        for sub_i in range(self.substeps):
            if sub_terminated:
                break
            n_actual += 1

            # 1. 주행 데이터
            v, a = self._get_cycle_values(self.sim_step_idx)
            acc_distance_m += v * self.dt_sim
            acc_v += v
            acc_a += a

            # 2. 요구 동력
            P_req_w = compute_P_req(v, a)
            P_low_w, P_high_w = self.dwt.decompose(P_req_w)
            P_low_kw = P_low_w / 1e3
            P_high_kw = P_high_w / 1e3
            acc_P_req += P_req_w / 1e3
            acc_P_low += P_low_kw
            acc_P_high += P_high_kw

            # 3. P_fc_star — 경사 제한 (서브스텝 단위 정밀 적용)
            max_ramp = 4.0 * self.dt_sim
            P_fc_star = np.clip(P_fc_target,
                                self.pemfc.P_fc_prev - max_ramp,
                                self.pemfc.P_fc_prev + max_ramp)
            P_fc_star = np.clip(P_fc_star, 0.0, 82.0)

            # 4. BoP
            last_bop_res = self.bop.compute(P_fc_star)
            acc_P_comp += last_bop_res["P_comp"]
            acc_P_fans += last_bop_res["P_fans"]
            acc_P_pump += last_bop_res["P_pump"]

            # 5. 수소 소비
            eta = eta_fcs(P_fc_star)
            if P_fc_star > 0.5 and eta > 0.01:
                m_dot_H2 = (P_fc_star * 1e3) / (eta * _LHV_H2)
            elif v < 0.5 and P_fc_star <= 0.5:
                m_dot_H2 = _IDLE_H2_KGS
            else:
                m_dot_H2 = 0.0

            # 6. FC 모델 (열, SOH, 과도응답)
            last_fc_res = self.pemfc.step(P_fc_star, self.dt_sim, self.T_amb)
            last_fc_res["m_dot_H2"] = m_dot_H2
            acc_P_fc += last_fc_res["P_fc_kwe"]
            acc_eta += eta
            acc_m_dot_H2 += m_dot_H2

            # 7. 수소 탱크
            last_tank_res = self.tank.step(m_dot_H2, self.T_amb, self.dt_sim)
            h2_consumed = m_dot_H2 * self.dt_sim
            acc_H2_kg += h2_consumed
            self.total_H2_kg += h2_consumed

            # 8. 회생 제동
            P_regen_w = 0.0
            if P_req_w < 0:
                m_v = VEHICLE_PARAMS["m_v"]
                F_brake = m_v * a
                P_regen_w = compute_regen(F_brake, v)
            acc_P_regen += P_regen_w / 1e3

            # 9. DC 버스 에너지 균형
            P_bat_kw = P_low_kw - P_fc_star
            if P_regen_w > 0:
                P_bat_kw -= P_regen_w / 1e3
            P_sc_kw = P_high_kw
            acc_P_bat += P_bat_kw
            acc_P_sc += P_sc_kw

            # 10. 배터리 & SC
            self.battery.step(P_bat_kw, self.dt_sim, self.T_amb)
            self.sc.step(P_sc_kw, self.dt_sim)

            # 11. 종료 조건
            self.sim_step_idx += 1
            if last_tank_res["shutdown_flag"]:
                sub_terminated = True
                self.terminated = True

        # 서브스테핑 완료 — 평균 계산
        n_actual = max(n_actual, 1)
        self.distance_m += acc_distance_m

        delta_SOH_fc = self.pemfc.SOH_fc - SOH_fc_before
        delta_SOH_bat = self.battery.SOH - SOH_bat_before

        avg_eta = acc_eta / n_actual
        avg_m_dot_H2 = acc_m_dot_H2 / n_actual
        avg_P_fc = acc_P_fc / n_actual
        avg_P_comp = acc_P_comp / n_actual

        reward = _compute_reward(
            avg_m_dot_H2, avg_P_fc, avg_eta,
            avg_P_comp, delta_SOH_fc, delta_SOH_bat,
            self.battery.SOC,
            last_fc_res["dP_dt"] if last_fc_res else 0.0,
            self.pemfc.T_st_C, self.dt_ctrl
        )

        self.ctrl_step_idx += 1
        if self.ctrl_step_idx >= self.total_ctrl_steps:
            self.truncated = True

        # 로그 (1초 간격)
        t_now = self.ctrl_step_idx * self.dt_ctrl
        tank_out = self.tank._make_output()
        self.log.append({
            "t": t_now,
            "v_ms": acc_v / n_actual,
            "v_kmh": acc_v / n_actual * 3.6,
            "a": acc_a / n_actual,
            "P_req_kW": acc_P_req / n_actual,
            "P_low_kW": acc_P_low / n_actual,
            "P_high_kW": acc_P_high / n_actual,
            "P_fc_kW": avg_P_fc,
            "P_fcs_net_kW": P_fc_star,
            "P_bat_kW": acc_P_bat / n_actual,
            "P_sc_kW": acc_P_sc / n_actual,
            "P_comp_kW": avg_P_comp,
            "P_fans_kW": acc_P_fans / n_actual,
            "P_coolpump_kW": acc_P_pump / n_actual,
            "m_H2_gs": avg_m_dot_H2 * 1e3,
            "M_H2_kg": self.total_H2_kg,
            "SOC_bat": self.battery.SOC,
            "SOH_fc": self.pemfc.SOH_fc,
            "SOH_bat": self.battery.SOH,
            "T_st_C": self.pemfc.T_st_C,
            "T_bat_C": self.battery.T_bat_C,
            "eta_fcs": avg_eta,
            "dPfc_dt": last_fc_res["dP_dt"] if last_fc_res else 0.0,
            "lambda_mem": self.pemfc.lambda_mem,
            "P_tank_bar": tank_out["P_tank_bar"],
            "distance_km": self.distance_m / 1e3,
            "P_regen_kW": acc_P_regen / n_actual,
            "action_norm": a_norm,
            "reward": reward,
        })

        obs = self._get_obs()
        return obs, float(reward), self.terminated, self.truncated, {}

    def get_log_dict(self) -> list:
        return self.log
