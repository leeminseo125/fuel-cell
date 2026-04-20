"""
FCHEVEnv-RL — 강화학습 최적화 환경
기본 FCHEVEnv를 상속하여 개선된 보상 함수, 커리큘럼 학습, 전문가 시연 지원 추가

핵심 개선:
  1. H2 소비 페널티 강화 (지수적, 탱크 잔량 반비례)
  2. 즉각적 km/kg 효율 보상 (지배적 항)
  3. 조기 종료 시 대규모 터미널 페널티
  4. 생존 보너스 (에피소드 완주 유도)
  5. 규칙 기반 베이스라인 대비 포텐셜 셰이핑
  6. 커리큘럼 학습 (에피소드 길이 점진적 증가)
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

_LHV_H2 = 120.0e6
_IDLE_H2_KGS = 12.4e-3 / 3600.0
_ETA_FC_DCDC = 0.97
_P_AUX_KW = 1.5


def _compute_reward_v2(m_dot_H2, P_fc_kwe, eta_fcs_val, P_eComp_kwe,
                        delta_SOH_fc, delta_SOH_bat, SOC_bat,
                        dPfc_dt, T_st_C, dt_ctrl, m_H2_0,
                        m_H2_remaining, distance_m_step,
                        t_step, total_steps, terminated_early):
    """
    보상 함수 v2 — km/kg H2 효율 극대화 설계

    기존 대비 변경:
      - R_H2: 탱크 잔량에 반비례하는 지수적 페널티
      - R_eff: 즉각적 km/kg 비율 보상 (지배적 항)
      - R_terminal: 조기 종료 시 대규모 페널티
      - R_alive: 매 스텝 생존 보너스
      - R_fc: 효율 보상 강화 (0.55 이상부터)
    """
    h2_consumed_step = m_dot_H2 * dt_ctrl
    tank_frac = max(m_H2_remaining / m_H2_0, 0.01)

    # 1. H2 소비 페널티 — 탱크 잔량 반비례 (핵심)
    R_H2 = -800.0 * (h2_consumed_step / m_H2_0) / (tank_frac ** 0.5)

    # 2. 즉각적 효율 보상 (km/kg H2) — 지배적 항
    dist_km_step = distance_m_step / 1000.0
    if h2_consumed_step > 1e-9:
        km_per_kg_inst = dist_km_step / h2_consumed_step
        # 규칙 기반 UDDS ~175, HWFET ~131; 정규화 기준 200 km/kg
        R_eff = 3.0 * np.clip(km_per_kg_inst / 200.0, 0.0, 2.0)
    elif dist_km_step > 0.001:
        # 주행 중인데 H2 소비 0 (회생 등)
        R_eff = 2.0
    else:
        R_eff = 0.0

    # 3. FC 효율 보상 (고효율 운전점 유도)
    if P_fc_kwe > 2.0:
        R_fc = 6.0 * max(0.0, eta_fcs_val - 0.55)
    else:
        R_fc = 0.2  # idle은 약간의 보상

    # 4. SOC 관리 (0.55~0.65 목표 범위)
    R_SOC = -40.0 * (SOC_bat - 0.60) ** 2

    # 5. 열화 페널티
    R_deg = -250.0 * abs(delta_SOH_fc) - 150.0 * abs(delta_SOH_bat)

    # 6. BoP 효율
    R_BoP = -2.5 * max(0.0, P_eComp_kwe - 0.5)

    # 7. 제약 조건 위반 페널티
    R_con = 0.0
    if SOC_bat < 0.25:
        R_con -= 800.0 * (0.25 - SOC_bat) ** 2
    if SOC_bat > 0.92:
        R_con -= 800.0 * (SOC_bat - 0.92) ** 2
    if abs(dPfc_dt) > 4.0:
        R_con -= 200.0 * (abs(dPfc_dt) - 4.0) ** 2
    if T_st_C > 85.0:
        R_con -= 150.0 * (T_st_C - 85.0) ** 2

    # 8. 생존 보너스 (에피소드 완주 유도)
    progress = t_step / max(total_steps, 1)
    R_alive = 0.8 + 0.5 * progress  # 후반일수록 보너스 증가

    # 9. 터미널 페널티 (조기 종료)
    R_terminal = 0.0
    if terminated_early:
        remaining_frac = 1.0 - progress
        R_terminal = -5000.0 * remaining_frac  # 일찍 끝날수록 큰 페널티

    total = (R_H2 + R_eff + R_fc + R_SOC + R_deg +
             R_BoP + R_con + R_alive + R_terminal)
    return total


def _compute_reward_v3(m_dot_H2, P_fc_kwe, eta_fcs_val, P_eComp_kwe,
                        delta_SOH_fc, delta_SOH_bat, SOC_bat,
                        dPfc_dt, T_st_C, dt_ctrl, m_H2_0,
                        m_H2_remaining, distance_m_step,
                        t_step, total_steps, terminated_early):
    """
    보상 함수 v3 — 규칙 기반 모방 + 효율 극대화

    v2보다 더 공격적인 H2 절약 + 규칙 기반의 운전점을 유도하는 셰이핑
    """
    h2_consumed_step = m_dot_H2 * dt_ctrl
    tank_frac = max(m_H2_remaining / m_H2_0, 0.01)
    dist_km_step = distance_m_step / 1000.0

    # 1. H2 소비 페널티 — 제곱 + 탱크 잔량 반비례
    h2_norm = h2_consumed_step / m_H2_0
    R_H2 = -1200.0 * h2_norm / (tank_frac ** 0.7)

    # 2. km/kg 즉각 효율 (강화)
    if h2_consumed_step > 1e-9:
        km_per_kg = dist_km_step / h2_consumed_step
        R_eff = 4.0 * np.clip(km_per_kg / 180.0, 0.0, 2.5)
    elif dist_km_step > 0.001:
        R_eff = 3.0
    else:
        R_eff = 0.0

    # 3. 최적 운전점 보상 (8~12 kW 구간 = 최대 효율)
    if 7.0 <= P_fc_kwe <= 13.0:
        R_opt = 2.0  # 최적 운전점 보너스
    elif 5.0 <= P_fc_kwe <= 20.0:
        R_opt = 0.8
    elif P_fc_kwe < 2.0:
        R_opt = 0.3  # idle
    else:
        R_opt = 0.0

    # 4. FC 효율
    R_fc = 5.0 * max(0.0, eta_fcs_val - 0.55)

    # 5. SOC 관리
    R_SOC = -50.0 * (SOC_bat - 0.60) ** 2

    # 6. 열화
    R_deg = -300.0 * abs(delta_SOH_fc) - 150.0 * abs(delta_SOH_bat)

    # 7. BoP
    R_BoP = -3.0 * max(0.0, P_eComp_kwe - 0.5)

    # 8. 제약
    R_con = 0.0
    if SOC_bat < 0.25:
        R_con -= 1000.0 * (0.25 - SOC_bat) ** 2
    if SOC_bat > 0.92:
        R_con -= 1000.0 * (SOC_bat - 0.92) ** 2
    if abs(dPfc_dt) > 4.0:
        R_con -= 300.0 * (abs(dPfc_dt) - 4.0) ** 2
    if T_st_C > 85.0:
        R_con -= 200.0 * (T_st_C - 85.0) ** 2

    # 9. 생존 보너스
    progress = t_step / max(total_steps, 1)
    R_alive = 1.0 + 1.0 * progress

    # 10. 터미널 페널티
    R_terminal = 0.0
    if terminated_early:
        remaining_frac = 1.0 - progress
        R_terminal = -8000.0 * remaining_frac

    return (R_H2 + R_eff + R_opt + R_fc + R_SOC + R_deg +
            R_BoP + R_con + R_alive + R_terminal)


REWARD_FUNCTIONS = {
    "v2": _compute_reward_v2,
    "v3": _compute_reward_v3,
}


class FCHEVRLEnv(gym.Env):
    """
    FCHEV 강화학습 최적화 환경

    FCHEVEnv와 동일한 물리 시뮬레이션 + 개선된 보상/관측 공간
    - 관측: 14차원 (기존 12 + 탱크잔량변화율, 에피소드 진행률 강화)
    - 보상: v2/v3 선택 가능
    - 커리큘럼: duration_s를 점진적으로 증가
    """

    metadata = {"render_modes": []}

    def __init__(self, cycle_v, cycle_a,
                 duration_s=18000, dt_sim=0.01, dt_ctrl=1.0,
                 T_amb_C=23.0, SOC_init=0.60, scenario_id="A",
                 reward_mode="v2", curriculum_duration_s=None):
        super().__init__()

        self.cycle_v = cycle_v
        self.cycle_a = cycle_a
        self.full_duration_s = duration_s
        self.duration_s = curriculum_duration_s or duration_s
        self.dt_sim = dt_sim
        self.dt_ctrl = dt_ctrl
        self.substeps = max(1, int(round(dt_ctrl / dt_sim)))
        self.T_amb = T_amb_C + 273.15
        self.T_amb_C = T_amb_C
        self.SOC_init = SOC_init
        self.scenario_id = scenario_id
        self.reward_mode = reward_mode
        self.total_ctrl_steps = int(self.duration_s / dt_ctrl)
        self.cycle_len = len(cycle_v)

        self.reward_fn = REWARD_FUNCTIONS.get(reward_mode, _compute_reward_v2)

        # 14-dim observation
        self.action_space = spaces.Box(low=-1.0, high=1.0,
                                       shape=(1,), dtype=np.float32)
        self.observation_space = spaces.Box(low=-3.0, high=3.0,
                                            shape=(14,), dtype=np.float32)

        # 모델
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
        self._prev_distance_m = 0.0
        self._prev_H2_kg = 0.0

    def set_curriculum_duration(self, duration_s):
        """커리큘럼 학습용 에피소드 길이 설정"""
        self.duration_s = min(duration_s, self.full_duration_s)
        self.total_ctrl_steps = int(self.duration_s / self.dt_ctrl)

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
        self._prev_distance_m = 0.0
        self._prev_H2_kg = 0.0
        return self._get_obs(), {}

    def _get_cycle_values(self, sim_idx):
        idx = sim_idx % self.cycle_len
        return float(self.cycle_v[idx]), float(self.cycle_a[idx])

    def _get_obs(self):
        v, a = self._get_cycle_values(self.sim_step_idx)
        P_req = compute_P_req(v, a)
        P_low, _ = self.dwt.decompose(P_req)

        obs = np.array([
            np.clip(P_low / 1e3 / 82.0, -1.0, 1.5),      # 0: 저주파 부하
            self.battery.SOC,                                # 1: 배터리 SOC
            self.pemfc.SOH_fc,                               # 2: FC 건전성
            self.battery.SOH,                                # 3: 배터리 건전성
            self.tank.m_H2 / self.tank.m_H2_0,              # 4: 탱크 잔량 비율
            (self.pemfc.T_st_C - 65.0) / 25.0,              # 5: 스택 온도
            (self.battery.T_bat_C - 30.0) / 20.0,           # 6: 배터리 온도
            v / 36.0,                                        # 7: 차속
            np.clip(a / 4.0, -1.0, 1.0),                    # 8: 가속도
            (self.pemfc.lambda_mem - 10.0) / 6.0,            # 9: 막 수화도
            (eta_fcs(self.pemfc.P_fc_actual) - 0.56) / 0.11, # 10: FC 효율
            self.ctrl_step_idx / max(self.total_ctrl_steps, 1), # 11: 진행률
            # 추가 관측 (v2)
            np.clip(self.pemfc.P_fc_actual / 82.0, 0.0, 1.0), # 12: 현재 FC 출력 비율
            np.clip((self.total_H2_kg / max(self.tank.m_H2_0, 0.01)
                     - self.ctrl_step_idx / max(self.total_ctrl_steps, 1)),
                    -1.0, 1.0),                              # 13: H2소비-진행률 차이
        ], dtype=np.float32)
        return obs

    def step(self, action):
        if self.terminated or self.truncated:
            return self._get_obs(), 0.0, True, False, {}

        a_norm = float(action[0]) if hasattr(action, '__len__') else float(action)
        a_norm = np.clip(a_norm, -1.0, 1.0)
        P_fc_target = 5.0 + 77.0 * (a_norm + 1.0) / 2.0

        # 서브스테핑
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
        self._prev_distance_m = self.distance_m
        self._prev_H2_kg = self.total_H2_kg

        sub_terminated = False
        n_actual = 0
        last_fc_res = None
        last_bop_res = None
        last_tank_res = None

        for sub_i in range(self.substeps):
            if sub_terminated:
                break
            n_actual += 1

            v, a = self._get_cycle_values(self.sim_step_idx)
            acc_distance_m += v * self.dt_sim
            acc_v += v
            acc_a += a

            P_req_w = compute_P_req(v, a)
            P_low_w, P_high_w = self.dwt.decompose(P_req_w)
            P_low_kw = P_low_w / 1e3
            P_high_kw = P_high_w / 1e3
            acc_P_req += P_req_w / 1e3
            acc_P_low += P_low_kw
            acc_P_high += P_high_kw

            max_ramp = 4.0 * self.dt_sim
            P_fc_star = np.clip(P_fc_target,
                                self.pemfc.P_fc_prev - max_ramp,
                                self.pemfc.P_fc_prev + max_ramp)
            P_fc_star = np.clip(P_fc_star, 0.0, 82.0)

            last_bop_res = self.bop.compute(P_fc_star)
            acc_P_comp += last_bop_res["P_comp"]
            acc_P_fans += last_bop_res["P_fans"]
            acc_P_pump += last_bop_res["P_pump"]

            eta = eta_fcs(P_fc_star)
            if P_fc_star > 0.5 and eta > 0.01:
                m_dot_H2 = (P_fc_star * 1e3) / (eta * _LHV_H2)
            elif v < 0.5 and P_fc_star <= 0.5:
                m_dot_H2 = _IDLE_H2_KGS
            else:
                m_dot_H2 = 0.0

            last_fc_res = self.pemfc.step(P_fc_star, self.dt_sim, self.T_amb)
            last_fc_res["m_dot_H2"] = m_dot_H2
            acc_P_fc += last_fc_res["P_fc_kwe"]
            acc_eta += eta
            acc_m_dot_H2 += m_dot_H2

            last_tank_res = self.tank.step(m_dot_H2, self.T_amb, self.dt_sim)
            h2_consumed = m_dot_H2 * self.dt_sim
            acc_H2_kg += h2_consumed
            self.total_H2_kg += h2_consumed

            P_regen_w = 0.0
            if P_req_w < 0:
                m_v = VEHICLE_PARAMS["m_v"]
                F_brake = m_v * a
                P_regen_w = compute_regen(F_brake, v)
            acc_P_regen += P_regen_w / 1e3

            P_bat_kw = P_low_kw - P_fc_star
            if P_regen_w > 0:
                P_bat_kw -= P_regen_w / 1e3
            P_sc_kw = P_high_kw
            acc_P_bat += P_bat_kw
            acc_P_sc += P_sc_kw

            self.battery.step(P_bat_kw, self.dt_sim, self.T_amb)
            self.sc.step(P_sc_kw, self.dt_sim)

            self.sim_step_idx += 1
            if last_tank_res["shutdown_flag"]:
                sub_terminated = True
                self.terminated = True

        n_actual = max(n_actual, 1)
        self.distance_m += acc_distance_m

        delta_SOH_fc = self.pemfc.SOH_fc - SOH_fc_before
        delta_SOH_bat = self.battery.SOH - SOH_bat_before

        avg_eta = acc_eta / n_actual
        avg_m_dot_H2 = acc_m_dot_H2 / n_actual
        avg_P_fc = acc_P_fc / n_actual
        avg_P_comp = acc_P_comp / n_actual

        distance_m_step = self.distance_m - self._prev_distance_m

        self.ctrl_step_idx += 1
        if self.ctrl_step_idx >= self.total_ctrl_steps:
            self.truncated = True

        reward = self.reward_fn(
            avg_m_dot_H2, avg_P_fc, avg_eta,
            avg_P_comp, delta_SOH_fc, delta_SOH_bat,
            self.battery.SOC,
            last_fc_res["dP_dt"] if last_fc_res else 0.0,
            self.pemfc.T_st_C, self.dt_ctrl, self.tank.m_H2_0,
            self.tank.m_H2, distance_m_step,
            self.ctrl_step_idx, self.total_ctrl_steps,
            self.terminated
        )

        # 로그
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
            "P_fcs_net_kW": P_fc_star if last_fc_res else 0.0,
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

    def get_log_dict(self):
        return self.log


class FCHEVCurriculumWrapper(gym.Wrapper):
    """
    커리큘럼 학습 래퍼 — 에피소드 길이를 점진적 증가

    학습 진행에 따라 에피소드 길이를 늘려서
    에이전트가 먼저 단기 효율을 학습한 후 장기 관리를 배우도록 유도
    """

    def __init__(self, env, stages=None):
        super().__init__(env)
        self.stages = stages or [
            (0, 1800),       # 0~: 30분
            (20000, 3600),   # 20k~: 1시간
            (50000, 7200),   # 50k~: 2시간
            (100000, 14400), # 100k~: 4시간
            (200000, 18000), # 200k~: 5시간 (full A/B)
        ]
        self.total_steps_trained = 0
        self._update_duration()

    def _update_duration(self):
        current_dur = self.stages[0][1]
        for threshold, dur in self.stages:
            if self.total_steps_trained >= threshold:
                current_dur = dur
        self.env.set_curriculum_duration(current_dur)

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        self.total_steps_trained += 1
        return obs, reward, terminated, truncated, info

    def reset(self, **kwargs):
        self._update_duration()
        return self.env.reset(**kwargs)
