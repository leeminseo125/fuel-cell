# FCHEV 완전 물리 시뮬레이터 & SAC EMS 비교 플랫폼
## PRD v3.0 — 수소 탱크→바퀴 물리 원리 + 시나리오 설계 완전 통합

> **에너지 경로**: H₂ Tank(700 bar) → Regulator → PEMFC Stack → BoP → DC Bus → Battery/SC → Inverter → PMSM → Gearbox → Wheel
> **실험 근거**: Sery & Leduc, Int. J. Engine Research, 2022, 23(5), pp.709–720

| 항목 | 내용 |
|------|------|
| 문서 버전 | PRD v3.0 |
| 핵심 목표 | 1회 수소 충전 최대 주행 거리 달성 |
| 시나리오 | 시내 5h/10h + 고속도로 5h/10h (4개) |
| RL 제어 범위 | H₂ 탱크~배터리 EMS만 (주행 속도·경로 완전 고정) |
| 비교 구조 | 규칙 기반 vs SAC × 4시나리오 = 8 인스턴스 동시 실행 |
| 참조 차종 | 현대 넥쏘 (700 bar × 3탱크, 6.33 kg H₂, 82 kWe sys, η_max=66.8%) |

---

## 목차

1. [프로젝트 개요](#1-프로젝트-개요)
2. [실험 데이터 기반 보정 근거](#2-실험-데이터-보정-근거)
3. [완전 물리 원리 및 모델](#3-완전-물리-원리-및-모델)
   - 3.1 [수소 고압 탱크](#31-수소-고압-탱크)
   - 3.2 [PEMFC 스택 전기화학 모델](#32-pemfc-스택)
   - 3.3 [BoP 상세 모델](#33-bop-상세-모델)
   - 3.4 [DC 버스 및 전력 분배](#34-dc-버스)
   - 3.5 [리튬이온 배터리 모델](#35-배터리-모델)
   - 3.6 [슈퍼커패시터 모델](#36-슈퍼커패시터)
   - 3.7 [3상 인버터 및 PMSM 모터](#37-인버터--pmsm-모터)
   - 3.8 [기어박스·차동 장치·휠 동역학](#38-기어박스--휠-동역학)
   - 3.9 [회생 제동 에너지 회수](#39-회생-제동)
   - 3.10 [열관리 시스템 (TMS)](#310-tms)
   - 3.11 [DWT 부하 분해기](#311-dwt-부하-분해기)
4. [에너지 흐름 경로 전체 명세](#4-에너지-흐름-경로)
5. [시나리오 설계](#5-시나리오-설계)
6. [강화학습 EMS 설계 (SAC)](#6-rl-ems-설계)
7. [병렬 비교 플랫폼](#7-병렬-비교-플랫폼)
8. [소프트웨어 아키텍처](#8-sw-아키텍처)
9. [로드맵 및 성공 기준](#9-로드맵--성공-기준)
10. [부록 A: 논문 실험 수치](#부록-a-논문-실험-수치)
11. [부록 B: 시나리오별 수소 소비 계산](#부록-b-수소-소비-계산)
12. [부록 C: 전체 물리 수식 참조표](#부록-c-전체-수식-참조표)

---

## 1. 프로젝트 개요

### 1.1 핵심 목표

동일한 주행 속도·경로에서 **수소 탱크에서 배터리까지 에너지 관리 전략만 변경**하여
1회 수소 충전 주행 가능 거리를 극대화한다.

```
⛔ 절대 변경 불가 항목
   ① 주행 속도 v(t)      — 외부 CSV, 읽기 전용
   ② 가속도 a(t)         — v(t) 수치 미분, 자동 계산
   ③ 도로 경사 θ(t)      — 외부 데이터, 읽기 전용
   ④ L0~L2 계층 제어기   — 고정 설계
   ⑤ SC 순간 전력        — DWT 자동 결정
   ⑥ BoP 전력 소비       — FC 출력 함수, 자동 계산

✅ 에이전트의 유일한 제어 변수
   P_fc*(t) [kWe]: 연료전지 출력 기준값 ∈ [5, 82] kWe
```

### 1.2 4개 시나리오 × 2개 EMS = 8 인스턴스

| ID | 환경 | 사이클 | 시간 | 예상 거리 | 예상 H₂ | 탱크 비율 | 핵심 포인트 |
|----|------|-------|------|---------|--------|---------|-----------|
| A | 시내 | UDDS | 5h | ~322 km | ~2.16 kg | 34% | FC 최대 효율점 운전 |
| B | 고속 | HWFET | 5h | ~380 km | ~2.74 kg | 43% | 압축기 BoP 최적화 |
| C | 시내 | UDDS | 10h | ~640 km | ~4.29 kg | 68% | 장시간 SOH 열화 |
| **D** | **고속** | **HWFET** | **10h** | **~776 km** | **~5.59 kg** | **88% ⚠** | **10h 완주 가능 여부** |

---

## 2. 실험 데이터 보정 근거

**논문**: Sery & Leduc (2022), Int. J. Engine Research, 23(5), pp.709–720

### 2.1 FC 시스템 효율 맵 — 2D 룩업 테이블 (논문 그림 8)

```python
# 파일: models/pemfc_model.py
ETA_FCS_TABLE = {
    # P_fcs_kwe: eta_fcs
    0:  0.00,
    5:  0.64,
    8:  0.668,   # ← 최대 효율점 (Sery & Leduc 그림 8)
    10: 0.668,
    20: 0.630,
    30: 0.610,
    40: 0.590,
    50: 0.560,
    60: 0.540,
    70: 0.515,
    82: 0.480,   # ← 정격 출력 (Sery & Leduc 실험 측정)
}

def eta_fcs(P_kwe):
    return np.interp(P_kwe,
                     list(ETA_FCS_TABLE.keys()),
                     list(ETA_FCS_TABLE.values()))
```

### 2.2 BoP 전력 소비 — 룩업 테이블 (논문 그림 9)

```python
# 파일: models/bop_model.py
# 모든 수치: Sery & Leduc 그림 9 정상 상태 실험 측정값

P_COMP_TABLE = {  # P_fcs_kwe: P_eComp_kwe
    0: 0.0,  10: 0.5,  20: 1.5,  30: 2.5,  40: 3.5,
    50: 4.5, 60: 6.0,  70: 7.5,  82: 9.1,  # 최대 9.1 kWe
}
P_FANS_TABLE = {  # P_fcs_kwe: P_fans_kwe
    0: 0.2, 40: 0.2, 50: 0.2, 60: 0.5, 70: 1.2, 82: 1.8,
}
P_PUMP_TABLE = {  # P_fcs_kwe: P_pump_kwe
    0: 0.05, 50: 0.15, 60: 0.3, 70: 0.5, 82: 0.6,
}
P_12V_KWE = 0.273   # 고정 상수 (Sery & Leduc 그림 14 평균값)
```

### 2.3 수소 소비 실험 기준선 (논문 그림 17)

| 사이클 | [kg/100km] | 시나리오 | 검증 허용 오차 |
|-------|-----------|--------|------------|
| FTP-75 (UDDS) | **0.67** | A/C | ±0.05 |
| HWFET | **0.72** | B/D | ±0.05 |
| 공전 | **12.4 g/h** | A/C 정차 | ±2.0 g/h |

### 2.4 차량 제원 (논문 Table 2)

```python
# 파일: models/drivetrain.py
VEHICLE_PARAMS = {
    "m_v":   2057.0,      # kg — Sery & Leduc Table 2
    "a_rl":   178.7,      # N
    "b_rl":   0.919,      # N/(km/h)
    "c_rl":   0.04037,    # N/(km/h)²
    "eta_dt": 0.93,       # 기어박스+디퍼렌셜
    "r_t":    0.346,      # m (235/55R19)
}

# 수소 탱크: 3개 (Sery & Leduc Figure 1)
TANK_PARAMS = {
    "n_tanks":    3,
    "V_per_tank": 0.0522,   # m³ (52.2 L)
    "V_tank":     0.1566,   # m³ 총합
    "m_H2_0":     6.33,     # kg
    "GS_tank":    70.0,     # W/K — Sery & Leduc 그림 4 실험 보정
    "P_min_bar":  10.0,     # bar — 운전 하한
}
```

---

## 3. 완전 물리 원리 및 모델

> 모든 내부 계산 단위: SI (W, A, V, K, kg, Pa)
> CSV 저장 시만 편의 단위 변환 (kW, °C, km/h)

---

### 3.1 수소 고압 탱크

**파일**: `models/h2_tank.py` | **클래스**: `H2TankModel`

#### 물리 원리
700 bar 고압으로 저장된 수소는 이상 기체 거동을 크게 벗어나므로 Abel-Noble 실기체 방정식을 사용한다.
수소가 공급될 때 감압에 따른 온도 강하가 발생하며, Sery & Leduc의 열역학 모델로 재현한다.

#### 수식

```
Abel-Noble 실기체 상태 방정식:
  P = ρ·R·T / (1 − b·ρ)          b = 2.661×10⁻⁵ m³/mol      (3.1)

Sery & Leduc 탱크 열역학 (수식 2~3):
  m_H₂·c_v·dT/dt = ṁ_H₂·(c_p−c_v)·T + GS_tank·(T₁−T)        (3.3)
  dm_H₂/dt = −ṁ_H₂_out                                          (3.4)

Z, c_p, c_v: CoolProp 라이브러리 계산
ODE 수치 적분: SciPy Runge-Kutta (논문 동일 방법)
```

```python
class H2TankModel:
    """
    입력:  ṁ_H₂_out [kg/s], T_amb [K], dt [s]
    출력:  P_tank [Pa], P_tank_bar [bar], m_H₂_rem [kg],
           T_tank [K], shutdown_flag [bool]
    """
    V_tank     = 0.1566   # m³ (3 × 52.2 L) — Sery & Leduc Figure 1
    m_H2_0     = 6.33     # kg
    GS_tank    = 70.0     # W/K — Sery & Leduc 그림 4 실험 보정
    P_min_bar  = 10.0     # bar — 운전 하한 (이하 shutdown)

    def step(self, m_dot_out: float, T_amb: float, dt: float) -> dict: ...
```

---

### 3.2 PEMFC 스택

**파일**: `models/pemfc_model.py` | **클래스**: `PEMFCModel`

#### 물리 원리 — 전기화학 반응
- 애노드: H₂ → 2H⁺ + 2e⁻ (수소 산화)
- 캐소드: ½O₂ + 2H⁺ + 2e⁻ → H₂O (산소 환원)
- 생성된 전자(e⁻)가 외부 회로를 흘러 전류 생성 → 모터 구동

#### 수식

```
단위 셀 출력 전압:
  V_cell = E_Nernst − V_act − V_ohm − V_conc                     (3.5)

넌스트 전위:
  E_N = 1.229 − 8.5×10⁻⁴(T−298.15) + (RT/2F)·ln(P_H₂·P_O₂^0.5) (3.6)

활성화 과전압 (Tafel):
  V_act = (RT/αF)·ln(I_fc/I_0)    α=0.5, I_0=0.001 A/cm²        (3.7)

옴 손실:
  V_ohm = I_fc·(R_mem + R_contact)                               (3.8)

농도 과전압:
  V_conc = −(RT/nF)·ln(1 − I_fc/I_lim)                          (3.9)

스택: V_st = N_cell × V_cell,  P_stack = V_st × I_st
N_cell=440, A_cell=312 cm²

수소 소비 (Sery & Leduc 수식 1):
  ṁ_H₂ = I_st·N_cell·M_H₂ / (n·F·η_F)    η_F=0.98              (3.10)

막 수분화:
  dλ_mem/dt = (Ṅ_prod−Ṅ_evap−Ṅ_EOD) / δ_mem    λ∈[6,14]        (3.11)
  λ < 6 → 막 건조 (R_mem ×3)
  λ > 14 → 플러딩 (I_lim 급감)

스택 열 모델:
  C_th_st·dT_st/dt = Q̇_gen − Q̇_cool − Q̇_loss                   (3.12)
  Q̇_gen = N_cell·I_st·(1.482 − V_cell)                          (3.13)

FC SOH 열화:
  dSOH_fc/dt = −(σ₁|dP/dt|² + σ₂·idle + σ₃·N_ss + σ₄·overheat)  (3.14)
  σ₁=1.8e-16, σ₂=8.4e-9/s, σ₃=1.1e-7/회, σ₄=2.0e-8/s
  T_st > 85°C: σ₄ ×5
```

| 파라미터 | 수치 | 출처 |
|---------|------|------|
| N_cell | **440** | Sery & Leduc 수식(1) |
| P_stack 정격 | **94 kWe** | 실험 측정 |
| P_fcs 정격 | **82 kWe** | 실험 측정 |
| η_fcs_max | **66.8% @ 8~10 kWe** | 그림 8 |
| 과도 응답 | **760 ms** (0→90 kWe) | BAB130 |

---

### 3.3 BoP 상세 모델

**파일**: `models/bop_model.py` | **클래스**: `BoPModel`

#### 물리 원리
FC 스택 외에 공기 공급(압축기), 냉각(팬·펌프), 12V 보조 부하가 전력을 소비한다.
이들 합산이 시스템 효율을 스택 효율보다 낮추는 주된 원인이다.

```python
def compute_bop(P_fcs_kwe: float) -> dict:
    """
    Sery & Leduc 그림 9 룩업 테이블 적용
    P_fcs_net = P_fcs_kwe - P_BoP  ← DC 버스 실제 공급 전력
    """
    P_comp = np.interp(P_fcs_kwe,
                       [0, 10, 20, 30, 40, 50, 60, 70, 82],
                       [0, 0.5, 1.5, 2.5, 3.5, 4.5, 6.0, 7.5, 9.1])
    P_fans = np.interp(P_fcs_kwe,
                       [0, 40, 50, 60, 70, 82],
                       [0.2, 0.2, 0.2, 0.5, 1.2, 1.8])
    P_pump = np.interp(P_fcs_kwe,
                       [0, 50, 60, 70, 82],
                       [0.05, 0.15, 0.3, 0.5, 0.6])
    P_12V  = 0.273  # kWe — 고정 (Sery & Leduc 그림 14)
    P_BoP  = P_comp + P_fans + P_pump + P_12V
    return {
        "P_comp": P_comp, "P_fans": P_fans,
        "P_pump": P_pump, "P_12V": P_12V,
        "P_BoP": P_BoP,
        "P_fcs_net": P_fcs_kwe - P_BoP,   # 수식 (3.17)
    }
```

---

### 3.4 DC 버스

**파일**: `models/dc_bus.py` | **클래스**: `DCBusModel`

```
에너지 균형:
  P_fcs_net + P_bat + P_sc = P_inv + P_BOP + P_aux              (3.18)
  P_aux ≈ 1.5 kW (공조·조명, 고정)
  V_bus = 300~400 V DC (Sery & Leduc Figure 1)

DC/DC 컨버터 효율:
  FC 단방향 (BHDC):    η = 0.97
  배터리 방전:          η = 0.96
  배터리 충전:          η = 0.97
  SC 양방향:           η = 0.98
```

---

### 3.5 배터리 모델

**파일**: `models/battery.py` | **클래스**: `BatteryModel`

#### 물리 원리 — 배터리가 충전되는 두 경로
1. **FC 잉여 전력**: P_fcs_net > P_wheel 조건에서 잉여분 → 배터리 충전
2. **회생 제동**: PMSM 발전 → 인버터 정류 → 배터리 충전

```
2RC 등가 회로:
  V_bat = V_oc(SOC) − I_bat·R_0 − V_R1 − V_R2                  (3.19)
  dV_R1/dt = −V_R1/(R_1·C_1) + I_bat/C_1                        (3.20)
  dV_R2/dt = −V_R2/(R_2·C_2) + I_bat/C_2                        (3.21)
  SOC(t) = SOC_0 − (η_bat/Q_bat)·∫I_bat dt                      (3.22)

SOH 열화:
  dSOH/dt = −k_cyc·|I|·f_stress/Q_nom − k_cal·exp(−E_a/RT)     (3.23)
  f_stress = (1+γ_DOD·(DOD−0.30)²)·(1+γ_C·(Crate−1)²)          (3.24)
```

| 파라미터 | 수치 | 출처 |
|---------|------|------|
| 용량 | 1.56 kWh / 6.5 Ah | Sery & Leduc Figure 1 |
| 공칭 전압 | 240 V | Sery & Leduc Figure 1 |
| **최대 방전** | **39 kWe** | **BAB130 실험** |
| **최대 회생** | **35 kWe** | **BAB130 실험** |
| R_0 | 0.10 Ω | |
| k_cyc | 9.6×10⁻⁶ A⁻¹ | |

---

### 3.6 슈퍼커패시터

**파일**: `models/sc_model.py`

```
V_sc = Q_sc/C_sc − I_sc·R_ESR                                   (3.26)
SOC_sc = (V_sc²−V_min²) / (V_max²−V_min²)                      (3.27)

C_sc=22F, V_max=370V, V_min=185V, R_ESR=0.3mΩ
⚠️ DWT 전처리가 P_high를 자동 할당 — RL 에이전트 제어 대상 아님
```

---

### 3.7 인버터 + PMSM 모터

**파일**: `models/inverter.py`, `models/motor.py`

#### 물리 원리 — DC 전력이 바퀴 토크가 되는 과정
DC 버스 → 3상 IGBT 인버터(DC→3상 AC) → PMSM(전자기 토크) → 기어박스 → 바퀴

```
인버터 손실:
  P_loss = f_sw·E_sw·I_rms + 3·I_rms²·R_DS_on/2               (3.28)
  f_sw=10kHz, η_inv=0.96~0.98, η_inv_regen=0.95

PMSM d-q 전압 방정식:
  v_d = R_s·i_d + L_d·di_d/dt − ω_e·L_q·i_q                   (3.29)
  v_q = R_s·i_q + L_q·di_q/dt + ω_e·(L_d·i_d + ψ_pm)          (3.30)

전자기 토크:
  T_em = (3/2)·p·[ψ_pm·i_q + (L_d−L_q)·i_d·i_q]              (3.31)

회전 동역학:
  J·dω_m/dt = T_em − T_load − B·ω_m                            (3.32)
  p=4, J=0.05kg·m², B=0.01N·m·s/rad
```

| 파라미터 | 수치 |
|---------|------|
| 정격 출력 | 120 kW (연속) |
| 최대 토크 | 395 N·m |
| 최대 속도 | 16,000 rpm |
| R_s | 0.012 Ω |
| L_d | 0.18 mH |
| L_q | 0.34 mH |
| ψ_pm | 0.082 Wb |

---

### 3.8 기어박스 + 휠 동역학

**파일**: `models/drivetrain.py`

#### 물리 원리 — 모터 토크가 추진력으로 변환

> v(t)는 외부 고정 입력. 모델은 backward 방식으로 P_wheel을 역산한다.

```python
def compute_P_req(v_ms: float, a: float) -> float:
    """
    Sery & Leduc Table 2 로드로드 파라미터 직접 적용
    """
    v_kmh  = v_ms * 3.6
    F_rl   = 178.7 + 0.919 * v_kmh + 0.04037 * v_kmh**2   # 수식 (3.33)
    F_trac = 2057.0 * a + F_rl                              # 수식 (3.34)
    P_wheel = F_trac * v_ms / 0.93                          # 수식 (3.35)
    return P_wheel  # [W]
```

---

### 3.9 회생 제동

**파일**: `models/regen_brake.py`

#### 물리 원리 — 운동 에너지 → 배터리 전기 에너지
바퀴 감속 → PMSM 발전기 모드 → 인버터 정류 → DC 버스 → 배터리 충전

```python
P_REGEN_MAX_KWE = 35.0  # Sery & Leduc BAB130 실험 측정값

def compute_regen(F_brake, v_ms, f_regen=0.70,
                  eta_gen=0.97, eta_inv=0.95):
    """수식 (3.36)"""
    P_raw   = f_regen * abs(F_brake) * v_ms * eta_gen * eta_inv
    P_regen = min(P_raw, P_REGEN_MAX_KWE * 1000)  # W
    return P_regen

# 충전 우선 순위: SC (SOC_sc < 0.95) → 배터리 (I < I_chg_max)
# 시내(A/C): 잦은 감속으로 회생량 큼
# 고속(B/D): 회생 기회 적음
```

---

### 3.10 TMS

**파일**: `models/thermal.py`

```
스택 냉각:
  Q̇_cool = ṁ_cool·c_p·(T_out−T_in)                             (3.37)
  Q̇_rad  = U·A_rad·(T_cool−T_amb)                              (3.38)

목표: T_st=65~75°C, T_bat=15~45°C (독립 냉각 루프)
TMS는 RL 에이전트 제어 대상 아님 — 규칙 기반 피드백 제어 고정
```

| 위반 조건 | 패널티 |
|----------|--------|
| T_st > 85°C | σ₄ ×5 + η_fc 패널티 |
| T_st < 40°C | η_fc −15% |
| T_bat > 45°C | k_cyc ×3 |

---

### 3.11 DWT 부하 분해기

**파일**: `control/dwt.py`

```python
import pywt
from collections import deque

class DWTDecomposer:
    """
    Daubechies-4, J=3, 100Hz
    슬라이딩 윈도우 256 샘플 (2.56초)
    P_req = P_low + P_high                                        (3.39)
    P_low  → SAC 에이전트 입력
    P_high → SC 자동 할당 (에이전트 제어 불가)
    """
    wavelet = 'db4'
    level   = 3
    window  = 256

    def decompose(self, p_req: float) -> tuple[float, float]:
        # 반환: (P_low [W], P_high [W])
        ...
```

---

## 4. 에너지 흐름 경로

### 4.1 추진 경로 — 수소에서 바퀴까지 12단계

| 단계 | 구성 요소 | 물리 변환 | 수식/손실 |
|-----|---------|---------|---------|
| ① | H₂ 탱크 (700 bar) | 화학 에너지 저장 | Abel-Noble EOS, GS=70 W/K |
| ② | 2단 감압 조절기 | 700 bar → 1.0~1.6 bar | 손실 ~1% |
| ③ | PEMFC 스택 440셀 | H₂+O₂ → 전기+열+H₂O | η_max=66.8% @ 8~10 kWe |
| ④ | 전동 압축기 | 공기 공급 (전력 소비) | 최대 9.1 kWe |
| ⑤ | FC DC/DC (BHDC) | FC 전압 → 버스 전압 | η=0.97 |
| ⑥ | 고전압 DC 버스 | 전력 분배 | 에너지 균형 수식 (3.18) |
| ⑦ | 배터리 양방향 DC/DC | 피크 보조/충전 | η_dis=0.96, η_chg=0.97 |
| ⑧ | SC 양방향 DC/DC | 과도 흡수 (DWT 자동) | η=0.98 |
| ⑨ | 3상 IGBT 인버터 | DC → 3상 AC | η=0.96~0.98 |
| ⑩ | PMSM 모터 | 전기 → 기계 토크 | 최대 120 kW |
| ⑪ | 기어박스+차동 장치 | 토크 증배·배분 | η=0.93 |
| ⑫ | 구동 바퀴 | 기계력 → 추진 | F_trac = m·a + F_rl |

### 4.2 배터리 충전 경로 — 두 가지

| 경로 | 물리 흐름 | 발생 조건 | EMS 제어 방법 |
|-----|---------|---------|------------|
| **경로 1: FC 잉여** | FC 순출력 > 구동 요구 → 잉여분 → DC 버스 → 배터리 | 저부하 (시내 정차 후) | SAC가 P_fc*를 높게 설정 |
| **경로 2: 회생 제동** | 바퀴 운동 에너지 → PMSM 발전 → 인버터 정류 → 배터리 | 감속·제동 시 | f_regen=0.70 고정 (제어 불가) |

### 4.3 제어 계층

| 계층 | 제어 대상 | 시간 상수 | 담당 |
|-----|---------|---------|------|
| **L3** | **EMS — FC/배터리 전력 분배** | **0.1~1 s** | **SAC 또는 규칙 기반** |
| L2 | DC 버스 전압, 컨버터 듀티 | 1~10 ms | PI 전압 제어기 (고정) |
| L1 | 모터 d-q 전류 루프 (FOC) | 0.1~1 ms | FOC 전류 제어기 (고정) |
| L0 | IGBT 게이트 PWM (SVPWM) | < 0.1 ms | 모듈레이터 (고정) |

---

## 5. 시나리오 설계

### 5.1 시나리오 설정 딕셔너리

```python
# 파일: scenarios/scenario_configs.py
SCENARIOS = {
    "A": {
        "name": "시내 저속 5시간",
        "cycle": "UDDS",
        "duration_s": 18_000,
        "temperature_C": 23,
        "include_idle": True,           # 정차 구간 12.4 g/h 적용
        "fc_per_100km_ref": 0.67,       # Sery & Leduc 기준선
        "expected_distance_km": 322,
        "expected_h2_kg": 2.16,
        "tank_fraction": 0.34,
    },
    "B": {
        "name": "고속도로 고속 5시간",
        "cycle": "HWFET",
        "duration_s": 18_000,
        "temperature_C": 23,
        "include_idle": False,
        "fc_per_100km_ref": 0.72,
        "expected_distance_km": 380,
        "expected_h2_kg": 2.74,
        "tank_fraction": 0.43,
    },
    "C": {
        "name": "시내 저속 10시간",
        "cycle": "UDDS",
        "duration_s": 36_000,
        "temperature_C": 23,
        "include_idle": True,
        "fc_per_100km_ref": 0.67,
        "expected_distance_km": 640,
        "expected_h2_kg": 4.29,
        "tank_fraction": 0.68,
    },
    "D": {
        "name": "고속도로 고속 10시간",
        "cycle": "HWFET",
        "duration_s": 36_000,
        "temperature_C": 23,
        "include_idle": False,
        "fc_per_100km_ref": 0.72,
        "expected_distance_km": 776,
        "expected_h2_kg": 5.59,
        "tank_fraction": 0.88,
        "critical": True,               # 탱크 고갈 위험 — 핵심 검증
    },
}
```

### 5.2 시나리오별 EMS 전략 포인트

**시나리오 A/C (시내 UDDS)**:
- FC를 최대 효율점(8~10 kWe, η=66.8%)에 최대한 고정
- 잦은 감속으로 회생 제동 에너지를 적극 배터리 충전
- 정차 중 FC 주기 가동(12.4 g/h) 최소화
- 저부하 구간에서 FC 잉여 → 배터리 사전 충전 (경로 1 활용)

**시나리오 B/D (고속 HWFET)**:
- 요구 출력 30~55 kWe 집중 → 압축기 전력(4~6 kWe) 최적화 핵심
- 회생 기회 적어 배터리는 주로 피크 보조 역할
- SAC: 압축기 소비 최소화되는 FC 출력 구간 학습

### 5.3 시나리오 D 핵심 검증

```
규칙 기반: 0.72 kg/100km × 776 km = 5.59 kg → 잔량 0.74 kg (12%) ⚠ 위험
SAC 목표:  0.68 kg/100km × 776 km = 5.27 kg → 잔량 1.06 kg (17%) ✅ 완주

동일한 주행(속도·경로 고정)에서
에너지 관리 전략만으로 10h 완주 가능 여부가 결정된다.
```

### 5.4 경계 조건 처리

```python
def _check_boundary(self):
    if self.battery.SOC < 0.30:
        self.forced_charge = True          # FC 출력 강제 증가 (Z6)
    if self.battery.SOC > 0.90:
        self.forced_reduce = True          # FC 출력 강제 감소 (Z7)
    if self.tank.P_tank_bar < 10.0:
        self.terminated = True             # 탱크 고갈 → 시뮬레이션 종료
        self.termination_time = self.t
    if self.pemfc.T_st_C > 85.0:
        self.pemfc.overheat_sigma4 = True  # σ₄ ×5 열화 가속
```

---

## 6. RL EMS 설계

### 6.1 상태 공간 s(t) ∈ ℝ¹²

| idx | 변수 | 물리 의미 | 범위 | 정규화 |
|-----|------|----------|------|--------|
| s₁ | P_low | DWT 저주파 동력 | [0, 82] kWe | ÷82 |
| s₂ | SOC_bat | 배터리 충전량 | [0.30, 0.90] | 그대로 |
| s₃ | SOH_fc | FC 건강도 | [0.85, 1.00] | 그대로 |
| s₄ | SOH_bat | 배터리 건강도 | [0.85, 1.00] | 그대로 |
| s₅ | m_H₂/m_H₂_0 | 탱크 잔량 비율 | [0, 1] | 그대로 |
| s₆ | T_st | FC 스택 온도 | [40, 90] °C | (T−65)/25 |
| s₇ | T_bat | 배터리 온도 | [15, 50] °C | (T−30)/20 |
| s₈ | v | 차량 속도 | [0, 36] m/s | ÷36 |
| s₉ | a | 가속도 | [−4, 4] m/s² | ÷4 |
| s₁₀ | λ_mem | 막 수분화 | [4, 16] | (λ−10)/6 |
| s₁₁ | η_fcs_cur | 현재 FC 효율 | [0.45, 0.67] | (η−0.56)/0.11 |
| **s₁₂** | **t_elapsed/t_total** | **시나리오 진행률** | **[0, 1]** | **그대로** |

### 6.2 행동 공간 및 역정규화

```python
# SAC Actor 출력: tanh → [-1, 1]
a_norm ∈ [-1, 1]

# 역정규화 (수식 6.1) — 상한 82 kWe (Sery & Leduc 실험 정격)
P_fc_star = 5.0 + 77.0 * (a_norm + 1.0) / 2.0    # [kWe]

# 전력 경사 제한 (수식 6.2)
P_fc_star = np.clip(P_fc_star,
                    P_fc_prev - 4.0 * dt,
                    P_fc_prev + 4.0 * dt)

# BoP 자동 계산 (수식 6.3)
bop_result  = bop_model.compute(P_fc_star)
P_fcs_net   = bop_result["P_fcs_net"]              # DC 버스 실제 공급

# 배터리 자동 결정 (수식 6.5)
P_bat = P_low - P_fcs_net
```

### 6.3 보상 함수

```python
def compute_reward(m_dot_H2, P_fc_kwe, eta_fcs, P_eComp_kwe,
                   delta_SOH_fc, delta_SOH_bat, SOC_bat,
                   dPfc_dt, T_st_C, dt, m_H2_0=6.33):

    # (6.6) 수소 효율 극대화 — 핵심 보상
    # 수소 1% 절감 ≡ 주행 거리 1% 증가
    R_H2  = -100.0 * m_dot_H2 * dt / m_H2_0

    # (6.7) FC 최대 효율점 인센티브 (66.8% @ 8~10 kWe)
    R_fc  =   8.0 * max(0, eta_fcs - 0.60)

    # (6.8) 압축기 과소비 패널티
    R_BoP = -3.0 * max(0, P_eComp_kwe - 0.5)

    # (6.9) SOH 열화 패널티
    R_deg = -350.0 * abs(delta_SOH_fc) - 200.0 * abs(delta_SOH_bat)

    # (6.10) SOC 추적 패널티
    R_SOC = -50.0 * (SOC_bat - 0.60)**2

    # (6.11) 구속 조건 위반 패널티
    R_con = (-1000.0 * max(0, 0.30 - SOC_bat)**2
             -1000.0 * max(0, SOC_bat - 0.90)**2
             - 500.0 * max(0, abs(dPfc_dt) - 4.0)**2
             - 200.0 * max(0, T_st_C - 85.0)**2)

    return R_H2 + R_fc + R_BoP + R_deg + R_SOC + R_con  # (6.12)
```

### 6.4 SAC 하이퍼파라미터

| 파라미터 | 값 | 설명 |
|---------|-----|------|
| 학습률 (Actor/Critic) | 3×10⁻⁴ | Adam |
| 할인 계수 γ | 0.99 | 장기 수소 절감 |
| 온도 α | 자동 튜닝 | H̄ = −1 |
| 리플레이 버퍼 | 1,000,000 | |
| 배치 크기 | 256 | |
| 은닉층 | [256, 256, 128] | Actor·Critic |
| 학습 에피소드 | 2,000 / 시나리오 | 시내·고속 독립 학습 |
| 제어 주기 Δt | 0.1 s | 10 Hz |

---

## 7. 병렬 비교 플랫폼

### 7.1 8 인스턴스 동시 실행

```
[시나리오 설정: A/B/C/D]
        ↓
[주행 사이클 v(t)] — 공유, 읽기 전용
        ↓
┌──────────┬──────────┬──────────┬──────────┐
│ A-Rule   │  A-SAC   │ B-Rule   │  B-SAC   │
│ UDDS 5h  │ UDDS 5h  │ HWFET 5h │ HWFET 5h │
├──────────┼──────────┼──────────┼──────────┤
│ C-Rule   │  C-SAC   │ D-Rule   │  D-SAC   │
│ UDDS 10h │ UDDS 10h │HWFET 10h │HWFET 10h │
└──────────┴──────────┴──────────┴──────────┘
        ↓
[comparison_report.csv] — 8개 결과 자동 비교
```

**공유 설정**: SOC₀=0.60, SOH₀=1.0, m_H₂_0=6.33 kg, T_amb=23°C

### 7.2 성능 비교 지표

| 지표 | 단위 | 기준선 | 비고 |
|-----|------|--------|------|
| km/kg H₂ (주) | km/kg | 시내 149.3 / 고속 138.9 | 핵심 |
| SAC 개선율 | % | > 5% | 성공 기준 |
| D시나리오 완주 | bool | 규칙:위험 / SAC:목표 | 가치 입증 |
| FC 평균 효율 | % | 48~63% | Sery & Leduc 범위 |
| 공전 소비 | g/h | 12.4 | Sery & Leduc |

---

## 8. SW 아키텍처

### 8.1 파일 구조

```
fchev_project/
├── fchev_pipeline.py           # 단일 진입점 (이 파일만 실행)
├── requirements.txt
├── scenarios/
│   └── scenario_configs.py
├── models/
│   ├── h2_tank.py              # Abel-Noble + GS=70 W/K
│   ├── pemfc_model.py          # 2D 룩업 η_max=66.8%
│   ├── bop_model.py            # 압축기+팬+펌프+12V
│   ├── dc_bus.py               # 에너지 균형
│   ├── battery.py              # 2RC-ECM + SOH
│   ├── sc_model.py             # 22F/370V
│   ├── inverter.py             # η_inv 룩업
│   ├── motor.py                # d-q + 효율 맵
│   ├── drivetrain.py           # Sery & Leduc 로드로드
│   ├── regen_brake.py          # 최대 35 kWe
│   └── thermal.py              # TMS (고정)
├── control/
│   ├── dwt.py                  # db4, J=3
│   ├── rule_ems.py             # Z0~Z6
│   └── sac_agent.py            # stable-baselines3 SAC
├── env/
│   └── env.py                  # FCHEVEnv (12차원, 1차원)
├── pipeline/
│   ├── cycle_loader.py         # UDDS/HWFET 자동 로드
│   ├── runner.py               # 8 인스턴스 병렬
│   └── validator.py            # Sery & Leduc 기준선 검증
└── dataset/                    # 자동 생성
    ├── A_rule_UDDS_5h.csv      # 28컬럼
    ├── A_sac_UDDS_5h.csv
    ├── B_rule_HWFET_5h.csv
    ├── B_sac_HWFET_5h.csv
    ├── C_rule_UDDS_10h.csv
    ├── C_sac_UDDS_10h.csv
    ├── D_rule_HWFET_10h.csv
    ├── D_sac_HWFET_10h.csv
    └── comparison_report.csv
```

### 8.2 requirements.txt

```
numpy>=1.24
pandas>=2.0
scipy>=1.11
PyWavelets>=1.4
stable-baselines3>=2.0
gymnasium>=0.29
wltp>=1.1.1
requests>=2.31
tqdm>=4.66
matplotlib>=3.7
CoolProp>=6.4
```

### 8.3 실행 명령

```bash
pip install -r requirements.txt
python fchev_pipeline.py
```

### 8.4 28개 CSV 컬럼

```
t, v_ms, v_kmh, a, P_req_kW, P_low_kW, P_high_kW,
P_fc_kW, P_fcs_net_kW, P_bat_kW, P_sc_kW,
P_comp_kW, P_fans_kW, P_coolpump_kW,
m_H2_gs, M_H2_kg,
SOC_bat, SOH_fc, SOH_bat,
T_st_C, T_bat_C, eta_fcs, dPfc_dt, lambda_mem,
P_tank_bar, distance_km, P_regen_kW,
ems_zone [규칙 전용] / action_norm + reward [SAC 전용]
```

---

## 9. 로드맵 & 성공 기준

### 9.1 마일스톤

| 단계 | 기간 | 산출물 | 검증 기준 |
|-----|------|--------|----------|
| M1 | 1~2주 | H₂ 탱크 + FC + BoP | η@10kWe=66.8±2% |
| M2 | 2~3주 | 배터리 + SC + DC 버스 | V_bat 오차 < 2% |
| M3 | 3~4주 | 인버터 + PMSM + 기어박스 + 차량 동역학 | UDDS P_req 오차 < 3% |
| M4 | 4~5주 | TMS + 회생 제동 + DWT | 회생 상한 35 kWe |
| M5 | 5~6주 | 규칙 기반 + Gym + 4개 시나리오 | UDDS 0.67±0.05 kg/100km |
| M6 | 6~8주 | SAC 학습 (시나리오별 2,000 에피소드) | 4개 시나리오 수렴 |
| M7 | 8~9주 | 8 인스턴스 병렬 + 비교 보고서 | 자동 생성 확인 |
| M8 | 9~10주 | 최적화 + 문서 | D시나리오 완주 여부 |

### 9.2 성공 기준 (Definition of Done)

| # | 기준 | 목표값 |
|---|------|--------|
| 1 | FC 효율 맵 재현 | η@10kWe = 66.8 ± 2% |
| 2 | UDDS 소비 재현 | 0.67 ± 0.05 kg/100km |
| 3 | HWFET 소비 재현 | 0.72 ± 0.05 kg/100km |
| 4 | 공전 소비 검증 | 12.4 ± 2.0 g/h |
| 5 | 에너지 균형 오차 | |오차| < 0.5 kW |
| **6** | **SAC 개선율** | **> 5% (km/kg H₂)** |
| **7** | **시나리오 D 10h 완주** | **SAC: 완주 / 규칙: 한계** |
| 8 | 28컬럼 CSV + 비교 보고서 | 자동 생성 |
| 9 | v(t) 완전 고정 확인 | 코드 리뷰 통과 |

---

## 부록 A. 논문 실험 수치

**전체 출처: Sery & Leduc (2022), Int. J. Engine Research, 23(5), pp.709–720**

| 항목 | 수치 | 단위 | 논문 위치 |
|-----|------|------|---------|
| 스택 정격 출력 | 94 | kWe | 실험 측정 |
| **시스템 정격 출력** | **82** | **kWe** | 실험 측정 |
| **최대 시스템 효율** | **66.8** | **%** | 그림 8 |
| **최대 효율 발생 출력** | **8~10** | **kWe** | 그림 8 |
| 정격 시스템 효율 | 48.0 | % | 그림 8 |
| **압축기 최대 소비** | **9.1** | **kWe** | 그림 9 |
| 냉각 팬 최대 | 1.8 | kWe | 그림 9 |
| 냉각수 펌프 최대 | 0.6 | kWe | 그림 9 |
| **12V LDC 평균** | **273** | **W** | 그림 14 |
| FC 과도 응답 | 760 | ms | BAB130 |
| **배터리 최대 방전** | **39** | **kWe** | BAB130 |
| **배터리 최대 회생** | **35** | **kWe** | BAB130 |
| **GS_tank** | **70** | **W/K** | 그림 4 |
| **공전 소비** | **12.4** | **g/h** | 그림 14 |
| FTP-75 (UDDS) | **0.67** | kg/100km | 그림 17 |
| HWFET | **0.72** | kg/100km | 그림 17 |
| WLTC @23°C | 0.92 | kg/100km | 그림 17 |
| WLTC @10°C | 1.11 | kg/100km | 그림 17 |
| **차량 질량** | **2,057** | **kg** | Table 2 |
| **로드로드 a** | **178.7** | **N** | Table 2 |
| **로드로드 b** | **0.919** | **N/(km/h)** | Table 2 |
| **로드로드 c** | **0.04037** | **N/(km/h)²** | Table 2 |
| 수소 탱크 | 3 × 52.2 L @700 bar | — | Figure 1 |
| N_cell | **440** | 개 | 수식 (1) |

---

## 부록 B. 수소 소비 계산

| ID | EMS | 거리 | 소비율 | 총 H₂ | 탱크 잔량 | 판정 |
|----|-----|------|--------|------|---------|------|
| A | 규칙 | 322 km | 0.67 | 2.16 kg | 4.17 kg (66%) | ✅ 충분 |
| A | SAC | 322 km | ~0.635 | ~2.05 kg | ~4.28 kg | ✅ 목표 |
| B | 규칙 | 380 km | 0.72 | 2.74 kg | 3.59 kg (57%) | ✅ 충분 |
| B | SAC | 380 km | ~0.684 | ~2.60 kg | ~3.73 kg | ✅ 목표 |
| C | 규칙 | 640 km | 0.67 | 4.29 kg | 2.04 kg (32%) | ✅ 충분 |
| C | SAC | 640 km | ~0.635 | ~4.06 kg | ~2.27 kg | ✅ 목표 |
| **D** | **규칙** | **776 km** | **0.72** | **5.59 kg** | **0.74 kg (12%)** | **⚠️ 위험** |
| **D** | **SAC** | **776 km** | **~0.68** | **~5.27 kg** | **~1.06 kg (17%)** | **✅ 완주 목표** |

---

## 부록 C. 전체 수식 참조표

| 번호 | 모듈 | 수식 | 주요 파라미터 |
|-----|------|------|------------|
| (3.1) | H₂ 탱크 | P=ρRT/(1−b·ρ) | b=2.661e-5 |
| (3.3) | H₂ 탱크 열역학 | m·c_v·dT/dt=ṁ·(c_p−c_v)·T+GS·ΔT | GS=70 W/K |
| (3.5) | PEMFC | V_cell=E_N−V_act−V_ohm−V_conc | N_cell=440 |
| (3.10) | PEMFC 수소 | ṁ_H₂=I·N·M/(n·F·η_F) | η_F=0.98 |
| (3.14) | PEMFC SOH | dSOH/dt=−(σ₁\|dP/dt\|²+σ₂+σ₃+σ₄) | σ₁=1.8e-16 |
| (3.17) | FC 순출력 | P_net=P_st·η_dc−P_comp−P_fans−P_pump−P_12V | η_dc=0.97 |
| (3.18) | DC 버스 | P_fcs+P_bat+P_sc=P_inv+P_BOP+P_aux | P_aux=1.5 kW |
| (3.19) | 배터리 | V_bat=V_oc−I·R_0−V_R1−V_R2 | R_0=0.10 Ω |
| (3.23) | 배터리 SOH | dSOH/dt=−k_cyc·\|I\|·f_stress−k_cal | k_cyc=9.6e-6 |
| (3.26) | SC | V_sc=Q/C−I·R_ESR | C=22F |
| (3.28) | 인버터 | P_loss=f_sw·E_sw·I+3I²·R/2 | f_sw=10kHz |
| (3.29~32) | PMSM | d-q 전압+토크+동역학 | p=4, ψ=0.082 Wb |
| (3.33~35) | 차량 동역학 | F_rl=a+b·v+c·v² → P_wheel | a=178.7N |
| (3.36) | 회생 제동 | P_regen=min(f·\|F\|·v·η, 35kWe) | f_max=0.70 |
| (6.1) | 행동 역정규화 | P_fc*=5+77·(a+1)/2 [kWe] | [5, 82] kWe |
| (6.12) | 보상 함수 | R=R_H₂+R_fc+R_BoP+R_deg+R_SOC+R_con | w₁=100 |

---

*Document End — FCHEV PRD v3.0 | 수소 탱크→바퀴 완전 물리 원리 + 4개 시나리오 + SAC EMS*
