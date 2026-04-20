# FCHEV Multi-Algorithm RL Training Manual

## 목차

1. [프로젝트 개요](#1-프로젝트-개요)
2. [시스템 요구사항](#2-시스템-요구사항)
3. [설치](#3-설치)
4. [프로젝트 구조](#4-프로젝트-구조)
5. [기존 파이프라인 (Rule vs SAC)](#5-기존-파이프라인-rule-vs-sac)
6. [RL 토너먼트 파이프라인](#6-rl-토너먼트-파이프라인)
7. [보상 함수 설계](#7-보상-함수-설계)
8. [알고리즘 상세](#8-알고리즘-상세)
9. [커리큘럼 학습](#9-커리큘럼-학습)
10. [결과 해석](#10-결과-해석)
11. [고급 사용법](#11-고급-사용법)
12. [트러블슈팅](#12-트러블슈팅)

---

## 1. 프로젝트 개요

### 배경

FCHEV(Fuel Cell Hybrid Electric Vehicle) 시뮬레이터는 수소연료전지 하이브리드 차량(현대 넥쏘 기반)의 에너지 관리 전략(EMS)을 비교 연구하는 플랫폼입니다.

### 목표

**규칙 기반 EMS의 수소 효율(130~175 km/kg H2)을 강화학습으로 초과 달성**

### 접근 방식

8개 RL 알고리즘을 토너먼트 방식으로 경쟁시켜 최적 에이전트를 선발합니다.

```
Phase 1: Screening    → 8개 알고리즘 × 2개 보상함수 = 16개 조합 스크리닝
Phase 2: Tournament   → km/kg H2 기준 순위 결정
Phase 3: Focus        → 상위 3~5개 집중 학습 (커리큘럼 + 장시간)
Phase 4: Evaluation   → 최종 에이전트를 6개 시나리오에서 평가
```

---

## 2. 시스템 요구사항

| 항목 | 최소 사양 | 권장 사양 |
|------|----------|----------|
| Python | 3.9+ | 3.10+ |
| RAM | 8 GB | 16 GB+ |
| GPU | 없음 (CPU 가능) | NVIDIA GPU (CUDA) 또는 Apple Silicon (MPS) |
| 저장공간 | 5 GB | 20 GB+ (모델 + 로그) |
| OS | macOS / Linux / Windows | - |

### GPU 지원

- **NVIDIA GPU**: CUDA 자동 감지, off-policy 알고리즘(SAC, TD3, TQC 등) 가속
- **Apple Silicon**: MPS(Metal Performance Shaders) 자동 감지
- **CPU**: 모든 알고리즘 지원, on-policy 알고리즘(PPO, A2C, TRPO)은 CPU가 오히려 효율적

---

## 3. 설치

```bash
# 저장소 클론
git clone https://github.com/leeminseo125/fuel-cell.git
cd fuel-cell

# 의존성 설치
pip install -r requirements.txt
```

### 주요 패키지

| 패키지 | 용도 |
|--------|------|
| `stable-baselines3` | SAC, PPO, TD3, DDPG, A2C |
| `sb3-contrib` | TQC, TRPO, ARS |
| `gymnasium` | RL 환경 인터페이스 |
| `torch` | 딥러닝 백엔드 |
| `CoolProp` | 수소 실기체 열역학 |
| `PyWavelets` | DWT 부하 분해 |
| `matplotlib` | 결과 시각화 |

---

## 4. 프로젝트 구조

```
fuel_cell/
├── fchev_pipeline.py          # 기존 파이프라인 (Rule vs SAC)
├── rl_train.py                # RL 토너먼트 파이프라인 (신규)
├── visualize_comparison.py    # 기존 결과 시각화
├── visualize_rl.py            # RL 토너먼트 시각화 (신규)
├── requirements.txt
├── MANUAL.md                  # 이 문서
│
├── env/
│   ├── env.py                 # 기본 시뮬레이터 환경
│   └── env_rl.py              # RL 최적화 환경 (신규)
│
├── control/
│   ├── rule_ems.py            # 규칙 기반 EMS (Z0~Z7)
│   ├── sac_agent.py           # 기존 SAC 에이전트
│   ├── multi_rl.py            # 멀티 알고리즘 에이전트 (신규)
│   └── dwt.py                 # 이산 웨이블릿 변환
│
├── models/                    # 물리 모델 (PEMFC, 배터리, 탱크 등)
├── pipeline/                  # 실행 프레임워크
├── scenarios/                 # 6개 시나리오 설정
├── dataset/                   # 시뮬레이션 CSV 로그
└── rl_results/                # RL 학습 결과 (자동 생성)
    ├── models/                # 학습된 모델 (.zip)
    ├── phase1_screening.csv
    ├── phase2_tournament.csv
    ├── phase3_focus.csv
    ├── final_comparison.csv
    └── fig_*.png              # 시각화 결과
```

---

## 5. 기존 파이프라인 (Rule vs SAC)

### 실행

```bash
# 전체 실행 (6 시나리오 × 2 EMS = 12 인스턴스)
python3 fchev_pipeline.py

# 규칙 기반만
python3 fchev_pipeline.py --rule-only

# SAC 학습 스텝 조정
python3 fchev_pipeline.py --timesteps 1000000

# 특정 시나리오만
python3 fchev_pipeline.py --scenarios A B D
```

### 시나리오 구성

| ID | 환경 | 사이클 | 시간 | 예상 거리 | 예상 H2 |
|----|------|--------|------|----------|---------|
| A | 시내 UDDS | UDDS | 5h | 322 km | 2.16 kg |
| B | 고속 HWFET | HWFET | 5h | 380 km | 2.74 kg |
| C | 시내 UDDS | UDDS | 10h | 640 km | 4.29 kg |
| D | 고속 HWFET | HWFET | 10h | 776 km | 5.59 kg |
| E | 시내 UDDS | UDDS | 24h | 1546 km | 6.33 kg |
| F | 고속 HWFET | HWFET | 24h | 1862 km | 6.33 kg |

---

## 6. RL 토너먼트 파이프라인

### 빠른 시작

```bash
# 기본 실행 (전체 4단계)
python3 rl_train.py

# 보상 함수 v2, v3 모두 시도 (권장)
python3 rl_train.py --reward both

# 학습 스텝 늘리기 (더 좋은 결과, 더 긴 시간)
python3 rl_train.py --screening-steps 300000 --focus-steps 2000000
```

### 단계별 실행

각 단계를 독립적으로 실행할 수 있습니다. 이전 단계의 결과가 `rl_results/`에 저장되어 있으면 자동으로 로드합니다.

```bash
# Phase 1만: 모든 알고리즘 스크리닝
python3 rl_train.py --phase 1

# Phase 2만: 토너먼트 순위 (Phase 1 결과 필요)
python3 rl_train.py --phase 2

# Phase 3만: 특정 알고리즘 집중 학습
python3 rl_train.py --phase 3 --algos SAC TQC PPO

# Phase 4만: 최종 평가 (Phase 3 결과 필요)
python3 rl_train.py --phase 4 --scenarios A B D
```

### 전체 옵션

```
옵션                        기본값      설명
─────────────────────────────────────────────────────────────
--phase N                   0 (전체)   실행할 단계 (1~4, 0=전체)
--screening-steps N         150,000    Phase 1 스크리닝 스텝 수
--focus-steps N             800,000    Phase 3 집중 학습 스텝 수
--reward {v2,v3,both}       v2         보상 함수 버전
--scenario S                A          학습 시나리오 ID
--scenarios S1 S2 ...       전체       Phase 4 평가 시나리오
--algos A1 A2 ...           자동       Phase 3 알고리즘 지정
--skip-rule                 false      규칙 기반 베이스라인 생략
```

### 출력 파일

```
rl_results/
├── models/
│   ├── screen_SAC_v2_A.zip       # Phase 1 스크리닝 모델
│   ├── screen_PPO_v2_A.zip
│   ├── focus_SAC_v2_A.zip        # Phase 3 집중 학습 모델
│   └── ...
├── phase1_screening.csv          # 스크리닝 결과 테이블
├── phase2_tournament.csv         # 토너먼트 순위
├── phase3_focus.csv              # 집중 학습 결과
├── final_comparison.csv          # 최종 비교 보고서
├── fig_tournament_ranking.png    # 토너먼트 순위 차트
├── fig_final_comparison.png      # 최종 비교 차트
├── fig_improvement_summary.png   # 개선율 요약
├── fig_ts_A_rule_vs_*.png        # 시계열 비교
├── fig_efficiency_dist.png       # 효율 분포
└── fig_operating_points.png      # 운전점 히트맵
```

---

## 7. 보상 함수 설계

### 기존 보상 (v1) — 문제점

```python
R_H2  = -100.0 * m_dot_H2 * dt / m_H2_0     # ~-0.016/step (너무 약함)
R_fc  = 8.0 * max(0, eta - 0.60)              # ~+0.4/step
R_SOC = -50.0 * (SOC - 0.60)^2                # SOC 관리
R_con = 제약 위반 페널티
# 문제: H2 페널티 < 효율 보상 → 에이전트가 H2를 낭비하며 학습
```

### 보상 v2 — 효율 극대화

```python
# 1. H2 소비 페널티 (강화, 탱크 잔량에 반비례)
R_H2 = -800 * (h2_consumed / m_H2_0) / sqrt(tank_fraction)

# 2. 즉각적 km/kg 효율 보상 (지배적 항)
R_eff = 3.0 * clip(km_per_kg_instant / 200, 0, 2.0)

# 3. FC 효율 보상
R_fc = 6.0 * max(0, eta - 0.55)

# 4. 생존 보너스 (에피소드 완주 유도)
R_alive = 0.8 + 0.5 * progress

# 5. 터미널 페널티 (조기 종료)
R_terminal = -5000 * remaining_fraction
```

### 보상 v3 — 규칙 기반 모방 + 효율 극대화

v2와 유사하지만 더 공격적:
- H2 페널티 -1200 (v2의 1.5배)
- 최적 운전점(8~12 kW) 보너스 +2.0
- 터미널 페널티 -8000

### 어떤 보상을 선택할까?

| 상황 | 권장 |
|------|------|
| 처음 실행 | `--reward both` (둘 다 시도) |
| 빠른 테스트 | `--reward v2` |
| 공격적 최적화 | `--reward v3` |

---

## 8. 알고리즘 상세

### 지원 알고리즘 (8개)

| 알고리즘 | 유형 | 특징 | 장점 |
|---------|------|------|------|
| **SAC** | Off-policy | 엔트로피 정규화 | 탐색-활용 균형, 연속 제어에 강함 |
| **PPO** | On-policy | 클리핑 기반 | 안정적, 튜닝 용이 |
| **TD3** | Off-policy | 쌍둥이 Q + 지연 갱신 | Q값 과추정 방지 |
| **DDPG** | Off-policy | 결정적 정책 | 연속 행동 기본형 |
| **A2C** | On-policy | 동기식 학습 | 빠른 학습, 멀티프로세싱 |
| **TQC** | Off-policy | 분포적 RL | SAC 개선, 분위수 기반 |
| **TRPO** | On-policy | 자연 경사법 | 이론적 보장, 안정적 |
| **ARS** | 무경사 | 랜덤 탐색 | 간단, 로컬 최적에 유리 |

### 하이퍼파라미터

모든 알고리즘은 GPU/CPU에 따라 자동 설정됩니다:

```
Off-policy (SAC, TD3, TQC):
  - learning_rate: 1e-4
  - gamma: 0.995 (장기 할인)
  - buffer_size: 2M (GPU) / 500K (CPU)
  - batch_size: 512 (GPU) / 256 (CPU)
  - net_arch: [512, 512, 256] (GPU) / [256, 256, 128] (CPU)

On-policy (PPO, A2C, TRPO):
  - learning_rate: 3e-4 ~ 7e-4
  - n_steps: 2048 ~ 4096
  - 항상 CPU에서 실행 (SB3 권장)
```

---

## 9. 커리큘럼 학습

에이전트가 먼저 단기 효율을 학습한 후 장기 관리 능력을 배우도록 에피소드 길이를 점진적으로 늘립니다.

### 스크리닝 (Phase 1) 커리큘럼

```
학습 스텝       에피소드 길이
──────────────────────────
0 ~  30,000    30분 (1,800 steps)
30K ~  60,000  1시간 (3,600 steps)
60K ~ 100,000  2시간 (7,200 steps)
100K ~ 120,000 4시간 (14,400 steps)
120K+          5시간 (18,000 steps)
```

### 집중 학습 (Phase 3) 커리큘럼

```
학습 스텝 비율    에피소드 길이
──────────────────────────
0 ~  5%         30분
5% ~ 10%        1시간
10% ~ 15%       1.5시간
15% ~ 25%       2시간
25% ~ 35%       3시간
35% ~ 50%       4시간
50% ~ 70%       5시간
70% ~ 85%       10시간 (해당 시)
85%+            전체 시나리오
```

---

## 10. 결과 해석

### 핵심 지표

| 지표 | 설명 | 규칙 기반 기준 |
|------|------|--------------|
| **km/kg H2** | 수소 1kg당 주행 거리 | UDDS: ~175, HWFET: ~131 |
| **kg/100km** | 100km당 수소 소비량 | UDDS: ~0.57, HWFET: ~0.76 |
| **improvement_pct** | 규칙 기반 대비 개선율 | 목표: +5% 이상 |
| **completed** | 시나리오 완주 여부 | True = 정상 완주 |
| **final_SOC** | 최종 배터리 잔량 | 0.30~0.90 정상 범위 |

### 결과 CSV 읽기

```python
import pandas as pd

# 최종 비교 보고서
df = pd.read_csv("rl_results/final_comparison.csv")
print(df[["scenario", "ems", "algo", "km_per_kg", "improvement_pct"]])

# 시뮬레이션 로그
log = pd.read_csv("dataset/A_SAC_UDDS_5h.csv")
print(log[["t", "P_fc_kW", "SOC_bat", "eta_fcs", "distance_km"]].describe())
```

### 시각화 재실행

```bash
# RL 토너먼트 시각화
python3 visualize_rl.py

# 기존 비교 시각화
python3 visualize_comparison.py
```

---

## 11. 고급 사용법

### 특정 알고리즘만 집중 학습

```bash
# SAC와 TQC만 200만 스텝 집중 학습
python3 rl_train.py --phase 3 --algos SAC TQC --focus-steps 2000000
```

### 다른 시나리오에서 학습

```bash
# 고속도로 시나리오(B)로 학습
python3 rl_train.py --scenario B

# 장거리 시나리오(D)로 학습
python3 rl_train.py --scenario D --screening-steps 200000
```

### Python API로 직접 사용

```python
from env.env_rl import FCHEVRLEnv
from control.multi_rl import MultiRLAgent, evaluate_agent
from scenarios.scenario_configs import SCENARIOS
from pipeline.cycle_loader import load_cycle

# 환경 생성
cfg = SCENARIOS["A"]
v, a = load_cycle(cfg["cycle"], cfg["duration_s"], dt=0.01)
env = FCHEVRLEnv(
    cycle_v=v, cycle_a=a,
    duration_s=cfg["duration_s"],
    dt_sim=0.01, dt_ctrl=1.0,
    T_amb_C=23.0, SOC_init=0.60,
    scenario_id="A",
    reward_mode="v2",
)

# 에이전트 생성 + 학습
agent = MultiRLAgent("SAC", env)
agent.build()
stats = agent.train(total_timesteps=500_000)

# 평가
result = evaluate_agent(agent, env)
print(f"효율: {result['km_per_kg']:.1f} km/kg H2")

# 모델 저장/로드
agent.save("my_model")
agent.load("my_model")
```

### 사용 가능한 알고리즘 확인

```python
from control.multi_rl import MultiRLAgent
print(MultiRLAgent.get_all_algo_names())
# ['SAC', 'PPO', 'TD3', 'DDPG', 'A2C', 'TQC', 'TRPO', 'ARS']
```

### 커스텀 보상 함수 추가

`env/env_rl.py`에 새 보상 함수를 정의하고 `REWARD_FUNCTIONS` 딕셔너리에 등록:

```python
def _compute_reward_custom(m_dot_H2, P_fc_kwe, eta_fcs_val, ...):
    # 커스텀 보상 로직
    return total_reward

REWARD_FUNCTIONS["custom"] = _compute_reward_custom
```

실행:
```bash
python3 rl_train.py --reward custom
```

---

## 12. 트러블슈팅

### sb3-contrib 설치 오류

```bash
pip install sb3-contrib>=2.0
# TQC, TRPO, ARS 없이도 SAC, PPO, TD3, DDPG, A2C는 사용 가능
```

### CUDA 메모리 부족

```bash
# 배치 크기와 버퍼 크기 줄이기 (control/multi_rl.py 수정)
# 또는 CPU 모드로 실행
CUDA_VISIBLE_DEVICES="" python3 rl_train.py
```

### CoolProp 설치 실패 (Apple Silicon)

```bash
# conda 환경에서 설치 권장
conda install -c conda-forge coolprop
```

### 학습이 너무 느림

- `--screening-steps 50000`으로 빠른 테스트
- `--phase 3 --algos SAC`로 단일 알고리즘만 집중
- GPU 확인: `python3 -c "import torch; print(torch.cuda.is_available())"`

### 메모리 부족 (RAM)

24시간 시나리오(E, F)는 86,400 스텝 로그를 생성하므로 메모리 사용량이 큽니다.
짧은 시나리오(A, B)부터 시작을 권장합니다.

---

## 참조

- **논문**: Sery & Leduc (2022), *Int. J. Engine Research*, 23(5), pp.709-720
- **차량**: 현대 넥쏘 (700bar x 3탱크, 6.33kg H2, 82kWe, eta_max=66.8%)
- **RL 프레임워크**: [Stable-Baselines3](https://stable-baselines3.readthedocs.io/)
- **환경 표준**: [Gymnasium](https://gymnasium.farama.org/)
