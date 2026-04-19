"""
FCHEV Rule-based vs SAC EMS 성능 비교 시각화
- 4 시나리오(A~D) × 2 EMS(rule/sac) 비교
- matplotlib 기반 다중 차트
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

plt.rcParams['font.size'] = 10
plt.rcParams['axes.titlesize'] = 12
plt.rcParams['figure.dpi'] = 150

DATA_DIR = Path(__file__).parent / "dataset"

SCENARIOS = {
    'A': ('UDDS_5h', 'UDDS 5h (시내)'),
    'B': ('HWFET_5h', 'HWFET 5h (고속)'),
    'C': ('UDDS_10h', 'UDDS 10h (시내 장거리)'),
    'D': ('HWFET_10h', 'HWFET 10h (고속 장거리)'),
}

COLORS = {'rule': '#2196F3', 'sac': '#FF5722'}


def load_scenario(scenario_key):
    """시나리오별 rule/sac CSV 로드 (시간 기준 리샘플링)"""
    suffix = SCENARIOS[scenario_key][0]
    data = {}
    for ems in ['rule', 'sac']:
        fpath = DATA_DIR / f"{scenario_key}_{ems}_{suffix}.csv"
        df = pd.read_csv(fpath)
        # rule 파일이 훨씬 크므로 시간 기준으로 1초 간격 리샘플링
        if len(df) > 50000:
            # 가장 가까운 정수 초로 그룹핑
            df['t_round'] = df['t'].round(0)
            df = df.groupby('t_round').last().reset_index(drop=True)
        data[ems] = df
    return data


def compute_summary(df):
    """단일 시뮬레이션 요약 지표 계산"""
    return {
        'distance_km': df['distance_km'].iloc[-1],
        'H2_consumed_kg': df['M_H2_kg'].iloc[-1],
        'final_SOC': df['SOC_bat'].iloc[-1],
        'final_SOH_fc': df['SOH_fc'].iloc[-1],
        'final_SOH_bat': df['SOH_bat'].iloc[-1],
        'mean_eta': df['eta_fcs'].mean(),
        'km_per_kg': df['distance_km'].iloc[-1] / max(df['M_H2_kg'].iloc[-1], 1e-9),
        'kg_per_100km': df['M_H2_kg'].iloc[-1] / max(df['distance_km'].iloc[-1], 1e-9) * 100,
        'total_time_h': df['t'].iloc[-1] / 3600,
    }


# ──────────────────────────────────────────────
# 1. 핵심 KPI 바 차트 (Figure 1)
# ──────────────────────────────────────────────
def plot_kpi_bars(summaries):
    fig, axes = plt.subplots(2, 3, figsize=(16, 9))
    fig.suptitle('FCHEV EMS 성능 비교: Rule-based vs SAC', fontsize=16, fontweight='bold')

    metrics = [
        ('distance_km', '주행 거리 (km)', '{:.1f}'),
        ('km_per_kg', '연비 (km/kg H₂)', '{:.1f}'),
        ('kg_per_100km', '수소 소비 (kg/100km)', '{:.3f}'),
        ('H2_consumed_kg', '총 수소 소비 (kg)', '{:.3f}'),
        ('final_SOC', '최종 배터리 SOC', '{:.3f}'),
        ('mean_eta', '평균 FCS 효율', '{:.3f}'),
    ]

    scenarios = list(summaries.keys())
    x = np.arange(len(scenarios))
    w = 0.35

    for idx, (key, title, fmt) in enumerate(metrics):
        ax = axes[idx // 3][idx % 3]
        rule_vals = [summaries[s]['rule'][key] for s in scenarios]
        sac_vals = [summaries[s]['sac'][key] for s in scenarios]

        bars_r = ax.bar(x - w/2, rule_vals, w, label='Rule', color=COLORS['rule'], alpha=0.85)
        bars_s = ax.bar(x + w/2, sac_vals, w, label='SAC', color=COLORS['sac'], alpha=0.85)

        # 값 표시
        for bar, val in zip(bars_r, rule_vals):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height(),
                    fmt.format(val), ha='center', va='bottom', fontsize=7)
        for bar, val in zip(bars_s, sac_vals):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height(),
                    fmt.format(val), ha='center', va='bottom', fontsize=7)

        ax.set_title(title)
        ax.set_xticks(x)
        ax.set_xticklabels([f"시나리오 {s}\n{SCENARIOS[s][1]}" for s in scenarios], fontsize=7)
        ax.legend(fontsize=8)

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(DATA_DIR / 'fig1_kpi_comparison.png', bbox_inches='tight')
    print("  [저장] fig1_kpi_comparison.png")
    return fig


# ──────────────────────────────────────────────
# 2. 개선율 요약 (Figure 2)
# ──────────────────────────────────────────────
def plot_improvement(summaries):
    fig, ax = plt.subplots(figsize=(12, 5))
    fig.suptitle('SAC 대비 Rule 개선율 (%)', fontsize=14, fontweight='bold')

    scenarios = list(summaries.keys())
    metrics = ['km_per_kg', 'kg_per_100km', 'mean_eta', 'distance_km']
    labels = ['연비 (km/kg)', '수소소비 (kg/100km)', '평균 효율', '주행 거리']
    colors_m = ['#4CAF50', '#FF9800', '#9C27B0', '#00BCD4']

    x = np.arange(len(scenarios))
    w = 0.18

    for i, (m, label, c) in enumerate(zip(metrics, labels, colors_m)):
        improvements = []
        for s in scenarios:
            rv = summaries[s]['rule'][m]
            sv = summaries[s]['sac'][m]
            if m == 'kg_per_100km':
                # 낮을수록 좋음 → 부호 반전
                imp = (rv - sv) / max(rv, 1e-9) * 100
            else:
                imp = (sv - rv) / max(rv, 1e-9) * 100
            improvements.append(imp)
        bars = ax.bar(x + i * w, improvements, w, label=label, color=c, alpha=0.85)
        for bar, val in zip(bars, improvements):
            ax.text(bar.get_x() + bar.get_width()/2,
                    bar.get_height() + (0.3 if val >= 0 else -1.2),
                    f"{val:+.1f}%", ha='center', va='bottom', fontsize=8)

    ax.axhline(0, color='black', linewidth=0.8)
    ax.axhline(5, color='green', linewidth=1, linestyle='--', alpha=0.5, label='목표 5%')
    ax.axhline(-5, color='red', linewidth=1, linestyle='--', alpha=0.5)
    ax.set_xticks(x + 1.5 * w)
    ax.set_xticklabels([f"시나리오 {s}" for s in scenarios])
    ax.set_ylabel('개선율 (%)')
    ax.legend(loc='upper left', fontsize=8)
    plt.tight_layout()
    fig.savefig(DATA_DIR / 'fig2_improvement.png', bbox_inches='tight')
    print("  [저장] fig2_improvement.png")
    return fig


# ──────────────────────────────────────────────
# 3. 시계열 비교 (시나리오별 Figure 3a~3d)
# ──────────────────────────────────────────────
def plot_timeseries(scenario_key, data):
    label = SCENARIOS[scenario_key][1]
    fig = plt.figure(figsize=(18, 14))
    fig.suptitle(f'시나리오 {scenario_key}: {label} — 시계열 비교', fontsize=15, fontweight='bold')
    gs = gridspec.GridSpec(4, 2, hspace=0.35, wspace=0.25)

    plots = [
        (0, 0, 'P_fc_kW',    '연료전지 출력 (kW)'),
        (0, 1, 'P_bat_kW',   '배터리 출력 (kW)'),
        (1, 0, 'SOC_bat',    '배터리 SOC'),
        (1, 1, 'M_H2_kg',    '누적 H₂ 소비 (kg)'),
        (2, 0, 'eta_fcs',    'FCS 효율'),
        (2, 1, 'distance_km','주행 거리 (km)'),
        (3, 0, 'T_st_C',     '스택 온도 (°C)'),
        (3, 1, 'reward',     '보상 (reward)'),
    ]

    for r, c, col, title in plots:
        ax = fig.add_subplot(gs[r, c])
        for ems in ['rule', 'sac']:
            df = data[ems]
            t_h = df['t'] / 3600
            if col in df.columns:
                ax.plot(t_h, df[col], label=ems.upper(), color=COLORS[ems],
                        alpha=0.7, linewidth=0.6)
        ax.set_title(title)
        ax.set_xlabel('시간 (h)')
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

    fname = f'fig3{scenario_key}_timeseries.png'
    fig.savefig(DATA_DIR / fname, bbox_inches='tight')
    print(f"  [저장] {fname}")
    return fig


# ──────────────────────────────────────────────
# 4. 효율 분포 히스토그램 (Figure 4)
# ──────────────────────────────────────────────
def plot_efficiency_dist(all_data):
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('FCS 효율 분포 비교 (Rule vs SAC)', fontsize=14, fontweight='bold')

    for idx, s in enumerate(SCENARIOS):
        ax = axes[idx // 2][idx % 2]
        for ems in ['rule', 'sac']:
            eta = all_data[s][ems]['eta_fcs'].dropna()
            eta = eta[(eta > 0) & (eta < 1)]
            ax.hist(eta, bins=80, alpha=0.5, label=ems.upper(),
                    color=COLORS[ems], density=True)
        ax.set_title(f"시나리오 {s}: {SCENARIOS[s][1]}")
        ax.set_xlabel('FCS 효율')
        ax.set_ylabel('밀도')
        ax.legend()
        ax.grid(True, alpha=0.3)

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(DATA_DIR / 'fig4_efficiency_dist.png', bbox_inches='tight')
    print("  [저장] fig4_efficiency_dist.png")
    return fig


# ──────────────────────────────────────────────
# 5. P_fc 운전점 히트맵 (Figure 5)
# ──────────────────────────────────────────────
def plot_operating_point(all_data):
    fig, axes = plt.subplots(2, 4, figsize=(20, 8))
    fig.suptitle('연료전지 출력 vs 차속 운전점 분포', fontsize=14, fontweight='bold')

    for idx, s in enumerate(SCENARIOS):
        for j, ems in enumerate(['rule', 'sac']):
            ax = axes[j][idx]
            df = all_data[s][ems]
            ax.hexbin(df['v_kmh'], df['P_fc_kW'], gridsize=40,
                      cmap='YlOrRd', mincnt=1)
            ax.set_title(f"{s} - {ems.upper()}", fontsize=10)
            ax.set_xlabel('속도 (km/h)')
            ax.set_ylabel('P_fc (kW)')

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(DATA_DIR / 'fig5_operating_point.png', bbox_inches='tight')
    print("  [저장] fig5_operating_point.png")
    return fig


# ──────────────────────────────────────────────
# 메인 실행
# ──────────────────────────────────────────────
def main():
    print("=" * 60)
    print("FCHEV Rule vs SAC 성능 비교 시각화")
    print("=" * 60)

    # 데이터 로드
    print("\n[1/6] 데이터 로드 중...")
    all_data = {}
    summaries = {}
    for s in SCENARIOS:
        print(f"  시나리오 {s} 로드 중...")
        all_data[s] = load_scenario(s)
        summaries[s] = {
            'rule': compute_summary(all_data[s]['rule']),
            'sac': compute_summary(all_data[s]['sac']),
        }

    # 요약 출력
    print("\n[2/6] 핵심 KPI 요약")
    print("-" * 90)
    print(f"{'시나리오':<10} {'EMS':<6} {'거리(km)':<10} {'H2(kg)':<10} "
          f"{'연비(km/kg)':<12} {'kg/100km':<10} {'SOC':<8} {'효율':<8}")
    print("-" * 90)
    for s in SCENARIOS:
        for ems in ['rule', 'sac']:
            sm = summaries[s][ems]
            print(f"  {s:<8} {ems:<6} {sm['distance_km']:>8.1f}  "
                  f"{sm['H2_consumed_kg']:>8.3f}  {sm['km_per_kg']:>10.1f}  "
                  f"{sm['kg_per_100km']:>8.3f}  {sm['final_SOC']:>6.3f}  "
                  f"{sm['mean_eta']:>6.3f}")
        # 개선율
        rv = summaries[s]['rule']['km_per_kg']
        sv = summaries[s]['sac']['km_per_kg']
        imp = (sv - rv) / rv * 100
        print(f"  {'':8} {'Δ':<6} {'':>8}  {'':>8}  {imp:>+9.1f}%")
    print("-" * 90)

    # 시각화
    print("\n[3/6] KPI 바 차트 생성...")
    plot_kpi_bars(summaries)

    print("\n[4/6] 개선율 차트 생성...")
    plot_improvement(summaries)

    print("\n[5/6] 시계열 차트 생성...")
    for s in SCENARIOS:
        plot_timeseries(s, all_data[s])

    print("\n[6/6] 효율 분포 & 운전점 차트 생성...")
    plot_efficiency_dist(all_data)
    plot_operating_point(all_data)

    print("\n" + "=" * 60)
    print("완료! dataset/ 폴더에 PNG 파일 저장됨")
    print("=" * 60)
    plt.show()


if __name__ == '__main__':
    main()
