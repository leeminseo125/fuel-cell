"""
FCHEV Multi-Algorithm RL Tournament 결과 시각화

- 토너먼트 순위 차트
- 알고리즘별 km/kg H2 비교
- 최고 RL vs Rule-based 시계열 비교
- 효율 분포, 운전점 히트맵
- 학습 커브 (보상 추이)
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

RESULTS_DIR = Path(__file__).parent / "rl_results"
DATASET_DIR = Path(__file__).parent / "dataset"

COLORS_ALGO = {
    'Rule-based': '#2196F3',
    'SAC': '#FF5722',
    'PPO': '#4CAF50',
    'TD3': '#9C27B0',
    'DDPG': '#FF9800',
    'A2C': '#00BCD4',
    'TQC': '#E91E63',
    'TRPO': '#795548',
    'ARS': '#607D8B',
}


def plot_tournament_ranking(report_df=None, save=True):
    """Phase 2 토너먼트 순위 바 차트"""
    if report_df is None:
        path = RESULTS_DIR / "phase2_tournament.csv"
        if not path.exists():
            print("  [SKIP] No tournament data")
            return
        report_df = pd.read_csv(path)

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    fig.suptitle('RL Tournament — Algorithm Screening Results',
                 fontsize=14, fontweight='bold')

    valid = report_df[~report_df.get("error", pd.Series(dtype=str)).notna() |
                       report_df.get("error", pd.Series(dtype=str)).isna()]
    if "km_per_kg" not in valid.columns:
        print("  [SKIP] No km_per_kg column in tournament data")
        return

    valid = valid.sort_values("km_per_kg", ascending=True)

    # km/kg H2 순위
    ax = axes[0]
    algos = valid["algo"].values
    km_kgs = valid["km_per_kg"].values
    colors = [COLORS_ALGO.get(a, '#999999') for a in algos]
    bars = ax.barh(range(len(algos)), km_kgs, color=colors, alpha=0.85)
    ax.set_yticks(range(len(algos)))
    ax.set_yticklabels([f"{a} ({valid.iloc[i].get('reward_mode', '?')})"
                        for i, a in enumerate(algos)], fontsize=9)
    ax.set_xlabel('km/kg H₂')
    ax.set_title('Algorithm Ranking (km/kg H₂)')
    for bar, val in zip(bars, km_kgs):
        ax.text(bar.get_width() + 0.5, bar.get_y() + bar.get_height()/2,
                f'{val:.1f}', ha='left', va='center', fontsize=9)

    # 주행 거리 비교
    ax = axes[1]
    dists = valid["distance_km"].values if "distance_km" in valid.columns else np.zeros(len(algos))
    bars = ax.barh(range(len(algos)), dists, color=colors, alpha=0.85)
    ax.set_yticks(range(len(algos)))
    ax.set_yticklabels(algos, fontsize=9)
    ax.set_xlabel('Distance (km)')
    ax.set_title('Distance Covered')

    plt.tight_layout(rect=[0, 0, 1, 0.93])
    if save:
        fig.savefig(RESULTS_DIR / 'fig_tournament_ranking.png', bbox_inches='tight')
        print("  [saved] fig_tournament_ranking.png")
    return fig


def plot_final_comparison(report_df=None, save=True):
    """최종 비교 — 시나리오별 Rule vs RL 에이전트"""
    if report_df is None:
        path = RESULTS_DIR / "final_comparison.csv"
        if not path.exists():
            print("  [SKIP] No final comparison data")
            return
        report_df = pd.read_csv(path)

    scenarios = sorted(report_df["scenario"].unique())
    ems_types = sorted(report_df["ems"].unique())
    n_ems = len(ems_types)

    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    fig.suptitle('FCHEV EMS 최종 비교: Rule vs RL Agents',
                 fontsize=16, fontweight='bold')

    metrics = [
        ('distance_km', 'Distance (km)', '{:.1f}'),
        ('km_per_kg', 'Efficiency (km/kg H₂)', '{:.1f}'),
        ('fc_per_100km', 'H₂ Consumption (kg/100km)', '{:.3f}'),
        ('h2_kg', 'Total H₂ Used (kg)', '{:.3f}'),
        ('final_SOC', 'Final Battery SOC', '{:.3f}'),
        ('improvement_pct', 'Improvement vs Rule (%)', '{:+.1f}'),
    ]

    x = np.arange(len(scenarios))
    w = 0.8 / n_ems

    for midx, (key, title, fmt) in enumerate(metrics):
        ax = axes[midx // 3][midx % 3]

        for eidx, ems in enumerate(ems_types):
            rows = report_df[report_df["ems"] == ems]
            vals = []
            for sid in scenarios:
                sid_rows = rows[rows["scenario"] == sid]
                if not sid_rows.empty and key in sid_rows.columns:
                    v = sid_rows.iloc[0][key]
                    vals.append(v if pd.notna(v) else 0)
                else:
                    vals.append(0)

            algo = rows.iloc[0]["algo"] if not rows.empty else ems
            color = COLORS_ALGO.get(algo, '#999999')
            bars = ax.bar(x + eidx * w - (n_ems - 1) * w / 2,
                          vals, w, label=algo, color=color, alpha=0.85)

            for bar, val in zip(bars, vals):
                if val != 0:
                    ax.text(bar.get_x() + bar.get_width()/2,
                            bar.get_height(),
                            fmt.format(val), ha='center', va='bottom',
                            fontsize=6, rotation=45)

        ax.set_title(title)
        ax.set_xticks(x)
        ax.set_xticklabels(scenarios, fontsize=9)
        ax.legend(fontsize=7, loc='best')
        ax.grid(True, alpha=0.2, axis='y')

        if key == 'improvement_pct':
            ax.axhline(0, color='black', linewidth=0.8)
            ax.axhline(5, color='green', linewidth=1, linestyle='--', alpha=0.5)

    plt.tight_layout(rect=[0, 0, 1, 0.93])
    if save:
        fig.savefig(RESULTS_DIR / 'fig_final_comparison.png', bbox_inches='tight')
        print("  [saved] fig_final_comparison.png")
    return fig


def plot_improvement_summary(report_df=None, save=True):
    """시나리오별 개선율 요약"""
    if report_df is None:
        path = RESULTS_DIR / "final_comparison.csv"
        if not path.exists():
            return
        report_df = pd.read_csv(path)

    rl_rows = report_df[report_df["ems"] != "rule"].copy()
    if "improvement_pct" not in rl_rows.columns or rl_rows.empty:
        return

    fig, ax = plt.subplots(figsize=(14, 6))
    fig.suptitle('RL Agent Improvement vs Rule-based (%)',
                 fontsize=14, fontweight='bold')

    scenarios = sorted(rl_rows["scenario"].unique())
    algos = sorted(rl_rows["algo"].unique())
    x = np.arange(len(scenarios))
    w = 0.8 / max(len(algos), 1)

    for aidx, algo in enumerate(algos):
        algo_rows = rl_rows[rl_rows["algo"] == algo]
        vals = []
        for sid in scenarios:
            sid_rows = algo_rows[algo_rows["scenario"] == sid]
            if not sid_rows.empty:
                vals.append(sid_rows.iloc[0].get("improvement_pct", 0) or 0)
            else:
                vals.append(0)

        color = COLORS_ALGO.get(algo, '#999999')
        bars = ax.bar(x + aidx * w - (len(algos) - 1) * w / 2,
                      vals, w, label=algo, color=color, alpha=0.85)

        for bar, val in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width()/2,
                    bar.get_height() + (0.3 if val >= 0 else -1.5),
                    f'{val:+.1f}%', ha='center', va='bottom', fontsize=8)

    ax.axhline(0, color='black', linewidth=0.8)
    ax.axhline(5, color='green', linewidth=1, linestyle='--', alpha=0.5,
               label='Target +5%')
    ax.set_xticks(x)
    ax.set_xticklabels([f'Scenario {s}' for s in scenarios])
    ax.set_ylabel('Improvement (%)')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.2, axis='y')

    plt.tight_layout(rect=[0, 0, 1, 0.93])
    if save:
        fig.savefig(RESULTS_DIR / 'fig_improvement_summary.png', bbox_inches='tight')
        print("  [saved] fig_improvement_summary.png")
    return fig


def plot_timeseries_comparison(all_results=None, scenario_id="A", save=True):
    """시나리오별 시계열 비교 (Rule vs Best RL)"""
    if all_results is None:
        return

    scenario_results = [r for r in all_results if r.get("scenario") == scenario_id]
    if not scenario_results:
        return

    rule_res = [r for r in scenario_results if r.get("ems") == "rule"]
    rl_res = [r for r in scenario_results if r.get("ems") != "rule"]

    if not rule_res or not rl_res:
        return

    rule_data = rule_res[0]
    # 최고 RL 에이전트 선택
    best_rl = max(rl_res, key=lambda r: r.get("km_per_kg", 0))

    if not rule_data.get("log") or not best_rl.get("log"):
        return

    rule_df = pd.DataFrame(rule_data["log"])
    rl_df = pd.DataFrame(best_rl["log"])

    fig = plt.figure(figsize=(18, 14))
    fig.suptitle(f'Scenario {scenario_id}: Rule vs {best_rl.get("algo", "RL")} '
                 f'— Time Series',
                 fontsize=15, fontweight='bold')
    gs = gridspec.GridSpec(4, 2, hspace=0.35, wspace=0.25)

    plots = [
        (0, 0, 'P_fc_kW',     'FC Power Output (kW)'),
        (0, 1, 'P_bat_kW',    'Battery Power (kW)'),
        (1, 0, 'SOC_bat',     'Battery SOC'),
        (1, 1, 'M_H2_kg',     'Cumulative H₂ (kg)'),
        (2, 0, 'eta_fcs',     'FCS Efficiency'),
        (2, 1, 'distance_km', 'Distance (km)'),
        (3, 0, 'T_st_C',      'Stack Temperature (C)'),
        (3, 1, 'reward',       'Reward'),
    ]

    algo_name = best_rl.get("algo", "RL")
    colors = {
        'rule': '#2196F3',
        'rl': COLORS_ALGO.get(algo_name, '#FF5722'),
    }

    for r, c, col, title in plots:
        ax = fig.add_subplot(gs[r, c])
        if col in rule_df.columns:
            ax.plot(rule_df['t'] / 3600, rule_df[col],
                    label='Rule', color=colors['rule'],
                    alpha=0.7, linewidth=0.6)
        if col in rl_df.columns:
            ax.plot(rl_df['t'] / 3600, rl_df[col],
                    label=algo_name, color=colors['rl'],
                    alpha=0.7, linewidth=0.6)
        ax.set_title(title)
        ax.set_xlabel('Time (h)')
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

    fname = f'fig_ts_{scenario_id}_rule_vs_{algo_name}.png'
    if save:
        fig.savefig(RESULTS_DIR / fname, bbox_inches='tight')
        print(f"  [saved] {fname}")
    return fig


def plot_efficiency_distribution(all_results=None, save=True):
    """효율 분포 히스토그램"""
    if all_results is None:
        return

    results_with_logs = [r for r in all_results
                         if r.get("log") and len(r["log"]) > 100]
    if not results_with_logs:
        return

    scenarios = sorted(set(r["scenario"] for r in results_with_logs))
    n_scenarios = min(len(scenarios), 4)

    fig, axes = plt.subplots(1, n_scenarios, figsize=(5 * n_scenarios, 5))
    if n_scenarios == 1:
        axes = [axes]
    fig.suptitle('FCS Efficiency Distribution', fontsize=14, fontweight='bold')

    for idx, sid in enumerate(scenarios[:n_scenarios]):
        ax = axes[idx]
        sid_results = [r for r in results_with_logs if r["scenario"] == sid]
        for r in sid_results:
            df = pd.DataFrame(r["log"])
            eta = df["eta_fcs"].dropna()
            eta = eta[(eta > 0) & (eta < 1)]
            algo = r.get("algo", r.get("ems", "?"))
            color = COLORS_ALGO.get(algo, '#999999')
            ax.hist(eta, bins=60, alpha=0.5, label=algo,
                    color=color, density=True)
        ax.set_title(f'Scenario {sid}')
        ax.set_xlabel('FCS Efficiency')
        ax.set_ylabel('Density')
        ax.legend(fontsize=7)
        ax.grid(True, alpha=0.3)

    plt.tight_layout(rect=[0, 0, 1, 0.93])
    if save:
        fig.savefig(RESULTS_DIR / 'fig_efficiency_dist.png', bbox_inches='tight')
        print("  [saved] fig_efficiency_dist.png")
    return fig


def plot_operating_points(all_results=None, save=True):
    """운전점 분포 (P_fc vs 차속)"""
    if all_results is None:
        return

    results_with_logs = [r for r in all_results
                         if r.get("log") and len(r["log"]) > 100]
    if not results_with_logs:
        return

    # scenario A만 표시
    sid = "A"
    sid_results = [r for r in results_with_logs if r["scenario"] == sid]
    if not sid_results:
        return

    n = len(sid_results)
    fig, axes = plt.subplots(1, n, figsize=(5 * n, 4))
    if n == 1:
        axes = [axes]
    fig.suptitle(f'FC Operating Points — Scenario {sid}',
                 fontsize=14, fontweight='bold')

    for idx, r in enumerate(sid_results):
        ax = axes[idx]
        df = pd.DataFrame(r["log"])
        algo = r.get("algo", r.get("ems", "?"))
        ax.hexbin(df['v_kmh'], df['P_fc_kW'], gridsize=40,
                  cmap='YlOrRd', mincnt=1)
        ax.set_title(algo, fontsize=10)
        ax.set_xlabel('Speed (km/h)')
        ax.set_ylabel('P_fc (kW)')

    plt.tight_layout(rect=[0, 0, 1, 0.93])
    if save:
        fig.savefig(RESULTS_DIR / 'fig_operating_points.png', bbox_inches='tight')
        print("  [saved] fig_operating_points.png")
    return fig


def main(report_df=None, all_results=None):
    """전체 시각화 실행"""
    print("\n" + "=" * 60)
    print("  RL Tournament Visualization")
    print("=" * 60)

    os.makedirs(RESULTS_DIR, exist_ok=True)

    print("\n[1/5] Tournament ranking...")
    plot_tournament_ranking()

    print("\n[2/5] Final comparison...")
    plot_final_comparison(report_df)

    print("\n[3/5] Improvement summary...")
    plot_improvement_summary(report_df)

    print("\n[4/5] Time series comparison...")
    if all_results:
        scenarios = sorted(set(r["scenario"] for r in all_results))
        for sid in scenarios[:4]:
            plot_timeseries_comparison(all_results, sid)

    print("\n[5/5] Efficiency & operating points...")
    plot_efficiency_distribution(all_results)
    plot_operating_points(all_results)

    print(f"\n  All figures saved to {RESULTS_DIR}/")


if __name__ == '__main__':
    main()
