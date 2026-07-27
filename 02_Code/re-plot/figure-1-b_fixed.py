#!/usr/bin/env python3
"""
Figure 1 (风廓线) — 全样本廓线 + 自由流/尾流廓线对比（重写版）

相对原版的改动
--------------
1. 掩膜换 70m 单高度判据 + flatline 剔除。

2. 样本口径与 figure-1 相关廓线、figure-2 各面板统一（复用 imf_masks 的 dfv）。

3. 补一张扇区差异诊断表（含 bootstrap CI）。
   原因：原稿称"10-m 风速的分布在自由流与尾流之间几乎相同"，
   并据此推出"该层基本不受尾流影响、仍反映背景大气入流"。
   实测 10m 平均风速 6.30 -> 5.93，即 5.8% 亏损 —— 不是"几乎相同"。
   R2 Point 12 正是攻击这句（并引 Archer 的工作与卫星 SAR 观测：
   尾流会降低摩擦速度，表层风必然受影响）。
   诊断表给出各高度的亏损量与 CI，用来支撑改写后的说法：
   亏损随高度单调递增（10m 最小），即表层通道的混合比例最低 —— 
   而不是"表层未被影响"。
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

from imf_masks import build_aligned_frame, build_sector_masks

# ---------------------------------------------------------------- 配置
DATA_PATH = '/Users/xiaxin/work/WindForecast_Project/01_Data/processed/matched_data/changma_matched.csv'
OUT_DIR = '/Users/xiaxin/work/WindForecast_Project/03_Results/re-plot-figures/figure-1/'

HEIGHTS = [10, 30, 50, 70]
N_PLOT = 200          # 背景细线采样条数
N_BOOT = 2000
RNG_SEED = 42

plt.rcParams.update({
    'font.family': 'Arial', 'font.size': 11,
    'figure.dpi': 300, 'savefig.dpi': 300,
})

C = {
    'all':  {'light': '#686767', 'dark': '#2d2d2d'},
    'wake': {'light': '#6fa8dc', 'dark': '#1996de'},
    'free': {'light': '#ed6e6e', 'dark': '#f41111'},
}


def boot_mean_diff(a, b, n_boot=N_BOOT, seed=RNG_SEED):
    """两组各自重采样 -> 均值差的 CI。"""
    rng = np.random.default_rng(seed)
    d = np.empty(n_boot)
    for i in range(n_boot):
        d[i] = (a[rng.integers(0, len(a), len(a))].mean()
                - b[rng.integers(0, len(b), len(b))].mean())
    return np.percentile(d, [2.5, 97.5])


def main():
    print("=" * 74)
    print("Figure 1 — 风廓线（70m 单高度判据）")
    print("=" * 74)

    dfv = build_aligned_frame(DATA_PATH)
    m_free, m_wake = build_sector_masks(dfv)

    from qc_common import flatline_mask, WS_COLS
    bad = np.zeros(len(dfv), bool)
    for c in WS_COLS:
        if c in dfv.columns:
            bad |= flatline_mask(dfv[c].values)
    m_all = ~bad

    X = dfv[[f'obs_wind_speed_{h}m' for h in HEIGHTS]].values

    stats_ = {}
    for tag, m in [('all', m_all), ('free', m_free), ('wake', m_wake)]:
        stats_[tag] = {'X': X[m], 'mean': X[m].mean(axis=0),
                       'std': X[m].std(axis=0), 'n': int(m.sum())}

    print(f"\n  样本: all={stats_['all']['n']}, "
          f"free={stats_['free']['n']}, wake={stats_['wake']['n']}")

    # ---------------- 扇区差异诊断
    print("\n" + "=" * 74)
    print("扇区间风速差异（回应 R2 Point 12）")
    print("=" * 74)
    print(f"  {'高度':>5} {'free':>7} {'wake':>7} {'亏损':>7} {'亏损%':>7} "
          f"{'95% CI of 差':>20}")

    rows = []
    for i, h in enumerate(HEIGHTS):
        a = stats_['free']['X'][:, i]
        b = stats_['wake']['X'][:, i]
        diff = a.mean() - b.mean()
        pct = diff / a.mean() * 100
        lo, hi = boot_mean_diff(a, b)
        print(f"  {h:>3}m {a.mean():>7.2f} {b.mean():>7.2f} "
              f"{diff:>7.2f} {pct:>6.1f}% [{lo:>7.3f},{hi:>7.3f}]")
        rows.append({'height_m': h, 'mean_free': a.mean(), 'mean_wake': b.mean(),
                     'deficit': diff, 'deficit_pct': pct,
                     'CI_lo': lo, 'CI_hi': hi,
                     'significant': not (lo <= 0 <= hi),
                     'std_free': a.std(), 'std_wake': b.std()})

    os.makedirs(OUT_DIR, exist_ok=True)
    pd.DataFrame(rows).to_csv(os.path.join(OUT_DIR, 'fig1_sector_deficit.csv'),
                              index=False, float_format='%.5g')

    # 剪切比
    for tag in ('free', 'wake'):
        mu = stats_[tag]['mean']
        print(f"  剪切比 (70m/10m) {tag}: {mu[3]/mu[0]:.3f}")

    print("\n  解读：若亏损随高度单调递增（10m 最小、70m 最大），")
    print("        则支持'表层通道混合比例最低'的改写说法；")
    print("        原稿'10m 分布几乎相同 -> 未受尾流影响'不再成立。")

    plot(stats_, OUT_DIR)
    print("\n" + "=" * 74)
    print("完成")
    print("=" * 74)


def plot(S, out_dir):
    rng = np.random.default_rng(RNG_SEED)
    fig = plt.figure(figsize=(12, 5))
    ax1 = fig.add_axes([0.08, 0.15, 0.28, 0.75])
    ax2 = fig.add_axes([0.46, 0.15, 0.28, 0.75])

    def thin_lines(ax, tag, n, alpha, zorder):
        Xs = S[tag]['X']
        idx = rng.choice(len(Xs), size=min(n, len(Xs)), replace=False)
        for p in Xs[idx]:
            ax.plot(p, HEIGHTS, '-', color=C[tag]['light'],
                    alpha=alpha, linewidth=0.8, zorder=zorder)

    def mean_line(ax, tag, zorder, shade_alpha):
        ax.fill_betweenx(HEIGHTS, S[tag]['mean'] - S[tag]['std'],
                         S[tag]['mean'] + S[tag]['std'],
                         color=C[tag]['dark'], alpha=shade_alpha, zorder=zorder)
        ax.plot(S[tag]['mean'], HEIGHTS, 'o-', color=C[tag]['dark'],
                linewidth=3, markersize=8, markerfacecolor='white',
                markeredgecolor=C[tag]['dark'], markeredgewidth=1.5,
                zorder=zorder + 1)

    # --- 左：全样本
    thin_lines(ax1, 'all', N_PLOT, 0.08, 1)
    mean_line(ax1, 'all', 2, 0.15)
    ax1.set_xlabel(r'Wind Speed (m$\cdot$s$^{-1}$)', fontweight='bold', fontsize=16)
    ax1.set_ylabel('Height (m)', fontweight='bold', fontsize=16)
    ax1.set_title('All Data', fontweight='bold', pad=10, fontsize=18)
    ax1.set_xlim(0, 20)
    ax1.legend(handles=[Line2D([0], [0], color='black', linewidth=3, marker='o',
                               markersize=8, markerfacecolor='white',
                               markeredgecolor='black', markeredgewidth=1.5,
                               label='All Data')],
               loc='lower right', frameon=False, fontsize=14, markerfirst=False)

    # --- 右：两扇区对比
    thin_lines(ax2, 'free', 150, 0.08, 1)
    mean_line(ax2, 'free', 2, 0.12)
    thin_lines(ax2, 'wake', 150, 0.12, 3)
    mean_line(ax2, 'wake', 5, 0.15)
    ax2.set_xlabel(r'Wind Speed (m$\cdot$s$^{-1}$)', fontweight='bold', fontsize=16)
    ax2.set_ylabel('Height (m)', fontweight='bold', fontsize=16)
    ax2.set_title('Free-stream vs. Wake Regime', fontweight='bold',
                  pad=10, fontsize=18)
    ax2.set_xlim(3, 12)
    ax2.set_xticks(np.arange(3, 12.1, 1))
    ax2.legend(handles=[
        Line2D([0], [0], color=C['wake']['dark'], linewidth=3, marker='o',
               markersize=8, markerfacecolor='white',
               markeredgecolor=C['wake']['dark'], markeredgewidth=1.5,
               label='Wake (Easterly)'),
        Line2D([0], [0], color=C['free']['dark'], linewidth=3, marker='o',
               markersize=8, markerfacecolor='white',
               markeredgecolor=C['free']['dark'], markeredgewidth=1.5,
               label='Free-stream (Westerly)')],
        loc='lower right', frameon=False, fontsize=13,
        labelspacing=0.14, markerfirst=False)

    for ax in (ax1, ax2):
        ax.grid(True, linestyle='dotted', alpha=0.5, linewidth=0.8, color='gray')
        ax.set_ylim(0, max(HEIGHTS) + 1)
        ax.set_yticks(np.arange(0, max(HEIGHTS) + 1, 10))
        ax.set_yticklabels([str(h) if h in HEIGHTS else ''
                            for h in np.arange(0, max(HEIGHTS) + 1, 10)])
        ax.tick_params(axis='both', which='major', labelsize=18,
                       width=1.2, length=5, direction='in')

    base = os.path.join(out_dir, 'fig1_wind_profiles')
    for ext in ('.png', '.pdf'):
        plt.savefig(base + ext, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"\n  ✓ {base}.png")


if __name__ == '__main__':
    main()
