#!/usr/bin/env python3
"""
Figure 1 (相关廓线) — 各高度风速与总功率的 Spearman 相关（重写版）

相对原版的改动
--------------
1. 掩膜换 70m 单高度判据 + flatline 剔除（原版是废弃的四高度 strict 判据）。

2. 样本口径：复用 imf_masks 的 dfv（四高度风速 + power 均有效），
   与 figure-2 各面板共用同一批样本。
   这一点对本图尤其关键：r(10m,P) 与 r(70m,P) 必须在同一批行上计算，
   否则 0.81 vs 0.79 连 like-for-like 都不是。

3. **补配对 bootstrap** —— 这是必须的，不是可选。
   R2 Point 10 原话："It's an interesting result, but is it statistically
   significant?" 问的就是 r(10m) 与 r(70m) 那 0.02 的差。
   原版只给每个相关各自的 p 值，那检验的是"相关是否 != 0"
   （两万样本下当然是），答非所问。
   要回答的是"两个相关是否不同" -> 对同一批行重采样，
   同时算两个相关、取差，看差的 CI 含不含 0。
   配对是关键：两个高度的风速高度共变（r ~ 0.87），
   非配对 bootstrap 会严重高估差值的不确定性。

4. 不施加 3-25 m/s 风速筛选。该筛选卡在 70m 上，会直接截断 70m 的量程、
   只间接截断 10m，属于 range restriction，会系统性压低 70m 的相关、
   抬高正在检验的 10m-vs-70m 对比。本图是高度间的公平比较，故不设。
   （重建/预报实验里设 3-25 是合理的：那里定义的是机组运行包线，
     且对 HH/SR/ER 一视同仁。）
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats

from imf_masks import build_aligned_frame, build_sector_masks

# ---------------------------------------------------------------- 配置
DATA_PATH = '/Users/xiaxin/work/WindForecast_Project/01_Data/processed/matched_data/changma_matched.csv'
OUT_DIR = '/Users/xiaxin/work/WindForecast_Project/03_Results/re-plot-figures/figure-1/'

HEIGHTS = [10, 30, 50, 70]
POWER_COL = 'power'
N_BOOT = 2000
RNG_SEED = 42

plt.rcParams.update({
    'font.family': 'Arial', 'font.size': 11,
    'figure.dpi': 300, 'savefig.dpi': 300,
})

COLORS = {'all': '#2d2d2d', 'free': '#f41111', 'wake': '#1996de'}


def spearman_pair(x, y):
    return stats.spearmanr(x, y)[0]


def paired_bootstrap_diff(ws_a, ws_b, power, n_boot=N_BOOT, seed=RNG_SEED):
    """同一批行重采样 -> 同时算 r(a,P) 与 r(b,P) -> 差的分布。"""
    rng = np.random.default_rng(seed)
    n = len(power)
    diffs = np.empty(n_boot)
    for i in range(n_boot):
        idx = rng.integers(0, n, n)
        ra = spearman_pair(ws_a[idx], power[idx])
        rb = spearman_pair(ws_b[idx], power[idx])
        diffs[i] = ra - rb
    return diffs


def main():
    print("=" * 74)
    print("Figure 1 — 相关廓线（70m 单高度判据）")
    print("=" * 74)

    dfv = build_aligned_frame(DATA_PATH)
    m_free, m_wake = build_sector_masks(dfv)

    # flatline 也要从 all 里剔，保持与扇区一致
    from qc_common import flatline_mask, WS_COLS
    bad = np.zeros(len(dfv), bool)
    for c in WS_COLS:
        if c in dfv.columns:
            bad |= flatline_mask(dfv[c].values)
    m_all = ~bad

    power = dfv[POWER_COL].values
    WS = {h: dfv[f'obs_wind_speed_{h}m'].values for h in HEIGHTS}

    subsets = {'all': m_all, 'free': m_free, 'wake': m_wake}
    print(f"\n  样本: all={m_all.sum()}, free={m_free.sum()}, wake={m_wake.sum()}")

    # ---------------- 相关系数
    corr = {}
    print("\n" + "=" * 74)
    print("Spearman 相关（风速 vs 总功率）")
    print("=" * 74)
    print(f"  {'高度':>5} {'all':>9} {'free':>9} {'wake':>9}")
    for h in HEIGHTS:
        row = []
        for tag, m in subsets.items():
            r = spearman_pair(WS[h][m], power[m])
            corr.setdefault(tag, []).append(r)
            row.append(r)
        print(f"  {h:>3}m {row[0]:>9.4f} {row[1]:>9.4f} {row[2]:>9.4f}")

    for tag in subsets:
        corr[tag] = np.array(corr[tag])

    # ---------------- 配对 bootstrap: 10m vs 70m
    print("\n" + "=" * 74)
    print("配对 bootstrap: r(10m,P) - r(70m,P)   [回应 R2 Point 10]")
    print("=" * 74)
    print(f"  {'子集':>6} {'r_10m':>8} {'r_70m':>8} {'差':>8} "
          f"{'95% CI of 差':>20}  判定")

    rows = []
    for tag, m in subsets.items():
        r10 = spearman_pair(WS[10][m], power[m])
        r70 = spearman_pair(WS[70][m], power[m])
        d = paired_bootstrap_diff(WS[10][m], WS[70][m], power[m])
        lo, hi = np.percentile(d, [2.5, 97.5])
        sig = "显著 (CI不含0)" if not (lo <= 0 <= hi) else "不显著 (CI含0)"
        print(f"  {tag:>6} {r10:>8.4f} {r70:>8.4f} {r10-r70:>8.4f} "
              f"[{lo:>7.4f},{hi:>7.4f}]  {sig}")
        rows.append({'subset': tag, 'N': int(m.sum()),
                     'r_10m': r10, 'r_70m': r70, 'diff': r10 - r70,
                     'CI_lo': lo, 'CI_hi': hi,
                     'significant': not (lo <= 0 <= hi)})

    pd.DataFrame(rows).to_csv(
        os.path.join(OUT_DIR, 'fig1_corr_bootstrap.csv'),
        index=False, float_format='%.5g')

    print("\n  解读：")
    print("    all 显著为正   -> 4.1 开场'10m 相关高于 70m'成立，且现在有检验支撑")
    print("    all 不显著     -> 那句必须降级为'两者相当'，4.1 开场要重写")
    print("    wake 显著为正  -> 这是 ER/WDA 的核心依据，最重要的一行")

    # 全高度相关矩阵（Figure S1 用）
    print("\n" + "-" * 74)
    print("各高度两两相关 + 与功率（Figure S1 数据）")
    print("-" * 74)
    for tag, m in subsets.items():
        cols = [f'obs_wind_speed_{h}m' for h in HEIGHTS] + [POWER_COL]
        cm = dfv.loc[m, cols].corr(method='spearman')
        cm.to_csv(os.path.join(OUT_DIR, f'fig1_corrmatrix_{tag}.csv'),
                  float_format='%.4f')
        print(f"  [{tag}] -> fig1_corrmatrix_{tag}.csv")

    plot_profile(corr, OUT_DIR)
    print("\n" + "=" * 74)
    print("完成")
    print("=" * 74)


def plot_profile(corr, out_dir):
    fig, ax = plt.subplots(figsize=(4, 5))

    for tag, label in [('all', 'All Data'),
                       ('wake', 'Wake (Easterly)'),
                       ('free', 'Free-stream (Westerly)')]:
        c = corr[tag]
        ax.plot(c, HEIGHTS, '-', color=COLORS[tag], linewidth=3, zorder=2)
        imax = int(np.nanargmax(c))
        for i, (v, h) in enumerate(zip(c, HEIGHTS)):
            ax.plot(v, h, 'o', color=COLORS[tag], markersize=8,
                    markerfacecolor=COLORS[tag] if i == imax else 'white',
                    markeredgecolor=COLORS[tag], markeredgewidth=1.5, zorder=3)
        ax.plot([], [], 'o-', color=COLORS[tag], linewidth=3, markersize=8,
                markerfacecolor='white', markeredgecolor=COLORS[tag],
                markeredgewidth=1.5, label=label)

    ax.set_ylabel('Height (m)', fontweight='bold', fontsize=18)
    ax.set_title('Spearman Correlation', fontweight='bold', pad=10, fontsize=19)
    ax.set_ylim(0, max(HEIGHTS) + 18)
    ax.set_yticks(np.arange(0, max(HEIGHTS) + 19, 10))
    ax.set_yticklabels([str(h) if h in HEIGHTS else ''
                        for h in np.arange(0, max(HEIGHTS) + 18, 10)])

    # 自适应 x 范围（原版写死 0.70 下限，新判据下可能裁掉数据）
    allc = np.concatenate([corr[t] for t in ('all', 'free', 'wake')])
    lo, hi = np.nanmin(allc), np.nanmax(allc)
    pad = max(0.01, (hi - lo) * 0.15)
    ax.set_xlim(lo - pad, hi + pad)

    ax.grid(True, linestyle='dotted', alpha=0.5, linewidth=0.8, color='gray')
    ax.tick_params(axis='both', which='major', labelsize=18, width=1.2,
                   length=5, direction='in')
    ax.legend(loc='upper right', frameon=False, fontsize=11, markerfirst=False)

    plt.tight_layout()
    os.makedirs(out_dir, exist_ok=True)
    base = os.path.join(out_dir, 'fig1_correlation_profile')
    for ext in ('.png', '.pdf'):
        plt.savefig(base + ext, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"\n  ✓ {base}.png")


if __name__ == '__main__':
    main()
