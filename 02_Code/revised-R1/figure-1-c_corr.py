#!/usr/bin/env python3
"""
Figure 1C — 各高度风速与全场总功率的 Spearman 相关廓线。

内容与上一轮的 `figure-1-c_fixed.py` 一致（掩膜、样本口径、配对 bootstrap
都没动），只是拆成独立一张、按最终印刷尺寸重排版面。

配对 bootstrap 回应 R2 Point 10（"is it statistically significant?"）：
问的是 r(10m) 与 r(70m) 那一点点差是否显著，所以必须对同一批行重采样、
同时算两个相关再取差。非配对会严重高估不确定性——两个高度的风速本身
就高度共变。检验结果写在 fig1_corr_bootstrap.csv，图上不标，留给正文/表。
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats

from fig1_common import (HEIGHTS, POWER_COL, N_BOOT, RNG_SEED, OUT_DIR,
                         COLORS, MARKERS, LABELS,
                         apply_style, load_observations, height_axis, save,
                         FS_LEGEND, FS_LABEL, AXES_LW, PanelCtx)
import os

PANEL_W, PANEL_H = 2.95, 3.30     # 本面板方框的英寸尺寸

# x 轴范围写死 0.80-0.90（间隔 0.02），不用自动缩放。
# 自动缩放会把 0.804-0.853 这 0.05 的跨度铺满整个面板宽度，
# 视觉上放大了扇区间那点差异；固定成 0.10 的跨度更保守，
# 代价是曲线只占面板左半边。
XLIM = (0.80, 0.90)
XTICK_STEP = 0.02
XLABEL_SIZE = FS_LABEL + 1        # 比其余面板的轴标题大一号（作者指定）
XLABEL_PAD = 6                    # 略微下移，与刻度拉开
CTX = PanelCtx()


def spearman_pair(x, y):
    return stats.spearmanr(x, y)[0]


def paired_bootstrap_diff(ws_a, ws_b, power, n_boot=N_BOOT, seed=RNG_SEED):
    """同一批行重采样 -> 同时算 r(a,P) 与 r(b,P) -> 差的分布。"""
    rng = np.random.default_rng(seed)
    n = len(power)
    d = np.empty(n_boot)
    for i in range(n_boot):
        idx = rng.integers(0, n, n)
        d[i] = spearman_pair(ws_a[idx], power[idx]) - spearman_pair(ws_b[idx], power[idx])
    return d


def draw(fig, x0=0.0, y0=0.0):
    CTX.bind(fig, x0, y0)
    dfv, m_all, m_free, m_wake = load_observations()

    power = dfv[POWER_COL].values
    WS = {h: dfv[f'obs_wind_speed_{h}m'].values for h in HEIGHTS}
    subsets = {'all': m_all, 'free': m_free, 'wake': m_wake}

    corr = {}
    print("\n  [Spearman 相关：风速 vs 总功率]")
    print(f"    {'高度':>5} {'all':>9} {'free':>9} {'wake':>9}")
    for tag, m in subsets.items():
        corr[tag] = np.array([spearman_pair(WS[h][m], power[m]) for h in HEIGHTS])
    for i, h in enumerate(HEIGHTS):
        print(f"    {h:>3}m {corr['all'][i]:>9.4f} "
              f"{corr['free'][i]:>9.4f} {corr['wake'][i]:>9.4f}")

    print("\n  [配对 bootstrap: r(10m,P) - r(70m,P)]  (R2 Point 10)")
    rows = []
    for tag, m in subsets.items():
        d = paired_bootstrap_diff(WS[10][m], WS[70][m], power[m])
        lo, hi = np.percentile(d, [2.5, 97.5])
        sig = not (lo <= 0 <= hi)
        print(f"    {tag:>5}  r10={corr[tag][0]:.4f}  r70={corr[tag][3]:.4f}  "
              f"diff={corr[tag][0]-corr[tag][3]:+.4f}  "
              f"CI[{lo:+.4f},{hi:+.4f}]  {'显著' if sig else '不显著'}")
        rows.append({'subset': tag, 'N': int(m.sum()),
                     'r_10m': corr[tag][0], 'r_70m': corr[tag][3],
                     'diff': corr[tag][0] - corr[tag][3],
                     'CI_lo': lo, 'CI_hi': hi, 'significant': sig})

    ax = CTX.ax(0.60, 0.90, 2.25, 2.28)

    for tag in ('all', 'free', 'wake'):
        ax.plot(corr[tag], HEIGHTS, '-', color=COLORS[tag], linewidth=1.6,
                marker=MARKERS[tag], markersize=4.6, markerfacecolor='white',
                markeredgecolor=COLORS[tag], markeredgewidth=1.2,
                label=LABELS[tag], clip_on=False, zorder=3)

    ax.set_xlabel('Spearman correlation with total power',
                  fontsize=XLABEL_SIZE, labelpad=XLABEL_PAD)
    ax.set_xlim(*XLIM)
    ax.set_xticks(np.arange(XLIM[0], XLIM[1] + 1e-9, XTICK_STEP))
    height_axis(ax)
    ax.grid(True, linestyle=':', linewidth=0.5, alpha=0.5, color='gray')
    ax.tick_params(direction='in', length=2.8, width=AXES_LW)

    # 图例放图框外，避免压住 10 m / 70 m 处的数据点
    ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.175), ncol=1,
              frameon=False, fontsize=FS_LEGEND, labelspacing=0.32,
              handlelength=1.8, handletextpad=0.55, borderaxespad=0)

    pd.DataFrame(rows).to_csv(os.path.join(OUT_DIR, 'fig1_corr_bootstrap.csv'),
                              index=False, float_format='%.5g')
    pd.DataFrame({'height_m': HEIGHTS, **{k: corr[k] for k in corr}}).to_csv(
        os.path.join(OUT_DIR, 'fig1_correlation_values.csv'),
        index=False, float_format='%.5g')
    return ax


def main():
    apply_style()
    fig = plt.figure(figsize=(PANEL_W, PANEL_H), facecolor='white')
    draw(fig)
    save(fig, 'figure-1c_correlation_profile')


if __name__ == '__main__':
    main()
