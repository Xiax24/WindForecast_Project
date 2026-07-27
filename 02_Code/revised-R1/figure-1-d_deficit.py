#!/usr/bin/env python3
"""
Figure 1D — 自由流 / 尾流两个扇区的平均风廓线，及两者之间的风速亏损。

由上一轮的两张图（全样本廓线 + 分扇区廓线）合并而来：
全样本那张整幅退出 Figure 1，腾出的位置改画亏损廓线（右侧窄轴）。

为什么要画亏损
--------------
原稿称"10 m 风速的分布在自由流与尾流之间几乎相同"，并据此推出
"该层基本不受尾流影响、仍反映背景大气入流"。实测 10 m 平均风速
6.30 -> 5.93 m/s，是 5.8% 的亏损，不是"几乎相同"。
R2 Point 12 正是攻击这句（并引 Archer 的工作与卫星 SAR 观测：
尾流会降低摩擦速度，表层风必然受影响）。
把亏损随高度单调递增（10 m 最小、70 m 最大）直接画出来，
支撑的是改写后的说法——表层通道的混合比例最低，而不是未被影响。

背景细廓线（原版每扇区 150 条）去掉了：在这个尺寸下只是噪声，
±1σ 阴影承载同样的离散度信息，而且是定量的。
"""

import os

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

from fig1_common import (HEIGHTS, N_BOOT, RNG_SEED, OUT_DIR, COLORS, LABELS,
                         apply_style, load_observations, height_axis, save,
                         FS_LEGEND, FS_ANNOT, AXES_LW, PanelCtx, align_xlabels)

PANEL_W, PANEL_H = 3.55, 2.90     # 本面板方框的英寸尺寸
LEGEND_Y = 0.34                   # 图例底边距面板底部（英寸）
DEFICIT_COLOR = '#7a7a7a'         # 亏损廓线用灰色
# 亏损才是这一面板的重点，把宽度从廓线那边匀过来（原来 1.78 : 0.62）。
# 右轴上限留到 28% 而不是 22%，是为了给每个点右侧的数值标注腾出位置。
PROF_W, DEF_W = 1.40, 1.05
DEF_XMAX = 28
CTX = PanelCtx()


def boot_mean_diff(a, b, n_boot=N_BOOT, seed=RNG_SEED):
    rng = np.random.default_rng(seed)
    d = np.empty(n_boot)
    for i in range(n_boot):
        d[i] = (a[rng.integers(0, len(a), len(a))].mean()
                - b[rng.integers(0, len(b), len(b))].mean())
    return np.percentile(d, [2.5, 97.5])


def draw(fig, x0=0.0, y0=0.0):
    CTX.bind(fig, x0, y0)
    dfv, _, m_free, m_wake = load_observations()

    X = dfv[[f'obs_wind_speed_{h}m' for h in HEIGHTS]].values
    S = {tag: {'X': X[m], 'mean': X[m].mean(axis=0), 'std': X[m].std(axis=0)}
         for tag, m in (('free', m_free), ('wake', m_wake))}

    rows = []
    for i, h in enumerate(HEIGHTS):
        a, b = S['free']['X'][:, i], S['wake']['X'][:, i]
        diff = a.mean() - b.mean()
        lo, hi = boot_mean_diff(a, b)
        rows.append({'height_m': h, 'mean_free': a.mean(), 'mean_wake': b.mean(),
                     'deficit': diff, 'deficit_pct': diff / a.mean() * 100,
                     'CI_lo': lo, 'CI_hi': hi,
                     'CI_lo_pct': lo / a.mean() * 100,
                     'CI_hi_pct': hi / a.mean() * 100,
                     'significant': not (lo <= 0 <= hi),
                     'std_free': a.std(), 'std_wake': b.std()})
    dfd = pd.DataFrame(rows)

    print("\n  [扇区风速亏损]")
    print(f"    {'高度':>5} {'free':>7} {'wake':>7} {'亏损':>7} {'亏损%':>7} "
          f"{'95% CI (m/s)':>20}")
    for r in rows:
        print(f"    {r['height_m']:>3}m {r['mean_free']:>7.2f} {r['mean_wake']:>7.2f} "
              f"{r['deficit']:>7.2f} {r['deficit_pct']:>6.1f}% "
              f"[{r['CI_lo']:>7.3f},{r['CI_hi']:>7.3f}]")
    for tag in ('free', 'wake'):
        mu = S[tag]['mean']
        print(f"    剪切比 (70m/10m) {tag}: {mu[3] / mu[0]:.3f}")

    # 左：分扇区平均廓线；右：亏损廓线。共用高度轴。
    ax = CTX.ax(0.58, 0.88, PROF_W, 2.00)
    axd = CTX.ax(0.58 + PROF_W + 0.16, 0.88, DEF_W, 2.00, sharey=ax)

    for tag in ('free', 'wake'):
        ax.fill_betweenx(HEIGHTS, S[tag]['mean'] - S[tag]['std'],
                         S[tag]['mean'] + S[tag]['std'],
                         color=COLORS[tag], alpha=0.13, linewidth=0, zorder=1)
        ax.plot(S[tag]['mean'], HEIGHTS, '-', color=COLORS[tag], linewidth=1.8,
                marker='o', markersize=4.6, markerfacecolor='white',
                markeredgecolor=COLORS[tag], markeredgewidth=1.2,
                clip_on=False, zorder=3)

    ax.set_xlabel(r'Wind speed (m$\cdot$s$^{-1}$)')
    height_axis(ax)
    ax.set_xlim(3, 12)
    ax.set_xticks(np.arange(4, 12.1, 2))
    ax.grid(True, linestyle=':', linewidth=0.5, alpha=0.5, color='gray')
    ax.tick_params(direction='in', length=2.8, width=AXES_LW)

    pct = dfd['deficit_pct'].values
    err = np.vstack([pct - dfd['CI_lo_pct'].values, dfd['CI_hi_pct'].values - pct])
    axd.errorbar(pct, HEIGHTS, xerr=err, fmt='-', color=DEFICIT_COLOR, linewidth=1.6,
                 marker='o', markersize=4.2, markerfacecolor='white',
                 markeredgecolor=DEFICIT_COLOR, markeredgewidth=1.2,
                 elinewidth=1.0, capsize=2.0, clip_on=False, zorder=3)
    axd.set_xlabel('Speed deficit (%)')
    axd.set_xlim(0, DEF_XMAX)
    axd.set_xticks([0, 10, 20])
    # 逐点标数值：正文第 65 段引的就是这四个数（5.8% -> 18.0%），
    # 图上直接给出，读者不必再去正文里翻。
    for h, v in zip(HEIGHTS, pct):
        axd.annotate(f'{v:.1f}%', xy=(v, h), xytext=(4, 0),
                     textcoords='offset points', ha='left', va='center',
                     fontsize=FS_ANNOT, color=DEFICIT_COLOR, zorder=6)
    axd.grid(True, linestyle=':', linewidth=0.5, alpha=0.5, color='gray')
    axd.tick_params(direction='in', length=2.8, width=AXES_LW)
    plt.setp(axd.get_yticklabels(), visible=False)

    # 两个 x 轴标签的基线对齐与高度，交给 align_xlabels 自动算，
    # 不再手写 y 值 —— 手调总在"顶到刻度"和"空一大片"之间来回。
    align_xlabels(fig, [ax, axd], gap_in=0.005)

    handles = [Line2D([], [], color=COLORS[t], linewidth=1.8, marker='o',
                      markersize=4.6, markerfacecolor='white',
                      markeredgecolor=COLORS[t], markeredgewidth=1.2,
                      label=LABELS[t]) for t in ('free', 'wake')]
    fig.legend(handles=handles, loc='lower left',
               bbox_to_anchor=CTX.pt(0.52, LEGEND_Y), ncol=2,
               frameon=False, fontsize=FS_LEGEND, handlelength=1.8,
               handletextpad=0.55, columnspacing=1.4, borderaxespad=0)

    dfd.to_csv(os.path.join(OUT_DIR, 'fig1_sector_deficit.csv'),
               index=False, float_format='%.5g')
    return ax


def main():
    apply_style()
    fig = plt.figure(figsize=(PANEL_W, PANEL_H), facecolor='white')
    draw(fig)
    save(fig, 'figure-1d_sector_profiles_deficit')


if __name__ == '__main__':
    main()
