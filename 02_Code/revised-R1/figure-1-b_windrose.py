#!/usr/bin/env python3
"""
Figure 1B — 70 m 轮毂高度风玫瑰。

相对原 `figure-1-a.py` 的修正
-----------------------------
1. 样本口径改成与 C、D 一致（IMF 对齐样本 + flatline 剔除，N = 47,448）。
   原来走的是 qc_power_only(原始 CSV)，与其余面板不是同一批行，
   同一张图里三个面板报不同的样本数说不过去。
2. 径向上限由实际最大扇区频率决定，不写死。
   E 扇区达 25.4%，任何 <26% 的固定上限都会把最长的花瓣直接截断。
3. windrose 包默认把方位刻度标成 "N-E"/"S-W"，改回气象惯例的 NE/SW。
4. 图例标签从 "[0.0 : 4.0)" 改成 "0-4" 的常规区间写法，并移到图框外；
   顶档并成 ">16"（原来按 p99.5 自动切会多出两个占比 0.7% 的隐形色块）。
5. 按最终印刷尺寸出图。
"""

import numpy as np
import matplotlib.pyplot as plt
from windrose import WindroseAxes   # noqa: F401  仅为注册 'windrose' projection

from fig1_common import (apply_style, load_observations, save,
                         FS_TICK, FS_LEGEND, FS_LABEL,
                         AXES_LW, AXES_COLOR, PanelCtx)

PANEL_W, PANEL_H = 3.40, 3.62     # 本面板方框的英寸尺寸
CTX = PanelCtx()

BIN_WIDTH = 4          # m/s
TOP_EDGE = 16          # 最高一档的下边界，其上并成 ">16"
NSECTOR = 16
PALETTE = ['#e0f3f8', '#abd9e9', '#74add1', '#4575b4', '#1a3d8f']


def draw(fig, x0=0.0, y0=0.0):
    CTX.bind(fig, x0, y0)
    dfv, m_all, _, _ = load_observations()

    ws = dfv['obs_wind_speed_70m'].values[m_all]
    wd = dfv['obs_wind_direction_70m'].values[m_all]
    ok = (np.isfinite(ws) & np.isfinite(wd)
          & (ws >= 0) & (ws < 50) & (wd >= 0) & (wd < 360))
    ws, wd = ws[ok], wd[ok]
    print(f"\n  [windrose] N = {len(ws):,} | 平均风速 {ws.mean():.2f} m/s")

    # 顶档定在 16 m/s，不按分位点自动延伸。
    # 按 p99.5 自动切会给出 0-4-8-12-16-20 六档，而 16-20 只占 0.66%、
    # >20 只占 0.048%（28 个样本）—— 六个图例色块里有两个在图上根本看不见，
    # 且那两档在印刷尺寸下都是分不开的深藏青。并成 ">16"（0.72%）后
    # 五档全部可见。
    bins = list(range(0, TOP_EDGE + 1, BIN_WIDTH))
    colors = PALETTE[:len(bins)]
    for lo, hi in [(0, 4), (4, 8), (8, 12), (12, 16), (16, 99)]:
        n = int(((ws >= lo) & (ws < hi)).sum())
        print(f"  [bin] {lo:>2}-{hi if hi < 99 else '  ':<3} {n:>7,}  {100 * n / len(ws):>6.2f}%")

    ax = CTX.ax(0.30, 0.62, 2.80, 2.80, projection='windrose')
    ax.bar(wd, ws, normed=True, opening=0.9, edgecolor='black',
           linewidth=0.35, bins=bins, colors=colors, nsector=NSECTOR)

    ax.set_xticklabels(['E', 'NE', 'N', 'NW', 'W', 'SW', 'S', 'SE'])

    # 径向上限必须由数据决定：写死会截断花瓣
    freq = ax._info['table'].sum(axis=0)
    rmax = float(freq.max())
    # 只留 3% 余量。向上取整到 5 的倍数会把 25.4% 的花瓣顶到 30% 的圈里，
    # 白白空掉外圈四分之一的面积。
    top = rmax * 1.03
    ax.set_ylim(0, top)
    ticks = np.arange(5, np.floor(top / 5) * 5 + 0.1, 5)
    ax.set_yticks(ticks)
    ax.set_yticklabels([f'{int(t)}%' for t in ticks], fontsize=FS_TICK)
    ax.tick_params(axis='x', labelsize=FS_TICK, pad=1)
    ax.grid(True, linewidth=0.5, alpha=0.4)
    for sp in ax.spines.values():
        sp.set_linewidth(AXES_LW)
        sp.set_edgecolor(AXES_COLOR)

    names = ['N', 'NNE', 'NE', 'ENE', 'E', 'ESE', 'SE', 'SSE',
             'S', 'SSW', 'SW', 'WSW', 'W', 'WNW', 'NW', 'NNW']
    order = np.argsort(freq)[::-1][:4]
    print("  [windrose] 频率最高的四个扇区: "
          + ', '.join(f'{names[i]} {freq[i]:.1f}%' for i in order))
    print(f"  [windrose] 径向上限 {top:.0f}%")

    labels = [f'{bins[i]}–{bins[i + 1]}' if i + 1 < len(bins) else f'>{bins[i]}'
              for i in range(len(bins))]
    lg = ax.legend(title=r'Wind speed (m$\cdot$s$^{-1}$)',
                   loc='upper center', bbox_to_anchor=(0.5, -0.045),
                   ncol=len(labels), frameon=False, fontsize=FS_LEGEND,
                   handlelength=1.0, handleheight=1.0, handletextpad=0.4,
                   columnspacing=0.7, labelspacing=0.3)
    for txt, lab in zip(lg.get_texts(), labels):
        txt.set_text(lab)
    lg.get_title().set_fontsize(FS_LABEL)

    return ax


def main():
    apply_style()
    fig = plt.figure(figsize=(PANEL_W, PANEL_H), facecolor='white')
    draw(fig)
    save(fig, 'figure-1b_windrose_70m')


if __name__ == '__main__':
    main()
