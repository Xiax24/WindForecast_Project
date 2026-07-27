#!/usr/bin/env python3
"""
Figure 2A — Scale-Resolved Variance Distribution（重写版）

相对原版的改动
--------------
1. 掩膜换成 70m 单高度判据（原版从 npz 读的是废弃的四高度 strict 判据），
   并把 flatline 行从掩膜中剔除。CEEMDAN 不用重跑。

2. 10m 也按扇区拆开。原版灰线用的是 np.ones(...)，即全样本；而红蓝是
   70m 的两个扇区。这意味着"10m 与 70m 在两个扇区下都趋于一致"这句话
   在图上没有对应物 —— 拿全样本的 10m 去比扇区里的 70m，不是同一批数据。
   现在是 4 条：10m-free / 10m-wake / 70m-free / 70m-wake。
   颜色 = 扇区（红 free / 蓝 wake），线型 = 高度（虚 10m / 实 70m）。

3. 同时输出三个纵轴版本，不预设哪个对：
     mode='E'      : E = mean(imf^2)             方差贡献，单位 m2/s2
     mode='ET'     : E * T                       原版画的量（正比于 PSD）
     mode='E_norm' : E / U_mean^2                相对变率（各扇区各自的均值）
   并输出诊断表，让数据决定正文用哪个。

关于 ×T（重要，别弄反了）
-------------------------
* 红 vs 蓝：共用同一份 imfs_70m，第 i 个 IMF 的 T 是同一个数，
  V_free/V_wake = E_free/E_wake，T 完全约掉。
  ==> "同一周期上谁大谁小" 不受 ×T 影响，原版画法在这件事上没问题。
* 灰(10m) vs 红/蓝(70m)：T_10m 与 T_70m 是各自算的，不是同一个数，
  所以跨高度比较时 ×T 不干净地约掉。诊断表里两个 T 都列了，可直接看差多少。
* 跨周期（"方差集中在天气尺度"这类论断）：T 从 ~1h 到 ~1000h 跨 3 个量级，
  ×T 会系统性抬高长周期端。这类论断应当用 E 来支撑，不是 E*T。

R1 l.261-263 那处矛盾
---------------------
与 ×T 无关（见上，T 在红蓝之间约掉了）。若绝对方差在 2-12h 段确实
wake < free，成因大概率是两扇区平均风速不同 —— 这正是 mode='E_norm'
要检验的。诊断表里 wake/free 的风速比一并给出。
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

from imf_masks import load_and_prepare

# ---------------------------------------------------------------- 配置
DATA_PATH = '/Users/xiaxin/work/WindForecast_Project/01_Data/processed/matched_data/changma_matched.csv'
DATA_DIR = '/Users/xiaxin/work/WindForecast_Project/03_Results/re-plot-figures/figure-2/'
NPZ_PATH = os.path.join(DATA_DIR, 'ceemdan_results_full.npz')

# 要出的版本：'E' / 'ET' / 'E_norm'
MODES = ['E', 'ET', 'E_norm']

plt.rcParams.update({
    'font.family': 'Arial',
    'font.size': 11,
    'figure.dpi': 300,
    'savefig.dpi': 300,
})

COLORS = {'free': '#f41111', 'wake': '#1996de'}

MODE_CFG = {
    'E': {
        'ylabel': r'Variance Contribution (m$^2$$\cdot$s$^{-2}$)',
        'title': 'Scale-Resolved Variance Distribution',
        'suffix': 'E',
    },
    'ET': {
        'ylabel': r'$E\cdot T$ (m$^2$$\cdot$s$^{-2}$$\cdot$h)',
        'title': 'Scale-Resolved Spectral Density',
        'suffix': 'ET',
    },
    'E_norm': {
        'ylabel': r'Normalized Variance $E/\overline{U}^{\,2}$ (-)',
        'title': 'Scale-Resolved Normalized Variance',
        'suffix': 'Enorm',
    },
}


# ---------------------------------------------------------------- 计算
def get_imf_period(imf):
    crossings = np.where(np.diff(np.sign(imf)))[0]
    if len(crossings) < 2:
        return np.nan
    return len(imf) / (len(crossings) / 2.0)


def calculate_energy(imfs, mask, dt):
    """E = mean(imf^2[mask])，T 由过零点估计（与掩膜无关，用全序列）。"""
    n = len(imfs)
    E = np.zeros(n)
    T = np.zeros(n)
    for i in range(n):
        imf = np.asarray(imfs[i])
        T[i] = get_imf_period(imf) * dt
        E[i] = np.mean(imf[mask] ** 2) if mask.sum() > 0 else np.nan
    return E, T


def main():
    data, dfv, m_free, m_wake, U = load_and_prepare(
        DATA_PATH, NPZ_PATH, ref_var='ws_70m'
    )
    dt = float(data['dt_hours'])
    imfs_10m = data['imfs_ws_10m']
    imfs_70m = data['imfs_ws_70m']

    print("\n" + "=" * 70)
    print("计算各扇区能量")
    print("=" * 70)

    E_10f, T_10 = calculate_energy(imfs_10m, m_free, dt)
    E_10w, _ = calculate_energy(imfs_10m, m_wake, dt)
    E_70f, T_70 = calculate_energy(imfs_70m, m_free, dt)
    E_70w, _ = calculate_energy(imfs_70m, m_wake, dt)

    n = min(len(E_10f), len(E_70f))
    print(f"  公共 IMF 数: {n}")
    E_10f, E_10w, T_10 = E_10f[:n], E_10w[:n], T_10[:n]
    E_70f, E_70w, T_70 = E_70f[:n], E_70w[:n], T_70[:n]

    # 归一化用的均值平方（各扇区各高度用自己的均值）
    U10f2, U10w2 = U[10]['free'] ** 2, U[10]['wake'] ** 2
    U70f2, U70w2 = U[70]['free'] ** 2, U[70]['wake'] ** 2

    curves = {
        'E': {
            ('10m', 'free'): (T_10, E_10f), ('10m', 'wake'): (T_10, E_10w),
            ('70m', 'free'): (T_70, E_70f), ('70m', 'wake'): (T_70, E_70w),
        },
        'ET': {
            ('10m', 'free'): (T_10, E_10f * T_10), ('10m', 'wake'): (T_10, E_10w * T_10),
            ('70m', 'free'): (T_70, E_70f * T_70), ('70m', 'wake'): (T_70, E_70w * T_70),
        },
        'E_norm': {
            ('10m', 'free'): (T_10, E_10f / U10f2), ('10m', 'wake'): (T_10, E_10w / U10w2),
            ('70m', 'free'): (T_70, E_70f / U70f2), ('70m', 'wake'): (T_70, E_70w / U70w2),
        },
    }

    # ------------------------------------------------------- 诊断表
    diag = pd.DataFrame({
        'imf': np.arange(1, n + 1),
        'T_10m_h': T_10,
        'T_70m_h': T_70,
        'T_ratio_70over10': T_70 / T_10,
        'E_10m_free': E_10f,
        'E_10m_wake': E_10w,
        'E_70m_free': E_70f,
        'E_70m_wake': E_70w,
        # 同周期扇区比：>1 表示 wake 更大
        'E70_wake_over_free': E_70w / E_70f,
        'E10_wake_over_free': E_10w / E_10f,
        # 归一化后的同周期扇区比
        'Enorm70_wake_over_free': (E_70w / U70w2) / (E_70f / U70f2),
        'Enorm10_wake_over_free': (E_10w / U10w2) / (E_10f / U10f2),
        # 跨高度：10m 相对 70m
        'E10_over_E70_free': E_10f / E_70f,
        'E10_over_E70_wake': E_10w / E_70w,
    })

    out_csv = os.path.join(DATA_DIR, 'fig2a_diagnostics.csv')
    diag.to_csv(out_csv, index=False, float_format='%.6g')

    print("\n" + "=" * 70)
    print("诊断表（同周期扇区比：>1 = wake 更大）")
    print("=" * 70)
    with pd.option_context('display.width', 200, 'display.max_columns', 50):
        print(diag[['imf', 'T_70m_h', 'E_70m_free', 'E_70m_wake',
                    'E70_wake_over_free', 'Enorm70_wake_over_free']].to_string(index=False))
    print(f"\n  ✓ 完整诊断表: {out_csv}")

    # R1 l.261-263 的直接检验：2-12h 段 wake 是否高于 free
    band = (T_70 >= 2) & (T_70 <= 12)
    if band.any():
        print("\n" + "-" * 70)
        print("R1 l.261-263 检验（2-12h 段，70m）")
        print("-" * 70)
        print(f"  绝对方差   wake/free 均值: {diag['E70_wake_over_free'][band].mean():.3f}  "
              f"({'wake 更高 → 与正文一致' if diag['E70_wake_over_free'][band].mean() > 1 else 'wake 更低 → 与正文矛盾'})")
        print(f"  归一化后   wake/free 均值: {diag['Enorm70_wake_over_free'][band].mean():.3f}  "
              f"({'wake 更高 → 归一化解决了矛盾' if diag['Enorm70_wake_over_free'][band].mean() > 1 else 'wake 仍更低 → 矛盾不是风速差造成的，正文需改'})")
        print(f"  风速比 wake/free @70m: {U[70]['wake']/U[70]['free']:.3f}")

    # 长周期段 10m 与 70m 的接近程度（正文 l.268 那句）
    longm = T_70 >= 24
    if longm.any():
        print("\n" + "-" * 70)
        print("正文 l.268 检验（≥24h，10m 与 70m 是否趋于一致）")
        print("-" * 70)
        print(f"  free 扇区 |E10/E70 - 1| 均值: {np.abs(diag['E10_over_E70_free'][longm] - 1).mean():.3f}")
        print(f"  wake 扇区 |E10/E70 - 1| 均值: {np.abs(diag['E10_over_E70_wake'][longm] - 1).mean():.3f}")

    # ------------------------------------------------------- 绘图
    for mode in MODES:
        plot_one(curves[mode], MODE_CFG[mode], DATA_DIR)

    print("\n" + "=" * 70)
    print("完成")
    print("=" * 70)


def plot_one(curve_dict, cfg, out_dir):
    fig, ax = plt.subplots(figsize=(10, 8))

    for t, label in {1: '1h', 6: '6h', 12: '12h', 24: '1d', 168: '1w'}.items():
        ax.axvline(t, color="#6C6C6E", linestyle='--', alpha=0.6, linewidth=1.5, zorder=1)
        ax.text(t * 1.3, 0.93, label, ha='center', va='bottom', fontsize=28,
                color='gray', alpha=0.8, transform=ax.get_xaxis_transform())

    style = {
        ('70m', 'free'): dict(color=COLORS['free'], ls='-', marker='o', mfc=COLORS['free'],
                              label='70 m, Free-stream (W)', zorder=5),
        ('70m', 'wake'): dict(color=COLORS['wake'], ls='-', marker='o', mfc=COLORS['wake'],
                              label='70 m, Wake (E)', zorder=4),
        ('10m', 'free'): dict(color=COLORS['free'], ls='--', marker='s', mfc='white',
                              label='10 m, Free-stream (W)', zorder=3),
        ('10m', 'wake'): dict(color=COLORS['wake'], ls='--', marker='s', mfc='white',
                              label='10 m, Wake (E)', zorder=2),
    }

    for key in [('70m', 'free'), ('70m', 'wake'), ('10m', 'free'), ('10m', 'wake')]:
        T, V = curve_dict[key]
        s = style[key]
        ax.plot(T, V, color=s['color'], linestyle=s['ls'], linewidth=1.2,
                marker=s['marker'], markersize=9, markerfacecolor=s['mfc'],
                markeredgecolor=s['color'], markeredgewidth=1.8,
                label=s['label'], zorder=s['zorder'])

    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xlabel(r'Period $T$ (hours)', fontsize=28)
    ax.set_ylabel(cfg['ylabel'], fontsize=24)
    ax.set_title(cfg['title'], fontsize=28, fontweight='bold', pad=15)
    ax.legend(loc='lower right', frameon=False, fontsize=18, markerfirst=False)
    ax.grid(True, which='both', alpha=0.3, linestyle=':', linewidth=0.5)
    ax.tick_params(axis='both', which='major', labelsize=28, width=1.2,
                   length=5, direction='in', pad=10)
    ax.tick_params(axis='both', which='minor', width=0.8, length=3, direction='in')
    for sp in ax.spines.values():
        sp.set_linewidth(1.2)

    ylim, xlim = ax.get_ylim(), ax.get_xlim()
    w0 = max(1, xlim[0])
    if 24 - w0 > 0:
        ax.add_patch(Rectangle((w0, ylim[0]), 24 - w0, ylim[1] - ylim[0],
                               linewidth=0, facecolor=COLORS['wake'], alpha=0.08, zorder=0))
    if xlim[1] - 24 > 0:
        ax.add_patch(Rectangle((24, ylim[0]), xlim[1] - 24, ylim[1] - ylim[0],
                               linewidth=0, facecolor=COLORS['free'], alpha=0.08, zorder=0))
    ax.set_ylim(ylim)
    ax.set_xlim(xlim)

    plt.tight_layout()
    base = os.path.join(out_dir, f"fig2a_variance_{cfg['suffix']}")
    plt.savefig(base + '.pdf', bbox_inches='tight', facecolor='white')
    plt.savefig(base + '.png', dpi=300, bbox_inches='tight', facecolor='white')
    print(f"  ✓ {base}.png")
    plt.close()


if __name__ == '__main__':
    main()
