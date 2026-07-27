#!/usr/bin/env python3
"""
Figure 2B — 各 IMF 的尺度分解相关系数（重写版）

相对原版的改动
--------------
1. 掩膜换成 70m 单高度判据 + flatline 剔除（原版从 npz 读的是废弃的
   四高度 strict 判据）。CEEMDAN 不用重跑。

2. 补齐 30m/50m 与 power 的分扇区相关。原版只算了 all-data 版本，
   分扇区只做了 10m 和 70m —— 但跨高度的"混合比例随高度变化"这条诊断
   需要四个高度都有分扇区的值。

3. 输出诊断表 CSV，含各扇区样本数。

术语提醒（R1 l.269-270）
------------------------
R1 明确指出 "coherence" 一词是留给谱计算的（spectral coherence），
这里算的是各 IMF 上的 Spearman 秩相关，不是相干谱。
正文/图注里不要再叫 coherence —— 建议用 "scale-resolved correlation"
或 "cross-scale correlation"。本脚本的变量名与输出一律用 correlation。
"""

import os
import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from imf_masks import load_and_prepare

# ---------------------------------------------------------------- 配置
DATA_PATH = '/Users/xiaxin/work/WindForecast_Project/01_Data/processed/matched_data/changma_matched.csv'
DATA_DIR = '/Users/xiaxin/work/WindForecast_Project/03_Results/re-plot-figures/figure-2/'
NPZ_PATH = os.path.join(DATA_DIR, 'ceemdan_results_full.npz')
OUT_NPZ = os.path.join(DATA_DIR, 'correlations_all.npz')
OUT_CSV = os.path.join(DATA_DIR, 'fig2b_diagnostics.csv')

MIN_N = 30   # 掩膜内样本少于此值则记 NaN（原版是 10，偏松）


def get_imf_period(imf):
    crossings = np.where(np.diff(np.sign(imf)))[0]
    if len(crossings) < 2:
        return np.nan
    return len(imf) / (len(crossings) / 2.0)


def calculate_correlation(imfs_a, imfs_b, mask, dt):
    """各 IMF 上的 Spearman 秩相关。mask=None 表示全样本。"""
    n = min(len(imfs_a), len(imfs_b))
    R = np.zeros(n)
    T = np.zeros(n)
    for i in range(n):
        a = np.asarray(imfs_a[i])
        b = np.asarray(imfs_b[i])
        T[i] = get_imf_period(a) * dt

        if mask is None:
            sa, sb = a, b
        else:
            sa, sb = a[mask], b[mask]

        if len(sa) < MIN_N:
            R[i] = np.nan
            continue
        try:
            R[i], _ = spearmanr(sa, sb)
        except Exception:
            R[i] = np.nan
    return R, T


def main():
    data, dfv, m_free, m_wake, U = load_and_prepare(
        DATA_PATH, NPZ_PATH, ref_var='ws_70m'
    )
    dt = float(data['dt_hours'])

    imfs = {}
    for h in (10, 30, 50, 70):
        k = f'imfs_ws_{h}m'
        if k in data:
            imfs[f'{h}m'] = data[k]
    has_power = 'imfs_power' in data
    if has_power:
        imfs['power'] = data['imfs_power']

    print("\n" + "=" * 70)
    print("计算尺度分解相关系数")
    print("=" * 70)
    print(f"  可用变量: {', '.join(imfs.keys())}")
    print(f"  样本数: free={m_free.sum()}, wake={m_wake.sum()}, all={len(dfv)}")

    masks = {'all': None, 'free': m_free, 'wake': m_wake}
    corr = {}

    # 高度对之间（垂直一致性）
    for a, b in [('10m', '70m'), ('30m', '70m'), ('50m', '70m'), ('10m', '30m')]:
        if a not in imfs or b not in imfs:
            continue
        for tag, mk in masks.items():
            R, T = calculate_correlation(imfs[a], imfs[b], mk, dt)
            corr[f'{a}-{b}-{tag}'] = (R, T)
            print(f"    ✓ {a}-{b} ({tag})")

    # 各高度与功率
    if has_power:
        for h in ('10m', '30m', '50m', '70m'):
            if h not in imfs:
                continue
            for tag, mk in masks.items():
                R, T = calculate_correlation(imfs[h], imfs['power'], mk, dt)
                corr[f'{h}-power-{tag}'] = (R, T)
                print(f"    ✓ {h}-power ({tag})")

    # ------------------------------------------------------- 保存
    n = min(len(v[0]) for v in corr.values())
    save = {
        'dt_hours': dt,
        'n_imfs': n,
        'correlation_names': list(corr.keys()),
        'n_free': int(m_free.sum()),
        'n_wake': int(m_wake.sum()),
        'n_all': int(len(dfv)),
        'sector_criterion': 'obs_wind_direction_70m single-height',
    }
    for k, (R, T) in corr.items():
        save[f'{k}_R'] = R[:n]
        save[f'{k}_T'] = T[:n]
    np.savez_compressed(OUT_NPZ, **save)
    print(f"\n  ✓ {OUT_NPZ}")

    # 诊断表
    ref_T = corr[list(corr.keys())[0]][1][:n]
    tab = {'imf': np.arange(1, n + 1), 'T_h': ref_T}
    for k, (R, _) in corr.items():
        tab[k] = R[:n]
    df = pd.DataFrame(tab)
    df.to_csv(OUT_CSV, index=False, float_format='%.4f')
    print(f"  ✓ {OUT_CSV}")

    # 关键对照：跨高度的扇区间落差随高度怎么变
    print("\n" + "-" * 70)
    print("跨高度诊断：与 power 的相关，free 与 wake 的落差随高度")
    print("-" * 70)
    if has_power:
        longm = ref_T >= 24
        print(f"  {'高度':>6} {'free':>8} {'wake':>8} {'落差':>8}   (T>=24h 平均)")
        for h in ('10m', '30m', '50m', '70m'):
            kf, kw = f'{h}-power-free', f'{h}-power-wake'
            if kf in corr and kw in corr:
                rf = np.nanmean(corr[kf][0][:n][longm])
                rw = np.nanmean(corr[kw][0][:n][longm])
                print(f"  {h:>6} {rf:>8.3f} {rw:>8.3f} {rf - rw:>8.3f}")
        print("\n  解读：若'落差'随高度单调增大（10m 最小、70m 最大），")
        print("        即为'表层通道混合比例最低'的直接证据 —— 这是 4.1 新叙述")
        print("        的落点，也同时回应 R2 Point 12 与 R1 的上下补充相对主导。")

    print("\n" + "=" * 70)
    print("完成")
    print("=" * 70)


if __name__ == '__main__':
    main()
