#!/usr/bin/env python3
"""
imf_masks.py — 重建与 CEEMDAN IMF 逐行对齐的掩膜（70m 单高度判据）

为什么需要这个模块
------------------
step1 保存的 ceemdan_results_full.npz 里烘死了 mask_west / mask_east，
它们来自已废弃的 strict_direction_mask（要求 10/30/50/70 四个高度同时落在
扇区内）。该判据被仪器缺陷主导（10m 风向标量化为 16 档、30m +12.2° 偏置且
2021-11 失效、50m -6.5° 偏置），仅保留约 20% 样本，且造成两扇区日间比例
严重失衡（22% vs 49%）。

npz 里没有保存时间轴，但 step1 里的 `valid` 是确定性的：
    按时间排序 -> 四高度风速非 NaN 且 power 非 NaN
所以可以从 CSV 精确复现出 IMF 每一列对应的是哪一行观测，
从而在不重跑 CEEMDAN（40-90 min）的前提下换掉掩膜。

flatline 的处理
---------------
不从分解里剔除（那会改变序列长度，必须重跑 CEEMDAN）。
E = mean(imf^2[mask]) 本来就是条件平均，把 flatline 行从 mask 里去掉即可，
效果等价且零成本。EMD 的筛分是局地的，卡值造成的污染基本留在局地。
"""

import numpy as np
import pandas as pd

from qc_common import (
    sector_mask, flatline_mask,
    SECTOR_FREE, SECTOR_WAKE, WS_COLS, WD_SECTOR_COL,
)

HEIGHTS = (10, 30, 50, 70)


def build_aligned_frame(data_path, heights=HEIGHTS, verbose=True):
    """复现 step1 的 `valid`，返回与 IMF 列一一对应的 DataFrame。

    必须与 step1 的逻辑逐字一致：
        df.sort_values(time_col).reset_index(drop=True)
        valid = 四高度风速非 NaN & power 非 NaN
    """
    df = pd.read_csv(data_path)

    if 'datetime' in df.columns:
        time_col = 'datetime'
    elif 'timestamp' in df.columns:
        time_col = 'timestamp'
    else:
        raise ValueError("找不到时间列（datetime / timestamp）")

    df[time_col] = pd.to_datetime(df[time_col])
    df = df.sort_values(time_col).reset_index(drop=True)

    valid = np.ones(len(df), dtype=bool)
    for h in heights:
        col = f'obs_wind_speed_{h}m'
        if col in df.columns:
            valid &= ~np.isnan(df[col].values)
    if 'power' in df.columns:
        valid &= ~np.isnan(df['power'].values)

    dfv = df[valid].reset_index(drop=True)
    dfv.attrs['time_col'] = time_col

    if verbose:
        print(f"  [align] 总样本 {len(df)} -> IMF 对齐样本 {len(dfv)} "
              f"({len(dfv)/len(df)*100:.1f}%)")
    return dfv


def check_alignment(dfv, imfs, name='imfs'):
    """确认复现出的行数与 IMF 长度一致。对不上就直接停，别往下算。"""
    n_imf = np.asarray(imfs[0]).shape[0]
    if len(dfv) != n_imf:
        raise AssertionError(
            f"对齐失败：复现样本 {len(dfv)} != {name} 长度 {n_imf}。\n"
            f"可能原因：CSV 已变动、或 step1 的 valid 逻辑与此处不一致。\n"
            f"此时必须重跑 step1，不能沿用旧 npz。"
        )
    print(f"  [align] ✓ {name}: {n_imf} 与复现行数一致")
    return True


def build_sector_masks(dfv, drop_flatlines=True, verbose=True):
    """70m 单高度扇区掩膜，并剔除 flatline 行。"""
    if WD_SECTOR_COL not in dfv.columns:
        raise ValueError(f"缺少扇区判据列 {WD_SECTOR_COL}")

    m_free = sector_mask(dfv, SECTOR_FREE)
    m_wake = sector_mask(dfv, SECTOR_WAKE)

    if verbose:
        print(f"  [sector] 判据列 = {WD_SECTOR_COL} (70m 单高度)")
        print(f"  [sector] flatline 剔除前: free={m_free.sum()}, wake={m_wake.sum()}")

    if drop_flatlines:
        bad = np.zeros(len(dfv), dtype=bool)
        for c in WS_COLS:
            if c in dfv.columns:
                m = flatline_mask(dfv[c].values)
                if verbose and m.sum():
                    print(f"    [QC] flatline {c}: {m.sum()} 行")
                bad |= m
        m_free &= ~bad
        m_wake &= ~bad
        if verbose:
            print(f"  [sector] flatline 剔除后: free={m_free.sum()}, wake={m_wake.sum()}")

    n = len(dfv)
    print(f"  [sector] free={m_free.sum()} ({100*m_free.sum()/n:.1f}%) | "
          f"wake={m_wake.sum()} ({100*m_wake.sum()/n:.1f}%)")

    # 日间比例：用于回应 R2 Point 7b（对流偏向）
    tc = dfv.attrs.get('time_col', 'datetime')
    if tc in dfv.columns:
        t = pd.to_datetime(dfv[tc])
        day = ((t.dt.hour >= 8) & (t.dt.hour < 18)).values
        print(f"  [sector] 日间比例: free={day[m_free].mean():.2f} | "
              f"wake={day[m_wake].mean():.2f}")

    return m_free, m_wake


def sector_mean_speeds(dfv, m_free, m_wake, heights=HEIGHTS):
    """各扇区各高度的平均风速 —— 归一化要用，也是判断 R1 那处矛盾成因的关键。"""
    out = {}
    for h in heights:
        c = f'obs_wind_speed_{h}m'
        if c in dfv.columns:
            out[h] = {
                'free': float(np.mean(dfv[c].values[m_free])),
                'wake': float(np.mean(dfv[c].values[m_wake])),
            }
    print("\n  [平均风速 m/s]")
    print(f"    {'高度':>6} {'free':>8} {'wake':>8} {'wake/free':>10}")
    for h, v in out.items():
        print(f"    {h:>4}m {v['free']:>8.2f} {v['wake']:>8.2f} "
              f"{v['wake']/v['free']:>10.3f}")
    return out


def load_and_prepare(data_path, npz_path, ref_var='ws_70m', verbose=True):
    """一步到位：加载 npz、复现对齐、造新掩膜、算扇区平均风速。"""
    print("=" * 70)
    print("重建掩膜（70m 单高度判据）")
    print("=" * 70)

    data = np.load(npz_path, allow_pickle=True)
    dfv = build_aligned_frame(data_path, verbose=verbose)
    check_alignment(dfv, data[f'imfs_{ref_var}'], name=f'imfs_{ref_var}')
    m_free, m_wake = build_sector_masks(dfv, verbose=verbose)
    U = sector_mean_speeds(dfv, m_free, m_wake)

    return data, dfv, m_free, m_wake, U
