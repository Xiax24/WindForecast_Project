#!/usr/bin/env python3
"""
步骤2b-1: 计算所有IMF相关系数并保存
运行一次后，绘图时直接加载，无需重复计算
"""

import pandas as pd
import numpy as np
import os
from scipy.stats import spearmanr
from datetime import datetime

def get_imf_period(imf):
    """计算IMF周期"""
    crossings = np.where(np.diff(np.sign(imf)))[0]
    if len(crossings) < 2:
        return np.nan
    return len(imf) / (len(crossings) / 2.0)

def calculate_correlation(imfs_a, imfs_b, mask, dt):
    """计算相关性"""
    n = min(len(imfs_a), len(imfs_b))
    correlations = np.zeros(n)
    periods = np.zeros(n)
    
    for i in range(n):
        periods[i] = get_imf_period(imfs_a[i]) * dt
        
        if mask is None:
            seg_a = imfs_a[i]
            seg_b = imfs_b[i]
        else:
            seg_a = imfs_a[i][mask]
            seg_b = imfs_b[i][mask]
        
        if len(seg_a) < 10 or len(seg_b) < 10:
            correlations[i] = np.nan
            continue
        
        try:
            corr, _ = spearmanr(seg_a, seg_b)
            correlations[i] = corr
        except:
            correlations[i] = np.nan
    
    return correlations, periods

# 路径
data_dir = '/Users/xiaxin/work/WindForecast_Project/03_Results/re-plot-figures/figure-2/'
ceemdan_file = os.path.join(data_dir, 'ceemdan_results_full.npz')
output_file = os.path.join(data_dir, 'correlations_all.npz')

print("=" * 70)
print("Calculate All IMF Correlations")
print("=" * 70)
print(f"Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

# 1. 加载CEEMDAN结果
print("\n[1/3] Loading CEEMDAN results...")
if not os.path.exists(ceemdan_file):
    print(f"ERROR: File not found: {ceemdan_file}")
    exit(1)

data = np.load(ceemdan_file, allow_pickle=True)

imfs_10m = data['imfs_ws_10m']
imfs_30m = data['imfs_ws_30m'] if 'imfs_ws_30m' in data else None
imfs_50m = data['imfs_ws_50m'] if 'imfs_ws_50m' in data else None
imfs_70m = data['imfs_ws_70m']

if 'imfs_power' in data:
    imfs_power = data['imfs_power']
    has_power = True
else:
    imfs_power = None
    has_power = False

mask_west = data['mask_west']
mask_east = data['mask_east']
dt = float(data['dt_hours'])

print(f"  ✓ Loaded: 10m={len(imfs_10m)}, 70m={len(imfs_70m)}")
if imfs_30m is not None:
    print(f"  ✓ 30m: {len(imfs_30m)}")
if imfs_50m is not None:
    print(f"  ✓ 50m: {len(imfs_50m)}")
if has_power:
    print(f"  ✓ Power: {len(imfs_power)}")

# 2. 计算所有相关性
print("\n[2/3] Calculating all correlations...")
start_time = datetime.now()

correlations = {}

# All data - 风速间相关性
print("  Computing all-data correlations...")
R, T = calculate_correlation(imfs_10m, imfs_70m, None, dt)
correlations['10m-70m-all'] = (R, T)
print(f"    ✓ 10m-70m (all)")

# All data - 功率相关性
if has_power:
    R, T = calculate_correlation(imfs_10m, imfs_power, None, dt)
    correlations['10m-power-all'] = (R, T)
    print(f"    ✓ 10m-Power (all)")
    
    if imfs_30m is not None:
        R, T = calculate_correlation(imfs_30m, imfs_power, None, dt)
        correlations['30m-power-all'] = (R, T)
        print(f"    ✓ 30m-Power (all)")
    
    if imfs_50m is not None:
        R, T = calculate_correlation(imfs_50m, imfs_power, None, dt)
        correlations['50m-power-all'] = (R, T)
        print(f"    ✓ 50m-Power (all)")
    
    R, T = calculate_correlation(imfs_70m, imfs_power, None, dt)
    correlations['70m-power-all'] = (R, T)
    print(f"    ✓ 70m-Power (all)")

# Conditional - 10m-70m
print("  Computing conditional correlations (10m-70m)...")
R, T = calculate_correlation(imfs_10m, imfs_70m, mask_east, dt)
correlations['10m-70m-wake'] = (R, T)
print(f"    ✓ 10m-70m Wake")

R, T = calculate_correlation(imfs_10m, imfs_70m, mask_west, dt)
correlations['10m-70m-free'] = (R, T)
print(f"    ✓ 10m-70m Free")

# Conditional - 30m-70m
if imfs_30m is not None:
    print("  Computing conditional correlations (30m-70m)...")
    R, T = calculate_correlation(imfs_30m, imfs_70m, mask_east, dt)
    correlations['30m-70m-wake'] = (R, T)
    print(f"    ✓ 30m-70m Wake")
    
    R, T = calculate_correlation(imfs_30m, imfs_70m, mask_west, dt)
    correlations['30m-70m-free'] = (R, T)
    print(f"    ✓ 30m-70m Free")

# Conditional - 10m-Power
if has_power:
    print("  Computing conditional correlations (10m-Power)...")
    R, T = calculate_correlation(imfs_10m, imfs_power, mask_east, dt)
    correlations['10m-power-wake'] = (R, T)
    print(f"    ✓ 10m-Power Wake")
    
    R, T = calculate_correlation(imfs_10m, imfs_power, mask_west, dt)
    correlations['10m-power-free'] = (R, T)
    print(f"    ✓ 10m-Power Free")
    
    # Conditional - 70m-Power
    print("  Computing conditional correlations (70m-Power)...")
    R, T = calculate_correlation(imfs_70m, imfs_power, mask_east, dt)
    correlations['70m-power-wake'] = (R, T)
    print(f"    ✓ 70m-Power Wake")
    
    R, T = calculate_correlation(imfs_70m, imfs_power, mask_west, dt)
    correlations['70m-power-free'] = (R, T)
    print(f"    ✓ 70m-Power Free")

elapsed = (datetime.now() - start_time).total_seconds()
print(f"\n  Calculated {len(correlations)} correlations in {elapsed:.1f} seconds")

# 3. 保存结果
print("\n[3/3] Saving correlations...")

# 对齐到最小公共长度
n = min(len(v[0]) for v in correlations.values())
save_dict = {
    'dt_hours': dt,
    'n_imfs': n,
    'timestamp': datetime.now().isoformat(),
    'correlation_names': list(correlations.keys()),
}

# 保存每个相关性
for key, (R, T) in correlations.items():
    save_dict[f'{key}_R'] = R[:n]
    save_dict[f'{key}_T'] = T[:n]

np.savez_compressed(output_file, **save_dict)

print(f"  ✓ Saved: {output_file}")
print(f"  File size: {os.path.getsize(output_file) / 1024:.1f} KB")
print(f"  Contains {len(correlations)} correlations with {n} IMFs each")

# 打印摘要
print(f"\n=== Summary ===")
print(f"Total correlations: {len(correlations)}")
print(f"IMFs per correlation: {n}")
T_periods = correlations['10m-70m-all'][1][:n]
print(f"Period range: {T_periods.min():.2f}h - {T_periods.max():.2f}h")

print("\nSaved correlations:")
for key in correlations.keys():
    print(f"  - {key}")

print("\n" + "=" * 70)
print("✓ Correlation calculation completed!")
print("=" * 70)
print(f"Total time: {elapsed:.1f} seconds")
print(f"\nNext step: Run 'step2b-2_plot_correlations.py' to create figures")