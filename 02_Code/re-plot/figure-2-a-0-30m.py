#!/usr/bin/env python3
"""
步骤2: 从保存的CEEMDAN结果快速绘图（含30m）
运行时间: <10秒
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
from datetime import datetime

# 配置
plt.rcParams.update({
    'font.family': 'Arial',
    'font.size': 11,
    'axes.labelsize': 12,
    'axes.titlesize': 12,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'legend.fontsize': 10,
    'figure.dpi': 300,
    'savefig.dpi': 300
})

COLORS = {'gray': '#686767', 'red': '#f41111', 'blue': '#1996de', 'purple': '#c5b0d5'}  # 添加30m颜色

def get_imf_period(imf):
    """计算IMF周期"""
    crossings = np.where(np.diff(np.sign(imf)))[0]
    if len(crossings) < 2:
        return np.nan
    return len(imf) / (len(crossings) / 2.0)

def calculate_energy(imfs, mask, dt):
    """计算能量和周期"""
    n = len(imfs)
    energies = np.zeros(n)
    periods = np.zeros(n)
    
    for i in range(n):
        periods[i] = get_imf_period(imfs[i]) * dt
        if mask.sum() > 0:
            energies[i] = np.mean(imfs[i][mask]**2)
        else:
            energies[i] = np.nan
    
    return energies, periods

# 路径
data_dir = '/Users/xiaxin/work/WindForecast_Project/03_Results/re-plot-figures/figure-2/'
saved_file = os.path.join(data_dir, 'ceemdan_results_full.npz')

print("=" * 70)
print("Fast Plotting from Saved CEEMDAN Results (Full Version)")
print("=" * 70)

# 1. 加载保存的结果
print("\n[1/4] Loading saved CEEMDAN results...")
if not os.path.exists(saved_file):
    print(f"ERROR: File not found: {saved_file}")
    print("Please run 'step1_ceemdan_full.py' first!")
    exit(1)

data = np.load(saved_file, allow_pickle=True)

# 检查可用变量
variables = data['variables'].tolist() if 'variables' in data else []
print(f"  ✓ Loaded from: {saved_file}")
print(f"  Method: {data['method']}")
print(f"  Processing date: {data['timestamp']}")
print(f"  Available variables: {', '.join(variables)}")

# 读取10m, 30m和70m
imfs_10m = data['imfs_ws_10m']
imfs_30m = data['imfs_ws_30m']  # 添加30m
imfs_70m = data['imfs_ws_70m']
mask_west = data['mask_west']
mask_east = data['mask_east']
dt = float(data['dt_hours'])

print(f"  Time resolution: {dt*60:.1f} min")
print(f"  10m IMFs: {len(imfs_10m)}")
print(f"  30m IMFs: {len(imfs_30m)}")  # 添加30m打印
print(f"  70m IMFs: {len(imfs_70m)}")

# 列出所有可用数据（供其他分析使用）
print(f"\n  All available data in file:")
for var in variables:
    imf_key = f'imfs_{var}'
    if imf_key in data:
        print(f"    - {var}: {len(data[imf_key])} IMFs")

# 2. 计算能量
print("\n[2/4] Calculating energies...")
E_10m, T_10m = calculate_energy(imfs_10m, np.ones(len(mask_west), dtype=bool), dt)
E_30m, T_30m = calculate_energy(imfs_30m, np.ones(len(mask_west), dtype=bool), dt)  # 添加30m计算
E_free, T_free = calculate_energy(imfs_70m, mask_west, dt)
E_wake, T_wake = calculate_energy(imfs_70m, mask_east, dt)

# 对齐
n = min(len(E_10m), len(E_30m), len(E_free), len(E_wake))  # 添加30m到对齐
print(f"  Using {n} IMFs")

E_10m, T_10m = E_10m[:n], T_10m[:n]
E_30m, T_30m = E_30m[:n], T_30m[:n]  # 添加30m对齐
E_free, T_free = E_free[:n], T_free[:n]
E_wake, T_wake = E_wake[:n], T_wake[:n]

V_10m = E_10m * T_10m
V_30m = E_30m * T_30m  # 添加30m能量贡献
V_free = E_free * T_free
V_wake = E_wake * T_wake

print(f"  IMFs: {n}")
print(f"  Period range: {T_free.min():.2f}h - {T_free.max():.2f}h")

# 3. 绘图
print("\n[3/4] Creating figure...")
fig, ax = plt.subplots(figsize=(10, 8))

# 区域文字标注
for t, label in {1: '1h', 6: '6h', 12: '12h', 24: '1d', 168: '1w'}.items():
    ax.axvline(t, color="#6C6C6E", linestyle='--', alpha=0.6, linewidth=1.5, zorder=1)
    t_offset = t * 1.3  # 向右偏移15%（对数空间）
    ax.text(t_offset, 0.93, label, ha='center', va='bottom', fontsize=28, color='gray', 
            alpha=0.8, transform=ax.get_xaxis_transform())

# 曲线（添加30m在10m和70m之间）
ax.plot(T_10m, V_10m, color=COLORS['gray'], linestyle='-', linewidth=1, 
        marker='o', markersize=10, markerfacecolor=COLORS['gray'], markeredgecolor=COLORS['gray'],
        markeredgewidth=1.8, label='10m Reference', zorder=3, alpha=0.9)

ax.plot(T_30m, V_30m, color=COLORS['purple'], linestyle='-', linewidth=1,  # 添加30m曲线
        marker='s', markersize=10, markerfacecolor=COLORS['purple'], markeredgecolor=COLORS['purple'], 
        markeredgewidth=1.8, label='30m Reference', zorder=3, alpha=0.9)

ax.plot(T_free, V_free, color=COLORS['red'], linestyle='-', linewidth=1,
        marker='o', markersize=10, markerfacecolor=COLORS['red'], markeredgecolor=COLORS['red'], 
        markeredgewidth=1.8, label='70m Free-stream (Westerly)', zorder=5)

ax.plot(T_wake, V_wake, color=COLORS['blue'], linestyle='-', linewidth=1,
        marker='o', markersize=10, markerfacecolor=COLORS['blue'], markeredgecolor=COLORS['blue'], 
        markeredgewidth=1.8, label='70m Wake (Easterly)', zorder=4)

# 设置坐标轴（在画背景前设置，以便获取正确的范围）
ax.set_xscale('log')
ax.set_yscale('log')
ax.set_xlabel('Period (Hours)', fontsize=28, fontweight='bold')
ax.set_ylabel(r'Variance Contribution (m$^2$$\cdot$s$^{-2}$)', fontsize=28, fontweight='bold')
ax.set_title('Scale-dependent Energy Spectrum', fontsize=30, fontweight='bold', pad=15)
# ax.legend(loc='upper left', frameon=False, fontsize=20, labelspacing=0.4)
ax.legend(loc='lower right', frameon=False, framealpha=0.9, fontsize=20, markerfirst=False)
ax.grid(True, which='both', alpha=0.3, linestyle=':', linewidth=0.5)
ax.tick_params(axis='both', which='major', labelsize=28, width=1.2, length=5, direction='in')
ax.tick_params(axis='both', which='minor', labelsize=28, width=0.8, length=3, direction='in')

for spine in ax.spines.values():
    spine.set_linewidth(1.2)

# 获取实际的Y轴范围（在对数坐标下）
ylim = ax.get_ylim()
xlim = ax.get_xlim()

# 定性区域标注（背景块）- 使用实际坐标范围
from matplotlib.patches import Rectangle

# 1h-24h: 蓝色背景 - Wake-Affected Variability
wake_x_start = max(1, xlim[0])
wake_x_width = 24 - wake_x_start
if wake_x_width > 0:
    wake_zone = Rectangle((wake_x_start, ylim[0]), wake_x_width, ylim[1]-ylim[0],
                           linewidth=0, facecolor=COLORS['blue'], alpha=0.08, zorder=0)
    ax.add_patch(wake_zone)

# 24h-xlim[1]: 红色背景 - Background Energy Proxy Zone  
proxy_x_start = 24
proxy_x_width = xlim[1] - proxy_x_start
if proxy_x_width > 0:
    proxy_zone = Rectangle((proxy_x_start, ylim[0]), proxy_x_width, ylim[1]-ylim[0],
                            linewidth=0, facecolor=COLORS['red'], alpha=0.08, zorder=0)
    ax.add_patch(proxy_zone)

# 文字标注 - 使用相对位置
# Wake-Affected区域 (蓝色)
# text_y = np.exp(np.log(ylim[0]) + 0.1 * (np.log(ylim[1]) - np.log(ylim[0])))  # Y轴10%位置
# ax.text(5, text_y, 'Wake-Affected\nVariability', 
#         fontsize=12, fontweight='bold', color=COLORS['blue'],
#         ha='center', va='center', alpha=0.8)

# # Background Proxy区域 (红色)
# ax.text(80, text_y, 'Background Energy\nProxy Zone', 
#         fontsize=12, fontweight='bold', color=COLORS['red'],
#         ha='center', va='center', alpha=0.8)

plt.tight_layout()

# 4. 保存
print("\nSaving figure...")
pdf_path = os.path.join(data_dir, 'final-energy_spectrum_CEEMDAN_with30m.pdf')
png_path = os.path.join(data_dir, 'final-energy_spectrum_CEEMDAN_with30m.png')

plt.savefig(pdf_path, bbox_inches='tight', facecolor='white')
plt.savefig(png_path, dpi=300, bbox_inches='tight', facecolor='white')

print(f"  ✓ {pdf_path}")
print(f"  ✓ {png_path}")

plt.show()

# 统计
print(f"\n=== Summary ===")
print(f"Method: CEEMDAN")
print(f"IMFs displayed: {n}")
print(f"Period range shown: {T_free.min():.2f}h - {T_free.max():.2f}h")

# 短周期统计 (<24h)
short_mask = T_free < 24
if short_mask.any():
    deficit = (V_free[short_mask] - V_wake[short_mask]) / V_free[short_mask] * 100
    print(f"Wake-affected zone (<24h) mean deficit: {deficit.mean():.1f}%")

# 长周期统计 (>=24h)
long_mask = T_free >= 24
if long_mask.any():
    diff = np.abs(V_10m[long_mask] - V_free[long_mask]) / V_free[long_mask] * 100
    wake_diff = np.abs(V_free[long_mask] - V_wake[long_mask]) / V_free[long_mask] * 100
    print(f"Proxy zone (≥24h) 10m-70m difference: {diff.mean():.1f}%")
    print(f"Proxy zone (≥24h) Wake-Free difference: {wake_diff.mean():.1f}%")

print("\n" + "=" * 70)
print("✓ Figure created successfully!")
print("=" * 70)
print(f"\nProcessing time: <10 seconds")