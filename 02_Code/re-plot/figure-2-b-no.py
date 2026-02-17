#!/usr/bin/env python3
"""
完全基于CEEMDAN结果的相关性分析
计算所有高度对和功率的相关性 + 条件相关性（70m Free vs Wake）
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
from scipy.stats import pearsonr

# Set style
plt.rcParams.update({
    'font.size': 10,
    'axes.titlesize': 11,
    'axes.labelsize': 10,
    'xtick.labelsize': 9,
    'ytick.labelsize': 9,
    'legend.fontsize': 7,
    'figure.titlesize': 13,
    'font.family': 'Arial'
})

def get_imf_period(imf):
    """计算IMF周期"""
    crossings = np.where(np.diff(np.sign(imf)))[0]
    if len(crossings) < 2:
        return np.nan
    return len(imf) / (len(crossings) / 2.0)

def calculate_correlation_all_data(imfs_a, imfs_b, dt):
    """
    计算两个变量之间所有IMF的相关性（使用全部数据）
    """
    n = min(len(imfs_a), len(imfs_b))
    correlations = []
    periods = []
    imf_numbers = []
    
    for i in range(n):
        period = get_imf_period(imfs_a[i]) * dt
        
        try:
            corr, _ = pearsonr(imfs_a[i], imfs_b[i])
            correlations.append(corr)
            periods.append(period)
            imf_numbers.append(i + 1)
        except:
            pass
    
    return np.array(correlations), np.array(periods), np.array(imf_numbers)

def calculate_conditional_correlation(imfs_a, imfs_b, mask, dt):
    """
    计算条件相关性（特定风向条件下）
    """
    n = min(len(imfs_a), len(imfs_b))
    correlations = []
    periods = []
    imf_numbers = []
    
    for i in range(n):
        period = get_imf_period(imfs_a[i]) * dt
        
        seg_a = imfs_a[i][mask]
        seg_b = imfs_b[i][mask]
        
        if len(seg_a) < 10:
            continue
        
        try:
            corr, _ = pearsonr(seg_a, seg_b)
            correlations.append(corr)
            periods.append(period)
            imf_numbers.append(i + 1)
        except:
            pass
    
    return np.array(correlations), np.array(periods), np.array(imf_numbers)

# ========== 路径设置 ==========
ceemdan_path = '/Users/xiaxin/work/WindForecast_Project/03_Results/re-plot-figures/figure-2/ceemdan_results_full.npz'
output_dir = '/Users/xiaxin/work/WindForecast_Project/03_Results/re-plot-figures/figure-2/'
os.makedirs(output_dir, exist_ok=True)

print("=" * 70)
print("Complete Correlation Analysis from CEEMDAN Results")
print("=" * 70)

# ========== 加载CEEMDAN结果 ==========
print("\n[1/3] Loading CEEMDAN results...")
if not os.path.exists(ceemdan_path):
    print(f"ERROR: CEEMDAN file not found: {ceemdan_path}")
    print("Please run 'step1_ceemdan_full.py' first!")
    exit(1)

data = np.load(ceemdan_path, allow_pickle=True)

# 提取所有变量
imfs_10m = data['imfs_ws_10m']
imfs_30m = data['imfs_ws_30m']
imfs_50m = data['imfs_ws_50m']
imfs_70m = data['imfs_ws_70m']
imfs_power = data['imfs_power'] if 'imfs_power' in data else None

mask_west = data['mask_west']
mask_east = data['mask_east']
dt = float(data['dt_hours'])

print(f"  ✓ Loaded CEEMDAN results")
print(f"    10m IMFs: {len(imfs_10m)}")
print(f"    30m IMFs: {len(imfs_30m)}")
print(f"    50m IMFs: {len(imfs_50m)}")
print(f"    70m IMFs: {len(imfs_70m)}")
if imfs_power is not None:
    print(f"    Power IMFs: {len(imfs_power)}")
print(f"    West wind samples: {mask_west.sum()}")
print(f"    East wind samples: {mask_east.sum()}")

# ========== 计算所有相关性 ==========
print("\n[2/3] Computing all correlations...")

correlations_dict = {}

# 定义所有变量对
pairs = [
    ('10m', '30m', imfs_10m, imfs_30m),
    ('10m', '50m', imfs_10m, imfs_50m),
    ('10m', '70m', imfs_10m, imfs_70m),
    ('30m', '50m', imfs_30m, imfs_50m),
    ('30m', '70m', imfs_30m, imfs_70m),
    ('50m', '70m', imfs_50m, imfs_70m),
]

# 如果有功率数据，添加功率相关性
if imfs_power is not None:
    pairs.extend([
        ('10m', 'power', imfs_10m, imfs_power),
        ('30m', 'power', imfs_30m, imfs_power),
        ('50m', 'power', imfs_50m, imfs_power),
        ('70m', 'power', imfs_70m, imfs_power),
    ])

# 计算全部数据的相关性
for name1, name2, imfs_a, imfs_b in pairs:
    key = f'{name1}-{name2}'
    print(f"  Computing {key}...")
    corr, period, imf_num = calculate_correlation_all_data(imfs_a, imfs_b, dt)
    correlations_dict[key] = {
        'correlation': corr,
        'period': period,
        'imf': imf_num,
        'type': 'all'
    }

# 计算条件相关性（10m-70m）
print(f"  Computing 10m-70m Free-stream (West)...")
corr_free, period_free, imf_free = calculate_conditional_correlation(
    imfs_10m, imfs_70m, mask_west, dt)
correlations_dict['10m-70m-free'] = {
    'correlation': corr_free,
    'period': period_free,
    'imf': imf_free,
    'type': 'conditional'
}

print(f"  Computing 10m-70m Wake (East)...")
corr_wake, period_wake, imf_wake = calculate_conditional_correlation(
    imfs_10m, imfs_70m, mask_east, dt)
correlations_dict['10m-70m-wake'] = {
    'correlation': corr_wake,
    'period': period_wake,
    'imf': imf_wake,
    'type': 'conditional'
}

print(f"  ✓ Total pairs computed: {len(correlations_dict)}")

# ========== 绘图 ==========
print("\n[3/3] Creating figure...")

# 颜色映射
color_mapping = {
    '30m-50m': '#2ca02c',
    '30m-70m': '#98df8a',
    '50m-70m': '#8c564b',
    '10m-30m': '#9467bd',
    '10m-50m': '#aec7e8',
    '10m-70m': '#c5b0d5',
    '10m-power': '#d62728',
    '30m-power': '#ff9896',
    '50m-power': '#ff7f0e',
    '70m-power': '#e377c2'
}

conditional_colors = {
    'free': '#f41111',
    'wake': '#1996de'
}

fig = plt.figure(figsize=(14, 7))
gs = fig.add_gridspec(1, 2, width_ratios=[4, 1.5], wspace=0.25)

# ========== 左图：Period vs Correlation ==========
ax1 = fig.add_subplot(gs[0, 0])

# 时间标记
time_markers = {1: '1h', 6: '6h', 24: '1d', 168: '1w', 720: '1m'}

for period_hours, label in time_markers.items():
    ax1.axvline(x=period_hours, color='gray', linestyle='--', alpha=0.6, linewidth=1.2)
    ax1.text(period_hours, 0.918, label, rotation=90, ha='center', va='bottom', 
             fontsize=18, color='gray', fontweight='bold',
             transform=ax1.get_xaxis_transform())

# 画所有常规相关性
for key in color_mapping.keys():
    if key in correlations_dict:
        data_dict = correlations_dict[key]
        period = data_dict['period']
        corr = data_dict['correlation']
        
        # 排序
        sort_idx = np.argsort(period)
        period = period[sort_idx]
        corr = corr[sort_idx]
        
        # 设置样式
        if '10m' in key:
            marker = 's'
            size = 80 if 'power' in key else 50
        else:
            marker = 'o'
            size = 60 if 'power' in key else 60
        
        alpha = 1.0 if 'power' in key else 0.7
        linewidth = 1.5 if 'power' in key else 1.0
        
        # 标签
        label = key.replace('m-', 'm-').replace('power', 'Power')
        
        # 画线
        ax1.plot(period, corr, color=color_mapping[key], alpha=alpha*0.6, 
                linewidth=linewidth, linestyle='-', zorder=2)
        
        # 画点
        ax1.scatter(period, corr, c=color_mapping[key], marker=marker, 
                   s=size, alpha=alpha, label=label, 
                   edgecolors='black', linewidth=0.5, zorder=3)
        
        # 标注70m-power的IMF编号
        if '70m-power' in key:
            imf_nums = data_dict['imf'][sort_idx]
            for p, c, imf in zip(period, corr, imf_nums):
                ax1.annotate(f'IMF{int(imf)}', (p, c),
                           xytext=(-16, -38), textcoords='offset points', 
                           fontsize=14, alpha=0.8, zorder=4)

# 画条件相关性（粗线）
if '10m-70m-free' in correlations_dict:
    data_free = correlations_dict['10m-70m-free']
    period_free = data_free['period']
    corr_free = data_free['correlation']
    
    sort_idx = np.argsort(period_free)
    period_free = period_free[sort_idx]
    corr_free = corr_free[sort_idx]
    
    ax1.plot(period_free, corr_free, color=conditional_colors['free'], 
            linestyle='-', linewidth=3.5, alpha=0.9, zorder=6, 
            label='10m-70m Free (West)')
    ax1.scatter(period_free, corr_free, c=conditional_colors['free'], 
               marker='D', s=120, edgecolors='white', linewidth=2, 
               zorder=7, alpha=0.95)

if '10m-70m-wake' in correlations_dict:
    data_wake = correlations_dict['10m-70m-wake']
    period_wake = data_wake['period']
    corr_wake = data_wake['correlation']
    
    sort_idx = np.argsort(period_wake)
    period_wake = period_wake[sort_idx]
    corr_wake = corr_wake[sort_idx]
    
    ax1.plot(period_wake, corr_wake, color=conditional_colors['wake'], 
            linestyle='-', linewidth=3.5, alpha=0.9, zorder=6, 
            label='10m-70m Wake (East)')
    ax1.scatter(period_wake, corr_wake, c=conditional_colors['wake'], 
               marker='D', s=120, edgecolors='white', linewidth=2, 
               zorder=7, alpha=0.95)

ax1.set_xlabel('Period (Hours)', fontsize=16)
ax1.set_ylabel('Correlation Coefficient', fontsize=16)
ax1.set_title('(a) Scale-dependent Correlations', fontsize=16)
ax1.set_xscale('log')
ax1.legend(fontsize=13, loc='lower right', ncol=2, framealpha=0.95)
ax1.grid(False)
ax1.set_ylim(0, 1.05)
ax1.set_yticks(np.arange(0, 1.1, 0.1))
ax1.tick_params(axis='both', which='major', labelsize=16)

# ========== 右图：Trend Component ==========
ax2 = fig.add_subplot(gs[0, 1])

if imfs_power is not None:
    heights = ['10m', '30m', '50m', '70m']
    height_colors = ['#d62728', '#ff9896', '#c5b0d5', '#e377c2']
    
    trend_correlations = []
    
    for height in heights:
        key = f'{height}-power'
        if key in correlations_dict:
            # 取最后一个IMF（趋势分量）
            corr_array = correlations_dict[key]['correlation']
            if len(corr_array) > 0:
                trend_correlations.append(corr_array[-1])
            else:
                trend_correlations.append(0)
        else:
            trend_correlations.append(0)
    
    bars = ax2.barh(range(len(heights)), trend_correlations, 
                    color=height_colors, alpha=0.85, height=0.2)
    
    for i, (bar, val) in enumerate(zip(bars, trend_correlations)):
        ax2.text(-0.056-0.02, bar.get_y() + bar.get_height()/2.,
                f'{val:.3f}', ha='right', va='center', fontsize=14)
    
    if trend_correlations and max(trend_correlations) > 0:
        best_idx = np.argmax(trend_correlations)
        bars[best_idx].set_color('#b91c1c')
    
    ax2.set_yticks(range(len(heights)))
    ax2.tick_params(axis='both', which='major', labelsize=14)
    ax2.set_yticklabels([f'WS {h}' for h in heights], fontsize=14)
    ax2.set_xlabel('Trend Component Correlation', fontsize=14)
    ax2.set_title('(b)', fontsize=14)
    ax2.axvline(x=0, color='black', linewidth=1.2, alpha=0.8)
    ax2.grid(False)
    
    if trend_correlations:
        max_abs_val = max(abs(min(trend_correlations)), abs(max(trend_correlations)))
        ax2.set_xlim(-max_abs_val * 1.3, max_abs_val * 1.3)
        ax2.set_ylim(-0.4, len(heights) - 0.6)

plt.tight_layout()

# ========== 保存 ==========
png_path = os.path.join(output_dir, 'CEEMDAN_full_correlation_analysis.png')
pdf_path = os.path.join(output_dir, 'CEEMDAN_full_correlation_analysis.pdf')

plt.savefig(png_path, dpi=300, bbox_inches='tight', facecolor='white')
plt.savefig(pdf_path, bbox_inches='tight', facecolor='white')

print(f"\n✓ Figures saved:")
print(f"  PNG: {png_path}")  
print(f"  PDF: {pdf_path}")

plt.show()

# ========== 统计摘要 ==========
print(f"\n=== Analysis Summary ===")
print(f"Total correlation pairs: {len(correlations_dict)}")

# 条件相关性统计
if '10m-70m-free' in correlations_dict and '10m-70m-wake' in correlations_dict:
    free_data = correlations_dict['10m-70m-free']
    wake_data = correlations_dict['10m-70m-wake']
    
    period_free = free_data['period']
    corr_free = free_data['correlation']
    period_wake = wake_data['period']
    corr_wake = wake_data['correlation']
    
    short_mask_free = period_free < 24
    short_mask_wake = period_wake < 24
    long_mask_free = period_free >= 24
    long_mask_wake = period_wake >= 24
    
    print(f"\nConditional correlations (10m-70m):")
    print(f"  Short-period (<24h):")
    print(f"    Free-stream: {np.mean(corr_free[short_mask_free]):.3f}")
    print(f"    Wake: {np.mean(corr_wake[short_mask_wake]):.3f}")
    print(f"    Decoupling: {np.mean(corr_free[short_mask_free]) - np.mean(corr_wake[short_mask_wake]):.3f}")
    
    print(f"  Long-period (≥24h):")
    print(f"    Free-stream: {np.mean(corr_free[long_mask_free]):.3f}")
    print(f"    Wake: {np.mean(corr_wake[long_mask_wake]):.3f}")
    print(f"    Difference: {abs(np.mean(corr_free[long_mask_free]) - np.mean(corr_wake[long_mask_wake])):.3f}")

print("\n" + "=" * 70)
print("✓ Complete correlation analysis finished!")
print("=" * 70)