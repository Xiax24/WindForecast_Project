#!/usr/bin/env python3
"""
顶刊级别风速廓线
- 展示所有样本廓线（透明）
- 叠加平均廓线（加粗）
- 精致配色和透明度
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

# Nature/Science 子刊标准配置
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

def strict_direction_mask(data, heights, direction_range):
    """严格筛选：所有高度都在指定风向区间"""
    min_deg, max_deg = direction_range
    mask = np.ones(len(data[f'wd_{heights[0]}m']), dtype=bool)
    
    for h in heights:
        wd = data[f'wd_{h}m']
        if min_deg > max_deg:
            h_mask = (wd >= min_deg) | (wd <= max_deg)
        else:
            h_mask = (wd >= min_deg) & (wd <= max_deg)
        mask = mask & h_mask & ~np.isnan(wd)
    
    return mask

def get_valid_profiles(data, heights, mask, max_samples=200):
    """
    获取有效的风廓线样本
    max_samples: 最多绘制的样本数（避免过于密集）
    """
    n_total = np.sum(mask)
    indices = np.where(mask)[0]
    
    # 如果样本太多，随机采样
    if len(indices) > max_samples:
        indices = np.random.choice(indices, max_samples, replace=False)
    
    profiles = []
    for idx in indices:
        profile = []
        valid = True
        for h in heights:
            ws = data[f'ws_{h}m'][idx]
            if np.isnan(ws):
                valid = False
                break
            profile.append(ws)
        if valid:
            profiles.append(profile)
    
    return np.array(profiles)

def plot_elegant_profiles(csv_file, heights, output_path):
    """绘制精致的风廓线图"""
    
    # 加载数据
    print("加载数据...")
    df = pd.read_csv(csv_file)
    data = {}
    for h in heights:
        data[f'ws_{h}m'] = df[f'obs_wind_speed_{h}m'].values
        data[f'wd_{h}m'] = df[f'obs_wind_direction_{h}m'].values
    
    # 严格筛选
    print("筛选数据...")
    mask_all = np.ones(len(df), dtype=bool)
    for h in heights:
        mask_all = mask_all & ~np.isnan(data[f'wd_{h}m'])
    
    mask_west = strict_direction_mask(data, heights, (225, 315))  # 西风
    mask_east = strict_direction_mask(data, heights, (45, 135))   # 东风
    
    # 获取样本廓线
    print("提取廓线样本...")
    profiles_all = get_valid_profiles(data, heights, mask_all, max_samples=200)
    profiles_west = get_valid_profiles(data, heights, mask_west, max_samples=150)
    profiles_east = get_valid_profiles(data, heights, mask_east, max_samples=150)
    
    # 计算平均廓线
    mean_all = np.nanmean(profiles_all, axis=0)
    mean_west = np.nanmean(profiles_west, axis=0)
    mean_east = np.nanmean(profiles_east, axis=0)
    
    # ========== 添加这3行 ==========
    std_all = np.nanstd(profiles_all, axis=0)
    std_west = np.nanstd(profiles_west, axis=0)
    std_east = np.nanstd(profiles_east, axis=0)
    # ==============================

    # 统计
    n_all = np.sum(mask_all)
    n_west = np.sum(mask_west)
    n_east = np.sum(mask_east)
    
    print(f"\n样本统计:")
    print(f"  整体: {n_all:,} (绘制 {len(profiles_all)} 条)")
    print(f"  西风: {n_west:,} (绘制 {len(profiles_west)} 条)")
    print(f"  东风: {n_east:,} (绘制 {len(profiles_east)} 条)")
    
    # 打印平均值
    print(f"\n平均风速 (m·s⁻¹):")
    print(f"  高度  整体    西风    东风")
    for i, h in enumerate(heights):
        print(f"  {h:2d}m  {mean_all[i]:5.2f}  {mean_west[i]:5.2f}  {mean_east[i]:5.2f}")
    
    # 统一X轴范围
    all_means = np.concatenate([mean_all, mean_west, mean_east])
    x_min = np.floor(np.min(all_means) - 0.5)
    x_max = np.ceil(np.max(all_means) + 0.5)
    
    print(f"\nX轴范围: {x_min:.1f} - {x_max:.1f} m·s⁻¹")
    
    # 精致配色（浅色透明 + 深色平均线）
    colors = {
        'all': {
            'light': "#686767",      # 浅灰（透明廓线）
            'dark': '#2d2d2d',       # 深灰（平均线）
        },
        'east': {
            'light': '#6fa8dc',      # 浅蓝（透明廓线）
            'dark': '#2e5f8a',       # 深蓝（平均线）
        },
        'west': {
            'light': '#e07b7b',      # 浅红（透明廓线）
            'dark': '#b63939',       # 深红（平均线）
        }
    }
    
    # 创建图形
    fig, axes = plt.subplots(1, 3, figsize=(12, 5), sharey=True)
    
    # 子图1：整体
    ax = axes[0]
    # 绘制所有样本廓线（透明）
    for profile in profiles_all:
        ax.plot(profile, heights, '-', color=colors['all']['light'], 
                alpha=0.08, linewidth=0.8, zorder=1)
    # ========== 添加这段（在平均线之前）==========
    # 标准差阴影
    ax.fill_betweenx(heights, 
                    mean_all - std_all,  # 左边界
                    mean_all + std_all,  # 右边界
                    color=colors['all']['dark'], 
                    alpha=0.15,  # 透明度
                    zorder=2)
    # ==========================================

    # 绘制平均廓线
    # 空心：markerfacecolor='white', markeredgecolor=color, markeredgewidth=1.5
    # 实心：直接用 color，不设置 facecolor
    ax.plot(mean_all, heights, 'o-', color=colors['all']['dark'], 
            linewidth=3, markersize=8, 
            markerfacecolor='white',  # 改为 color 则为实心
            markeredgecolor=colors['all']['dark'],
            markeredgewidth=1.5,
            label=f'Mean (n={n_all:,})', zorder=3)

    ax.set_xlabel(r'Wind Speed (m$\cdot$s$^{-1}$)', fontweight='bold', fontsize=16)
    ax.set_ylabel('Height (m)', fontweight='bold', fontsize=16)
    ax.set_title('Overall', fontweight='bold', pad=10, fontsize=18)

    # 去除网格
    ax.grid(True, linestyle='dotted', alpha=0.5, linewidth=0.8, color='gray')
    
    # ax.legend(loc='lower right', frameon=True, framealpha=0.9)
    # 添加图例（包含阴影和透明线说明）
    from matplotlib.patches import Patch
    from matplotlib.lines import Line2D

    legend_elements = [
        Line2D([0], [0], color=colors['all']['dark'], linewidth=3, 
            marker='o', markersize=8, markerfacecolor='white',
            markeredgecolor=colors['all']['dark'], markeredgewidth=1.5,
            label=f'Mean (n={n_all:,})'),
        Patch(facecolor=colors['all']['dark'], alpha=0.15, 
            label='Mean ± 1σ'),
        Line2D([0], [0], color=colors['all']['light'], linewidth=0.8, 
            alpha=0.3, label='Individual profiles')
    ]
    ax.legend(handles=legend_elements, loc='lower right', 
            frameon=True, framealpha=0.9, fontsize=12)
    # 设置Y轴：刻度每10m，但只显示测量高度的标签
    ax.set_ylim(0, max(heights) + 10)
    ax.set_yticks(np.arange(0, max(heights) + 11, 10))  # 每10m一个刻度
    ax.set_yticklabels([str(h) if h in heights else '' for h in np.arange(0, max(heights) + 11, 10)])
    
    # 统一X轴范围
    # ax.set_xlim(x_min, x_max)
    ax.set_xlim(0, 20)
    # 去除上侧和右侧坐标轴
    ax.spines['top'].set_visible(True)
    ax.spines['right'].set_visible(True)
    
    # 放大tick
    ax.tick_params(axis='both', which='major', labelsize=18, width=1.2, length=5)
    
    # 子图2：西风（自由流）
    ax = axes[1]
    for profile in profiles_west:
        ax.plot(profile, heights, '-', color=colors['west']['light'], 
                alpha=0.12, linewidth=0.8, zorder=1)
        
    # ========== 添加这段 ==========
    ax.fill_betweenx(heights, 
                    mean_west - std_west, 
                    mean_west + std_west,
                    color=colors['west']['dark'], 
                    alpha=0.15, 
                    zorder=2)
    # ============================

    ax.plot(mean_west, heights, 'o-', color=colors['west']['dark'], 
            linewidth=3, markersize=8,
            markerfacecolor='white',  # 改为 color 则为实心
            markeredgecolor=colors['west']['dark'],
            markeredgewidth=1.5,
            label=f'Mean (n={n_west:,})', zorder=3)

    ax.set_xlabel(r'Wind Speed (m$\cdot$s$^{-1}$)', fontweight='bold', fontsize=16)
    ax.set_title('Westerly (Free-stream)', fontweight='bold', pad=10, fontsize=18)
    ax.grid(True, linestyle='dotted', alpha=0.5, linewidth=0.8, color='gray')
    # ax.legend(loc='lower right', frameon=True, framealpha=0.9)
    legend_elements = [
    Line2D([0], [0], color=colors['west']['dark'], linewidth=3, 
           marker='o', markersize=8, markerfacecolor='white',
           markeredgecolor=colors['west']['dark'], markeredgewidth=1.5,
           label=f'Mean (n={n_west:,})'),
    Patch(facecolor=colors['west']['dark'], alpha=0.15, 
          label='Mean ± 1σ'),
    Line2D([0], [0], color=colors['west']['light'], linewidth=0.8, 
           alpha=0.3, label='Individual profiles')
    ]
    ax.legend(handles=legend_elements, loc='lower right', 
            frameon=True, framealpha=0.9, fontsize=12)
    # ax.set_xlim(x_min, x_max)
    ax.set_xlim(0, 20)
    ax.spines['top'].set_visible(True)
    ax.spines['right'].set_visible(True)
    ax.tick_params(axis='both', which='major', labelsize=18, width=1.2, length=5)
    
    # 子图3：东风（尾流）
    ax = axes[2]
    for profile in profiles_east:
        ax.plot(profile, heights, '-', color=colors['east']['light'], 
                alpha=0.12, linewidth=0.8, zorder=1)
    # ========== 添加这段 ==========
    ax.fill_betweenx(heights, 
                    mean_east - std_east, 
                    mean_east + std_east,
                    color=colors['east']['dark'], 
                    alpha=0.15, 
                    zorder=2)
    # ============================
    ax.plot(mean_east, heights, 'o-', color=colors['east']['dark'], 
            linewidth=3, markersize=8,
            markerfacecolor='white',  # 改为 color 则为实心
            markeredgecolor=colors['east']['dark'],
            markeredgewidth=1.5,
            label=f'Mean (n={n_east:,})', zorder=3)
    
    ax.set_xlabel(r'Wind Speed (m$\cdot$s$^{-1}$)', fontweight='bold', fontsize=16)
    ax.set_title('Easterly (Wake)', fontweight='bold', pad=10, fontsize=18)
    ax.grid(True, linestyle='dotted', alpha=0.5, linewidth=0.8, color='gray')
    # ax.legend(loc='lower right', frameon=True, framealpha=0.9)
    legend_elements = [
    Line2D([0], [0], color=colors['east']['dark'], linewidth=3, 
           marker='o', markersize=8, markerfacecolor='white',
           markeredgecolor=colors['east']['dark'], markeredgewidth=1.5,
           label=f'Mean (n={n_east:,})'),
    Patch(facecolor=colors['east']['dark'], alpha=0.15, 
          label='Mean ± 1σ'),
    Line2D([0], [0], color=colors['east']['light'], linewidth=0.8, 
           alpha=0.3, label='Individual profiles')
    ]
    ax.legend(handles=legend_elements, loc='lower right', 
            frameon=True, framealpha=0.9, fontsize=12)
    
    ax.set_xlim(0, 20)
    
    ax.spines['top'].set_visible(True)
    ax.spines['right'].set_visible(True)
    ax.tick_params(axis='both', which='major', labelsize=18, width=1.2, length=5)
    
    # 调整布局
    plt.tight_layout()
    
    # 保存
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    
    print(f"\n✓ 已保存: {output_path}")

if __name__ == "__main__":
    csv_file = '/Users/xiaxin/work/WindForecast_Project/01_Data/processed/matched_data/changma_matched.csv'
    output_path = '/Users/xiaxin/work/WindForecast_Project/03_Results/re-plot-figures/figure-1/final-wind_profiles.png'
    heights = [10, 30, 50, 70]
    
    plot_elegant_profiles(csv_file, heights, output_path)
    print("\n✅ 完成！")