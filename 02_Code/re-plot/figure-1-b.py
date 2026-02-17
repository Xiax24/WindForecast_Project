#!/usr/bin/env python3
"""
顶刊级别风速廓线
- 展示所有样本廓线（透明）
- 叠加平均廓线（加粗）
- 第三张子图叠加西风线条用于对比
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
    获取有效的风廓线样本用于绘图
    返回：采样的profiles（用于背景透明线）
    """
    indices = np.where(mask)[0]
    
    # 如果样本太多，随机采样用于绘图
    if len(indices) > max_samples:
        sampled_indices = np.random.choice(indices, max_samples, replace=False)
    else:
        sampled_indices = indices
    
    profiles = []
    for idx in sampled_indices:
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

def calculate_mean_std(data, heights, mask):
    """
    计算所有有效数据的平均值和标准差
    用于统计，不受采样影响
    """
    indices = np.where(mask)[0]
    
    # 收集所有有效廓线
    all_profiles = []
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
            all_profiles.append(profile)
    
    all_profiles = np.array(all_profiles)
    
    # 基于全部数据计算
    mean = np.nanmean(all_profiles, axis=0)
    std = np.nanstd(all_profiles, axis=0)
    
    return mean, std

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
    
    mask_west = strict_direction_mask(data, heights, (225, 315))
    mask_east = strict_direction_mask(data, heights, (45, 135))
    
    # 获取样本廓线（用于绘图，采样）
    print("提取廓线样本用于绘图...")
    profiles_all = get_valid_profiles(data, heights, mask_all, max_samples=200)
    profiles_west = get_valid_profiles(data, heights, mask_west, max_samples=150)
    profiles_east = get_valid_profiles(data, heights, mask_east, max_samples=150)
    
    # 计算平均和标准差（基于所有数据，不采样）
    print("计算统计量（基于所有数据）...")
    mean_all, std_all = calculate_mean_std(data, heights, mask_all)
    mean_west, std_west = calculate_mean_std(data, heights, mask_west)
    mean_east, std_east = calculate_mean_std(data, heights, mask_east)

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
    
    # 配色
    # colors = {
    #     'all': {
    #         'light': "#686767",
    #         'dark': '#2d2d2d',
    #     },
    #     'east': {
    #         'light': "#74b0e8",
    #         'dark': "#1c68a9",
    #     },
    #     'west': {
    #         'light': "#F45F31",
    #         'dark': "#E83E0B",
    #     }
    # }
    colors = {
        'all': {
            'light': "#686767",      # 浅灰（透明廓线）
            'dark': '#2d2d2d',       # 深灰（平均线）
        },
        'east': {
            'light': '#6fa8dc',      # 浅蓝（透明廓线）
            'dark': "#1996de",      # 深蓝（平均线）
        },
        'west': {
            'light': "#ed6e6e",      # 浅红（透明廓线）
            'dark': "#f41111",       # 深红（平均线）
        }
    }
    
    # 创建图形：2个子图 + 1个图例区域
    fig = plt.figure(figsize=(12, 5))
    
    # 手动创建子图位置 [left, bottom, width, height]
    ax1 = fig.add_axes([0.08, 0.15, 0.28, 0.75])  # 子图1：Overall
    ax2 = fig.add_axes([0.46, 0.15, 0.28, 0.75])  # 子图2：Easterly (原子图3)
    ax_legend = fig.add_axes([0.69, 0.15, 0.20, 0.75])  # 图例区域
    ax_legend.axis('off')  # 隐藏坐标轴
    
    axes = [ax1, ax2]
    
    # 子图1：整体
    ax = axes[0]
    for profile in profiles_all:
        ax.plot(profile, heights, '-', color=colors['all']['light'], 
                alpha=0.08, linewidth=0.8, zorder=1)
    
    ax.fill_betweenx(heights, 
                    mean_all - std_all,
                    mean_all + std_all,
                    color=colors['all']['dark'], 
                    alpha=0.15,
                    zorder=2)

    ax.plot(mean_all, heights, 'o-', color=colors['all']['dark'], 
            linewidth=3, markersize=8, 
            markerfacecolor='white',
            markeredgecolor=colors['all']['dark'],
            markeredgewidth=1.5,
            zorder=3)

    ax.set_xlabel(r'Wind Speed (m$\cdot$s$^{-1}$)', fontweight='bold', fontsize=16)
    ax.set_ylabel('Height (m)', fontweight='bold', fontsize=16)
    ax.set_title('All Data ', fontweight='bold', pad=10, fontsize=18)
    ax.grid(True, linestyle='dotted', alpha=0.5, linewidth=0.8, color='gray')
    
    # 保存图例元素（不显示，稍后统一显示在右侧）
    from matplotlib.patches import Patch
    from matplotlib.lines import Line2D
    
    # legend_elements_overall = [
    #     Line2D([0], [0], color=colors['all']['dark'], linewidth=3, 
    #         marker='o', markersize=8, markerfacecolor='white',
    #         markeredgecolor=colors['all']['dark'], markeredgewidth=1.5,
    #         label=f'Overall Mean'),
    #     Patch(facecolor=colors['all']['dark'], alpha=0.15, 
    #         label='Overall ± 1σ'),
    #     Line2D([0], [0], color=colors['all']['light'], linewidth=0.8, 
    #         alpha=0.3, label='Overall profiles')
    # ]
    # legend_elements_overall = [
    # Line2D([0], [0], color='black', linewidth=3, 
    #        marker='o', markersize=8, markerfacecolor='white',
    #        markeredgecolor='black', markeredgewidth=1.5,
    #        label='All Data')  # 简单明了，代表整个黑色系
    # ]
    legend_overall = [
        Line2D([0], [0], color='black', linewidth=3, 
            marker='o', markersize=8, markerfacecolor='white',
            markeredgecolor='black', markeredgewidth=1.5,
            label='All Data')
    ]

    # 直接放在 ax1 内部 (通常左上角是空的，适合放图例)
    ax1.legend(handles=legend_overall, 
            loc='lower right',    # 放在左上角空白处
            frameon=False,       # 去掉边框，融为一体
            fontsize=14, markerfirst=False)        # 字号大一点
    ax.set_ylim(0, max(heights) + 1)
    ax.set_yticks(np.arange(0, max(heights) + 1, 10))
    ax.set_yticklabels([str(h) if h in heights else '' for h in np.arange(0, max(heights) + 1, 10)])
    ax.set_xlim(0, 20)
    ax.spines['bottom'].set_visible(True)
    ax.spines['right'].set_visible(True)
    ax.tick_params(axis='both', which='major', labelsize=18, width=1.2, length=5, direction='in')
    
    # 子图2（原子图3）：东风（尾流） + 完整叠加西风（自由流）
    ax = axes[1]
    
    # ========== 先绘制西风的所有元素（作为对比背景）==========
    # 西风透明线
    for profile in profiles_west:
        ax.plot(profile, heights, '-', color=colors['west']['light'], 
                alpha=0.08, linewidth=0.8, zorder=1)
    
    # 西风阴影
    ax.fill_betweenx(heights, 
                    mean_west - std_west, 
                    mean_west + std_west,
                    color=colors['west']['dark'], 
                    alpha=0.12,
                    zorder=2)
    
    # 西风平均线
    ax.plot(mean_west, heights, 'o-', color=colors['west']['dark'], 
            linewidth=3, markersize=8,
            markerfacecolor='white',
            markeredgecolor=colors['west']['dark'],
            markeredgewidth=1.5,
            zorder=4)
    # =======================================================
    
    # ========== 再绘制东风的所有元素（前景）==========
    # 东风透明线
    for profile in profiles_east:
        ax.plot(profile, heights, '-', color=colors['east']['light'], 
                alpha=0.12, linewidth=0.8, zorder=3)
    
    # 东风阴影
    ax.fill_betweenx(heights, 
                    mean_east - std_east, 
                    mean_east + std_east,
                    color=colors['east']['dark'], 
                    alpha=0.15, 
                    zorder=5)
    
    # 东风平均线
    ax.plot(mean_east, heights, 'o-', color=colors['east']['dark'], 
            linewidth=3, markersize=8,
            markerfacecolor='white',
            markeredgecolor=colors['east']['dark'],
            markeredgewidth=1.5,
            zorder=6)
    # =================================================
    
    ax.set_xlabel(r'Wind Speed (m$\cdot$s$^{-1}$)', fontweight='bold', fontsize=16)
    ax.set_title('Free-stream vs. Wake Regime', fontweight='bold', pad=10, fontsize=18)
    ax.grid(True, linestyle='dotted', alpha=0.5, linewidth=0.8, color='gray')
    ax.set_ylabel('Height (m)', fontweight='bold', fontsize=16)
    # 保存图例元素（不显示，稍后统一显示在右侧）
    # legend_elements_easterly = [
    #     Line2D([0], [0], color=colors['east']['dark'], linewidth=3, 
    #            marker='o', markersize=8, markerfacecolor='white',
    #            markeredgecolor=colors['east']['dark'], markeredgewidth=1.5,
    #            label=f'Wake (Easterly) Mean'),
    #     Patch(facecolor=colors['east']['dark'], alpha=0.15, 
    #           label='Wake ± 1σ'),
    #     Line2D([0], [0], color=colors['east']['light'], linewidth=0.8, 
    #            alpha=0.3, label='Wake profiles'),
    #     Line2D([0], [0], color=colors['west']['dark'], linewidth=3, 
    #            marker='o', markersize=8, markerfacecolor='white',
    #            markeredgecolor=colors['west']['dark'], markeredgewidth=1.5,
    #            label=f'Free-stream (Westerly) Mean'),
    #     Patch(facecolor=colors['west']['dark'], alpha=0.12, 
    #           label='Free-stream ± 1σ'),
    #     Line2D([0], [0], color=colors['west']['light'], linewidth=0.8, 
    #            alpha=0.3, label='Free-stream profiles')
    # ]
    # legend_elements_comparison = [
    # # Free-stream (Westerly)
    # Line2D([0], [0], color=colors['west']['dark'], linewidth=3, 
    #        marker='o', markersize=8, markerfacecolor='white',
    #        markeredgecolor=colors['west']['dark'], markeredgewidth=1.5,
    #        label='Free-stream (Westerly)'),
           
    # # Wake (Easterly)
    # Line2D([0], [0], color=colors['east']['dark'], linewidth=3, 
    #        marker='o', markersize=8, markerfacecolor='white',
    #        markeredgecolor=colors['east']['dark'], markeredgewidth=1.5,
    #        label='Wake (Easterly)')
    # ]
    legend_compare = [
        # # Free-stream (Westerly)
        # Line2D([0], [0], color=colors['west']['dark'], linewidth=3, 
        #     marker='o', markersize=8, markerfacecolor='white',
        #     markeredgecolor=colors['west']['dark'], markeredgewidth=1.5,
        #     label='Free-stream (Westerly)'),
            
        # Wake (Easterly)
        Line2D([0], [0], color=colors['east']['dark'], linewidth=3, 
            marker='o', markersize=8, markerfacecolor='white',
            markeredgecolor=colors['east']['dark'], markeredgewidth=1.5,
            label='Wake (Easterly)'),
        
        # Free-stream (Westerly)
        Line2D([0], [0], color=colors['west']['dark'], linewidth=3, 
            marker='o', markersize=8, markerfacecolor='white',
            markeredgecolor=colors['west']['dark'], markeredgewidth=1.5,
            label='Free-stream (Westerly)')
    ]
    # 直接放在 ax2 内部
    ax2.legend(handles=legend_compare, 
            loc='lower right',    # 放在左上角空白处
            frameon=False, 
            fontsize=13,labelspacing=0.14, markerfirst=False)

    ax.set_xlim(3, 12)
    ax.set_xticks(np.arange(3, 12.1, 1))
    ax.set_ylim(0, max(heights)+1)
    ax.set_yticks(np.arange(0, max(heights)+1, 10))
    ax.set_yticklabels([str(h) if h in heights else '' for h in np.arange(0, max(heights)+1, 10)])
    ax.spines['bottom'].set_visible(True)
    ax.spines['right'].set_visible(True)
    ax.tick_params(axis='both', which='major', labelsize=18, width=1.2, length=5, direction='in')
  
    # 保存
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    
    print(f"\n✓ 已保存: {output_path}")

if __name__ == "__main__":
    csv_file = '/Users/xiaxin/work/WindForecast_Project/01_Data/processed/matched_data/changma_matched.csv'
    output_path = '/Users/xiaxin/work/WindForecast_Project/03_Results/re-plot-figures/figure-1/final-wind_profiles.pdf'
    heights = [10, 30, 50, 70]
    
    plot_elegant_profiles(csv_file, heights, output_path)
    print("\n✅ 完成！")