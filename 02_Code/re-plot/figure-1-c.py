#!/usr/bin/env python3
"""
风速-功率相关性廓线图（竖向）
与风速廓线图形成视觉联动
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
import os

# 与风廓线一致的配置
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

def strict_direction_mask(df, heights, direction_range):
    """严格筛选：所有高度都在指定风向区间"""
    min_deg, max_deg = direction_range
    mask = np.ones(len(df), dtype=bool)
    
    for h in heights:
        wd_col = f'obs_wind_direction_{h}m'
        if wd_col not in df.columns:
            continue
        
        wd = df[wd_col].values
        
        if min_deg > max_deg:
            h_mask = (wd >= min_deg) | (wd <= max_deg)
        else:
            h_mask = (wd >= min_deg) & (wd <= max_deg)
        
        h_mask = h_mask & ~np.isnan(wd)
        mask = mask & h_mask
    
    return mask

def calculate_spearman_with_pvalue(df, var1, var2):
    """计算Spearman相关系数和p值"""
    paired_data = df[[var1, var2]].dropna()
    if len(paired_data) > 3:
        try:
            corr, p_val = stats.spearmanr(paired_data[var1], paired_data[var2])
            return corr, p_val
        except:
            return np.nan, 1.0
    else:
        return np.nan, 1.0

def plot_correlation_profile(csv_file, heights, output_path, power_column='power'):
    """绘制竖向相关性廓线图"""
    
    print("加载数据...")
    df = pd.read_csv(csv_file)
    
    # 检查功率列
    if power_column not in df.columns:
        power_cols = [col for col in df.columns if 'power' in col.lower()]
        if power_cols:
            power_column = power_cols[0]
        else:
            print("❌ 未找到功率列")
            return
    
    print(f"✓ 使用功率列: {power_column}")
    
    # 筛选数据
    print("筛选数据...")
    mask_all = np.ones(len(df), dtype=bool)
    for h in heights:
        wd_col = f'obs_wind_direction_{h}m'
        if wd_col in df.columns:
            mask_all = mask_all & ~np.isnan(df[wd_col].values)
    
    mask_west = strict_direction_mask(df, heights, (225, 315))
    mask_east = strict_direction_mask(df, heights, (45, 135))
    
    # 计算相关系数
    print("计算相关系数...")
    corr_all = []
    pval_all = []
    corr_west = []
    pval_west = []
    corr_east = []
    pval_east = []
    
    for h in heights:
        ws_col = f'obs_wind_speed_{h}m'
        
        # Overall
        df_all = df.loc[mask_all, [ws_col, power_column]].dropna()
        if len(df_all) > 10:
            corr, pval = calculate_spearman_with_pvalue(df_all, ws_col, power_column)
            corr_all.append(corr)
            pval_all.append(pval)
        else:
            corr_all.append(np.nan)
            pval_all.append(1.0)
        
        # 西风
        df_west = df.loc[mask_west, [ws_col, power_column]].dropna()
        if len(df_west) > 10:
            corr, pval = calculate_spearman_with_pvalue(df_west, ws_col, power_column)
            corr_west.append(corr)
            pval_west.append(pval)
        else:
            corr_west.append(np.nan)
            pval_west.append(1.0)
        
        # 东风
        df_east = df.loc[mask_east, [ws_col, power_column]].dropna()
        if len(df_east) > 10:
            corr, pval = calculate_spearman_with_pvalue(df_east, ws_col, power_column)
            corr_east.append(corr)
            pval_east.append(pval)
        else:
            corr_east.append(np.nan)
            pval_east.append(1.0)
    
    corr_all = np.array(corr_all)
    corr_west = np.array(corr_west)
    corr_east = np.array(corr_east)
    pval_all = np.array(pval_all)
    pval_west = np.array(pval_west)
    pval_east = np.array(pval_east)
    
    # 统计
    n_all = np.sum(mask_all)
    n_west = np.sum(mask_west)
    n_east = np.sum(mask_east)
    
    print(f"\n样本统计:")
    print(f"  整体: {n_all:,}")
    print(f"  西风: {n_west:,}")
    print(f"  东风: {n_east:,}")
    
    # 打印相关系数
    print(f"\n相关系数:")
    print(f"  高度   整体      西风      东风")
    for i, h in enumerate(heights):
        sig_a = '***' if pval_all[i] < 0.001 else '**' if pval_all[i] < 0.01 else '*' if pval_all[i] < 0.05 else 'ns'
        sig_w = '***' if pval_west[i] < 0.001 else '**' if pval_west[i] < 0.01 else '*' if pval_west[i] < 0.05 else 'ns'
        sig_e = '***' if pval_east[i] < 0.001 else '**' if pval_east[i] < 0.01 else '*' if pval_east[i] < 0.05 else 'ns'
        print(f"  {h:2d}m  {corr_all[i]:.4f}{sig_a:>4s}  {corr_west[i]:.4f}{sig_w:>4s}  {corr_east[i]:.4f}{sig_e:>4s}")
    
    # 配色（与风廓线一致）
    colors = {
        'all': '#2d2d2d',   # 深灰（Overall）
        'west': "#f41111",  # 红色（Westerly）
        'east': "#1996de",  # 蓝色（Easterly）
    }
    
    # 创建图形（竖长条，与风廓线子图尺寸一致）
    fig, ax = plt.subplots(figsize=(4, 5))
    
    # 找出每条线的最强相关性点的索引
    idx_all_max = np.nanargmax(corr_all)
    idx_west_max = np.nanargmax(corr_west)
    idx_east_max = np.nanargmax(corr_east)
    
    # 绘制Overall相关性曲线（灰色）
    # 先画线
    ax.plot(corr_all, heights, '-', 
            color=colors['all'], 
            linewidth=3, 
            zorder=2)
    # 再画空心点
    for i, (corr, h) in enumerate(zip(corr_all, heights)):
        if i == idx_all_max:
            # 最强点：实心
            ax.plot(corr, h, 'o',
                   color=colors['all'],
                   markersize=8,
                   markerfacecolor=colors['all'],
                   markeredgecolor=colors['all'],
                   markeredgewidth=1.5,
                   zorder=3)
        else:
            # 其他点：空心
            ax.plot(corr, h, 'o',
                   color=colors['all'],
                   markersize=8,
                   markerfacecolor='white',
                   markeredgecolor=colors['all'],
                   markeredgewidth=1.5,
                   zorder=3)
    
    # 添加图例项（只用于显示）
    ax.plot([], [], 'o-', 
            color=colors['all'], 
            linewidth=3, 
            markersize=8,
            markerfacecolor='white',
            markeredgecolor=colors['all'],
            markeredgewidth=1.5,
            label=f'All Data')
    
    # 绘制东风相关性曲线（蓝色）
    ax.plot(corr_east, heights, '-', 
            color=colors['east'], 
            linewidth=3, 
            zorder=2)
    for i, (corr, h) in enumerate(zip(corr_east, heights)):
        if i == idx_east_max:
            ax.plot(corr, h, 'o',
                   color=colors['east'],
                   markersize=8,
                   markerfacecolor=colors['east'],
                   markeredgecolor=colors['east'],
                   markeredgewidth=1.5,
                   zorder=3)
        else:
            ax.plot(corr, h, 'o',
                   color=colors['east'],
                   markersize=8,
                   markerfacecolor='white',
                   markeredgecolor=colors['east'],
                   markeredgewidth=1.5,
                   zorder=3)
    
    ax.plot([], [], 'o-', 
            color=colors['east'], 
            linewidth=3, 
            markersize=8,
            markerfacecolor='white',
            markeredgecolor=colors['east'],
            markeredgewidth=1.5,
            label=f'Wake (Easterly)')

    
    # 绘制西风相关性曲线（红色）
    ax.plot(corr_west, heights, '-', 
            color=colors['west'], 
            linewidth=3, 
            zorder=2)
    for i, (corr, h) in enumerate(zip(corr_west, heights)):
        if i == idx_west_max:
            ax.plot(corr, h, 'o',
                   color=colors['west'],
                   markersize=8,
                   markerfacecolor=colors['west'],
                   markeredgecolor=colors['west'],
                   markeredgewidth=1.5,
                   zorder=3)
        else:
            ax.plot(corr, h, 'o',
                   color=colors['west'],
                   markersize=8,
                   markerfacecolor='white',
                   markeredgecolor=colors['west'],
                   markeredgewidth=1.5,
                   zorder=3)
    
    ax.plot([], [], 'o-', 
            color=colors['west'], 
            linewidth=3, 
            markersize=8,
            markerfacecolor='white',
            markeredgecolor=colors['west'],
            markeredgewidth=1.5,
            label=f'Free-stream (Westerly)')
    
    # # 绘制东风相关性曲线（蓝色）
    # ax.plot(corr_east, heights, '-', 
    #         color=colors['east'], 
    #         linewidth=3, 
    #         zorder=2)
    # for i, (corr, h) in enumerate(zip(corr_east, heights)):
    #     if i == idx_east_max:
    #         ax.plot(corr, h, 'o',
    #                color=colors['east'],
    #                markersize=8,
    #                markerfacecolor=colors['east'],
    #                markeredgecolor=colors['east'],
    #                markeredgewidth=1.5,
    #                zorder=3)
    #     else:
    #         ax.plot(corr, h, 'o',
    #                color=colors['east'],
    #                markersize=8,
    #                markerfacecolor='white',
    #                markeredgecolor=colors['east'],
    #                markeredgewidth=1.5,
    #                zorder=3)
    
    # ax.plot([], [], 'o-', 
    #         color=colors['east'], 
    #         linewidth=3, 
    #         markersize=8,
    #         markerfacecolor='white',
    #         markeredgecolor=colors['east'],
    #         markeredgewidth=1.5,
    #         label=f'Wake (Easterly)')
    
    # 坐标轴设置
    # ax.set_xlabel(r'Spearman Correlation', fontweight='bold', fontsize=18)
    ax.set_ylabel('Height (m)', fontweight='bold', fontsize=18)
    ax.set_title('Spearman Correlation', fontweight='bold', pad=10, fontsize=19)
    
    # Y轴：与风廓线一致
    ax.set_ylim(0, max(heights) + 18)
    ax.set_yticks(np.arange(0, max(heights) + 19, 10))
    ax.set_yticklabels([str(h) if h in heights else '' for h in np.arange(0, max(heights) + 18, 10)])
    
    # X轴范围：建议0.70-0.90
    all_corrs = np.concatenate([corr_all, corr_west, corr_east])
    x_min = max(0.70, np.floor(np.nanmin(all_corrs) * 20) / 20 - 0.02)
    x_max = min(0.90, np.ceil(np.nanmax(all_corrs) * 20) / 20 + 0.02)
    ax.set_xlim(x_min, x_max)
    
    # 网格和框线（与风廓线一致）
    ax.grid(True, linestyle='dotted', alpha=0.5, linewidth=0.8, color='gray')
    ax.spines['top'].set_visible(True)
    ax.spines['right'].set_visible(True)
    ax.spines['bottom'].set_visible(True)
    ax.spines['left'].set_visible(True)
    
    # Tick设置（向内）
    ax.tick_params(axis='both', which='major', labelsize=18, width=1.2, length=5, direction='in')
    
    # 图例
    ax.legend(loc='upper right', frameon=False, framealpha=0.9, fontsize=11,markerfirst=False)
    
    # 调整布局
    plt.tight_layout()
    
    # 保存
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    
    print(f"\n✓ 已保存: {output_path}")

if __name__ == "__main__":
    csv_file = '/Users/xiaxin/work/WindForecast_Project/01_Data/processed/matched_data/changma_matched.csv'
    output_path = '/Users/xiaxin/work/WindForecast_Project/03_Results/re-plot-figures/figure-1/correlation_profile.pdf'
    heights = [10, 30, 50, 70]
    power_column = 'power'
    
    plot_correlation_profile(csv_file, heights, output_path, power_column)
    print("\n✅ 完成！")