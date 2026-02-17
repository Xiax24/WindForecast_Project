#!/usr/bin/env python3
"""
风速-功率Spearman相关性热图
参考原有代码的风格和配色
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import spearmanr
import matplotlib.colors as mcolors
import os

# 设置风格（与原代码一致）
sns.set_style("whitegrid")
plt.rcParams['figure.dpi'] = 500
plt.rcParams['font.size'] = 28
plt.rcParams['axes.labelsize'] = 28
plt.rcParams['axes.titlesize'] = 28
plt.rcParams['xtick.labelsize'] = 28
plt.rcParams['ytick.labelsize'] = 28
plt.rcParams['legend.fontsize'] = 28
plt.rcParams['figure.titlesize'] = 28
plt.rcParams['font.family'] = 'Arial'

def create_custom_colormap():
    """创建自定义配色（与原代码一致）"""
    colors = ["#A32903", "#F45F31", "white", "#3FC2E7", "#016DB5"]
    n_bins = 256
    custom_cmap = mcolors.LinearSegmentedColormap.from_list('custom', colors, N=n_bins)
    return custom_cmap

def strict_direction_mask(df, heights, direction_range):
    """严格筛选：所有高度都在指定风向区间"""
    min_deg, max_deg = direction_range
    
    # 使用DataFrame的长度
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
            corr, p_val = spearmanr(paired_data[var1], paired_data[var2])
            return corr, p_val
        except:
            return np.nan, 1.0
    else:
        return np.nan, 1.0

def plot_windspeed_power_heatmap_multi(csv_file, heights, output_dir, power_column='power'):
    """绘制三个风向分类的风速-功率相关性热图"""
    
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
    
    # 选择变量：各高度风速 + 功率
    wind_speed_cols = [f'obs_wind_speed_{h}m' for h in heights]
    wind_dir_cols = [f'obs_wind_direction_{h}m' for h in heights]
    selected_vars = wind_speed_cols + [power_column]
    
    # 检查列是否存在
    missing_cols = [col for col in selected_vars if col not in df.columns]
    if missing_cols:
        print(f"❌ 缺失列: {missing_cols}")
        return
    
    # 三种分类
    classifications = {
        'Overall': None,
        'Westerly (Free-stream)': (225, 315),
        'Easterly (Wake)': (45, 135)
    }
    
    # 配色
    colors_map = {
        'Overall': '#2d2d2d',
        'Westerly': '#2e5f8a', 
        'Easterly': '#b63939'
    }
    
    # 创建自定义配色
    custom_cmap = create_custom_colormap()
    
    # 创建图形，手动设置子图位置
    fig = plt.figure(figsize=(24, 7))
    
    # 手动创建三个子图，确保位置一致
    # [left, bottom, width, height]
    ax1 = fig.add_axes([0.18, 0.15, 0.25, 0.70])   # 子图1
    ax2 = fig.add_axes([0.42, 0.15, 0.25, 0.70])   # 子图2  
    ax3 = fig.add_axes([0.684, 0.15, 0.25, 0.70])   # 子图3
    
    axes = [ax1, ax2, ax3]
    
    for idx, (class_name, direction_range) in enumerate(classifications.items()):
        ax = axes[idx]
        
        print(f"\n处理 {class_name}...")
        
        # 筛选数据（直接用df）
        if direction_range is None:
            mask = np.ones(len(df), dtype=bool)

        else:
            mask = strict_direction_mask(df, heights, direction_range)
        
        # 创建筛选后的DataFrame
        df_filtered = df.loc[mask, selected_vars].copy()
        df_clean = df_filtered.dropna()
        
        n_samples = len(df_clean)
        print(f"  有效样本: {n_samples:,}")
        
        if n_samples < 10:
            print(f"  ⚠️ 样本数不足，跳过")
            continue
        
        # 计算相关系数矩阵
        n_vars = len(selected_vars)
        corr_matrix = np.zeros((n_vars, n_vars))
        pval_matrix = np.zeros((n_vars, n_vars))
        
        for i, var1 in enumerate(selected_vars):
            for j, var2 in enumerate(selected_vars):
                if i == j:
                    corr_matrix[i, j] = 1.0
                    pval_matrix[i, j] = 0.0
                else:
                    corr, pval = calculate_spearman_with_pvalue(df_clean, var1, var2)
                    corr_matrix[i, j] = corr
                    pval_matrix[i, j] = pval
        
        # 打印风速-功率相关系数
        print(f"  风速与功率的相关系数:")
        for i, h in enumerate(heights):
            corr = corr_matrix[i, -1]
            pval = pval_matrix[i, -1]
            sig = '***' if pval < 0.001 else '**' if pval < 0.01 else '*' if pval < 0.05 else 'ns'
            print(f"    {h}m: ρ={corr:.4f} {sig}")
        
        # 创建标签（倒序：Power, 70m, 50m, 30m, 10m）
        simplified_labels = ['Power'] + [f'{h}m' for h in reversed(heights)]
        
        # 重新排列矩阵（倒序）
        reversed_order = list(range(n_vars-1, -1, -1))  # [4, 3, 2, 1, 0]
        corr_matrix_reordered = corr_matrix[reversed_order, :][:, reversed_order]
        pval_matrix_reordered = pval_matrix[reversed_order, :][:, reversed_order]
        
        corr_df = pd.DataFrame(corr_matrix_reordered, index=simplified_labels, columns=simplified_labels)
        
        # 创建显著性标注
        annotations = np.empty((n_vars, n_vars), dtype=object)
        for i in range(n_vars):
            for j in range(n_vars):
                corr_val = corr_matrix_reordered[i, j]
                p_val = pval_matrix_reordered[i, j]
                
                if p_val < 0.001:
                    sig_mark = '***'
                elif p_val < 0.01:
                    sig_mark = '**'
                elif p_val < 0.05:
                    sig_mark = '*'
                else:
                    sig_mark = ''
                
                annotations[i, j] = f'{corr_val:.2f}\n{sig_mark}'
        
        # 创建上三角mask
        mask_tri = np.triu(np.ones_like(corr_df, dtype=bool))
        # 额外隐藏Power行（第一行全部）
        mask_tri[0, :] = True

        # 额外隐藏10m列（最后一列全部）
        mask_tri[:, -1] = True
        # 绘制热图（反转colormap，只在最后一个子图显示colorbar）
        sns.heatmap(corr_df, 
                    mask=mask_tri, 
                    annot=annotations, 
                    cmap=custom_cmap.reversed(),  # 反转配色
                    center=0.75,  # 中心点设为0.75
                    square=True, 
                    fmt='',
                    vmin=0.5, 
                    vmax=1.0,
                    cbar=(idx == 2),  # 只在第三个子图显示colorbar
                    cbar_kws={
                        "shrink": .8,
                    } if idx == 2 else None,
                    annot_kws={
                        'size': 28,
                        'weight': 'normal'
                    },
                    ax=ax)
        
        # 设置标题（使用默认位置即可）
        ax.set_title(f'{class_name}', 
                    fontsize=28, fontweight='bold', 
                    pad=15,
                    color='black')
                    # color=colors_map[class_name])
        
        # 设置刻度
        ax.tick_params(axis='x', which='major', labelsize=28, rotation=45, labelcolor='black')
        ax.tick_params(axis='y', which='major', labelsize=28, rotation=0, labelcolor='black')
        ax.grid(False)
        # 自定义colorbar（只在第三个子图）
        if idx == 2:
            cbar = ax.collections[0].colorbar
            cbar.ax.tick_params(labelsize=28)
            cbar.set_label('Spearman ρ', 
                          fontsize=28, fontweight='normal', labelpad=15)
    
    # 去除网格
    plt.grid(False)
    
    # 不需要调整布局，因为已经手动设置了子图位置
    
    # 保存
    os.makedirs(output_dir, exist_ok=True)
    output_png = os.path.join(output_dir, 'windspeed_power_correlation_heatmap_all.png')
    output_pdf = os.path.join(output_dir, 'windspeed_power_correlation_heatmap_all.pdf')
    
    plt.savefig(output_png, dpi=500, bbox_inches='tight', facecolor='white')
    plt.savefig(output_pdf, bbox_inches='tight', facecolor='white')
    plt.close()
    
    print(f"\n✓ 热图已保存:")
    print(f"  PNG: {output_png}")
    print(f"  PDF: {output_pdf}")

if __name__ == "__main__":
    csv_file = '/Users/xiaxin/work/WindForecast_Project/01_Data/processed/matched_data/changma_matched.csv'
    output_dir = '/Users/xiaxin/work/WindForecast_Project/03_Results/re-plot-figures/figure-1'
    heights = [10, 30, 50, 70]
    power_column = 'power'
    
    plot_windspeed_power_heatmap_multi(csv_file, heights, output_dir, power_column)
    print("\n✅ 完成！")