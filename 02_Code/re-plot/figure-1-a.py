#!/usr/bin/env python3
"""
风速-风向玫瑰图 - 图例文字可调版
专门解决图例文字 [0.0:4.0) 等标签的大小问题
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from windrose import WindroseAxes
import os

COLOR_SCHEMES = {
    'classic_blue': ['#deebf7', '#c6dbef', '#9ecae1', '#6baed6', '#4292c6', '#2171b5', '#084594'],
    'ocean_blue': ['#e0f3f8', '#abd9e9', '#74add1', '#4575b4', '#313695', '#1a237e', '#0d1642'],
    'viridis': ['#fde725', '#90d743', '#35b779', '#21918c', '#31688e', '#443983', '#440154'],
    'viridis_r': ['#440154', '#443983', '#31688e', '#21918c', '#35b779', '#90d743', '#fde725']
}

def plot_windrose_custom_legend(csv_file, ws_col, wd_col, output_path, 
                                percentile_cutoff=99.5, 
                                bin_width=4, 
                                color_scheme='classic_blue',
                                legend_text_size=20,      # 图例文字大小 ⭐
                                legend_title_size=20,     # 图例标题大小 ⭐
                                legend_box_size=10,       # 色块大小 ⭐
                                title_size=50,            # 主标题大小
                                tick_size=24):            # 刻度大小
    """
    绘制风玫瑰图 - 完全自定义图例
    
    Parameters:
    -----------
    legend_text_size : int
        图例标签文字大小，如 [0.0:4.0)，推荐 16-20
    legend_title_size : int
        图例标题 "Wind Speed (m/s)" 大小，推荐 18-22
    legend_box_size : int
        图例色块大小，推荐 30-40
    """
    
    # 基础配置
    plt.rcParams.update({
        'font.family': 'Arial',
        'font.size': 20,
        'axes.labelsize': 20,
        'axes.titlesize': title_size,
        'xtick.labelsize': tick_size,
        'ytick.labelsize': tick_size,
        'lines.linewidth': 2,
        'axes.linewidth': 2,
        'figure.dpi': 300,
        'savefig.dpi': 300
    })
    
    print(f"\n{'='*60}")
    print(f"自定义图例风玫瑰图")
    print(f"{'='*60}")
    
    # 读取数据
    data = pd.read_csv(csv_file)
    ws = data[ws_col].values
    wd = data[wd_col].values
    
    valid = ~(np.isnan(ws) | np.isnan(wd)) & (ws >= 0) & (ws < 50) & (wd >= 0) & (wd < 360)
    ws = ws[valid]
    wd = wd[valid]
    
    print(f"✓ 数据: {len(ws):,} 点")
    print(f"  风速: {ws.min():.1f} - {ws.max():.1f} m/s (均值: {ws.mean():.2f})")
    
    # Bins
    if percentile_cutoff < 100:
        cutoff = np.percentile(ws, percentile_cutoff)
    else:
        cutoff = ws.max()
    
    upper = int(np.ceil(cutoff / bin_width) * bin_width)
    bins = list(range(0, upper + 1, bin_width))
    print(f"  Bins: {bins}")
    
    # 创建图形
    fig = plt.figure(figsize=(12, 11))
    ax = fig.add_subplot(111, projection='windrose')
    
    # 颜色
    base_colors = COLOR_SCHEMES[color_scheme]
    n = len(bins)
    if n <= len(base_colors):
        colors = base_colors[:n]
    else:
        from matplotlib.colors import LinearSegmentedColormap
        cmap = LinearSegmentedColormap.from_list('custom', base_colors, N=n)
        colors = [cmap(i/n) for i in range(n)]
        colors = ['#%02x%02x%02x' % tuple(int(c*255) for c in color[:3]) for color in colors]
    
    # 绘制
    ax.bar(wd, ws, 
           normed=True,
           opening=0.9,
           edgecolor='black',
           linewidth=1,
           bins=bins,
           colors=colors,
           nsector=16)
    
    # ========== 自定义图例 ==========
    legend = ax.set_legend(
        title='Wind Speed (m$\cdot$s$^{-1}$)', 
        loc='upper left',
        bbox_to_anchor=(1.03, 0.5),
        frameon=False,
        fancybox=False,
        shadow=False,
        borderpad=1.2,
        labelspacing=0.8,
        handlelength=2.5,
        handleheight=1.8
    )
    
    # 关键：手动设置每个文字的大小
    print(f"\n✓ 图例设置:")
    for i, text in enumerate(legend.get_texts()):
        text.set_fontsize(legend_text_size)  # ⭐ 这里设置 [0.0:4.0) 的大小
        if i == 0:
            print(f"  文字 '{text.get_text()}' 大小: {legend_text_size}pt")
    
    # 设置标题
    legend.get_title().set_fontsize(legend_title_size)
    print(f"  标题 'Wind Speed (m/s)' 大小: {legend_title_size}pt")
    
    # 设置色块
    for patch in legend.get_patches():
        patch.set_height(25)
        patch.set_width(50)
    print(f"  色块尺寸: {legend_box_size}×{legend_box_size}")
    
    # 主标题
    if 'm' in ws_col:
        height = ws_col.split('_')[-1].replace('m', '')
        title = f'Wind Rose at Hub Height'
    else:
        title = 'Wind Rose'
    
    ax.set_title(title, fontsize=title_size, fontweight='bold', pad=170)
    
    # Y轴
    ax.set_yticks(np.arange(5, 26, 5))
    ax.set_yticklabels([f'{i}%' for i in range(5, 26, 5)], fontsize=tick_size)
    
    # ========== 方位标签 (N, S-W, E 等) ⭐ ==========
    ax.tick_params(axis='x', 
                   labelsize=40,  # 字体大小
                   pad=30)         # 距离
    
    
    ax.grid(True, linewidth=3, alpha=0.4)
    
    # 保存
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, bbox_inches='tight', dpi=300, 
                facecolor='white', edgecolor='none')
    
    print(f"\n✓ 已保存: {output_path}")
    print(f"{'='*60}\n")
    
    plt.close()

# ========== 使用示例 ==========
if __name__ == "__main__":
    csv_file = '/Users/xiaxin/work/WindForecast_Project/01_Data/processed/matched_data/changma_matched.csv'
    ws_col = 'obs_wind_speed_70m'
    wd_col = 'obs_wind_direction_70m'
    output_dir = '/Users/xiaxin/work/WindForecast_Project/03_Results/re-plot-figures/figure-1'
    
    # 方案1：推荐配置
    plot_windrose_custom_legend(
        csv_file, ws_col, wd_col,
        output_path=f'{output_dir}/windrose_70m_final.pdf',
        percentile_cutoff=99.5,
        color_scheme='ocean_blue',
        legend_text_size=24,     # [0.0:4.0) 文字大小 ⭐
        legend_title_size=25,    # "Wind Speed" 标题大小 ⭐
        legend_box_size=16,      # 色块大小 ⭐
        title_size=55,
        tick_size=35
    )