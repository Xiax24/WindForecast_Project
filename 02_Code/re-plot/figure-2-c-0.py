#!/usr/bin/env python3
"""
Figure 2c: 垂直 EOF 第一模态 (Vertical EOF Mode 1 Profiles)
对比 Free-stream 和 Wake 条件下的垂直结构一致性
完全可配置版本：所有视觉参数都可以调整
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import os
import warnings
warnings.filterwarnings('ignore')

# ==================== 全局配置区 ====================
# ========== 1. 路径配置 ==========
data_path = '/Users/xiaxin/work/WindForecast_Project/01_Data/processed/matched_data/changma_matched.csv'
output_dir = '/Users/xiaxin/work/WindForecast_Project/03_Results/re-plot-figures/figure-2/'

# ========== 2. 图形尺寸配置 ==========
FIGURE_CONFIG = {
    'figsize': (8, 8),           # 图形尺寸 (宽, 高)
    'dpi': 300,                  # 分辨率
}

# ========== 3. 颜色配置 ==========
COLORS = {
    'all': {
        'line': '#686767',       # All Data 线条颜色
        'marker_edge': '#686767', # All Data 标记边缘颜色
        'marker_face': 'white',   # All Data 标记填充颜色
    },
    'west': {  # Free-stream (Westerly)
        'line': "#f41111",        # 深红色
        'marker_edge': "#f41111",
        'marker_face': 'white',
    },
    'east': {  # Wake (Easterly)
        'line': "#1996de",        # 深蓝色
        'marker_edge': "#1996de",
        'marker_face': 'white',
    },
    'zero_line': 'gray',          # 零参考线颜色
    'grid': 'gray',               # 网格颜色
}

# ========== 4. 线条和标记配置 ==========
LINE_CONFIG = {
    'all': {
        'linewidth': 3,
        'markersize': 10,
        'marker': 'o',            # 圆圈
        'markeredgewidth': 2,
        'zorder': 2,
        'alpha': 1.0,
    },
    'west': {
        'linewidth': 3,
        'markersize': 10,
        'marker': 'o',            # 圆圈
        'markeredgewidth': 2,
        'zorder': 3,
        'alpha': 1.0,
    },
    'east': {
        'linewidth': 3,
        'markersize': 10,
        'marker': 'o',            # 方块
        'markeredgewidth': 2,
        'zorder': 3,
        'alpha': 1.0,
    },
    'zero_line': {
        'linewidth': 1.2,
        'linestyle': '--',
        'alpha': 0.5,
        'zorder': 1,
    }
}

# ========== 5. 坐标轴配置 ==========
AXIS_CONFIG = {
    'xlabel': {
        'text': 'EOF1 Loading',
        'fontsize': 22,
        'fontweight': 'normal',
        'labelpad': 10,
    },
    'ylabel': {
        'text': r'Height $z$ (m)',
        'fontsize': 22,
        'fontweight': 'normal',
        'labelpad': 18,
    },
    'title': {
        'text': 'First Vertical EOF Mode (EOF1)',
        'fontsize': 21,
        'fontweight': 'bold',
        'pad': 12,
    },
    
    # ========== X轴配置 ==========
    'xlim': (0.4, 0.6),           # X轴范围 (min, max) 或 None (自动)
    'xticks': None,               # X轴刻度列表 或 None (自动生成)
                                  # 例如: [0.40, 0.45, 0.50, 0.55, 0.60]
    'xtick_interval': 0.05,       # X轴刻度间隔 (仅在xticks=None时生效)
                                  # 例如: 0.05 表示每隔0.05一个刻度
    'xtick_labels': None,         # X轴刻度标签 或 None (使用默认数值)
                                  # 例如: ['0.40', '0.45', '0.50', '0.55', '0.60']
    'xtick_rotation': 0,          # X轴刻度标签旋转角度 (0, 45, 90等)
    'xtick_format': None,         # X轴刻度格式化 或 None (默认)
                                  # 例如: '%.2f' 表示保留2位小数
    
    # ========== Y轴配置 ==========
    'ylim': None,                 # Y轴范围 (min, max) 或 None (自动)
                                  # None表示使用 (0, max(heights) + ylim_offset)
    'ylim_offset': 5,             # Y轴上限相对最大高度的偏移 (仅当ylim=None时生效)
    'yticks': None,               # Y轴刻度列表 或 None (使用heights)
                                  # 例如: [0, 10, 20, 30, 40, 50, 60, 70]
    'ytick_labels': None,         # Y轴刻度标签 或 None (自动生成)
                                  # 例如: ['0 m', '10 m', '20 m', ...]
    'ytick_interval': None,       # Y轴刻度间隔 或 None (不使用等间隔)
                                  # 例如: 10 表示每隔10一个刻度
    
    # ========== 刻度参数 ==========
    'tick_params': {
        'labelsize': 25,
        'width': 1.2,
        'length': 5,
        'direction': 'in',        # 'in', 'out', 'inout'
    },
    'spine_linewidth': 1.2,       # 边框线宽
}

# ========== 6. 网格配置 ==========
GRID_CONFIG = {
    'show': True,
    'linestyle': 'dotted',
    'alpha': 0.5,
    'linewidth': 0.8,
}

# ========== 7. 图例配置 ==========
LEGEND_CONFIG = {
    'loc': 'lower right',         # 位置: 'upper left', 'upper right', 'lower left', 'lower right', 'center', etc.
    'fontsize': 20,
    'frameon': False,             # 是否显示边框
    'labelspacing': 0.5,          # 标签间距
    'labels': {
        'all': 'All Data',
        'west': 'Free (Westerly)',
        'east': 'Wake (Easterly)',
    }
}

# ========== 8. 统计文本配置（三个独立text）==========
STATS_TEXT_CONFIG = {
    # All Data 文本配置
    'all_data': {
        'show': True,
        'position': (0.01, 0.97),     # 文本位置 (x, y) 在axes坐标系中
        'text_format': 'EV: {:.1f}% (All)',  # 文本格式，{}会被variance替换
        'fontsize': 24,
        'fontweight': 'normal',
        'color': '#686767',           # 灰色
        'va': 'top',
        'ha': 'left',
    },
    # Free-stream 文本配置
    'freestream': {
        'show': True,
        'position': (0.01, 0.87),     # 文本位置 (x, y)
        'text_format': 'EV: {:.1f}% (Free)',  # 文本格式
        'fontsize': 24,
        'fontweight': 'normal',
        'color': '#f41111',           # 红色
        'va': 'top',
        'ha': 'left',
    },
    # Wake 文本配置
    'wake': {
        'show': True,
        'position': (0.01, 0.77),     # 文本位置 (x, y)
        'text_format': 'EV: {:.1f}% (Wake)',  # 文本格式
        'fontsize': 24,
        'fontweight': 'normal',
        'color': '#1996de',           # 蓝色
        'va': 'top',
        'ha': 'left',
    },
}

# ========== 9. 其他配置 ==========
heights = [10, 30, 50, 70]        # 高度列表
wind_direction_ranges = {
    'west': (225, 315),           # Free-stream风向范围
    'east': (45, 135),            # Wake风向范围
}

# matplotlib全局配置
plt.rcParams.update({
    'font.family': 'Arial',
    'font.size': 28,
    'axes.labelsize': 30,
    'axes.titlesize': 30,
    'xtick.labelsize': 28,
    'ytick.labelsize': 28,
    'legend.fontsize': 28,
    'figure.dpi': FIGURE_CONFIG['dpi'],
    'savefig.dpi': FIGURE_CONFIG['dpi'],
})

# ==================== 数据处理函数 ====================
def strict_direction_mask(df, heights, direction_range):
    """严格的风向掩码：要求所有高度的风向都满足条件"""
    min_deg, max_deg = direction_range
    mask = np.ones(len(df), dtype=bool)
    
    for h in heights:
        wd_col = f'obs_wind_direction_{h}m'
        if wd_col not in df.columns:
            raise ValueError(f"Missing wind direction column: {wd_col}")
        
        wd = df[wd_col].values
        
        if min_deg < max_deg:
            mask &= (wd >= min_deg) & (wd <= max_deg)
        else:
            mask &= (wd >= min_deg) | (wd <= max_deg)
    
    return mask

def perform_eof(data, mask, label):
    """对指定掩码的数据进行EOF分析"""
    data_subset = data[mask]
    
    if len(data_subset) < 10:
        raise ValueError(f"Not enough samples for {label}: {len(data_subset)}")
    
    # 标准化
    data_std = (data_subset - data_subset.mean()) / data_subset.std()
    
    # PCA (EOF分析)
    pca = PCA(n_components=4)
    pca.fit(data_std)
    
    # 获取第一模态
    eof1_pattern = pca.components_[0]
    explained_var = pca.explained_variance_ratio_[0]
    
    print(f"  {label}:")
    print(f"    Samples: {len(data_subset)}")
    print(f"    EOF1 explained variance: {explained_var*100:.2f}%")
    print(f"    EOF1 loadings: {eof1_pattern}")
    
    return eof1_pattern, explained_var

# ==================== 主程序 ====================
if __name__ == '__main__':
    # 确保输出目录存在
    os.makedirs(output_dir, exist_ok=True)
    
    # ==================== 1. 读取数据 ====================
    print("=" * 70)
    print("EOF Mode 1 Vertical Profile Comparison")
    print("=" * 70)
    print("\n[1/5] Loading data...")
    
    df = pd.read_csv(data_path)
    
    # 检测时间列
    if 'datetime' in df.columns:
        time_col = 'datetime'
    elif 'timestamp' in df.columns:
        time_col = 'timestamp'
    else:
        raise ValueError("No time column found in data")
    
    df[time_col] = pd.to_datetime(df[time_col])
    df = df.sort_values(time_col).reset_index(drop=True)
    
    print(f"  ✓ Loaded {len(df)} samples")
    
    # ==================== 2. 提取风速数据 ====================
    print("\n[2/5] Extracting wind speed data...")
    
    wind_variables = [f'obs_wind_speed_{h}m' for h in heights]
    
    # 检查所有变量是否存在
    missing_vars = [v for v in wind_variables if v not in df.columns]
    if missing_vars:
        raise ValueError(f"Missing variables: {missing_vars}")
    
    # 提取风速数据
    wind_data = df[wind_variables].copy()
    wind_data_clean = wind_data.dropna()
    valid_indices = wind_data_clean.index
    
    print(f"  Valid samples: {len(wind_data_clean)} ({len(wind_data_clean)/len(df)*100:.1f}%)")
    
    # ==================== 3. 创建风向掩码 ====================
    print("\n[3/5] Creating wind direction masks...")
    
    mask_west_full = strict_direction_mask(df, heights, wind_direction_ranges['west'])
    mask_east_full = strict_direction_mask(df, heights, wind_direction_ranges['east'])
    
    mask_west = mask_west_full[valid_indices]
    mask_east = mask_east_full[valid_indices]
    
    print(f"  Free-stream: {mask_west.sum()} samples ({mask_west.sum()/len(mask_west)*100:.1f}%)")
    print(f"  Wake: {mask_east.sum()} samples ({mask_east.sum()/len(mask_east)*100:.1f}%)")
    
    # ==================== 4. EOF分析 ====================
    print("\n[4/5] Performing EOF analysis...")
    
    mask_all = np.ones(len(wind_data_clean), dtype=bool)
    eof1_all, var_all = perform_eof(wind_data_clean, mask_all, "All Data")
    eof1_freestream, var_freestream = perform_eof(wind_data_clean, mask_west, "Free-stream")
    eof1_wake, var_wake = perform_eof(wind_data_clean, mask_east, "Wake")
    
    # ==================== 5. 绘图 ====================
    print("\n[5/5] Creating visualization...")
    
    fig, ax = plt.subplots(figsize=FIGURE_CONFIG['figsize'])
    
    # 绘制All Data EOF1
    ax.plot(eof1_all, heights, 
            marker=LINE_CONFIG['all']['marker'],
            color=COLORS['all']['line'],
            linewidth=LINE_CONFIG['all']['linewidth'],
            markersize=LINE_CONFIG['all']['markersize'],
            markerfacecolor=COLORS['all']['marker_face'],
            markeredgecolor=COLORS['all']['marker_edge'],
            markeredgewidth=LINE_CONFIG['all']['markeredgewidth'],
            label=LEGEND_CONFIG['labels']['all'],
            zorder=LINE_CONFIG['all']['zorder'],
            alpha=LINE_CONFIG['all']['alpha'])
    
    # 绘制Free-stream EOF1
    ax.plot(eof1_freestream, heights,
            marker=LINE_CONFIG['west']['marker'],
            color=COLORS['west']['line'],
            linewidth=LINE_CONFIG['west']['linewidth'],
            markersize=LINE_CONFIG['west']['markersize'],
            markerfacecolor=COLORS['west']['marker_face'],
            markeredgecolor=COLORS['west']['marker_edge'],
            markeredgewidth=LINE_CONFIG['west']['markeredgewidth'],
            label=LEGEND_CONFIG['labels']['west'],
            zorder=LINE_CONFIG['west']['zorder'],
            alpha=LINE_CONFIG['west']['alpha'])
    
    # 绘制Wake EOF1
    ax.plot(eof1_wake, heights,
            marker=LINE_CONFIG['east']['marker'],
            color=COLORS['east']['line'],
            linewidth=LINE_CONFIG['east']['linewidth'],
            markersize=LINE_CONFIG['east']['markersize'],
            markerfacecolor=COLORS['east']['marker_face'],
            markeredgecolor=COLORS['east']['marker_edge'],
            markeredgewidth=LINE_CONFIG['east']['markeredgewidth'],
            label=LEGEND_CONFIG['labels']['east'],
            zorder=LINE_CONFIG['east']['zorder'],
            alpha=LINE_CONFIG['east']['alpha'])
    
    # 添加零参考线
    ax.axvline(x=0, 
               color=COLORS['zero_line'],
               linestyle=LINE_CONFIG['zero_line']['linestyle'],
               alpha=LINE_CONFIG['zero_line']['alpha'],
               linewidth=LINE_CONFIG['zero_line']['linewidth'],
               zorder=LINE_CONFIG['zero_line']['zorder'])
    
    # 设置标签 - 应用labelpad
    ax.set_xlabel(AXIS_CONFIG['xlabel']['text'],
                  fontsize=AXIS_CONFIG['xlabel']['fontsize'],
                  fontweight=AXIS_CONFIG['xlabel']['fontweight'],
                  labelpad=AXIS_CONFIG['xlabel']['labelpad'])
    
    ax.set_ylabel(AXIS_CONFIG['ylabel']['text'],
                  fontsize=AXIS_CONFIG['ylabel']['fontsize'],
                  fontweight=AXIS_CONFIG['ylabel']['fontweight'],
                  labelpad=AXIS_CONFIG['ylabel']['labelpad'])
    
    ax.set_title(AXIS_CONFIG['title']['text'],
                 fontsize=AXIS_CONFIG['title']['fontsize'],
                 fontweight=AXIS_CONFIG['title']['fontweight'],
                 pad=AXIS_CONFIG['title']['pad'])
    
    # 设置y轴
    if AXIS_CONFIG['ylim'] is not None:
        # 使用手动指定的ylim
        ax.set_ylim(AXIS_CONFIG['ylim'])
    else:
        # 自动计算ylim
        ax.set_ylim(0, max(heights) + AXIS_CONFIG['ylim_offset'])
    
    # 设置y轴刻度
    if AXIS_CONFIG['yticks'] is not None:
        # 使用手动指定的刻度
        ax.set_yticks(AXIS_CONFIG['yticks'])
    elif AXIS_CONFIG['ytick_interval'] is not None:
        # 使用间隔自动生成刻度
        ylim = ax.get_ylim()
        yticks = np.arange(ylim[0], ylim[1] + AXIS_CONFIG['ytick_interval']/2, 
                          AXIS_CONFIG['ytick_interval'])
        ax.set_yticks(yticks)
    else:
        # 使用heights作为刻度
        ax.set_yticks(heights)
    
    # 设置y轴刻度标签
    if AXIS_CONFIG['ytick_labels'] is not None:
        ax.set_yticklabels(AXIS_CONFIG['ytick_labels'])
    elif AXIS_CONFIG['yticks'] is None and AXIS_CONFIG['ytick_interval'] is None:
        # 默认：只在heights位置显示刻度值
        ax.set_yticklabels([str(h) for h in heights])
    
    # 设置x轴范围
    if AXIS_CONFIG['xlim'] is not None:
        ax.set_xlim(AXIS_CONFIG['xlim'])
        
        # 设置x轴刻度
        if AXIS_CONFIG['xticks'] is not None:
            # 手动指定刻度
            ax.set_xticks(AXIS_CONFIG['xticks'])
        elif AXIS_CONFIG['xtick_interval'] is not None:
            # 使用间隔自动生成刻度
            xmin, xmax = AXIS_CONFIG['xlim']
            xticks = np.arange(xmin, xmax + AXIS_CONFIG['xtick_interval']/2, 
                              AXIS_CONFIG['xtick_interval'])
            ax.set_xticks(xticks)
        
        # 设置x轴刻度标签
        if AXIS_CONFIG['xtick_labels'] is not None:
            # 使用自定义标签
            ax.set_xticklabels(AXIS_CONFIG['xtick_labels'], 
                              rotation=AXIS_CONFIG['xtick_rotation'])
        elif AXIS_CONFIG['xtick_format'] is not None:
            # 使用格式化字符串
            xticks = ax.get_xticks()
            labels = [AXIS_CONFIG['xtick_format'] % x for x in xticks]
            ax.set_xticklabels(labels, rotation=AXIS_CONFIG['xtick_rotation'])
        elif AXIS_CONFIG['xtick_rotation'] != 0:
            # 只设置旋转角度
            ax.tick_params(axis='x', rotation=AXIS_CONFIG['xtick_rotation'])
    else:
        # 自动计算对称范围
        x_max = max(abs(eof1_all.max()), abs(eof1_all.min()),
                    abs(eof1_freestream.max()), abs(eof1_freestream.min()),
                    abs(eof1_wake.max()), abs(eof1_wake.min())) * 1.15
        ax.set_xlim(-x_max, x_max)
    
    # 设置刻度参数
    ax.tick_params(axis='both', which='major', 
                   labelsize=AXIS_CONFIG['tick_params']['labelsize'],
                   width=AXIS_CONFIG['tick_params']['width'],
                   length=AXIS_CONFIG['tick_params']['length'],
                   direction=AXIS_CONFIG['tick_params']['direction'])
    
    # 网格
    if GRID_CONFIG['show']:
        ax.grid(True,
                linestyle=GRID_CONFIG['linestyle'],
                alpha=GRID_CONFIG['alpha'],
                linewidth=GRID_CONFIG['linewidth'],
                color=COLORS['grid'])
    
    # ==================== 添加三个独立的统计文本 ====================
    # All Data 文本
    if STATS_TEXT_CONFIG['all_data']['show']:
        cfg = STATS_TEXT_CONFIG['all_data']
        text = cfg['text_format'].format(var_all * 100)
        ax.text(cfg['position'][0], cfg['position'][1],
                text,
                transform=ax.transAxes,
                verticalalignment=cfg['va'],
                horizontalalignment=cfg['ha'],
                fontsize=cfg['fontsize'],
                fontweight=cfg['fontweight'],
                color=cfg['color'])
    
    # Free-stream 文本
    if STATS_TEXT_CONFIG['freestream']['show']:
        cfg = STATS_TEXT_CONFIG['freestream']
        text = cfg['text_format'].format(var_freestream * 100)
        ax.text(cfg['position'][0], cfg['position'][1],
                text,
                transform=ax.transAxes,
                verticalalignment=cfg['va'],
                horizontalalignment=cfg['ha'],
                fontsize=cfg['fontsize'],
                fontweight=cfg['fontweight'],
                color=cfg['color'])
    
    # Wake 文本
    if STATS_TEXT_CONFIG['wake']['show']:
        cfg = STATS_TEXT_CONFIG['wake']
        text = cfg['text_format'].format(var_wake * 100)
        ax.text(cfg['position'][0], cfg['position'][1],
                text,
                transform=ax.transAxes,
                verticalalignment=cfg['va'],
                horizontalalignment=cfg['ha'],
                fontsize=cfg['fontsize'],
                fontweight=cfg['fontweight'],
                color=cfg['color'])
    
    # 图例
    ax.legend(loc=LEGEND_CONFIG['loc'],
              fontsize=LEGEND_CONFIG['fontsize'],
              frameon=LEGEND_CONFIG['frameon'],
              labelspacing=LEGEND_CONFIG['labelspacing'],
              markerfirst=False)
    
    # 美化边框
    for spine in ax.spines.values():
        spine.set_linewidth(AXIS_CONFIG['spine_linewidth'])
    
    # 调整布局
    plt.tight_layout()
    
    # 保存图形
    save_path_png = os.path.join(output_dir, 'final_EOF_mode1_vertical_comparison.png')
    save_path_pdf = os.path.join(output_dir, 'final_EOF_mode1_vertical_comparison.pdf')
    
    plt.savefig(save_path_png, dpi=FIGURE_CONFIG['dpi'], bbox_inches='tight', facecolor='white')
    plt.savefig(save_path_pdf, bbox_inches='tight', facecolor='white')
    
    print(f"\n  ✓ Figures saved:")
    print(f"    PNG: {save_path_png}")
    print(f"    PDF: {save_path_pdf}")
    
    # plt.show()