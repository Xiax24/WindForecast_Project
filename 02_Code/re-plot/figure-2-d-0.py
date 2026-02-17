#!/usr/bin/env python3
"""
Figure 2: Surface and Upper-Level Wind vs Vertical Mode 1
GRL Journal Quality Figure
Panel (a): 10 m wind speed vs PC1
Panel (b): 70 m wind speed vs PC1
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from scipy import stats
from scipy.stats import gaussian_kde
import os
import warnings
warnings.filterwarnings('ignore')

# ==================== 路径配置 ====================
DATA_PATH = '/Users/xiaxin/work/WindForecast_Project/01_Data/processed/matched_data/changma_matched.csv'
OUTPUT_DIR = '/Users/xiaxin/work/WindForecast_Project/03_Results/re-plot-figures/figure-2/'

# ==================== 图形整体配置 ====================
FIGURE = {
    'width': 10,                    # 图形宽度 (inches)
    'height': 15,                    # 图形高度 (inches) - 纵向排列适合用较高的比例
    'dpi': 600,                      # 分辨率 (GRL推荐300-600)
    'facecolor': 'white',            # 背景色
}

# 子图间距配置
LAYOUT = {
    'left': 0.12,                    # 左边距
    'right': 0.95,                   # 右边距
    'top': 0.96,                     # 上边距
    'bottom': 0.06,                  # 下边距
    'hspace': 0.1,                  # 子图垂直间距
}

# ==================== 颜色方案 ====================
COLORS = {
    'free_cmap': 'Reds',             # Free-stream密度图色系
    'wake_cmap': 'Blues',            # Wake密度图色系
    'free_line': '#D62728',          # Free-stream拟合线（鲜艳红）
    'wake_line': '#1F77B4',          # Wake拟合线（科研蓝）
    'grid': 'black',               # 网格颜色
    'spine': 'black',              # 边框颜色
}

# ==================== 散点和线条配置 ====================
SCATTER = {
    'size': 30,                       # 散点大小
    'alpha': 1,                   # 散点透明度
    'edgecolor': 'none',             # 散点边缘颜色
    'rasterized': False,              # 栅格化（减小PDF文件大小）
}

LINE = {
    'width': 2.5,                    # 拟合线宽度
    'style': '-',                    # 拟合线样式
    'alpha': 0.95,                   # 拟合线透明度
}

# ==================== 坐标轴配置 ====================
AXIS = {
    # X轴范围
    'xlim_10m': (0, 18),
    'xlim_70m': (0, 20),
    
    # Y轴范围
    'ylim': (-6, 8),
    
    # 轴标签
    'xlabel_10m': '',#r'Surface $WS_{10}$ (m$\cdot$s$^{-1}$)',
    'xlabel_70m': 'Wind Speed (m$\cdot$s$^{-1}$)',# r'Hub-height $WS_{70}$ (m$\cdot$s$^{-1}$)',
    'ylabel': r'PC1 Amplitude',
    
    # 子图标题
    'title_10m': 'Surface-to-PC1 Mapping',
    'title_70m': '',
}

# ==================== 字体配置 ====================
FONTS = {
    'family': 'Arial',               # 字体族（GRL推荐Arial或Helvetica）
    
    'size_panel_label': 48,          # 面板标签 (a), (b) 字体大小
    'weight_panel_label': 'bold',    # 面板标签粗细
    
    'size_axis_label': 45,           # 轴标签字体大小
    'weight_axis_label': 'normal',   # 轴标签粗细
    
    'size_tick_label': 40,           # 刻度标签字体大小
    
    'size_stats': 40,                # 统计文本字体大小
    'weight_stats': 'normal',        # 统计文本粗细
}

# ==================== 边距配置 ====================
SPACING = {
    'labelpad_x': 5,                 # X轴标签到轴的距离
    'labelpad_y': 5,                # Y轴标签到轴的距离
    'title_x': 0.5,                 # 面板标签X位置（在axes坐标系中）
    'title_y': 1.15,                 # 面板标签Y位置（在axes坐标系中）
}

# ==================== 刻度配置 ====================
TICKS = {
    'major_width': 1.0,              # 主刻度线宽
    'major_length': 5,               # 主刻度线长度
    'direction': 'in',               # 刻度方向 ('in', 'out', 'inout')
    'labelsize': 40,                 # 刻度标签大小
}

# ==================== 边框配置 ====================
SPINES = {
    'linewidth': 2.0,                # 边框线宽
    'color': 'black',              # 边框颜色
}

# ==================== 网格配置 ====================
GRID = {
    'show': False,                    # 是否显示网格
    'style': ':',                    # 网格线样式
    'width': 0.5,                    # 网格线宽
    'alpha': 1,                   # 网格透明度
    'color': 'black',              # 网格颜色
}

# ==================== 统计文本框配置 ====================
STATS_BOX = {
    # Free-stream文本框
    'free': {
        'position': (0.05, 0.96),    # 位置 (x, y)
        'va': 'top',                 # 垂直对齐
        'ha': 'left',                # 水平对齐
        'fontsize': 40
    },
    
    # Wake文本框
    'wake': {
        'position': (0.55, 0.44),    # 位置 (x, y)
        'va': 'top',                 # 垂直对齐
        'ha': 'left',                # 水平对齐
        'fontsize': 40
    },
}

# ==================== 数据配置 ====================
DATA = {
    'heights': [10, 30, 50, 70],
    'wind_dir_free': (225, 315),     # Free-stream风向范围
    'wind_dir_wake': (45, 135),      # Wake风向范围
}

# ==================== 应用matplotlib全局设置 ====================
plt.rcParams.update({
    'font.family': FONTS['family'],
    'font.size': FONTS['size_tick_label'],
    'figure.dpi': FIGURE['dpi'],
    'savefig.dpi': FIGURE['dpi'],
    'savefig.bbox': 'tight',
    'savefig.pad_inches': 0.05,
    'pdf.fonttype': 42,              # TrueType字体（期刊要求）
    'ps.fonttype': 42,
    'mathtext.fontset': 'custom',    # 自定义数学字体
    'mathtext.rm': 'Arial',          # 数学模式罗马字体
    'mathtext.it': 'Arial:italic',   # 数学模式斜体
    'mathtext.bf': 'Arial:bold',     # 数学模式粗体
})

# ==================== 辅助函数 ====================
def get_wind_direction_mask(df, direction_range):
    """创建风向掩码"""
    min_deg, max_deg = direction_range
    mask = np.ones(len(df), dtype=bool)
    for h in DATA['heights']:
        wd = df[f'obs_wind_direction_{h}m'].values
        if min_deg < max_deg:
            mask &= (wd >= min_deg) & (wd <= max_deg)
        else:
            mask &= (wd >= min_deg) | (wd <= max_deg)
    return mask

def perform_pca(wind_data):
    """执行主成分分析"""
    wind_std = (wind_data - wind_data.mean()) / wind_data.std()
    pca = PCA(n_components=4)
    scores = pca.fit_transform(wind_std)
    return scores[:, 0], pca.explained_variance_ratio_[0]

def fit_linear_model(x, y):
    """线性回归拟合"""
    X = x.reshape(-1, 1)
    reg = LinearRegression().fit(X, y)
    y_pred = reg.predict(X)
    r2 = r2_score(y, y_pred)
    
    # 计算p值
    n = len(x)
    residuals = y - y_pred
    mse = np.sum(residuals**2) / (n - 2)
    se_slope = np.sqrt(mse / np.sum((x - np.mean(x))**2))
    t_stat = reg.coef_[0] / se_slope
    p_value = 2 * (1 - stats.t.cdf(abs(t_stat), n - 2))
    
    return {
        'r2': r2,
        'slope': reg.coef_[0],
        'intercept': reg.intercept_,
        'prediction': y_pred,
        'p_value': p_value
    }

def plot_kde_density(ax, x, y, cmap, alpha, xlim, ylim):
    """绘制KDE密度散点"""
    mask = (x >= xlim[0]) & (x <= xlim[1]) & (y >= ylim[0]) & (y <= ylim[1])
    x_valid, y_valid = x[mask], y[mask]
    
    try:
        xy = np.vstack([x_valid, y_valid])
        density = gaussian_kde(xy)(xy)
        sorted_idx = density.argsort()
        
        ax.scatter(x_valid[sorted_idx], y_valid[sorted_idx], 
                  c=density[sorted_idx], 
                  s=SCATTER['size'],
                  cmap=cmap,
                  alpha=alpha,
                  edgecolors=SCATTER['edgecolor'],
                  rasterized=SCATTER['rasterized'])
    except:
        # 如果KDE失败，使用普通散点
        color = 'red' if 'Red' in cmap else 'blue'
        ax.scatter(x_valid, y_valid, s=SCATTER['size'], c=color, alpha=0.3)

def create_panel(ax, u_free, pc1_free, u_wake, pc1_wake, 
                xlim, xlabel, panel_label):
    """创建单个面板"""
    
    # 拟合模型
    fit_free = fit_linear_model(u_free, pc1_free)
    fit_wake = fit_linear_model(u_wake, pc1_wake)
    
    # 绘制Wake数据（蓝色，底层）
    plot_kde_density(ax, u_wake, pc1_wake, COLORS['wake_cmap'], 
                    SCATTER['alpha'], xlim, AXIS['ylim'])
    
    # 绘制Wake拟合线
    sort_idx = np.argsort(u_wake)
    ax.plot(u_wake[sort_idx], fit_wake['prediction'][sort_idx],
           color=COLORS['wake_line'], linewidth=LINE['width'],
           linestyle=LINE['style'], alpha=LINE['alpha'], zorder=10)
    
    # 绘制Free-stream数据（红色，上层）
    plot_kde_density(ax, u_free, pc1_free, COLORS['free_cmap'],
                    SCATTER['alpha'], xlim, AXIS['ylim'])
    
    # 绘制Free-stream拟合线
    sort_idx = np.argsort(u_free)
    ax.plot(u_free[sort_idx], fit_free['prediction'][sort_idx],
           color=COLORS['free_line'], linewidth=LINE['width'],
           linestyle=LINE['style'], alpha=LINE['alpha'], zorder=11)
    
    # 设置坐标轴
    ax.set_xlim(xlim)
    ax.set_ylim(AXIS['ylim'])
    ax.set_xlabel(xlabel, fontsize=FONTS['size_axis_label'],
                 fontweight=FONTS['weight_axis_label'],
                 labelpad=SPACING['labelpad_x'])
    ax.set_ylabel(AXIS['ylabel'], fontsize=FONTS['size_axis_label'],
                 fontweight=FONTS['weight_axis_label'],
                 labelpad=SPACING['labelpad_y'])
    
    # 面板标签
    ax.text(SPACING['title_x'], SPACING['title_y'], panel_label,
           transform=ax.transAxes, fontsize=FONTS['size_panel_label'],
           fontweight=FONTS['weight_panel_label'], va='top', ha='center')
    
    # 刻度设置
    ax.tick_params(axis='both', which='major',
                  labelsize=TICKS['labelsize'],
                  width=TICKS['major_width'],
                  length=TICKS['major_length'],
                  direction=TICKS['direction'])
    
    # 边框设置
    for spine in ax.spines.values():
        spine.set_linewidth(SPINES['linewidth'])
        spine.set_color(SPINES['color'])
    
    # 网格
    if GRID['show']:
        ax.grid(True, linestyle=GRID['style'], linewidth=GRID['width'],
               alpha=GRID['alpha'], color=GRID['color'], zorder=0)
    
    # 统计文本框 - Free-stream（使用Arial字体，去掉LaTeX）
    cfg_free = STATS_BOX['free']
    stats_text_free = (
        f"y = {fit_free['slope']:.2f}x {fit_free['intercept']:+.2f}\n"
        f"N = {len(u_free)}\n"
        f"p < 0.01\n"
        f"R² = {fit_free['r2']:.3f}"
    )
    ax.text(cfg_free['position'][0], cfg_free['position'][1],
           stats_text_free, transform=ax.transAxes, color='#D62728',
           fontsize=cfg_free['fontsize'], fontfamily='Arial',
           va=cfg_free['va'], ha=cfg_free['ha'])
    
    # 统计文本框 - Wake（使用Arial字体，去掉LaTeX）
    cfg_wake = STATS_BOX['wake']
    stats_text_wake = (
        f"y = {fit_wake['slope']:.2f}x {fit_wake['intercept']:+.2f}\n"
        f"N = {len(u_wake)}\n"
        f"p < 0.01\n"
        f"R² = {fit_wake['r2']:.3f}"
    )
    ax.text(cfg_wake['position'][0], cfg_wake['position'][1],
           stats_text_wake, transform=ax.transAxes, color='#1F77B4',
           fontsize=cfg_wake['fontsize'], fontfamily='Arial',
           va=cfg_wake['va'], ha=cfg_wake['ha'])

# ==================== 主程序 ====================
if __name__ == '__main__':
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    print("=" * 80)
    print("Figure 2: Surface and Upper-Level Wind vs Vertical Mode 1")
    print("GRL Journal Quality")
    print("=" * 80)
    
    # 读取数据
    print("\n[1/4] Loading data...")
    df = pd.read_csv(DATA_PATH)
    time_col = 'datetime' if 'datetime' in df.columns else 'timestamp'
    df[time_col] = pd.to_datetime(df[time_col])
    
    # 提取风速数据
    print("[2/4] Extracting wind speed data...")
    wind_vars = [f'obs_wind_speed_{h}m' for h in DATA['heights']]
    wind_data = df[wind_vars].dropna()
    valid_idx = wind_data.index
    
    # 创建风向掩码
    print("[3/4] Creating wind direction masks...")
    mask_free = get_wind_direction_mask(df, DATA['wind_dir_free'])[valid_idx]
    mask_wake = get_wind_direction_mask(df, DATA['wind_dir_wake'])[valid_idx]
    
    # PCA分析
    print("[4/4] Performing PCA...")
    pc1_all, variance = perform_pca(wind_data)
    
    # 数据分组
    pc1_free = pc1_all[mask_free]
    pc1_wake = pc1_all[mask_wake]
    
    u10m_free = wind_data.loc[mask_free, 'obs_wind_speed_10m'].values
    u10m_wake = wind_data.loc[mask_wake, 'obs_wind_speed_10m'].values
    u70m_free = wind_data.loc[mask_free, 'obs_wind_speed_70m'].values
    u70m_wake = wind_data.loc[mask_wake, 'obs_wind_speed_70m'].values
    
    print(f"\n  PC1 explained variance: {variance*100:.1f}%")
    print(f"  Free-stream samples: {len(u10m_free)}")
    print(f"  Wake samples: {len(u10m_wake)}")
    
    # 创建图形
    print("\n[5/5] Creating figure...")
    fig, (ax1, ax2) = plt.subplots(2, 1, 
                                    figsize=(FIGURE['width'], FIGURE['height']),
                                    facecolor=FIGURE['facecolor'])
    
    # 面板(a): 10m
    create_panel(ax1, u10m_free, pc1_free, u10m_wake, pc1_wake,
                AXIS['xlim_10m'], AXIS['xlabel_10m'], AXIS['title_10m'])
    
    # 面板(b): 70m  
    create_panel(ax2, u70m_free, pc1_free, u70m_wake, pc1_wake,
                AXIS['xlim_70m'], AXIS['xlabel_70m'], AXIS['title_70m'])
    
    # 调整布局
    plt.subplots_adjust(left=LAYOUT['left'], right=LAYOUT['right'],
                       top=LAYOUT['top'], bottom=LAYOUT['bottom'],
                       hspace=LAYOUT['hspace'])
    
    # 保存图形
    png_path = os.path.join(OUTPUT_DIR, 'final_Figure2d.png')
    pdf_path = os.path.join(OUTPUT_DIR, 'final_Figure2d.pdf')

    plt.savefig(png_path, dpi=FIGURE['dpi'], facecolor=FIGURE['facecolor'])
    plt.savefig(pdf_path, facecolor=FIGURE['facecolor'])
