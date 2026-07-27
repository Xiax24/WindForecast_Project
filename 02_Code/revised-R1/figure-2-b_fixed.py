#!/usr/bin/env python3
"""
步骤2b-2: 从保存的相关系数快速绘图
运行时间: <1秒
前提: 先运行 step2b-1_calculate_correlations.py
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import os

# ==================== 可调参数 ====================
# 字体大小
FONT_MAIN = 25           # 主要文字
FONT_LABEL = 34          # 轴标签
FONT_TITLE = 34          # 子图标题
FONT_LEGEND = 25         # 图例
FONT_TICK = 25           # 刻度标签
FONT_TIME_MARKER = 30    # 时间标记

# 子图布局 (宽, 总高, panel1高度比例)
FIG_WIDTH = 13           # 增加宽度以容纳右侧图例
FIG_HEIGHT = 8.5
PANEL1_HEIGHT_RATIO = 0.8    # Panel 1 占比
PANEL23_HEIGHT_RATIO = 1.0   # Panel 2+3 总占比 (等于Panel 1)

# 子图间距控制
PANEL_SPACING_1_2 = 0.25      # Panel 1 和 Panel 2 之间的间距
PANEL_SPACING_2_3 = 0.1      # Panel 2 和 Panel 3 之间的间距

# 线条和标记大小
LINE_WIDTH = 3.0
MARKER_SIZE = 8
MARKER_EDGE_WIDTH = 2.0

# DPI
FIGURE_DPI = 300

# ==================== 标题位置参数 ====================
# 标题水平对齐: 'left', 'center', 'right'
TITLE_LOC_PANEL1 = 'center'
TITLE_LOC_PANEL2 = 'center'
TITLE_LOC_PANEL3 = 'left'

# 标题位置微调 (使用 x, y 参数)
# x: 0=左, 0.5=中, 1=右
# y: 1.0=axes顶部, >1.0向上移动, <1.0向下移动
# 例如: y=1.05 表示比axes顶部高5%的位置
TITLE_X_PANEL1 = 0.5   # center对应0.5
TITLE_Y_PANEL1 = 1.05  # 1.0是默认位置，>1.0向上移动
TITLE_X_PANEL2 = 0.5
TITLE_Y_PANEL2 = 1.05
TITLE_X_PANEL3 = 0.1
TITLE_Y_PANEL3 = -0.95

# ==================== 图例位置参数 ====================
# 是否将所有图例放到右侧
USE_EXTERNAL_LEGENDS = True

# 外部图例位置参数 (相对于axes的位置)
# bbox_to_anchor格式: (x, y) 其中x,y相对于axes坐标系
# x > 1.0 表示在axes右侧
LEGEND_BBOX_PANEL1 = (0.99, 1.0)    # Panel 1图例位置 (右上)
LEGEND_BBOX_PANEL2 = (0.99, 1.0)    # Panel 2图例位置 (右上)  
LEGEND_BBOX_PANEL3 = (0.99, 1.0)    # Panel 3图例位置 (右上)

# 图例锚点位置: 'upper left'表示bbox_to_anchor指定的是图例的左上角
LEGEND_LOC_EXTERNAL = 'upper left'

# 图例列数
LEGEND_NCOL_PANEL1 = 1
LEGEND_NCOL_PANEL2 = 1
LEGEND_NCOL_PANEL3 = 1

# ==================== Y轴标签参数 ====================
# 共用Y轴标签设置
USE_SHARED_YLABEL = True              # True=共用一个ylabel, False=每个panel独立ylabel
SHARED_YLABEL_TEXT = 'Correlation Coefficient (R)'
SHARED_YLABEL_X = 0.00               # X位置（负数=左侧，相对于figure）
SHARED_YLABEL_Y = 0.5                 # Y位置（0.5=中间，相对于figure）
SHARED_YLABEL_FONTWEIGHT = 'normal'

# 如果不共用ylabel，则每个panel的ylabel设置
YLABEL_PANEL1 = 'Correlation (R)'
YLABEL_PANEL2 = 'Correlation (R)'
YLABEL_PANEL3 = 'Correlation (R)'

# ==================== Y轴范围和刻度参数 ====================
# Y轴范围 (ymin, ymax)
YLIM_PANEL1 = (0, 0.71)
YLIM_PANEL2 = (0, 0.61)
YLIM_PANEL3 = (0, 0.45)

# Y轴主刻度间隔 (None表示自动)
YTICK_INTERVAL_PANEL1 = 0.2   # 例如: 0.1表示每0.1一个刻度
YTICK_INTERVAL_PANEL2 = 0.2
YTICK_INTERVAL_PANEL3 = 0.2

# ==================== 配色 ====================
COLORS = {
    '10m-70m': "#5c5c5c",
    '30m-70m': "#ff7f0e",
    '10m-power': "#ff7f0e",
    '30m-power': '#2ca02c',
    '50m-power': '#ff7f0e',
    '70m-power': "#9ebfeb",
    'wake': '#1996de',
    'free': '#f41111',## #ff9896
}

# 配置matplotlib
plt.rcParams.update({
    'font.family': 'Arial',
    'font.size': FONT_MAIN,
    'axes.labelsize': FONT_LABEL,
    'axes.titlesize': FONT_TITLE,
    'xtick.labelsize': FONT_TICK,
    'ytick.labelsize': FONT_TICK,
    'legend.fontsize': FONT_LEGEND,
    'figure.dpi': FIGURE_DPI,
    'savefig.dpi': FIGURE_DPI,
    'axes.linewidth': 1.5,
    'xtick.major.width': 1.5,
    'ytick.major.width': 1.5,
})

def add_time_markers(ax):
    """添加时间参考线"""
    for t, label in {1: '1h', 6: '6h', 12: '12h', 24: '1d'}.items():
        ax.axvline(t, color='lightgray', linestyle='--', alpha=0.4, linewidth=1.5, zorder=1)
        t_offset = t * 1.12
        ax.text(t_offset, 0.850, label, ha='left', va='bottom', 
               fontsize=FONT_TIME_MARKER, color='gray', alpha=0.6, 
               transform=ax.get_xaxis_transform())

def add_background_zones(ax):
    """添加背景区域"""
    ylim = ax.get_ylim()
    xlim = ax.get_xlim()
    
    from matplotlib.patches import Rectangle
    
    # 短周期区域 (1h-24h)
    wake_x_start = max(1, xlim[0]) if xlim[0] is not None else 1
    wake_x_width = 24 - wake_x_start
    if wake_x_width > 0:
        zone = Rectangle((wake_x_start, ylim[0]), wake_x_width, ylim[1]-ylim[0],
                        linewidth=0, facecolor='#1996de', alpha=0.08, zorder=0)
        ax.add_patch(zone)
    
    # 长周期区域 (24h-168h)
    proxy_x_start = 24
    proxy_x_end = min(168, xlim[1]) if xlim[1] is not None else 168
    proxy_x_width = proxy_x_end - proxy_x_start
    if proxy_x_width > 0:
        zone = Rectangle((proxy_x_start, ylim[0]), proxy_x_width, ylim[1]-ylim[0],
                        linewidth=0, facecolor='#f41111', alpha=0.08, zorder=0)
        ax.add_patch(zone)

def remove_top_right_spines(ax):
    """移除上边框和右边框"""
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.tick_params(top=False, right=False)

def set_yticks_interval(ax, interval):
    """设置Y轴刻度间隔"""
    if interval is not None:
        ylim = ax.get_ylim()
        yticks = np.arange(ylim[0], ylim[1] + interval/2, interval)
        ax.set_yticks(yticks)

# 路径
data_dir = '/Users/xiaxin/work/WindForecast_Project/03_Results/re-plot-figures/figure-2/'
corr_file = os.path.join(data_dir, 'correlations_all.npz')

print("=" * 70)
print("Fast Plotting from Saved Correlations")
print("=" * 70)

# 1. 加载相关系数
print("\n[1/2] Loading saved correlations...")
if not os.path.exists(corr_file):
    print(f"ERROR: File not found: {corr_file}")
    print("Please run 'step2b-1_calculate_correlations.py' first!")
    exit(1)

data = np.load(corr_file, allow_pickle=True)

# 读取所有相关性
correlations = {}
corr_names = data['correlation_names']

for name in corr_names:
    R = data[f'{name}_R']
    T = data[f'{name}_T']
    correlations[name] = (R, T)

n = int(data['n_imfs'])
T_periods = correlations['10m-70m-all'][1]

print(f"  ✓ Loaded {len(correlations)} correlations")
print(f"  IMFs: {n}")
print(f"  Period range: {T_periods.min():.2f}h - {T_periods.max():.2f}h")
# print(f"  Processed: {data['timestamp']}")

# 检查是否有功率数据
has_power = '10m-power-all' in correlations
has_30m = '30m-70m-wake' in correlations

# 2. 创建图形布局
print("\n[2/2] Creating 3-panel figure with custom layout...")

# 使用嵌套GridSpec创建自定义布局，支持不同间距
fig = plt.figure(figsize=(FIG_WIDTH, FIG_HEIGHT))

# 外层GridSpec: 将figure分为Panel1和Panel2+3两部分
# 缩小right值为原来的比例，为右侧图例留出空间
outer_gs = gridspec.GridSpec(2, 1, 
                             height_ratios=[PANEL1_HEIGHT_RATIO, PANEL23_HEIGHT_RATIO],
                             hspace=PANEL_SPACING_1_2,
                             left=0.10, right=0.73,  # 从0.95改为0.73，留出右侧空间
                             top=0.96, bottom=0.06)

# Panel 1
ax1 = fig.add_subplot(outer_gs[0])

# 内层GridSpec: 将Panel2+3部分分为2个子图，使用不同的间距
inner_gs = gridspec.GridSpecFromSubplotSpec(2, 1, 
                                            subplot_spec=outer_gs[1],
                                            height_ratios=[1, 1],
                                            hspace=PANEL_SPACING_2_3)

# Panel 2 和 Panel 3
ax2 = fig.add_subplot(inner_gs[0], sharex=ax1)
ax3 = fig.add_subplot(inner_gs[1], sharex=ax1)

axes = [ax1, ax2, ax3]

# ========== Panel 1: Vertical Coupling ==========
ax = ax1
add_time_markers(ax)

# 10m-70m Wake
ax.plot(T_periods, correlations['10m-70m-wake'][0],
       color=COLORS['wake'], linestyle='--', linewidth=LINE_WIDTH,
       marker='^', markersize=MARKER_SIZE, markerfacecolor='white',
       markeredgecolor=COLORS['wake'], markeredgewidth=MARKER_EDGE_WIDTH,
       label='10 m-70 m (Wake)', alpha=1, zorder=3)

# 10m-70m Free
ax.plot(T_periods, correlations['10m-70m-free'][0],
       color=COLORS['free'], linestyle=':', linewidth=LINE_WIDTH,
       marker='^', markersize=MARKER_SIZE, markerfacecolor='white',
       markeredgecolor=COLORS['free'], markeredgewidth=MARKER_EDGE_WIDTH,
       label='10 m-70 m (Free)', alpha=1, zorder=3)

# 30m-70m (如果有)
if has_30m:
    ax.plot(T_periods, correlations['30m-70m-wake'][0],
           color=COLORS['wake'], linestyle='--', linewidth=LINE_WIDTH-0.5,
           marker='o', markersize=MARKER_SIZE-1, markerfacecolor='white',
           markeredgecolor=COLORS['wake'], markeredgewidth=MARKER_EDGE_WIDTH-0.5,
           label='30 m-70 m (Wake)', alpha=1, zorder=2)
    
    ax.plot(T_periods, correlations['30m-70m-free'][0],
           color=COLORS['free'], linestyle=':', linewidth=LINE_WIDTH-0.5,
           marker='o', markersize=MARKER_SIZE-1, markerfacecolor='white',
           markeredgecolor=COLORS['free'], markeredgewidth=MARKER_EDGE_WIDTH-0.5,
           label='30 m-70 m (Free)', alpha=1, zorder=2)

ax.set_ylim(*YLIM_PANEL1)

# 设置ylabel（如果不共用）
if not USE_SHARED_YLABEL:
    ax.set_ylabel(YLABEL_PANEL1, fontweight='bold')

# 设置标题（使用可调参数）
title1 = ax.set_title('Scale-Dependent Vertical Coherence', 
                       fontweight='bold', loc=TITLE_LOC_PANEL1, fontsize=FONT_TITLE, 
                       pad=17)
title1.set_position([TITLE_X_PANEL1, TITLE_Y_PANEL1])

# 设置图例（放到右侧）
if USE_EXTERNAL_LEGENDS:
    ax.legend(loc=LEGEND_LOC_EXTERNAL, 
             bbox_to_anchor=LEGEND_BBOX_PANEL1,
             frameon=False, framealpha=0.95, 
             ncol=LEGEND_NCOL_PANEL1,
             borderaxespad=0)

ax.grid(True, which='both', alpha=0.2, linestyle=':', linewidth=0.5)
set_yticks_interval(ax, YTICK_INTERVAL_PANEL1)
add_background_zones(ax)
plt.setp(ax.get_xticklabels(), visible=False)

# ========== Panel 2: 10m-Power Conditional ==========
if has_power:
    ax = ax2
    remove_top_right_spines(ax)
    
    ax.plot(T_periods, correlations['10m-power-wake'][0],
           color=COLORS['wake'], linestyle='--', linewidth=LINE_WIDTH,
           marker='s', markersize=MARKER_SIZE, markerfacecolor='white',
           markeredgecolor=COLORS['wake'], markeredgewidth=MARKER_EDGE_WIDTH,
           label='Wake (Easterly)', alpha=1, zorder=3)
    
    ax.plot(T_periods, correlations['10m-power-free'][0],
           color=COLORS['free'], linestyle=':', linewidth=LINE_WIDTH,
           marker='s', markersize=MARKER_SIZE, markerfacecolor='white',
           markeredgecolor=COLORS['free'], markeredgewidth=MARKER_EDGE_WIDTH,
           label='Free (Westerly)', alpha=1, zorder=3)
    
    ax.set_ylim(*YLIM_PANEL2)
    
    if not USE_SHARED_YLABEL:
        ax.set_ylabel(YLABEL_PANEL2, fontweight='normal')
    
    title2 = ax.set_title('Multiscale Coupling with Power Output', pad=15,
                           fontweight='bold', loc=TITLE_LOC_PANEL2)
    title2.set_position([TITLE_X_PANEL2, TITLE_Y_PANEL2])
    # ax.text(0.25, 0.79, 'WS 10m vs Power', ha='center', va='center',
    #         fontsize=30, fontweight='normal', transform=ax.transAxes)
    ax.text(0.25, 0.79, r'Correlation: $ws_\mathrm{10m}$ vs. $P$', ha='center', va='center',
            fontsize=25, fontweight='normal', transform=ax.transAxes)
    # 设置图例（放到右侧）
    if USE_EXTERNAL_LEGENDS:
        ax.legend(loc=LEGEND_LOC_EXTERNAL,
                 bbox_to_anchor=LEGEND_BBOX_PANEL2,
                 frameon=False, framealpha=0.95,
                 ncol=LEGEND_NCOL_PANEL2,
                 borderaxespad=0)
    
    ax.grid(True, which='both', alpha=0.2, linestyle=':', linewidth=0.5)
    set_yticks_interval(ax, YTICK_INTERVAL_PANEL2)
    add_background_zones(ax)
    plt.setp(ax.get_xticklabels(), visible=False)

# ========== Panel 3: 70m-Power Conditional ==========
if has_power:
    ax = ax3
    remove_top_right_spines(ax)
    
    ax.plot(T_periods, correlations['70m-power-wake'][0],
           color=COLORS['wake'], linestyle='--', linewidth=LINE_WIDTH,
           marker='s', markersize=MARKER_SIZE, markerfacecolor='white',
           markeredgecolor=COLORS['wake'], markeredgewidth=MARKER_EDGE_WIDTH,
           label='Wake (Easterly)', alpha=1, zorder=3)
    
    ax.plot(T_periods, correlations['70m-power-free'][0],
           color=COLORS['free'], linestyle=':', linewidth=LINE_WIDTH,
           marker='s', markersize=MARKER_SIZE, markerfacecolor='white',
           markeredgecolor=COLORS['free'], markeredgewidth=MARKER_EDGE_WIDTH,
           label='Free (Westerly)', alpha=1, zorder=3)
    
    ax.set_ylim(*YLIM_PANEL3)
    
    if not USE_SHARED_YLABEL:
        ax.set_ylabel(YLABEL_PANEL3, fontweight='normal')

    ax.text(0.25, 0.79, r'Correlation: $ws_\mathrm{70m}$ vs. $P$', ha='center', va='center',
            fontsize=25, fontweight='normal', transform=ax.transAxes)
    # ax.set_xlabel('Period (Hours)', fontweight='bold')
    ax.set_xlabel(r'Period $T$ (hours)', fontweight='normal',fontsize=32)
    
    # 设置图例（放到右侧）
    # if USE_EXTERNAL_LEGENDS:
    #     ax.legend(loc=LEGEND_LOC_EXTERNAL,
    #              bbox_to_anchor=LEGEND_BBOX_PANEL3,
    #              frameon=True, framealpha=0.95,
    #              ncol=LEGEND_NCOL_PANEL3,
    #              borderaxespad=0)
    
    ax.grid(True, which='both', alpha=0.2, linestyle=':', linewidth=0.5)
    set_yticks_interval(ax, YTICK_INTERVAL_PANEL3)
    add_background_zones(ax)

# ========== 添加共用的Y轴标签 ==========
if USE_SHARED_YLABEL:
    fig.text(SHARED_YLABEL_X, SHARED_YLABEL_Y, SHARED_YLABEL_TEXT,
             va='center', ha='center', rotation='vertical',
             fontsize=FONT_LABEL, fontweight=SHARED_YLABEL_FONTWEIGHT)

# 设置X轴范围和scale
ax3.set_xscale('log')
ax3.set_xlim(None, 168)

# 3. 保存
print("\nSaving figure...")
pdf_path = os.path.join(data_dir, 'final-correlation_4panels_extended.pdf')
png_path = os.path.join(data_dir, 'final-correlation_4panels_extended.png')

plt.savefig(pdf_path, bbox_inches='tight', facecolor='white')
plt.savefig(png_path, dpi=FIGURE_DPI, bbox_inches='tight', facecolor='white')

print(f"  ✓ {pdf_path}")
print(f"  ✓ {png_path}")

# plt.show()

# 统计
short_mask = T_periods < 24
long_mask = (T_periods >= 24) & (T_periods <= 168)

print(f"\n=== Summary ===")
print(f"Panels: 3")
print(f"IMFs: {n}")
print(f"Display range: up to 168h (1 week)")

print(f"\nShort-period (<24h): {short_mask.sum()} IMFs")
print(f"  10m-70m Wake: {np.nanmean(correlations['10m-70m-wake'][0][short_mask]):.3f}")
print(f"  10m-70m Free: {np.nanmean(correlations['10m-70m-free'][0][short_mask]):.3f}")

print(f"\nLong-period (24h-168h): {long_mask.sum()} IMFs")
print(f"  10m-70m Wake: {np.nanmean(correlations['10m-70m-wake'][0][long_mask]):.3f}")
print(f"  10m-70m Free: {np.nanmean(correlations['10m-70m-free'][0][long_mask]):.3f}")

print("\n" + "=" * 70)
print("✓ Fast plotting completed (<1 second)!")
print("=" * 70)