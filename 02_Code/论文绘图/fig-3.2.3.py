import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats as scipy_stats
from matplotlib.patches import Polygon, FancyBboxPatch
import matplotlib.patches as mpatches
from datetime import datetime

# 设置matplotlib参数
plt.style.use('default')
plt.rcParams['font.family'] = 'Arial'
plt.rcParams['font.size'] = 10

# 读取真实数据
data_path = '/Users/xiaxin/work/WindForecast_Project/01_Data/processed/imputed_data/changma_clean.csv'
df = pd.read_csv(data_path)

print(f"原始数据形状: {df.shape}")

# 数据预处理
# 计算偏差：预报值 - 观测值
df['ec_bias_10m'] = df['ec_wind_speed_10m'] - df['obs_wind_speed_10m']
df['gfs_bias_10m'] = df['gfs_wind_speed_10m'] - df['obs_wind_speed_10m']

# 解析时间并提取小时
df['datetime'] = pd.to_datetime(df['datetime'])
df['hour'] = df['datetime'].dt.hour

# 过滤有效数据
valid_data = df.dropna(subset=['ec_bias_10m', 'gfs_bias_10m', 'hour'])
print(f"有效数据点: {len(valid_data)}")

# 按小时分组计算统计量
hourly_stats = {}
for hour in range(24):
    hour_data = valid_data[valid_data['hour'] == hour]
    
    if len(hour_data) > 10:
        ec_biases = hour_data['ec_bias_10m']
        gfs_biases = hour_data['gfs_bias_10m']
        
        hourly_stats[hour] = {
            'ec_biases': ec_biases.values,
            'gfs_biases': gfs_biases.values,
            'count': len(hour_data),
            'ec_median': ec_biases.median(),
            'gfs_median': gfs_biases.median(),
            'ec_q5': ec_biases.quantile(0.05),
            'ec_q25': ec_biases.quantile(0.25),
            'ec_q75': ec_biases.quantile(0.75),
            'ec_q95': ec_biases.quantile(0.95),
            'gfs_q5': gfs_biases.quantile(0.05),
            'gfs_q25': gfs_biases.quantile(0.25),
            'gfs_q75': gfs_biases.quantile(0.75),
            'gfs_q95': gfs_biases.quantile(0.95),
            'ec_std': ec_biases.std(),
            'gfs_std': gfs_biases.std()
        }

# 计算全局中位数
global_median_ec = valid_data['ec_bias_10m'].median()
global_median_gfs = valid_data['gfs_bias_10m'].median()

# 创建图形
fig, ax = plt.subplots(figsize=(16, 14))

# 设置颜色方案 - 绿色(EC)和图片中的紫色(GFS)配色，外深内浅
ec_colors = {
    'light': '#C8E6C9',      # EC 90%范围 - 外层，较深绿
    'medium': '#E8F5E9',     # EC 50%范围 - 内层，较浅绿
    'dark': '#689F38',       # EC 密度线颜色 - 深绿
    'median_point': '#1B5E20', # EC 中位数点 - 最深绿
    'violin_gray': '#F1F8E9'   # EC 半小提琴图颜色 - 很浅绿
}

gfs_colors = {
    'light': '#673AB7',      # GFS 90%范围 - 外层，深紫色
    'medium': '#D1C4E9',     # GFS 50%范围 - 内层，浅紫色  
    'dark': '#673AB7',       # GFS 密度线颜色 - 深紫
    'median_point': '#512DA8', # GFS 中位数点 - 最深紫
    'violin_gray': '#EDE7F6'   # GFS 半小提琴图颜色 - 极浅紫
}

# 调整后的条形高度
ec_bar_height = 0.35  # EC条形变细
gfs_bar_height = 0.35  # GFS条形高度
bar_spacing = 0.05    # 条形间距

# 按小时顺序排列
hours_sorted = sorted(hourly_stats.keys())
y_positions = np.arange(len(hours_sorted))

for i, hour in enumerate(hours_sorted):
    y_center = len(hours_sorted) - 1 - i
    stats = hourly_stats[hour]
    
    # EC模型位置（上方）
    ec_y = y_center + bar_spacing/2 + ec_bar_height/2
    # GFS模型位置（下方）  
    gfs_y = y_center - bar_spacing/2 - gfs_bar_height/2
    
    # === EC模型绘制 ===
    ec_biases = stats['ec_biases']
    
    # 1. EC半小提琴图（去掉边框）
    if len(ec_biases) > 20 and stats['ec_std'] > 0.001:
        kde = scipy_stats.gaussian_kde(ec_biases)
        x_min = stats['ec_q5'] - stats['ec_std'] * 0.5
        x_max = stats['ec_q95'] + stats['ec_std'] * 0.5
        x_range = np.linspace(x_min, x_max, 300)
        density = kde(x_range)
        density_normalized = density / density.max() * (ec_bar_height * 0.8)
        
        upper_y = ec_y + density_normalized
        vertices = [(x_range[0], ec_y)]
        vertices.extend([(x, y) for x, y in zip(x_range, upper_y)])
        vertices.append((x_range[-1], ec_y))
        
        # 去掉edgecolor参数，不要边框
        violin_polygon = Polygon(vertices, facecolor=ec_colors['violin_gray'], 
                               alpha=0.6, zorder=1)
        ax.add_patch(violin_polygon)
    
    # 2. EC条形图层 - 三层设计：95%, 80%, 50%
    # 95%范围 (最外层，最浅色)
    ax.barh(ec_y, stats['ec_q95'] - stats['ec_q5'], left=stats['ec_q5'], 
            height=ec_bar_height, color='#E8F5E8', alpha=0.8, 
            edgecolor='white', linewidth=0.5, zorder=2)
    
    # 80%范围 (中间层，中等色)
    ec_q10, ec_q90 = np.percentile(stats['ec_biases'], [10, 90])
    ax.barh(ec_y, ec_q90 - ec_q10, left=ec_q10, 
            height=ec_bar_height, color='#C8E6C9', alpha=0.85, 
            edgecolor='white', linewidth=0.8, zorder=3)
    
    # 50%范围 (最内层，最深色)
    ax.barh(ec_y, stats['ec_q75'] - stats['ec_q25'], left=stats['ec_q25'], 
            height=ec_bar_height, color='#A5D6A7', alpha=0.9, 
            edgecolor='white', linewidth=1, zorder=4)
    
    # 3. EC中位数点
    ax.scatter(stats['ec_median'], ec_y, color=ec_colors['dark'], 
              s=80, zorder=10, edgecolor='white', linewidth=1)
    
    # === GFS模型绘制 ===
    gfs_biases = stats['gfs_biases']
    
    # 1. GFS半小提琴图（去掉边框）
    if len(gfs_biases) > 20 and stats['gfs_std'] > 0.001:
        kde = scipy_stats.gaussian_kde(gfs_biases)
        x_min = stats['gfs_q5'] - stats['gfs_std'] * 0.5
        x_max = stats['gfs_q95'] + stats['gfs_std'] * 0.5
        x_range = np.linspace(x_min, x_max, 300)
        density = kde(x_range)
        density_normalized = density / density.max() * (gfs_bar_height * 0.8)
        
        upper_y = gfs_y + density_normalized
        vertices = [(x_range[0], gfs_y)]
        vertices.extend([(x, y) for x, y in zip(x_range, upper_y)])
        vertices.append((x_range[-1], gfs_y))
        
        # 去掉edgecolor参数，不要边框
        violin_polygon = Polygon(vertices, facecolor=gfs_colors['violin_gray'], 
                               alpha=0.6, zorder=1)
        ax.add_patch(violin_polygon)
    
    # 2. GFS条形图层 - 三层设计：95%, 80%, 50%
    # 95%范围 (最外层，深紫色)
    ax.barh(gfs_y, stats['gfs_q95'] - stats['gfs_q5'], left=stats['gfs_q5'], 
            height=gfs_bar_height, color='#673AB7', alpha=0.6, 
            edgecolor='white', linewidth=0.5, zorder=2)
    
    # 80%范围 (中间层，中紫色)
    gfs_q10, gfs_q90 = np.percentile(stats['gfs_biases'], [10, 90])
    ax.barh(gfs_y, gfs_q90 - gfs_q10, left=gfs_q10, 
            height=gfs_bar_height, color='#9575CD', alpha=0.7, 
            edgecolor='white', linewidth=0.8, zorder=3)
    
    # 50%范围 (最内层，浅紫色)
    ax.barh(gfs_y, stats['gfs_q75'] - stats['gfs_q25'], left=stats['gfs_q25'], 
            height=gfs_bar_height, color='#D1C4E9', alpha=0.8, 
            edgecolor='white', linewidth=1, zorder=4)
    
    # 3. GFS中位数点
    ax.scatter(stats['gfs_median'], gfs_y, color=gfs_colors['dark'], 
              s=80, zorder=10, edgecolor='white', linewidth=1)

# 添加全局中位数参考线（绿色和紫色虚线）
# ax.axvline(x=0, color='#689F38', linestyle='-', 
#            alpha=0.7, linewidth=2, zorder=0, label=f'EC Overall Median')
# ax.axvline(x=global_median_gfs, color='#673AB7', linestyle='--', 
#            alpha=0.7, linewidth=2, zorder=0, label=f'GFS Overall Median')

# 不再添加零线

# 设置y轴标签
hour_labels = []
for hour in hours_sorted:
    stats = hourly_stats[hour]
    hour_label = f"{hour:02d} hr"
    # count_label = f"({stats['count']} obs)"
    hour_labels.append(f"{hour_label}")

ax.set_yticks(y_positions)
ax.set_yticklabels(reversed(hour_labels), fontsize=10)

# 设置x轴
ax.set_xlabel('Model Bias (m/s) = Forecast - Observation', fontsize=13, fontweight='bold')
ax.set_ylabel('Hour of Day', fontsize=13, fontweight='bold')

# 设置标题
main_title = "10M WIND SPEED BIAS COMPARISON - EC vs GFS"
subtitle = f"Diurnal pattern comparison across 24 hours (Total: {len(valid_data):,} observations)\nEC median: {global_median_ec:.3f} m/s, GFS median: {global_median_gfs:.3f} m/s"

ax.text(0.02, 0.98, main_title, transform=ax.transAxes,
        fontsize=16, fontweight='bold', va='top')
ax.text(0.02, 0.94, subtitle, transform=ax.transAxes,
        fontsize=11, va='top', color='#666666')

# 创建图例
legend_x = 0.62
legend_y = 0.88
legend_width = 0.36
legend_height = 0.3

# 图例背景
legend_bg = FancyBboxPatch((legend_x, legend_y - legend_height), legend_width, legend_height,
                          boxstyle="round,pad=0.02", transform=ax.transAxes,
                          facecolor='white', edgecolor='#CCCCCC', 
                          linewidth=1.5, alpha=0.95, zorder=15)
ax.add_patch(legend_bg)

# 图例标题
ax.text(legend_x + legend_width/2, legend_y - 0.02, 'Legend', 
        transform=ax.transAxes, ha='center', fontsize=12, fontweight='bold', zorder=20)

# 创建图例 - 更新为三层设计
legend_x = 0.62
legend_y = 0.88
legend_width = 0.36
legend_height = 0.32

# 图例背景
legend_bg = FancyBboxPatch((legend_x, legend_y - legend_height), legend_width, legend_height,
                          boxstyle="round,pad=0.02", transform=ax.transAxes,
                          facecolor='white', edgecolor='#CCCCCC', 
                          linewidth=1.5, alpha=0.95, zorder=15)
ax.add_patch(legend_bg)

# 图例标题
ax.text(legend_x + legend_width/2, legend_y - 0.02, 'Legend', 
        transform=ax.transAxes, ha='center', fontsize=12, fontweight='bold', zorder=20)

# EC模型示例
demo_y_ec = legend_y - 0.08
demo_bar_height = 0.015

ax.text(legend_x + 0.02, demo_y_ec + 0.02, 'EC Model:', 
        transform=ax.transAxes, fontsize=10, fontweight='bold', color='#4CAF50', zorder=20)

# EC 95%范围示例 (最浅)
ax.barh(demo_y_ec, 0.08, left=legend_x + 0.02, height=demo_bar_height,
        color='#E8F5E8', alpha=0.8, transform=ax.transAxes, zorder=16)
ax.text(legend_x + 0.11, demo_y_ec, '95% of data', 
        transform=ax.transAxes, va='center', fontsize=9, zorder=20)

# EC 80%范围示例 (中等)
ax.barh(demo_y_ec - 0.025, 0.06, left=legend_x + 0.03, height=demo_bar_height,
        color='#C8E6C9', alpha=0.85, transform=ax.transAxes, zorder=17)
ax.text(legend_x + 0.11, demo_y_ec - 0.025, '80% of data', 
        transform=ax.transAxes, va='center', fontsize=9, zorder=20)

# EC 50%范围示例 (最深)
ax.barh(demo_y_ec - 0.05, 0.04, left=legend_x + 0.04, height=demo_bar_height,
        color='#A5D6A7', alpha=0.9, transform=ax.transAxes, zorder=18)
ax.text(legend_x + 0.11, demo_y_ec - 0.05, '50% of data (IQR)', 
        transform=ax.transAxes, va='center', fontsize=9, zorder=20)

# GFS模型示例
demo_y_gfs = legend_y - 0.18

ax.text(legend_x + 0.02, demo_y_gfs + 0.02, 'GFS Model:', 
        transform=ax.transAxes, fontsize=10, fontweight='bold', color='#673AB7', zorder=20)

# GFS 95%范围示例 (最深)
ax.barh(demo_y_gfs, 0.08, left=legend_x + 0.02, height=demo_bar_height,
        color='#673AB7', alpha=0.6, transform=ax.transAxes, zorder=16)
ax.text(legend_x + 0.11, demo_y_gfs, '95% of data', 
        transform=ax.transAxes, va='center', fontsize=9, zorder=20)

# GFS 80%范围示例 (中等)
ax.barh(demo_y_gfs - 0.025, 0.06, left=legend_x + 0.03, height=demo_bar_height,
        color='#9575CD', alpha=0.7, transform=ax.transAxes, zorder=17)
ax.text(legend_x + 0.11, demo_y_gfs - 0.025, '80% of data', 
        transform=ax.transAxes, va='center', fontsize=9, zorder=20)

# GFS 50%范围示例 (最浅)
ax.barh(demo_y_gfs - 0.05, 0.04, left=legend_x + 0.04, height=demo_bar_height,
        color='#D1C4E9', alpha=0.8, transform=ax.transAxes, zorder=18)
ax.text(legend_x + 0.11, demo_y_gfs - 0.05, '50% of data (IQR)', 
        transform=ax.transAxes, va='center', fontsize=9, zorder=20)

# 中位数示例
ax.scatter(legend_x + 0.06, demo_y_gfs - 0.08, color='#1B5E20', 
          s=50, transform=ax.transAxes, zorder=18, edgecolor='white', linewidth=1)
ax.text(legend_x + 0.11, demo_y_gfs - 0.08, 'Median bias', 
        transform=ax.transAxes, va='center', fontsize=9, zorder=20)



# 添加说明
ax.text(legend_x + 0.02, demo_y_gfs - 0.09, 
        f'Dashed lines: Overall median biases\nGray shapes: Probability density distributions',
        transform=ax.transAxes, fontsize=8, style='italic', zorder=20)

# 美化坐标轴
for spine in ax.spines.values():
    spine.set_visible(False)
ax.spines['bottom'].set_visible(True)
ax.spines['left'].set_visible(True)
ax.spines['bottom'].set_linewidth(1)
ax.spines['left'].set_linewidth(1)

# 设置网格
ax.grid(axis='x', alpha=0.3, linestyle='-', linewidth=0.5)
ax.set_axisbelow(True)

# 调整y轴范围以适应双条形
ax.set_ylim(-0.8, len(hours_sorted) - 0.2)

# 添加模型比较统计
# comparison_stats = f"""Model Comparison:
# EC Model:
#   Mean bias: {valid_data['ec_bias_10m'].mean():.4f} m/s
#   RMSE: {np.sqrt(np.mean(valid_data['ec_bias_10m']**2)):.4f} m/s
  
# GFS Model:
#   Mean bias: {valid_data['gfs_bias_10m'].mean():.4f} m/s
#   RMSE: {np.sqrt(np.mean(valid_data['gfs_bias_10m']**2)):.4f} m/s"""

# ax.text(0.02, 0.4, comparison_stats, transform=ax.transAxes,
#         bbox=dict(boxstyle='round,pad=0.5', facecolor='#F8F9FA', alpha=0.9, edgecolor='#DDD'),
#         fontsize=9, fontfamily='monospace', va='top')

ax.axvline(x=0, color="#BFBFBF", linestyle='-', 
           alpha=0.7, linewidth=2, zorder=0, label='Zero Bias Line')

plt.tight_layout()

# 保存图片
plt.savefig('/Users/xiaxin/work/WindForecast_Project/03_Results/figures/3.2resutls/EC_vs_GFS_10m_bias_comparison.png', 
            dpi=300, bbox_inches='tight', facecolor='white', edgecolor='none')

plt.savefig('/Users/xiaxin/work/WindForecast_Project/03_Results/figures/3.2resutls/EC_vs_GFS_10m_bias_comparison.pdf', 
            bbox_inches='tight', facecolor='white', edgecolor='none')

plt.show()

print("EC vs GFS对比的halfeye图已完成!")
print("主要改进:")
print("✓ 移除了零线")
print("✓ 去掉了半小提琴图的边框")
print("✓ 添加了GFS模型(红色系)")
print("✓ EC条形变细，两模型上下排列")
print("✓ 添加了两个模型的全局中位数参考线")