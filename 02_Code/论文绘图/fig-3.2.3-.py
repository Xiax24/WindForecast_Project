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
plt.rcParams['font.size'] = 9

# 读取真实数据
data_path = '/Users/xiaxin/work/WindForecast_Project/01_Data/processed/imputed_data/changma_clean.csv'
df = pd.read_csv(data_path)

print(f"原始数据形状: {df.shape}")

# 数据预处理
# 计算四个高度的偏差
heights = ['10m', '30m', '50m', '70m']
for height in heights:
    df[f'ec_bias_{height}'] = df[f'ec_wind_speed_{height}'] - df[f'obs_wind_speed_{height}']
    df[f'gfs_bias_{height}'] = df[f'gfs_wind_speed_{height}'] - df[f'obs_wind_speed_{height}']

# 解析时间并提取小时
df['datetime'] = pd.to_datetime(df['datetime'])
df['hour'] = df['datetime'].dt.hour

# 创建1行4列的子图
fig, axes = plt.subplots(1, 4, figsize=(20, 30))

# 设置总标题
# fig.suptitle('EC vs GFS Wind Speed Bias Comparison - Multi-Height Analysis\n24-Hour Diurnal Pattern Across Different Heights', 
#              fontsize=16, fontweight='bold', y=0.95)

for subplot_idx, height in enumerate(heights):
    ax = axes[subplot_idx]
    
    # 过滤有效数据
    valid_cols = [f'ec_bias_{height}', f'gfs_bias_{height}', 'hour']
    valid_data = df.dropna(subset=valid_cols)
    
    # 按小时分组计算统计量
    hourly_stats = {}
    for hour in range(24):
        hour_data = valid_data[valid_data['hour'] == hour]
        
        if len(hour_data) > 10:
            ec_biases = hour_data[f'ec_bias_{height}']
            gfs_biases = hour_data[f'gfs_bias_{height}']
            
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
    global_median_ec = valid_data[f'ec_bias_{height}'].median()
    global_median_gfs = valid_data[f'gfs_bias_{height}'].median()
    
    # 调整后的条形高度（适应更小的子图）
    ec_bar_height = 0.3
    gfs_bar_height = 0.3
    bar_spacing = 0.05
    
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
        
        # 1. EC半小提琴图
        # if len(ec_biases) > 20 and stats['ec_std'] > 0.001:
        #     kde = scipy_stats.gaussian_kde(ec_biases)
        #     x_min = stats['ec_q5'] - stats['ec_std'] * 0.5
        #     x_max = stats['ec_q95'] + stats['ec_std'] * 0.5
        #     x_range = np.linspace(x_min, x_max, 200)
        #     density = kde(x_range)
        #     density_normalized = density / density.max() * (ec_bar_height * 0.6)
            
        #     upper_y = ec_y + density_normalized
        #     vertices = [(x_range[0], ec_y)]
        #     vertices.extend([(x, y) for x, y in zip(x_range, upper_y)])
        #     vertices.append((x_range[-1], ec_y))
            
        #     violin_polygon = Polygon(vertices, facecolor='#F1F8E9', 
        #                            alpha=0.6, zorder=1)
        #     ax.add_patch(violin_polygon)
        
        # 2. EC条形图层 - 三层设计：95%, 80%, 50% (只显示上半部分)
        # 95%范围 (最浅) - 只显示上半部分
        ax.barh(ec_y + ec_bar_height/4, stats['ec_q95'] - stats['ec_q5'], left=stats['ec_q5'], 
                height=ec_bar_height/2, color="#71D175", alpha=1, 
                edgecolor='white', linewidth=1, zorder=2)
        
        # 80%范围 (中等) - 只显示上半部分
        ec_q10, ec_q90 = np.percentile(stats['ec_biases'], [10, 90])
        ax.barh(ec_y + ec_bar_height/4, ec_q90 - ec_q10, left=ec_q10, 
                height=ec_bar_height/2, color='#C8E6C9', alpha=0.85, 
                edgecolor='white', linewidth=0.8, zorder=3)
        
        # 50%范围 (最深) - 只显示上半部分
        ax.barh(ec_y + ec_bar_height/4, stats['ec_q75'] - stats['ec_q25'], left=stats['ec_q25'], 
                height=ec_bar_height/2, color='#E8F5E8', alpha=0.8, 
                edgecolor='white', linewidth=1, zorder=4)
        
        # 3. EC中位数点
        ax.scatter(stats['ec_median'], y_center+0.25, color="#10DC1E", 
                  s=200, zorder=10, edgecolor='white', linewidth=1.5)
        
        # === GFS模型绘制 ===
        gfs_biases = stats['gfs_biases']
        
        # 1. GFS半小提琴图
        # if len(gfs_biases) > 20 and stats['gfs_std'] > 0.001:
        #     kde = scipy_stats.gaussian_kde(gfs_biases)
        #     x_min = stats['gfs_q5'] - stats['gfs_std'] * 0.5
        #     x_max = stats['gfs_q95'] + stats['gfs_std'] * 0.5
        #     x_range = np.linspace(x_min, x_max, 200)
        #     density = kde(x_range)
        #     density_normalized = density / density.max() * (gfs_bar_height * 0.6)
            
        #     upper_y = gfs_y + density_normalized
        #     vertices = [(x_range[0], gfs_y)]
        #     vertices.extend([(x, y) for x, y in zip(x_range, upper_y)])
        #     vertices.append((x_range[-1], gfs_y))
            
        #     violin_polygon = Polygon(vertices, facecolor='#EDE7F6', 
        #                            alpha=0.6, zorder=1)
        #     ax.add_patch(violin_polygon)
        
        # 2. GFS条形图层 - 三层设计：95%, 80%, 50% (只显示下半部分)
        # 95%范围 (最深) - 只显示下半部分
        ax.barh(gfs_y - gfs_bar_height/4, stats['gfs_q95'] - stats['gfs_q5'], left=stats['gfs_q5'], 
                height=gfs_bar_height/2, color="#4C4C4E", alpha=0.9, 
                edgecolor='white', linewidth=0.5, zorder=2)#512DA8
        
        # 80%范围 (中等) - 只显示下半部分
        gfs_q10, gfs_q90 = np.percentile(stats['gfs_biases'], [10, 90])
        ax.barh(gfs_y - gfs_bar_height/4, gfs_q90 - gfs_q10, left=gfs_q10, 
                height=gfs_bar_height/2, color="#C3C2C3", alpha=0.7, 
                edgecolor='white', linewidth=0.8, zorder=3)
        
        # 50%范围 (最浅) - 只显示下半部分
        ax.barh(gfs_y - gfs_bar_height/4, stats['gfs_q75'] - stats['gfs_q25'], left=stats['gfs_q25'], 
                height=gfs_bar_height/2, color="#E3E2E4", alpha=0.8, 
                edgecolor='white', linewidth=1, zorder=4)
        
        # 3. GFS中位数点
        ax.scatter(stats['gfs_median'], gfs_y, color="#444345", 
                  s=200, zorder=10, edgecolor='white', linewidth=1.5)# "#1D0557" '#9575CD'  '#D1C4E9'
    
    # 添加全局中位数参考线
    # ax.axvline(x=global_median_ec, color='#689F38', linestyle='--', 
    #            alpha=0.6, linewidth=1.5, zorder=0)
    # ax.axvline(x=global_median_gfs, color='#673AB7', linestyle='--', 
    #            alpha=0.6, linewidth=1.5, zorder=0)
    ax.axvline(x=0, color="black", linestyle='-', alpha=0.6, linewidth=1.5, zorder=1)
    # 设置y轴标签（只在第一个子图显示完整标签）
    if subplot_idx == 0:
        hour_labels = [f"{hour:02d}" for hour in reversed(hours_sorted)]
        ax.set_yticks(y_positions)
        ax.set_yticklabels(hour_labels, fontsize=25)
        ax.set_ylabel('Hour of Day', fontsize=25, fontweight='normal')
    else:
        ax.set_yticks(y_positions)
        ax.set_yticklabels([])
    
    # 设置x轴刻度标签大小
    ax.tick_params(axis='x', labelsize=23)  # 将20改为你想要的字体大小
    # 设置x轴
    ax.set_xlabel(f'Bias (m·s$^{-1}$)', fontsize=25, fontweight='normal')
    ax.set_xlim(-4.9, 5)
    ax.set_xticks(np.arange(-5, 5, 1))
    # 设置子图标题
    ax.set_title(f'Height Layer {height.upper()}', 
                fontsize=25, fontweight='normal', pad=15)
    
    # 美化坐标轴
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.spines['bottom'].set_visible(True)
    ax.spines['bottom'].set_linewidth(1)
    if subplot_idx == 0:
        ax.spines['left'].set_visible(True)
        ax.spines['left'].set_linewidth(1)
    
    # 设置网格
    ax.grid(False)
    ax.set_axisbelow(True)
    
    # 调整y轴范围
    ax.set_ylim(-0.8, len(hours_sorted) - 0.2)

# 创建统一的图例（放在右上角）
legend_elements = [
    plt.Rectangle((0, 0), 1, 1, facecolor='#71D175', alpha=0.9, label='EC: 95% of data'),
    plt.Rectangle((0, 0), 1, 1, facecolor='#C8E6C9', alpha=0.85, label='EC: 80% of data'),
    plt.Rectangle((0, 0), 1, 1, facecolor='#E8F5E8', alpha=0.8, label='EC: 50% of data'),
    plt.Rectangle((0, 0), 1, 1, facecolor='#4C4C4E', alpha=0.9, label='GFS: 95% of data'),
    plt.Rectangle((0, 0), 1, 1, facecolor='#C3C2C3', alpha=0.9, label='GFS: 80% of data'),
    plt.Rectangle((0, 0), 1, 1, facecolor='#E3E2E4', alpha=0.8, label='GFS: 50% of data'),
    plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='#10DC1E', 
               markersize=20, label='EC Median', markeredgecolor='white', markeredgewidth=1),
    plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='#444345', 
               markersize=20, label='GFS Median', markeredgecolor='white', markeredgewidth=1)
]

fig.legend(handles=legend_elements, loc='upper center', bbox_to_anchor=(0.5, 0.17),
           ncol=3, fontsize=25, frameon=True, framealpha=0.9)

# 添加数据统计信息
# data_info = f"""Dataset: {len(df):,} total observations | Period: {df['datetime'].min().strftime('%Y-%m-%d')} to {df['datetime'].max().strftime('%Y-%m-%d')}
# Gray shapes: Probability density distributions | Colored bars: Data percentiles | Dots: Median values | Dashed lines: Overall medians"""

# fig.text(0.02, 0.08, data_info, fontsize=9, style='italic', 
#          bbox=dict(boxstyle='round,pad=0.5', facecolor='#F8F9FA', alpha=0.9))

# 调整布局
plt.tight_layout()
plt.subplots_adjust(top=0.85, bottom=0.2, hspace=0.3, wspace=0.05)

# 保存图片
plt.savefig('/Users/xiaxin/work/WindForecast_Project/03_Results/figures/3.2resutls/EC_vs_GFS_all_heights_comparison.png', 
            dpi=600, bbox_inches='tight', facecolor='white', edgecolor='none')

plt.savefig('/Users/xiaxin/work/WindForecast_Project/03_Results/figures/3.2resutls/EC_vs_GFS_all_heights_comparison.pdf', 
            bbox_inches='tight', facecolor='white', edgecolor='none')

plt.show()

print("四个高度的EC vs GFS对比图已完成!")
print("保存位置:")
print("- PNG格式: EC_vs_GFS_all_heights_comparison.png")
print("- PDF格式: EC_vs_GFS_all_heights_comparison.pdf")

# 输出各高度的统计摘要
print(f"\n各高度全局偏差对比:")
print("高度   EC中位数   GFS中位数   EC优势")
print("-" * 40)
for height in heights:
    valid_data = df.dropna(subset=[f'ec_bias_{height}', f'gfs_bias_{height}'])
    ec_median = valid_data[f'ec_bias_{height}'].median()
    gfs_median = valid_data[f'gfs_bias_{height}'].median()
    ec_better = abs(ec_median) < abs(gfs_median)
    print(f"{height:4s}   {ec_median:8.4f}   {gfs_median:9.4f}   {'✓' if ec_better else '✗'}")