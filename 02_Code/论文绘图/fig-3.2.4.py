import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Rectangle

# 设置matplotlib参数
plt.style.use('default')
plt.rcParams['font.family'] = 'Arial'
plt.rcParams['font.size'] = 10

# 读取数据
df = pd.read_csv('/Users/xiaxin/work/WindForecast_Project/03_Results/figures/3.2resutls/practical_nwp_evaluation.csv')

# 只选择风速数据
wind_data = df[df['Variable'].isin(['wind_speed_10m', 'wind_speed_70m'])].copy()

# 创建2行2列的子图
fig, axes = plt.subplots(2, 2, figsize=(16, 12))

# 设置颜色
ec_color = '#2196F3'  # 蓝色
gfs_color = '#E91E63'  # 粉色

# 子图配置：[高度, 时间]
subplot_configs = [
    ('10m', 'day', 0, 0),    # 10m日间
    ('10m', 'night', 0, 1),  # 10m夜间
    ('70m', 'day', 1, 0),    # 70m日间
    ('70m', 'night', 1, 1)   # 70m夜间
]

# 定义需要调整的子图的标签偏移量，避免重叠
label_offsets = {
    ('70m', 'day'): {
        'weak': {'ec': (-60, 10), 'gfs': (-60, -25)},  # 调整到左侧
        'moderate': {'ec': (-40, -30), 'gfs': (-60, 15)},  # 错开位置
        'strong': {'ec': (10, -25), 'gfs': (10, 10)}   # 上下错开
    },
    ('70m', 'night'): {
        'weak': {'ec': (-10, 10), 'gfs': (-60, -25)},  # 调整到左侧
        'moderate': {'ec': (10, -25), 'gfs': (-40, -30)},  # 错开位置
        'strong': {'ec': (10, 10), 'gfs': (10, 15)}
    }
}

for height_str, time_period, row, col in subplot_configs:
    ax = axes[row, col]
    
    # 确定对应的变量名
    variable = f'wind_speed_{height_str}'
    
    # 筛选对应高度和时间的数据
    height_data = wind_data[wind_data['Variable'] == variable]
    time_filtered = height_data[height_data['Classification'].str.contains(time_period)]
    
    # 按风切变强度分组
    shear_types = ['weak', 'moderate', 'strong']
    markers = {'weak': 'o', 'moderate': 's', 'strong': '^'}
    marker_sizes = {'weak': 400, 'moderate': 400, 'strong': 400}
    
    for shear in shear_types:
        shear_data = time_filtered[time_filtered['Classification'].str.contains(shear)]
        
        if len(shear_data) > 0:
            # 获取数据
            ec_corr = shear_data['EC_CORR'].iloc[0]
            ec_rmse = shear_data['EC_RMSE'].iloc[0]
            gfs_corr = shear_data['GFS_CORR'].iloc[0]
            gfs_rmse = shear_data['GFS_RMSE'].iloc[0]
            
            # 绘制EC和GFS点
            ax.scatter(ec_corr, ec_rmse, c=ec_color, s=marker_sizes[shear], 
                      alpha=0.8, marker=markers[shear], 
                      edgecolors='white', linewidth=2, zorder=5)
            
            ax.scatter(gfs_corr, gfs_rmse, c=gfs_color, s=marker_sizes[shear], 
                      alpha=0.8, marker=markers[shear], 
                      edgecolors='white', linewidth=2, zorder=5)
            
            # 获取当前子图的标签偏移量（只有70m的子图使用自定义偏移）
            if (height_str, time_period) in label_offsets:
                current_offsets = label_offsets[(height_str, time_period)][shear]
                # 添加标签（使用自定义偏移量）
                ax.annotate(shear.capitalize(), (ec_corr, ec_rmse), 
                           xytext=current_offsets['ec'], textcoords='offset points',
                           fontsize=18, alpha=0.9, color=ec_color, fontweight='normal')
                
                ax.annotate(shear.capitalize(), (gfs_corr, gfs_rmse),
                           xytext=current_offsets['gfs'], textcoords='offset points', 
                           fontsize=18, alpha=0.9, color=gfs_color, fontweight='normal')
            else:
                # 其他子图使用原来的默认偏移
                ax.annotate(shear.capitalize(), (ec_corr, ec_rmse), 
                           xytext=(10, 10), textcoords='offset points',
                           fontsize=18, alpha=0.9, color=ec_color, fontweight='normal')
                
                ax.annotate(shear.capitalize(), (gfs_corr, gfs_rmse),
                           xytext=(10, 10), textcoords='offset points', 
                           fontsize=18, alpha=0.9, color=gfs_color, fontweight='normal')
    
    # ====== 只添加高度-昼夜标签 ======
    time_label = 'Day' if time_period == 'day' else 'Night'
    label_text = f"{height_str[:]} {time_label}"
    ax.text(0.02, 0.98, label_text, 
            transform=ax.transAxes,
            verticalalignment='top',
            horizontalalignment='left',
            fontsize=23,
            fontweight='bold')
    
    # ====== 只在合适位置显示轴标签 ======
    # 只在底部行显示x轴标签
    if row == 1:  # 底部行
        ax.set_xlabel('Correlation Coefficient', fontsize=23, fontweight='normal')
    
    # 只在左侧列显示y轴标签
    if col == 0:  # 左侧列
        ax.set_ylabel('RMSE (m·s$^{-1}$)', fontsize=23, fontweight='normal')
    
    # 确保所有子图都显示刻度数字
    ax.tick_params(labelbottom=True, labelleft=True, labelsize=20)
    # 设置标题
    time_label = 'Daytime' if time_period == 'day' else 'Nighttime'
    # ax.set_title(f'{height_str.upper()} Wind Speed - {time_label}', 
    #             fontsize=12, fontweight='normal', pad=20)
    
    # 动态设置坐标轴范围
    all_data = time_filtered
    if len(all_data) > 0:
        all_corr = np.concatenate([all_data['EC_CORR'].values, all_data['GFS_CORR'].values])
        all_rmse = np.concatenate([all_data['EC_RMSE'].values, all_data['GFS_RMSE'].values])
        
        corr_min, corr_max = all_corr.min(), all_corr.max()
        rmse_min, rmse_max = all_rmse.min(), all_rmse.max()
        
        corr_margin = (corr_max - corr_min) * 0.2
        rmse_margin = (rmse_max - rmse_min) * 0.22
        
        ax.set_xlim(corr_min - corr_margin, corr_max + corr_margin)
        ax.set_ylim(rmse_min - rmse_margin, rmse_max + rmse_margin)
        # ====== 自定义刻度设置 ======
        # 根据不同子图设置不同的刻度
        if height_str == '10m' and time_period == 'day':
            ax.set_xticks([0.45, 0.50, 0.55, 0.60, 0.65, 0.7, 0.75])  # 自定义x轴刻度
            ax.set_ylim(1.9,3.5)
            ax.set_yticks([2.00, 2.20,2.40, 2.60,2.80, 3.00, 3.20, 3.40])  # 自定义y轴刻度
        elif height_str == '10m' and time_period == 'night':
            ax.set_xticks([0.45, 0.50, 0.55, 0.60, 0.65, 0.7, 0.75])
            ax.set_ylim(1.9,3.5)
            ax.set_yticks([2.00, 2.20,2.40, 2.60,2.80, 3.00, 3.20, 3.40])
        elif height_str == '70m' and time_period == 'day':
            ax.set_xticks([0.45, 0.50, 0.55, 0.60, 0.65, 0.7, 0.75])
            ax.set_ylim(2.5,4.1)
            ax.set_yticks([2.60, 2.80,3.0, 3.20,3.40, 3.6, 3.80, 4.0])
        elif height_str == '70m' and time_period == 'night':
            ax.set_xticks([0.45, 0.50, 0.55, 0.60, 0.65, 0.7, 0.75])
            ax.set_ylim(2.5,4.1)
            ax.set_yticks([2.60, 2.80,3.0, 3.20,3.40, 3.6, 3.80, 4.0])
        
        # 或者你可以用更自动化的方式：
        # # 自动生成合适数量的刻度
        # x_ticks = np.linspace(corr_min, corr_max, 5)  # 5个x轴刻度
        # y_ticks = np.linspace(rmse_min, rmse_max, 5)  # 5个y轴刻度
        # ax.set_xticks(x_ticks)
        # ax.set_yticks(y_ticks)
    
    
    # 添加网格
    ax.grid(True, alpha=0.3, linestyle='--')
    
    # 美化边框
    for spine in ax.spines.values():
        spine.set_linewidth(1.2)

# 移除多余的setp设置，避免重复标签

# 创建统一图例
legend_elements = [
    # 模型颜色
    plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=ec_color, 
               markersize=12, label='EC Model', markeredgecolor='white', markeredgewidth=2),
    plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=gfs_color, 
               markersize=12, label='GFS Model', markeredgecolor='white', markeredgewidth=2),
    # 风切变强度
    plt.Line2D([0], [0], marker='o', color='gray', markersize=10, 
               label='Weak Shear', linestyle='None'),
    plt.Line2D([0], [0], marker='s', color='gray', markersize=10, 
               label='Moderate Shear', linestyle='None'),
    plt.Line2D([0], [0], marker='^', color='gray', markersize=12, 
               label='Strong Shear', linestyle='None')
]
# 在图的上方添加图例，调整布局
fig.legend(handles=legend_elements, 
           loc='upper center', 
           bbox_to_anchor=(0.5, 0.945),  # 稍微上移
           ncol=5,  # 或者改成 ncol=2 分两行显示
           fontsize=22, 
           framealpha=0.9,
           columnspacing=1.5,  # 增加列间距
           handletextpad=0.2)  # 调整图标和文字间距

# 相应调整 subplots_adjust
# plt.subplots_adjust(top=0.85)  # 为图例留更多空间
# 在图的右上角添加图例
# fig.legend(handles=legend_elements, loc='upper center', bbox_to_anchor=(0.5, 0.95),
#            ncol=5, fontsize=20, framealpha=0.9)

# 调整布局和子图间距
plt.tight_layout()
# 精细调整子图间距
plt.subplots_adjust(
    top=0.88,      # 图例区域预留
    hspace=0.15,   # 垂直间距（行之间）- 减小可使上下图更紧凑
    wspace=0.08    # 水平间距（列之间）- 减小可使左右图更紧凑
)

# 保存图片
plt.savefig('/Users/xiaxin/work/WindForecast_Project/03_Results/figures/3.2resutls/day_night_shear_comparison.png', 
            dpi=500, bbox_inches='tight', facecolor='white')

plt.savefig('/Users/xiaxin/work/WindForecast_Project/03_Results/figures/3.2resutls/day_night_shear_comparison.pdf', 
            bbox_inches='tight', facecolor='white')

plt.show()

print("昼夜分离的风切变预报性能对比图已完成!")
print("保存位置:")
print("- PNG格式: day_night_shear_comparison.png")
print("- PDF格式: day_night_shear_comparison.pdf")

# 分析昼夜差异
print("\n昼夜差异分析:")
print("="*60)

for height_str in ['10m', '70m']:
    variable = f'wind_speed_{height_str}'
    height_data = wind_data[wind_data['Variable'] == variable]
    
    print(f"\n{height_str.upper()} 风速:")
    print("-" * 30)
    
    for time_period in ['day', 'night']:
        time_data = height_data[height_data['Classification'].str.contains(time_period)]
        
        if len(time_data) > 0:
            avg_ec_corr = time_data['EC_CORR'].mean()
            avg_ec_rmse = time_data['EC_RMSE'].mean()
            avg_gfs_corr = time_data['GFS_CORR'].mean()
            avg_gfs_rmse = time_data['GFS_RMSE'].mean()
            
            print(f"{time_period.capitalize()}:")
            print(f"  EC  - CORR: {avg_ec_corr:.3f}, RMSE: {avg_ec_rmse:.3f}")
            print(f"  GFS - CORR: {avg_gfs_corr:.3f}, RMSE: {avg_gfs_rmse:.3f}")
            
            # 找最佳条件
            best_ec = time_data.loc[time_data['EC_CORR'].idxmax()]
            print(f"  最佳EC条件: {best_ec['Classification']}")