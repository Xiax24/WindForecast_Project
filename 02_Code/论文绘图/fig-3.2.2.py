import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def create_unified_seasonal_boxplot(df, figsize=(20, 8)):
    """
    创建统一的季节特征盒须图
    - 所有高度合在一张图上
    - 10m黑色，30m蓝色，50m绿色，70m紫色
    - EC用深色，GFS用对应的浅色
    - 盒子宽度较细
    """
    
    # 添加月份信息
    df_copy = df.copy()
    df_copy['month_year'] = df_copy.index.to_period('M')
    
    # 创建单个图表
    fig, ax = plt.subplots(figsize=figsize)
    
    heights = ['10m', '70m']
    
    # 定义颜色方案：深色(EC) + 浅色(GFS)，只要10m和70m
    color_scheme = {
        '10m': {'ec': '#2C3E50', 'gfs': '#7F8C8D'},  # 深黑 + 浅灰 "#051CED", 'gfs': "#8AB6ED".  '#2C3E50', 'gfs': '#7F8C8D'
        '70m': {'ec': "#ED0543", 'gfs': "#ED8AA1"}   # 深紫 + 浅紫
    }
    
    # 获取唯一月份
    unique_months = sorted(df_copy['month_year'].unique())
    n_months = len(unique_months)
    n_heights = len(heights)
    
    # 为每个月份和高度组合创建位置
    position_counter = 0
    positions_info = []  # 存储位置信息，用于设置X轴标签
    
    for j, month in enumerate(unique_months):
        month_data = df_copy[df_copy['month_year'] == month]
        month_start_pos = position_counter
        
        for i, height in enumerate(heights):
            # 查找对应列名
            obs_col = f'obs_wind_speed_{height}'
            ec_col = f'ec_wind_speed_{height}'
            gfs_col = f'gfs_wind_speed_{height}'
            
            # 检查列是否存在
            if all(col in df.columns for col in [obs_col, ec_col, gfs_col]):
                # 计算偏差
                ec_bias = (month_data[ec_col] - month_data[obs_col]).dropna().values
                gfs_bias = (month_data[gfs_col] - month_data[obs_col]).dropna().values
                
                # 绘制ECMWF盒须图（深色）
                if len(ec_bias) > 0:
                    bp_ec = ax.boxplot([ec_bias], positions=[position_counter], widths=0.5,
                                     patch_artist=True,
                                     boxprops=dict(facecolor='none', edgecolor=color_scheme[height]['ec'], linewidth=1.5),
                                     whiskerprops=dict(color=color_scheme[height]['ec'], linewidth=1.5),
                                     capprops=dict(color=color_scheme[height]['ec'], linewidth=1.5),
                                     medianprops=dict(color=color_scheme[height]['ec'], linewidth=2),
                                     flierprops=dict(marker='o', markerfacecolor='none', 
                                                    markeredgecolor=color_scheme[height]['ec'], markersize=3))
                
                position_counter += 0.8  # 同一高度内的EC和GFS距离较近
                
                # 绘制GFS盒须图（浅色）
                if len(gfs_bias) > 0:
                    bp_gfs = ax.boxplot([gfs_bias], positions=[position_counter], widths=0.5,
                                      patch_artist=True,
                                      boxprops=dict(facecolor='none', edgecolor=color_scheme[height]['gfs'], linewidth=1.5),
                                      whiskerprops=dict(color=color_scheme[height]['gfs'], linewidth=1.5),
                                      capprops=dict(color=color_scheme[height]['gfs'], linewidth=1.5),
                                      medianprops=dict(color=color_scheme[height]['gfs'], linewidth=2),
                                      flierprops=dict(marker='o', markerfacecolor='none',
                                                     markeredgecolor=color_scheme[height]['gfs'], markersize=3))
                
                position_counter += 1.4  # 不同高度之间的距离较远
            else:
                # 如果数据缺失，跳过相应位置
                position_counter += 2
        
        # 记录每个月的中心位置，用于X轴标签
        month_center = month_start_pos + (n_heights * 2 - 1) / 2
        positions_info.append((month_center, month.strftime('%b %Y')))
        
        # 在月份之间添加间隔
        position_counter += 1
    
    # 设置X轴
    month_centers, month_labels = zip(*positions_info)
    ax.set_xticks(month_centers)
    ax.set_xticklabels(month_labels, rotation=45, ha='right', fontsize=23)
    # ax.set_xlabel('Month', fontsize=22, weight='normal')
    
    # 添加零线
    ax.axhline(y=0, color='gray', linestyle='-', alpha=0.7, linewidth=1)
    
    # 设置Y轴
    ax.set_ylabel('Forecast Bias (m·s$^{-1}$)', fontsize=25, weight='normal')
    ax.tick_params(axis='y', labelsize=23)  # 设置Y轴刻度标签字体大小为14
    ax.set_ylim(-10, 10)
    ax.grid(False)
    
    # # 创建图例
    from matplotlib.lines import Line2D
    legend_elements = []
    
    # 高度图例 - 只有10m和70m
    legend_elements.extend([
        Line2D([0], [0], color=color_scheme['10m']['ec'], lw=3, label='10m EC-WRF'),
        Line2D([0], [0], color=color_scheme['10m']['gfs'], lw=3, label='10m GFS-WRF'),
        Line2D([0], [0], color=color_scheme['70m']['ec'], lw=3, label='70m EC-WRF'),
        Line2D([0], [0], color=color_scheme['70m']['gfs'], lw=3, label='70m GFS-WRF')
    ])
    
    # 添加分隔线
    # legend_elements.append(Line2D([0], [0], color='white', lw=0, label=''))
    
    # # 模型图例  
    # legend_elements.extend([
    #     Line2D([0], [0], color='black', lw=3, label='ECMWF (Dark)'),
    #     Line2D([0], [0], color='gray', lw=3, label='GFS (Light)')
    # ])
    
    # ax.legend(handles=legend_elements, loc='upper right', fontsize=10, 
    #          frameon=True, facecolor='white', framealpha=0.9, ncol=1)
    ax.legend(
        handles=legend_elements,
        loc='upper center',          # 水平居中
        bbox_to_anchor=(0.5, 1.15),  # 向上挪动一点，1.15 可调
        fontsize=25,
        frameon=True,
        facecolor='white',
        framealpha=0.9,
        ncol=4                       # 一行显示多个
    )

    
    # 设置标题
    # plt.title('3.2.2 Seasonal Forecast Bias Characteristics\n(All Heights Combined - Color by Height, Shade by Model)', 
    #          fontsize=16, weight='bold', pad=20)
    
    # 调整布局
    plt.tight_layout()
    
    return fig, ax

def main():
    """主函数"""
    # 数据路径
    data_path = '/Users/xiaxin/work/WindForecast_Project/01_Data/processed/matched_data/changma_matched.csv'
    
    try:
        # 加载数据
        print("Loading data...")
        df = pd.read_csv(data_path)
        df['datetime'] = pd.to_datetime(df['datetime'])
        df = df.set_index('datetime')
        
        print(f"Data loaded successfully. Shape: {df.shape}")
        print(f"Time range: {df.index.min()} to {df.index.max()}")
        
        # 创建统一季节特征盒须图
        print("\nCreating unified seasonal boxplot...")
        fig, ax = create_unified_seasonal_boxplot(df)
        
        # 保存图片
        save_path = '/Users/xiaxin/work/WindForecast_Project/03_Results/figures/3.2resutls/seasonal_boxplot_3_2_2.png'
        savepdf_path = '/Users/xiaxin/work/WindForecast_Project/03_Results/figures/3.2resutls/seasonal_boxplot_3_2_2.pdf'
        plt.savefig(save_path, dpi=600, bbox_inches='tight')
        plt.savefig(savepdf_path, bbox_inches='tight', facecolor='white')
        print(f"Unified seasonal boxplot saved: {save_path}")
        
        # 显示图片
        plt.show()
        
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()