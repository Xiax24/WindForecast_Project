import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

# 设置图形样式
plt.rcParams['font.family'] = 'Arial'
plt.rcParams['font.sans-serif'] = ['Arial', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['figure.dpi'] = 600

def load_data(file_path):
    """加载数据"""
    df = pd.read_csv(file_path)
    # 确保按高度排序
    layer_order = ['10m', '30m', '50m', '70m']
    df['Layer'] = pd.Categorical(df['Layer'], categories=layer_order, ordered=True)
    df = df.sort_values('Layer').reset_index(drop=True)
    return df

def create_triple_axis_chart(df, output_base_path):
    """
    创建三轴点线图：左y轴-相关系数，右y轴1-RMSE，右y轴2-偏差
    """
    fig, ax1 = plt.subplots(figsize=(11, 8))
    
    # 准备x轴数据
    layers = df['Layer'].tolist()
    x_pos = np.arange(len(layers))
    
    # 颜色设置
    corr_color = 'black'    # 蓝色 - 相关系数
    rmse_color = '#1f77b4'    # 红色 - RMSE  
    bias_color = '#2ca02c'    # 绿色 - 偏差
    
    # 线型设置
    ec_style = '-'    # 实线 - ECMWF
    gfs_style = '-'  # 虚线 - GFS
    
    # 第一个y轴：相关系数
    ax1.set_xlabel('Height Layer', fontsize=23, fontweight='normal')
    ax1.set_ylabel('Correlation Coefficient', fontsize=23, fontweight='normal', color=corr_color)
    
    line1 = ax1.plot(x_pos, df['EC_Correlation'], 'o-', linewidth=1, markersize=18, 
                     color=corr_color, linestyle=ec_style, label='EC Correlation', alpha=0.8)
    line2 = ax1.plot(x_pos, df['GFS_Correlation'], 's-', linewidth=1, markersize=18,
                     color=corr_color, linestyle=gfs_style, label='GFS Correlation', alpha=0.4)
    
    ax1.tick_params(axis='y', labelcolor=corr_color, labelsize=12)
    ax1.spines['left'].set_color(corr_color)  # 左轴线颜色
    ax1.spines['left'].set_linewidth(2)       # 左轴线粗细
    # 隐藏不需要的轴线
    # ax1.spines['top'].set_visible(False)
    ax1.spines['top'].set_visible(True)
    # ax1.spines['top'].set_color('black')  # 或者你想要的颜色
    # ax1.spines['top'].set_linewidth(1)   # 设置线宽
    ax1.spines['right'].set_visible(False)
    ax1.set_ylim(0.55, 0.64)
    ax1.set_yticks(np.arange(0.55, 0.65, 0.01))
    ax1.grid(False)
    
    # 添加相关系数数值标签
    # for i, (ec_val, gfs_val) in enumerate(zip(df['EC_Correlation'], df['GFS_Correlation'])):
    #     ax1.text(i, ec_val + 0.003, f'{ec_val:.3f}', ha='center', va='bottom', 
    #             fontsize=20, color=corr_color, fontweight='bold')
    #     ax1.text(i, gfs_val - 0.004, f'{gfs_val:.3f}', ha='center', va='top',
    #             fontsize=20, color=corr_color, fontweight='bold')
    
    # 第二个y轴：RMSE
    ax2 = ax1.twinx()
    ax2.set_ylabel('RMSE (m·s$^{-1}$)', fontsize=23, fontweight='normal', color=rmse_color)
    
    line3 = ax2.plot(x_pos, df['EC_RMSE'], 'o-', linewidth=1, markersize=18,
                     color=rmse_color, linestyle=ec_style, label='EC RMSE', alpha=0.8)
    line4 = ax2.plot(x_pos, df['GFS_RMSE'], 's-', linewidth=1, markersize=18,
                     color=rmse_color, linestyle=gfs_style, label='GFS RMSE', alpha=0.4)
    
    ax2.tick_params(axis='y', labelcolor=rmse_color, labelsize=12)
    # 隐藏ax2不需要的轴线，只保留右边的红色轴
    ax2.spines['top'].set_visible(False)
    ax2.spines['left'].set_visible(False)
    ax2.spines['bottom'].set_visible(False)
    ax2.spines['right'].set_color(rmse_color)   # 右轴线颜色
    ax2.spines['right'].set_linewidth(2)        # 右轴线粗细
    ax2.set_ylim(2.2, 3.8)
    
    # 添加RMSE数值标签
    # for i, (ec_val, gfs_val) in enumerate(zip(df['EC_RMSE'], df['GFS_RMSE'])):
    #     ax2.text(i + 0.1, ec_val - 0.015, f'{ec_val:.2f}', ha='center', va='bottom',
    #             fontsize=20, color=rmse_color, fontweight='bold')
    #     ax2.text(i + 0.1, gfs_val + 0.005, f'{gfs_val:.2f}', ha='center', va='top',
    #             fontsize=20, color=rmse_color, fontweight='bold')
    
    # 第三个y轴：偏差（绝对值）
    ax3 = ax1.twinx()
    # 调整第三个轴的位置
    ax3.spines['right'].set_position(('outward', 80))
    ax3.set_ylabel('Absolute Bias (m·s$^{-1}$)', fontsize=23, fontweight='normal', color=bias_color)    
    ec_bias_abs = df['EC_bias'].abs()
    gfs_bias_abs = df['GFS_bias'].abs()
    
    line5 = ax3.plot(x_pos, ec_bias_abs, 'o-', linewidth=1, markersize=18,
                     color=bias_color, linestyle=ec_style, label='EC Bias', alpha=0.8)
    line6 = ax3.plot(x_pos, gfs_bias_abs, 's-', linewidth=1, markersize=18,
                     color=bias_color, linestyle=gfs_style, label='GFS Bias', alpha=0.4)
    
    ax3.tick_params(axis='y', labelcolor=bias_color, labelsize=12)
    # 隐藏ax3不需要的轴线，只保留右边的绿色轴
    ax3.spines['top'].set_visible(False)
    ax3.spines['left'].set_visible(False) 
    ax3.spines['bottom'].set_visible(False)
    ax3.spines['right'].set_color(bias_color)   # 第三轴线颜色  
    ax3.spines['right'].set_linewidth(2)        # 第三轴线粗细
    ax3.set_ylim(-0.1, 1.15)
    
    # 添加偏差数值标签
    # for i, (ec_val, gfs_val) in enumerate(zip(ec_bias_abs, gfs_bias_abs)):
    #     ax3.text(i - 0.1, ec_val - 0.03, f'{ec_val:.3f}', ha='center', va='bottom',
    #             fontsize=20, color=bias_color, fontweight='bold')
    #     ax3.text(i - 0.1, gfs_val + 0.04, f'{gfs_val:.3f}', ha='center', va='top',
    #             fontsize=20, color=bias_color, fontweight='bold')
    
    # 设置x轴
    ax1.set_xticks(x_pos)
    ax1.set_xticklabels(layers, fontsize=24, fontweight='normal')
    
    # 创建图例
    lines1 = line1 + line2 + line3 + line4 + line5 + line6
    labels1 = ['EC Correlation', 'GFS Correlation', 'EC RMSE', 'GFS RMSE', 'EC Bias', 'GFS Bias']
    ax1.legend(lines1, labels1, loc='upper center', fontsize=21, frameon=False, ncol=3,
              fancybox=False, shadow=False, columnspacing=0.1)  # 默认是2.0，改为0.5
    
    # 设置标题
    # plt.title('Wind Speed Forecast Performance Across Different Heights\nTriple-Axis Comparison of Key Metrics', 
    #           fontsize=16, fontweight='bold', pad=20)
    
    # 添加说明文本框
    # textstr = 'Performance Ranking (Best to Worst):\n• Correlation: Higher is better\n• RMSE: Lower is better\n• Bias: Lower absolute value is better'
    # props = dict(boxstyle='round', facecolor='wheat', alpha=0.8)
    # ax1.text(0.02, 0.98, textstr, transform=ax1.transAxes, fontsize=11,
    #         verticalalignment='top', bbox=props)
    
    # 高亮10m层（添加背景色）
    # ax1.axvspan(-0.4, 0.4, alpha=0.1, color='gold', label='Best Layer (10m)')
    # 第一个轴
    ax1.tick_params(axis='y', labelcolor=corr_color, labelsize=23)  # 从12改为16

    # 第二个轴  
    ax2.tick_params(axis='y', labelcolor=rmse_color, labelsize=23)  # 从12改为16

    # 第三个轴
    ax3.tick_params(axis='y', labelcolor=bias_color, labelsize=23)  # 从12改为16
    plt.tight_layout()
    
    # 保存文件
    png_path = f"{output_base_path}_triple_axis.png"
    pdf_path = f"{output_base_path}_triple_axis.pdf"
    
    plt.savefig(png_path, dpi=600, bbox_inches='tight', facecolor='white')
    plt.savefig(pdf_path, bbox_inches='tight', facecolor='white')
    
    print(f"Triple-axis chart saved:")
    print(f"  PNG: {png_path}")
    print(f"  PDF: {pdf_path}")
    
    plt.show()
    return fig, (ax1, ax2, ax3)

def create_simplified_triple_axis(df, output_base_path):
    """
    创建简化版三轴图，突出最优层级
    """
    fig, ax1 = plt.subplots(figsize=(12, 8))
    
    # 数据准备
    layers = df['Layer'].tolist()
    x_pos = np.arange(len(layers))
    
    # 计算综合性能指标（标准化后的加权平均）
    # 先标准化每个指标
    corr_norm = (df['EC_Correlation'] - df['EC_Correlation'].min()) / (df['EC_Correlation'].max() - df['EC_Correlation'].min())
    rmse_norm = 1 - (df['EC_RMSE'] - df['EC_RMSE'].min()) / (df['EC_RMSE'].max() - df['EC_RMSE'].min())  # 反转
    bias_norm = 1 - (df['EC_bias'].abs() - df['EC_bias'].abs().min()) / (df['EC_bias'].abs().max() - df['EC_bias'].abs().min())  # 反转
    
    # 计算加权综合评分
    ec_composite = (corr_norm * 0.4 + rmse_norm * 0.4 + bias_norm * 0.2) * 100  # 转换为百分制
    
    # 对GFS做同样处理
    corr_norm_gfs = (df['GFS_Correlation'] - df['GFS_Correlation'].min()) / (df['GFS_Correlation'].max() - df['GFS_Correlation'].min())
    rmse_norm_gfs = 1 - (df['GFS_RMSE'] - df['GFS_RMSE'].min()) / (df['GFS_RMSE'].max() - df['GFS_RMSE'].min())
    bias_norm_gfs = 1 - (df['GFS_bias'].abs() - df['GFS_bias'].abs().min()) / (df['GFS_bias'].abs().max() - df['GFS_bias'].abs().min())
    
    gfs_composite = (corr_norm_gfs * 0.4 + rmse_norm_gfs * 0.4 + bias_norm_gfs * 0.2) * 100
    
    # 主y轴：综合评分
    ax1.set_xlabel('Height Layer', fontsize=20, fontweight='bold')
    ax1.set_ylabel('Composite Performance Score', fontsize=20, fontweight='bold', color='purple')
    
    bars1 = ax1.bar(x_pos - 0.2, ec_composite, 0.35, label='ECMWF Overall', 
                    color='#1f77b4', alpha=0.7)
    bars2 = ax1.bar(x_pos + 0.2, gfs_composite, 0.35, label='GFS Overall',
                    color='#ff7f0e', alpha=0.7)
    
    ax1.set_ylim(0, 100)
    ax1.tick_params(axis='y', labelcolor='purple', labelsize=12)
    
    # 添加综合评分标签
    for bar in bars1:
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height + 1,
                f'{height:.1f}', ha='center', va='bottom', fontsize=20, fontweight='bold')
    for bar in bars2:
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height + 1,
                f'{height:.1f}', ha='center', va='bottom', fontsize=20, fontweight='bold')
    
    # 第二y轴：显示原始RMSE作为参考
    ax2 = ax1.twinx()
    ax2.set_ylabel('RMSE (m/s)', fontsize=20, fontweight='bold', color='red')
    
    ax2.plot(x_pos, df['EC_RMSE'], 'o-', linewidth=2, color='red', alpha=0.8, label='EC RMSE')
    ax2.plot(x_pos, df['GFS_RMSE'], 's--', linewidth=2, color='orange', alpha=0.8, label='GFS RMSE')
    ax2.tick_params(axis='y', labelcolor='red', labelsize=12)
    
    # x轴设置
    ax1.set_xticks(x_pos)
    ax1.set_xticklabels(layers, fontsize=20, fontweight='bold')
    ax1.grid(False)
    
    # 图例
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper right', fontsize=20)
    
    plt.title('Wind Speed Forecast Performance: Composite Score vs RMSE\nHigher Composite Score = Better Overall Performance', 
              fontsize=20, fontweight='bold', pad=20)
    
    plt.tight_layout()
    
    # 保存
    png_path = f"{output_base_path}_composite.png"
    pdf_path = f"{output_base_path}_composite.pdf"
    
    plt.savefig(png_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.savefig(pdf_path, bbox_inches='tight', facecolor='white')
    
    print(f"Composite score chart saved:")
    print(f"  PNG: {png_path}")  
    print(f"  PDF: {pdf_path}")
    
    plt.show()
    return fig, (ax1, ax2)

def main():
    """
    主函数
    """
    # 文件路径
    data_path = "/Users/xiaxin/work/WindForecast_Project/03_Results/figures/3.2resutls/layer_wise_summary.csv"
    output_base_path = "/Users/xiaxin/work/WindForecast_Project/03_Results/figures/3.2resutls/layer_comparison"
    
    try:
        # 1. 加载数据
        df = load_data(data_path)
        print("Data loaded:")
        print(df)
        
        # 2. 创建三轴点线图
        print("\nCreating triple-axis line chart...")
        fig1, (ax1, ax2, ax3) = create_triple_axis_chart(df, output_base_path)
        
        print("\nTriple-axis visualization completed!")
        print("The chart should clearly show 10m layer as the best performer.")
        
    except Exception as e:
        print(f"Error occurred: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()