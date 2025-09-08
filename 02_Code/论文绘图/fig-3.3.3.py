import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import mean_squared_error
import warnings
warnings.filterwarnings('ignore')

# 设置图形样式 - 全局设置Arial字体
plt.rcParams['font.family'] = 'Arial'
plt.rcParams['font.sans-serif'] = ['Arial', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['figure.dpi'] = 300

def load_and_classify_data(file_path):
    """
    加载数据并进行风切变和昼夜分类
    """
    print("Loading and processing data...")
    df = pd.read_csv(file_path)
    df_clean = df.dropna()
    
    # 确保datetime列是datetime类型
    df_clean['datetime'] = pd.to_datetime(df_clean['datetime'])
    df_clean['hour'] = df_clean['datetime'].dt.hour
    
    # 定义昼夜：白天6:00-18:00，夜间18:00-6:00
    df_clean['time_period'] = df_clean['hour'].apply(lambda x: 'Day' if 6 <= x < 18 else 'Night')
    
    # 风切变分类
    conditions = [
        df_clean['windshear'] < 0.1,
        (df_clean['windshear'] >= 0.1) & (df_clean['windshear'] <= 0.3),
        df_clean['windshear'] > 0.3
    ]
    
    windshear_categories = ['Low Shear', 'Moderate Shear', 'High Shear']
    df_clean['windshear_base'] = np.select(conditions, windshear_categories, default='Unclassified')
    
    # 组合风切变和昼夜分类
    df_clean['windshear_category'] = df_clean['windshear_base'] + ' ' + df_clean['time_period']
    
    return df_clean

def calculate_rmse_by_category(df):
    """
    计算每个风切变+昼夜类别下各策略的RMSE
    """
    # 策略顺序（根据你的表格排名）
    prediction_cols = [
        'Fusion-M2',
        'X-M2-10m', 'X-M2-70m', 
        'E-M2-10m', 'E-M2-70m', 
        'G-M2-10m', 'G-M2-70m', 
        'E-M1-10m', 'E-M1-70m', 
        'G-M1-10m', 'G-M1-70m'
    ]
    
    categories = [
        'Low Shear Day', 'Low Shear Night',
        'Moderate Shear Day', 'Moderate Shear Night', 
        'High Shear Day', 'High Shear Night'
    ]
    
    results = {}
    
    for category in categories:
        category_data = df[df['windshear_category'] == category]
        category_rmse = {}
        
        print(f"{category} (n={len(category_data)}):")
        
        for strategy in prediction_cols:
            if strategy in df.columns:
                valid_data = category_data.dropna(subset=[strategy, 'actual_power'])
                if len(valid_data) > 0:
                    rmse = np.sqrt(mean_squared_error(valid_data['actual_power'], valid_data[strategy]))
                    category_rmse[strategy] = rmse
                    print(f"  {strategy}: {rmse:.4f}")
                else:
                    category_rmse[strategy] = np.nan
            else:
                category_rmse[strategy] = np.nan
        
        results[category] = category_rmse
    
    return results, prediction_cols

def create_rmse_grouped_bar_chart(rmse_results, prediction_cols, output_base_path=None):
    """
    创建RMSE分组条形图，支持保存为多种格式
    """
    categories = [
        'Low Shear Day', 'Low Shear Night',
        'Moderate Shear Day', 'Moderate Shear Night', 
        'High Shear Day', 'High Shear Night'
    ]
    
    # 改进的配色方案 - 更加专业和学术化
    colors = [
        '#B3D9FF',  # 浅蓝色 - Low Shear Day
        '#1E88E5',  # 深蓝色 - Low Shear Night
        '#C8E6C9',  # 浅绿色 - Moderate Shear Day  
        '#43A047',  # 深绿色 - Moderate Shear Night
        '#E1BEE7',  # 浅紫色 - High Shear Day
        '#8E24AA'   # 深紫色 - High Shear Night
    ]

    # 准备数据
    data_for_plot = []
    for strategy in prediction_cols:
        strategy_data = []
        for category in categories:
            if category in rmse_results:
                value = rmse_results[category].get(strategy, np.nan)
                strategy_data.append(value if not pd.isna(value) else 0)
            else:
                strategy_data.append(0)
        data_for_plot.append(strategy_data)
    
    # 转换为numpy数组
    data_array = np.array(data_for_plot)
    
    # 设置图形 - 调整为更合适的尺寸
    fig, ax = plt.subplots(figsize=(16, 8))
    
    # 设置条形图参数
    x = np.arange(len(prediction_cols))
    width = 0.13  # 调整柱子宽度
    
    # 绘制分组条形图
    for i, (category, color) in enumerate(zip(categories, colors)):
        offset = (i - 2.5) * width  # 居中偏移
        
        # 绘制所有策略的柱子
        bars = ax.bar(x + offset, data_array[:, i], width, 
                     color=color, alpha=0.85, edgecolor='white', linewidth=0.5)
        
        # 特别处理Fusion-M2柱子（第一个策略）- 在绘制后单独设置样式
        fusion_bar = bars[0]  # Fusion-M2是第一个策略
        fusion_bar.set_edgecolor('black')
        fusion_bar.set_linewidth(2.0)
        fusion_bar.set_alpha(0.95)
    
    # 设置标签和标题 - 使用全局Arial字体设置
    ax.set_ylabel('RMSE (MW)', fontsize=22, fontweight='normal')
    ax.set_xticks(x)
    ax.set_xticklabels(prediction_cols, rotation=45, ha='right', fontsize=22)
    
    # 图例处理 - 创建干净的图例，不继承柱子的边框样式
    legend_elements = []
    for category, color in zip(categories, colors):
        legend_elements.append(plt.Rectangle((0,0),1,1, facecolor=color, alpha=0.85, 
                                           edgecolor='white', linewidth=0.5))
    
    ax.legend(legend_elements, categories, loc='upper center', bbox_to_anchor=(0.5, 1.0), 
              fontsize=22, frameon=True, fancybox=True, shadow=True, ncol=3)
    
    # 添加网格 - 更细致的网格
    ax.grid(True, which='both', linestyle='--', linewidth=0.7, alpha=0.7)
    ax.set_axisbelow(True)
    
    # 设置y轴范围和刻度 - 留出更多空间给标签和注释
    valid_data = data_array[data_array > 0]
    if len(valid_data) > 0:
        y_max = valid_data.max() * 1.3
        ax.set_ylim(0, y_max)
        # 设置y轴刻度间隔为5
        ax.set_yticks(np.arange(0, y_max + 1, 5))

    # 改进坐标轴外观 - 全局字体设置会自动应用
    ax.tick_params(axis='both', which='major', labelsize=22)
    
    plt.tight_layout()
    
    # 保存多种格式的图片
    if output_base_path:
        # 去掉原有的扩展名（如果有的话）
        base_path = output_base_path.rsplit('.', 1)[0]  # 移除扩展名
        
        # 保存PNG格式 - 高分辨率，适合网页和演示
        png_path = f"{base_path}.png"
        plt.savefig(png_path, dpi=300, bbox_inches='tight', facecolor='white', format='png')
        print(f"PNG chart saved to: {png_path}")
        
        # 保存PDF格式 - 矢量图，适合论文和打印
        pdf_path = f"{base_path}.pdf"
        plt.savefig(pdf_path, bbox_inches='tight', facecolor='white', format='pdf')
        print(f"PDF chart saved to: {pdf_path}")
        
        # 可选：也保存SVG格式 - 矢量图，适合编辑
        # svg_path = f"{base_path}.svg"
        # plt.savefig(svg_path, bbox_inches='tight', facecolor='white', format='svg')
        # print(f"SVG chart saved to: {svg_path}")
        
        # 可选：也保存EPS格式 - 适合学术期刊
        # eps_path = f"{base_path}.eps"
        # plt.savefig(eps_path, bbox_inches='tight', facecolor='white', format='eps')
        # print(f"EPS chart saved to: {eps_path}")
    
    plt.show()
    
    return fig, ax

def main():
    """
    主函数 - 生成风切变条件下的RMSE对比图
    """
    # 文件路径 - 请根据你的实际路径修改
    data_path = "/Users/xiaxin/work/WindForecast_Project/01_Data/processed/imputed_data/windshear.csv"
    # 注意：这里只需要提供基础路径，不需要扩展名
    output_base_path = "/Users/xiaxin/work/WindForecast_Project/03_Results/figures/windshear_rmse_comparison"
    
    try:
        # 1. 加载和分类数据
        df = load_and_classify_data(data_path)
        
        # 2. 计算各类别下的RMSE
        rmse_results, prediction_cols = calculate_rmse_by_category(df)
        
        # 3. 创建可视化 - 现在会自动保存PNG和PDF两种格式
        fig, ax = create_rmse_grouped_bar_chart(rmse_results, prediction_cols, output_base_path)
        
        print("Visualization completed successfully!")
        print("Files saved:")
        print("  - PNG format: suitable for presentations and web")
        print("  - PDF format: suitable for academic papers and printing")
        
        # 显示数据摘要
        print("\nData Summary:")
        category_counts = df['windshear_category'].value_counts()
        for category in ['Low Shear Day', 'Low Shear Night', 'Moderate Shear Day', 
                        'Moderate Shear Night', 'High Shear Day', 'High Shear Night']:
            count = category_counts.get(category, 0)
            print(f"- {category}: {count} samples")
        
    except Exception as e:
        print(f"Error occurred: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()