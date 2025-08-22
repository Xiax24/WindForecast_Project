import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from sklearn.metrics import mean_squared_error
import warnings
warnings.filterwarnings('ignore')

# 设置中文字体和图形样式
plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial Unicode MS', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False
sns.set_style("whitegrid")
plt.rcParams['figure.dpi'] = 300

def load_and_preprocess_data(file_path):
    """
    加载数据并进行预处理
    """
    print("正在加载数据...")
    df = pd.read_csv(file_path)
    print(f"数据形状: {df.shape}")
    print(f"列名: {df.columns.tolist()}")
    
    # 检查数据完整性
    print(f"\n数据完整性检查:")
    print(f"总行数: {len(df)}")
    print(f"windshear缺失值: {df['windshear'].isna().sum()}")
    print(f"actual_power缺失值: {df['actual_power'].isna().sum()}")
    
    # 移除包含缺失值的行
    df_clean = df.dropna()
    print(f"清理后数据行数: {len(df_clean)}")
    
    return df_clean

def classify_windshear_with_time(df):
    """
    根据风切变值和时间对数据进行分类（6类）
    """
    print("\n进行风切变和昼夜分类...")
    
    # 确保datetime列是datetime类型
    df['datetime'] = pd.to_datetime(df['datetime'])
    
    # 提取小时信息
    df['hour'] = df['datetime'].dt.hour
    
    # 定义昼夜：白天6:00-18:00，夜间18:00-6:00
    df['time_period'] = df['hour'].apply(lambda x: 'Day' if 6 <= x < 18 else 'Night')
    
    # 风切变分类
    conditions = [
        df['windshear'] < 0.1,
        (df['windshear'] >= 0.1) & (df['windshear'] <= 0.3),
        df['windshear'] > 0.3
    ]
    
    windshear_categories = ['Low Shear', 'Moderate Shear', 'High Shear']
    df['windshear_base'] = np.select(conditions, windshear_categories, default='Unclassified')
    
    # 组合风切变和昼夜分类
    df['windshear_category'] = df['windshear_base'] + ' ' + df['time_period']
    
    # 统计各类别数量
    category_counts = df['windshear_category'].value_counts()
    print("风切变和昼夜分类统计:")
    total_samples = len(df)
    
    # 按照逻辑顺序排序
    ordered_categories = [
        'Low Shear Day', 'Low Shear Night',
        'Moderate Shear Day', 'Moderate Shear Night', 
        'High Shear Day', 'High Shear Night'
    ]
    
    for category in ordered_categories:
        if category in category_counts:
            count = category_counts[category]
            percentage = count / total_samples * 100
            print(f"- {category}: {count} 条 ({percentage:.1f}%)")
        else:
            print(f"- {category}: 0 条 (0.0%)")
    
    return df

def get_prediction_columns(df):
    """
    获取所有预测策略列名（按指定顺序）
    """
    # 指定的策略顺序
    desired_order = [
        'Fusion-M2',
        'X-M2-10m', 'X-M2-70m', 
        'E-M2-10m', 'E-M2-70m', 
        'G-M2-10m', 'G-M2-70m', 
        'E-M1-10m', 'E-M1-70m', 
        'G-M1-10m', 'G-M1-70m'
    ]
    
    # 排除基础列，获取预测策略列
    exclude_cols = ['datetime', 'windshear', 'actual_power', 'windshear_category', 
                   'hour', 'time_period', 'windshear_base']
    available_cols = [col for col in df.columns if col not in exclude_cols]
    
    # 按照指定顺序排列，只包含实际存在的列
    prediction_cols = [col for col in desired_order if col in available_cols]
    
    # 添加任何不在指定顺序中但存在的列（如果有的话）
    remaining_cols = [col for col in available_cols if col not in prediction_cols]
    prediction_cols.extend(remaining_cols)
    
    print(f"\n识别到的预测策略: {len(prediction_cols)} 个（按指定顺序排列）")
    for i, col in enumerate(prediction_cols, 1):
        print(f"  {i}. {col}")
    
    return prediction_cols

def calculate_rmse_by_category(df, prediction_cols):
    """
    计算每个风切变+昼夜类别下各策略的RMSE
    """
    print("\n计算各风切变+昼夜类别下的RMSE...")
    
    categories = [
        'Low Shear Day', 'Low Shear Night',
        'Moderate Shear Day', 'Moderate Shear Night', 
        'High Shear Day', 'High Shear Night'
    ]
    results = {}
    
    for category in categories:
        category_data = df[df['windshear_category'] == category]
        category_rmse = {}
        
        print(f"\n{category} (n={len(category_data)}):")
        
        if len(category_data) == 0:
            print("  无数据")
            for strategy in prediction_cols:
                category_rmse[strategy] = np.nan
            results[category] = category_rmse
            continue
        
        for strategy in prediction_cols:
            # 移除该策略下的缺失值
            valid_data = category_data.dropna(subset=[strategy, 'actual_power'])
            
            if len(valid_data) > 0:
                rmse = np.sqrt(mean_squared_error(valid_data['actual_power'], valid_data[strategy]))
                category_rmse[strategy] = rmse
                print(f"  {strategy}: {rmse:.4f}")
            else:
                category_rmse[strategy] = np.nan
                print(f"  {strategy}: 无有效数据")
        
        results[category] = category_rmse
    
    return results

def create_rmse_comparison_table(rmse_results, output_dir):
    """
    创建RMSE对比表
    """
    print("\n创建RMSE对比表...")
    
    # 转换为DataFrame
    rmse_df = pd.DataFrame(rmse_results).round(4)
    
    # 保存详细表格
    table_path = f"{output_dir}/windshear_rmse_detailed_6categories.csv"
    rmse_df.to_csv(table_path, encoding='utf-8-sig')
    print(f"详细RMSE表格已保存: {table_path}")
    
    # 创建排序表格
    ranking_results = {}
    for category in rmse_df.columns:
        if not rmse_df[category].isna().all():  # 如果该类别有有效数据
            sorted_strategies = rmse_df[category].sort_values()
            ranking_results[f"{category}_Strategy"] = sorted_strategies.index.tolist()
            ranking_results[f"{category}_RMSE"] = sorted_strategies.values.tolist()
    
    if ranking_results:
        ranking_df = pd.DataFrame(dict([(k, pd.Series(v)) for k, v in ranking_results.items()]))
        ranking_path = f"{output_dir}/windshear_strategy_ranking_6categories.csv"
        ranking_df.to_csv(ranking_path, index=False, encoding='utf-8-sig')
        print(f"策略排序表格已保存: {ranking_path}")
    else:
        ranking_df = pd.DataFrame()
    
    return rmse_df, ranking_df

def create_sample_size_table(df, output_dir):
    """
    创建数据量统计表
    """
    print("\n创建数据量统计表...")
    
    # 创建数据量统计表
    category_counts = df['windshear_category'].value_counts()
    
    # 按照逻辑顺序重新排序
    ordered_categories = [
        'Low Shear Day', 'Low Shear Night',
        'Moderate Shear Day', 'Moderate Shear Night', 
        'High Shear Day', 'High Shear Night'
    ]
    
    sample_data = []
    for category in ordered_categories:
        count = category_counts.get(category, 0)
        percentage = count / len(df) * 100
        sample_data.append({
            'Wind Shear Category': category,
            'Sample Size': count,
            'Percentage (%)': round(percentage, 1)
        })
    
    sample_size_df = pd.DataFrame(sample_data)
    
    # 保存数据量统计表
    sample_size_path = f"{output_dir}/windshear_sample_size_6categories.csv"
    sample_size_df.to_csv(sample_size_path, index=False, encoding='utf-8-sig')
    print(f"风切变分类数据量统计表已保存: {sample_size_path}")
    
    # 打印数据量统计
    print(f"\n数据量统计:")
    print(sample_size_df.to_string(index=False))
    
    return sample_size_df

def perform_significance_tests(df, prediction_cols, output_dir):
    """
    进行显著性检验
    """
    print("\n进行显著性检验...")
    
    categories = [
        'Low Shear Day', 'Low Shear Night',
        'Moderate Shear Day', 'Moderate Shear Night', 
        'High Shear Day', 'High Shear Night'
    ]
    significance_results = {}
    
    for category in categories:
        category_data = df[df['windshear_category'] == category]
        
        if len(category_data) < 10:  # 样本太小跳过
            print(f"\n{category}: 样本量太小，跳过显著性检验")
            continue
            
        print(f"\n{category}显著性检验:")
        category_tests = {}
        
        # 对策略两两进行t检验
        for i, strategy1 in enumerate(prediction_cols):
            for strategy2 in prediction_cols[i+1:]:
                # 计算残差
                valid_data = category_data.dropna(subset=[strategy1, strategy2, 'actual_power'])
                
                if len(valid_data) > 10:
                    residuals1 = valid_data['actual_power'] - valid_data[strategy1]
                    residuals2 = valid_data['actual_power'] - valid_data[strategy2]
                    
                    # 配对t检验
                    t_stat, p_value = stats.ttest_rel(np.abs(residuals1), np.abs(residuals2))
                    
                    test_key = f"{strategy1} vs {strategy2}"
                    category_tests[test_key] = {
                        't_statistic': t_stat,
                        'p_value': p_value,
                        'significant': p_value < 0.05
                    }
                    
                    if p_value < 0.05:
                        better_strategy = strategy1 if np.mean(np.abs(residuals1)) < np.mean(np.abs(residuals2)) else strategy2
                        print(f"  {test_key}: p={p_value:.4f} (显著, {better_strategy}更好)")
        
        significance_results[category] = category_tests
    
    # 保存显著性检验结果
    significance_summary = []
    for category, tests in significance_results.items():
        for test_pair, result in tests.items():
            significance_summary.append({
                'Windshear Category': category,
                'Strategy Comparison': test_pair,
                't-statistic': result['t_statistic'],
                'p-value': result['p_value'],
                'Significant': result['significant']
            })
    
    if significance_summary:
        sig_df = pd.DataFrame(significance_summary)
        sig_path = f"{output_dir}/significance_tests_6categories.csv"
        sig_df.to_csv(sig_path, index=False, encoding='utf-8-sig')
        print(f"显著性检验结果已保存: {sig_path}")
    
    return significance_results

def create_visualizations(rmse_df, df, output_dir):
    """
    创建可视化图表
    """
    print("\n创建可视化图表...")
    
    # 1. RMSE热力图
    plt.figure(figsize=(16, 10))
    
    # 过滤掉全为NaN的列
    valid_columns = rmse_df.columns[~rmse_df.isna().all()]
    rmse_df_clean = rmse_df[valid_columns]
    
    if not rmse_df_clean.empty:
        sns.heatmap(rmse_df_clean.T, annot=True, cmap='RdYlBu_r', fmt='.4f', 
                    cbar_kws={'label': 'RMSE'})
        plt.title('RMSE Comparison Across Wind Shear and Time Conditions', fontsize=16, fontweight='bold')
        plt.xlabel('Prediction Strategy', fontsize=12)
        plt.ylabel('Wind Shear & Time Category', fontsize=12)
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        plt.savefig(f"{output_dir}/rmse_heatmap_6categories.png", dpi=300, bbox_inches='tight')
        plt.show()
    
    # 2. 分组条形图对比所有策略在6类条件下的表现
    categories = [
        'Low Shear Day', 'Low Shear Night',
        'Moderate Shear Day', 'Moderate Shear Night', 
        'High Shear Day', 'High Shear Night'
    ]
    
    # 使用seaborn风格的渐变配色方案
    colors = [
        '#B3D9FF',  # 中浅蓝色 - Low Shear Day
        "#1379CD",  # 钢蓝色 - Low Shear Night
        '#C8E6C9',  # 中浅绿色 - Moderate Shear Day  
        "#0BAB50",  # 海绿色 - Moderate Shear Night
        '#E1BEE7',  # 中浅紫色 - High Shear Day
        '#8A2BE2'   # 深紫色 - High Shear Night
    ]
    
    # 获取所有策略
    all_strategies = rmse_df.index.tolist()
    
    # 准备数据
    data_for_plot = []
    for strategy in all_strategies:
        strategy_data = []
        for category in categories:
            if category in rmse_df.columns:
                value = rmse_df.loc[strategy, category]
                strategy_data.append(value if not pd.isna(value) else 0)
            else:
                strategy_data.append(0)
        data_for_plot.append(strategy_data)
    
    # 转换为numpy数组
    data_array = np.array(data_for_plot)
    
    # 设置图形
    fig, ax = plt.subplots(figsize=(16, 8))
    
    # 设置条形图参数
    x = np.arange(len(all_strategies))
    width = 0.12  # 每个柱子的宽度
    
    # 绘制分组条形图
    for i, (category, color) in enumerate(zip(categories, colors)):
        offset = (i - 2.5) * width  # 居中偏移
        bars = ax.bar(x + offset, data_array[:, i], width, 
                     label=category, color=color, alpha=0.8)
        
        # 在柱子上标注数值（只标注非零值）
        for j, bar in enumerate(bars):
            height = bar.get_height()
            if height > 0:
                ax.text(bar.get_x() + bar.get_width()/2., height + 0.5,
                       f'{height:.1f}', ha='center', va='bottom', fontsize=7, rotation=0)
    
    # 设置标签和标题
    ax.set_xlabel('Prediction Strategy', fontsize=12, fontweight='bold')
    ax.set_ylabel('RMSE', fontsize=12, fontweight='bold')
    ax.set_title('RMSE Comparison: All Strategies Across 6 Wind Shear & Time Conditions', 
                fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(all_strategies, rotation=45, ha='right')
    
    # 添加图例
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    
    # 添加网格
    ax.grid(True, alpha=0.3, axis='y')
    
    # 设置y轴范围
    ax.set_ylim(0, data_array.max() * 1.15)
    
    plt.tight_layout()
    plt.savefig(f"{output_dir}/rmse_grouped_bar_chart_6categories.png", dpi=300, bbox_inches='tight')
    plt.show()
    
    # 3. 昼夜和风切变分布图
    plt.figure(figsize=(15, 6))
    
    # 子图1：风切变指数分布
    plt.subplot(1, 3, 1)
    df['windshear'].hist(bins=50, alpha=0.7, color='skyblue', edgecolor='black')
    plt.axvline(x=0.1, color='red', linestyle='--', label='Low Shear Threshold (0.1)')
    plt.axvline(x=0.3, color='orange', linestyle='--', label='High Shear Threshold (0.3)')
    plt.xlabel('Wind Shear Index')
    plt.ylabel('Frequency')
    plt.title('Wind Shear Index Distribution')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # 子图2：昼夜分布
    plt.subplot(1, 3, 2)
    time_counts = df['time_period'].value_counts()
    plt.pie(time_counts.values, labels=time_counts.index, autopct='%1.1f%%', 
            colors=['gold', 'darkblue'], startangle=90)
    plt.title('Day/Night Distribution')
    
    # 子图3：6类组合分布
    plt.subplot(1, 3, 3)
    category_counts = df['windshear_category'].value_counts()
    ordered_categories = [
        'Low Shear Day', 'Low Shear Night',
        'Moderate Shear Day', 'Moderate Shear Night', 
        'High Shear Day', 'High Shear Night'
    ]
    
    # 按顺序获取数据
    ordered_counts = [category_counts.get(cat, 0) for cat in ordered_categories]
    
    plt.bar(range(len(ordered_categories)), ordered_counts, 
            color=['lightcoral', 'darkred', 'skyblue', 'darkblue', 'lightgreen', 'darkgreen'],
            alpha=0.8)
    plt.xlabel('Wind Shear & Time Category')
    plt.ylabel('Sample Count')
    plt.title('6-Category Distribution')
    plt.xticks(range(len(ordered_categories)), 
               [cat.replace(' ', '\n') for cat in ordered_categories], 
               rotation=45, ha='right')
    
    # 添加数值标签
    for i, count in enumerate(ordered_counts):
        plt.text(i, count + 50, str(count), ha='center', va='bottom')
    
    plt.tight_layout()
    plt.savefig(f"{output_dir}/distribution_6categories.png", dpi=300, bbox_inches='tight')
    plt.show()

def generate_analysis_summary(rmse_df, significance_results, df, output_dir):
    """
    生成分析总结报告
    """
    print("\n生成分析总结报告...")
    
    summary_text = []
    summary_text.append("# 风切变和昼夜条件下预测策略性能分析报告\n")
    
    # 数据概况
    summary_text.append("## 1. 数据概况")
    summary_text.append(f"- 总数据量: {len(df)} 条")
    
    category_counts = df['windshear_category'].value_counts()
    ordered_categories = [
        'Low Shear Day', 'Low Shear Night',
        'Moderate Shear Day', 'Moderate Shear Night', 
        'High Shear Day', 'High Shear Night'
    ]
    
    for category in ordered_categories:
        count = category_counts.get(category, 0)
        percentage = count / len(df) * 100
        summary_text.append(f"- {category}: {count} 条 ({percentage:.1f}%)")
    summary_text.append("")
    
    # 各条件下的最优策略
    summary_text.append("## 2. 各风切变和昼夜条件下的最优策略")
    for category in rmse_df.columns:
        if not rmse_df[category].isna().all():
            valid_rmse = rmse_df[category].dropna()
            if len(valid_rmse) > 0:
                best_strategy = valid_rmse.idxmin()
                best_rmse = valid_rmse.min()
                worst_strategy = valid_rmse.idxmax()
                worst_rmse = valid_rmse.max()
                
                summary_text.append(f"### {category}")
                summary_text.append(f"- 最优策略: {best_strategy} (RMSE: {best_rmse:.4f})")
                summary_text.append(f"- 最差策略: {worst_strategy} (RMSE: {worst_rmse:.4f})")
                summary_text.append(f"- 性能差距: {((worst_rmse - best_rmse) / best_rmse * 100):.1f}%")
                summary_text.append("")
        else:
            summary_text.append(f"### {category}")
            summary_text.append("- 无有效数据")
            summary_text.append("")
    
    # 昼夜对比分析
    summary_text.append("## 3. 昼夜对比分析")
    summary_text.append("### 各风切变条件下的昼夜差异:")
    
    for base_category in ['Low Shear', 'Moderate Shear', 'High Shear']:
        day_category = f"{base_category} Day"
        night_category = f"{base_category} Night"
        
        if day_category in rmse_df.columns and night_category in rmse_df.columns:
            day_rmse = rmse_df[day_category].dropna()
            night_rmse = rmse_df[night_category].dropna()
            
            if len(day_rmse) > 0 and len(night_rmse) > 0:
                day_best = day_rmse.min()
                night_best = night_rmse.min()
                day_best_strategy = day_rmse.idxmin()
                night_best_strategy = night_rmse.idxmin()
                
                summary_text.append(f"#### {base_category}")
                summary_text.append(f"- 白天最优: {day_best_strategy} (RMSE: {day_best:.4f})")
                summary_text.append(f"- 夜间最优: {night_best_strategy} (RMSE: {night_best:.4f})")
                summary_text.append(f"- 昼夜差异: {abs(day_best - night_best):.4f}")
                summary_text.append("")
    
    summary_text.append("")
    
    # 保存总结报告
    summary_path = f"{output_dir}/analysis_summary_6categories.md"
    with open(summary_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(summary_text))
    
    print(f"分析总结报告已保存: {summary_path}")
    
    return summary_text

def main():
    """
    主分析流程
    """
    # 文件路径
    data_path = "/Users/xiaxin/work/WindForecast_Project/01_Data/processed/imputed_data/windshear.csv"
    output_dir = "/Users/xiaxin/work/WindForecast_Project/03_Results/03-08"
    
    # 创建输出目录
    import os
    os.makedirs(output_dir, exist_ok=True)
    
    try:
        # 1. 加载和预处理数据
        df = load_and_preprocess_data(data_path)
        
        # 2. 风切变和昼夜分类
        df = classify_windshear_with_time(df)
        
        # 3. 获取预测策略列
        prediction_cols = get_prediction_columns(df)
        
        # 4. 计算各类别下的RMSE
        rmse_results = calculate_rmse_by_category(df, prediction_cols)
        
        # 5. 创建对比表格
        rmse_df, ranking_df = create_rmse_comparison_table(rmse_results, output_dir)
        
        # 5.1 创建数据量统计表
        sample_size_df = create_sample_size_table(df, output_dir)
        
        # 6. 显著性检验
        significance_results = perform_significance_tests(df, prediction_cols, output_dir)
        
        # 7. 创建可视化
        create_visualizations(rmse_df, df, output_dir)
        
        # 8. 生成分析总结
        summary = generate_analysis_summary(rmse_df, significance_results, df, output_dir)
        
        print("\n" + "="*60)
        print("6类分析完成！所有结果已保存到:")
        print(f"{output_dir}")
        print("="*60)
        
        # 显示关键发现
        print("\n关键发现 (6类分析):")
        for category in rmse_df.columns:
            if not rmse_df[category].isna().all():
                valid_rmse = rmse_df[category].dropna()
                if len(valid_rmse) > 0:
                    best_strategy = valid_rmse.idxmin()
                    best_rmse = valid_rmse.min()
                    print(f"- {category}: {best_strategy} (RMSE: {best_rmse:.4f})")
        
    except Exception as e:
        print(f"分析过程中出现错误: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()