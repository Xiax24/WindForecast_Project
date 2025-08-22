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

def classify_windshear(df):
    """
    根据风切变值对数据进行分类
    """
    print("\n进行风切变分类...")
    
    # 定义分类条件
    conditions = [
        df['windshear'] < 0.1,
        (df['windshear'] >= 0.1) & (df['windshear'] <= 0.3),
        df['windshear'] > 0.3
    ]
    
    choices = ['Low Shear', 'Moderate Shear', 'High Shear']
    
    df['windshear_category'] = np.select(conditions, choices, default='Unclassified')
    
    # 统计各类别数量
    category_counts = df['windshear_category'].value_counts()
    print("风切变分类统计:")
    for category, count in category_counts.items():
        percentage = count / len(df) * 100
        print(f"- {category}: {count} 条 ({percentage:.1f}%)")
        
        # 创建对应的英文标签用于图表
        english_labels = {
            'Low Shear': '弱切变',
            'Moderate Shear': '中等切变', 
            'High Shear': '强切变'
        }
    
    return df

def get_prediction_columns(df):
    """
    获取所有预测策略列名
    """
    # 排除基础列，获取预测策略列
    exclude_cols = ['datetime', 'windshear', 'actual_power', 'windshear_category']
    prediction_cols = [col for col in df.columns if col not in exclude_cols]
    
    print(f"\n识别到的预测策略: {len(prediction_cols)} 个")
    for i, col in enumerate(prediction_cols, 1):
        print(f"  {i}. {col}")
    
    return prediction_cols

def calculate_rmse_by_category(df, prediction_cols):
    """
    计算每个风切变类别下各策略的RMSE
    """
    print("\n计算各风切变类别下的RMSE...")
    
    categories = ['Low Shear', 'Moderate Shear', 'High Shear']
    results = {}
    
    for category in categories:
        category_data = df[df['windshear_category'] == category]
        category_rmse = {}
        
        print(f"\n{category} (n={len(category_data)}):")
        
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

def create_rmse_comparison_table(results, output_dir):
    """
    创建RMSE对比表
    """
    print("\n创建RMSE对比表...")
    
    # 转换为DataFrame
    rmse_df = pd.DataFrame(results).round(4)
    
    # 保存详细表格
    table_path = f"{output_dir}/windshear_rmse_detailed.csv"
    rmse_df.to_csv(table_path, encoding='utf-8-sig')
    print(f"详细RMSE表格已保存: {table_path}")
    
    # 创建排序表格
    ranking_results = {}
    for category in rmse_df.columns:
        sorted_strategies = rmse_df[category].sort_values()
        ranking_results[f"{category}_Strategy"] = sorted_strategies.index.tolist()
        ranking_results[f"{category}_RMSE"] = sorted_strategies.values.tolist()
    
    ranking_df = pd.DataFrame(dict([(k, pd.Series(v)) for k, v in ranking_results.items()]))
    
    ranking_path = f"{output_dir}/windshear_strategy_ranking.csv"
    ranking_df.to_csv(ranking_path, index=False, encoding='utf-8-sig')
    print(f"策略排序表格已保存: {ranking_path}")
    
    return rmse_df, ranking_df

def perform_significance_tests(df, prediction_cols, output_dir):
    """
    进行显著性检验
    """
    print("\n进行显著性检验...")
    
    categories = ['Low Shear', 'Moderate Shear', 'High Shear']
    significance_results = {}
    
    for category in categories:
        category_data = df[df['windshear_category'] == category]
        
        if len(category_data) < 10:  # 样本太小跳过
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
        sig_path = f"{output_dir}/significance_tests.csv"
        sig_df.to_csv(sig_path, index=False, encoding='utf-8-sig')
        print(f"显著性检验结果已保存: {sig_path}")
    
    return significance_results

def create_visualizations(rmse_df, df, output_dir):
    """
    创建可视化图表
    """
    print("\n创建可视化图表...")
    
    # 1. RMSE热力图
    plt.figure(figsize=(12, 8))
    sns.heatmap(rmse_df.T, annot=True, cmap='RdYlBu_r', fmt='.4f', 
                cbar_kws={'label': 'RMSE'})
    plt.title('RMSE Comparison Across Wind Shear Conditions', fontsize=16, fontweight='bold')
    plt.xlabel('Prediction Strategy', fontsize=12)
    plt.ylabel('Wind Shear Category', fontsize=12)
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig(f"{output_dir}/rmse_heatmap.png", dpi=300, bbox_inches='tight')
    plt.show()
    
    # 2. RMSE条形图对比
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    categories = ['Low Shear', 'Moderate Shear', 'High Shear']
    colors = ['lightcoral', 'skyblue', 'lightgreen']
    
    for idx, category in enumerate(categories):
        rmse_values = rmse_df[category].sort_values()
        axes[idx].bar(range(len(rmse_values)), rmse_values.values, color=colors[idx], alpha=0.7)
        axes[idx].set_title(f'RMSE under {category} Conditions', fontweight='bold')
        axes[idx].set_ylabel('RMSE')
        axes[idx].set_xlabel('Prediction Strategy')
        axes[idx].set_xticks(range(len(rmse_values)))
        axes[idx].set_xticklabels(rmse_values.index, rotation=45, ha='right')
        
        # 标注数值
        for i, v in enumerate(rmse_values.values):
            axes[idx].text(i, v + 0.01, f'{v:.3f}', ha='center', va='bottom', fontsize=8)
    
    plt.tight_layout()
    plt.savefig(f"{output_dir}/rmse_comparison_bars.png", dpi=300, bbox_inches='tight')
    plt.show()
    
    # 3. 风切变分布图
    plt.figure(figsize=(12, 6))
    
    plt.subplot(1, 2, 1)
    df['windshear'].hist(bins=50, alpha=0.7, color='skyblue', edgecolor='black')
    plt.axvline(x=0.1, color='red', linestyle='--', label='Low Shear Threshold (0.1)')
    plt.axvline(x=0.3, color='orange', linestyle='--', label='High Shear Threshold (0.3)')
    plt.xlabel('Wind Shear Index')
    plt.ylabel('Frequency')
    plt.title('Wind Shear Index Distribution')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.subplot(1, 2, 2)
    category_counts = df['windshear_category'].value_counts()
    colors_pie = ['lightcoral', 'skyblue', 'lightgreen']
    plt.pie(category_counts.values, labels=category_counts.index, autopct='%1.1f%%', 
            colors=colors_pie, startangle=90)
    plt.title('Wind Shear Category Distribution')
    
    plt.tight_layout()
    plt.savefig(f"{output_dir}/windshear_distribution.png", dpi=300, bbox_inches='tight')
    plt.show()

def generate_analysis_summary(rmse_df, significance_results, df, output_dir):
    """
    生成分析总结报告
    """
    print("\n生成分析总结报告...")
    
    summary_text = []
    summary_text.append("# 风切变条件下预测策略性能分析报告\n")
    
    # 数据概况
    summary_text.append("## 1. 数据概况")
    summary_text.append(f"- 总数据量: {len(df)} 条")
    category_counts = df['windshear_category'].value_counts()
    for category, count in category_counts.items():
        percentage = count / len(df) * 100
        summary_text.append(f"- {category}: {count} 条 ({percentage:.1f}%)")
    summary_text.append("")
    
    # 各风切变条件下的最优策略
    summary_text.append("## 2. 各风切变条件下的最优策略")
    for category in rmse_df.columns:
        best_strategy = rmse_df[category].idxmin()
        best_rmse = rmse_df[category].min()
        worst_strategy = rmse_df[category].idxmax()
        worst_rmse = rmse_df[category].max()
        
        summary_text.append(f"### {category}")
        summary_text.append(f"- 最优策略: {best_strategy} (RMSE: {best_rmse:.4f})")
        summary_text.append(f"- 最差策略: {worst_strategy} (RMSE: {worst_rmse:.4f})")
        summary_text.append(f"- 性能差距: {((worst_rmse - best_rmse) / best_rmse * 100):.1f}%")
        summary_text.append("")
    
    # 策略一致性分析
    summary_text.append("## 3. 策略排序一致性分析")
    
    # 计算各策略的平均排名
    rankings = {}
    for strategy in rmse_df.index:
        ranks = []
        for category in rmse_df.columns:
            rank = rmse_df[category].rank().loc[strategy]
            ranks.append(rank)
        rankings[strategy] = np.mean(ranks)
    
    sorted_strategies = sorted(rankings.items(), key=lambda x: x[1])
    
    summary_text.append("### 综合排序 (基于平均排名):")
    for i, (strategy, avg_rank) in enumerate(sorted_strategies, 1):
        summary_text.append(f"{i}. {strategy} (平均排名: {avg_rank:.1f})")
    
    summary_text.append("")
    
    # 保存总结报告
    summary_path = f"{output_dir}/analysis_summary.md"
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
        
        # 2. 风切变分类
        df = classify_windshear(df)
        
        # 3. 获取预测策略列
        prediction_cols = get_prediction_columns(df)
        
        # 4. 计算各类别下的RMSE
        rmse_results = calculate_rmse_by_category(df, prediction_cols)
        
        # 5. 创建对比表格
        rmse_df, ranking_df = create_rmse_comparison_table(rmse_results, output_dir)
        
        # 6. 显著性检验
        significance_results = perform_significance_tests(df, prediction_cols, output_dir)
        
        # 7. 创建可视化
        create_visualizations(rmse_df, df, output_dir)
        
        # 8. 生成分析总结
        summary = generate_analysis_summary(rmse_df, significance_results, df, output_dir)
        
        print("\n" + "="*50)
        print("分析完成！所有结果已保存到:")
        print(f"{output_dir}")
        print("="*50)
        
        # 显示关键发现
        print("\n关键发现:")
        for category in rmse_df.columns:
            best_strategy = rmse_df[category].idxmin()
            best_rmse = rmse_df[category].min()
            print(f"- {category} Best Strategy: {best_strategy} (RMSE: {best_rmse:.4f})")
        
    except Exception as e:
        print(f"分析过程中出现错误: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()