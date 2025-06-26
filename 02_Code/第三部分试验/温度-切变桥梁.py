#!/usr/bin/env python3
"""
10m温度与风切变关系分析器
验证温度是否可以作为风切变条件的代理变量
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import accuracy_score, classification_report
import seaborn as sns
import os
import warnings
warnings.filterwarnings('ignore')

# 设置英文显示
plt.rcParams['font.family'] = ['Arial', 'DejaVu Sans', 'Liberation Sans']
plt.rcParams['axes.unicode_minus'] = False

def analyze_temp_shear_relationship(data_path, save_path):
    """分析10m温度与风切变的关系"""
    print("=" * 80)
    print("🌡️ 10m温度与风切变关系分析")
    print("目标：验证温度是否可作为切变条件的代理变量")
    print("=" * 80)
    
    os.makedirs(save_path, exist_ok=True)
    
    # 1. 数据加载和预处理
    print("\n🔄 步骤1: 数据加载和预处理")
    data = pd.read_csv(data_path)
    data['datetime'] = pd.to_datetime(data['datetime'])
    print(f"原始数据形状: {data.shape}")
    
    # 数据清理
    temp_col = 'obs_temperature_10m'
    wind_10m_col = 'obs_wind_speed_10m'
    wind_70m_col = 'obs_wind_speed_70m'
    
    # 检查必需列是否存在
    required_cols = [temp_col, wind_10m_col, wind_70m_col]
    missing_cols = [col for col in required_cols if col not in data.columns]
    if missing_cols:
        print(f"❌ 缺少必需列: {missing_cols}")
        return False
    
    # 数据清理
    data[temp_col] = data[temp_col].where((data[temp_col] >= -50) & (data[temp_col] <= 60))
    data[wind_10m_col] = data[wind_10m_col].where((data[wind_10m_col] >= 0) & (data[wind_10m_col] <= 50))
    data[wind_70m_col] = data[wind_70m_col].where((data[wind_70m_col] >= 0) & (data[wind_70m_col] <= 50))
    
    # 风切变计算
    print("\n🔄 步骤2: 风切变计算")
    v1 = data[wind_10m_col]
    v2 = data[wind_70m_col]
    
    valid_wind_mask = (v1 > 0.5) & (v2 > 0.5) & (~v1.isna()) & (~v2.isna())
    data = data[valid_wind_mask].copy()
    
    # 计算风切变系数 alpha
    data['wind_shear_alpha'] = np.log(data[wind_70m_col] / data[wind_10m_col]) / np.log(70 / 10)
    
    # 计算简单风切变 (线性)
    data['wind_shear_linear'] = (data[wind_70m_col] - data[wind_10m_col]) / (70 - 10)
    
    # 过滤有效切变数据
    alpha = data['wind_shear_alpha']
    valid_alpha = (~np.isnan(alpha)) & (~np.isinf(alpha)) & (alpha > -1) & (alpha < 2)
    data = data[valid_alpha].copy()
    
    # 过滤有效温度数据
    valid_temp = ~data[temp_col].isna()
    data = data[valid_temp].copy()
    
    print(f"有效数据点: {len(data)}")
    
    # 添加时间变量
    data['hour'] = data['datetime'].dt.hour
    data['month'] = data['datetime'].dt.month
    data['is_daytime'] = ((data['hour'] >= 6) & (data['hour'] < 18))
    data['season'] = data['month'].map({
        12: 'winter', 1: 'winter', 2: 'winter',
        3: 'spring', 4: 'spring', 5: 'spring',
        6: 'summer', 7: 'summer', 8: 'summer',
        9: 'autumn', 10: 'autumn', 11: 'autumn'
    })
    
    # 风切变分类
    shear_thresholds = {'weak_upper': 0.1, 'moderate_upper': 0.3}
    alpha = data['wind_shear_alpha']
    conditions = [
        alpha < shear_thresholds['weak_upper'],
        (alpha >= shear_thresholds['weak_upper']) & (alpha < shear_thresholds['moderate_upper']),
        alpha >= shear_thresholds['moderate_upper']
    ]
    choices = ['weak', 'moderate', 'strong']
    data['shear_category'] = np.select(conditions, choices, default='unknown')
    
    # 2. 基础相关性分析
    print("\n🔄 步骤3: 基础相关性分析")
    
    temp_data = data[temp_col]
    alpha_data = data['wind_shear_alpha']
    linear_shear_data = data['wind_shear_linear']
    
    # 计算相关系数
    corr_alpha_pearson, p_alpha_pearson = pearsonr(temp_data, alpha_data)
    corr_alpha_spearman, p_alpha_spearman = spearmanr(temp_data, alpha_data)
    corr_linear_pearson, p_linear_pearson = pearsonr(temp_data, linear_shear_data)
    corr_linear_spearman, p_linear_spearman = spearmanr(temp_data, linear_shear_data)
    
    print(f"10m温度 vs 风切变系数(alpha):")
    print(f"  Pearson相关系数: {corr_alpha_pearson:.4f} (p-value: {p_alpha_pearson:.6f})")
    print(f"  Spearman相关系数: {corr_alpha_spearman:.4f} (p-value: {p_alpha_spearman:.6f})")
    
    print(f"\n10m温度 vs 线性风切变:")
    print(f"  Pearson相关系数: {corr_linear_pearson:.4f} (p-value: {p_linear_pearson:.6f})")
    print(f"  Spearman相关系数: {corr_linear_spearman:.4f} (p-value: {p_linear_spearman:.6f})")
    
    # 3. 分时段分析
    print("\n🔄 步骤4: 分时段相关性分析")
    
    time_periods = {
        'dawn': (0, 6),
        'morning': (6, 12), 
        'afternoon': (12, 18),
        'evening': (18, 24)
    }
    
    time_correlations = {}
    for period_name, (start_hour, end_hour) in time_periods.items():
        mask = (data['hour'] >= start_hour) & (data['hour'] < end_hour)
        period_data = data[mask]
        
        if len(period_data) > 30:  # 至少30个样本
            temp_period = period_data[temp_col]
            alpha_period = period_data['wind_shear_alpha']
            
            corr_period, p_period = pearsonr(temp_period, alpha_period)
            time_correlations[period_name] = {
                'correlation': corr_period,
                'p_value': p_period,
                'sample_size': len(period_data),
                'hours': f"{start_hour:02d}-{end_hour:02d}"
            }
            
            print(f"  {period_name} ({start_hour:02d}-{end_hour:02d}h): "
                  f"相关系数={corr_period:.4f}, p={p_period:.6f}, N={len(period_data)}")
    
    # 4. 基于温度预测切变类别的能力
    print("\n🔄 步骤5: 温度预测切变类别能力分析")
    
    def create_temp_based_classifier(temp_data, shear_categories, is_daytime):
        """基于温度创建切变分类器"""
        # 计算不同类别的温度统计
        temp_stats = {}
        for category in ['weak', 'moderate', 'strong']:
            cat_mask = shear_categories == category
            if cat_mask.sum() > 0:
                temp_stats[category] = {
                    'mean': temp_data[cat_mask].mean(),
                    'std': temp_data[cat_mask].std(),
                    'count': cat_mask.sum()
                }
        
        # 简单的基于温度阈值的分类器
        def classify_by_temp(temp, is_day):
            if is_day:
                if temp < 10:
                    return 'strong'  # 低温通常对应强切变
                elif temp < 20:
                    return 'moderate'
                else:
                    return 'weak'   # 高温通常对应弱切变
            else:  # 夜间
                if temp < 5:
                    return 'strong'
                elif temp < 15:
                    return 'moderate'
                else:
                    return 'weak'
        
        predicted = [classify_by_temp(t, d) for t, d in zip(temp_data, is_daytime)]
        return predicted, temp_stats
    
    predicted_categories, temp_stats = create_temp_based_classifier(
        data[temp_col], data['shear_category'], data['is_daytime']
    )
    
    # 计算分类准确率
    accuracy = accuracy_score(data['shear_category'], predicted_categories)
    print(f"\n基于温度的切变分类准确率: {accuracy:.3f}")
    
    print("\n各切变类别的温度统计:")
    for category, stats in temp_stats.items():
        print(f"  {category}: 均值={stats['mean']:.2f}°C, "
              f"标准差={stats['std']:.2f}°C, 样本数={stats['count']}")
    
    # 5. 可视化分析
    print("\n🔄 步骤6: 创建可视化图表")
    
    # 设置图表样式
    plt.style.use('default')
    sns.set_palette("husl")
    
    fig = plt.figure(figsize=(20, 16))
    
    # 1. 温度vs风切变散点图
    ax1 = plt.subplot(3, 3, 1)
    scatter = ax1.scatter(data[temp_col], data['wind_shear_alpha'], 
                         c=data['hour'], cmap='viridis', alpha=0.6, s=20)
    ax1.set_xlabel('10m Temperature (°C)')
    ax1.set_ylabel('Wind Shear Alpha')
    ax1.set_title(f'Temperature vs Wind Shear\n(r={corr_alpha_pearson:.3f}, p={p_alpha_pearson:.1e})')
    ax1.grid(True, alpha=0.3)
    cbar = plt.colorbar(scatter, ax=ax1)
    cbar.set_label('Hour of Day')
    
    # 2. 分时段相关性柱状图
    ax2 = plt.subplot(3, 3, 2)
    periods = list(time_correlations.keys())
    correlations = [time_correlations[p]['correlation'] for p in periods]
    colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4']
    bars = ax2.bar(periods, correlations, color=colors, alpha=0.7)
    ax2.set_ylabel('Correlation Coefficient')
    ax2.set_title('Temperature-Shear Correlation by Time Period')
    ax2.grid(True, alpha=0.3)
    ax2.axhline(y=0, color='black', linestyle='-', alpha=0.5)
    
    # 添加数值标签
    for bar, corr in zip(bars, correlations):
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height + 0.01 if height >= 0 else height - 0.01,
                f'{corr:.3f}', ha='center', va='bottom' if height >= 0 else 'top', fontweight='bold')
    
    # 3. 不同切变类别的温度分布箱线图
    ax3 = plt.subplot(3, 3, 3)
    shear_cats = ['weak', 'moderate', 'strong']
    temp_by_shear = [data[data['shear_category'] == cat][temp_col] for cat in shear_cats]
    box_plot = ax3.boxplot(temp_by_shear, labels=shear_cats, patch_artist=True)
    
    # 设置箱线图颜色
    colors_box = ['#FFE5B4', '#FFCC99', '#FF9999']
    for patch, color in zip(box_plot['boxes'], colors_box):
        patch.set_facecolor(color)
    
    ax3.set_xlabel('Wind Shear Category')
    ax3.set_ylabel('10m Temperature (°C)')
    ax3.set_title('Temperature Distribution by Shear Category')
    ax3.grid(True, alpha=0.3)
    
    # 4. 日变化模式对比
    ax4 = plt.subplot(3, 3, 4)
    hourly_temp = data.groupby('hour')[temp_col].mean()
    hourly_shear = data.groupby('hour')['wind_shear_alpha'].mean()
    
    ax4_twin = ax4.twinx()
    line1 = ax4.plot(hourly_temp.index, hourly_temp.values, 'r-o', label='Temperature', linewidth=2)
    line2 = ax4_twin.plot(hourly_shear.index, hourly_shear.values, 'b-s', label='Wind Shear', linewidth=2)
    
    ax4.set_xlabel('Hour of Day')
    ax4.set_ylabel('Temperature (°C)', color='red')
    ax4_twin.set_ylabel('Wind Shear Alpha', color='blue')
    ax4.set_title('Diurnal Pattern: Temperature vs Wind Shear')
    ax4.grid(True, alpha=0.3)
    
    # 图例
    lines1, labels1 = ax4.get_legend_handles_labels()
    lines2, labels2 = ax4_twin.get_legend_handles_labels()
    ax4.legend(lines1 + lines2, labels1 + labels2, loc='upper left')
    
    # 5. 季节性分析
    ax5 = plt.subplot(3, 3, 5)
    seasonal_corr = []
    seasons = ['spring', 'summer', 'autumn', 'winter']
    season_colors = ['#98D8C8', '#F7DC6F', '#F8C471', '#AED6F1']
    
    for season in seasons:
        season_data = data[data['season'] == season]
        if len(season_data) > 30:
            corr_season, _ = pearsonr(season_data[temp_col], season_data['wind_shear_alpha'])
            seasonal_corr.append(corr_season)
        else:
            seasonal_corr.append(0)
    
    bars_season = ax5.bar(seasons, seasonal_corr, color=season_colors, alpha=0.7)
    ax5.set_ylabel('Correlation Coefficient')
    ax5.set_title('Temperature-Shear Correlation by Season')
    ax5.grid(True, alpha=0.3)
    ax5.axhline(y=0, color='black', linestyle='-', alpha=0.5)
    
    # 添加数值标签
    for bar, corr in zip(bars_season, seasonal_corr):
        height = bar.get_height()
        ax5.text(bar.get_x() + bar.get_width()/2., height + 0.01 if height >= 0 else height - 0.01,
                f'{corr:.3f}', ha='center', va='bottom' if height >= 0 else 'top', fontweight='bold')
    
    # 6. 温度分级的切变分布
    ax6 = plt.subplot(3, 3, 6)
    temp_bins = pd.cut(data[temp_col], bins=5, labels=['Very Cold', 'Cold', 'Mild', 'Warm', 'Hot'])
    temp_shear_cross = pd.crosstab(temp_bins, data['shear_category'], normalize='index') * 100
    
    temp_shear_cross.plot(kind='bar', stacked=True, ax=ax6, 
                         color=['#FFE5B4', '#FFCC99', '#FF9999'])
    ax6.set_xlabel('Temperature Category')
    ax6.set_ylabel('Percentage (%)')
    ax6.set_title('Shear Category Distribution by Temperature Range')
    ax6.legend(title='Shear Category', bbox_to_anchor=(1.05, 1), loc='upper left')
    ax6.grid(True, alpha=0.3)
    plt.setp(ax6.xaxis.get_majorticklabels(), rotation=45)
    
    # 7. 分类准确率混淆矩阵
    ax7 = plt.subplot(3, 3, 7)
    from sklearn.metrics import confusion_matrix
    cm = confusion_matrix(data['shear_category'], predicted_categories, labels=['weak', 'moderate', 'strong'])
    cm_percentage = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis] * 100
    
    im = ax7.imshow(cm_percentage, interpolation='nearest', cmap='Blues')
    ax7.set_title(f'Classification Confusion Matrix\n(Accuracy: {accuracy:.3f})')
    tick_marks = np.arange(3)
    ax7.set_xticks(tick_marks)
    ax7.set_yticks(tick_marks)
    ax7.set_xticklabels(['weak', 'moderate', 'strong'])
    ax7.set_yticklabels(['weak', 'moderate', 'strong'])
    ax7.set_xlabel('Predicted by Temperature')
    ax7.set_ylabel('Actual Shear Category')
    
    # 添加数值标签
    for i in range(3):
        for j in range(3):
            ax7.text(j, i, f'{cm_percentage[i, j]:.1f}%\n({cm[i, j]})',
                    ha="center", va="center", color="white" if cm_percentage[i, j] > 50 else "black")
    
    # 8. 风速条件下的温度-切变关系
    ax8 = plt.subplot(3, 3, 8)
    wind_speed_avg = (data[wind_10m_col] + data[wind_70m_col]) / 2
    wind_categories = pd.cut(wind_speed_avg, bins=3, labels=['Low Wind', 'Medium Wind', 'High Wind'])
    
    for i, wind_cat in enumerate(['Low Wind', 'Medium Wind', 'High Wind']):
        wind_mask = wind_categories == wind_cat
        if wind_mask.sum() > 20:
            wind_data = data[wind_mask]
            corr_wind, _ = pearsonr(wind_data[temp_col], wind_data['wind_shear_alpha'])
            ax8.scatter(wind_data[temp_col], wind_data['wind_shear_alpha'], 
                       alpha=0.6, s=15, label=f'{wind_cat} (r={corr_wind:.3f}, N={wind_mask.sum()})')
    
    ax8.set_xlabel('10m Temperature (°C)')
    ax8.set_ylabel('Wind Shear Alpha')
    ax8.set_title('Temperature-Shear Relationship by Wind Speed')
    ax8.legend()
    ax8.grid(True, alpha=0.3)
    
    # 9. 综合评估雷达图
    ax9 = plt.subplot(3, 3, 9, projection='polar')
    
    # 评估指标
    metrics = ['Overall\nCorrelation', 'Morning\nCorrelation', 'Afternoon\nCorrelation', 
              'Evening\nCorrelation', 'Classification\nAccuracy']
    values = [
        abs(corr_alpha_pearson),
        abs(time_correlations.get('morning', {}).get('correlation', 0)),
        abs(time_correlations.get('afternoon', {}).get('correlation', 0)),
        abs(time_correlations.get('evening', {}).get('correlation', 0)),
        accuracy
    ]
    
    # 归一化到0-1
    values_normalized = [(v + 1) / 2 if i < 4 else v for i, v in enumerate(values)]
    
    angles = np.linspace(0, 2 * np.pi, len(metrics), endpoint=False).tolist()
    values_normalized += values_normalized[:1]  # 闭合
    angles += angles[:1]
    
    ax9.plot(angles, values_normalized, 'o-', linewidth=2, color='#FF6B6B')
    ax9.fill(angles, values_normalized, alpha=0.25, color='#FF6B6B')
    ax9.set_xticks(angles[:-1])
    ax9.set_xticklabels(metrics)
    ax9.set_ylim(0, 1)
    ax9.set_title('Temperature-Shear Relationship\nComprehensive Assessment', y=1.08)
    ax9.grid(True)
    
    plt.tight_layout()
    plt.savefig(f"{save_path}/temperature_shear_relationship_analysis.png", dpi=300, bbox_inches='tight')
    plt.show()
    
    # 6. 生成分析报告
    print("\n🔄 步骤7: 生成分析报告")
    
    report_data = {
        'overall_correlation_alpha': corr_alpha_pearson,
        'overall_p_value_alpha': p_alpha_pearson,
        'overall_correlation_linear': corr_linear_pearson,
        'overall_p_value_linear': p_linear_pearson,
        'classification_accuracy': accuracy,
        'sample_size': len(data),
        'time_period_correlations': time_correlations,
        'seasonal_correlations': dict(zip(seasons, seasonal_corr)),
        'temperature_statistics_by_shear': temp_stats
    }
    
    # 保存详细结果
    results_df = data[[temp_col, 'wind_shear_alpha', 'wind_shear_linear', 'shear_category', 
                      'hour', 'is_daytime', 'season']].copy()
    results_df['predicted_shear_category'] = predicted_categories
    results_df.to_csv(f"{save_path}/temperature_shear_analysis_data.csv", index=False)
    
    # 保存汇总报告
    import json
    with open(f"{save_path}/temperature_shear_relationship_report.json", 'w') as f:
        json.dump(report_data, f, indent=2, default=str)
    
    # 7. 结论总结
    print("\n" + "=" * 80)
    print("📋 分析结论总结")
    print("=" * 80)
    
    print(f"\n🔍 主要发现:")
    print(f"1. 10m温度与风切变系数的总体相关性: {corr_alpha_pearson:.4f}")
    if abs(corr_alpha_pearson) > 0.3:
        print("   ✅ 存在中等强度相关性")
    elif abs(corr_alpha_pearson) > 0.1:
        print("   ⚠️ 存在弱相关性")
    else:
        print("   ❌ 相关性很弱")
    
    print(f"\n2. 统计显著性: p-value = {p_alpha_pearson:.6f}")
    if p_alpha_pearson < 0.001:
        print("   ✅ 高度显著 (p < 0.001)")
    elif p_alpha_pearson < 0.05:
        print("   ✅ 统计显著 (p < 0.05)")
    else:
        print("   ❌ 不显著 (p >= 0.05)")
    
    print(f"\n3. 基于温度的切变分类准确率: {accuracy:.3f}")
    if accuracy > 0.6:
        print("   ✅ 分类效果较好")
    elif accuracy > 0.4:
        print("   ⚠️ 分类效果一般")
    else:
        print("   ❌ 分类效果较差")
    
    print(f"\n4. 不同时段的相关性变化:")
    for period, info in time_correlations.items():
        print(f"   {period} ({info['hours']}): r={info['correlation']:.3f}, N={info['sample_size']}")
    
    print(f"\n🎯 实用性评估:")
    if abs(corr_alpha_pearson) > 0.2 and p_alpha_pearson < 0.05 and accuracy > 0.4:
        print("✅ 温度可以作为风切变条件的代理变量")
        print("   建议：可以在融合策略中使用温度来动态调整权重")
    elif abs(corr_alpha_pearson) > 0.1 and p_alpha_pearson < 0.05:
        print("⚠️ 温度与风切变存在一定关系，但强度有限")
        print("   建议：可以尝试使用，但需要结合其他变量")
    else:
        print("❌ 温度与风切变关系较弱，不建议直接使用")
        print("   建议：寻找其他更强的代理变量")
    
    print(f"\n📁 输出文件:")
    print(f"   - temperature_shear_relationship_analysis.png: 综合分析图表")
    print(f"   - temperature_shear_analysis_data.csv: 详细分析数据")
    print(f"   - temperature_shear_relationship_report.json: 分析报告")
    
    return True

if __name__ == "__main__":
    # 配置路径
    DATA_PATH = "/Users/xiaxin/work/WindForecast_Project/01_Data/processed/imputed_data/changma_imputed_complete.csv"
    SAVE_PATH = "/Users/xiaxin/work/WindForecast_Project/03_Results/temperature_shear_analysis"
    
    success = analyze_temp_shear_relationship(DATA_PATH, SAVE_PATH)
    
    if success:
        print("\n🎉 温度与风切变关系分析完成!")
    else:
        print("\n⚠️ 分析失败")