#!/usr/bin/env python3
"""
Fusion-M1 vs Fusion-M2 预测结果对比可视化
在测试集上对比三个序列：观测值、Fusion-M1预测、Fusion-M2预测
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import os
import json

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['Arial Unicode MS', 'SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

# 设置绘图风格
sns.set_style("whitegrid")
plt.rcParams['figure.dpi'] = 300
plt.rcParams['savefig.dpi'] = 300

def load_results(fusion_m1_dir, fusion_m2_dir):
    """加载两个模型的结果"""
    
    print("📂 加载模型结果...")
    
    # 加载Fusion-M1结果
    m1_detailed_path = os.path.join(fusion_m1_dir, 'Fusion-M1_detailed_results.csv')
    if not os.path.exists(m1_detailed_path):
        raise FileNotFoundError(f"未找到Fusion-M1结果文件: {m1_detailed_path}")
    
    m1_results = pd.read_csv(m1_detailed_path)
    m1_results['datetime'] = pd.to_datetime(m1_results['datetime'])
    
    print(f"   ✅ Fusion-M1: {len(m1_results)} 个测试样本")
    
    # 加载Fusion-M2结果
    m2_detailed_path = os.path.join(fusion_m2_dir, 'detailed_results.csv')
    if not os.path.exists(m2_detailed_path):
        raise FileNotFoundError(f"未找到Fusion-M2结果文件: {m2_detailed_path}")
    
    m2_results = pd.read_csv(m2_detailed_path)
    m2_results['datetime'] = pd.to_datetime(m2_results['datetime'])
    
    print(f"   ✅ Fusion-M2: {len(m2_results)} 个测试样本")
    
    # 确保两个结果有相同的时间索引
    merged = pd.merge(
        m1_results[['datetime', 'actual_power', 'predicted_power']],
        m2_results[['datetime', 'predicted_power', 'wind_category']],
        on='datetime',
        how='inner',
        suffixes=('_m1', '_m2')
    )
    
    print(f"   🔗 合并后: {len(merged)} 个共同测试样本")
    
    return merged

def calculate_metrics(actual, predicted, model_name):
    """计算评估指标"""
    
    # 过滤掉NaN和预测为0的样本
    valid_mask = (~np.isnan(actual)) & (~np.isnan(predicted)) & (predicted != 0)
    
    if valid_mask.sum() == 0:
        return None
    
    actual_valid = actual[valid_mask]
    predicted_valid = predicted[valid_mask]
    
    rmse = np.sqrt(mean_squared_error(actual_valid, predicted_valid))
    mae = mean_absolute_error(actual_valid, predicted_valid)
    r2 = r2_score(actual_valid, predicted_valid)
    corr = np.corrcoef(actual_valid, predicted_valid)[0, 1]
    
    # 计算MAPE（避免除以0）
    mape = np.mean(np.abs((actual_valid - predicted_valid) / (actual_valid + 1e-6))) * 100
    
    metrics = {
        'model': model_name,
        'RMSE': rmse,
        'MAE': mae,
        'R2': r2,
        'Correlation': corr,
        'MAPE': mape,
        'Valid_Samples': valid_mask.sum()
    }
    
    return metrics

def plot_complete_test_series(merged_data, save_dir):
    """
    绘制完整测试集的三条序列对比（单独一张图，无中文）
    """
    
    print("\n📊 绘制完整测试集三条序列对比...")
    
    # 创建图形 - 单图
    fig, ax = plt.subplots(1, 1, figsize=(20, 6))
    
    # 三条线对比
    ax.plot(merged_data['datetime'], merged_data['actual_power'], 
            label='Observed', color='black', linewidth=1, alpha=0.5)
    ax.plot(merged_data['datetime'], merged_data['predicted_power_m1'], 
            label='Fusion-M1', color="#f22023c9", linewidth=1.2, alpha=0.7, linestyle='-')
    ax.plot(merged_data['datetime'], merged_data['predicted_power_m2'], 
            label='Fusion-M2', color="#0e12ff", linewidth=1.2, alpha=0.7, linestyle='-')
    
    # ax.set_xlabel('Time', fontsize=20, fontweight='bold')
    ax.set_ylabel('Power (MW)', fontsize=20, fontweight='bold')
    ax.set_title('Comparison of Observed and Predicted Power on Test Set', fontsize=20, fontweight='bold')
    ax.legend(loc='upper right', fontsize=20, framealpha=0.9)
    ax.grid(True, alpha=0.3)
    # 设置刻度标签字体大小（关键！）
    ax.tick_params(axis='both', which='major', labelsize=18)
    ax.tick_params(axis='both', which='minor', labelsize=16)
    # 调整x轴日期显示
    import matplotlib.dates as mdates
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
    ax.xaxis.set_major_locator(mdates.AutoDateLocator())
    plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha='right')
    
    plt.tight_layout()
    
    # 保存图形
    save_path = os.path.join(save_dir, 'complete_test_series_comparison.png')
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"   ✅ 完整测试集序列对比图已保存: {save_path}")
    plt.close()

def plot_error_series_comparison(merged_data, save_dir):
    """
    绘制两个模型的误差序列对比（单独一张图，两条误差线在同一子图，无中文）
    """
    
    print("\n📊 绘制误差序列对比...")
    
    # 计算误差
    error_m1 = merged_data['predicted_power_m1'] - merged_data['actual_power']
    error_m2 = merged_data['predicted_power_m2'] - merged_data['actual_power']
    
    # 创建图形 - 单图
    fig, ax = plt.subplots(1, 1, figsize=(20, 6))
    
    # 两条误差线
    ax.plot(merged_data['datetime'], error_m1, 
            label='Fusion-M1 Error', color='#1f77b4', linewidth=1.0, alpha=0.7)
    ax.plot(merged_data['datetime'], error_m2, 
            label='Fusion-M2 Error', color='#ff7f0e', linewidth=1.0, alpha=0.7)
    
    # 零线
    ax.axhline(y=0, color='red', linestyle='--', linewidth=1.5, label='Zero Line')
    
    ax.set_xlabel('Time', fontsize=14, fontweight='bold')
    ax.set_ylabel('Prediction Error (MW)', fontsize=14, fontweight='bold')
    ax.set_title('Comparison of Prediction Errors on Test Set', fontsize=16, fontweight='bold')
    ax.legend(loc='upper right', fontsize=12, framealpha=0.9)
    ax.grid(True, alpha=0.3)
    
    # 调整x轴日期显示
    import matplotlib.dates as mdates
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
    ax.xaxis.set_major_locator(mdates.AutoDateLocator())
    plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha='right')
    
    plt.tight_layout()
    
    # 保存图形
    save_path = os.path.join(save_dir, 'error_series_comparison.png')
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"   ✅ 误差序列对比图已保存: {save_path}")
    plt.close()

def plot_scatter_comparison(merged_data, save_dir):
    """绘制散点图对比（无中文）"""
    
    print("\n📊 绘制散点图对比...")
    
    fig, axes = plt.subplots(1, 2, figsize=(16, 7))
    
    # Fusion-M1散点图
    ax1 = axes[0]
    ax1.scatter(merged_data['actual_power'], merged_data['predicted_power_m1'], 
                alpha=0.5, s=20, color='#1f77b4', edgecolors='none')
    
    # 添加1:1线
    max_val = max(merged_data['actual_power'].max(), merged_data['predicted_power_m1'].max())
    min_val = min(merged_data['actual_power'].min(), merged_data['predicted_power_m1'].min())
    ax1.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2, label='1:1 Line')
    
    # 计算指标
    m1_metrics = calculate_metrics(
        merged_data['actual_power'].values, 
        merged_data['predicted_power_m1'].values,
        'Fusion-M1'
    )
    
    # 添加指标文本
    metrics_text = f"RMSE: {m1_metrics['RMSE']:.4f}\nMAE: {m1_metrics['MAE']:.4f}\nR²: {m1_metrics['R2']:.4f}\nCorr: {m1_metrics['Correlation']:.4f}"
    ax1.text(0.05, 0.95, metrics_text, transform=ax1.transAxes, 
             fontsize=11, verticalalignment='top',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    
    ax1.set_xlabel('Observed Power (MW)', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Predicted Power (MW)', fontsize=12, fontweight='bold')
    ax1.set_title('Fusion-M1: Predicted vs Observed', fontsize=14, fontweight='bold')
    ax1.legend(fontsize=11)
    ax1.grid(True, alpha=0.3)
    
    # Fusion-M2散点图
    ax2 = axes[1]
    
    # 根据风向类别上色
    for category, color, label in [
        ('east', '#2ca02c', 'East Wind'),
        ('west', '#ff7f0e', 'West Wind'),
        ('other', '#9467bd', 'Other')
    ]:
        mask = merged_data['wind_category'] == category
        if mask.sum() > 0:
            ax2.scatter(merged_data.loc[mask, 'actual_power'], 
                       merged_data.loc[mask, 'predicted_power_m2'],
                       alpha=0.5, s=20, color=color, label=label, edgecolors='none')
    
    # 添加1:1线
    max_val = max(merged_data['actual_power'].max(), merged_data['predicted_power_m2'].max())
    min_val = min(merged_data['actual_power'].min(), merged_data['predicted_power_m2'].min())
    ax2.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2, label='1:1 Line')
    
    # 计算指标
    m2_metrics = calculate_metrics(
        merged_data['actual_power'].values, 
        merged_data['predicted_power_m2'].values,
        'Fusion-M2'
    )
    
    # 添加指标文本
    metrics_text = f"RMSE: {m2_metrics['RMSE']:.4f}\nMAE: {m2_metrics['MAE']:.4f}\nR²: {m2_metrics['R2']:.4f}\nCorr: {m2_metrics['Correlation']:.4f}"
    ax2.text(0.05, 0.95, metrics_text, transform=ax2.transAxes, 
             fontsize=11, verticalalignment='top',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    
    ax2.set_xlabel('Observed Power (MW)', fontsize=12, fontweight='bold')
    ax2.set_ylabel('Predicted Power (MW)', fontsize=12, fontweight='bold')
    ax2.set_title('Fusion-M2: Predicted vs Observed (by Wind Direction)', fontsize=14, fontweight='bold')
    ax2.legend(fontsize=11)
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    save_path = os.path.join(save_dir, 'scatter_comparison.png')
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"   ✅ Scatter plot comparison saved: {save_path}")
    plt.close()
    
    return m1_metrics, m2_metrics

def plot_error_distribution(merged_data, save_dir):
    """绘制误差分布对比（无中文）"""
    
    print("\n📊 绘制误差分布对比...")
    
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    # 计算误差
    error_m1 = merged_data['predicted_power_m1'] - merged_data['actual_power']
    error_m2 = merged_data['predicted_power_m2'] - merged_data['actual_power']
    
    # 图1：误差直方图对比
    ax1 = axes[0, 0]
    ax1.hist(error_m1, bins=50, alpha=0.6, color='#1f77b4', label='Fusion-M1', density=True)
    ax1.hist(error_m2, bins=50, alpha=0.6, color='#ff7f0e', label='Fusion-M2', density=True)
    ax1.axvline(x=0, color='red', linestyle='--', linewidth=2)
    ax1.set_xlabel('Prediction Error (MW)', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Density', fontsize=12, fontweight='bold')
    ax1.set_title('Error Distribution Comparison', fontsize=14, fontweight='bold')
    ax1.legend(fontsize=11)
    ax1.grid(True, alpha=0.3)
    
    # 图2：误差箱线图
    ax2 = axes[0, 1]
    box_data = [error_m1, error_m2]
    bp = ax2.boxplot(box_data, labels=['Fusion-M1', 'Fusion-M2'], 
                     patch_artist=True, widths=0.6)
    
    # 设置箱线图颜色
    colors = ['#1f77b4', '#ff7f0e']
    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.6)
    
    ax2.axhline(y=0, color='red', linestyle='--', linewidth=2)
    ax2.set_ylabel('Prediction Error (MW)', fontsize=12, fontweight='bold')
    ax2.set_title('Error Box Plot Comparison', fontsize=14, fontweight='bold')
    ax2.grid(True, alpha=0.3, axis='y')
    
    # 添加统计信息
    stats_text = f"Fusion-M1: μ={error_m1.mean():.2f}, σ={error_m1.std():.2f}\n"
    stats_text += f"Fusion-M2: μ={error_m2.mean():.2f}, σ={error_m2.std():.2f}"
    ax2.text(0.05, 0.95, stats_text, transform=ax2.transAxes,
             fontsize=10, verticalalignment='top',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    
    # 图3：绝对误差对比
    ax3 = axes[1, 0]
    abs_error_m1 = np.abs(error_m1)
    abs_error_m2 = np.abs(error_m2)
    
    ax3.hist(abs_error_m1, bins=50, alpha=0.6, color='#1f77b4', 
             label=f'Fusion-M1 (MAE={abs_error_m1.mean():.4f})', density=True)
    ax3.hist(abs_error_m2, bins=50, alpha=0.6, color='#ff7f0e', 
             label=f'Fusion-M2 (MAE={abs_error_m2.mean():.4f})', density=True)
    
    ax3.set_xlabel('Absolute Error (MW)', fontsize=12, fontweight='bold')
    ax3.set_ylabel('Density', fontsize=12, fontweight='bold')
    ax3.set_title('Absolute Error Distribution Comparison', fontsize=14, fontweight='bold')
    ax3.legend(fontsize=11)
    ax3.grid(True, alpha=0.3)
    
    # 图4：误差改进可视化
    ax4 = axes[1, 1]
    error_improvement = abs_error_m1 - abs_error_m2  # 正值表示M2更好
    
    # 分类统计
    better = (error_improvement > 0).sum()
    worse = (error_improvement < 0).sum()
    same = (error_improvement == 0).sum()
    
    improvement_pct = better / len(error_improvement) * 100
    
    ax4.hist(error_improvement, bins=50, alpha=0.7, color='green', edgecolor='black')
    ax4.axvline(x=0, color='red', linestyle='--', linewidth=2, label='No Improvement')
    ax4.set_xlabel('Error Improvement (M1-M2, MW)', fontsize=12, fontweight='bold')
    ax4.set_ylabel('Sample Count', fontsize=12, fontweight='bold')
    ax4.set_title('Fusion-M2 Error Improvement over M1', fontsize=14, fontweight='bold')
    
    # 添加统计信息
    improve_text = f"M2 Better: {better} ({improvement_pct:.1f}%)\n"
    improve_text += f"M2 Worse: {worse} ({worse/len(error_improvement)*100:.1f}%)\n"
    improve_text += f"Same: {same}"
    ax4.text(0.05, 0.95, improve_text, transform=ax4.transAxes,
             fontsize=10, verticalalignment='top',
             bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.8))
    
    ax4.legend(fontsize=11)
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    save_path = os.path.join(save_dir, 'error_distribution_comparison.png')
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"   ✅ Error distribution comparison saved: {save_path}")
    plt.close()

def plot_category_performance(merged_data, save_dir):
    """按风向类别绘制性能对比（无中文）"""
    
    print("\n📊 绘制分类别性能对比...")
    
    categories = ['east', 'west', 'other']
    category_labels = {'east': 'East Wind', 'west': 'West Wind', 'other': 'Other'}
    
    # 计算每个类别的指标
    category_metrics = []
    
    for category in categories:
        mask = merged_data['wind_category'] == category
        if mask.sum() == 0:
            continue
        
        cat_data = merged_data[mask]
        
        m1_metrics = calculate_metrics(
            cat_data['actual_power'].values,
            cat_data['predicted_power_m1'].values,
            f'M1-{category}'
        )
        
        m2_metrics = calculate_metrics(
            cat_data['actual_power'].values,
            cat_data['predicted_power_m2'].values,
            f'M2-{category}'
        )
        
        if m1_metrics and m2_metrics:
            category_metrics.append({
                'category': category_labels[category],
                'samples': mask.sum(),
                'm1_rmse': m1_metrics['RMSE'],
                'm2_rmse': m2_metrics['RMSE'],
                'm1_mae': m1_metrics['MAE'],
                'm2_mae': m2_metrics['MAE'],
                'm1_corr': m1_metrics['Correlation'],
                'm2_corr': m2_metrics['Correlation']
            })
    
    if not category_metrics:
        print("   ⚠️ Not enough category data for plotting")
        return
    
    # 转换为DataFrame
    cat_df = pd.DataFrame(category_metrics)
    
    # 创建对比图
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    
    x = np.arange(len(cat_df))
    width = 0.35
    
    # RMSE对比
    ax1 = axes[0]
    bars1 = ax1.bar(x - width/2, cat_df['m1_rmse'], width, 
                    label='Fusion-M1', color="#f22023c9", alpha=0.7)
    bars2 = ax1.bar(x + width/2, cat_df['m2_rmse'], width, 
                    label='Fusion-M2', color="#0e12ff", alpha=0.7)
    
    ax1.set_xlabel('Wind Direction Category', fontsize=12, fontweight='bold')
    ax1.set_ylabel('RMSE', fontsize=12, fontweight='bold')
    ax1.set_title('RMSE Comparison by Category', fontsize=14, fontweight='bold')
    ax1.set_xticks(x)
    ax1.set_xticklabels(cat_df['category'])
    ax1.legend(fontsize=11)
    ax1.grid(True, alpha=0.3, axis='y')
    
    # 添加数值标签
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height,
                    f'{height:.3f}', ha='center', va='bottom', fontsize=9)
    
    # MAE对比
    ax2 = axes[1]
    bars1 = ax2.bar(x - width/2, cat_df['m1_mae'], width, 
                    label='Fusion-M1', color='#1f77b4', alpha=0.8)
    bars2 = ax2.bar(x + width/2, cat_df['m2_mae'], width, 
                    label='Fusion-M2', color='#ff7f0e', alpha=0.8)
    
    ax2.set_xlabel('Wind Direction Category', fontsize=12, fontweight='bold')
    ax2.set_ylabel('MAE', fontsize=12, fontweight='bold')
    ax2.set_title('MAE Comparison by Category', fontsize=14, fontweight='bold')
    ax2.set_xticks(x)
    ax2.set_xticklabels(cat_df['category'])
    ax2.legend(fontsize=11)
    ax2.grid(True, alpha=0.3, axis='y')
    
    # 添加数值标签
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width()/2., height,
                    f'{height:.3f}', ha='center', va='bottom', fontsize=9)
    
    # 相关系数对比
    ax3 = axes[2]
    bars1 = ax3.bar(x - width/2, cat_df['m1_corr'], width, 
                    label='Fusion-M1', color='#1f77b4', alpha=0.8)
    bars2 = ax3.bar(x + width/2, cat_df['m2_corr'], width, 
                    label='Fusion-M2', color='#ff7f0e', alpha=0.8)
    
    ax3.set_xlabel('Wind Direction Category', fontsize=12, fontweight='bold')
    ax3.set_ylabel('Correlation', fontsize=12, fontweight='bold')
    ax3.set_title('Correlation Comparison by Category', fontsize=14, fontweight='bold')
    ax3.set_xticks(x)
    ax3.set_xticklabels(cat_df['category'])
    ax3.legend(fontsize=11)
    ax3.grid(True, alpha=0.3, axis='y')
    ax3.set_ylim([0.8, 1.0])  # 相关系数通常在这个范围
    
    # 添加数值标签
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            ax3.text(bar.get_x() + bar.get_width()/2., height,
                    f'{height:.4f}', ha='center', va='bottom', fontsize=9)
    
    plt.tight_layout()
    
    save_path = os.path.join(save_dir, 'category_performance_comparison.png')
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"   ✅ Category performance comparison saved: {save_path}")
    plt.close()
    
    return cat_df

def generate_comparison_report(merged_data, m1_metrics, m2_metrics, category_df, save_dir):
    """生成对比报告"""
    
    print("\n📝 生成对比报告...")
    
    report_lines = []
    report_lines.append("=" * 80)
    report_lines.append("Fusion-M1 vs Fusion-M2 对比报告")
    report_lines.append("=" * 80)
    report_lines.append("")
    
    # 1. 整体性能对比
    report_lines.append("1. 整体性能对比 (测试集)")
    report_lines.append("-" * 80)
    report_lines.append(f"{'指标':<15} {'Fusion-M1':<15} {'Fusion-M2':<15} {'改进':<15}")
    report_lines.append("-" * 80)
    
    rmse_improve = (m1_metrics['RMSE'] - m2_metrics['RMSE']) / m1_metrics['RMSE'] * 100
    mae_improve = (m1_metrics['MAE'] - m2_metrics['MAE']) / m1_metrics['MAE'] * 100
    
    report_lines.append(f"{'RMSE':<15} {m1_metrics['RMSE']:<15.4f} {m2_metrics['RMSE']:<15.4f} {rmse_improve:>+.2f}%")
    report_lines.append(f"{'MAE':<15} {m1_metrics['MAE']:<15.4f} {m2_metrics['MAE']:<15.4f} {mae_improve:>+.2f}%")
    report_lines.append(f"{'R²':<15} {m1_metrics['R2']:<15.4f} {m2_metrics['R2']:<15.4f}")
    report_lines.append(f"{'Correlation':<15} {m1_metrics['Correlation']:<15.4f} {m2_metrics['Correlation']:<15.4f}")
    report_lines.append(f"{'MAPE (%)':<15} {m1_metrics['MAPE']:<15.2f} {m2_metrics['MAPE']:<15.2f}")
    report_lines.append("")
    
    # 2. 风向分类统计
    report_lines.append("2. 风向分类统计")
    report_lines.append("-" * 80)
    category_counts = merged_data['wind_category'].value_counts()
    total = len(merged_data)
    
    for category in ['east', 'west', 'other']:
        if category in category_counts:
            count = category_counts[category]
            pct = count / total * 100
            label = {'east': '东风', 'west': '西风', 'other': '其他'}[category]
            report_lines.append(f"{label}: {count} 样本 ({pct:.2f}%)")
    report_lines.append("")
    
    # 3. 分类别性能对比
    if category_df is not None and len(category_df) > 0:
        report_lines.append("3. 分类别性能对比")
        report_lines.append("-" * 80)
        report_lines.append(f"{'类别':<10} {'样本数':<10} {'M1-RMSE':<12} {'M2-RMSE':<12} {'改进':<10}")
        report_lines.append("-" * 80)
        
        for _, row in category_df.iterrows():
            improve = (row['m1_rmse'] - row['m2_rmse']) / row['m1_rmse'] * 100
            report_lines.append(
                f"{row['category']:<10} {row['samples']:<10} "
                f"{row['m1_rmse']:<12.4f} {row['m2_rmse']:<12.4f} {improve:>+.2f}%"
            )
        report_lines.append("")
    
    # 4. 误差分析
    report_lines.append("4. 误差分析")
    report_lines.append("-" * 80)
    
    error_m1 = merged_data['predicted_power_m1'] - merged_data['actual_power']
    error_m2 = merged_data['predicted_power_m2'] - merged_data['actual_power']
    abs_error_m1 = np.abs(error_m1)
    abs_error_m2 = np.abs(error_m2)
    
    report_lines.append(f"Fusion-M1 误差统计:")
    report_lines.append(f"  均值: {error_m1.mean():.4f}, 标准差: {error_m1.std():.4f}")
    report_lines.append(f"  最大正误差: {error_m1.max():.4f}, 最大负误差: {error_m1.min():.4f}")
    report_lines.append(f"  平均绝对误差: {abs_error_m1.mean():.4f}")
    report_lines.append("")
    
    report_lines.append(f"Fusion-M2 误差统计:")
    report_lines.append(f"  均值: {error_m2.mean():.4f}, 标准差: {error_m2.std():.4f}")
    report_lines.append(f"  最大正误差: {error_m2.max():.4f}, 最大负误差: {error_m2.min():.4f}")
    report_lines.append(f"  平均绝对误差: {abs_error_m2.mean():.4f}")
    report_lines.append("")
    
    # 5. 改进总结
    report_lines.append("5. 改进总结")
    report_lines.append("-" * 80)
    
    error_improvement = abs_error_m1 - abs_error_m2
    better = (error_improvement > 0).sum()
    worse = (error_improvement < 0).sum()
    
    report_lines.append(f"Fusion-M2相对于Fusion-M1:")
    report_lines.append(f"  预测更好的样本: {better} ({better/len(error_improvement)*100:.2f}%)")
    report_lines.append(f"  预测更差的样本: {worse} ({worse/len(error_improvement)*100:.2f}%)")
    report_lines.append(f"  RMSE改进: {rmse_improve:+.2f}%")
    report_lines.append(f"  MAE改进: {mae_improve:+.2f}%")
    report_lines.append("")
    
    # 6. 结论
    report_lines.append("6. 结论")
    report_lines.append("-" * 80)
    
    if m2_metrics['RMSE'] < m1_metrics['RMSE']:
        report_lines.append("✅ Fusion-M2在整体性能上优于Fusion-M1")
        report_lines.append(f"   基于风向分类的动态变量选择策略带来了 {rmse_improve:.2f}% 的RMSE改进")
    else:
        report_lines.append("⚠️ Fusion-M2在整体性能上未超过Fusion-M1")
        report_lines.append("   可能需要进一步优化风向分类策略或调整模型参数")
    
    report_lines.append("")
    report_lines.append("=" * 80)
    
    # 保存报告
    report_text = "\n".join(report_lines)
    report_path = os.path.join(save_dir, 'comparison_report.txt')
    
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write(report_text)
    
    print(f"   ✅ 对比报告已保存: {report_path}")
    
    # 同时打印到控制台
    print("\n" + report_text)
    
    # 保存为JSON
    json_report = {
        'overall_performance': {
            'fusion_m1': {k: float(v) if isinstance(v, (int, float, np.number)) else v 
                         for k, v in m1_metrics.items()},
            'fusion_m2': {k: float(v) if isinstance(v, (int, float, np.number)) else v 
                         for k, v in m2_metrics.items()},
            'improvement': {
                'rmse_improve_pct': float(rmse_improve),
                'mae_improve_pct': float(mae_improve)
            }
        },
        'category_statistics': category_counts.to_dict(),
        'category_performance': category_df.to_dict('records') if category_df is not None else []
    }
    
    json_path = os.path.join(save_dir, 'comparison_report.json')
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(json_report, f, indent=2, ensure_ascii=False)
    
    print(f"   ✅ JSON报告已保存: {json_path}")

def main(fusion_m1_dir, fusion_m2_dir, save_dir, sample_hours=168):
    """
    主函数
    
    Args:
        fusion_m1_dir: Fusion-M1结果目录
        fusion_m2_dir: Fusion-M2结果目录
        save_dir: 对比结果保存目录
        sample_hours: 时间序列图采样小时数（默认168小时=7天，None表示全部）
    """
    
    print("=" * 80)
    print("🔬 Fusion-M1 vs Fusion-M2 对比分析")
    print("=" * 80)
    
    # 创建保存目录
    os.makedirs(save_dir, exist_ok=True)
    
    # 1. 加载结果
    merged_data = load_results(fusion_m1_dir, fusion_m2_dir)
    
    # 2. 绘制完整测试集三条序列对比（新增）
    plot_complete_test_series(merged_data, save_dir)
    
    # 3. 绘制误差序列对比（新增）
    plot_error_series_comparison(merged_data, save_dir)
    
    # 4. 绘制散点图对比
    m1_metrics, m2_metrics = plot_scatter_comparison(merged_data, save_dir)
    
    # 5. 绘制误差分布对比
    plot_error_distribution(merged_data, save_dir)
    
    # 6. 绘制分类别性能对比
    category_df = plot_category_performance(merged_data, save_dir)
    
    # 7. 生成对比报告
    generate_comparison_report(merged_data, m1_metrics, m2_metrics, category_df, save_dir)
    
    print("\n" + "=" * 80)
    print("🎉 对比分析完成!")
    print(f"📁 所有结果已保存至: {save_dir}")
    print("=" * 80)
    
    print("\n📊 生成的文件:")
    print("   - complete_test_series_comparison.png: 完整测试集三条序列对比 (NEW)")
    print("   - error_series_comparison.png: 误差序列对比 (NEW)")
    print("   - scatter_comparison.png: 散点图对比")
    print("   - error_distribution_comparison.png: 误差分布对比")
    print("   - category_performance_comparison.png: 分类别性能对比")
    print("   - comparison_report.txt: 文本格式对比报告")
    print("   - comparison_report.json: JSON格式对比报告")

if __name__ == "__main__":
    # 配置路径
    FUSION_M1_DIR = "/Users/xiaxin/work/WindForecast_Project/03_Results/建模试验/simplified_enhanced_experiments/Fusion-M1"
    FUSION_M2_DIR = "/Users/xiaxin/work/WindForecast_Project/03_Results/建模试验/wind_direction_based_fusion"
    SAVE_DIR = "/Users/xiaxin/work/WindForecast_Project/03_Results/建模试验/fusion_m1_vs_m2_comparison"
    
    # 运行对比分析
    # sample_hours=168 表示随机采样7天(168小时)数据绘制时间序列图
    # 如果想绘制全部数据，设置为None，但可能会很密集
    main(FUSION_M1_DIR, FUSION_M2_DIR, SAVE_DIR, sample_hours=168)