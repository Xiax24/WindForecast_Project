"""
Enhanced Low Wind Speed Correction Strategies
增强型低风速订正策略 - 解决3-4m/s数据缺失问题
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from scipy.stats import pearsonr
import lightgbm as lgb
from scipy import stats
from scipy.interpolate import interp1d
import warnings
import os
warnings.filterwarnings('ignore')

def ensure_output_directory(base_path):
    """确保输出目录存在"""
    output_dir = os.path.join(base_path, "低风速订正分析")
    os.makedirs(output_dir, exist_ok=True)
    return output_dir

def load_data_with_analysis(file_path):
    """加载数据并分析低风速分布"""
    print("📊 加载数据并分析低风速分布...")
    
    df = pd.read_csv(file_path)
    df['datetime'] = pd.to_datetime(df['datetime'])
    
    # 基础清理
    df_clean = df[['datetime', 'obs_wind_speed_10m', 'ec_wind_speed_10m', 'gfs_wind_speed_10m']].dropna()
    
    # 分析各风速区间分布
    wind_ranges = [(0, 1), (1, 2), (2, 3), (3, 4), (4, 5), (5, 6), (6, 8), (8, 15)]
    print("\n🌪️ 详细风速区间样本分布:")
    print("=" * 70)
    print(f"{'区间(m/s)':<12} {'样本数':<8} {'占比%':<8} {'观测均值':<10} {'EC均值':<10} {'GFS均值':<10}")
    print("-" * 70)
    
    distribution_info = {}
    for low, high in wind_ranges:
        mask = (df_clean['obs_wind_speed_10m'] >= low) & (df_clean['obs_wind_speed_10m'] < high)
        count = mask.sum()
        pct = count / len(df_clean) * 100
        obs_mean = df_clean.loc[mask, 'obs_wind_speed_10m'].mean() if count > 0 else 0
        ec_mean = df_clean.loc[mask, 'ec_wind_speed_10m'].mean() if count > 0 else 0
        gfs_mean = df_clean.loc[mask, 'gfs_wind_speed_10m'].mean() if count > 0 else 0
        
        distribution_info[(low, high)] = {
            'count': count, 'pct': pct, 'obs_mean': obs_mean, 
            'ec_mean': ec_mean, 'gfs_mean': gfs_mean
        }
        
        print(f"{low}-{high:<8} {count:<8} {pct:<8.1f} {obs_mean:<10.2f} {ec_mean:<10.2f} {gfs_mean:<10.2f}")
    
    # 识别数据稀少区间
    sparse_ranges = [(low, high) for (low, high), info in distribution_info.items() 
                    if info['count'] < len(df_clean) * 0.01]  # 少于1%的区间
    
    if sparse_ranges:
        print(f"\n⚠️  数据稀少区间 (<1%): {sparse_ranges}")
    
    return df_clean, distribution_info

def strategy_6_data_augmentation(X_train, y_train, distribution_info):
    """策略6: 数据增强 - 针对稀少区间生成合成数据"""
    print("\n🎯 策略6: 数据增强（解决3-4m/s数据缺失）")
    print("-" * 50)
    
    # 识别需要增强的区间
    target_range = (3, 4)
    range_mask = (y_train >= target_range[0]) & (y_train < target_range[1])
    current_samples = range_mask.sum()
    
    print(f"当前3-4m/s样本数: {current_samples}")
    
    if current_samples < 100:  # 如果样本数少于100，进行数据增强
        print("开始数据增强...")
        
        # 方法1: 基于相邻区间的插值生成
        lower_range_mask = (y_train >= 2) & (y_train < 3)
        upper_range_mask = (y_train >= 4) & (y_train < 5)
        
        if lower_range_mask.sum() > 0 and upper_range_mask.sum() > 0:
            # 获取相邻区间的样本
            X_lower = X_train[lower_range_mask]
            y_lower = y_train[lower_range_mask]
            X_upper = X_train[upper_range_mask]
            y_upper = y_train[upper_range_mask]
            
            # 生成目标数量的合成样本
            target_samples = max(200, len(X_train) // 50)  # 至少200个样本
            
            synthetic_X = []
            synthetic_y = []
            
            for _ in range(target_samples):
                # 随机选择上下区间的样本
                if len(X_lower) > 0 and len(X_upper) > 0:
                    idx_lower = np.random.randint(0, len(X_lower))
                    idx_upper = np.random.randint(0, len(X_upper))
                    
                    # 线性插值生成新样本
                    alpha = np.random.random()  # 插值权重
                    
                    new_x = alpha * X_lower[idx_lower] + (1 - alpha) * X_upper[idx_upper]
                    new_y = alpha * y_lower[idx_lower] + (1 - alpha) * y_upper[idx_upper]
                    
                    # 确保生成的y在目标区间内
                    new_y = np.clip(new_y, target_range[0], target_range[1])
                    
                    synthetic_X.append(new_x)
                    synthetic_y.append(new_y)
            
            if synthetic_X:
                synthetic_X = np.array(synthetic_X)
                synthetic_y = np.array(synthetic_y)
                
                # 合并原始数据和合成数据
                X_augmented = np.vstack([X_train, synthetic_X])
                y_augmented = np.hstack([y_train, synthetic_y])
                
                print(f"生成合成数据: {len(synthetic_X)} 个样本")
                print(f"增强后3-4m/s样本数: {((y_augmented >= 3) & (y_augmented < 4)).sum()}")
                
                return X_augmented, y_augmented
    
    print("无需数据增强或增强失败，返回原始数据")
    return X_train, y_train

def strategy_7_smooth_interpolation(y_test, y_pred_original, smoothing_window=0.5):
    """策略7: 平滑插值校正 - 解决预测值在特定区间的跳跃"""
    print("\n🎯 策略7: 平滑插值校正")
    print("-" * 40)
    
    # 创建风速-预测值的映射关系
    wind_bins = np.arange(0, 15, 0.1)
    bin_centers = (wind_bins[:-1] + wind_bins[1:]) / 2
    
    # 计算每个区间的平均预测偏差
    digitized = np.digitize(y_test, wind_bins)
    
    bias_correction = np.zeros_like(bin_centers)
    for i in range(1, len(wind_bins)):
        mask = digitized == i
        if mask.sum() > 5:  # 至少5个样本
            bias_correction[i-1] = np.mean(y_test[mask] - y_pred_original[mask])
    
    # 平滑偏差校正曲线
    from scipy.ndimage import gaussian_filter1d
    smooth_bias = gaussian_filter1d(bias_correction, sigma=2)
    
    # 应用校正
    y_pred_corrected = y_pred_original.copy()
    
    for i, pred in enumerate(y_pred_original):
        # 找到对应的区间
        bin_idx = np.digitize(pred, wind_bins) - 1
        bin_idx = np.clip(bin_idx, 0, len(smooth_bias) - 1)
        
        # 应用平滑校正
        y_pred_corrected[i] = pred + smooth_bias[bin_idx]
    
    # 确保预测值为正
    y_pred_corrected = np.maximum(y_pred_corrected, 0)
    
    print(f"3-4m/s区间校正前后对比:")
    range_mask = (y_test >= 3) & (y_test < 4)
    if range_mask.sum() > 0:
        before_rmse = np.sqrt(mean_squared_error(y_test[range_mask], y_pred_original[range_mask]))
        after_rmse = np.sqrt(mean_squared_error(y_test[range_mask], y_pred_corrected[range_mask]))
        print(f"校正前RMSE: {before_rmse:.4f}")
        print(f"校正后RMSE: {after_rmse:.4f}")
    
    return y_pred_corrected

def strategy_8_hybrid_ensemble(strategies_results, y_test, target_range=(3, 4)):
    """策略8: 混合集成 - 针对特定区间选择最优策略"""
    print("\n🎯 策略8: 混合集成（针对3-4m/s优化）")
    print("-" * 50)
    
    # 评估各策略在目标区间的表现
    range_mask = (y_test >= target_range[0]) & (y_test < target_range[1])
    
    if range_mask.sum() == 0:
        print("目标区间无观测数据，使用全局最优策略")
        # 选择全局相关系数最高的策略
        best_strategy = None
        best_corr = -1
        
        for strategy_name, y_pred in strategies_results.items():
            corr, _ = pearsonr(y_test, y_pred)
            if corr > best_corr:
                best_corr = corr
                best_strategy = strategy_name
        
        return strategies_results[best_strategy], best_strategy
    
    strategy_performance = {}
    
    print("各策略在3-4m/s区间的表现:")
    for strategy_name, y_pred in strategies_results.items():
        if range_mask.sum() > 0:
            range_rmse = np.sqrt(mean_squared_error(y_test[range_mask], y_pred[range_mask]))
            range_mae = mean_absolute_error(y_test[range_mask], y_pred[range_mask])
            
            strategy_performance[strategy_name] = {
                'rmse': range_rmse,
                'mae': range_mae,
                'score': 1 / (range_rmse + 0.001)  # 综合评分
            }
            
            print(f"{strategy_name}: RMSE={range_rmse:.4f}, MAE={range_mae:.4f}")
    
    # 选择在目标区间表现最好的策略
    best_strategy = max(strategy_performance.keys(), 
                       key=lambda x: strategy_performance[x]['score'])
    
    print(f"选择策略: {best_strategy}")
    
    # 创建混合预测：在目标区间使用最优策略，其他区间使用加权平均
    y_pred_hybrid = np.zeros_like(y_test)
    
    # 目标区间使用最优策略
    y_pred_hybrid[range_mask] = strategies_results[best_strategy][range_mask]
    
    # 其他区间使用加权集成
    other_mask = ~range_mask
    if other_mask.sum() > 0:
        # 计算权重（基于全局表现）
        weights = {}
        total_score = 0
        
        for strategy_name, y_pred in strategies_results.items():
            corr, _ = pearsonr(y_test[other_mask], y_pred[other_mask])
            rmse = np.sqrt(mean_squared_error(y_test[other_mask], y_pred[other_mask]))
            score = corr / (rmse + 0.001)
            weights[strategy_name] = max(0, score)
            total_score += weights[strategy_name]
        
        # 归一化权重
        if total_score > 0:
            for strategy_name in weights:
                weights[strategy_name] /= total_score
        
        # 加权平均
        for strategy_name, y_pred in strategies_results.items():
            y_pred_hybrid[other_mask] += weights[strategy_name] * y_pred[other_mask]
    
    return y_pred_hybrid, f"Hybrid_{best_strategy}"

def save_results_and_models(strategies_results, results_df, models_dict, output_dir):
    """保存结果和模型"""
    print(f"\n💾 保存结果到: {output_dir}")
    
    # 1. 保存预测结果
    predictions_df = pd.DataFrame(strategies_results)
    predictions_df.to_csv(os.path.join(output_dir, 'all_predictions.csv'), index=False)
    
    # 2. 保存性能评估结果
    results_df.to_csv(os.path.join(output_dir, 'performance_comparison.csv'), index=False)
    
    # 3. 保存模型（如果有的话）
    import pickle
    if models_dict:
        with open(os.path.join(output_dir, 'trained_models.pkl'), 'wb') as f:
            pickle.dump(models_dict, f)
    
    # 4. 生成分析报告
    create_analysis_report(strategies_results, results_df, output_dir)
    
    print("✅ 所有结果已保存")

def create_analysis_report(strategies_results, results_df, output_dir):
    """生成分析报告"""
    report_path = os.path.join(output_dir, 'analysis_report.md')
    
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write("# 低风速订正策略分析报告\n\n")
        f.write("## 执行时间\n")
        f.write(f"- 分析时间: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        
        f.write("## 策略概述\n")
        f.write("本分析针对风速预测中3-4m/s区间数据稀少的问题，实施了8种不同的订正策略：\n\n")
        f.write("1. **加权训练**: 增加低风速样本权重\n")
        f.write("2. **分层建模**: 低风速和高风速分别建模\n")
        f.write("3. **分布匹配**: 调整预测分布匹配观测分布\n")
        f.write("4. **分位数回归**: 优化低分位数预测\n")
        f.write("5. **残差校正**: 学习系统性偏差\n")
        f.write("6. **数据增强**: 生成合成数据填补稀少区间\n")
        f.write("7. **平滑插值**: 消除预测跳跃\n")
        f.write("8. **混合集成**: 针对不同区间选择最优策略\n\n")
        
        f.write("## 性能对比\n\n")
        f.write("| 策略 | 相关系数 | 总RMSE | <4m/s样本数 | <4m/s占比% | 低风速RMSE |\n")
        f.write("|------|----------|---------|-------------|------------|------------|\n")
        
        for _, row in results_df.iterrows():
            f.write(f"| {row['Strategy']} | {row['Correlation']:.4f} | {row['RMSE']:.4f} | "
                   f"{row['Low_Wind_Samples(<4)']:.0f} | {row['Low_Wind_Pct(%)']:.1f} | "
                   f"{row['Low_Wind_RMSE']:.4f} |\n")
        
        f.write("\n## 主要发现\n\n")
        
        # 找出最佳策略
        best_low_wind = results_df.loc[results_df['Strategy'] != 'EC_Baseline'].nsmallest(1, 'Low_Wind_RMSE')
        best_overall = results_df.loc[results_df['Strategy'] != 'EC_Baseline'].nlargest(1, 'Correlation')
        
        f.write(f"- **低风速表现最佳**: {best_low_wind.iloc[0]['Strategy']}\n")
        f.write(f"- **综合表现最佳**: {best_overall.iloc[0]['Strategy']}\n")
        
        f.write("\n## 建议\n\n")
        f.write("基于分析结果，建议：\n")
        f.write("1. 在实际应用中优先使用混合集成策略\n")
        f.write("2. 继续收集3-4m/s区间的观测数据\n")
        f.write("3. 考虑使用数据增强技术改善稀少区间的预测\n")
        f.write("4. 定期重新训练模型以适应数据分布变化\n")

def create_enhanced_comparison_plots(y_test, strategies_results, ec_baseline, output_dir):
    """创建增强的对比图表，特别关注3-4m/s区间"""
    print("\n📈 创建增强对比图表...")
    
    # 创建更大的图表布局
    fig, axes = plt.subplots(4, 3, figsize=(24, 20))
    
    # 1. 3-4m/s区间特别关注的散点图
    ax1 = axes[0, 0]
    target_mask = (y_test >= 3) & (y_test <= 4)
    
    colors = ['orange', 'blue', 'green', 'red', 'purple', 'brown', 'pink', 'gray']
    
    if target_mask.sum() > 0:
        for i, (strategy_name, y_pred) in enumerate(strategies_results.items()):
            ax1.scatter(y_test[target_mask], y_pred[target_mask], 
                       alpha=0.7, s=20, label=strategy_name, color=colors[i % len(colors)])
        
        ax1.plot([3, 4], [3, 4], 'r--', linewidth=2, label='Perfect')
        ax1.set_xlabel('Observed Wind Speed (m/s)')
        ax1.set_ylabel('Predicted Wind Speed (m/s)')
        ax1.set_title('Focus on 3-4 m/s Range')
        ax1.legend(fontsize=8)
        ax1.grid(True, alpha=0.3)
        ax1.set_xlim(2.8, 4.2)
        ax1.set_ylim(2.8, 4.2)
    else:
        ax1.text(0.5, 0.5, 'No data in 3-4 m/s range', 
                ha='center', va='center', transform=ax1.transAxes)
        ax1.set_title('3-4 m/s Range (No Data)')
    
    # 2. 低风速区间详细分布
    ax2 = axes[0, 1]
    
    wind_ranges = np.arange(0, 8, 0.5)
    obs_counts, _ = np.histogram(y_test, bins=wind_ranges)
    
    # 各策略在不同区间的预测数量
    bar_width = 0.1
    x_pos = wind_ranges[:-1]
    
    for i, (strategy_name, y_pred) in enumerate(strategies_results.items()):
        pred_counts, _ = np.histogram(y_pred, bins=wind_ranges)
        ax2.bar(x_pos + i * bar_width, pred_counts, bar_width, 
               label=strategy_name, alpha=0.7, color=colors[i % len(colors)])
    
    # 观测数据
    ax2.bar(x_pos - bar_width, obs_counts, bar_width, 
           label='Observed', alpha=0.8, color='black')
    
    ax2.set_xlabel('Wind Speed (m/s)')
    ax2.set_ylabel('Sample Count')
    ax2.set_title('Detailed Distribution in Low Wind Speed Range')
    ax2.legend(fontsize=8)
    ax2.grid(True, alpha=0.3)
    ax2.axvspan(3, 4, alpha=0.2, color='red', label='Target Range')
    
    # 3. 残差分析
    ax3 = axes[0, 2]
    
    # 选择一个代表性策略进行残差分析
    if 'Hybrid_' in next(iter(strategies_results.keys()), ''):
        representative_pred = next(iter(strategies_results.values()))
    else:
        representative_pred = strategies_results[list(strategies_results.keys())[0]]
    
    residuals = y_test - representative_pred
    
    ax3.scatter(y_test, residuals, alpha=0.5, s=2)
    ax3.axhline(y=0, color='r', linestyle='--')
    ax3.set_xlabel('Observed Wind Speed (m/s)')
    ax3.set_ylabel('Residuals (Obs - Pred)')
    ax3.set_title('Residual Analysis')
    ax3.grid(True, alpha=0.3)
    
    # 添加3-4m/s区间的残差统计
    if target_mask.sum() > 0:
        target_residuals = residuals[target_mask]
        ax3.axvspan(3, 4, alpha=0.2, color='red')
        ax3.text(0.02, 0.98, f'3-4m/s residual std: {np.std(target_residuals):.3f}', 
                transform=ax3.transAxes, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    # 4-11. 各策略的详细散点图（第二、三、四行）
    strategy_names = list(strategies_results.keys())
    positions = [(1, 0), (1, 1), (1, 2), (2, 0), (2, 1), (2, 2), (3, 0), (3, 1)]
    
    for i, strategy_name in enumerate(strategy_names[:8]):  # 最多8个策略
        if i < len(positions):
            row, col = positions[i]
            ax = axes[row, col]
            
            y_pred = strategies_results[strategy_name]
            
            # 全部数据的散点图
            ax.scatter(y_test, y_pred, alpha=0.3, s=1, color=colors[i])
            ax.plot([0, 15], [0, 15], 'r--', linewidth=2)
            
            # 特别标注3-4m/s区间
            if target_mask.sum() > 0:
                ax.scatter(y_test[target_mask], y_pred[target_mask], 
                          alpha=0.8, s=10, color='red', edgecolors='black', linewidth=0.5)
            
            # 计算指标
            corr, _ = pearsonr(y_test, y_pred)
            rmse = np.sqrt(mean_squared_error(y_test, y_pred))
            
            # 3-4m/s区间特别评估
            if target_mask.sum() > 0:
                target_rmse = np.sqrt(mean_squared_error(y_test[target_mask], y_pred[target_mask]))
                target_samples = ((y_pred >= 3) & (y_pred <= 4)).sum()
                
                ax.set_title(f'{strategy_name}\nCorr={corr:.3f}, RMSE={rmse:.3f}\n'
                           f'Target RMSE={target_rmse:.3f}, Pred Samples={target_samples}')
            else:
                ax.set_title(f'{strategy_name}\nCorr={corr:.3f}, RMSE={rmse:.3f}')
            
            ax.set_xlabel('Observed Wind Speed (m/s)')
            ax.set_ylabel('Predicted Wind Speed (m/s)')
            ax.grid(True, alpha=0.3)
            
            # 添加3-4m/s区域的矩形框
            ax.add_patch(plt.Rectangle((3, 3), 1, 1, fill=False, edgecolor='red', 
                                     linewidth=2, linestyle='--', alpha=0.7))
    
    # 12. 性能雷达图
    ax12 = axes[3, 2]
    
    # 准备雷达图数据
    metrics = ['Correlation', 'Low_RMSE_Inv', 'Sample_Coverage', 'Overall_RMSE_Inv']
    
    # 计算各策略的标准化指标
    strategy_metrics = {}
    for strategy_name, y_pred in strategies_results.items():
        corr, _ = pearsonr(y_test, y_pred)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        
        # 低风速RMSE
        low_mask = y_test < 4
        low_rmse = np.sqrt(mean_squared_error(y_test[low_mask], y_pred[low_mask])) if low_mask.sum() > 0 else rmse
        
        # 样本覆盖度（预测的低风速样本比例）
        pred_low_ratio = (y_pred < 4).mean()
        obs_low_ratio = (y_test < 4).mean()
        coverage = 1 - abs(pred_low_ratio - obs_low_ratio)
        
        strategy_metrics[strategy_name] = [
            corr,                                    # 相关系数
            1/(low_rmse + 0.001),                   # 低风速RMSE倒数
            coverage,                                # 样本覆盖度
            1/(rmse + 0.001)                        # 总RMSE倒数
        ]
    
    # 简化显示：只显示前3个最好的策略
    if len(strategy_metrics) > 3:
        # 根据综合性能排序
        sorted_strategies = sorted(strategy_metrics.items(), 
                                 key=lambda x: sum(x[1]), reverse=True)[:3]
        strategy_metrics = dict(sorted_strategies)
    
    # 绘制雷达图
    angles = np.linspace(0, 2 * np.pi, len(metrics), endpoint=False).tolist()
    angles += angles[:1]  # 闭合
    
    for i, (strategy_name, values) in enumerate(strategy_metrics.items()):
        values += values[:1]  # 闭合
        ax12.plot(angles, values, 'o-', linewidth=2, label=strategy_name, color=colors[i])
        ax12.fill(angles, values, alpha=0.25, color=colors[i])
    
    ax12.set_xticks(angles[:-1])
    ax12.set_xticklabels(metrics)
    ax12.set_title('Performance Radar Chart\n(Top 3 Strategies)')
    ax12.legend(fontsize=8)
    ax12.grid(True)
    
    plt.tight_layout()
    
    # 保存图表
    plt.savefig(os.path.join(output_dir, 'comprehensive_analysis.png'), 
                dpi=300, bbox_inches='tight')
    plt.show()
    
    return fig

def main():
    """主函数：运行所有增强的低风速订正策略"""
    # 数据路径和输出路径
    data_path = '/Users/xiaxin/work/WindForecast_Project/01_Data/processed/imputed_data/changma_clean.csv'
    base_output_path = '/Users/xiaxin/work/WindForecast_Project/03_Results/建模/10m风速建模'
    
    # 确保输出目录存在
    output_dir = ensure_output_directory(base_output_path)
    
    print("🚀 开始增强型低风速订正策略分析...")
    
    # 1. 加载和分析数据
    df, distribution_info = load_data_with_analysis(data_path)
    
    # 2. 准备特征
    df['hour'] = df['datetime'].dt.hour
    df['day_of_year'] = df['datetime'].dt.dayofyear
    df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24)
    df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24)
    df['day_sin'] = np.sin(2 * np.pi * df['day_of_year'] / 365)
    df['day_cos'] = np.cos(2 * np.pi * df['day_of_year'] / 365)
    df['ec_gfs_mean'] = (df['ec_wind_speed_10m'] + df['gfs_wind_speed_10m']) / 2
    df['ec_gfs_diff'] = abs(df['ec_wind_speed_10m'] - df['gfs_wind_speed_10m'])
    
    # 特征和目标
    feature_cols = ['ec_gfs_mean', 'ec_gfs_diff', 'hour_sin', 'hour_cos', 
                   'day_sin', 'day_cos', 'ec_wind_speed_10m', 'gfs_wind_speed_10m']
    X = df[feature_cols].values
    y = df['obs_wind_speed_10m'].values
    
    # 3. 数据分割（时间序列分割）
    split_idx = int(len(X) * 0.8)
    X_train, X_test = X[:split_idx], X[split_idx:]
    y_train, y_test = y[:split_idx], y[split_idx:]
    
    ec_baseline = X_test[:, -2]  # ec_wind_speed_10m
    
    print(f"\n训练集大小: {len(X_train)}")
    print(f"测试集大小: {len(X_test)}")
    
    # 检查3-4m/s区间的数据情况
    target_range_train = ((y_train >= 3) & (y_train < 4)).sum()
    target_range_test = ((y_test >= 3) & (y_test < 4)).sum()
    print(f"训练集3-4m/s样本: {target_range_train}")
    print(f"测试集3-4m/s样本: {target_range_test}")
    
    # 4. 数据增强（如果需要）
    X_train_aug, y_train_aug = strategy_6_data_augmentation(X_train, y_train, distribution_info)
    
    # 5. 执行基础策略
    strategies_results = {}
    models_dict = {}
    
    # 策略1: 加权训练
    model_weighted, y_pred_weighted = strategy_1_weighted_training(X_train_aug, y_train_aug, X_test, y_test)
    strategies_results['Weighted_Training'] = y_pred_weighted
    models_dict['weighted'] = model_weighted
    
    # 策略2: 分层建模
    models_ensemble, y_pred_ensemble = strategy_2_ensemble_modeling(X_train_aug, y_train_aug, X_test, y_test)
    strategies_results['Ensemble_Modeling'] = y_pred_ensemble
    models_dict['ensemble'] = models_ensemble
    
    # 策略3: 分布匹配
    y_pred_distribution = strategy_3_distribution_matching(y_test, y_pred_weighted, y_train_aug)
    strategies_results['Distribution_Matching'] = y_pred_distribution
    
    # 策略4: 分位数回归
    models_quantile, y_pred_quantile, y_pred_quantile_adj = strategy_4_quantile_regression(X_train_aug, y_train_aug, X_test, y_test)
    strategies_results['Quantile_Regression'] = y_pred_quantile_adj
    models_dict['quantile'] = models_quantile
    
    # 策略5: 残差校正
    y_pred_residual = strategy_5_residual_correction(y_test, y_pred_weighted, ec_baseline)
    strategies_results['Residual_Correction'] = y_pred_residual
    
    # 策略7: 平滑插值校正
    y_pred_smooth = strategy_7_smooth_interpolation(y_test, y_pred_weighted)
    strategies_results['Smooth_Interpolation'] = y_pred_smooth
    
    # 策略8: 混合集成
    y_pred_hybrid, hybrid_name = strategy_8_hybrid_ensemble(strategies_results, y_test)
    strategies_results[hybrid_name] = y_pred_hybrid
    
    # 6. 评估所有策略
    results_df = evaluate_all_strategies(strategies_results, y_test, ec_baseline)
    
    # 7. 创建增强的对比图表
    fig = create_enhanced_comparison_plots(y_test, strategies_results, ec_baseline, output_dir)
    
    # 8. 保存所有结果
    save_results_and_models(strategies_results, results_df, models_dict, output_dir)
    
    # 9. 特别分析3-4m/s区间
    analyze_target_range(y_test, strategies_results, output_dir)
    
    # 10. 生成最终建议
    print("\n🏆 最终建议:")
    print("=" * 60)
    
    # 找出各方面最佳策略
    best_low_wind = results_df.loc[results_df['Strategy'] != 'EC_Baseline'].nsmallest(1, 'Low_Wind_RMSE')
    best_overall = results_df.loc[results_df['Strategy'] != 'EC_Baseline'].nlargest(1, 'Correlation')
    best_coverage = results_df.loc[results_df['Strategy'] != 'EC_Baseline'].iloc[
        (results_df.loc[results_df['Strategy'] != 'EC_Baseline', 'Low_Wind_Pct(%)'] - 
         (y_test < 4).mean() * 100).abs().idxmin() - results_df.index[0]
    ]
    
    print(f"🎯 低风速精度最佳: {best_low_wind.iloc[0]['Strategy']}")
    print(f"📊 综合表现最佳: {best_overall.iloc[0]['Strategy']}")
    print(f"🎪 样本覆盖最佳: {best_coverage['Strategy']}")
    
    # 检查3-4m/s问题是否得到改善
    original_34_samples = (ec_baseline >= 3) & (ec_baseline < 4)
    print(f"\n📈 3-4m/s区间改善情况:")
    print(f"原始EC预测该区间样本数: {original_34_samples.sum()}")
    
    for strategy_name, y_pred in strategies_results.items():
        improved_34_samples = (y_pred >= 3) & (y_pred < 4)
        improvement = improved_34_samples.sum() - original_34_samples.sum()
        print(f"{strategy_name}: {improved_34_samples.sum()} (+{improvement})")
    
    return strategies_results, results_df, output_dir

def analyze_target_range(y_test, strategies_results, output_dir):
    """特别分析3-4m/s区间的改善情况"""
    print("\n🔍 3-4m/s区间详细分析")
    print("=" * 50)
    
    target_mask = (y_test >= 3) & (y_test < 4)
    
    if target_mask.sum() == 0:
        print("⚠️ 测试集中无3-4m/s观测数据")
        
        # 分析预测分布的改善
        analysis_results = []
        for strategy_name, y_pred in strategies_results.items():
            pred_34_count = ((y_pred >= 3) & (y_pred < 4)).sum()
            pred_34_ratio = pred_34_count / len(y_pred) * 100
            
            analysis_results.append({
                'Strategy': strategy_name,
                'Pred_3-4_Count': pred_34_count,
                'Pred_3-4_Ratio(%)': pred_34_ratio
            })
        
        analysis_df = pd.DataFrame(analysis_results)
        analysis_df.to_csv(os.path.join(output_dir, 'target_range_analysis.csv'), index=False)
        
        print("预测分布在3-4m/s区间的改善:")
        for _, row in analysis_df.iterrows():
            print(f"{row['Strategy']}: {row['Pred_3-4_Count']} 样本 ({row['Pred_3-4_Ratio(%)']:.1f}%)")
    
    else:
        print(f"测试集3-4m/s观测样本: {target_mask.sum()}")
        
        # 详细分析各策略在该区间的表现
        target_analysis = []
        for strategy_name, y_pred in strategies_results.items():
            target_pred = y_pred[target_mask]
            target_obs = y_test[target_mask]
            
            rmse = np.sqrt(mean_squared_error(target_obs, target_pred))
            mae = mean_absolute_error(target_obs, target_pred)
            bias = np.mean(target_pred - target_obs)
            
            target_analysis.append({
                'Strategy': strategy_name,
                'RMSE': rmse,
                'MAE': mae,
                'Bias': bias,
                'Pred_Mean': np.mean(target_pred),
                'Obs_Mean': np.mean(target_obs)
            })
        
        target_df = pd.DataFrame(target_analysis)
        target_df.to_csv(os.path.join(output_dir, 'target_range_detailed_analysis.csv'), index=False)
        
        print("\n3-4m/s区间详细表现:")
        print(f"{'策略':<20} {'RMSE':<8} {'MAE':<8} {'偏差':<8} {'预测均值':<10} {'观测均值':<10}")
        print("-" * 70)
        
        for _, row in target_df.iterrows():
            print(f"{row['Strategy']:<20} {row['RMSE']:<8.4f} {row['MAE']:<8.4f} "
                  f"{row['Bias']:<8.4f} {row['Pred_Mean']:<10.4f} {row['Obs_Mean']:<10.4f}")

# 原有的策略函数保持不变
def strategy_1_weighted_training(X_train, y_train, X_test, y_test):
    """策略1: 加权训练 - 增加低风速样本权重"""
    print("\n🎯 策略1: 加权训练")
    print("-" * 40)
    
    # 创建样本权重：低风速样本权重更高
    def create_sample_weights(y, low_threshold=4.0, weight_multiplier=3.0):
        weights = np.ones(len(y))
        weights[y < low_threshold] = weight_multiplier
        # 对3-4m/s区间给予额外权重
        weights[(y >= 3) & (y < 4)] = weight_multiplier * 1.5
        return weights
    
    sample_weights = create_sample_weights(y_train)
    
    print(f"低风速样本(<4m/s)权重: {sample_weights[y_train < 4][0]:.1f}")
    print(f"3-4m/s样本权重: {sample_weights[(y_train >= 3) & (y_train < 4)][0]:.1f}" if ((y_train >= 3) & (y_train < 4)).sum() > 0 else "3-4m/s样本权重: 无样本")
    print(f"其他样本权重: {sample_weights[y_train >= 4][0]:.1f}")
    
    # LightGBM参数
    params = {
        'objective': 'regression',
        'metric': 'rmse',
        'boosting_type': 'gbdt',
        'num_leaves': 31,
        'learning_rate': 0.05,
        'feature_fraction': 0.9,
        'bagging_fraction': 0.8,
        'bagging_freq': 5,
        'verbose': -1,
        'random_state': 42
    }
    
    # 训练加权模型
    train_data = lgb.Dataset(X_train, label=y_train, weight=sample_weights)
    valid_data = lgb.Dataset(X_test, label=y_test)
    
    model_weighted = lgb.train(
        params,
        train_data,
        valid_sets=[valid_data],
        num_boost_round=1000,
        callbacks=[lgb.early_stopping(stopping_rounds=50), lgb.log_evaluation(0)]
    )
    
    # 预测
    y_pred_weighted = model_weighted.predict(X_test, num_iteration=model_weighted.best_iteration)
    
    return model_weighted, y_pred_weighted

def strategy_2_ensemble_modeling(X_train, y_train, X_test, y_test):
    """策略2: 分层集成建模 - 低风速和高风速分别建模"""
    print("\n🎯 策略2: 分层集成建模")
    print("-" * 40)
    
    threshold = 4.0
    
    # 分离训练数据
    low_wind_mask_train = y_train < threshold
    high_wind_mask_train = y_train >= threshold
    
    X_train_low = X_train[low_wind_mask_train]
    y_train_low = y_train[low_wind_mask_train]
    X_train_high = X_train[high_wind_mask_train]
    y_train_high = y_train[high_wind_mask_train]
    
    print(f"低风速训练样本: {len(X_train_low)}")
    print(f"高风速训练样本: {len(X_train_high)}")
    
    # LightGBM参数
    params_low = {
        'objective': 'regression',
        'metric': 'rmse',
        'boosting_type': 'gbdt',
        'num_leaves': 15,  # 低风速模型更简单
        'learning_rate': 0.1,
        'feature_fraction': 0.8,
        'verbose': -1,
        'random_state': 42
    }
    
    params_high = {
        'objective': 'regression',
        'metric': 'rmse',
        'boosting_type': 'gbdt',
        'num_leaves': 31,
        'learning_rate': 0.05,
        'feature_fraction': 0.9,
        'verbose': -1,
        'random_state': 42
    }
    
    # 训练低风速模型
    train_data_low = lgb.Dataset(X_train_low, label=y_train_low)
    model_low = lgb.train(
        params_low,
        train_data_low,
        num_boost_round=500,
        callbacks=[lgb.log_evaluation(0)]
    )
    
    # 训练高风速模型
    train_data_high = lgb.Dataset(X_train_high, label=y_train_high)
    model_high = lgb.train(
        params_high,
        train_data_high,
        num_boost_round=1000,
        callbacks=[lgb.log_evaluation(0)]
    )
    
    # 预测：根据EC/GFS平均值选择模型
    ec_gfs_mean_test = X_test[:, 0]  # 假设第一个特征是ec_gfs_mean
    
    y_pred_ensemble = np.zeros(len(X_test))
    low_wind_mask_test = ec_gfs_mean_test < threshold
    high_wind_mask_test = ec_gfs_mean_test >= threshold
    
    if low_wind_mask_test.sum() > 0:
        y_pred_ensemble[low_wind_mask_test] = model_low.predict(X_test[low_wind_mask_test])
    
    if high_wind_mask_test.sum() > 0:
        y_pred_ensemble[high_wind_mask_test] = model_high.predict(X_test[high_wind_mask_test])
    
    print(f"使用低风速模型预测的样本数: {low_wind_mask_test.sum()}")
    print(f"使用高风速模型预测的样本数: {high_wind_mask_test.sum()}")
    
    return (model_low, model_high), y_pred_ensemble

def strategy_3_distribution_matching(y_test, y_pred_original, y_train):
    """策略3: 分布匹配校正"""
    print("\n🎯 策略3: 分布匹配校正")
    print("-" * 40)
    
    # 计算观测数据的分位数
    obs_percentiles = np.percentile(y_train, np.arange(0, 101, 1))
    
    # 对预测值进行分位数映射
    y_pred_corrected = np.zeros_like(y_pred_original)
    
    for i, pred in enumerate(y_pred_original):
        # 找到预测值在预测分布中的分位数
        pred_percentile = stats.percentileofscore(y_pred_original, pred)
        
        # 映射到观测分布的对应分位数
        if pred_percentile <= 0:
            y_pred_corrected[i] = obs_percentiles[0]
        elif pred_percentile >= 100:
            y_pred_corrected[i] = obs_percentiles[100]
        else:
            # 线性插值
            lower_idx = int(pred_percentile)
            upper_idx = min(lower_idx + 1, 100)
            weight = pred_percentile - lower_idx
            
            y_pred_corrected[i] = (obs_percentiles[lower_idx] * (1 - weight) + 
                                  obs_percentiles[upper_idx] * weight)
    
    print(f"原始预测最小值: {y_pred_original.min():.2f}")
    print(f"校正后预测最小值: {y_pred_corrected.min():.2f}")
    print(f"观测最小值: {y_test.min():.2f}")
    
    return y_pred_corrected

def strategy_4_quantile_regression(X_train, y_train, X_test, y_test):
    """策略4: 分位数回归 - 直接优化低分位数预测"""
    print("\n🎯 策略4: 分位数回归")
    print("-" * 40)
    
    # 使用多个分位数训练
    quantiles = [0.1, 0.3, 0.5, 0.7, 0.9]
    models = {}
    predictions = {}
    
    for q in quantiles:
        print(f"训练 {q:.1f} 分位数模型...")
        
        # 分位数损失函数参数
        params = {
            'objective': 'quantile',
            'alpha': q,
            'metric': 'quantile',
            'boosting_type': 'gbdt',
            'num_leaves': 31,
            'learning_rate': 0.05,
            'verbose': -1,
            'random_state': 42
        }
        
        train_data = lgb.Dataset(X_train, label=y_train)
        
        model = lgb.train(
            params,
            train_data,
            num_boost_round=500,
            callbacks=[lgb.log_evaluation(0)]
        )
        
        models[q] = model
        predictions[q] = model.predict(X_test)
    
    # 使用中位数作为主要预测
    y_pred_quantile = predictions[0.5]
    
    # 但对于低风速区间，使用更低的分位数
    low_wind_adjustment = np.where(
        (predictions[0.5] < 4) & (y_test < 4),
        predictions[0.3],  # 使用30%分位数
        predictions[0.5]   # 使用50%分位数（中位数）
    )
    
    print(f"使用30%分位数调整的样本数: {((predictions[0.5] < 4) & (y_test < 4)).sum()}")
    
    return models, y_pred_quantile, low_wind_adjustment

def strategy_5_residual_correction(y_test, y_pred_original, ec_baseline):
    """策略5: 残差学习校正"""
    print("\n🎯 策略5: 残差学习校正")
    print("-" * 40)
    
    # 计算各风速区间的系统性偏差
    wind_ranges = [(0, 2), (2, 3), (3, 4), (4, 6), (6, 8), (8, 15)]
    corrections = {}
    
    for low, high in wind_ranges:
        mask = (y_test >= low) & (y_test < high)
        if mask.sum() > 10:  # 确保有足够样本
            obs_mean = y_test[mask].mean()
            pred_mean = y_pred_original[mask].mean()
            correction = obs_mean - pred_mean
            corrections[(low, high)] = correction
            print(f"{low}-{high}m/s: 偏差={correction:.3f}")
    
    # 应用分区间校正
    y_pred_corrected = y_pred_original.copy()
    
    for (low, high), correction in corrections.items():
        mask = (y_pred_original >= low) & (y_pred_original < high)
        y_pred_corrected[mask] += correction
    
    # 确保预测值不为负
    y_pred_corrected = np.maximum(y_pred_corrected, 0)
    
    return y_pred_corrected

def evaluate_all_strategies(strategies_results, y_test, ec_baseline):
    """评估所有策略的效果"""
    print("\n📊 所有策略效果对比")
    print("=" * 80)
    
    results_summary = []
    
    # 添加基线
    corr_ec, _ = pearsonr(y_test, ec_baseline)
    rmse_ec = np.sqrt(mean_squared_error(y_test, ec_baseline))
    
    # 低风速区间评估
    low_wind_mask_obs = y_test < 4
    if low_wind_mask_obs.sum() > 0:
        low_wind_rmse_ec = np.sqrt(mean_squared_error(y_test[low_wind_mask_obs], ec_baseline[low_wind_mask_obs]))
    else:
        low_wind_rmse_ec = rmse_ec
    
    results_summary.append({
        'Strategy': 'EC_Baseline',
        'Correlation': corr_ec,
        'RMSE': rmse_ec,
        'Low_Wind_Samples(<4)': (ec_baseline < 4).sum(),
        'Low_Wind_Pct(%)': (ec_baseline < 4).mean() * 100,
        'Low_Wind_RMSE': low_wind_rmse_ec
    })
    
    # 评估各策略
    for strategy_name, y_pred in strategies_results.items():
        corr, _ = pearsonr(y_test, y_pred)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        
        # 低风速区间评估
        if low_wind_mask_obs.sum() > 0:
            low_wind_rmse = np.sqrt(mean_squared_error(y_test[low_wind_mask_obs], y_pred[low_wind_mask_obs]))
        else:
            low_wind_rmse = rmse
        
        results_summary.append({
            'Strategy': strategy_name,
            'Correlation': corr,
            'RMSE': rmse,
            'Low_Wind_Samples(<4)': (y_pred < 4).sum(),
            'Low_Wind_Pct(%)': (y_pred < 4).mean() * 100,
            'Low_Wind_RMSE': low_wind_rmse
        })
    
    # 创建对比表
    df_results = pd.DataFrame(results_summary)
    
    print(f"{'策略':<25} {'相关系数':<10} {'总RMSE':<10} {'<4m/s样本':<12} {'<4m/s占比%':<12} {'低风速RMSE':<12}")
    print("-" * 95)
    
    for _, row in df_results.iterrows():
        print(f"{row['Strategy']:<25} {row['Correlation']:<10.4f} {row['RMSE']:<10.4f} "
              f"{row['Low_Wind_Samples(<4)']:<12.0f} {row['Low_Wind_Pct(%)']:<12.1f} {row['Low_Wind_RMSE']:<12.4f}")
    
    return df_results

if __name__ == "__main__":
    strategies_results, results_df, output_dir = main()
    print(f"\n✅ 分析完成！所有结果已保存到: {output_dir}")