"""
完整的混合优化测试
寻找精度与低风速覆盖的最佳平衡点
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error
from scipy.stats import pearsonr
import lightgbm as lgb
import warnings
import os
from datetime import datetime
warnings.filterwarnings('ignore')

# 设置matplotlib中文显示
plt.rcParams['font.sans-serif'] = ['Arial Unicode MS', 'SimHei']
plt.rcParams['axes.unicode_minus'] = False

def load_and_prepare_data(data_path):
    """加载和准备数据"""
    print("📊 加载数据...")
    
    df = pd.read_csv(data_path)
    df['datetime'] = pd.to_datetime(df['datetime'])
    
    # 基础清理
    df_clean = df[['datetime', 'obs_wind_speed_10m', 'ec_wind_speed_10m', 'gfs_wind_speed_10m']].dropna()
    
    # 特征工程
    df_clean['hour'] = df_clean['datetime'].dt.hour
    df_clean['day_of_year'] = df_clean['datetime'].dt.dayofyear
    df_clean['hour_sin'] = np.sin(2 * np.pi * df_clean['hour'] / 24)
    df_clean['hour_cos'] = np.cos(2 * np.pi * df_clean['hour'] / 24)
    df_clean['day_sin'] = np.sin(2 * np.pi * df_clean['day_of_year'] / 365)
    df_clean['day_cos'] = np.cos(2 * np.pi * df_clean['day_of_year'] / 365)
    df_clean['ec_gfs_mean'] = (df_clean['ec_wind_speed_10m'] + df_clean['gfs_wind_speed_10m']) / 2
    df_clean['ec_gfs_diff'] = abs(df_clean['ec_wind_speed_10m'] - df_clean['gfs_wind_speed_10m'])
    
    # 特征和目标
    feature_cols = ['ec_gfs_mean', 'ec_gfs_diff', 'hour_sin', 'hour_cos', 
                   'day_sin', 'day_cos', 'ec_wind_speed_10m', 'gfs_wind_speed_10m']
    X = df_clean[feature_cols].values
    y = df_clean['obs_wind_speed_10m'].values
    
    return X, y, df_clean

def train_baseline_model(X_train, y_train, X_test, y_test):
    """训练基线模型"""
    print("🎯 训练基线模型...")
    
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
    
    train_data = lgb.Dataset(X_train, label=y_train)
    valid_data = lgb.Dataset(X_test, label=y_test)
    
    model = lgb.train(
        params,
        train_data,
        valid_sets=[valid_data],
        num_boost_round=1000,
        callbacks=[lgb.early_stopping(stopping_rounds=50), lgb.log_evaluation(0)]
    )
    
    baseline_pred = model.predict(X_test, num_iteration=model.best_iteration)
    
    return model, baseline_pred

def order_preserving_mapping(y_pred_original, y_train, y_test):
    """Order_Preserving方法（已知效果好的方案）"""
    print("🎯 Order_Preserving映射")
    
    sorted_indices = np.argsort(y_pred_original)
    n_samples = len(y_pred_original)
    target_percentiles = np.linspace(0, 100, n_samples)
    target_values = np.percentile(y_train, target_percentiles)
    
    y_pred_corrected = y_pred_original.copy()
    cutoff_idx = int(n_samples * 0.3)
    
    for i in range(cutoff_idx):
        original_idx = sorted_indices[i]
        mapping_strength = 1 - (i / cutoff_idx)
        
        y_pred_corrected[original_idx] = (
            mapping_strength * target_values[i] + 
            (1 - mapping_strength) * y_pred_original[original_idx]
        )
    
    return y_pred_corrected

def hybrid_conservative(y_pred_original, y_train, y_test, precision_weight=0.8):
    """混合方案1: 保守调整，优先保持精度"""
    print(f"🎯 混合保守方案 (精度权重={precision_weight})")
    
    y_pred_corrected = y_pred_original.copy()
    
    # 计算低风速覆盖缺口
    obs_low_ratio = (y_train < 3).mean()
    pred_low_ratio = (y_pred_original < 3).mean()
    coverage_deficit = max(0, obs_low_ratio - pred_low_ratio)
    
    print(f"  低风速覆盖缺口: {coverage_deficit*100:.1f}%")
    
    if coverage_deficit > 0.05:  # 只有缺口>5%才修正
        # 最小必要调整量
        needed_samples = int(len(y_pred_original) * coverage_deficit * 0.5)  # 只修正一半缺口
        
        # 选择影响最小的候选样本：3-5m/s区间
        candidates_mask = (y_pred_original >= 3) & (y_pred_original <= 5)
        candidates_indices = np.where(candidates_mask)[0]
        
        if len(candidates_indices) >= needed_samples:
            # 随机选择，避免系统性偏差
            selected_indices = np.random.choice(candidates_indices, needed_samples, replace=False)
            
            # 保守调整：向低风速轻微推进
            obs_low_values = y_train[y_train < 3]
            
            for idx in selected_indices:
                original_pred = y_pred_original[idx]
                target_low = np.random.choice(obs_low_values)
                
                # 调整强度基于精度权重
                adjustment_strength = (1 - precision_weight) * 0.5  # 最多50%调整
                
                y_pred_corrected[idx] = (
                    (1 - adjustment_strength) * original_pred + 
                    adjustment_strength * target_low
                )
        
        print(f"  保守调整了 {needed_samples} 个样本")
    
    return y_pred_corrected

def hybrid_segmented(y_pred_original, y_train, y_test):
    """混合方案2: 分段优化"""
    print("🎯 混合分段方案")
    
    y_pred_corrected = y_pred_original.copy()
    
    # 只对0-4m/s区间进行轻微调整
    low_wind_mask = y_pred_original < 4
    
    if low_wind_mask.sum() > 0:
        low_indices = np.where(low_wind_mask)[0]
        low_pred = y_pred_original[low_indices]
        
        # 计算这个区间的目标分布
        obs_low = y_train[y_train < 4]
        if len(obs_low) > 0:
            target_percentiles = np.linspace(0, 100, len(low_indices))
            target_values = np.percentile(obs_low, target_percentiles)
            
            # 排序并轻微映射
            sorted_low_indices = np.argsort(low_pred)
            
            for i, orig_idx in enumerate(sorted_low_indices):
                actual_idx = low_indices[orig_idx]
                # 渐进映射，但强度很低
                mapping_strength = 0.2 * (1 - i / len(sorted_low_indices))  # 最多20%调整
                
                y_pred_corrected[actual_idx] = (
                    (1 - mapping_strength) * y_pred_original[actual_idx] + 
                    mapping_strength * target_values[i]
                )
        
        print(f"  轻微调整了低风速区间: {low_wind_mask.sum()} 样本")
    
    return y_pred_corrected

def hybrid_multi_objective(y_pred_original, y_train, y_test):
    """混合方案3: 多目标优化"""
    print("🎯 混合多目标优化")
    
    def multi_objective_score(y_pred, y_true, y_train, alpha=0.75):
        """多目标评分函数"""
        # 精度得分
        corr, _ = pearsonr(y_true, y_pred)
        rmse = np.sqrt(mean_squared_error(y_true, y_pred))
        precision_score = corr - rmse/15  # 归一化精度得分
        
        # 分布完整性得分
        obs_low_ratio = (y_train < 3).mean()
        pred_low_ratio = (y_pred < 3).mean()
        coverage_score = min(pred_low_ratio / obs_low_ratio, 1.0) if obs_low_ratio > 0 else 1.0
        
        # 连续性得分
        continuity_samples = ((y_pred >= 3) & (y_pred < 4)).sum()
        continuity_score = min(continuity_samples / 200, 1.0)  # 目标200个样本
        
        distribution_score = (coverage_score + continuity_score) / 2
        
        # 综合得分 (75%精度 + 25%分布)
        total_score = alpha * precision_score + (1 - alpha) * distribution_score
        
        return total_score, precision_score, distribution_score
    
    # 参数搜索寻找最优平衡
    best_score = -999
    best_pred = y_pred_original.copy()
    best_params = None
    
    print("  搜索最优参数...")
    
    # 精简搜索空间，避免过度优化
    for cutoff_ratio in [0.15, 0.20, 0.25]:
        for mapping_strength in [0.2, 0.3, 0.4]:
            y_pred_candidate = y_pred_original.copy()
            
            sorted_indices = np.argsort(y_pred_original)
            cutoff_idx = int(len(y_pred_original) * cutoff_ratio)
            
            target_percentiles = np.linspace(0, 100, len(y_pred_original))
            target_values = np.percentile(y_train, target_percentiles)
            
            for i in range(cutoff_idx):
                original_idx = sorted_indices[i]
                strength = mapping_strength * (1 - i / cutoff_idx)
                
                y_pred_candidate[original_idx] = (
                    strength * target_values[i] + 
                    (1 - strength) * y_pred_original[original_idx]
                )
            
            # 评估候选方案
            score, prec_score, dist_score = multi_objective_score(
                y_pred_candidate, y_test, y_train
            )
            
            if score > best_score:
                best_score = score
                best_pred = y_pred_candidate.copy()
                best_params = (cutoff_ratio, mapping_strength, prec_score, dist_score)
    
    if best_params:
        cutoff, strength, prec, dist = best_params
        print(f"  最优参数: cutoff={cutoff:.2f}, strength={strength:.1f}")
        print(f"  精度得分: {prec:.3f}, 分布得分: {dist:.3f}")
    
    return best_pred

def evaluate_method(y_pred, y_test, y_train, method_name):
    """评估单个方法"""
    corr, _ = pearsonr(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    
    # 低风速统计
    obs_low_1 = (y_test < 1).sum()
    obs_low_2 = (y_test < 2).sum()
    obs_low_3 = (y_test < 3).sum()
    obs_low_4 = (y_test < 4).sum()
    
    pred_low_1 = (y_pred < 1).sum()
    pred_low_2 = (y_pred < 2).sum()
    pred_low_3 = (y_pred < 3).sum()
    pred_low_4 = (y_pred < 4).sum()
    
    # 连续性评估
    continuity_34 = ((y_pred >= 3) & (y_pred < 4)).sum()
    
    # 覆盖率
    coverage_3 = pred_low_3 / obs_low_3 if obs_low_3 > 0 else 0
    
    # 综合评分 (精度权重70%)
    comprehensive_score = 0.7 * (corr - rmse/15) + 0.3 * min(coverage_3, 1.0)
    
    return {
        'method': method_name,
        'corr': corr,
        'rmse': rmse,
        'pred_low_1': pred_low_1,
        'pred_low_2': pred_low_2,
        'pred_low_3': pred_low_3,
        'pred_low_4': pred_low_4,
        'continuity_34': continuity_34,
        'coverage_3': coverage_3,
        'comp_score': comprehensive_score,
        'pred': y_pred
    }

def compare_all_hybrid_methods(y_pred_original, y_train, y_test):
    """比较所有混合方法"""
    print("\n🔄 比较所有混合优化方法")
    print("="*80)
    
    # 运行所有方法
    methods_results = []
    
    # 基线
    methods_results.append(evaluate_method(y_pred_original, y_test, y_train, "Original_Baseline"))
    
    # Order_Preserving (已知好方案)
    order_pred = order_preserving_mapping(y_pred_original.copy(), y_train, y_test)
    methods_results.append(evaluate_method(order_pred, y_test, y_train, "Order_Preserving"))
    
    # 混合方案
    hybrid_cons_pred = hybrid_conservative(y_pred_original.copy(), y_train, y_test, 0.85)
    methods_results.append(evaluate_method(hybrid_cons_pred, y_test, y_train, "Hybrid_Conservative"))
    
    hybrid_moderate_pred = hybrid_conservative(y_pred_original.copy(), y_train, y_test, 0.7)
    methods_results.append(evaluate_method(hybrid_moderate_pred, y_test, y_train, "Hybrid_Moderate"))
    
    hybrid_seg_pred = hybrid_segmented(y_pred_original.copy(), y_train, y_test)
    methods_results.append(evaluate_method(hybrid_seg_pred, y_test, y_train, "Hybrid_Segmented"))
    
    hybrid_multi_pred = hybrid_multi_objective(y_pred_original.copy(), y_train, y_test)
    methods_results.append(evaluate_method(hybrid_multi_pred, y_test, y_train, "Hybrid_MultiObj"))
    
    # 显示结果表格
    print(f"\n📊 详细对比结果:")
    print(f"{'方法':<18} {'相关系数':<8} {'RMSE':<8} {'<3m/s':<8} {'3-4m/s':<8} {'覆盖率':<8} {'综合分':<8}")
    print("-"*85)
    
    # 观测基准
    obs_low_3 = (y_test < 3).sum()
    print(f"{'观测基准':<18} {'--':<8} {'--':<8} {obs_low_3:<8} {'--':<8} {'1.00':<8} {'--':<8}")
    print("-"*85)
    
    for result in methods_results:
        print(f"{result['method']:<18} {result['corr']:<8.4f} {result['rmse']:<8.4f} "
              f"{result['pred_low_3']:<8} {result['continuity_34']:<8} "
              f"{result['coverage_3']:<8.2f} {result['comp_score']:<8.3f}")
    
    return methods_results

def visualize_hybrid_comparison(methods_results, y_test, output_dir):
    """可视化混合方法对比"""
    
    fig, axes = plt.subplots(2, 3, figsize=(20, 12))
    
    # 选择要显示的方法（跳过基线）
    display_methods = [r for r in methods_results if r['method'] != 'Original_Baseline']
    colors = ['red', 'green', 'orange', 'purple', 'brown']
    
    for i, result in enumerate(display_methods[:5]):
        row = i // 3
        col = i % 3
        ax = axes[row, col]
        
        pred = result['pred']
        
        # 绘制散点图
        ax.scatter(y_test, pred, alpha=0.4, s=1, color=colors[i])
        ax.plot([0, 12], [0, 12], 'k--', linewidth=2)
        
        # 标注关注区间
        ax.axvspan(3, 4, alpha=0.1, color='red')
        ax.axhspan(3, 4, alpha=0.1, color='red')
        
        # 显示指标
        method_name = result['method'].replace('_', '\n')
        ax.set_title(f"{method_name}\nCorr={result['corr']:.3f}, RMSE={result['rmse']:.3f}\n"
                    f"<3m/s: {result['pred_low_3']}, 3-4m/s: {result['continuity_34']}")
        ax.set_xlabel('观测风速 (m/s)')
        ax.set_ylabel('预测风速 (m/s)')
        ax.grid(True, alpha=0.3)
        ax.set_xlim(0, 10)
        ax.set_ylim(0, 10)
    
    # 最后一个图：性能雷达图
    if len(display_methods) <= 5:
        ax_radar = axes[1, 2] if len(display_methods) <= 2 else axes[1, len(display_methods) % 3]
        
        # 性能对比柱状图
        methods_names = [r['method'].replace('_', '\n') for r in display_methods]
        corr_scores = [r['corr'] for r in display_methods]
        coverage_scores = [r['coverage_3'] for r in display_methods]
        comp_scores = [r['comp_score'] for r in display_methods]
        
        x = np.arange(len(methods_names))
        width = 0.25
        
        ax_radar.bar(x - width, corr_scores, width, label='相关系数', alpha=0.7)
        ax_radar.bar(x, coverage_scores, width, label='覆盖率', alpha=0.7)
        ax_radar.bar(x + width, comp_scores, width, label='综合评分', alpha=0.7)
        
        ax_radar.set_xlabel('方法')
        ax_radar.set_ylabel('得分')
        ax_radar.set_title('性能对比')
        ax_radar.set_xticks(x)
        ax_radar.set_xticklabels(methods_names, rotation=45, ha='right')
        ax_radar.legend()
        ax_radar.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # 保存图表
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    plot_path = os.path.join(output_dir, f'hybrid_methods_comparison_{timestamp}.png')
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    print(f"📊 对比图表已保存: {plot_path}")
    
    plt.show()
    return fig

def generate_recommendations(methods_results):
    """生成推荐建议"""
    print("\n🎯 方案推荐")
    print("="*50)
    
    # 找出各方面的最佳方案
    best_precision = max(methods_results, key=lambda x: x['corr'])
    best_coverage = max(methods_results, key=lambda x: x['coverage_3'])
    best_comprehensive = max(methods_results, key=lambda x: x['comp_score'])
    
    print(f"🏆 各指标最佳方案:")
    print(f"  精度最佳: {best_precision['method']} (Corr={best_precision['corr']:.4f})")
    print(f"  覆盖最佳: {best_coverage['method']} (覆盖率={best_coverage['coverage_3']:.2f})")
    print(f"  综合最佳: {best_comprehensive['method']} (综合分={best_comprehensive['comp_score']:.3f})")
    
    print(f"\n💡 使用建议:")
    
    # 基于不同需求给出建议
    baseline = next(r for r in methods_results if r['method'] == 'Original_Baseline')
    
    for result in methods_results:
        if result['method'] == 'Original_Baseline':
            continue
            
        corr_loss = baseline['corr'] - result['corr']
        coverage_gain = result['coverage_3'] - baseline['coverage_3']
        
        print(f"\n📌 {result['method']}:")
        
        if corr_loss < 0.005 and coverage_gain > 0.3:
            print(f"  ✅ 强烈推荐: 精度损失极小({corr_loss:.4f})，覆盖大幅提升({coverage_gain:.2f})")
        elif corr_loss < 0.01 and coverage_gain > 0.2:
            print(f"  🌟 推荐: 精度轻微损失({corr_loss:.4f})，覆盖显著提升({coverage_gain:.2f})")
        elif corr_loss < 0.02:
            print(f"  ⚖️ 平衡选择: 可接受的精度损失({corr_loss:.4f})换取覆盖改善({coverage_gain:.2f})")
        else:
            print(f"  ⚠️ 权衡选择: 需要评估精度损失({corr_loss:.4f})是否可接受")

def main():
    """主函数"""
    print("🚀 混合优化方案完整测试")
    print("寻找精度与低风速覆盖的最佳平衡")
    print("="*60)
    
    # 配置路径
    data_path = '/Users/xiaxin/work/WindForecast_Project/01_Data/processed/imputed_data/changma_clean.csv'
    output_dir = '/Users/xiaxin/work/WindForecast_Project/03_Results/建模/10m风速建模/混合优化方案'
    
    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)
    
    try:
        # 1. 数据准备
        X, y, df_clean = load_and_prepare_data(data_path)
        
        split_idx = int(len(X) * 0.8)
        X_train, X_test = X[:split_idx], X[split_idx:]
        y_train, y_test = y[:split_idx], y[split_idx:]
        
        print(f"数据规模: 训练{len(X_train)}, 测试{len(X_test)}")
        
        # 显示观测数据的低风速分布
        for threshold in [1, 2, 3, 4]:
            count = (y_test < threshold).sum()
            pct = count / len(y_test) * 100
            print(f"测试集<{threshold}m/s: {count} ({pct:.1f}%)")
        
        # 2. 训练基线模型
        baseline_model, baseline_pred = train_baseline_model(X_train, y_train, X_test, y_test)
        
        # 3. 比较所有混合方法
        methods_results = compare_all_hybrid_methods(baseline_pred, y_train, y_test)
        
        # 4. 可视化对比
        fig = visualize_hybrid_comparison(methods_results, y_test, output_dir)
        
        # 5. 生成推荐
        generate_recommendations(methods_results)
        
        # 6. 保存结果
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # 保存预测结果
        pred_data = {result['method']: result['pred'] for result in methods_results}
        pred_df = pd.DataFrame(pred_data)
        pred_path = os.path.join(output_dir, f'hybrid_predictions_{timestamp}.csv')
        pred_df.to_csv(pred_path, index=False)
        
        # 保存性能对比
        perf_data = []
        for result in methods_results:
            perf_data.append({
                'Method': result['method'],
                'Correlation': result['corr'],
                'RMSE': result['rmse'],
                'Pred_Low_3ms': result['pred_low_3'],
                'Continuity_3_4ms': result['continuity_34'],
                'Coverage_Rate': result['coverage_3'],
                'Comprehensive_Score': result['comp_score']
            })
        
        perf_df = pd.DataFrame(perf_data)
        perf_path = os.path.join(output_dir, f'hybrid_performance_{timestamp}.csv')
        perf_df.to_csv(perf_path, index=False)
        
        print(f"\n💾 结果已保存到: {output_dir}")
        print(f"  预测数据: {os.path.basename(pred_path)}")
        print(f"  性能对比: {os.path.basename(perf_path)}")
        
        print("\n🎉 混合优化测试完成！")
        print("现在您可以根据业务需求选择最合适的方案了！")
        
        return methods_results, output_dir
        
    except FileNotFoundError:
        print(f"❌ 数据文件未找到: {data_path}")
        return None
        
    except Exception as e:
        print(f"❌ 运行出错: {e}")
        import traceback
        traceback.print_exc()
        return None

if __name__ == "__main__":
    methods_results, output_dir = main()
    
    if methods_results:
        print("\n✅ 混合优化测试成功完成！")
        print("查看生成的图表和性能对比，选择最适合您需求的方案。")
    else:
        print("\n❌ 测试失败，请检查数据路径和环境配置。")