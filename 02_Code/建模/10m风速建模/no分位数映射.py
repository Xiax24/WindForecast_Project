"""
Distribution Matching 分布匹配方法详解
通过分位数映射让预测分布匹配观测分布
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats
from scipy.interpolate import interp1d
import seaborn as sns

def distribution_matching_detailed(y_test, y_pred_original, y_train, method='quantile'):
    """
    分布匹配的详细实现和解释
    
    Parameters:
    -----------
    y_test : array
        测试集观测值
    y_pred_original : array  
        原始预测值
    y_train : array
        训练集观测值（用于构建目标分布）
    method : str
        方法选择：'quantile', 'cdf', 'histogram'
    """
    
    print("🎯 分布匹配方法详解")
    print("=" * 50)
    
    if method == 'quantile':
        return quantile_mapping(y_test, y_pred_original, y_train)
    elif method == 'cdf':
        return cdf_mapping(y_test, y_pred_original, y_train)
    elif method == 'histogram':
        return histogram_matching(y_test, y_pred_original, y_train)

def quantile_mapping(y_test, y_pred_original, y_train):
    """
    方法1: 分位数映射 (Quantile Mapping)
    这是昨天代码中使用的方法
    """
    print("\n📊 方法1: 分位数映射")
    print("-" * 30)
    
    # 步骤1: 计算观测数据的分位数
    print("步骤1: 计算目标分布（观测数据）的分位数")
    percentiles = np.arange(0, 101, 1)  # 0%, 1%, 2%, ..., 100%
    obs_quantiles = np.percentile(y_train, percentiles)
    
    print(f"  观测数据范围: {y_train.min():.2f} - {y_train.max():.2f}")
    print(f"  分位数数量: {len(obs_quantiles)}")
    
    # 步骤2: 对每个预测值进行映射
    print("\n步骤2: 预测值 → 分位数 → 观测分布对应值")
    y_pred_corrected = np.zeros_like(y_pred_original)
    
    mapping_examples = []
    
    for i, pred in enumerate(y_pred_original):
        # 2.1: 计算预测值在预测分布中的分位数位置
        pred_percentile = stats.percentileofscore(y_pred_original, pred)
        
        # 2.2: 映射到观测分布的对应分位数值
        if pred_percentile <= 0:
            corrected_value = obs_quantiles[0]
        elif pred_percentile >= 100:
            corrected_value = obs_quantiles[100]
        else:
            # 线性插值
            lower_idx = int(pred_percentile)
            upper_idx = min(lower_idx + 1, 100)
            weight = pred_percentile - lower_idx
            
            corrected_value = (obs_quantiles[lower_idx] * (1 - weight) + 
                             obs_quantiles[upper_idx] * weight)
        
        y_pred_corrected[i] = corrected_value
        
        # 收集一些示例用于说明
        if i < 5:
            mapping_examples.append({
                'original_pred': pred,
                'percentile': pred_percentile,
                'corrected_pred': corrected_value
            })
    
    print("  映射示例:")
    for ex in mapping_examples:
        print(f"    {ex['original_pred']:.2f} → {ex['percentile']:.1f}% → {ex['corrected_pred']:.2f}")
    
    print(f"\n结果: 原始预测范围 {y_pred_original.min():.2f}-{y_pred_original.max():.2f}")
    print(f"      校正后范围   {y_pred_corrected.min():.2f}-{y_pred_corrected.max():.2f}")
    print(f"      观测范围     {y_test.min():.2f}-{y_test.max():.2f}")
    
    return y_pred_corrected

def cdf_mapping(y_test, y_pred_original, y_train):
    """
    方法2: 累积分布函数映射 (CDF Mapping)
    更平滑的分布匹配方法
    """
    print("\n📊 方法2: CDF映射")
    print("-" * 30)
    
    # 构建观测数据的经验CDF
    print("步骤1: 构建目标分布的经验CDF")
    obs_sorted = np.sort(y_train)
    obs_cdf = np.arange(1, len(obs_sorted) + 1) / len(obs_sorted)
    
    # 构建预测数据的经验CDF
    pred_sorted = np.sort(y_pred_original)
    pred_cdf = np.arange(1, len(pred_sorted) + 1) / len(pred_sorted)
    
    # 创建插值函数
    print("步骤2: 创建CDF插值函数")
    # 预测值 → CDF值的映射
    pred_to_cdf = interp1d(pred_sorted, pred_cdf, 
                          bounds_error=False, fill_value=(0, 1))
    
    # CDF值 → 观测值的映射
    cdf_to_obs = interp1d(obs_cdf, obs_sorted, 
                         bounds_error=False, 
                         fill_value=(obs_sorted[0], obs_sorted[-1]))
    
    # 执行映射
    print("步骤3: 执行CDF映射")
    pred_cdf_values = pred_to_cdf(y_pred_original)
    y_pred_corrected = cdf_to_obs(pred_cdf_values)
    
    print(f"CDF映射完成，处理了 {len(y_pred_original)} 个预测值")
    
    return y_pred_corrected

def histogram_matching(y_test, y_pred_original, y_train):
    """
    方法3: 直方图匹配 (Histogram Matching)
    基于直方图的分布校正
    """
    print("\n📊 方法3: 直方图匹配")
    print("-" * 30)
    
    # 定义统一的分箱
    bins = np.linspace(min(y_train.min(), y_pred_original.min()), 
                      max(y_train.max(), y_pred_original.max()), 50)
    
    # 计算观测和预测的直方图
    obs_hist, _ = np.histogram(y_train, bins=bins, density=True)
    pred_hist, _ = np.histogram(y_pred_original, bins=bins, density=True)
    
    # 计算校正因子
    correction_factors = np.divide(obs_hist, pred_hist, 
                                  out=np.ones_like(obs_hist), 
                                  where=pred_hist!=0)
    
    # 应用校正
    y_pred_corrected = y_pred_original.copy()
    bin_indices = np.digitize(y_pred_original, bins) - 1
    bin_indices = np.clip(bin_indices, 0, len(correction_factors) - 1)
    
    # 这里简化为调整预测值的方差
    pred_mean = np.mean(y_pred_original)
    pred_std = np.std(y_pred_original)
    obs_std = np.std(y_train)
    
    # 调整方差匹配观测数据
    y_pred_corrected = pred_mean + (y_pred_original - pred_mean) * (obs_std / pred_std)
    
    print(f"直方图匹配完成")
    print(f"  预测数据标准差: {pred_std:.3f} → {np.std(y_pred_corrected):.3f}")
    print(f"  观测数据标准差: {obs_std:.3f}")
    
    return y_pred_corrected

def visualize_distribution_matching(y_test, y_pred_original, y_train, methods=['quantile', 'cdf']):
    """可视化分布匹配的效果"""
    
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    
    # 应用不同方法
    results = {}
    for method in methods:
        results[method] = distribution_matching_detailed(y_test, y_pred_original, y_train, method)
    
    # 1. 原始分布对比
    ax1 = axes[0, 0]
    ax1.hist(y_train, bins=30, alpha=0.7, density=True, label='观测(训练)', color='black')
    ax1.hist(y_pred_original, bins=30, alpha=0.7, density=True, label='原始预测', color='blue')
    ax1.set_xlabel('风速 (m/s)')
    ax1.set_ylabel('密度')
    ax1.set_title('原始分布对比')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # 2. 校正后分布对比
    ax2 = axes[0, 1]
    ax2.hist(y_test, bins=30, alpha=0.7, density=True, label='观测(测试)', color='black')
    colors = ['red', 'green', 'orange']
    for i, (method, corrected) in enumerate(results.items()):
        ax2.hist(corrected, bins=30, alpha=0.6, density=True, 
                label=f'{method}校正', color=colors[i])
    ax2.set_xlabel('风速 (m/s)')
    ax2.set_ylabel('密度')
    ax2.set_title('校正后分布对比')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # 3. Q-Q图
    ax3 = axes[0, 2]
    
    # 计算分位数
    quantiles = np.linspace(0.01, 0.99, 50)
    obs_quantiles = np.percentile(y_test, quantiles * 100)
    pred_quantiles = np.percentile(y_pred_original, quantiles * 100)
    
    ax3.scatter(obs_quantiles, pred_quantiles, alpha=0.6, label='原始预测', color='blue')
    
    for i, (method, corrected) in enumerate(results.items()):
        corr_quantiles = np.percentile(corrected, quantiles * 100)
        ax3.scatter(obs_quantiles, corr_quantiles, alpha=0.6, 
                   label=f'{method}校正', color=colors[i], s=20)
    
    ax3.plot([obs_quantiles.min(), obs_quantiles.max()], 
            [obs_quantiles.min(), obs_quantiles.max()], 'k--', label='理想线')
    ax3.set_xlabel('观测分位数')
    ax3.set_ylabel('预测分位数')
    ax3.set_title('Q-Q图对比')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # 4. 散点图对比 - 原始
    ax4 = axes[1, 0]
    ax4.scatter(y_test, y_pred_original, alpha=0.5, s=2, color='blue')
    ax4.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')
    ax4.set_xlabel('观测风速')
    ax4.set_ylabel('原始预测风速')
    ax4.set_title('原始预测散点图')
    ax4.grid(True, alpha=0.3)
    
    # 5. 散点图对比 - 校正后
    ax5 = axes[1, 1]
    for i, (method, corrected) in enumerate(results.items()):
        ax5.scatter(y_test, corrected, alpha=0.6, s=2, 
                   label=f'{method}校正', color=colors[i])
    ax5.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'k--')
    ax5.set_xlabel('观测风速')
    ax5.set_ylabel('校正后预测风速')
    ax5.set_title('校正后预测散点图')
    ax5.legend()
    ax5.grid(True, alpha=0.3)
    
    # 6. 3-4m/s区间特别关注
    ax6 = axes[1, 2]
    target_mask = (y_test >= 3) & (y_test < 4)
    
    if target_mask.sum() > 0:
        ax6.scatter(y_test[target_mask], y_pred_original[target_mask], 
                   alpha=0.8, s=30, label='原始预测', color='blue')
        
        for i, (method, corrected) in enumerate(results.items()):
            ax6.scatter(y_test[target_mask], corrected[target_mask], 
                       alpha=0.8, s=30, label=f'{method}校正', color=colors[i])
        
        ax6.plot([3, 4], [3, 4], 'k--')
        ax6.set_xlim(2.8, 4.2)
        ax6.set_ylim(2.8, 4.2)
        ax6.set_xlabel('观测风速')
        ax6.set_ylabel('预测风速')
        ax6.set_title('3-4m/s区间对比')
        ax6.legend()
        ax6.grid(True, alpha=0.3)
    else:
        ax6.text(0.5, 0.5, '3-4m/s区间无数据', ha='center', va='center', transform=ax6.transAxes)
    
    plt.tight_layout()
    plt.show()
    
    return results

def explain_distribution_matching_theory():
    """解释分布匹配的理论基础"""
    
    print("🎓 分布匹配理论基础")
    print("=" * 50)
    
    print("\n1. 核心假设:")
    print("   ✓ 模型能够学习正确的相对关系（排序）")
    print("   ✓ 但预测分布的形状可能与观测不同")
    print("   ✓ 通过重新映射可以校正分布偏差")
    
    print("\n2. 数学原理:")
    print("   设 F_obs 为观测数据的CDF，F_pred 为预测数据的CDF")
    print("   对于预测值 y_pred，其分位数为: p = F_pred(y_pred)")
    print("   校正值为: y_corrected = F_obs^(-1)(p)")
    
    print("\n3. 适用场景:")
    print("   ✓ 预测趋势正确但分布偏移")
    print("   ✓ 系统性的偏差（如整体偏高或偏低）")
    print("   ✓ 分布形状差异（如方差不匹配）")
    
    print("\n4. 优势:")
    print("   ✓ 保持预测值的相对排序")
    print("   ✓ 自动匹配目标分布的统计特性")
    print("   ✓ 不需要额外的特征工程")
    
    print("\n5. 局限性:")
    print("   ⚠ 假设训练集分布代表真实分布")
    print("   ⚠ 可能过度依赖历史数据")
    print("   ⚠ 对极值的处理可能不够稳健")

def demo_distribution_matching():
    """演示分布匹配的效果"""
    
    # 创建模拟数据
    np.random.seed(42)
    
    # 模拟观测数据（训练集）
    y_train = np.random.gamma(2, 2, 1000)  # 伽马分布
    
    # 模拟测试集观测
    y_test = np.random.gamma(2, 2, 300)
    
    # 模拟有偏差的预测（正态分布，均值偏高）
    y_pred_original = np.random.normal(y_test.mean() + 1, y_test.std() * 0.8, len(y_test))
    y_pred_original = np.maximum(y_pred_original, 0)  # 确保非负
    
    print("🧪 分布匹配演示")
    print("=" * 50)
    print(f"观测数据统计:")
    print(f"  训练集: 均值={y_train.mean():.2f}, 标准差={y_train.std():.2f}")
    print(f"  测试集: 均值={y_test.mean():.2f}, 标准差={y_test.std():.2f}")
    print(f"原始预测统计:")
    print(f"  均值={y_pred_original.mean():.2f}, 标准差={y_pred_original.std():.2f}")
    
    # 应用分布匹配
    results = visualize_distribution_matching(y_test, y_pred_original, y_train)
    
    # 评估改善效果
    print(f"\n📊 改善效果:")
    from scipy.stats import pearsonr
    from sklearn.metrics import mean_squared_error
    
    orig_corr, _ = pearsonr(y_test, y_pred_original)
    orig_rmse = np.sqrt(mean_squared_error(y_test, y_pred_original))
    
    print(f"原始预测: 相关系数={orig_corr:.4f}, RMSE={orig_rmse:.4f}")
    
    for method, corrected in results.items():
        corr, _ = pearsonr(y_test, corrected)
        rmse = np.sqrt(mean_squared_error(y_test, corrected))
        print(f"{method}校正: 相关系数={corr:.4f}, RMSE={rmse:.4f}")

if __name__ == "__main__":
    # 运行演示
    explain_distribution_matching_theory()
    demo_distribution_matching()