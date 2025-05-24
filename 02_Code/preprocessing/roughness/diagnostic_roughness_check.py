#!/usr/bin/env python3
"""
粗糙度计算诊断检查 - 验证Wang方法的结果
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

def diagnostic_check(detailed_results_path, output_dir):
    """诊断检查Wang方法的计算结果"""
    
    print("=== 粗糙度计算诊断检查 ===")
    
    # 读取详细结果
    df = pd.read_csv(detailed_results_path)
    print(f"总样本数: {len(df)}")
    
    # 1. 基本统计
    print("\n1. 粗糙度基本统计:")
    print(f"   唯一值数量: {df['z0'].nunique()}")
    print(f"   最小值: {df['z0'].min():.6f} m")
    print(f"   最大值: {df['z0'].max():.6f} m")
    print(f"   中位数: {df['z0'].median():.6f} m")
    print(f"   平均值: {df['z0'].mean():.6f} m")
    print(f"   标准差: {df['z0'].std():.6f} m")
    
    # 2. 检查z0值的分布
    print("\n2. z0值分布检查:")
    value_counts = df['z0'].value_counts()
    print(f"   最常见的5个值:")
    for val, count in value_counts.head().items():
        print(f"   {val:.6f} m: {count}次 ({count/len(df)*100:.1f}%)")
    
    # 3. 检查u*一致性
    print("\n3. u*一致性检查:")
    print(f"   u*相对标准差 < 1%: {len(df[df['u_star_std'] < 0.01])} ({len(df[df['u_star_std'] < 0.01])/len(df)*100:.1f}%)")
    print(f"   u*相对标准差 < 3%: {len(df[df['u_star_std'] < 0.03])} ({len(df[df['u_star_std'] < 0.03])/len(df)*100:.1f}%)")
    print(f"   u*相对标准差 < 5%: {len(df[df['u_star_std'] < 0.05])} ({len(df[df['u_star_std'] < 0.05])/len(df)*100:.1f}%)")
    
    # 4. 创建诊断图
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    # 4.1 z0的对数分布
    ax = axes[0, 0]
    ln_z0 = np.log(df['z0'])
    ax.hist(ln_z0, bins=50, alpha=0.7, color='blue', edgecolor='black')
    ax.set_xlabel('ln(z₀)', fontsize=12)
    ax.set_ylabel('Count', fontsize=12)
    ax.set_title('ln(z₀) Distribution', fontsize=14)
    ax.axvline(np.median(ln_z0), color='red', linestyle='--', label=f'Median: {np.median(ln_z0):.2f}')
    ax.legend()
    
    # 4.2 z0 vs u*
    ax = axes[0, 1]
    ax.scatter(df['u_star'], df['z0'], alpha=0.5, s=20)
    ax.set_xlabel('u* (m/s)', fontsize=12)
    ax.set_ylabel('z₀ (m)', fontsize=12)
    ax.set_title('z₀ vs Friction Velocity', fontsize=14)
    ax.set_yscale('log')
    
    # 4.3 z0 vs R²
    ax = axes[1, 0]
    ax.scatter(df['r_squared'], df['z0'], alpha=0.5, s=20)
    ax.set_xlabel('R²', fontsize=12)
    ax.set_ylabel('z₀ (m)', fontsize=12)
    ax.set_title('z₀ vs Fitting Quality', fontsize=14)
    ax.set_yscale('log')
    
    # 4.4 零平面位移高度检查
    ax = axes[1, 1]
    d_values = df['d']
    ax.hist(d_values, bins=50, alpha=0.7, color='green', edgecolor='black')
    ax.set_xlabel('d (m)', fontsize=12)
    ax.set_ylabel('Count', fontsize=12)
    ax.set_title('Zero-plane Displacement Height Distribution', fontsize=14)
    ax.axvline(np.median(d_values), color='red', linestyle='--', 
               label=f'Median d: {np.median(d_values):.4f} m')
    ax.legend()
    
    plt.tight_layout()
    plt.savefig(f"{output_dir}/diagnostic_plots.png", dpi=300)
    plt.close()
    
    # 5. 检查可能的问题
    print("\n4. 潜在问题诊断:")
    
    # 检查是否有过多相同值
    most_common_value = value_counts.iloc[0]
    if most_common_value > len(df) * 0.5:
        print(f"   ⚠️ 警告: 超过50%的样本有相同的z0值！")
    
    # 检查d值是否合理
    median_d = df['d'].median()
    if median_d < 0.001:
        print(f"   ⚠️ 警告: 零平面位移高度过小 (中位数={median_d:.4f} m)")
    
    # 检查RMSE
    median_rmse = df['rmse'].median()
    if median_rmse > 5:
        print(f"   ⚠️ 警告: 拟合误差较大 (中位数RMSE={median_rmse:.2f} m/s)")
    
    # 6. 建议使用传统方法重新计算几个样本进行对比
    print("\n5. 建议验证步骤:")
    print("   1) 随机选择几个风速廓线，用传统线性回归方法计算z0")
    print("   2) 检查是否考虑零平面位移高度导致了问题")
    print("   3) 尝试放宽u*一致性要求到10%")
    print("   4) 检查优化算法是否收敛到合理值")
    
    return df

def compare_with_traditional_method(df, sample_size=5):
    """用传统方法重新计算几个样本"""
    print("\n=== 传统方法验证 ===")
    
    # 随机选择样本
    samples = df.sample(n=min(sample_size, len(df)))
    
    for idx, row in samples.iterrows():
        print(f"\n样本 {idx}:")
        print(f"Wang方法 z0: {row['z0']:.6f} m")
        
        # 重构风速廓线
        winds = np.array(eval(row['wind_profile']))
        heights = np.array(eval(row['height_profile']))
        
        # 传统方法：直接对ln(z)和u进行线性回归
        ln_z = np.log(heights)
        slope, intercept, r_value, _, _ = stats.linregress(ln_z, winds)
        
        # 计算z0
        z0_traditional = np.exp(-intercept / slope)
        u_star_traditional = slope * 0.4
        
        print(f"传统方法 z0: {z0_traditional:.6f} m")
        print(f"比值: {row['z0'] / z0_traditional:.2f}")
        print(f"u* (Wang): {row['u_star']:.3f} m/s")
        print(f"u* (传统): {u_star_traditional:.3f} m/s")

# 主函数
if __name__ == "__main__":
    import os
    
    # 设置路径
    results_dir = "/Users/xiaxin/work/WindForecast_Project/03_Results/wang_method_roughness_analysis"
    detailed_results_path = os.path.join(results_dir, "wang_method_detailed_results.csv")
    
    # 运行诊断
    df = diagnostic_check(detailed_results_path, results_dir)
    
    # 传统方法对比
    compare_with_traditional_method(df, sample_size=10)
    
    print("\n诊断完成！")