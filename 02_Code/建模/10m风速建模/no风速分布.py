"""
Wind Speed Distribution Analysis
分析风速分布和模型预测特性
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

def analyze_wind_speed_distribution(data_path):
    """分析风速分布"""
    print("📊 分析风速分布...")
    
    # 加载数据
    df = pd.read_csv(data_path)
    
    # 基础统计
    obs_stats = df['obs_wind_speed_10m'].describe()
    ec_stats = df['ec_wind_speed_10m'].describe()
    gfs_stats = df['gfs_wind_speed_10m'].describe()
    
    print("🔍 风速基础统计:")
    print("=" * 60)
    print(f"{'统计量':<12} {'观测':<10} {'EC':<10} {'GFS':<10}")
    print("-" * 60)
    for stat in ['count', 'mean', 'std', 'min', '25%', '50%', '75%', 'max']:
        print(f"{stat:<12} {obs_stats[stat]:<10.2f} {ec_stats[stat]:<10.2f} {gfs_stats[stat]:<10.2f}")
    
    # 低风速占比分析
    thresholds = [1, 2, 3, 4, 5]
    print(f"\n🌪️ 低风速样本占比:")
    print("=" * 50)
    print(f"{'阈值(m/s)':<12} {'观测%':<10} {'EC%':<10} {'GFS%':<10}")
    print("-" * 50)
    
    for threshold in thresholds:
        obs_pct = (df['obs_wind_speed_10m'] < threshold).mean() * 100
        ec_pct = (df['ec_wind_speed_10m'] < threshold).mean() * 100
        gfs_pct = (df['gfs_wind_speed_10m'] < threshold).mean() * 100
        print(f"<{threshold}m/s       {obs_pct:<10.1f} {ec_pct:<10.1f} {gfs_pct:<10.1f}")
    
    # 创建分布图
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    # 1. 整体分布直方图
    ax1 = axes[0, 0]
    bins = np.arange(0, 16, 0.5)
    ax1.hist(df['obs_wind_speed_10m'], bins=bins, alpha=0.6, label='观测', color='black', density=True)
    ax1.hist(df['ec_wind_speed_10m'], bins=bins, alpha=0.6, label='EC', color='orange', density=True)
    ax1.hist(df['gfs_wind_speed_10m'], bins=bins, alpha=0.6, label='GFS', color='blue', density=True)
    ax1.set_xlabel('风速 (m/s)')
    ax1.set_ylabel('密度')
    ax1.set_title('风速分布对比')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.axvline(4, color='red', linestyle='--', linewidth=2, label='4m/s阈值')
    
    # 2. 低风速区域放大
    ax2 = axes[0, 1]
    low_wind_mask = (df['obs_wind_speed_10m'] < 8) & (df['ec_wind_speed_10m'] < 8) & (df['gfs_wind_speed_10m'] < 8)
    low_wind_data = df[low_wind_mask]
    
    bins_low = np.arange(0, 8, 0.2)
    ax2.hist(low_wind_data['obs_wind_speed_10m'], bins=bins_low, alpha=0.6, label='观测', color='black', density=True)
    ax2.hist(low_wind_data['ec_wind_speed_10m'], bins=bins_low, alpha=0.6, label='EC', color='orange', density=True)
    ax2.hist(low_wind_data['gfs_wind_speed_10m'], bins=bins_low, alpha=0.6, label='GFS', color='blue', density=True)
    ax2.set_xlabel('风速 (m/s)')
    ax2.set_ylabel('密度')
    ax2.set_title('低风速区域分布 (<8m/s)')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    ax2.axvline(4, color='red', linestyle='--', linewidth=2, label='4m/s阈值')
    
    # 3. 散点图：观测 vs EC (关注低风速)
    ax3 = axes[1, 0]
    sample_size = min(5000, len(df))
    sample_idx = np.random.choice(len(df), sample_size, replace=False)
    
    obs_sample = df['obs_wind_speed_10m'].iloc[sample_idx]
    ec_sample = df['ec_wind_speed_10m'].iloc[sample_idx]
    
    ax3.scatter(obs_sample, ec_sample, alpha=0.3, s=1)
    ax3.plot([0, 15], [0, 15], 'r--', linewidth=2, label='完美预报')
    ax3.axhline(4, color='red', linestyle=':', alpha=0.7, label='4m/s阈值')
    ax3.axvline(4, color='red', linestyle=':', alpha=0.7)
    ax3.set_xlabel('观测风速 (m/s)')
    ax3.set_ylabel('EC预报风速 (m/s)')
    ax3.set_title('观测 vs EC预报')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    ax3.set_xlim(0, 15)
    ax3.set_ylim(0, 15)
    
    # 4. EC vs GFS 低风速对比
    ax4 = axes[1, 1]
    ax4.scatter(ec_sample, df['gfs_wind_speed_10m'].iloc[sample_idx], alpha=0.3, s=1, color='green')
    ax4.plot([0, 15], [0, 15], 'r--', linewidth=2, label='EC=GFS')
    ax4.axhline(4, color='red', linestyle=':', alpha=0.7, label='4m/s阈值')
    ax4.axvline(4, color='red', linestyle=':', alpha=0.7)
    ax4.set_xlabel('EC预报风速 (m/s)')
    ax4.set_ylabel('GFS预报风速 (m/s)')
    ax4.set_title('EC vs GFS预报对比')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    ax4.set_xlim(0, 15)
    ax4.set_ylim(0, 15)
    
    plt.tight_layout()
    plt.show()
    
    return df

def analyze_model_prediction_bias(y_test, y_pred, ec_baseline, gfs_baseline):
    """分析模型预测偏差"""
    print("\n🎯 分析模型预测偏差...")
    
    # 按风速区间分析
    wind_ranges = [(0, 2), (2, 4), (4, 6), (6, 8), (8, 10), (10, 15)]
    
    print("📊 分风速区间的预测性能:")
    print("=" * 80)
    print(f"{'风速区间':<12} {'样本数':<8} {'观测均值':<10} {'EC均值':<10} {'GFS均值':<10} {'LGB均值':<10}")
    print("-" * 80)
    
    for low, high in wind_ranges:
        mask = (y_test >= low) & (y_test < high)
        
        if mask.sum() > 0:
            obs_mean = y_test[mask].mean()
            ec_mean = ec_baseline[mask].mean()
            gfs_mean = gfs_baseline[mask].mean()
            lgb_mean = y_pred[mask].mean()
            
            print(f"{low}-{high}m/s     {mask.sum():<8} {obs_mean:<10.2f} {ec_mean:<10.2f} {gfs_mean:<10.2f} {lgb_mean:<10.2f}")
    
    # 分析预测下限
    print(f"\n🔍 预测统计:")
    print(f"观测最小值: {y_test.min():.2f} m/s")
    print(f"EC预测最小值: {ec_baseline.min():.2f} m/s")
    print(f"GFS预测最小值: {gfs_baseline.min():.2f} m/s")
    print(f"LightGBM预测最小值: {y_pred.min():.2f} m/s")
    
    print(f"\n观测中<4m/s的样本: {(y_test < 4).sum()} ({(y_test < 4).mean()*100:.1f}%)")
    print(f"LightGBM预测中<4m/s的样本: {(y_pred < 4).sum()} ({(y_pred < 4).mean()*100:.1f}%)")

# 主函数
def main():
    data_path = '/Users/xiaxin/work/WindForecast_Project/01_Data/processed/imputed_data/changma_clean.csv'
    
    # 分析原始数据分布
    df = analyze_wind_speed_distribution(data_path)
    
    print("\n" + "="*60)
    print("💡 可能的解决方案:")
    print("="*60)
    print("1. 📈 数据增强: 增加低风速样本的权重")
    print("2. 🎯 分层建模: 低风速和高风速分别建模")
    print("3. 🔧 特征工程: 添加能表达静风条件的特征")
    print("4. ⚖️ 损失函数: 使用对低风速更敏感的损失函数")
    print("5. 📊 后处理: 对预测结果进行分布匹配校正")

if __name__ == "__main__":
    main()