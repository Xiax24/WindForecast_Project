import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
from datetime import datetime, timedelta

# 设置绘图样式和配色
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['figure.figsize'] = (16, 10)
sns.set_style("whitegrid")

# 使用专业的配色方案
colors = {
    'obs': '#2C3E50',      # 深蓝灰 - 观测值
    'ec': '#E74C3C',       # 红色 - EC模型
    'gfs': '#3498DB',      # 蓝色 - GFS模型
    'fill_obs': '#34495E', # 观测值填充
    'fill_ec': '#EC7063',  # EC填充
    'fill_gfs': '#5DADE2'  # GFS填充
}

# 数据和结果路径
data_path = '/Users/xiaxin/work/WindForecast_Project/01_Data/processed/imputed_data/changma_imputed_complete.csv'
results_dir = '/Users/xiaxin/work/WindForecast_Project/03_Results/6diurnal_sequence/'

# 创建结果目录
os.makedirs(results_dir, exist_ok=True)

def load_and_prepare_data():
    """读取和预处理数据"""
    print("Loading wind speed data...")
    df = pd.read_csv(data_path)
    
    # 转换时间列
    df['datetime'] = pd.to_datetime(df['datetime'])
    df['hour'] = df['datetime'].dt.hour
    df['date'] = df['datetime'].dt.date
    df['month'] = df['datetime'].dt.month
    df['season'] = df['month'].map({
        12: 'Winter', 1: 'Winter', 2: 'Winter',
        3: 'Spring', 4: 'Spring', 5: 'Spring',
        6: 'Summer', 7: 'Summer', 8: 'Summer',
        9: 'Autumn', 10: 'Autumn', 11: 'Autumn'
    })
    
    print(f"Data shape: {df.shape}")
    print(f"Time range: {df['datetime'].min()} to {df['datetime'].max()}")
    
    return df

def get_wind_speed_variables(df):
    """获取风速变量"""
    heights = ['10m', '30m', '50m', '70m']
    variables = []
    
    for height in heights:
        obs_col = f'obs_wind_speed_{height}'
        ec_col = f'ec_wind_speed_{height}'
        gfs_col = f'gfs_wind_speed_{height}'
        
        if all(col in df.columns for col in [obs_col, ec_col, gfs_col]):
            variables.append({
                'height': height,
                'obs': obs_col,
                'ec': ec_col,
                'gfs': gfs_col
            })
    
    return variables

def calculate_diurnal_statistics(df, variables):
    """计算昼夜循环统计量"""
    results = []
    
    for var in variables:
        for hour in range(24):
            hour_data = df[df['hour'] == hour]
            
            # 计算统计量
            obs_stats = {
                'mean': hour_data[var['obs']].mean(),
                'std': hour_data[var['obs']].std(),
                'median': hour_data[var['obs']].median(),
                'q25': hour_data[var['obs']].quantile(0.25),
                'q75': hour_data[var['obs']].quantile(0.75),
                'count': hour_data[var['obs']].count()
            }
            
            ec_stats = {
                'mean': hour_data[var['ec']].mean(),
                'std': hour_data[var['ec']].std(),
                'median': hour_data[var['ec']].median(),
                'q25': hour_data[var['ec']].quantile(0.25),
                'q75': hour_data[var['ec']].quantile(0.75),
                'count': hour_data[var['ec']].count()
            }
            
            gfs_stats = {
                'mean': hour_data[var['gfs']].mean(),
                'std': hour_data[var['gfs']].std(),
                'median': hour_data[var['gfs']].median(),
                'q25': hour_data[var['gfs']].quantile(0.25),
                'q75': hour_data[var['gfs']].quantile(0.75),
                'count': hour_data[var['gfs']].count()
            }
            
            results.append({
                'height': var['height'],
                'hour': hour,
                'obs_mean': obs_stats['mean'],
                'obs_std': obs_stats['std'],
                'obs_median': obs_stats['median'],
                'obs_q25': obs_stats['q25'],
                'obs_q75': obs_stats['q75'],
                'obs_count': obs_stats['count'],
                'ec_mean': ec_stats['mean'],
                'ec_std': ec_stats['std'],
                'ec_median': ec_stats['median'],
                'ec_q25': ec_stats['q25'],
                'ec_q75': ec_stats['q75'],
                'ec_count': ec_stats['count'],
                'gfs_mean': gfs_stats['mean'],
                'gfs_std': gfs_stats['std'],
                'gfs_median': gfs_stats['median'],
                'gfs_q25': gfs_stats['q25'],
                'gfs_q75': gfs_stats['q75'],
                'gfs_count': gfs_stats['count']
            })
    
    return pd.DataFrame(results)

def plot_diurnal_sequence_by_height(stats_df, height):
    """为每个高度绘制昼夜循环序列图"""
    height_data = stats_df[stats_df['height'] == height].sort_values('hour')
    
    if height_data.empty:
        print(f"No data for height {height}")
        return
    
    # 创建两个子图：均值±标准差 和 中位数+四分位数
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(16, 12))
    
    hours = height_data['hour'].values
    
    # === 上图：均值 ± 标准差 ===
    # 观测值
    obs_mean = height_data['obs_mean'].values
    obs_std = height_data['obs_std'].values
    
    ax1.plot(hours, obs_mean, color=colors['obs'], linewidth=3, 
             marker='o', markersize=6, label='Observed', alpha=0.9)
    ax1.fill_between(hours, obs_mean - obs_std, obs_mean + obs_std, 
                     color=colors['fill_obs'], alpha=0.2)
    
    # EC模型
    ec_mean = height_data['ec_mean'].values
    ec_std = height_data['ec_std'].values
    
    ax1.plot(hours, ec_mean, color=colors['ec'], linewidth=2.5, 
             marker='s', markersize=5, label='EC Model', alpha=0.8)
    ax1.fill_between(hours, ec_mean - ec_std, ec_mean + ec_std, 
                     color=colors['fill_ec'], alpha=0.15)
    
    # GFS模型
    gfs_mean = height_data['gfs_mean'].values
    gfs_std = height_data['gfs_std'].values
    
    ax1.plot(hours, gfs_mean, color=colors['gfs'], linewidth=2.5, 
             marker='^', markersize=5, label='GFS Model', alpha=0.8)
    ax1.fill_between(hours, gfs_mean - gfs_std, gfs_mean + gfs_std, 
                     color=colors['fill_gfs'], alpha=0.15)
    
    # 设置上图
    ax1.set_xlabel('Hour', fontsize=12)
    ax1.set_ylabel('Wind Speed (m/s)', fontsize=12)
    ax1.set_title(f'Wind Speed {height} - Diurnal Cycle (Mean ± Std)', 
                  fontsize=14, fontweight='bold')
    ax1.legend(fontsize=12, loc='best')
    ax1.grid(True, alpha=0.3)
    ax1.set_xticks(range(0, 24, 2))
    ax1.set_xlim(0, 23)
    
    # === 下图：中位数 + 四分位数范围 ===
    # 观测值
    obs_median = height_data['obs_median'].values
    obs_q25 = height_data['obs_q25'].values
    obs_q75 = height_data['obs_q75'].values
    
    ax2.plot(hours, obs_median, color=colors['obs'], linewidth=3, 
             marker='o', markersize=6, label='Observed (Median)', alpha=0.9)
    ax2.fill_between(hours, obs_q25, obs_q75, 
                     color=colors['fill_obs'], alpha=0.2, label='Observed IQR')
    
    # EC模型
    ec_median = height_data['ec_median'].values
    ec_q25 = height_data['ec_q25'].values
    ec_q75 = height_data['ec_q75'].values
    
    ax2.plot(hours, ec_median, color=colors['ec'], linewidth=2.5, 
             marker='s', markersize=5, label='EC Model (Median)', alpha=0.8)
    ax2.fill_between(hours, ec_q25, ec_q75, 
                     color=colors['fill_ec'], alpha=0.15, label='EC IQR')
    
    # GFS模型
    gfs_median = height_data['gfs_median'].values
    gfs_q25 = height_data['gfs_q25'].values
    gfs_q75 = height_data['gfs_q75'].values
    
    ax2.plot(hours, gfs_median, color=colors['gfs'], linewidth=2.5, 
             marker='^', markersize=5, label='GFS Model (Median)', alpha=0.8)
    ax2.fill_between(hours, gfs_q25, gfs_q75, 
                     color=colors['fill_gfs'], alpha=0.15, label='GFS IQR')
    
    # 设置下图
    ax2.set_xlabel('Hour', fontsize=12)
    ax2.set_ylabel('Wind Speed (m/s)', fontsize=12)
    ax2.set_title(f'Wind Speed {height} - Diurnal Cycle (Median with IQR)', 
                  fontsize=14, fontweight='bold')
    ax2.legend(fontsize=11, loc='best')
    ax2.grid(True, alpha=0.3)
    ax2.set_xticks(range(0, 24, 2))
    ax2.set_xlim(0, 23)
    
    # 统一y轴范围以便对比
    all_values = np.concatenate([
        obs_mean - obs_std, obs_mean + obs_std,
        ec_mean - ec_std, ec_mean + ec_std,
        gfs_mean - gfs_std, gfs_mean + gfs_std,
        obs_q25, obs_q75, ec_q25, ec_q75, gfs_q25, gfs_q75
    ])
    all_values = all_values[~np.isnan(all_values)]
    
    if len(all_values) > 0:
        y_min = max(0, min(all_values) * 0.95)  # 风速不能为负
        y_max = max(all_values) * 1.05
        ax1.set_ylim(y_min, y_max)
        ax2.set_ylim(y_min, y_max)
    
    plt.tight_layout()
    filename = f'wind_speed_{height}_diurnal_sequence.png'
    plt.savefig(os.path.join(results_dir, filename), dpi=300, bbox_inches='tight')
    plt.show()
    print(f"Saved: {filename}")

def plot_seasonal_comparison(df, variables):
    """绘制季节对比图"""
    seasons = ['Spring', 'Summer', 'Autumn', 'Winter']
    seasonal_colors = ['#27AE60', '#F39C12', '#D35400', '#3498DB']  # 绿、橙、红、蓝
    
    for var in variables:
        height = var['height']
        
        # 计算季节性统计
        seasonal_stats = []
        for season in seasons:
            season_data = df[df['season'] == season]
            
            for hour in range(24):
                hour_data = season_data[season_data['hour'] == hour]
                
                seasonal_stats.append({
                    'season': season,
                    'hour': hour,
                    'obs_mean': hour_data[var['obs']].mean(),
                    'ec_mean': hour_data[var['ec']].mean(),
                    'gfs_mean': hour_data[var['gfs']].mean()
                })
        
        seasonal_df = pd.DataFrame(seasonal_stats)
        
        # 绘制季节对比图
        fig, axes = plt.subplots(1, 3, figsize=(18, 6))
        
        # 观测值季节对比
        for i, season in enumerate(seasons):
            season_data = seasonal_df[seasonal_df['season'] == season].sort_values('hour')
            axes[0].plot(season_data['hour'], season_data['obs_mean'], 
                        color=seasonal_colors[i], linewidth=2.5, marker='o', 
                        markersize=4, label=season, alpha=0.8)
        
        axes[0].set_title(f'Observed Wind Speed {height} - Seasonal Comparison', 
                         fontsize=13, fontweight='bold')
        axes[0].set_xlabel('Hour', fontsize=12)
        axes[0].set_ylabel('Wind Speed (m/s)', fontsize=12)
        axes[0].legend(fontsize=10)
        axes[0].grid(True, alpha=0.3)
        axes[0].set_xlim(0, 23)
        axes[0].set_xticks(range(0, 24, 4))
        
        # EC模型季节对比
        for i, season in enumerate(seasons):
            season_data = seasonal_df[seasonal_df['season'] == season].sort_values('hour')
            axes[1].plot(season_data['hour'], season_data['ec_mean'], 
                        color=seasonal_colors[i], linewidth=2.5, marker='s', 
                        markersize=4, label=season, alpha=0.8)
        
        axes[1].set_title(f'EC Model Wind Speed {height} - Seasonal Comparison', 
                         fontsize=13, fontweight='bold')
        axes[1].set_xlabel('Hour', fontsize=12)
        axes[1].set_ylabel('Wind Speed (m/s)', fontsize=12)
        axes[1].legend(fontsize=10)
        axes[1].grid(True, alpha=0.3)
        axes[1].set_xlim(0, 23)
        axes[1].set_xticks(range(0, 24, 4))
        
        # GFS模型季节对比
        for i, season in enumerate(seasons):
            season_data = seasonal_df[seasonal_df['season'] == season].sort_values('hour')
            axes[2].plot(season_data['hour'], season_data['gfs_mean'], 
                        color=seasonal_colors[i], linewidth=2.5, marker='^', 
                        markersize=4, label=season, alpha=0.8)
        
        axes[2].set_title(f'GFS Model Wind Speed {height} - Seasonal Comparison', 
                         fontsize=13, fontweight='bold')
        axes[2].set_xlabel('Hour', fontsize=12)
        axes[2].set_ylabel('Wind Speed (m/s)', fontsize=12)
        axes[2].legend(fontsize=10)
        axes[2].grid(True, alpha=0.3)
        axes[2].set_xlim(0, 23)
        axes[2].set_xticks(range(0, 24, 4))
        
        # 统一y轴
        all_means = np.concatenate([
            seasonal_df['obs_mean'].dropna(),
            seasonal_df['ec_mean'].dropna(),
            seasonal_df['gfs_mean'].dropna()
        ])
        if len(all_means) > 0:
            y_min = max(0, min(all_means) * 0.95)
            y_max = max(all_means) * 1.05
            for ax in axes:
                ax.set_ylim(y_min, y_max)
        
        plt.tight_layout()
        filename = f'wind_speed_{height}_seasonal_comparison.png'
        plt.savefig(os.path.join(results_dir, filename), dpi=300, bbox_inches='tight')
        plt.show()
        print(f"Saved: {filename}")

def calculate_sequence_metrics(stats_df):
    """计算序列指标"""
    results = []
    
    heights = stats_df['height'].unique()
    
    for height in heights:
        height_data = stats_df[stats_df['height'] == height]
        
        # 计算相关系数
        obs_ec_corr = np.corrcoef(height_data['obs_mean'], height_data['ec_mean'])[0, 1]
        obs_gfs_corr = np.corrcoef(height_data['obs_mean'], height_data['gfs_mean'])[0, 1]
        
        # 计算RMSE
        obs_vals = height_data['obs_mean'].values
        ec_vals = height_data['ec_mean'].values
        gfs_vals = height_data['gfs_mean'].values
        
        ec_rmse = np.sqrt(np.mean((obs_vals - ec_vals)**2))
        gfs_rmse = np.sqrt(np.mean((obs_vals - gfs_vals)**2))
        
        # 计算偏差
        ec_bias = np.mean(ec_vals - obs_vals)
        gfs_bias = np.mean(gfs_vals - obs_vals)
        
        # 计算变异系数
        obs_cv = height_data['obs_std'].mean() / height_data['obs_mean'].mean()
        ec_cv = height_data['ec_std'].mean() / height_data['ec_mean'].mean()
        gfs_cv = height_data['gfs_std'].mean() / height_data['gfs_mean'].mean()
        
        results.append({
            'height': height,
            'obs_ec_correlation': obs_ec_corr,
            'obs_gfs_correlation': obs_gfs_corr,
            'ec_rmse': ec_rmse,
            'gfs_rmse': gfs_rmse,
            'ec_bias': ec_bias,
            'gfs_bias': gfs_bias,
            'obs_coeff_variation': obs_cv,
            'ec_coeff_variation': ec_cv,
            'gfs_coeff_variation': gfs_cv,
            'obs_mean_daily_range': height_data['obs_mean'].max() - height_data['obs_mean'].min(),
            'ec_mean_daily_range': height_data['ec_mean'].max() - height_data['ec_mean'].min(),
            'gfs_mean_daily_range': height_data['gfs_mean'].max() - height_data['gfs_mean'].min()
        })
    
    return pd.DataFrame(results)

def print_sequence_summary(metrics_df, stats_df):
    """打印序列分析摘要"""
    print("\n" + "="*70)
    print("WIND SPEED DIURNAL SEQUENCE ANALYSIS SUMMARY")
    print("="*70)
    
    for _, row in metrics_df.iterrows():
        height = row['height']
        print(f"\nWind Speed {height}:")
        print("-" * 50)
        
        print(f"Correlation with Observations:")
        print(f"  EC Model:  {row['obs_ec_correlation']:>6.3f}")
        print(f"  GFS Model: {row['obs_gfs_correlation']:>6.3f}")
        
        print(f"\nPrediction Accuracy (RMSE):")
        print(f"  EC Model:  {row['ec_rmse']:>6.3f} m/s")
        print(f"  GFS Model: {row['gfs_rmse']:>6.3f} m/s")
        
        print(f"\nBias (Model - Observed):")
        print(f"  EC Model:  {row['ec_bias']:>+6.3f} m/s")
        print(f"  GFS Model: {row['gfs_bias']:>+6.3f} m/s")
        
        print(f"\nDaily Variation Range:")
        print(f"  Observed:  {row['obs_mean_daily_range']:>6.3f} m/s")
        print(f"  EC Model:  {row['ec_mean_daily_range']:>6.3f} m/s")
        print(f"  GFS Model: {row['gfs_mean_daily_range']:>6.3f} m/s")
    
    # 总体比较
    ec_better_corr = (metrics_df['obs_ec_correlation'] > metrics_df['obs_gfs_correlation']).sum()
    ec_better_rmse = (metrics_df['ec_rmse'] < metrics_df['gfs_rmse']).sum()
    total_heights = len(metrics_df)
    
    print(f"\n{'='*70}")
    print("OVERALL MODEL COMPARISON")
    print(f"{'='*70}")
    print(f"Heights where EC has higher correlation: {ec_better_corr}/{total_heights}")
    print(f"Heights where EC has lower RMSE: {ec_better_rmse}/{total_heights}")
    
    # 找出最佳和最差表现的高度
    best_corr_height = metrics_df.loc[metrics_df[['obs_ec_correlation', 'obs_gfs_correlation']].max(axis=1).idxmax(), 'height']
    worst_rmse_height = metrics_df.loc[metrics_df[['ec_rmse', 'gfs_rmse']].min(axis=1).idxmax(), 'height']
    
    print(f"\nBest overall correlation at: {best_corr_height}")
    print(f"Best overall accuracy at: {worst_rmse_height}")

def main():
    """主分析流程"""
    print("="*70)
    print("WIND SPEED DIURNAL SEQUENCE ANALYSIS")
    print("="*70)
    
    # 1. 读取数据
    df = load_and_prepare_data()
    
    # 2. 获取风速变量
    variables = get_wind_speed_variables(df)
    print(f"\nFound {len(variables)} wind speed heights:")
    for var in variables:
        print(f"  - {var['height']}")
    
    if not variables:
        print("No wind speed variables found!")
        return
    
    # 3. 计算昼夜循环统计
    print("\nCalculating diurnal statistics...")
    stats_df = calculate_diurnal_statistics(df, variables)
    
    # 4. 计算序列指标
    print("Calculating sequence metrics...")
    metrics_df = calculate_sequence_metrics(stats_df)
    
    # 5. 保存数据
    print("Saving results...")
    stats_df.to_csv(os.path.join(results_dir, 'wind_speed_diurnal_sequence_stats.csv'), index=False)
    metrics_df.to_csv(os.path.join(results_dir, 'wind_speed_sequence_metrics.csv'), index=False)
    
    # 6. 绘制每个高度的序列图
    print("\nGenerating sequence plots for each height...")
    for var in variables:
        print(f"Plotting {var['height']}...")
        plot_diurnal_sequence_by_height(stats_df, var['height'])
    
    # 7. 绘制季节对比图
    print("\nGenerating seasonal comparison plots...")
    plot_seasonal_comparison(df, variables)
    
    # 8. 打印摘要
    print_sequence_summary(metrics_df, stats_df)
    
    print(f"\n{'='*70}")
    print("ANALYSIS COMPLETED SUCCESSFULLY")
    print(f"All results saved to: {results_dir}")
    print("Generated files:")
    print("  - wind_speed_diurnal_sequence_stats.csv")
    print("  - wind_speed_sequence_metrics.csv")
    print("  - wind_speed_[height]_diurnal_sequence.png (for each height)")
    print("  - wind_speed_[height]_seasonal_comparison.png (for each height)")
    print(f"{'='*70}")

if __name__ == "__main__":
    main()