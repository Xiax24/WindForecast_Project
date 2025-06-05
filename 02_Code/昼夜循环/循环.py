import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import os
from scipy import stats

# 设置绘图参数
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['figure.dpi'] = 100
sns.set_palette("husl")

def create_comprehensive_diurnal_analysis(data_path, results_dir):
    """
    创建综合的昼夜循环分析
    """
    # 创建结果目录
    os.makedirs(results_dir, exist_ok=True)
    
    # 读取数据
    print("正在读取数据...")
    df = pd.read_csv(data_path)
    df['datetime'] = pd.to_datetime(df['datetime'])
    df['hour'] = df['datetime'].dt.hour
    df['month'] = df['datetime'].dt.month
    df['season'] = df['month'].map({12: 'Winter', 1: 'Winter', 2: 'Winter',
                                   3: 'Spring', 4: 'Spring', 5: 'Spring',
                                   6: 'Summer', 7: 'Summer', 8: 'Summer',
                                   9: 'Autumn', 10: 'Autumn', 11: 'Autumn'})
    
    print(f"数据时间范围: {df['datetime'].min()} 到 {df['datetime'].max()}")
    print(f"数据点数量: {len(df)}")
    
    # 定义观测数据列
    wind_cols = [col for col in df.columns if col.startswith('obs_wind_speed')]
    temp_cols = [col for col in df.columns if col.startswith('obs_temperature')]
    power_col = 'power'
    
    return df, wind_cols, temp_cols, power_col

def plot_diurnal_cycle_with_uncertainty(df, columns, col_type, results_dir, 
                                       ylabel, title_prefix, filename_prefix):
    """
    绘制包含不确定性的昼夜循环图
    """
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 12))
    
    # 计算小时统计
    hourly_stats = []
    for hour in range(24):
        hour_data = df[df['hour'] == hour]
        stats_dict = {'hour': hour}
        
        for col in columns:
            if col in df.columns:
                values = hour_data[col].dropna()
                if len(values) > 0:
                    stats_dict[f'{col}_mean'] = values.mean()
                    stats_dict[f'{col}_std'] = values.std()
                    stats_dict[f'{col}_median'] = values.median()
                    stats_dict[f'{col}_q25'] = values.quantile(0.25)
                    stats_dict[f'{col}_q75'] = values.quantile(0.75)
                    stats_dict[f'{col}_count'] = len(values)
        
        hourly_stats.append(stats_dict)
    
    stats_df = pd.DataFrame(hourly_stats)
    
    # 上图：均值 ± 标准差
    colors = plt.cm.Set1(np.linspace(0, 1, len(columns)))
    
    for i, col in enumerate(columns):
        if f'{col}_mean' in stats_df.columns:
            mean_col = f'{col}_mean'
            std_col = f'{col}_std'
            
            # 获取高度信息（如果是风速）
            if 'wind_speed' in col:
                height = col.split('_')[-1]
                label = f'Observed {height}'
            elif 'temperature' in col:
                height = col.split('_')[-1]
                label = f'Observed {height}'
            else:
                label = col_type
            
            ax1.plot(stats_df['hour'], stats_df[mean_col], 
                    color=colors[i], linewidth=2.5, marker='o', markersize=5,
                    label=label, alpha=0.8)
            
            ax1.fill_between(stats_df['hour'], 
                           stats_df[mean_col] - stats_df[std_col],
                           stats_df[mean_col] + stats_df[std_col],
                           color=colors[i], alpha=0.15)
    
    ax1.set_xlabel('Hour', fontsize=12)
    ax1.set_ylabel(ylabel, fontsize=12)
    ax1.set_title(f'{title_prefix} Diurnal Cycle (Mean ± Std)', fontsize=14, fontweight='bold')
    ax1.legend(fontsize=10, loc='best')
    ax1.grid(True, alpha=0.3)
    ax1.set_xticks(range(0, 24, 2))
    ax1.set_xlim(0, 23)
    
    # 下图：箱线图显示分布
    for i, col in enumerate(columns):
        if f'{col}_median' in stats_df.columns:
            median_col = f'{col}_median'
            q25_col = f'{col}_q25'
            q75_col = f'{col}_q75'
            
            if 'wind_speed' in col:
                height = col.split('_')[-1]
                label = f'Observed {height}'
            elif 'temperature' in col:
                height = col.split('_')[-1]
                label = f'Observed {height}'
            else:
                label = col_type
            
            ax2.plot(stats_df['hour'], stats_df[median_col], 
                    color=colors[i], linewidth=2, marker='s', markersize=4,
                    label=f'{label} (Median)', alpha=0.8)
            
            ax2.fill_between(stats_df['hour'], 
                           stats_df[q25_col], stats_df[q75_col],
                           color=colors[i], alpha=0.2)
    
    ax2.set_xlabel('Hour', fontsize=12)
    ax2.set_ylabel(ylabel, fontsize=12)
    ax2.set_title(f'{title_prefix} Diurnal Cycle (Median with IQR)', fontsize=14, fontweight='bold')
    ax2.legend(fontsize=10, loc='best')
    ax2.grid(True, alpha=0.3)
    ax2.set_xticks(range(0, 24, 2))
    ax2.set_xlim(0, 23)
    
    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, f'{filename_prefix}_comprehensive_diurnal.png'), 
                dpi=300, bbox_inches='tight')
    plt.show()
    
    return stats_df

def plot_seasonal_diurnal(df, columns, col_type, results_dir, ylabel, title_prefix, filename_prefix):
    """
    绘制季节性昼夜循环对比图
    """
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    axes = axes.flatten()
    
    seasons = ['Spring', 'Summer', 'Autumn', 'Winter']
    colors = plt.cm.Set1(np.linspace(0, 1, len(columns)))
    
    # 首先计算所有季节的数据范围，用于统一y轴
    all_means = []
    all_stds = []
    seasonal_data = {}
    
    for season in seasons:
        season_data = df[df['season'] == season]
        seasonal_data[season] = {}
        
        for col in columns:
            if col in df.columns:
                hourly_means = []
                hourly_stds = []
                hours = []
                
                for hour in range(24):
                    hour_data = season_data[season_data['hour'] == hour][col].dropna()
                    if len(hour_data) > 0:
                        mean_val = hour_data.mean()
                        std_val = hour_data.std()
                        hourly_means.append(mean_val)
                        hourly_stds.append(std_val)
                        hours.append(hour)
                        all_means.append(mean_val)
                        all_stds.append(std_val)
                
                seasonal_data[season][col] = {
                    'means': hourly_means,
                    'stds': hourly_stds,
                    'hours': hours
                }
    
    # 计算统一的y轴范围
    if all_means:
        y_min = min(np.array(all_means) - np.array(all_stds)) * 0.95
        y_max = max(np.array(all_means) + np.array(all_stds)) * 1.05
    else:
        y_min, y_max = 0, 1
    
    # 绘制每个季节的图
    for season_idx, season in enumerate(seasons):
        ax = axes[season_idx]
        
        for col_idx, col in enumerate(columns):
            if col in seasonal_data[season] and seasonal_data[season][col]['means']:
                data = seasonal_data[season][col]
                
                if 'wind_speed' in col:
                    height = col.split('_')[-1]
                    label = f'{height}'
                elif 'temperature' in col:
                    height = col.split('_')[-1]
                    label = f'{height}'
                else:
                    label = col_type
                
                ax.plot(data['hours'], data['means'], color=colors[col_idx], 
                       linewidth=2, marker='o', markersize=4, label=label)
                
                ax.fill_between(data['hours'], 
                               np.array(data['means']) - np.array(data['stds']),
                               np.array(data['means']) + np.array(data['stds']),
                               color=colors[col_idx], alpha=0.2)
        
        ax.set_title(f'{season} - {title_prefix}', fontsize=12, fontweight='bold')
        ax.set_xlabel('Hour', fontsize=10)
        ax.set_ylabel(ylabel, fontsize=10)
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)
        ax.set_xticks(range(0, 24, 4))
        ax.set_xlim(0, 23)
        ax.set_ylim(y_min, y_max)  # 统一y轴范围
    
    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, f'{filename_prefix}_seasonal_diurnal.png'), 
                dpi=300, bbox_inches='tight')
    plt.show()

def generate_statistical_summary(df, columns, results_dir, filename_prefix):
    """
    生成统计摘要报告
    """
    summary_stats = []
    
    for col in columns:
        if col in df.columns:
            # 整体统计
            overall_stats = df[col].describe()
            
            # 昼夜差异
            day_hours = df[df['hour'].between(6, 18)][col].dropna()
            night_hours = df[~df['hour'].between(6, 18)][col].dropna()
            
            # 峰值时间
            hourly_means = df.groupby('hour')[col].mean()
            peak_hour = hourly_means.idxmax()
            trough_hour = hourly_means.idxmin()
            
            summary_stats.append({
                'variable': col,
                'mean': overall_stats['mean'],
                'std': overall_stats['std'],
                'min': overall_stats['min'],
                'max': overall_stats['max'],
                'day_mean': day_hours.mean() if len(day_hours) > 0 else np.nan,
                'night_mean': night_hours.mean() if len(night_hours) > 0 else np.nan,
                'day_night_diff': (day_hours.mean() - night_hours.mean()) if len(day_hours) > 0 and len(night_hours) > 0 else np.nan,
                'peak_hour': peak_hour,
                'trough_hour': trough_hour,
                'diurnal_range': hourly_means.max() - hourly_means.min()
            })
    
    summary_df = pd.DataFrame(summary_stats)
    summary_df.to_csv(os.path.join(results_dir, f'{filename_prefix}_statistical_summary.csv'), index=False)
    
    return summary_df

# 主分析流程
def main():
    # 路径设置
    data_path = '/Users/xiaxin/work/WindForecast_Project/01_Data/processed/imputed_data/changma_imputed_complete.csv'
    results_dir = '/Users/xiaxin/work/WindForecast_Project/03_Results/4diurnal/'
    
    # 读取和准备数据
    df, wind_cols, temp_cols, power_col = create_comprehensive_diurnal_analysis(data_path, results_dir)
    
    print("\n=== Starting Diurnal Cycle Analysis ===")
    
    # 1. 风速分析
    print("\n1. Analyzing wind speed diurnal cycle...")
    wind_stats = plot_diurnal_cycle_with_uncertainty(
        df, wind_cols, 'Wind Speed', results_dir, 
        'Wind Speed (m/s)', 'Wind Speed', 'wind_speed'
    )
    
    plot_seasonal_diurnal(
        df, wind_cols, 'Wind Speed', results_dir,
        'Wind Speed (m/s)', 'Wind Speed Diurnal Cycle', 'wind_speed'
    )
    
    wind_summary = generate_statistical_summary(df, wind_cols, results_dir, 'wind_speed')
    
    # 2. 温度分析
    print("\n2. Analyzing temperature diurnal cycle...")
    temp_stats = plot_diurnal_cycle_with_uncertainty(
        df, temp_cols, 'Temperature', results_dir,
        'Temperature (°C)', 'Temperature', 'temperature'
    )
    
    plot_seasonal_diurnal(
        df, temp_cols, 'Temperature', results_dir,
        'Temperature (°C)', 'Temperature Diurnal Cycle', 'temperature'
    )
    
    temp_summary = generate_statistical_summary(df, temp_cols, results_dir, 'temperature')
    
    # 3. 功率分析
    print("\n3. Analyzing power diurnal cycle...")
    power_stats = plot_diurnal_cycle_with_uncertainty(
        df, [power_col], 'Power', results_dir,
        'Power (kW)', 'Power', 'power'
    )
    
    plot_seasonal_diurnal(
        df, [power_col], 'Power', results_dir,
        'Power (kW)', 'Power Diurnal Cycle', 'power'
    )
    
    power_summary = generate_statistical_summary(df, [power_col], results_dir, 'power')
    
    print(f"\n=== Analysis Completed ===")
    print(f"All results saved to: {results_dir}")
    
    # 打印一些关键发现
    print("\n=== Key Findings ===")
    print("Wind speed diurnal variation:")
    for _, row in wind_summary.iterrows():
        print(f"  {row['variable']}: Peak at {row['peak_hour']:02d}:00, Trough at {row['trough_hour']:02d}:00, Daily range {row['diurnal_range']:.2f} m/s")
    
    print("\nTemperature diurnal variation:")
    for _, row in temp_summary.iterrows():
        print(f"  {row['variable']}: Peak at {row['peak_hour']:02d}:00, Trough at {row['trough_hour']:02d}:00, Daily range {row['diurnal_range']:.2f} °C")
    
    print("\nPower diurnal variation:")
    for _, row in power_summary.iterrows():
        print(f"  {row['variable']}: Peak at {row['peak_hour']:02d}:00, Trough at {row['trough_hour']:02d}:00, Daily range {row['diurnal_range']:.2f} kW")

if __name__ == "__main__":
    main()