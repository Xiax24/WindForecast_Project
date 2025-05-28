"""
1.2.1 变量分布特征分析
分析各气象变量和功率的概率分布特征
"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

# 设置图表样式
sns.set_style("whitegrid")
plt.rcParams['figure.dpi'] = 300

def analyze_wind_speed_distributions(df, output_dir):
    """
    风速分布分析
    """
    print("=== Wind Speed Distribution Analysis ===")
    
    # 找到所有观测风速变量
    wind_speed_cols = [col for col in df.columns if col.startswith('obs_wind_speed')]
    
    if not wind_speed_cols:
        print("No wind speed observation data found")
        return
    
    # 创建图形
    n_cols = len(wind_speed_cols)
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('Wind Speed Distribution Analysis', fontsize=16, fontweight='bold')
    axes = axes.flatten()
    
    # 统计信息存储
    distribution_stats = {}
    
    for i, col in enumerate(wind_speed_cols):
        ax = axes[i]
        wind_data = df[col].dropna()
        
        # 绘制直方图
        n, bins, patches = ax.hist(wind_data, bins=50, density=True, alpha=0.7, 
                                  color='skyblue', edgecolor='black', label='Observed data')
        
        # 拟合威布尔分布（风速的理论分布）
        try:
            # 威布尔分布参数估计
            shape, loc, scale = stats.weibull_min.fit(wind_data, floc=0)
            x = np.linspace(wind_data.min(), wind_data.max(), 200)
            weibull_pdf = stats.weibull_min.pdf(x, shape, loc, scale)
            ax.plot(x, weibull_pdf, 'r-', linewidth=2, 
                   label=f'Weibull (k={shape:.2f}, c={scale:.2f})')
            
            # Kolmogorov-Smirnov检验
            ks_stat, ks_p = stats.kstest(wind_data, 
                                        lambda x: stats.weibull_min.cdf(x, shape, loc, scale))
            
            distribution_stats[col] = {
                'mean': wind_data.mean(),
                'std': wind_data.std(),
                'weibull_k': shape,
                'weibull_c': scale,
                'ks_statistic': ks_stat,
                'ks_p_value': ks_p
            }
            
        except Exception as e:
            print(f"Weibull distribution fitting failed {col}: {e}")
        
        # 拟合正态分布进行对比
        try:
            mu, sigma = stats.norm.fit(wind_data)
            normal_pdf = stats.norm.pdf(x, mu, sigma)
            ax.plot(x, normal_pdf, 'g--', linewidth=2, 
                   label=f'Normal (μ={mu:.2f}, σ={sigma:.2f})')
        except:
            pass
        
        # 图形设置
        height = col.replace('obs_wind_speed_', '').replace('m', '')
        ax.set_title(f'{height}m Height Wind Speed Distribution')
        ax.set_xlabel('Wind Speed (m/s)')
        ax.set_ylabel('Probability Density')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # 添加统计信息文本
        stats_text = f'Mean: {wind_data.mean():.2f}\nStd: {wind_data.std():.2f}\nSamples: {len(wind_data)}'
        ax.text(0.02, 0.98, stats_text, transform=ax.transAxes, fontsize=9,
               verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.tight_layout()
    plt.savefig(f'{output_dir}/wind_speed_distributions.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # 保存统计结果
    stats_df = pd.DataFrame(distribution_stats).T
    stats_df.to_csv(f'{output_dir}/wind_speed_distribution_stats.csv')
    
    print("Wind speed distribution analysis completed")
    print(f"Statistical summary:\n{df[wind_speed_cols].describe().round(3)}")
    
    return distribution_stats

def analyze_wind_direction_distributions(df, output_dir):
    """
    风向分布分析（风向玫瑰图）
    """
    print("\n=== Wind Direction Distribution Analysis ===")
    
    # 找到所有观测风向变量
    wind_dir_cols = [col for col in df.columns if col.startswith('obs_wind_direction')]
    
    if not wind_dir_cols:
        print("No wind direction observation data found")
        return
    
    # 创建极坐标图
    fig, axes = plt.subplots(2, 2, figsize=(16, 16), subplot_kw=dict(projection='polar'))
    fig.suptitle('Wind Direction Distribution (Wind Rose)', fontsize=16, fontweight='bold')
    axes = axes.flatten()
    
    # 16个方向的定义
    direction_names = ['N', 'NNE', 'NE', 'ENE', 'E', 'ESE', 'SE', 'SSE',
                      'S', 'SSW', 'SW', 'WSW', 'W', 'WNW', 'NW', 'NNW']
    
    wind_dir_stats = {}
    
    for i, col in enumerate(wind_dir_cols):
        ax = axes[i]
        wind_dir_data = df[col].dropna()
        
        # 转换为弧度
        wind_dir_rad = np.deg2rad(wind_dir_data)
        
        # 计算16个方向的频率
        bins = np.linspace(0, 2*np.pi, 17)  # 16个扇区
        hist, bin_edges = np.histogram(wind_dir_rad, bins=bins)
        
        # 计算每个扇区的中心角度
        theta = bins[:-1] + np.pi/16
        
        # 绘制风向玫瑰图
        bars = ax.bar(theta, hist, width=2*np.pi/16, alpha=0.7, color='lightcoral', 
                     edgecolor='black', linewidth=0.5)
        
        # 设置图形属性
        ax.set_theta_zero_location('N')  # 北方向为0度
        ax.set_theta_direction(-1)       # 顺时针方向
        ax.set_thetagrids(np.arange(0, 360, 22.5), 
                         ['N', 'NNE', 'NE', 'ENE', 'E', 'ESE', 'SE', 'SSE',
                          'S', 'SSW', 'SW', 'WSW', 'W', 'WNW', 'NW', 'NNW'])
        
        height = col.replace('obs_wind_direction_', '').replace('m', '')
        ax.set_title(f'{height}m Height Wind Direction\n(Total samples: {len(wind_dir_data)})', 
                    fontsize=12, pad=20)
        
        # 计算主导风向
        dominant_direction_idx = np.argmax(hist)
        dominant_direction = direction_names[dominant_direction_idx]
        dominant_freq = hist[dominant_direction_idx] / len(wind_dir_data) * 100
        
        # 计算风向统计量
        # 平均风向（圆形平均）
        mean_angle = np.angle(np.mean(np.exp(1j * wind_dir_rad))) * 180 / np.pi
        if mean_angle < 0:
            mean_angle += 360
        
        # 风向集中度（向量强度）
        concentration = np.abs(np.mean(np.exp(1j * wind_dir_rad)))
        
        wind_dir_stats[col] = {
            'dominant_direction': dominant_direction,
            'dominant_frequency_percent': dominant_freq,
            'mean_direction_deg': mean_angle,
            'concentration': concentration,
            'sample_count': len(wind_dir_data)
        }
        
        # 添加统计信息
        stats_text = f'Dominant: {dominant_direction}\nFreq: {dominant_freq:.1f}%\nConcentration: {concentration:.3f}'
        ax.text(0.02, 0.98, stats_text, transform=ax.transAxes, fontsize=10,
               verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    
    plt.tight_layout()
    plt.savefig(f'{output_dir}/wind_direction_distributions.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # 保存风向统计
    wind_dir_stats_df = pd.DataFrame(wind_dir_stats).T
    wind_dir_stats_df.to_csv(f'{output_dir}/wind_direction_stats.csv')
    
    print("Wind direction distribution analysis completed")
    print("Wind direction statistical summary:")
    for col, stats in wind_dir_stats.items():
        height = col.replace('obs_wind_direction_', '').replace('m', '')
        print(f"  {height}m: Dominant direction {stats['dominant_direction']} ({stats['dominant_frequency_percent']:.1f}%)")
    
    return wind_dir_stats

def analyze_other_variables(df, output_dir):
    """
    分析温度、湿度、密度等其他变量的分布
    """
    print("\n=== Other Meteorological Variables Distribution Analysis ===")
    
    # 分类变量
    temp_cols = [col for col in df.columns if col.startswith('obs_temperature')]
    humidity_cols = [col for col in df.columns if col.startswith('obs_humidity')]
    density_cols = [col for col in df.columns if col.startswith('obs_density')]
    
    other_vars = temp_cols + humidity_cols + density_cols
    
    if not other_vars:
        print("No other meteorological variables found")
        return
    
    # 创建图形
    n_vars = len(other_vars)
    n_cols = min(3, n_vars)
    n_rows = (n_vars + n_cols - 1) // n_cols
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 5*n_rows))
    fig.suptitle('Other Meteorological Variables Distribution', fontsize=16, fontweight='bold')
    
    if n_vars == 1:
        axes = [axes]
    elif n_rows == 1:
        axes = axes if n_vars > 1 else [axes]
    else:
        axes = axes.flatten()
    
    other_var_stats = {}
    
    for i, col in enumerate(other_vars):
        ax = axes[i]
        var_data = df[col].dropna()
        
        # 绘制直方图
        ax.hist(var_data, bins=50, density=True, alpha=0.7, 
               color='lightgreen', edgecolor='black')
        
        # 拟合正态分布
        mu, sigma = stats.norm.fit(var_data)
        x = np.linspace(var_data.min(), var_data.max(), 200)
        normal_pdf = stats.norm.pdf(x, mu, sigma)
        ax.plot(x, normal_pdf, 'r-', linewidth=2, 
               label=f'Normal (μ={mu:.2f}, σ={sigma:.2f})')
        
        # 正态性检验
        shapiro_stat, shapiro_p = stats.shapiro(var_data[:5000] if len(var_data) > 5000 else var_data)
        
        # 设置标题和标签
        var_name = col.replace('obs_', '').replace('_', ' ').title()
        ax.set_title(var_name)
        
        if 'temperature' in col:
            ax.set_xlabel('Temperature (°C)')
        elif 'humidity' in col:
            ax.set_xlabel('Relative Humidity (%)')
        elif 'density' in col:
            ax.set_xlabel('Air Density (kg/m³)')
        else:
            ax.set_xlabel('Value')
        
        ax.set_ylabel('Probability Density')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # 添加统计信息
        stats_text = f'Mean: {mu:.3f}\nStd: {sigma:.3f}\nSkewness: {stats.skew(var_data):.3f}\nKurtosis: {stats.kurtosis(var_data):.3f}'
        ax.text(0.02, 0.98, stats_text, transform=ax.transAxes, fontsize=9,
               verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        
        # 保存统计信息
        other_var_stats[col] = {
            'mean': mu,
            'std': sigma,
            'skewness': stats.skew(var_data),
            'kurtosis': stats.kurtosis(var_data),
            'shapiro_statistic': shapiro_stat,
            'shapiro_p_value': shapiro_p,
            'min': var_data.min(),
            'max': var_data.max()
        }
    
    # 隐藏多余的子图
    for j in range(n_vars, len(axes)):
        axes[j].set_visible(False)
    
    plt.tight_layout()
    plt.savefig(f'{output_dir}/other_variables_distributions.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # 保存统计结果
    other_stats_df = pd.DataFrame(other_var_stats).T
    other_stats_df.to_csv(f'{output_dir}/other_variables_stats.csv')
    
    print("Other variables distribution analysis completed")
    print(f"Statistical summary:\n{df[other_vars].describe().round(3)}")
    
    return other_var_stats

def analyze_power_distribution(df, output_dir):
    """
    功率分布特征分析
    """
    print("\n=== Power Distribution Analysis ===")
    
    # 找到功率变量
    power_cols = [col for col in df.columns if 'power' in col.lower()]
    
    if not power_cols:
        print("No power data found")
        return
    
    power_col = power_cols[0]
    power_data = df[power_col].dropna()
    
    # 创建图形
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('Power Distribution Analysis', fontsize=16, fontweight='bold')
    
    # 1. 功率分布直方图
    ax1 = axes[0, 0]
    n, bins, patches = ax1.hist(power_data, bins=50, density=True, alpha=0.7, 
                               color='orange', edgecolor='black')
    
    # 标记不同功率区间
    # 零功率区间
    zero_power = (power_data == 0).sum()
    zero_percent = zero_power / len(power_data) * 100
    
    ax1.set_title(f'Power Distribution Histogram\nZero Power: {zero_percent:.1f}% ({zero_power} samples)')
    ax1.set_xlabel('Power')
    ax1.set_ylabel('Probability Density')
    ax1.grid(True, alpha=0.3)
    
    # 2. 功率盒图
    ax2 = axes[0, 1]
    bp = ax2.boxplot(power_data, patch_artist=True)
    bp['boxes'][0].set_facecolor('lightblue')
    
    # 计算分位数
    q25, q50, q75 = np.percentile(power_data, [25, 50, 75])
    iqr = q75 - q25
    
    ax2.set_title(f'Power Distribution Boxplot\nIQR: {iqr:.2f}')
    ax2.set_ylabel('Power')
    ax2.grid(True, alpha=0.3)
    
    # 添加统计信息
    stats_text = f'Q25: {q25:.2f}\nQ50: {q50:.2f}\nQ75: {q75:.2f}\nIQR: {iqr:.2f}'
    ax2.text(1.1, 0.5, stats_text, transform=ax2.transAxes, fontsize=10,
            verticalalignment='center', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    # 3. Q-Q图检验正态性
    ax3 = axes[1, 0]
    stats.probplot(power_data, dist="norm", plot=ax3)
    ax3.set_title('Power Normal Q-Q Plot')
    ax3.grid(True, alpha=0.3)
    
    # 4. 累积分布函数
    ax4 = axes[1, 1]
    sorted_power = np.sort(power_data)
    cdf = np.arange(1, len(sorted_power) + 1) / len(sorted_power)
    ax4.plot(sorted_power, cdf, 'b-', linewidth=2)
    ax4.set_title('Power Cumulative Distribution Function')
    ax4.set_xlabel('Power')
    ax4.set_ylabel('Cumulative Probability')
    ax4.grid(True, alpha=0.3)
    
    # 标记重要分位点
    percentiles = [10, 25, 50, 75, 90, 95, 99]
    for p in percentiles:
        value = np.percentile(power_data, p)
        ax4.axvline(value, color='red', linestyle='--', alpha=0.5)
        ax4.text(value, p/100, f'P{p}', rotation=90, fontsize=8)
    
    plt.tight_layout()
    plt.savefig(f'{output_dir}/power_distribution_analysis.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # 详细统计分析
    power_stats = {
        'count': len(power_data),
        'mean': power_data.mean(),
        'std': power_data.std(),
        'min': power_data.min(),
        'max': power_data.max(),
        'q25': q25,
        'q50': q50,
        'q75': q75,
        'skewness': stats.skew(power_data),
        'kurtosis': stats.kurtosis(power_data),
        'zero_power_count': zero_power,
        'zero_power_percent': zero_percent
    }
    
    # 正态性检验
    shapiro_stat, shapiro_p = stats.shapiro(power_data[:5000] if len(power_data) > 5000 else power_data)
    power_stats['shapiro_statistic'] = shapiro_stat
    power_stats['shapiro_p_value'] = shapiro_p
    
    # 保存功率统计
    power_stats_df = pd.DataFrame([power_stats]).T
    power_stats_df.columns = ['Value']
    power_stats_df.to_csv(f'{output_dir}/power_distribution_stats.csv')
    
    print("Power distribution analysis completed")
    print(f"Power statistical summary:")
    for key, value in power_stats.items():
        if isinstance(value, float):
            print(f"  {key}: {value:.3f}")
        else:
            print(f"  {key}: {value}")
    
    return power_stats

def run_distribution_analysis(data_path, output_dir):
    """
    运行完整的变量分布分析
    """
    import os
    
    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)
    
    print("=== 1.2.1 Variable Distribution Analysis ===")
    print(f"Input data: {data_path}")
    print(f"Output directory: {output_dir}")
    
    # 加载数据
    df = pd.read_csv(data_path)
    print(f"Data shape: {df.shape}")
    
    # 执行各项分析
    wind_speed_stats = analyze_wind_speed_distributions(df, output_dir)
    wind_dir_stats = analyze_wind_direction_distributions(df, output_dir)
    other_var_stats = analyze_other_variables(df, output_dir)
    power_stats = analyze_power_distribution(df, output_dir)
    
    # 生成综合报告
    generate_distribution_report(df, output_dir, wind_speed_stats, wind_dir_stats, 
                               other_var_stats, power_stats)
    
    print(f"\n✓ Variable distribution analysis completed!")
    print(f"✓ All charts saved to: {output_dir}")

def generate_distribution_report(df, output_dir, wind_speed_stats, wind_dir_stats, 
                               other_var_stats, power_stats):
    """
    生成分布分析综合报告
    """
    report = []
    report.append("=== 1.2.1 Variable Distribution Analysis Report ===")
    report.append(f"Analysis time: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}")
    report.append(f"Data file: {df.shape[0]} rows × {df.shape[1]} columns")
    
    report.append("\n【Wind Speed Distribution】")
    if wind_speed_stats:
        for col, stats in wind_speed_stats.items():
            height = col.replace('obs_wind_speed_', '').replace('m', '')
            report.append(f"  {height}m height:")
            report.append(f"    Mean wind speed: {stats['mean']:.2f} m/s")
            report.append(f"    Standard deviation: {stats['std']:.2f} m/s")
            report.append(f"    Weibull parameters: k={stats['weibull_k']:.2f}, c={stats['weibull_c']:.2f}")
            if stats['ks_p_value'] > 0.05:
                report.append(f"    Weibull fit: Good (p={stats['ks_p_value']:.3f})")
            else:
                report.append(f"    Weibull fit: Fair (p={stats['ks_p_value']:.3f})")
    
    report.append("\n【Wind Direction Distribution】")
    if wind_dir_stats:
        for col, stats in wind_dir_stats.items():
            height = col.replace('obs_wind_direction_', '').replace('m', '')
            report.append(f"  {height}m height:")
            report.append(f"    Dominant direction: {stats['dominant_direction']} ({stats['dominant_frequency_percent']:.1f}%)")
            report.append(f"    Wind direction concentration: {stats['concentration']:.3f}")
    
    report.append("\n【Other Variables Distribution】")
    if other_var_stats:
        for col, stats in other_var_stats.items():
            var_name = col.replace('obs_', '').replace('_', ' ')
            report.append(f"  {var_name}:")
            report.append(f"    Mean±Std: {stats['mean']:.3f}±{stats['std']:.3f}")
            report.append(f"    Skewness: {stats['skewness']:.3f}, Kurtosis: {stats['kurtosis']:.3f}")
            if stats['shapiro_p_value'] > 0.05:
                report.append(f"    Normality: Normal distribution (p={stats['shapiro_p_value']:.3f})")
            else:
                report.append(f"    Normality: Non-normal distribution (p={stats['shapiro_p_value']:.3f})")
    
    report.append("\n【Power Distribution】")
    if power_stats:
        report.append(f"  Mean power: {power_stats['mean']:.2f}")
        report.append(f"  Power standard deviation: {power_stats['std']:.2f}")
        report.append(f"  Power range: {power_stats['min']:.2f} - {power_stats['max']:.2f}")
        report.append(f"  Zero power percentage: {power_stats['zero_power_percent']:.1f}%")
        report.append(f"  Distribution skewness: {power_stats['skewness']:.3f}")
        if power_stats['shapiro_p_value'] > 0.05:
            report.append(f"  Normality: Normal distribution")
        else:
            report.append(f"  Normality: Non-normal distribution")
    
    # 保存报告
    report_text = '\n'.join(report)
    with open(f'{output_dir}/distribution_analysis_report.txt', 'w', encoding='utf-8') as f:
        f.write(report_text)
    
    print("Analysis report generated")

# 使用示例
if __name__ == "__main__":
    # 设置路径
    data_path = "/Users/xiaxin/work/WindForecast_Project/01_Data/processed/imputed_data/changma_imputed_complete.csv"
    output_dir = "/Users/xiaxin/work/WindForecast_Project/03_Results/1_2_1_distributions"
    
    # 运行分析
    run_distribution_analysis(data_path, output_dir)