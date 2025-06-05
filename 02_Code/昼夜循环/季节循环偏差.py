import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
from datetime import datetime

# 设置绘图样式和配色
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['figure.figsize'] = (12, 8)
sns.set_style("whitegrid")

# 使用更好看的配色方案
colors_main = ['#2E86AB', '#A23B72']  # 深蓝和深粉
colors_seasonal = ['#264653', '#2A9D8F', '#E9C46A', '#F4A261']  # 深绿、青绿、金黄、橙色
colors_heights = ['#6A4C93', '#9B59B6', '#3498DB', '#1ABC9C']  # 紫、浅紫、蓝、青

# 数据和结果路径
data_path = '/Users/xiaxin/work/WindForecast_Project/01_Data/processed/imputed_data/changma_imputed_complete.csv'
results_dir = '/Users/xiaxin/work/WindForecast_Project/03_Results/5diurnal_error/'

# 创建结果目录
os.makedirs(results_dir, exist_ok=True)

def calculate_bias(y_obs, y_pred):
    """
    计算Bias = mean(predicted - observed)
    正值表示高估，负值表示低估
    """
    mask = ~(np.isnan(y_obs) | np.isnan(y_pred))
    if mask.sum() == 0:
        return np.nan
    return np.mean(y_pred[mask] - y_obs[mask])

def load_and_prepare_data():
    """读取和预处理数据"""
    print("Loading wind speed data...")
    df = pd.read_csv(data_path)
    
    # 转换时间列
    df['datetime'] = pd.to_datetime(df['datetime'])
    df['hour'] = df['datetime'].dt.hour
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

def get_wind_speed_pairs(df):
    """获取风速变量对"""
    pairs = []
    
    for height in ['10m', '30m', '50m', '70m']:
        obs_col = f'obs_wind_speed_{height}'
        ec_col = f'ec_wind_speed_{height}'
        gfs_col = f'gfs_wind_speed_{height}'
        
        if all(col in df.columns for col in [obs_col, ec_col, gfs_col]):
            pairs.append({
                'variable': f'Wind Speed {height}',
                'height': height,
                'obs': obs_col,
                'ec': ec_col,
                'gfs': gfs_col
            })
    
    return pairs

def calculate_diurnal_bias(df, pairs):
    """计算昼夜循环Bias"""
    results = []
    
    for pair in pairs:
        for hour in range(24):
            hour_data = df[df['hour'] == hour]
            
            ec_bias = calculate_bias(
                hour_data[pair['obs']].values,
                hour_data[pair['ec']].values
            )
            
            gfs_bias = calculate_bias(
                hour_data[pair['obs']].values,
                hour_data[pair['gfs']].values
            )
            
            results.append({
                'variable': pair['variable'],
                'height': pair['height'],
                'hour': hour,
                'EC_bias': ec_bias,
                'GFS_bias': gfs_bias
            })
    
    return pd.DataFrame(results)

def calculate_seasonal_diurnal_bias(df, pairs):
    """计算季节性昼夜循环Bias"""
    results = []
    seasons = ['Spring', 'Summer', 'Autumn', 'Winter']
    
    for pair in pairs:
        for season in seasons:
            season_data = df[df['season'] == season]
            
            for hour in range(24):
                hour_data = season_data[season_data['hour'] == hour]
                
                ec_bias = calculate_bias(
                    hour_data[pair['obs']].values,
                    hour_data[pair['ec']].values
                )
                
                gfs_bias = calculate_bias(
                    hour_data[pair['obs']].values,
                    hour_data[pair['gfs']].values
                )
                
                results.append({
                    'variable': pair['variable'],
                    'height': pair['height'],
                    'season': season,
                    'hour': hour,
                    'EC_bias': ec_bias,
                    'GFS_bias': gfs_bias
                })
    
    return pd.DataFrame(results)

def calculate_overall_bias(df, pairs):
    """计算总体Bias"""
    results = []
    
    for pair in pairs:
        ec_bias = calculate_bias(
            df[pair['obs']].values,
            df[pair['ec']].values
        )
        
        gfs_bias = calculate_bias(
            df[pair['obs']].values,
            df[pair['gfs']].values
        )
        
        ec_abs_bias = abs(ec_bias) if not np.isnan(ec_bias) else np.inf
        gfs_abs_bias = abs(gfs_bias) if not np.isnan(gfs_bias) else np.inf
        better_model = 'EC' if ec_abs_bias < gfs_abs_bias else 'GFS'
        
        ec_tendency = 'Overestimate' if ec_bias > 0 else 'Underestimate' if ec_bias < 0 else 'Neutral'
        gfs_tendency = 'Overestimate' if gfs_bias > 0 else 'Underestimate' if gfs_bias < 0 else 'Neutral'
        
        results.append({
            'variable': pair['variable'],
            'height': pair['height'],
            'EC_bias': ec_bias,
            'GFS_bias': gfs_bias,
            'EC_abs_bias': ec_abs_bias,
            'GFS_abs_bias': gfs_abs_bias,
            'better_model': better_model,
            'EC_tendency': ec_tendency,
            'GFS_tendency': gfs_tendency
        })
    
    return pd.DataFrame(results)

def plot_diurnal_bias(diurnal_df):
    """绘制昼夜循环Bias图"""
    variables = diurnal_df['variable'].unique()
    n_vars = len(variables)
    
    fig, axes = plt.subplots(n_vars, 1, figsize=(14, 4*n_vars))
    if n_vars == 1:
        axes = [axes]
    
    for i, var in enumerate(variables):
        var_data = diurnal_df[diurnal_df['variable'] == var].sort_values('hour')
        
        x = np.arange(24)
        width = 0.35
        
        # 绘制柱状图
        bars1 = axes[i].bar(x - width/2, var_data['EC_bias'], width, 
                           label='EC Model', color=colors_main[0], alpha=0.8)
        bars2 = axes[i].bar(x + width/2, var_data['GFS_bias'], width, 
                           label='GFS Model', color=colors_main[1], alpha=0.8)
        
        # 添加零线
        axes[i].axhline(y=0, color='black', linestyle='--', alpha=0.7, linewidth=1)
        
        # 设置标签和标题
        axes[i].set_xlabel('Hour', fontsize=12)
        axes[i].set_ylabel('Bias (m/s)', fontsize=12)
        axes[i].set_title(f'{var} - Diurnal Bias (Positive=Overestimate, Negative=Underestimate)', 
                         fontsize=13, fontweight='bold')
        axes[i].legend(fontsize=11)
        axes[i].grid(True, alpha=0.3)
        axes[i].set_xticks(range(0, 24, 2))
        axes[i].set_xlim(-0.5, 23.5)
        
        # 设置y轴范围（对称）
        all_values = np.concatenate([
            var_data['EC_bias'].dropna().values,
            var_data['GFS_bias'].dropna().values
        ])
        if len(all_values) > 0:
            y_range = max(abs(all_values)) * 1.1
            axes[i].set_ylim(-y_range, y_range)
    
    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, 'wind_speed_diurnal_bias.png'), 
                dpi=300, bbox_inches='tight')
    plt.show()
    print("Saved: wind_speed_diurnal_bias.png")

def plot_overall_bias(overall_df):
    """绘制总体Bias对比图"""
    heights = [var.split()[-1] for var in overall_df['variable']]
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    x = np.arange(len(heights))
    width = 0.35
    
    # 绘制柱状图
    bars1 = ax.bar(x - width/2, overall_df['EC_bias'], width, 
                   label='EC Model', color=colors_main[0], alpha=0.8)
    bars2 = ax.bar(x + width/2, overall_df['GFS_bias'], width, 
                   label='GFS Model', color=colors_main[1], alpha=0.8)
    
    # 添加零线
    ax.axhline(y=0, color='black', linestyle='--', alpha=0.7, linewidth=1)
    
    # 设置标签和标题
    ax.set_xlabel('Height', fontsize=12)
    ax.set_ylabel('Bias (m/s)', fontsize=12)
    ax.set_title('Wind Speed - Overall Bias Comparison (Positive=Overestimate)', 
                fontsize=14, fontweight='bold')
    ax.legend(fontsize=12)
    ax.grid(True, alpha=0.3)
    ax.set_xticks(x)
    ax.set_xticklabels(heights, rotation=0)
    
    # 设置y轴范围（对称）
    all_values = np.concatenate([
        overall_df['EC_bias'].dropna().values,
        overall_df['GFS_bias'].dropna().values
    ])
    if len(all_values) > 0:
        y_range = max(abs(all_values)) * 1.2
        ax.set_ylim(-y_range, y_range)
    
    # 添加数值标签
    for i, (ec_val, gfs_val) in enumerate(zip(overall_df['EC_bias'], overall_df['GFS_bias'])):
        if not np.isnan(ec_val):
            y_pos = ec_val + (0.02 * y_range if ec_val >= 0 else -0.02 * y_range)
            ax.text(i - width/2, y_pos, f'{ec_val:.3f}', 
                   ha='center', va='bottom' if ec_val >= 0 else 'top', fontsize=10, fontweight='bold')
        if not np.isnan(gfs_val):
            y_pos = gfs_val + (0.02 * y_range if gfs_val >= 0 else -0.02 * y_range)
            ax.text(i + width/2, y_pos, f'{gfs_val:.3f}', 
                   ha='center', va='bottom' if gfs_val >= 0 else 'top', fontsize=10, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, 'wind_speed_overall_bias.png'), 
                dpi=300, bbox_inches='tight')
    plt.show()
    print("Saved: wind_speed_overall_bias.png")

def plot_seasonal_diurnal_bias(seasonal_df):
    """绘制季节性昼夜循环Bias图 - EC和GFS分别绘制"""
    variables = seasonal_df['variable'].unique()
    seasons = ['Spring', 'Summer', 'Autumn', 'Winter']
    
    # 为EC模型绘制季节对比图
    for model in ['EC', 'GFS']:
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        axes = axes.flatten()
        
        bias_col = f'{model}_bias'
        
        # 计算全局y轴范围
        all_values = []
        for var in variables:
            var_data = seasonal_df[seasonal_df['variable'] == var]
            all_values.extend(var_data[bias_col].dropna().values)
        
        global_y_range = max(abs(np.array(all_values))) * 1.1 if all_values else 1
        
        for season_idx, season in enumerate(seasons):
            ax = axes[season_idx]
            
            # 为每个高度绘制线条
            for var_idx, var in enumerate(variables):
                var_season_data = seasonal_df[
                    (seasonal_df['variable'] == var) & 
                    (seasonal_df['season'] == season)
                ].sort_values('hour')
                
                height = var.split()[-1]
                
                if not var_season_data.empty:
                    ax.plot(var_season_data['hour'], var_season_data[bias_col], 
                           marker='o', linewidth=2.5, markersize=5, 
                           color=colors_heights[var_idx], alpha=0.8,
                           label=height)
            
            # 添加零线
            ax.axhline(y=0, color='black', linestyle='--', alpha=0.7, linewidth=1.5)
            
            # 设置标签和标题
            ax.set_xlabel('Hour', fontsize=12)
            ax.set_ylabel('Bias (m/s)', fontsize=12)
            ax.set_title(f'{season} - {model} Model Wind Speed Bias', 
                        fontsize=13, fontweight='bold')
            ax.legend(fontsize=10, title='Height')
            ax.grid(True, alpha=0.3)
            ax.set_xticks(range(0, 24, 2))
            ax.set_xlim(0, 23)
            ax.set_ylim(-global_y_range, global_y_range)
        
        plt.tight_layout()
        plt.savefig(os.path.join(results_dir, f'wind_speed_{model.lower()}_seasonal_diurnal_bias.png'), 
                    dpi=300, bbox_inches='tight')
        plt.show()
        print(f"Saved: wind_speed_{model.lower()}_seasonal_diurnal_bias.png")

def plot_seasonal_comparison_by_height(seasonal_df):
    """按高度绘制季节对比图"""
    variables = seasonal_df['variable'].unique()
    seasons = ['Spring', 'Summer', 'Autumn', 'Winter']
    
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    axes = axes.flatten()
    
    for var_idx, var in enumerate(variables):
        ax = axes[var_idx]
        var_data = seasonal_df[seasonal_df['variable'] == var]
        height = var.split()[-1]
        
        # 计算这个高度的y轴范围
        all_bias_values = np.concatenate([
            var_data['EC_bias'].dropna().values,
            var_data['GFS_bias'].dropna().values
        ])
        y_range = max(abs(all_bias_values)) * 1.1 if len(all_bias_values) > 0 else 1
        
        # 为每个季节绘制EC和GFS的对比
        x_positions = np.arange(len(seasons))
        width = 0.35
        
        ec_seasonal_means = []
        gfs_seasonal_means = []
        
        for season in seasons:
            season_data = var_data[var_data['season'] == season]
            ec_mean = season_data['EC_bias'].mean()
            gfs_mean = season_data['GFS_bias'].mean()
            ec_seasonal_means.append(ec_mean)
            gfs_seasonal_means.append(gfs_mean)
        
        bars1 = ax.bar(x_positions - width/2, ec_seasonal_means, width, 
                      label='EC Model', color=colors_main[0], alpha=0.8)
        bars2 = ax.bar(x_positions + width/2, gfs_seasonal_means, width, 
                      label='GFS Model', color=colors_main[1], alpha=0.8)
        
        # 添加零线
        ax.axhline(y=0, color='black', linestyle='--', alpha=0.7, linewidth=1)
        
        # 设置标签和标题
        ax.set_xlabel('Season', fontsize=12)
        ax.set_ylabel('Average Bias (m/s)', fontsize=12)
        ax.set_title(f'Wind Speed {height} - Seasonal Average Bias', 
                    fontsize=13, fontweight='bold')
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3)
        ax.set_xticks(x_positions)
        ax.set_xticklabels(seasons)
        ax.set_ylim(-y_range, y_range)
        
        # 添加数值标签
        for i, (ec_val, gfs_val) in enumerate(zip(ec_seasonal_means, gfs_seasonal_means)):
            if not np.isnan(ec_val):
                y_pos = ec_val + (0.02 * y_range if ec_val >= 0 else -0.02 * y_range)
                ax.text(i - width/2, y_pos, f'{ec_val:.3f}', 
                       ha='center', va='bottom' if ec_val >= 0 else 'top', fontsize=9)
            if not np.isnan(gfs_val):
                y_pos = gfs_val + (0.02 * y_range if gfs_val >= 0 else -0.02 * y_range)
                ax.text(i + width/2, y_pos, f'{gfs_val:.3f}', 
                       ha='center', va='bottom' if gfs_val >= 0 else 'top', fontsize=9)
    
    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, 'wind_speed_seasonal_average_bias.png'), 
                dpi=300, bbox_inches='tight')
    plt.show()
    print("Saved: wind_speed_seasonal_average_bias.png")

def print_analysis_summary(overall_df, seasonal_df):
    """打印分析摘要"""
    print("\n" + "="*60)
    print("WIND SPEED BIAS ANALYSIS SUMMARY")
    print("="*60)
    print("Bias = Predicted - Observed")
    print("Positive Bias: Model OVERESTIMATES")
    print("Negative Bias: Model UNDERESTIMATES")
    print("="*60)
    
    print(f"\nOverall Bias by Height:")
    print("-" * 40)
    
    for _, row in overall_df.iterrows():
        print(f"  {row['variable']}:")
        print(f"    EC Model:  {row['EC_bias']:>7.4f} m/s ({row['EC_tendency']})")
        print(f"    GFS Model: {row['GFS_bias']:>7.4f} m/s ({row['GFS_tendency']})")
        print(f"    More accurate: {row['better_model']}")
        print()
    
    # 季节性摘要
    print("\nSeasonal Bias Patterns:")
    print("-" * 40)
    seasons = ['Spring', 'Summer', 'Autumn', 'Winter']
    
    for season in seasons:
        season_data = seasonal_df[seasonal_df['season'] == season]
        ec_mean = season_data['EC_bias'].mean()
        gfs_mean = season_data['GFS_bias'].mean()
        
        print(f"  {season}:")
        print(f"    EC average bias:  {ec_mean:>7.4f} m/s")
        print(f"    GFS average bias: {gfs_mean:>7.4f} m/s")
    
    # 总体统计
    total_vars = len(overall_df)
    ec_better = (overall_df['better_model'] == 'EC').sum()
    gfs_better = (overall_df['better_model'] == 'GFS').sum()
    
    print(f"\n{'='*60}")
    print("OVERALL MODEL PERFORMANCE")
    print(f"{'='*60}")
    print(f"Total heights analyzed: {total_vars}")
    print(f"EC more accurate: {ec_better}/{total_vars} ({ec_better/total_vars*100:.1f}%)")
    print(f"GFS more accurate: {gfs_better}/{total_vars} ({gfs_better/total_vars*100:.1f}%)")

def main():
    """主分析流程"""
    print("="*60)
    print("WIND SPEED DIURNAL BIAS ANALYSIS")
    print("="*60)
    
    # 1. 读取数据
    df = load_and_prepare_data()
    
    # 2. 获取风速变量对
    pairs = get_wind_speed_pairs(df)
    print(f"\nFound {len(pairs)} wind speed variables:")
    for pair in pairs:
        print(f"  - {pair['variable']}")
    
    if not pairs:
        print("No wind speed variables found!")
        return
    
    # 3. 计算各种bias
    print("\nCalculating diurnal bias...")
    diurnal_df = calculate_diurnal_bias(df, pairs)
    
    print("Calculating seasonal diurnal bias...")
    seasonal_df = calculate_seasonal_diurnal_bias(df, pairs)
    
    print("Calculating overall bias...")
    overall_df = calculate_overall_bias(df, pairs)
    
    # 4. 保存数据
    print("Saving results...")
    diurnal_df.to_csv(os.path.join(results_dir, 'wind_speed_diurnal_bias.csv'), index=False)
    seasonal_df.to_csv(os.path.join(results_dir, 'wind_speed_seasonal_diurnal_bias.csv'), index=False)
    overall_df.to_csv(os.path.join(results_dir, 'wind_speed_overall_bias.csv'), index=False)
    
    # 5. 绘制图表
    print("\nGenerating plots...")
    
    print("Plotting overall diurnal bias...")
    plot_diurnal_bias(diurnal_df)
    
    print("Plotting overall bias comparison...")
    plot_overall_bias(overall_df)
    
    print("Plotting seasonal diurnal bias...")
    plot_seasonal_diurnal_bias(seasonal_df)
    
    print("Plotting seasonal comparison by height...")
    plot_seasonal_comparison_by_height(seasonal_df)
    
    # 6. 打印摘要
    print_analysis_summary(overall_df, seasonal_df)
    
    print(f"\n{'='*60}")
    print("ANALYSIS COMPLETED SUCCESSFULLY")
    print(f"All results saved to: {results_dir}")
    print("Generated files:")
    print("  - wind_speed_diurnal_bias.csv")
    print("  - wind_speed_seasonal_diurnal_bias.csv") 
    print("  - wind_speed_overall_bias.csv")
    print("  - wind_speed_diurnal_bias.png")
    print("  - wind_speed_overall_bias.png")
    print("  - wind_speed_ec_seasonal_diurnal_bias.png")
    print("  - wind_speed_gfs_seasonal_diurnal_bias.png")
    print("  - wind_speed_seasonal_average_bias.png")
    print(f"{'='*60}")

if __name__ == "__main__":
    main()