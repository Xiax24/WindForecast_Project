import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
from datetime import datetime

# 设置绘图样式
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['figure.figsize'] = (12, 8)
sns.set_style("whitegrid")

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
    # 去除缺失值
    mask = ~(np.isnan(y_obs) | np.isnan(y_pred))
    if mask.sum() == 0:
        return np.nan
    
    return np.mean(y_pred[mask] - y_obs[mask])

def load_and_prepare_data():
    """读取和预处理数据"""
    print("Loading data...")
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

def get_variable_pairs(df):
    """获取观测-预测变量对"""
    pairs = []
    
    # 风速变量
    for height in ['10m', '30m', '50m', '70m']:
        obs_col = f'obs_wind_speed_{height}'
        ec_col = f'ec_wind_speed_{height}'
        gfs_col = f'gfs_wind_speed_{height}'
        
        if all(col in df.columns for col in [obs_col, ec_col, gfs_col]):
            pairs.append({
                'group': 'Wind Speed',
                'variable': f'Wind Speed {height}',
                'height': height,
                'obs': obs_col,
                'ec': ec_col,
                'gfs': gfs_col,
                'unit': 'm/s'
            })
    
    # 风向变量
    for height in ['10m', '30m', '50m', '70m']:
        obs_col = f'obs_wind_direction_{height}'
        ec_col = f'ec_wind_direction_{height}'
        gfs_col = f'gfs_wind_direction_{height}'
        
        if all(col in df.columns for col in [obs_col, ec_col, gfs_col]):
            pairs.append({
                'group': 'Wind Direction',
                'variable': f'Wind Direction {height}',
                'height': height,
                'obs': obs_col,
                'ec': ec_col,
                'gfs': gfs_col,
                'unit': 'degrees'
            })
    
    # 温度变量
    for height in ['10m']:
        obs_col = f'obs_temperature_{height}'
        ec_col = f'ec_temperature_{height}'
        gfs_col = f'gfs_temperature_{height}'
        
        if all(col in df.columns for col in [obs_col, ec_col, gfs_col]):
            pairs.append({
                'group': 'Temperature',
                'variable': f'Temperature {height}',
                'height': height,
                'obs': obs_col,
                'ec': ec_col,
                'gfs': gfs_col,
                'unit': '°C'
            })
    
    # 密度变量
    for height in ['10m']:
        obs_col = f'obs_density_{height}'
        ec_col = f'ec_density_{height}'
        gfs_col = f'gfs_density_{height}'
        
        if all(col in df.columns for col in [obs_col, ec_col, gfs_col]):
            pairs.append({
                'group': 'Air Density',
                'variable': f'Air Density {height}',
                'height': height,
                'obs': obs_col,
                'ec': ec_col,
                'gfs': gfs_col,
                'unit': 'kg/m³'
            })
    
    return pairs

def calculate_diurnal_bias(df, pairs):
    """计算昼夜循环Bias"""
    results = []
    
    for pair in pairs:
        for hour in range(24):
            hour_data = df[df['hour'] == hour]
            
            # 计算EC和GFS的bias
            ec_bias = calculate_bias(
                hour_data[pair['obs']].values,
                hour_data[pair['ec']].values
            )
            
            gfs_bias = calculate_bias(
                hour_data[pair['obs']].values,
                hour_data[pair['gfs']].values
            )
            
            results.append({
                'group': pair['group'],
                'variable': pair['variable'],
                'height': pair['height'],
                'hour': hour,
                'EC_bias': ec_bias,
                'GFS_bias': gfs_bias,
                'unit': pair['unit']
            })
    
    return pd.DataFrame(results)

def calculate_overall_bias(df, pairs):
    """计算总体Bias"""
    results = []
    
    for pair in pairs:
        # 计算总体bias
        ec_bias = calculate_bias(
            df[pair['obs']].values,
            df[pair['ec']].values
        )
        
        gfs_bias = calculate_bias(
            df[pair['obs']].values,
            df[pair['gfs']].values
        )
        
        # 判断哪个模型更准确（绝对偏差更小）
        ec_abs_bias = abs(ec_bias) if not np.isnan(ec_bias) else np.inf
        gfs_abs_bias = abs(gfs_bias) if not np.isnan(gfs_bias) else np.inf
        better_model = 'EC' if ec_abs_bias < gfs_abs_bias else 'GFS'
        
        # 判断偏差倾向
        ec_tendency = 'Overestimate' if ec_bias > 0 else 'Underestimate' if ec_bias < 0 else 'Neutral'
        gfs_tendency = 'Overestimate' if gfs_bias > 0 else 'Underestimate' if gfs_bias < 0 else 'Neutral'
        
        results.append({
            'group': pair['group'],
            'variable': pair['variable'],
            'height': pair['height'],
            'EC_bias': ec_bias,
            'GFS_bias': gfs_bias,
            'EC_abs_bias': ec_abs_bias,
            'GFS_abs_bias': gfs_abs_bias,
            'better_model': better_model,
            'EC_tendency': ec_tendency,
            'GFS_tendency': gfs_tendency,
            'unit': pair['unit']
        })
    
    return pd.DataFrame(results)

def plot_diurnal_bias_by_group(diurnal_df, group_name):
    """绘制单个变量组的昼夜Bias图"""
    group_data = diurnal_df[diurnal_df['group'] == group_name]
    
    if group_data.empty:
        print(f"No data for group: {group_name}")
        return
    
    variables = group_data['variable'].unique()
    n_vars = len(variables)
    unit = group_data['unit'].iloc[0]
    
    fig, axes = plt.subplots(n_vars, 1, figsize=(14, 4*n_vars))
    if n_vars == 1:
        axes = [axes]
    
    colors = ['#3498db', '#e74c3c']  # Blue for EC, Red for GFS
    
    for i, var in enumerate(variables):
        var_data = group_data[group_data['variable'] == var].sort_values('hour')
        
        x = np.arange(24)
        width = 0.35
        
        # 绘制柱状图
        bars1 = axes[i].bar(x - width/2, var_data['EC_bias'], width, 
                           label='EC Model', color=colors[0], alpha=0.8)
        bars2 = axes[i].bar(x + width/2, var_data['GFS_bias'], width, 
                           label='GFS Model', color=colors[1], alpha=0.8)
        
        # 添加零线
        axes[i].axhline(y=0, color='black', linestyle='--', alpha=0.7, linewidth=1)
        
        # 设置标签和标题
        axes[i].set_xlabel('Hour', fontsize=12)
        axes[i].set_ylabel(f'Bias ({unit})', fontsize=12)
        axes[i].set_title(f'{var} - Diurnal Bias (Positive=Overestimate, Negative=Underestimate)', 
                         fontsize=13, fontweight='bold')
        axes[i].legend(fontsize=10)
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
    filename = f'{group_name.lower().replace(" ", "_")}_diurnal_bias.png'
    plt.savefig(os.path.join(results_dir, filename), dpi=300, bbox_inches='tight')
    plt.show()
    print(f"Saved: {filename}")

def plot_overall_bias_by_group(overall_df, group_name):
    """绘制单个变量组的总体Bias对比图"""
    group_data = overall_df[overall_df['group'] == group_name]
    
    if group_data.empty:
        print(f"No data for group: {group_name}")
        return
    
    unit = group_data['unit'].iloc[0]
    variables = group_data['variable'].values
    heights = [var.split()[-1] for var in variables]  # 提取高度信息
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    x = np.arange(len(variables))
    width = 0.35
    colors = ['#3498db', '#e74c3c']
    
    # 绘制柱状图
    bars1 = ax.bar(x - width/2, group_data['EC_bias'], width, 
                   label='EC Model', color=colors[0], alpha=0.8)
    bars2 = ax.bar(x + width/2, group_data['GFS_bias'], width, 
                   label='GFS Model', color=colors[1], alpha=0.8)
    
    # 添加零线
    ax.axhline(y=0, color='black', linestyle='--', alpha=0.7, linewidth=1)
    
    # 设置标签和标题
    ax.set_xlabel('Variable', fontsize=12)
    ax.set_ylabel(f'Bias ({unit})', fontsize=12)
    ax.set_title(f'{group_name} - Overall Bias Comparison (Positive=Overestimate)', 
                fontsize=14, fontweight='bold')
    ax.legend(fontsize=12)
    ax.grid(True, alpha=0.3)
    ax.set_xticks(x)
    ax.set_xticklabels(heights, rotation=0)
    
    # 设置y轴范围（对称）
    all_values = np.concatenate([
        group_data['EC_bias'].dropna().values,
        group_data['GFS_bias'].dropna().values
    ])
    if len(all_values) > 0:
        y_range = max(abs(all_values)) * 1.2
        ax.set_ylim(-y_range, y_range)
    
    # 添加数值标签
    for i, (ec_val, gfs_val) in enumerate(zip(group_data['EC_bias'], group_data['GFS_bias'])):
        if not np.isnan(ec_val):
            y_pos = ec_val + (0.02 * y_range if ec_val >= 0 else -0.02 * y_range)
            ax.text(i - width/2, y_pos, f'{ec_val:.3f}', 
                   ha='center', va='bottom' if ec_val >= 0 else 'top', fontsize=10)
        if not np.isnan(gfs_val):
            y_pos = gfs_val + (0.02 * y_range if gfs_val >= 0 else -0.02 * y_range)
            ax.text(i + width/2, y_pos, f'{gfs_val:.3f}', 
                   ha='center', va='bottom' if gfs_val >= 0 else 'top', fontsize=10)
    
    plt.tight_layout()
    filename = f'{group_name.lower().replace(" ", "_")}_overall_bias.png'
    plt.savefig(os.path.join(results_dir, filename), dpi=300, bbox_inches='tight')
    plt.show()
    print(f"Saved: {filename}")

def plot_comprehensive_comparison(overall_df):
    """绘制所有变量的综合对比图"""
    groups = overall_df['group'].unique()
    
    fig, axes = plt.subplots(len(groups), 1, figsize=(14, 5*len(groups)))
    if len(groups) == 1:
        axes = [axes]
    
    colors = ['#3498db', '#e74c3c']
    
    for i, group in enumerate(groups):
        group_data = overall_df[overall_df['group'] == group]
        unit = group_data['unit'].iloc[0]
        
        variables = group_data['variable'].values
        heights = [var.split()[-1] for var in variables]
        x = np.arange(len(variables))
        width = 0.35
        
        # 绘制柱状图
        axes[i].bar(x - width/2, group_data['EC_bias'], width, 
                   label='EC Model', color=colors[0], alpha=0.8)
        axes[i].bar(x + width/2, group_data['GFS_bias'], width, 
                   label='GFS Model', color=colors[1], alpha=0.8)
        
        # 添加零线
        axes[i].axhline(y=0, color='black', linestyle='--', alpha=0.7, linewidth=1)
        
        # 设置标签和标题
        axes[i].set_xlabel('Variable', fontsize=12)
        axes[i].set_ylabel(f'Bias ({unit})', fontsize=12)
        axes[i].set_title(f'{group} - Bias Comparison', fontsize=14, fontweight='bold')
        axes[i].legend(fontsize=12)
        axes[i].grid(True, alpha=0.3)
        axes[i].set_xticks(x)
        axes[i].set_xticklabels(heights, rotation=0)
        
        # 设置y轴范围
        all_values = np.concatenate([
            group_data['EC_bias'].dropna().values,
            group_data['GFS_bias'].dropna().values
        ])
        if len(all_values) > 0:
            y_range = max(abs(all_values)) * 1.2
            axes[i].set_ylim(-y_range, y_range)
        
        # 添加数值标签
        for j, (ec_val, gfs_val) in enumerate(zip(group_data['EC_bias'], group_data['GFS_bias'])):
            if not np.isnan(ec_val):
                y_pos = ec_val + (0.02 * y_range if ec_val >= 0 else -0.02 * y_range)
                axes[i].text(j - width/2, y_pos, f'{ec_val:.3f}', 
                           ha='center', va='bottom' if ec_val >= 0 else 'top', fontsize=9)
            if not np.isnan(gfs_val):
                y_pos = gfs_val + (0.02 * y_range if gfs_val >= 0 else -0.02 * y_range)
                axes[i].text(j + width/2, y_pos, f'{gfs_val:.3f}', 
                           ha='center', va='bottom' if gfs_val >= 0 else 'top', fontsize=9)
    
    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, 'comprehensive_bias_comparison.png'), 
                dpi=300, bbox_inches='tight')
    plt.show()
    print("Saved: comprehensive_bias_comparison.png")

def print_analysis_summary(overall_df):
    """打印分析摘要"""
    print("\n" + "="*60)
    print("BIAS ANALYSIS SUMMARY")
    print("="*60)
    print("Bias = Predicted - Observed")
    print("Positive Bias: Model OVERESTIMATES (predicts higher than actual)")
    print("Negative Bias: Model UNDERESTIMATES (predicts lower than actual)")
    print("Lower absolute bias = more accurate model")
    print("="*60)
    
    groups = overall_df['group'].unique()
    
    for group in groups:
        group_data = overall_df[overall_df['group'] == group]
        print(f"\n{group}:")
        print("-" * 40)
        
        for _, row in group_data.iterrows():
            print(f"  {row['variable']}:")
            print(f"    EC Model:  {row['EC_bias']:>7.4f} {row['unit']} ({row['EC_tendency']})")
            print(f"    GFS Model: {row['GFS_bias']:>7.4f} {row['unit']} ({row['GFS_tendency']})")
            print(f"    More accurate: {row['better_model']} (lower absolute bias)")
            print()
    
    # 总体统计
    total_vars = len(overall_df)
    ec_better = (overall_df['better_model'] == 'EC').sum()
    gfs_better = (overall_df['better_model'] == 'GFS').sum()
    
    ec_overestimate = (overall_df['EC_tendency'] == 'Overestimate').sum()
    ec_underestimate = (overall_df['EC_tendency'] == 'Underestimate').sum()
    gfs_overestimate = (overall_df['GFS_tendency'] == 'Overestimate').sum()
    gfs_underestimate = (overall_df['GFS_tendency'] == 'Underestimate').sum()
    
    print("\n" + "="*60)
    print("OVERALL MODEL PERFORMANCE")
    print("="*60)
    print(f"Total variables analyzed: {total_vars}")
    print(f"EC more accurate: {ec_better}/{total_vars} ({ec_better/total_vars*100:.1f}%)")
    print(f"GFS more accurate: {gfs_better}/{total_vars} ({gfs_better/total_vars*100:.1f}%)")
    print()
    print("Bias Tendencies:")
    print(f"  EC Model  - Overestimate: {ec_overestimate}, Underestimate: {ec_underestimate}")
    print(f"  GFS Model - Overestimate: {gfs_overestimate}, Underestimate: {gfs_underestimate}")
    
    # 计算平均绝对偏差
    ec_mean_abs_bias = overall_df['EC_abs_bias'].replace([np.inf, -np.inf], np.nan).mean()
    gfs_mean_abs_bias = overall_df['GFS_abs_bias'].replace([np.inf, -np.inf], np.nan).mean()
    
    print(f"\nAverage Absolute Bias:")
    print(f"  EC Model:  {ec_mean_abs_bias:.4f}")
    print(f"  GFS Model: {gfs_mean_abs_bias:.4f}")

def main():
    """主分析流程"""
    print("="*60)
    print("DIURNAL CYCLE BIAS ANALYSIS")
    print("="*60)
    
    # 1. 读取数据
    df = load_and_prepare_data()
    
    # 2. 获取变量对
    pairs = get_variable_pairs(df)
    print(f"\nFound {len(pairs)} variable pairs for analysis:")
    for pair in pairs:
        print(f"  - {pair['variable']}")
    
    # 3. 计算昼夜循环bias
    print("\nCalculating diurnal bias...")
    diurnal_df = calculate_diurnal_bias(df, pairs)
    
    # 4. 计算总体bias
    print("Calculating overall bias...")
    overall_df = calculate_overall_bias(df, pairs)
    
    # 5. 保存数据
    print("Saving results...")
    diurnal_df.to_csv(os.path.join(results_dir, 'diurnal_bias_data.csv'), index=False)
    overall_df.to_csv(os.path.join(results_dir, 'overall_bias_data.csv'), index=False)
    
    # 6. 绘制图表
    print("\nGenerating plots...")
    groups = overall_df['group'].unique()
    
    for group in groups:
        print(f"\nPlotting {group}...")
        plot_diurnal_bias_by_group(diurnal_df, group)
        plot_overall_bias_by_group(overall_df, group)
    
    print("\nPlotting comprehensive comparison...")
    plot_comprehensive_comparison(overall_df)
    
    # 7. 打印摘要
    print_analysis_summary(overall_df)
    
    print(f"\n{'='*60}")
    print("ANALYSIS COMPLETED SUCCESSFULLY")
    print(f"All results saved to: {results_dir}")
    print("Generated files:")
    print("  - diurnal_bias_data.csv: Hourly bias data")
    print("  - overall_bias_data.csv: Overall bias summary")
    print("  - *_diurnal_bias.png: Diurnal bias charts")
    print("  - *_overall_bias.png: Overall bias charts")
    print("  - comprehensive_bias_comparison.png: All variables comparison")
    print(f"{'='*60}")

if __name__ == "__main__":
    main()