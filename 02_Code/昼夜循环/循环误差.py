import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import os
from sklearn.metrics import mean_squared_error

# 设置绘图样式
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['figure.figsize'] = (12, 8)
sns.set_style("whitegrid")
sns.set_palette("husl")

# 数据和结果路径
data_path ='/Users/xiaxin/work/WindForecast_Project/01_Data/processed/imputed_data/changma_imputed_complete.csv'
results_dir = '/Users/xiaxin/work/WindForecast_Project/03_Results/5diurnal_error/'

# 创建结果目录
os.makedirs(results_dir, exist_ok=True)

# 读取数据
print("Loading data...")
df = pd.read_csv(data_path)
df['datetime'] = pd.to_datetime(df['datetime'])
df['hour'] = df['datetime'].dt.hour
df['season'] = df['datetime'].dt.month.map({
    12: 'Winter', 1: 'Winter', 2: 'Winter',
    3: 'Spring', 4: 'Spring', 5: 'Spring',
    6: 'Summer', 7: 'Summer', 8: 'Summer',
    9: 'Autumn', 10: 'Autumn', 11: 'Autumn'
})

print(f"Data shape: {df.shape}")
print(f"Time range: {df['datetime'].min()} to {df['datetime'].max()}")

def calculate_rmse(y_true, y_pred):
    """计算RMSE"""
    mask = ~(np.isnan(y_true) | np.isnan(y_pred))
    if mask.sum() == 0:
        return np.nan
    return np.sqrt(mean_squared_error(y_true[mask], y_pred[mask]))

def get_variable_pairs():
    """获取观测-预测变量对"""
    # 风速变量对
    wind_pairs = []
    for height in ['10m', '30m', '50m', '70m']:
        obs_col = f'obs_wind_speed_{height}'
        ec_col = f'ec_wind_speed_{height}'
        gfs_col = f'gfs_wind_speed_{height}'
        if all(col in df.columns for col in [obs_col, ec_col, gfs_col]):
            wind_pairs.append({
                'variable': f'Wind Speed {height}',
                'obs': obs_col,
                'ec': ec_col,
                'gfs': gfs_col,
                'unit': 'm/s'
            })
    
    # 风向变量对
    direction_pairs = []
    for height in ['10m', '30m', '50m', '70m']:
        obs_col = f'obs_wind_direction_{height}'
        ec_col = f'ec_wind_direction_{height}'
        gfs_col = f'gfs_wind_direction_{height}'
        if all(col in df.columns for col in [obs_col, ec_col, gfs_col]):
            direction_pairs.append({
                'variable': f'Wind Direction {height}',
                'obs': obs_col,
                'ec': ec_col,
                'gfs': gfs_col,
                'unit': '°'
            })
    
    # 温度变量对
    temp_pairs = []
    for height in ['10m']:
        obs_col = f'obs_temperature_{height}'
        ec_col = f'ec_temperature_{height}'
        gfs_col = f'gfs_temperature_{height}'
        if all(col in df.columns for col in [obs_col, ec_col, gfs_col]):
            temp_pairs.append({
                'variable': f'Temperature {height}',
                'obs': obs_col,
                'ec': ec_col,
                'gfs': gfs_col,
                'unit': '°C'
            })
    
    # 密度变量对
    density_pairs = []
    for height in ['10m']:
        obs_col = f'obs_density_{height}'
        ec_col = f'ec_density_{height}'
        gfs_col = f'gfs_density_{height}'
        if all(col in df.columns for col in [obs_col, ec_col, gfs_col]):
            density_pairs.append({
                'variable': f'Air Density {height}',
                'obs': obs_col,
                'ec': ec_col,
                'gfs': gfs_col,
                'unit': 'kg/m³'
            })
    
    return wind_pairs, direction_pairs, temp_pairs, density_pairs

def calculate_diurnal_rmse(df, var_pairs, var_type):
    """计算昼夜循环的RMSE"""
    results = []
    
    for pair in var_pairs:
        for hour in range(24):
            hour_data = df[df['hour'] == hour]
            
            # EC模型RMSE
            ec_rmse = calculate_rmse(
                hour_data[pair['obs']].values,
                hour_data[pair['ec']].values
            )
            
            # GFS模型RMSE
            gfs_rmse = calculate_rmse(
                hour_data[pair['obs']].values,
                hour_data[pair['gfs']].values
            )
            
            results.append({
                'variable': pair['variable'],
                'hour': hour,
                'EC_RMSE': ec_rmse,
                'GFS_RMSE': gfs_rmse,
                'unit': pair['unit'],
                'var_type': var_type
            })
    
    return pd.DataFrame(results)

def calculate_seasonal_rmse(df, var_pairs, var_type):
    """计算季节性RMSE"""
    results = []
    
    for pair in var_pairs:
        for season in ['Spring', 'Summer', 'Autumn', 'Winter']:
            season_data = df[df['season'] == season]
            
            # EC模型RMSE
            ec_rmse = calculate_rmse(
                season_data[pair['obs']].values,
                season_data[pair['ec']].values
            )
            
            # GFS模型RMSE
            gfs_rmse = calculate_rmse(
                season_data[pair['obs']].values,
                season_data[pair['gfs']].values
            )
            
            results.append({
                'variable': pair['variable'],
                'season': season,
                'EC_RMSE': ec_rmse,
                'GFS_RMSE': gfs_rmse,
                'unit': pair['unit'],
                'var_type': var_type
            })
    
    return pd.DataFrame(results)

def plot_diurnal_rmse_bars(rmse_df, var_type, unit):
    """绘制昼夜循环RMSE柱状图"""
    variables = rmse_df['variable'].unique()
    n_vars = len(variables)
    
    fig, axes = plt.subplots(n_vars, 1, figsize=(16, 4*n_vars))
    if n_vars == 1:
        axes = [axes]
    
    for i, var in enumerate(variables):
        var_data = rmse_df[rmse_df['variable'] == var]
        
        x = np.arange(24)
        width = 0.35
        
        axes[i].bar(x - width/2, var_data['EC_RMSE'], width, 
                   label='EC Model', color='#3498db', alpha=0.8)
        axes[i].bar(x + width/2, var_data['GFS_RMSE'], width, 
                   label='GFS Model', color='#e74c3c', alpha=0.8)
        
        axes[i].set_xlabel('Hour', fontsize=12)
        axes[i].set_ylabel(f'RMSE ({unit})', fontsize=12)
        axes[i].set_title(f'{var} - Diurnal RMSE Variation', fontsize=14, fontweight='bold')
        axes[i].legend(fontsize=10)
        axes[i].grid(True, alpha=0.3)
        axes[i].set_xticks(range(0, 24, 2))
        axes[i].set_xlim(-0.5, 23.5)
        
        # 添加数值标签（可选，防止图表过于拥挤可以注释掉）
        # for j, (ec_val, gfs_val) in enumerate(zip(var_data['EC_RMSE'], var_data['GFS_RMSE'])):
        #     if not np.isnan(ec_val):
        #         axes[i].text(j - width/2, ec_val + max(var_data['EC_RMSE'])*0.01, 
        #                     f'{ec_val:.2f}', ha='center', va='bottom', fontsize=8)
        #     if not np.isnan(gfs_val):
        #         axes[i].text(j + width/2, gfs_val + max(var_data['GFS_RMSE'])*0.01, 
        #                     f'{gfs_val:.2f}', ha='center', va='bottom', fontsize=8)
    
    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, f'{var_type.lower()}_diurnal_rmse.png'), 
                dpi=300, bbox_inches='tight')
    plt.show()

def plot_seasonal_rmse_bars(rmse_df, var_type, unit):
    """绘制季节性RMSE柱状图"""
    variables = rmse_df['variable'].unique()
    n_vars = len(variables)
    
    fig, axes = plt.subplots(n_vars, 1, figsize=(12, 4*n_vars))
    if n_vars == 1:
        axes = [axes]
    
    seasons = ['Spring', 'Summer', 'Autumn', 'Winter']
    
    for i, var in enumerate(variables):
        var_data = rmse_df[rmse_df['variable'] == var]
        
        x = np.arange(len(seasons))
        width = 0.35
        
        ec_values = [var_data[var_data['season'] == s]['EC_RMSE'].iloc[0] for s in seasons]
        gfs_values = [var_data[var_data['season'] == s]['GFS_RMSE'].iloc[0] for s in seasons]
        
        axes[i].bar(x - width/2, ec_values, width, 
                   label='EC Model', color='#3498db', alpha=0.8)
        axes[i].bar(x + width/2, gfs_values, width, 
                   label='GFS Model', color='#e74c3c', alpha=0.8)
        
        axes[i].set_xlabel('Season', fontsize=12)
        axes[i].set_ylabel(f'RMSE ({unit})', fontsize=12)
        axes[i].set_title(f'{var} - Seasonal RMSE Comparison', fontsize=14, fontweight='bold')
        axes[i].legend(fontsize=10)
        axes[i].grid(True, alpha=0.3)
        axes[i].set_xticks(x)
        axes[i].set_xticklabels(seasons)
        
        # 添加数值标签
        for j, (ec_val, gfs_val) in enumerate(zip(ec_values, gfs_values)):
            if not np.isnan(ec_val):
                axes[i].text(j - width/2, ec_val + max(ec_values + gfs_values)*0.01, 
                            f'{ec_val:.2f}', ha='center', va='bottom', fontsize=10)
            if not np.isnan(gfs_val):
                axes[i].text(j + width/2, gfs_val + max(ec_values + gfs_values)*0.01, 
                            f'{gfs_val:.2f}', ha='center', va='bottom', fontsize=10)
    
    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, f'{var_type.lower()}_seasonal_rmse.png'), 
                dpi=300, bbox_inches='tight')
    plt.show()

def plot_overall_rmse_comparison(all_results):
    """绘制总体RMSE对比图"""
    # 计算总体RMSE
    overall_results = []
    
    for var_type, (pairs, unit) in [
        ('Wind Speed', (wind_pairs, 'm/s')),
        ('Wind Direction', (direction_pairs, '°')),
        ('Temperature', (temp_pairs, '°C')),
        ('Air Density', (density_pairs, 'kg/m³'))
    ]:
        if not pairs:
            continue
            
        for pair in pairs:
            # 计算总体RMSE
            ec_rmse = calculate_rmse(df[pair['obs']].values, df[pair['ec']].values)
            gfs_rmse = calculate_rmse(df[pair['obs']].values, df[pair['gfs']].values)
            
            overall_results.append({
                'Variable': pair['variable'],
                'EC_RMSE': ec_rmse,
                'GFS_RMSE': gfs_rmse,
                'Type': var_type,
                'Unit': unit
            })
    
    overall_df = pd.DataFrame(overall_results)
    
    # 按变量类型分组绘图
    var_types = overall_df['Type'].unique()
    
    for var_type in var_types:
        type_data = overall_df[overall_df['Type'] == var_type]
        
        fig, ax = plt.subplots(figsize=(12, 6))
        
        variables = type_data['Variable'].values
        x = np.arange(len(variables))
        width = 0.35
        
        ax.bar(x - width/2, type_data['EC_RMSE'], width, 
               label='EC Model', color='#3498db', alpha=0.8)
        ax.bar(x + width/2, type_data['GFS_RMSE'], width, 
               label='GFS Model', color='#e74c3c', alpha=0.8)
        
        ax.set_xlabel('Variable', fontsize=12)
        ax.set_ylabel(f'RMSE ({type_data.iloc[0]["Unit"]})', fontsize=12)
        ax.set_title(f'{var_type} - Overall RMSE Comparison', fontsize=14, fontweight='bold')
        ax.legend(fontsize=12)
        ax.grid(True, alpha=0.3)
        ax.set_xticks(x)
        ax.set_xticklabels(variables, rotation=45, ha='right')
        
        # 添加数值标签
        for i, (ec_val, gfs_val) in enumerate(zip(type_data['EC_RMSE'], type_data['GFS_RMSE'])):
            if not np.isnan(ec_val):
                ax.text(i - width/2, ec_val + max(type_data['EC_RMSE'])*0.01, 
                       f'{ec_val:.3f}', ha='center', va='bottom', fontsize=10)
            if not np.isnan(gfs_val):
                ax.text(i + width/2, gfs_val + max(type_data['GFS_RMSE'])*0.01, 
                       f'{gfs_val:.3f}', ha='center', va='bottom', fontsize=10)
        
        plt.tight_layout()
        plt.savefig(os.path.join(results_dir, f'{var_type.lower().replace(" ", "_")}_overall_rmse.png'), 
                    dpi=300, bbox_inches='tight')
        plt.show()
    
    return overall_df

# 主分析流程
def main():
    print("\n=== Starting Diurnal Error Analysis ===")
    
    # 获取变量对
    global wind_pairs, direction_pairs, temp_pairs, density_pairs
    wind_pairs, direction_pairs, temp_pairs, density_pairs = get_variable_pairs()
    
    print(f"Found {len(wind_pairs)} wind speed pairs")
    print(f"Found {len(direction_pairs)} wind direction pairs") 
    print(f"Found {len(temp_pairs)} temperature pairs")
    print(f"Found {len(density_pairs)} density pairs")
    
    all_results = {}
    
    # 分析各类变量
    for var_type, pairs in [
        ('Wind_Speed', wind_pairs),
        ('Wind_Direction', direction_pairs), 
        ('Temperature', temp_pairs),
        ('Air_Density', density_pairs)
    ]:
        if not pairs:
            print(f"\nSkipping {var_type} - no valid pairs found")
            continue
            
        print(f"\n=== Analyzing {var_type} ===")
        
        # 获取单位
        unit = pairs[0]['unit']
        
        # 计算昼夜循环RMSE
        diurnal_rmse = calculate_diurnal_rmse(df, pairs, var_type)
        plot_diurnal_rmse_bars(diurnal_rmse, var_type, unit)
        
        # 计算季节性RMSE
        seasonal_rmse = calculate_seasonal_rmse(df, pairs, var_type)
        plot_seasonal_rmse_bars(seasonal_rmse, var_type, unit)
        
        # 保存结果
        diurnal_rmse.to_csv(os.path.join(results_dir, f'{var_type.lower()}_diurnal_rmse.csv'), index=False)
        seasonal_rmse.to_csv(os.path.join(results_dir, f'{var_type.lower()}_seasonal_rmse.csv'), index=False)
        
        all_results[var_type] = {
            'diurnal': diurnal_rmse,
            'seasonal': seasonal_rmse,
            'pairs': pairs,
            'unit': unit
        }
    
    # 绘制总体对比图
    print("\n=== Creating Overall Comparison ===")
    overall_df = plot_overall_rmse_comparison(all_results)
    overall_df.to_csv(os.path.join(results_dir, 'overall_rmse_comparison.csv'), index=False)
    
    print(f"\n=== Analysis Completed ===")
    print(f"All results saved to: {results_dir}")
    
    # 打印一些关键发现
    print("\n=== Key Findings ===")
    for var_type, results in all_results.items():
        if results['pairs']:
            print(f"\n{var_type}:")
            for pair in results['pairs']:
                overall_ec = calculate_rmse(df[pair['obs']].values, df[pair['ec']].values)
                overall_gfs = calculate_rmse(df[pair['obs']].values, df[pair['gfs']].values)
                
                better_model = "EC" if overall_ec < overall_gfs else "GFS"
                improvement = abs(overall_ec - overall_gfs) / max(overall_ec, overall_gfs) * 100
                
                print(f"  {pair['variable']}: EC={overall_ec:.3f}, GFS={overall_gfs:.3f} {pair['unit']}")
                print(f"    -> {better_model} performs {improvement:.1f}% better")

if __name__ == "__main__":
    main()