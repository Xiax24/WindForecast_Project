import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import warnings
import os
from pathlib import Path

warnings.filterwarnings('ignore')

class ObsWRFMatcher:
    """观测数据与WRF模拟数据完整匹配分析系统 - 修正版"""
    
    def __init__(self, project_root="/Users/xiaxin/work/WindForecast_Project"):
        self.project_root = Path(project_root)
        self.data_base_path = self.project_root / "01_Data"
        
        # 基于实际观测数据结构的站点配置
        self.station_configs = {
            'changma': {
                'obs_file': 'raw/processed/cleaned-15min/changma_20210501_20221031_15min.csv',
                'ec_wrf_file': 'raw/wrf/ec_driven/changma.csv',
                'gfs_wrf_file': 'raw/wrf/gfs_driven/changma.csv',
                'time_range': ('2021-05-01', '2022-10-31'),
                'obs_variables': {
                    'wind_speed_10m': 'wind_speed_10m',
                    'wind_speed_30m': 'wind_speed_30m', 
                    'wind_speed_50m': 'wind_speed_50m',
                    'wind_speed_70m': 'wind_speed_70m',
                    'wind_direction_10m': 'wind_direction_10m',
                    'wind_direction_30m': 'wind_direction_30m',
                    'wind_direction_50m': 'wind_direction_50m', 
                    'wind_direction_70m': 'wind_direction_70m',
                    'temperature_10m': 'temperature_10m',
                    'humidity_10m': 'humidity_10m',
                    'density_10m': 'density_10m'
                }
            },
            'kuangqu': {
                'obs_file': 'raw/processed/cleaned-15min/kuangqu_20210501_20220601_15min.csv',
                'ec_wrf_file': 'raw/wrf/ec_driven/kuangqu.csv',
                'gfs_wrf_file': 'raw/wrf/gfs_driven/kuangqu.csv',
                'time_range': ('2021-05-01', '2022-06-01'),
                'obs_variables': {
                    'wind_speed_30m': 'wind_speed_30m',
                    'wind_speed_50m': 'wind_speed_50m',
                    'wind_speed_70m': 'wind_speed_70m',
                    'wind_direction_30m': 'wind_direction_30m',
                    'wind_direction_50m': 'wind_direction_50m',
                    'wind_direction_70m': 'wind_direction_70m'
                }
            },
            'sanlijijingzi': {
                'obs_file': 'raw/processed/cleaned-15min/sanlijijingzi_20210601_20220616_15min.csv',
                'ec_wrf_file': 'raw/wrf/ec_driven/sanlijijingzi.csv', 
                'gfs_wrf_file': 'raw/wrf/gfs_driven/sanlijijingzi.csv',
                'time_range': ('2021-06-01', '2022-06-16'),
                'obs_variables': {
                    'wind_speed_10m': 'wind_speed_10m',
                    'wind_speed_30m': 'wind_speed_30m',
                    'wind_speed_50m': 'wind_speed_50m', 
                    'wind_speed_70m': 'wind_speed_70m',
                    'wind_direction_10m': 'wind_direction_10m',
                    'wind_direction_30m': 'wind_direction_30m',
                    'wind_direction_50m': 'wind_direction_50m',
                    'wind_direction_70m': 'wind_direction_70m',
                    'wind_speed_max_10m': 'wind_speed_max_10m',
                    'wind_speed_max_30m': 'wind_speed_max_30m',
                    'wind_speed_max_50m': 'wind_speed_max_50m',
                    'wind_speed_max_70m': 'wind_speed_max_70m',
                    'wind_speed_std_10m': 'wind_speed_std_10m',
                    'wind_speed_std_30m': 'wind_speed_std_30m',
                    'wind_speed_std_50m': 'wind_speed_std_50m',
                    'wind_speed_std_70m': 'wind_speed_std_70m'
                }
            }
        }
        
        # WRF变量映射和计算规则 - 修正版
        self.wrf_mapping_rules = {
            'wind_speed_10m': {'wrf_var': 'wind_speed_10m', 'calc_method': 'direct'},
            'wind_speed_30m': {'wrf_var': 'ws_30m', 'calc_method': 'direct'},
            'wind_speed_50m': {'wrf_var': 'ws_50m', 'calc_method': 'direct'}, 
            'wind_speed_70m': {'wrf_var': 'ws_70m', 'calc_method': 'direct'},
            'wind_direction_10m': {'wrf_var': 'wind_direction_10m', 'calc_method': 'direct'},
            'wind_direction_70m': {'wrf_var': 'wind_direction_70m', 'calc_method': 'direct'},
            'wind_direction_30m': {'wrf_var': ['u_30m', 'v_30m'], 'calc_method': 'wind_direction'},
            'wind_direction_50m': {'wrf_var': ['u_50m', 'v_50m'], 'calc_method': 'wind_direction'},
            'temperature_10m': {'wrf_var': 'tk_30m', 'calc_method': 'temp_convert_and_extrapolate'},
            'humidity_10m': {'wrf_var': None, 'calc_method': 'nan'},
            'density_10m': {'wrf_var': ['tk_30m', 'p_30m'], 'calc_method': 'density_calculate'},
            'wind_speed_max_10m': {'wrf_var': None, 'calc_method': 'nan'},
            'wind_speed_max_30m': {'wrf_var': None, 'calc_method': 'nan'},
            'wind_speed_max_50m': {'wrf_var': None, 'calc_method': 'nan'},
            'wind_speed_max_70m': {'wrf_var': None, 'calc_method': 'nan'},
            'wind_speed_std_10m': {'wrf_var': None, 'calc_method': 'nan'},
            'wind_speed_std_30m': {'wrf_var': None, 'calc_method': 'nan'},
            'wind_speed_std_50m': {'wrf_var': None, 'calc_method': 'nan'},
            'wind_speed_std_70m': {'wrf_var': None, 'calc_method': 'nan'}
        }
        
        self.matched_datasets = {}
        self.R_specific = 287.05  # J/(kg·K) 干空气比气体常数
    
    def load_observation_data(self, station_name):
        """加载观测数据"""
        print(f"正在加载 {station_name} 观测数据...")
        
        obs_file = self.data_base_path / self.station_configs[station_name]['obs_file']
        
        try:
            obs_df = pd.read_csv(obs_file)
            obs_df['datetime'] = pd.to_datetime(obs_df['datetime'])
            obs_df.set_index('datetime', inplace=True)
            
            config_vars = self.station_configs[station_name]['obs_variables']
            available_vars = []
            missing_vars = []
            
            for var_name, col_name in config_vars.items():
                if col_name in obs_df.columns:
                    available_vars.append(var_name)
                else:
                    missing_vars.append(var_name)
            
            print(f"  - 观测数据加载成功: {obs_df.shape[0]} 个时间点")
            print(f"  - 时间范围: {obs_df.index.min()} 到 {obs_df.index.max()}")
            print(f"  - 可用变量: {len(available_vars)} 个")
            if missing_vars:
                print(f"  - 缺失变量: {len(missing_vars)} 个: {missing_vars}")
            
            return obs_df
            
        except Exception as e:
            print(f"  - 观测数据加载失败: {e}")
            return None
    
    def load_wrf_data(self, station_name, model_type='ec'):
        """加载WRF数据"""
        print(f"正在加载 {station_name} {model_type.upper()}-WRF数据...")
        
        wrf_file_key = f'{model_type}_wrf_file'
        wrf_file = self.data_base_path / self.station_configs[station_name][wrf_file_key]
        
        try:
            wrf_df = pd.read_csv(wrf_file)
            
            print(f"  - 原始WRF数据形状: {wrf_df.shape}")
            
            possible_time_cols = ['datetime', 'date', 'time', 'Date', 'DateTime']
            time_column = None
            
            for col in possible_time_cols:
                if col in wrf_df.columns:
                    time_column = col
                    break
            
            if time_column is None:
                print(f"  - ❌ 在WRF数据中找不到时间列!")
                return None
            
            print(f"  - 使用时间列: {time_column}")
            
            wrf_df['datetime'] = pd.to_datetime(wrf_df[time_column])
            wrf_df.set_index('datetime', inplace=True)
            
            if time_column != 'datetime' and time_column in wrf_df.columns:
                wrf_df.drop(columns=[time_column], inplace=True)
            
            print(f"  - {model_type.upper()}-WRF数据加载成功: {wrf_df.shape[0]} 个时间点")
            print(f"  - 时间范围: {wrf_df.index.min()} 到 {wrf_df.index.max()}")
            
            return wrf_df
            
        except Exception as e:
            print(f"  - {model_type.upper()}-WRF数据加载失败: {e}")
            import traceback
            traceback.print_exc()
            return None
    
    def calculate_wind_direction(self, u_component, v_component):
        """从u, v分量计算风向"""
        wind_dir = (270 - np.arctan2(v_component, u_component) * 180 / np.pi) % 360
        return wind_dir
    
    def calculate_density_from_temp_pressure(self, temperature_k, pressure_pa):
        """根据理想气体定律计算空气密度 ρ = P/(R*T)"""
        density = pressure_pa / (self.R_specific * temperature_k)
        return density
    
    def convert_kelvin_to_celsius(self, temp_kelvin):
        """开尔文转摄氏度"""
        return temp_kelvin - 273.15
    
    def extrapolate_temp_to_10m(self, temp_30m_celsius):
        """从30m温度外推10m温度"""
        lapse_rate = 0.0065  # K/m
        temp_10m = temp_30m_celsius + lapse_rate * (30 - 10)
        return temp_10m
    
    def calculate_wrf_variable(self, wrf_df, obs_var_name):
        """根据规则计算WRF中的对应变量 - 修正版"""
        
        if obs_var_name not in self.wrf_mapping_rules:
            print(f"    ! 未知变量 {obs_var_name}，设为 NaN")
            return np.full(len(wrf_df), np.nan)
        
        rule = self.wrf_mapping_rules[obs_var_name]
        calc_method = rule['calc_method']
        wrf_var = rule['wrf_var']
        
        if calc_method == 'direct':
            if wrf_var in wrf_df.columns:
                return wrf_df[wrf_var]
            else:
                print(f"    ! WRF变量 {wrf_var} 不存在，设为 NaN")
                return np.full(len(wrf_df), np.nan)
        
        elif calc_method == 'wind_direction':
            u_var, v_var = wrf_var
            if u_var in wrf_df.columns and v_var in wrf_df.columns:
                calculated = self.calculate_wind_direction(wrf_df[u_var], wrf_df[v_var])
                print(f"    ✓ 从 {u_var}, {v_var} 计算了 {obs_var_name}")
                return calculated
            else:
                print(f"    ! u/v分量 {u_var}, {v_var} 不存在，设为 NaN")
                return np.full(len(wrf_df), np.nan)
        
        elif calc_method == 'temp_convert_and_extrapolate':
            if wrf_var in wrf_df.columns:
                temp_celsius_30m = self.convert_kelvin_to_celsius(wrf_df[wrf_var])
                temp_celsius_10m = self.extrapolate_temp_to_10m(temp_celsius_30m)
                print(f"    ✓ 从 {wrf_var} 转换并外推了 {obs_var_name}")
                return temp_celsius_10m
            else:
                print(f"    ! 用于温度计算的变量 {wrf_var} 不存在，设为 NaN")
                return np.full(len(wrf_df), np.nan)
        
        elif calc_method == 'density_calculate':
            temp_var, pres_var = wrf_var
            if temp_var in wrf_df.columns and pres_var in wrf_df.columns:
                temp_kelvin = wrf_df[temp_var]
                pressure = wrf_df[pres_var]
                
                mean_pressure = pressure.mean()
                if mean_pressure > 50000:
                    pressure_pa = pressure
                elif 500 < mean_pressure < 1200:
                    pressure_pa = pressure * 100
                else:
                    print(f"    ! 气压单位未知 (均值: {mean_pressure:.1f})，设为 NaN")
                    return np.full(len(wrf_df), np.nan)
                
                calculated_density = self.calculate_density_from_temp_pressure(temp_kelvin, pressure_pa)
                
                mean_density = calculated_density.mean()
                if 0.5 < mean_density < 2.0:
                    print(f"    ✓ 从 {temp_var}, {pres_var} 计算了 {obs_var_name} (均值: {mean_density:.3f} kg/m³)")
                    return calculated_density
                else:
                    print(f"    ! 密度计算结果异常 (均值: {mean_density:.3f})，设为 NaN")
                    return np.full(len(wrf_df), np.nan)
            else:
                print(f"    ! 温度/气压变量 {temp_var}, {pres_var} 不存在，设为 NaN")
                return np.full(len(wrf_df), np.nan)
        
        elif calc_method == 'nan':
            print(f"    - {obs_var_name} 设为 NaN (WRF无法提供)")
            return np.full(len(wrf_df), np.nan)
        
        else:
            print(f"    ! 未知计算方法 {calc_method}，设为 NaN")
            return np.full(len(wrf_df), np.nan)
    
    def align_time_series(self, obs_df, ec_wrf_df, gfs_wrf_df):
        """对齐三个数据集的时间序列"""
        print("  - 正在对齐时间序列...")
        
        common_start = max(obs_df.index.min(), ec_wrf_df.index.min(), gfs_wrf_df.index.min())
        common_end = min(obs_df.index.max(), ec_wrf_df.index.max(), gfs_wrf_df.index.max())
        
        print(f"    共同时间范围: {common_start} 到 {common_end}")
        
        obs_aligned = obs_df.loc[common_start:common_end]
        ec_wrf_aligned = ec_wrf_df.loc[common_start:common_end]
        gfs_wrf_aligned = gfs_wrf_df.loc[common_start:common_end]
        
        common_times = obs_aligned.index.intersection(ec_wrf_aligned.index).intersection(gfs_wrf_aligned.index)
        
        print(f"    共同时间点数量: {len(common_times)}")
        
        return (obs_aligned.loc[common_times], 
                ec_wrf_aligned.loc[common_times], 
                gfs_wrf_aligned.loc[common_times])
    
    def create_matched_dataset(self, station_name):
        """为指定站点创建匹配的数据集"""
        print(f"\n=== 处理 {station_name} 站点 ===")
        
        obs_df = self.load_observation_data(station_name)
        if obs_df is None:
            return None
            
        ec_wrf_df = self.load_wrf_data(station_name, 'ec')
        if ec_wrf_df is None:
            return None
            
        gfs_wrf_df = self.load_wrf_data(station_name, 'gfs')
        if gfs_wrf_df is None:
            return None
        
        obs_aligned, ec_aligned, gfs_aligned = self.align_time_series(obs_df, ec_wrf_df, gfs_wrf_df)
        
        matched_data = pd.DataFrame(index=obs_aligned.index)
        
        print("\n  计算WRF对应变量:")
        
        obs_vars = self.station_configs[station_name]['obs_variables']
        
        for obs_var_name, obs_col_name in obs_vars.items():
            
            if obs_col_name in obs_aligned.columns:
                matched_data[f'obs_{obs_var_name}'] = obs_aligned[obs_col_name]
            else:
                matched_data[f'obs_{obs_var_name}'] = np.nan
                print(f"    ! 观测变量 {obs_col_name} 不存在")
            
            print(f"  处理 {obs_var_name} (EC-WRF):")
            ec_values = self.calculate_wrf_variable(ec_aligned, obs_var_name)
            matched_data[f'ec_{obs_var_name}'] = ec_values
            
            print(f"  处理 {obs_var_name} (GFS-WRF):")
            gfs_values = self.calculate_wrf_variable(gfs_aligned, obs_var_name)
            matched_data[f'gfs_{obs_var_name}'] = gfs_values
        
        print(f"\n  - 匹配数据集创建完成: {matched_data.shape}")
        print(f"  - 变量数量: {matched_data.shape[1]} 个")
        
        self.print_data_quality_summary(matched_data, station_name)
        
        return matched_data
    
    def print_data_quality_summary(self, matched_data, station_name):
        """输出数据质量摘要"""
        print(f"\n  --- {station_name} 数据质量摘要 ---")
        
        obs_vars = [col for col in matched_data.columns if col.startswith('obs_')]
        
        for obs_var in obs_vars:
            var_name = obs_var.replace('obs_', '')
            ec_var = f'ec_{var_name}'
            gfs_var = f'gfs_{var_name}'
            
            obs_valid = matched_data[obs_var].notna().sum()
            ec_valid = matched_data[ec_var].notna().sum()
            gfs_valid = matched_data[gfs_var].notna().sum()
            total = len(matched_data)
            
            all_valid = matched_data[[obs_var, ec_var, gfs_var]].notna().all(axis=1).sum()
            
            print(f"  {var_name}:")
            print(f"    OBS: {obs_valid}/{total} ({obs_valid/total*100:.1f}%)")
            print(f"    EC:  {ec_valid}/{total} ({ec_valid/total*100:.1f}%)")
            print(f"    GFS: {gfs_valid}/{total} ({gfs_valid/total*100:.1f}%)")
            print(f"    全部有效: {all_valid}/{total} ({all_valid/total*100:.1f}%)")
    
    def calculate_basic_statistics(self, matched_data, station_name):
        """计算基本统计指标"""
        print(f"\n--- {station_name} 统计指标 ---")
        
        stats_results = {}
        obs_vars = [col for col in matched_data.columns if col.startswith('obs_')]
        
        for obs_var in obs_vars:
            var_name = obs_var.replace('obs_', '')
            ec_var = f'ec_{var_name}'
            gfs_var = f'gfs_{var_name}'
            
            if ec_var in matched_data.columns and gfs_var in matched_data.columns:
                mask = matched_data[[obs_var, ec_var, gfs_var]].notna().all(axis=1)
                
                if mask.sum() > 10:
                    obs_valid = matched_data.loc[mask, obs_var]
                    ec_valid = matched_data.loc[mask, ec_var]
                    gfs_valid = matched_data.loc[mask, gfs_var]
                    
                    ec_bias = np.mean(ec_valid - obs_valid)
                    gfs_bias = np.mean(gfs_valid - obs_valid)
                    
                    ec_rmse = np.sqrt(np.mean((ec_valid - obs_valid)**2))
                    gfs_rmse = np.sqrt(np.mean((gfs_valid - obs_valid)**2))
                    
                    ec_mae = np.mean(np.abs(ec_valid - obs_valid))
                    gfs_mae = np.mean(np.abs(gfs_valid - obs_valid))
                    
                    ec_corr = np.corrcoef(obs_valid, ec_valid)[0, 1] if len(obs_valid) > 1 else np.nan
                    gfs_corr = np.corrcoef(obs_valid, gfs_valid)[0, 1] if len(obs_valid) > 1 else np.nan
                    
                    better_model = 'EC' if ec_rmse < gfs_rmse else 'GFS'
                    improvement = abs(ec_rmse - gfs_rmse) / max(ec_rmse, gfs_rmse) * 100
                    
                    stats_results[var_name] = {
                        'n_samples': mask.sum(),
                        'ec_bias': ec_bias,
                        'gfs_bias': gfs_bias,
                        'ec_rmse': ec_rmse,
                        'gfs_rmse': gfs_rmse,
                        'ec_mae': ec_mae,
                        'gfs_mae': gfs_mae,
                        'ec_corr': ec_corr,
                        'gfs_corr': gfs_corr,
                        'better_model': better_model,
                        'improvement_pct': improvement
                    }
                    
                    print(f"{var_name} ({mask.sum()} 样本):")
                    print(f"  EC  - BIAS: {ec_bias:6.3f}, RMSE: {ec_rmse:6.3f}, MAE: {ec_mae:6.3f}, CORR: {ec_corr:6.3f}")
                    print(f"  GFS - BIAS: {gfs_bias:6.3f}, RMSE: {gfs_rmse:6.3f}, MAE: {gfs_mae:6.3f}, CORR: {gfs_corr:6.3f}")
                    print(f"  更优: {better_model} (改善 {improvement:.1f}%)")
                    print()
                else:
                    print(f"{var_name}: 有效样本不足 ({mask.sum()} < 10)")
        
        return stats_results
    
    def create_validation_plots(self, matched_data, station_name, output_dir=None):
        """创建验证图表"""
        if output_dir is None:
            output_dir = self.project_root / "03_Results" / "validation_plots"
        else:
            output_dir = Path(output_dir)
            
        output_dir.mkdir(parents=True, exist_ok=True)
        
        wind_speed_vars = [col for col in matched_data.columns 
                          if col.startswith('obs_wind_speed') and 'std' not in col and 'max' not in col]
        
        n_vars = len(wind_speed_vars)
        if n_vars == 0:
            print("没有找到风速变量用于可视化")
            return
        
        fig, axes = plt.subplots(n_vars, 3, figsize=(18, 6*n_vars))
        if n_vars == 1:
            axes = axes.reshape(1, -1)
        
        for i, obs_var in enumerate(wind_speed_vars):
            var_name = obs_var.replace('obs_', '')
            ec_var = f'ec_{var_name}'
            gfs_var = f'gfs_{var_name}'
            
            mask = matched_data[[obs_var, ec_var, gfs_var]].notna().all(axis=1)
            
            if mask.sum() > 10:
                plot_data = matched_data.loc[mask]
                
                obs_plot = plot_data[obs_var]
                ec_plot = plot_data[ec_var]
                gfs_plot = plot_data[gfs_var]
                
                sample_size = min(1000, len(plot_data))
                sample_idx = np.linspace(0, len(plot_data)-1, sample_size, dtype=int)
                
                axes[i, 0].plot(plot_data.index[sample_idx], obs_plot.iloc[sample_idx], 'k-', 
                               label='OBS', alpha=0.7, linewidth=1)
                axes[i, 0].plot(plot_data.index[sample_idx], ec_plot.iloc[sample_idx], 'r-', 
                               label='EC-WRF', alpha=0.7, linewidth=1)
                axes[i, 0].plot(plot_data.index[sample_idx], gfs_plot.iloc[sample_idx], 'b-', 
                               label='GFS-WRF', alpha=0.7, linewidth=1)
                axes[i, 0].set_title(f'{station_name} - {var_name} Time Series')
                axes[i, 0].set_ylabel('Wind Speed (m/s)')
                axes[i, 0].legend()
                axes[i, 0].grid(True, alpha=0.3)
                
                max_val = max(obs_plot.max(), ec_plot.max(), gfs_plot.max())
                
                axes[i, 1].scatter(obs_plot, ec_plot, alpha=0.4, s=8, label='EC-WRF', color='red')
                axes[i, 1].scatter(obs_plot, gfs_plot, alpha=0.4, s=8, label='GFS-WRF', color='blue')
                axes[i, 1].plot([0, max_val], [0, max_val], 'k--', alpha=0.5)
                axes[i, 1].set_xlabel('Observed Wind Speed (m/s)')
                axes[i, 1].set_ylabel('Predicted Wind Speed (m/s)')
                axes[i, 1].set_title(f'{var_name} Scatter Plot')
                axes[i, 1].legend()
                axes[i, 1].grid(True, alpha=0.3)
                
                ec_error = ec_plot - obs_plot
                gfs_error = gfs_plot - obs_plot
                
                axes[i, 2].hist(ec_error, bins=30, alpha=0.6, label='EC-WRF Error', 
                               color='red', density=True)
                axes[i, 2].hist(gfs_error, bins=30, alpha=0.6, label='GFS-WRF Error', 
                               color='blue', density=True)
                axes[i, 2].axvline(0, color='k', linestyle='--', alpha=0.5)
                axes[i, 2].set_xlabel('Error (m/s)')
                axes[i, 2].set_ylabel('Density')
                axes[i, 2].set_title(f'{var_name} Error Distribution')
                axes[i, 2].legend()
                axes[i, 2].grid(True, alpha=0.3)
        
        plt.tight_layout()
        output_file = output_dir / f'{station_name}_validation.png'
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        plt.show()
        
        print(f"验证图表已保存到: {output_file}")
    
    def save_matched_data(self, matched_data, station_name, output_dir=None):
        """保存匹配数据到文件"""
        if output_dir is None:
            output_dir = self.project_root / "01_Data" / "processed" / "matched_data"
        else:
            output_dir = Path(output_dir)
            
        output_dir.mkdir(parents=True, exist_ok=True)
        
        output_file = output_dir / f'{station_name}_matched.csv'
        matched_data.to_csv(output_file)
        
        # 生成数据描述文件
        desc_file = output_dir / f'{station_name}_matched_description.txt'
        with open(desc_file, 'w', encoding='utf-8') as f:
            f.write(f"{station_name} 匹配数据描述 (修正版)\n")
            f.write("=" * 60 + "\n\n")
            f.write(f"数据形状: {matched_data.shape}\n")
            f.write(f"时间范围: {matched_data.index.min()} 到 {matched_data.index.max()}\n")
            f.write(f"数据期间: {(matched_data.index.max() - matched_data.index.min()).days} 天\n\n")
            
            f.write("数据单位说明:\n")
            f.write("  风速: m/s\n")
            f.write("  风向: 度 (°)\n") 
            f.write("  温度: 摄氏度 (°C) - 已从开尔文转换\n")
            f.write("  密度: kg/m³ - 从温度和气压重新计算\n")
            f.write("  湿度: % (仅观测数据)\n\n")
            
            f.write("WRF数据处理说明:\n")
            f.write("  - 温度: 从tk_30m转换K→°C并外推至10m\n")
            f.write("  - 密度: 从tk_30m和p_30m用理想气体定律计算\n")
            f.write("  - 风向: 30m和50m从u/v分量计算\n")
            f.write("  - 湍流指标: WRF无法提供，设为NaN\n\n")
            
            f.write("变量完整性:\n")
            for col in matched_data.columns:
                valid_count = matched_data[col].notna().sum()
                valid_pct = valid_count / len(matched_data) * 100
                f.write(f"  {col}: {valid_count}/{len(matched_data)} ({valid_pct:.1f}%)\n")
            
            f.write(f"\n基本统计:\n")
            f.write(matched_data.describe().to_string())
        
        print(f"匹配数据已保存到: {output_file}")
        print(f"数据描述已保存到: {desc_file}")
    
    def process_single_station(self, station_name):
        """处理单个站点"""
        print(f"开始处理 {station_name} 站点...")
        
        try:
            matched_data = self.create_matched_dataset(station_name)
            
            if matched_data is not None:
                self.matched_datasets[station_name] = matched_data
                
                stats = self.calculate_basic_statistics(matched_data, station_name)
                
                self.create_validation_plots(matched_data, station_name)
                
                self.save_matched_data(matched_data, station_name)
                
                print(f"\n✅ {station_name} 站点处理完成!")
                return stats
            else:
                print(f"\n❌ {station_name} 站点处理失败!")
                return None
                
        except Exception as e:
            print(f"❌ 处理 {station_name} 时出错: {e}")
            import traceback
            traceback.print_exc()
            return None
    
    def process_all_stations(self):
        """处理所有站点"""
        print("开始处理所有站点的数据匹配...")
        
        all_stats = {}
        
        for station_name in self.station_configs.keys():
            stats = self.process_single_station(station_name)
            if stats:
                all_stats[station_name] = stats
        
        self.generate_summary_report(all_stats)
        
        print("\n🎉 所有站点处理完成!")
        return all_stats
    
    def generate_summary_report(self, all_stats, output_file=None):
        """生成总体结果摘要报告"""
        if output_file is None:
            output_file = self.project_root / "03_Results" / "obs_wrf_comparison_summary.txt"
        else:
            output_file = Path(output_file)
            
        output_file.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write("观测-WRF数据匹配分析总体报告 (修正版)\n")
            f.write("=" * 60 + "\n\n")
            
            f.write(f"处理时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"处理站点数: {len(all_stats)}\n\n")
            
            f.write("数据处理说明:\n")
            f.write("  - 温度单位: 开尔文(K) → 摄氏度(°C)\n")
            f.write("  - 密度计算: 从WRF温度和气压重新计算\n")
            f.write("  - 风向计算: 从u/v分量计算\n")
            f.write("  - 时间对齐: 观测与WRF数据精确匹配\n\n")
            
            for station_name, stats in all_stats.items():
                f.write(f"=== {station_name} 站点结果摘要 ===\n")
                
                if stats:
                    wind_speed_vars = [var for var in stats.keys() if 'wind_speed' in var and 'std' not in var and 'max' not in var]
                    
                    if wind_speed_vars:
                        f.write("主要风速变量:\n")
                        for var in wind_speed_vars:
                            metrics = stats[var]
                            f.write(f"  {var}:\n")
                            f.write(f"    样本数: {metrics['n_samples']}\n")
                            f.write(f"    EC  - RMSE: {metrics['ec_rmse']:.3f}, CORR: {metrics['ec_corr']:.3f}\n")
                            f.write(f"    GFS - RMSE: {metrics['gfs_rmse']:.3f}, CORR: {metrics['gfs_corr']:.3f}\n")
                            f.write(f"    更优模式: {metrics['better_model']} (改善 {metrics['improvement_pct']:.1f}%)\n\n")
                    
                    other_vars = [var for var in stats.keys() if var not in wind_speed_vars]
                    if other_vars:
                        f.write("其他变量:\n")
                        for var in other_vars:
                            metrics = stats[var]
                            f.write(f"  {var}: {metrics['better_model']} 更优 (RMSE改善 {metrics['improvement_pct']:.1f}%)\n")
                        f.write("\n")
                    
                    ec_better_count = sum(1 for v in stats.values() if v['better_model'] == 'EC')
                    gfs_better_count = sum(1 for v in stats.values() if v['better_model'] == 'GFS')
                    total_vars = len(stats)
                    
                    f.write(f"总体表现:\n")
                    f.write(f"  EC更优变量数: {ec_better_count}/{total_vars} ({ec_better_count/total_vars*100:.1f}%)\n")
                    f.write(f"  GFS更优变量数: {gfs_better_count}/{total_vars} ({gfs_better_count/total_vars*100:.1f}%)\n")
                    
                    if ec_better_count > gfs_better_count:
                        f.write(f"  推荐: 该站点优先使用 EC-WRF\n")
                    elif gfs_better_count > ec_better_count:
                        f.write(f"  推荐: 该站点优先使用 GFS-WRF\n")
                    else:
                        f.write(f"  推荐: 两种模式表现相当，可结合使用\n")
                
                f.write("\n" + "-" * 40 + "\n")
        
        print(f"\n📊 总体报告已保存到: {output_file}")

# 使用示例和快速测试函数
def quick_test_changma():
    """快速测试昌马站点"""
    print("🧪 快速测试昌马站点...")
    
    matcher = ObsWRFMatcher()
    stats = matcher.process_single_station('changma')
    
    if stats:
        print("\n✅ 昌马站点测试成功!")
        return matcher, stats
    else:
        print("\n❌ 昌马站点测试失败!")
        return None, None

def quick_test_all():
    """快速测试所有站点"""
    print("🧪 测试所有站点...")
    
    matcher = ObsWRFMatcher()
    all_stats = matcher.process_all_stations()
    
    return matcher, all_stats

# 主执行函数
if __name__ == "__main__":
    print("🚀 观测-WRF数据匹配系统启动 (修正版)")
    print("=" * 60)
    
    # 方案1: 仅测试昌马站点（推荐开始）
    # matcher, stats = quick_test_changma()
    
    # 方案2: 测试所有站点
    matcher, all_stats = quick_test_all()
    
    if matcher:
        print("\n📈 匹配数据可通过以下方式访问:")
        print("   matcher.matched_datasets['changma']  # 昌马站匹配数据")
        print("   matcher.matched_datasets['kuangqu']  # 矿区站匹配数据") 
        print("   matcher.matched_datasets['sanlijijingzi']  # 三十里井子站匹配数据")
        
        print("\n📂 输出文件位置:")
        print("   01_Data/processed/matched_data/     # 匹配数据CSV文件")
        print("   03_Results/validation_plots/        # 验证图表")
        print("   03_Results/obs_wrf_comparison_summary.txt # 总体报告")
        
        print("\n🔧 修正内容:")
        print("   ✅ 温度单位: K → °C 转换和外推")
        print("   ✅ 密度计算: 从温度气压重新计算") 
        print("   ✅ 风向计算: 从u/v分量计算")
        print("   ✅ 数据描述: 包含单位和处理说明")