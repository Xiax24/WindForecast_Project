#!/usr/bin/env python3
"""
昌马站小时稳定度计算和粗糙度长度推导系统（hourly计算版）
基于Wang et al. (2024)方法的实现

参考文献:
- Wang, J., Yang, K., Yuan, L., et al. (2024). Deducing aerodynamic roughness 
  length from abundant anemometer tower data to inform wind resource modeling. 
  Geophysical Research Letters, 51, e2024GL111056.

日期: 2025-05-24
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
import warnings
import os
from collections import Counter
from scipy import stats
from scipy.optimize import minimize_scalar, minimize
import json

warnings.filterwarnings('ignore')

class ChangmaHourlyStabilityAnalyzer:
    """昌马站小时稳定度分析器"""
    
    def __init__(self):
        self.alpha_thresholds = {
            'unstable': 0.1,
            'neutral_upper': 0.3,
            'stable': 0.3
        }
        self.daytime_hours = (6, 18)
        self.low_wind_threshold = 1.0
        self.temp_change_thresholds = {
            'rapid_warming': 0.5,
            'rapid_cooling': -0.5
        }
    
    def load_data(self, file_path):
        """加载15分钟数据"""
        print(f"加载数据: {file_path}")
        data = pd.read_csv(file_path)
        
        time_columns = [col for col in data.columns if 'time' in col.lower() or 'date' in col.lower()]
        
        if time_columns:
            time_col = time_columns[0]
            data['timestamp'] = pd.to_datetime(data[time_col])
        else:
            first_col = data.columns[0]
            data['timestamp'] = pd.to_datetime(data[first_col])
        
        print(f"数据形状: {data.shape}")
        print(f"时间范围: {data['timestamp'].min()} 到 {data['timestamp'].max()}")
        
        return data
    
    def resample_to_hourly(self, data):
        """将15分钟数据重采样为小时数据"""
        print("将15分钟数据重采样为小时数据...")
        
        data_indexed = data.set_index('timestamp')
        
        wind_cols = [col for col in data.columns if 'wind_speed' in col]
        temp_cols = [col for col in data.columns if 'temperature' in col]
        other_cols = [col for col in data.columns if col not in wind_cols + temp_cols and col != 'timestamp']
        
        agg_dict = {}
        
        for col in wind_cols + temp_cols:
            agg_dict[col] = 'mean'
        
        for col in other_cols:
            if data[col].dtype in ['float64', 'int64']:
                agg_dict[col] = 'mean'
        
        hourly_data = data_indexed.resample('H').agg(agg_dict)
        hourly_data = hourly_data.reset_index()
        
        print(f"重采样后数据形状: {hourly_data.shape}")
        print(f"小时数据时间范围: {hourly_data['timestamp'].min()} 到 {hourly_data['timestamp'].max()}")
        
        return hourly_data
    
    def calculate_wind_shear_alpha(self, wind_lower, wind_upper, height_lower, height_upper):
        """计算风切变参数α"""
        if pd.isna(wind_lower) or pd.isna(wind_upper) or wind_lower <= 0 or wind_upper <= 0:
            return np.nan
        
        try:
            alpha = np.log(wind_upper / wind_lower) / np.log(height_upper / height_lower)
            return alpha
        except:
            return np.nan
    
    def classify_stability_by_alpha(self, alpha):
        """基于α值分类稳定度"""
        if pd.isna(alpha):
            return 'unknown'
        elif alpha > self.alpha_thresholds['stable']:
            return 'stable'
        elif alpha < self.alpha_thresholds['unstable']:
            return 'unstable'
        else:
            return 'neutral'
    
    def calculate_temp_trend(self, temp_series, current_idx, lookback_hours=[1, 2, 3]):
        """计算温度变化趋势"""
        current_temp = temp_series.iloc[current_idx]
        
        if pd.isna(current_temp):
            return np.nan, 'current_missing'
        
        for hours in lookback_hours:
            if current_idx >= hours:
                past_temp = temp_series.iloc[current_idx - hours]
                if not pd.isna(past_temp):
                    temp_change_rate = (current_temp - past_temp) / hours
                    return temp_change_rate, f'lookback_{hours}h'
        
        return np.nan, 'insufficient_history'
    
    def get_expected_stability_by_time(self, timestamp):
        """基于时间获取预期稳定度"""
        hour = timestamp.hour
        
        if self.daytime_hours[0] <= hour < self.daytime_hours[1]:
            return 'unstable'
        else:
            return 'stable'
    
    def get_expected_stability_by_temp(self, temp_change_rate):
        """基于温度变化获取预期稳定度"""
        if pd.isna(temp_change_rate):
            return 'unknown'
        elif temp_change_rate > self.temp_change_thresholds['rapid_warming']:
            return 'unstable'
        elif temp_change_rate < self.temp_change_thresholds['rapid_cooling']:
            return 'stable'
        else:
            return 'neutral'
    
    def check_physical_consistency(self, stability_alpha, stability_time, stability_temp, alpha_value):
        """检查物理一致性"""
        consistencies = []
        
        if stability_alpha == stability_time:
            consistencies.append('time_consistent')
        else:
            consistencies.append('time_inconsistent')
        
        if stability_temp != 'unknown':
            if stability_alpha == stability_temp:
                consistencies.append('temp_consistent')
            else:
                consistencies.append('temp_inconsistent')
        
        if not pd.isna(alpha_value) and abs(alpha_value) > 1.0:
            consistencies.append('extreme_shear')
        
        if 'time_consistent' in consistencies and 'temp_consistent' in consistencies:
            return 'consistent'
        elif 'extreme_shear' in consistencies:
            return 'questionable'
        elif len([c for c in consistencies if 'inconsistent' in c]) >= 2:
            return 'inconsistent'
        else:
            return 'acceptable'
    
    def get_season(self, timestamp):
        """根据月份获取季节"""
        month = timestamp.month
        if month in [3, 4, 5]:
            return 'spring'
        elif month in [6, 7, 8]:
            return 'summer'
        elif month in [9, 10, 11]:
            return 'autumn'
        else:
            return 'winter'
    
    def assess_data_availability(self, row):
        """评估单个时间点的数据可用性"""
        wind_10m = not pd.isna(row['obs_wind_speed_10m'])
        wind_30m = not pd.isna(row['obs_wind_speed_30m'])
        wind_50m = not pd.isna(row['obs_wind_speed_50m'])
        wind_70m = not pd.isna(row['obs_wind_speed_70m'])
        temp_10m = not pd.isna(row['obs_temperature_10m'])
        
        availability = {
            'wind_10m': wind_10m,
            'wind_30m': wind_30m,
            'wind_50m': wind_50m,
            'wind_70m': wind_70m,
            'temp_10m': temp_10m
        }
        
        if wind_10m and wind_70m:
            if wind_30m and wind_50m:
                return 'complete', availability
            else:
                return 'sufficient', availability
        elif sum([wind_10m, wind_30m, wind_50m, wind_70m]) >= 2:
            return 'limited', availability
        else:
            return 'insufficient', availability
    
    def calculate_hourly_stability(self, row, temp_series, row_idx):
        """计算单个小时的稳定度"""
        result = {
            'timestamp': row['timestamp'],
            'stability_final': 'unknown',
            'confidence_final': 0.0,
            'alpha_main': np.nan,
            'data_quality': 'insufficient',
            'quality_flags': [],
            'wind_10m': row.get('obs_wind_speed_10m', np.nan),
            'wind_30m': row.get('obs_wind_speed_30m', np.nan),
            'wind_50m': row.get('obs_wind_speed_50m', np.nan),
            'wind_70m': row.get('obs_wind_speed_70m', np.nan),
            'temp_10m': row.get('obs_temperature_10m', np.nan),
            'alpha_10_30': np.nan,
            'alpha_30_50': np.nan,
            'alpha_50_70': np.nan,
            'alpha_10_70': np.nan,
            'stability_10_30': 'unknown',
            'stability_30_50': 'unknown',
            'stability_50_70': 'unknown',
            'stability_10_70': 'unknown',
            'hour': row['timestamp'].hour,
            'month': row['timestamp'].month,
            'season': self.get_season(row['timestamp']),
            'is_daytime': self.daytime_hours[0] <= row['timestamp'].hour < self.daytime_hours[1]
        }
        
        availability_level, availability_detail = self.assess_data_availability(row)
        result['data_availability'] = availability_level
        
        if availability_level == 'insufficient':
            result['quality_flags'].append('insufficient_data')
            return result
        
        # 计算各层α值
        if availability_detail['wind_10m'] and availability_detail['wind_30m']:
            result['alpha_10_30'] = self.calculate_wind_shear_alpha(
                result['wind_10m'], result['wind_30m'], 10, 30
            )
            result['stability_10_30'] = self.classify_stability_by_alpha(result['alpha_10_30'])
        
        if availability_detail['wind_30m'] and availability_detail['wind_50m']:
            result['alpha_30_50'] = self.calculate_wind_shear_alpha(
                result['wind_30m'], result['wind_50m'], 30, 50
            )
            result['stability_30_50'] = self.classify_stability_by_alpha(result['alpha_30_50'])
        
        if availability_detail['wind_50m'] and availability_detail['wind_70m']:
            result['alpha_50_70'] = self.calculate_wind_shear_alpha(
                result['wind_50m'], result['wind_70m'], 50, 70
            )
            result['stability_50_70'] = self.classify_stability_by_alpha(result['alpha_50_70'])
        
        if availability_detail['wind_10m'] and availability_detail['wind_70m']:
            result['alpha_10_70'] = self.calculate_wind_shear_alpha(
                result['wind_10m'], result['wind_70m'], 10, 70
            )
            result['stability_10_70'] = self.classify_stability_by_alpha(result['alpha_10_70'])
            result['alpha_main'] = result['alpha_10_70']
        
        # 备选方案
        if pd.isna(result['alpha_main']):
            for key in ['alpha_10_30', 'alpha_30_50', 'alpha_50_70']:
                if not pd.isna(result[key]):
                    result['alpha_main'] = result[key]
                    result['quality_flags'].append(f'alternative_alpha_{key}')
                    break
        
        if pd.isna(result['alpha_main']):
            result['quality_flags'].append('no_valid_alpha')
            return result
        
        # 主要分类
        stability_alpha = self.classify_stability_by_alpha(result['alpha_main'])
        result['stability_primary'] = stability_alpha
        
        # 一致性检验
        valid_classifications = []
        for key in ['stability_10_30', 'stability_30_50', 'stability_50_70', 'stability_10_70']:
            if result[key] != 'unknown':
                valid_classifications.append(result[key])
        
        if len(valid_classifications) > 1:
            consistency_score = valid_classifications.count(stability_alpha) / len(valid_classifications)
            result['consistency_score'] = consistency_score
            
            if consistency_score >= 0.8:
                result['quality_flags'].append('high_consistency')
            elif consistency_score < 0.5:
                result['quality_flags'].append('low_consistency')
        else:
            result['consistency_score'] = 1.0
        
        # 时间和温度辅助判断
        stability_time = self.get_expected_stability_by_time(row['timestamp'])
        result['expected_stability_time'] = stability_time
        
        temp_change_rate, temp_status = self.calculate_temp_trend(temp_series, row_idx)
        result['temp_change_rate'] = temp_change_rate
        result['temp_status'] = temp_status
        
        stability_temp = self.get_expected_stability_by_temp(temp_change_rate)
        result['expected_stability_temp'] = stability_temp
        
        # 物理一致性
        physical_consistency = self.check_physical_consistency(
            stability_alpha, stability_time, stability_temp, result['alpha_main']
        )
        result['physical_consistency'] = physical_consistency
        
        # 最终结果
        result['stability_final'] = stability_alpha
        
        # 置信度计算
        confidence = 0.7
        
        if result['consistency_score'] >= 0.8:
            confidence += 0.15
        elif result['consistency_score'] < 0.5:
            confidence -= 0.1
        
        if physical_consistency == 'consistent':
            confidence += 0.1
        elif physical_consistency == 'questionable':
            confidence -= 0.05
        elif physical_consistency == 'inconsistent':
            confidence -= 0.15
        
        if (result['wind_10m'] < self.low_wind_threshold or 
            result['wind_70m'] < self.low_wind_threshold):
            confidence -= 0.1
            result['quality_flags'].append('low_wind_speed')
        
        if not pd.isna(result['alpha_main']) and abs(result['alpha_main']) > 1.0:
            confidence -= 0.1
            result['quality_flags'].append('extreme_shear')
        
        result['confidence_final'] = max(min(confidence, 0.95), 0.1)
        
        # 数据质量等级
        if result['confidence_final'] >= 0.8:
            result['data_quality'] = 'high'
        elif result['confidence_final'] >= 0.6:
            result['data_quality'] = 'medium'
        else:
            result['data_quality'] = 'low'
        
        return result
    
    def analyze_hourly_stability(self, data):
        """批量分析小时稳定度"""
        print("开始小时稳定度分析...")
        
        hourly_data = self.resample_to_hourly(data)
        
        results = []
        temp_series = hourly_data['obs_temperature_10m']
        total_points = len(hourly_data)
        
        for idx, row in hourly_data.iterrows():
            result = self.calculate_hourly_stability(row, temp_series, idx)
            results.append(result)
            
            if idx > 0 and idx % 500 == 0:
                print(f"进度: {idx}/{total_points} ({idx/total_points*100:.1f}%)")
        
        results_df = pd.DataFrame(results)
        print(f"\n小时稳定度分析完成！共处理 {len(results_df)} 个小时")
        
        return results_df


class ChangmaRoughnessCalculator:
    """昌马站粗糙度长度计算器"""
    
    def __init__(self):
        self.neutral_stability_threshold = 0.85
        self.von_karman = 0.4
        self.displacement_ratio = 20/3
        self.min_wind_speed = 0.4
        self.max_wind_speed = 50.0
        self.terrain_uniformity_threshold = 30
    
    def filter_neutral_conditions(self, stability_df):
        """筛选中性稳定条件的数据"""
        print("筛选中性稳定条件的数据...")
        
        neutral_mask = (
            (stability_df['stability_final'] == 'neutral') &
            (stability_df['confidence_final'] >= self.neutral_stability_threshold) &
            (stability_df['data_quality'] == 'high')
        )
        
        neutral_data = stability_df[neutral_mask].copy()
        
        print(f"原始小时数据点数: {len(stability_df)}")
        print(f"符合中性条件的数据点数: {len(neutral_data)}")
        print(f"中性条件比例: {len(neutral_data)/len(stability_df)*100:.2f}%")
        
        if len(neutral_data) == 0:
            print("警告：没有符合中性条件的数据！")
            return pd.DataFrame()
        
        return neutral_data
    
    def quality_control_wind_profiles(self, neutral_data):
        """风廓线质量控制"""
        print("执行风廓线质量控制...")
        
        initial_count = len(neutral_data)
        
        wind_cols = ['wind_10m', 'wind_30m', 'wind_50m', 'wind_70m']
        
        valid_mask = pd.Series(True, index=neutral_data.index)
        
        for col in wind_cols:
            if col in neutral_data.columns:
                valid_mask &= (
                    (neutral_data[col] >= self.min_wind_speed) &
                    (neutral_data[col] <= self.max_wind_speed) &
                    (~pd.isna(neutral_data[col]))
                )
        
        # 检查风廓线单调性（允许5%容差）
        tolerance = 0.05
        wind_monotonic_relaxed = (
            (neutral_data['wind_10m'] <= neutral_data['wind_30m'] * (1 + tolerance)) &
            (neutral_data['wind_30m'] <= neutral_data['wind_50m'] * (1 + tolerance)) &
            (neutral_data['wind_50m'] <= neutral_data['wind_70m'] * (1 + tolerance))
        )
        
        final_mask = valid_mask & wind_monotonic_relaxed
        
        qc_data = neutral_data[final_mask].copy()
        
        print(f"质量控制前数据点数: {initial_count}")
        print(f"质量控制后数据点数: {len(qc_data)}")
        print(f"质量控制通过率: {len(qc_data)/initial_count*100:.2f}%")
        
        return qc_data
    
    def logarithmic_wind_profile_equation(self, z, u_star, z0, d=0):
        """对数风廓线方程"""
        if z <= d or z0 <= 0:
            return np.nan
        
        return (u_star / self.von_karman) * np.log((z - d) / z0)
    
    def calculate_rmse_for_profile(self, params, heights, wind_speeds):
        """计算给定参数下风廓线拟合的RMSE"""
        ln_z0, u_star = params
        z0 = np.exp(ln_z0)
        d = self.displacement_ratio * z0
        
        if z0 <= 0 or u_star <= 0:
            return 1e10
        
        predicted_winds = []
        for i, h in enumerate(heights):
            if h <= d:
                return 1e10
            
            pred_wind = self.logarithmic_wind_profile_equation(h, u_star, z0, d)
            if np.isnan(pred_wind) or pred_wind <= 0:
                return 1e10
            predicted_winds.append(pred_wind)
        
        predicted_winds = np.array(predicted_winds)
        rmse = np.sqrt(np.mean((wind_speeds - predicted_winds) ** 2))
        
        return rmse
    
    def derive_z0_single_profile(self, wind_profile, heights=[10, 30, 50, 70]):
        """从单个风廓线推导z0值"""
        valid_indices = []
        valid_winds = []
        valid_heights = []
        
        wind_cols = ['wind_10m', 'wind_30m', 'wind_50m', 'wind_70m']
        
        for i, (col, h) in enumerate(zip(wind_cols, heights)):
            if col in wind_profile and not pd.isna(wind_profile[col]) and wind_profile[col] > 0:
                valid_winds.append(wind_profile[col])
                valid_heights.append(h)
                valid_indices.append(i)
        
        if len(valid_winds) < 3:
            return np.nan, np.nan, np.nan, 'insufficient_heights'
        
        valid_winds = np.array(valid_winds)
        valid_heights = np.array(valid_heights)
        
        # 初始猜测
        ln_z0_init = -4.0
        u_star_init = valid_winds[0] * self.von_karman / np.log(valid_heights[0] / 0.01)
        
        best_rmse = float('inf')
        best_ln_z0 = np.nan
        best_u_star = np.nan
        
        # 尝试多个初始值
        ln_z0_range = np.linspace(-6, -2, 5)
        
        for ln_z0_guess in ln_z0_range:
            try:
                def objective_u_star(u_star):
                    if u_star <= 0:
                        return 1e10
                    return self.calculate_rmse_for_profile([ln_z0_guess, u_star], valid_heights, valid_winds)
                
                result = minimize_scalar(objective_u_star, bounds=(0.01, 5.0), method='bounded')
                
                if result.success and result.fun < best_rmse:
                    best_rmse = result.fun
                    best_ln_z0 = ln_z0_guess
                    best_u_star = result.x
                    
            except:
                continue
        
        # 进一步优化
        if not np.isnan(best_ln_z0):
            try:
                def objective_both(params):
                    return self.calculate_rmse_for_profile(params, valid_heights, valid_winds)
                
                initial_guess = [best_ln_z0, best_u_star]
                bounds = [(-8, -1), (0.01, 5.0)]
                
                result = minimize(objective_both, initial_guess, bounds=bounds, method='L-BFGS-B')
                
                if result.success and result.fun < best_rmse:
                    best_rmse = result.fun
                    best_ln_z0, best_u_star = result.x
                    
            except:
                pass
        
        if np.isnan(best_ln_z0):
            return np.nan, np.nan, np.nan, 'optimization_failed'
        
        # 验证结果合理性
        z0 = np.exp(best_ln_z0)
        d = self.displacement_ratio * z0
        
        if d >= min(valid_heights):
            return np.nan, np.nan, np.nan, 'unreasonable_displacement'
        
        if z0 < 0.001 or z0 > 1.0:
            return np.nan, np.nan, np.nan, 'unreasonable_z0'
        
        # 计算拟合质量
        predicted_winds = []
        for h in valid_heights:
            pred_wind = self.logarithmic_wind_profile_equation(h, best_u_star, z0, d)
            predicted_winds.append(pred_wind)
        
        predicted_winds = np.array(predicted_winds)
        
        if len(predicted_winds) > 1:
            correlation = np.corrcoef(valid_winds, predicted_winds)[0, 1]
        else:
            correlation = 0
        
        # 检查拟合质量
        if correlation < 0.9 or best_rmse > 2.0:
            return np.nan, np.nan, np.nan, 'poor_fit_quality'
        
        return best_ln_z0, best_u_star, best_rmse, 'success'
    
    def derive_z0_from_neutral_data(self, neutral_qc_data):
        """从中性条件数据推导z0值"""
        print("开始从中性条件数据推导z0值...")
        
        z0_results = []
        
        for idx, row in neutral_qc_data.iterrows():
            ln_z0, u_star, rmse, status = self.derive_z0_single_profile(row)
            
            result = {
                'timestamp': row['timestamp'],
                'ln_z0': ln_z0,
                'z0': np.exp(ln_z0) if not pd.isna(ln_z0) else np.nan,
                'u_star': u_star,
                'rmse': rmse,
                'status': status,
                'wind_10m': row['wind_10m'],
                'wind_30m': row['wind_30m'],
                'wind_50m': row['wind_50m'],
                'wind_70m': row['wind_70m'],
                'alpha_main': row['alpha_main'],
                'confidence': row['confidence_final']
            }
            
            z0_results.append(result)
            
            # 进度提示
            if len(z0_results) % 100 == 0:
                print(f"已处理 {len(z0_results)}/{len(neutral_qc_data)} 个风廓线")
        
        z0_df = pd.DataFrame(z0_results)
        
        # 统计成功率
        success_count = len(z0_df[z0_df['status'] == 'success'])
        success_rate = success_count / len(z0_df) * 100
        
        print(f"\nz0推导完成！")
        print(f"总风廓线数: {len(z0_df)}")
        print(f"成功推导数: {success_count}")
        print(f"成功率: {success_rate:.2f}%")
        
        # 状态统计
        status_counts = z0_df['status'].value_counts()
        print(f"\n状态统计:")
        for status, count in status_counts.items():
            print(f"  {status}: {count} ({count/len(z0_df)*100:.1f}%)")
        
        return z0_df
    
    def statistical_analysis_z0(self, z0_df):
        """对推导的z0值进行统计分析"""
        print("\n开始z0值统计分析...")
        
        # 筛选成功的z0值
        valid_z0 = z0_df[z0_df['status'] == 'success']['ln_z0'].dropna()
        
        if len(valid_z0) == 0:
            print("错误：没有有效的z0值！")
            return None
        
        print(f"有效ln_z0样本数: {len(valid_z0)}")
        
        # 基本统计
        ln_z0_mean = valid_z0.mean()
        ln_z0_median = valid_z0.median()
        ln_z0_std = valid_z0.std()
        ln_z0_min = valid_z0.min()
        ln_z0_max = valid_z0.max()
        
        print(f"\nln_z0统计:")
        print(f"  平均值: {ln_z0_mean:.3f}")
        print(f"  中位数: {ln_z0_median:.3f}")
        print(f"  标准差: {ln_z0_std:.3f}")
        print(f"  范围: [{ln_z0_min:.3f}, {ln_z0_max:.3f}]")
        
        # 转换为z0统计
        z0_values = np.exp(valid_z0)
        z0_mean = z0_values.mean()
        z0_median = z0_values.median()
        z0_std = z0_values.std()
        
        print(f"\nz0统计 (m):")
        print(f"  平均值: {z0_mean:.4f}")
        print(f"  中位数: {z0_median:.4f}")
        print(f"  标准差: {z0_std:.4f}")
        print(f"  范围: [{z0_values.min():.4f}, {z0_values.max():.4f}]")
        
        # 计算分布特征
        from scipy import stats as scipy_stats
        
        skewness = scipy_stats.skew(valid_z0)
        kurtosis = scipy_stats.kurtosis(valid_z0)
        
        print(f"\n分布特征:")
        print(f"  偏度: {skewness:.3f}")
        print(f"  峰度: {kurtosis:.3f}")
        
        # 检查中位数和众数的接近程度
        hist, bin_edges = np.histogram(valid_z0, bins=20)
        mode_bin_idx = np.argmax(hist)
        mode_center = (bin_edges[mode_bin_idx] + bin_edges[mode_bin_idx + 1]) / 2
        
        median_mode_diff = abs(ln_z0_median - mode_center)
        
        print(f"  中位数: {ln_z0_median:.3f}")
        print(f"  众数(近似): {mode_center:.3f}")
        print(f"  中位数-众数差异: {median_mode_diff:.3f}")
        
        # 质量评估
        quality_flag = 'good'
        
        if median_mode_diff > 0.5:
            quality_flag = 'questionable'
            print("警告：中位数和众数差异较大，可能存在多个风向的影响")
        
        if abs(skewness) > 1.0:
            quality_flag = 'questionable'
            print("警告：分布偏度较大，可能需要进一步质量控制")
        
        # 最终z0值确定
        final_ln_z0 = ln_z0_median
        final_z0 = np.exp(final_ln_z0)
        
        # 计算置信区间（bootstrap方法）
        n_bootstrap = 1000
        bootstrap_medians = []
        
        for _ in range(n_bootstrap):
            bootstrap_sample = np.random.choice(valid_z0, size=len(valid_z0), replace=True)
            bootstrap_medians.append(np.median(bootstrap_sample))
        
        confidence_interval = np.percentile(bootstrap_medians, [2.5, 97.5])
        
        print(f"\n最终结果:")
        print(f"  最终ln_z0: {final_ln_z0:.3f}")
        print(f"  最终z0: {final_z0:.4f} m")
        print(f"  95%置信区间(ln_z0): [{confidence_interval[0]:.3f}, {confidence_interval[1]:.3f}]")
        print(f"  95%置信区间(z0): [{np.exp(confidence_interval[0]):.4f}, {np.exp(confidence_interval[1]):.4f}] m")
        print(f"  数据质量: {quality_flag}")
        
        # 返回结果
        result = {
            'final_ln_z0': final_ln_z0,
            'final_z0': final_z0,
            'ln_z0_statistics': {
                'mean': ln_z0_mean,
                'median': ln_z0_median,
                'std': ln_z0_std,
                'min': ln_z0_min,
                'max': ln_z0_max,
                'skewness': skewness,
                'kurtosis': kurtosis
            },
            'z0_statistics': {
                'mean': z0_mean,
                'median': z0_median,
                'std': z0_std,
                'min': z0_values.min(),
                'max': z0_values.max()
            },
            'confidence_interval_ln_z0': confidence_interval,
            'confidence_interval_z0': [np.exp(confidence_interval[0]), np.exp(confidence_interval[1])],
            'sample_size': len(valid_z0),
            'quality_flag': quality_flag,
            'median_mode_difference': median_mode_diff
        }
        
        return result
    
    def validate_z0_result(self, final_result, z0_df, neutral_qc_data):
        """验证z0结果的可靠性"""
        print("\n开始验证z0结果的可靠性...")
        
        if final_result is None:
            print("无法验证：没有有效的z0结果")
            return None
        
        final_z0 = final_result['final_z0']
        final_d = self.displacement_ratio * final_z0
        
        print(f"使用z0 = {final_z0:.4f} m, d = {final_d:.4f} m进行验证")
        
        # 选择成功推导z0的数据进行验证
        valid_z0_data = z0_df[z0_df['status'] == 'success'].copy()
        
        validation_results = []
        
        for idx, row in valid_z0_data.iterrows():
            # 使用最终z0值和该时刻的u_star进行外推
            u_star = row['u_star']
            
            if pd.isna(u_star) or u_star <= 0:
                continue
            
            # 外推各高度风速
            heights = [10, 30, 50, 70]
            wind_cols = ['wind_10m', 'wind_30m', 'wind_50m', 'wind_70m']
            
            observed_winds = []
            predicted_winds = []
            valid_heights = []
            
            for h, col in zip(heights, wind_cols):
                if not pd.isna(row[col]) and row[col] > 0:
                    observed = row[col]
                    predicted = self.logarithmic_wind_profile_equation(h, u_star, final_z0, final_d)
                    
                    if not pd.isna(predicted) and predicted > 0:
                        observed_winds.append(observed)
                        predicted_winds.append(predicted)
                        valid_heights.append(h)
            
            if len(observed_winds) >= 3:  # 至少3个高度
                observed_winds = np.array(observed_winds)
                predicted_winds = np.array(predicted_winds)
                
                # 计算验证指标
                correlation = np.corrcoef(observed_winds, predicted_winds)[0, 1]
                rmse = np.sqrt(np.mean((observed_winds - predicted_winds) ** 2))
                mean_bias = np.mean(predicted_winds - observed_winds)
                mean_bias_pct = mean_bias / np.mean(observed_winds) * 100
                
                validation_results.append({
                    'timestamp': row['timestamp'],
                    'correlation': correlation,
                    'rmse': rmse,
                    'mean_bias': mean_bias,
                    'mean_bias_pct': mean_bias_pct,
                    'n_heights': len(observed_winds)
                })
        
        if len(validation_results) == 0:
            print("无法进行验证：没有足够的有效数据")
            return None
        
        validation_df = pd.DataFrame(validation_results)
        
        # 计算总体验证指标
        mean_correlation = validation_df['correlation'].mean()
        mean_rmse = validation_df['rmse'].mean()
        mean_bias_pct = validation_df['mean_bias_pct'].mean()
        
        # 计算通过验证的比例
        good_correlation = (validation_df['correlation'] > 0.9).sum()
        good_rmse = (validation_df['rmse'] < 2.0).sum()
        good_bias = (abs(validation_df['mean_bias_pct']) < 10).sum()
        
        total_samples = len(validation_df)
        
        print(f"\n验证结果:")
        print(f"  验证样本数: {total_samples}")
        print(f"  平均相关系数: {mean_correlation:.3f}")
        print(f"  平均RMSE: {mean_rmse:.3f} m/s")
        print(f"  平均偏差: {mean_bias_pct:.2f}%")
        print(f"\n通过率统计:")
        print(f"  相关系数>0.9: {good_correlation}/{total_samples} ({good_correlation/total_samples*100:.1f}%)")
        print(f"  RMSE<2.0 m/s: {good_rmse}/{total_samples} ({good_rmse/total_samples*100:.1f}%)")
        print(f"  偏差<10%: {good_bias}/{total_samples} ({good_bias/total_samples*100:.1f}%)")
        
        # 总体评价
        if mean_correlation > 0.9 and mean_rmse < 2.0 and abs(mean_bias_pct) < 10:
            validation_quality = 'excellent'
        elif mean_correlation > 0.8 and mean_rmse < 3.0 and abs(mean_bias_pct) < 15:
            validation_quality = 'good'
        elif mean_correlation > 0.7 and mean_rmse < 4.0 and abs(mean_bias_pct) < 20:
            validation_quality = 'acceptable'
        else:
            validation_quality = 'poor'
        
        print(f"  总体验证质量: {validation_quality}")
        
        validation_summary = {
            'sample_size': total_samples,
            'mean_correlation': mean_correlation,
            'mean_rmse': mean_rmse,
            'mean_bias_pct': mean_bias_pct,
            'good_correlation_rate': good_correlation / total_samples,
            'good_rmse_rate': good_rmse / total_samples,
            'good_bias_rate': good_bias / total_samples,
            'validation_quality': validation_quality,
            'detailed_results': validation_df
        }
        
        return validation_summary
    
    def run_complete_analysis(self, stability_df):
        """运行完整的粗糙度分析流程"""
        print("=" * 60)
        print("开始昌马站粗糙度长度推导分析")
        print("=" * 60)
        
        # 1. 筛选中性稳定条件
        neutral_data = self.filter_neutral_conditions(stability_df)
        if len(neutral_data) == 0:
            print("分析终止：没有符合条件的中性稳定数据")
            return None
        
        # 2. 风廓线质量控制
        neutral_qc_data = self.quality_control_wind_profiles(neutral_data)
        if len(neutral_qc_data) == 0:
            print("分析终止：质量控制后没有有效数据")
            return None
        
        # 3. 推导z0值
        z0_df = self.derive_z0_from_neutral_data(neutral_qc_data)
        
        # 4. 统计分析确定最终z0
        final_result = self.statistical_analysis_z0(z0_df)
        
        # 5. 验证结果
        validation_summary = self.validate_z0_result(final_result, z0_df, neutral_qc_data)
        
        # 6. 整合所有结果
        complete_result = {
            'final_z0_result': final_result,
            'validation_summary': validation_summary,
            'z0_derivation_details': z0_df,
            'neutral_data_used': neutral_qc_data,
            'analysis_metadata': {
                'total_hourly_samples': len(stability_df),
                'neutral_samples': len(neutral_data),
                'qc_passed_samples': len(neutral_qc_data),
                'successful_z0_derivations': len(z0_df[z0_df['status'] == 'success']),
                'analysis_timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            }
        }
        
        print("\n" + "=" * 60)
        print("昌马站粗糙度长度推导分析完成")
        print("=" * 60)
        
        return complete_result


def main():
    """主执行函数"""
    # 设置路径
    data_path = "/Users/xiaxin/work/WindForecast_Project/01_Data/processed/matched_data/changma_matched.csv"
    output_dir = "/Users/xiaxin/work/WindForecast_Project/03_Results/roughness/corrected_roughness_wang_hourly"
    
    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)
    
    try:
        # 1. 初始化小时稳定度分析器
        print("步骤1: 初始化小时稳定度分析器")
        stability_analyzer = ChangmaHourlyStabilityAnalyzer()
        
        # 2. 加载数据并计算小时稳定度
        print("\n步骤2: 加载数据并计算小时稳定度")
        data = stability_analyzer.load_data(data_path)
        hourly_stability_df = stability_analyzer.analyze_hourly_stability(data)
        
        # 保存小时稳定度结果
        stability_output_path = os.path.join(output_dir, "changma_hourly_stability.csv")
        hourly_stability_df.to_csv(stability_output_path, index=False)
        print(f"小时稳定度结果已保存到: {stability_output_path}")
        
        # 3. 初始化粗糙度计算器
        print("\n步骤3: 初始化粗糙度计算器")
        roughness_calculator = ChangmaRoughnessCalculator()
        
        # 4. 运行完整的粗糙度分析
        print("\n步骤4: 运行粗糙度分析")
        roughness_result = roughness_calculator.run_complete_analysis(hourly_stability_df)
        
        if roughness_result is None:
            print("粗糙度分析失败")
            return
        
        # 5. 保存结果
        print("\n步骤5: 保存分析结果")
        
        # 保存最终z0结果
        if roughness_result['final_z0_result'] is not None:
            z0_summary = {
                'final_z0_meters': roughness_result['final_z0_result']['final_z0'],
                'final_ln_z0': roughness_result['final_z0_result']['final_ln_z0'],
                'confidence_interval_z0': roughness_result['final_z0_result']['confidence_interval_z0'],
                'sample_size': roughness_result['final_z0_result']['sample_size'],
                'quality_flag': roughness_result['final_z0_result']['quality_flag'],
                'analysis_metadata': roughness_result['analysis_metadata']
            }
            
            # 保存为JSON
            z0_summary_path = os.path.join(output_dir, "changma_z0_summary.json")
            with open(z0_summary_path, 'w', encoding='utf-8') as f:
                json.dump(z0_summary, f, indent=2, ensure_ascii=False, default=str)
            
            print(f"z0摘要结果已保存到: {z0_summary_path}")
        
        # 保存详细的z0推导结果
        z0_details_path = os.path.join(output_dir, "changma_z0_derivation_details.csv")
        roughness_result['z0_derivation_details'].to_csv(z0_details_path, index=False)
        print(f"z0推导详细结果已保存到: {z0_details_path}")
        
        # 保存验证结果
        if roughness_result['validation_summary'] is not None:
            validation_path = os.path.join(output_dir, "changma_z0_validation.csv")
            roughness_result['validation_summary']['detailed_results'].to_csv(validation_path, index=False)
            print(f"z0验证结果已保存到: {validation_path}")
        
        # 生成分析报告
        print("\n步骤6: 生成分析报告")
        generate_analysis_report(roughness_result, output_dir)
        
        print(f"\n分析完成！所有结果已保存到: {output_dir}")
        
    except Exception as e:
        print(f"分析过程中出现错误: {str(e)}")
        import traceback
        traceback.print_exc()
        raise


def generate_analysis_report(roughness_result, output_dir):
    """生成分析报告"""
    report_path = os.path.join(output_dir, "changma_roughness_analysis_report.txt")
    
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write("昌马站空气动力学粗糙度长度推导分析报告\n")
        f.write("=" * 60 + "\n\n")
        f.write(f"分析时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"基于Wang et al. (2024)方法\n\n")
        
        # 数据统计
        metadata = roughness_result['analysis_metadata']
        f.write("数据统计:\n")
        f.write(f"  总小时样本数: {metadata['total_hourly_samples']}\n")
        f.write(f"  中性稳定条件样本数: {metadata['neutral_samples']}\n")
        f.write(f"  质量控制通过样本数: {metadata['qc_passed_samples']}\n")
        f.write(f"  成功推导z0的样本数: {metadata['successful_z0_derivations']}\n\n")
        
        # 最终z0结果
        if roughness_result['final_z0_result'] is not None:
            final_result = roughness_result['final_z0_result']
            f.write("最终粗糙度长度结果:\n")
            f.write(f"  z0 = {final_result['final_z0']:.4f} m\n")
            f.write(f"  ln(z0) = {final_result['final_ln_z0']:.3f}\n")
            f.write(f"  95%置信区间: [{final_result['confidence_interval_z0'][0]:.4f}, {final_result['confidence_interval_z0'][1]:.4f}] m\n")
            f.write(f"  样本数量: {final_result['sample_size']}\n")
            f.write(f"  数据质量: {final_result['quality_flag']}\n\n")
            
            # 统计信息
            stats = final_result['ln_z0_statistics']
            f.write("ln(z0)统计信息:\n")
            f.write(f"  均值: {stats['mean']:.3f}\n")
            f.write(f"  中位数: {stats['median']:.3f}\n")
            f.write(f"  标准差: {stats['std']:.3f}\n")
            f.write(f"  偏度: {stats['skewness']:.3f}\n")
            f.write(f"  峰度: {stats['kurtosis']:.3f}\n\n")
        
        # 验证结果
        if roughness_result['validation_summary'] is not None:
            validation = roughness_result['validation_summary']
            f.write("验证结果:\n")
            f.write(f"  验证样本数: {validation['sample_size']}\n")
            f.write(f"  平均相关系数: {validation['mean_correlation']:.3f}\n")
            f.write(f"  平均RMSE: {validation['mean_rmse']:.3f} m/s\n")
            f.write(f"  平均偏差: {validation['mean_bias_pct']:.2f}%\n")
            f.write(f"  验证质量: {validation['validation_quality']}\n\n")
        
        f.write("分析方法说明:\n")
        f.write("1. 基于15分钟观测数据计算小时稳定度\n")
        f.write("2. 筛选中性稳定条件且置信度>0.8的数据\n")
        f.write("3. 使用对数风廓线方程拟合推导z0\n")
        f.write("4. 采用中位数作为最终z0值\n")
        f.write("5. 通过风速外推验证结果可靠性\n\n")
        
        f.write("参考文献:\n")
        f.write("Wang, J., Yang, K., Yuan, L., et al. (2024). Deducing aerodynamic \n")
        f.write("roughness length from abundant anemometer tower data to inform \n")
        f.write("wind resource modeling. Geophysical Research Letters, 51, e2024GL111056.\n")
    
    print(f"分析报告已保存到: {report_path}")


if __name__ == "__main__":
    main()