#!/usr/bin/env python3
"""
昌马站大气稳定度分析
基于风切变参数α的多因素稳定度分类系统

作者: 研究团队
日期: 2025-05-23
文件位置: /Users/xiaxin/work/WindForecast_Project/02_Code/preprocessing/changma_stability_analysis.py

参考文献:
- Emeis, S. (2013). Wind energy meteorology
- Wharton & Lundquist (2012). Atmospheric stability affects wind turbine power collection  
- Gualtieri, G. (2019). Comprehensive review on wind resource extrapolation models
- Risan et al. (2018). Wind in complex terrain - lidar measurements for evaluation of CFD simulations
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
import warnings
import os
from collections import Counter
warnings.filterwarnings('ignore')

# 设置matplotlib中文字体
import matplotlib
import platform

# 根据操作系统设置字体
system = platform.system()
if system == 'Darwin':  # macOS
    # Mac系统常用字体
    matplotlib.rcParams['font.family'] = ['PingFang SC', 'Arial Unicode MS', 'STHeiti', 'SimHei']
elif system == 'Windows':
    # Windows系统常用字体
    matplotlib.rcParams['font.family'] = ['Microsoft YaHei', 'SimHei', 'KaiTi', 'SimSun']
else:  # Linux
    # Linux系统常用字体
    matplotlib.rcParams['font.family'] = ['WenQuanYi Micro Hei', 'SimHei', 'DejaVu Sans']

matplotlib.rcParams['axes.unicode_minus'] = False

# 检测字体支持
def check_chinese_font_support():
    """检测中文字体支持"""
    try:
        import matplotlib.font_manager as fm
        # 获取所有可用字体
        available_fonts = set(f.name for f in fm.fontManager.ttflist)
        
        # 中文字体列表
        chinese_fonts = [
            'PingFang SC', 'PingFang TC', 'STHeiti', 'STSong', 'STKaiti',
            'Arial Unicode MS', 'Microsoft YaHei', 'SimHei', 'SimSun', 'KaiTi',
            'WenQuanYi Micro Hei', 'WenQuanYi Zen Hei', 'Noto Sans CJK SC'
        ]
        
        # 找到第一个可用的中文字体
        for font in chinese_fonts:
            if font in available_fonts:
                matplotlib.rcParams['font.sans-serif'] = [font]
                print(f"检测到中文字体: {font}")
                return True, font
        
        print("警告: 未检测到中文字体支持")
        return False, None
        
    except Exception as e:
        print(f"字体检测错误: {e}")
        return False, None

# 执行字体检测
chinese_support, font_name = check_chinese_font_support()

# 如果没有中文字体支持，使用英文
if not chinese_support:
    matplotlib.rcParams['font.sans-serif'] = ['DejaVu Sans', 'Arial', 'sans-serif']

class ChangmaStabilityAnalyzer:
    """
    昌马站大气稳定度分析器
    基于多层风速数据和温度数据的综合稳定度分类
    """
    
    def __init__(self):
        # 稳定度分类阈值（基于文献）
        self.alpha_thresholds = {
            'unstable': 0.1,      # α < 0.1
            'neutral_upper': 0.3, # 0.1 ≤ α < 0.3  
            'stable': 0.3         # α ≥ 0.3
        }
        
        # 时间分类参数（简化，实际应考虑季节和纬度）
        self.daytime_hours = (6, 18)
        
        # 低风速阈值（α计算不稳定的临界值）
        self.low_wind_threshold = 1.0
        
        # 温度变化率阈值
        self.temp_change_thresholds = {
            'rapid_warming': 0.5,   # °C/h
            'rapid_cooling': -0.5   # °C/h
        }
    
    def load_data(self, file_path):
        """加载昌马站匹配数据"""
        print(f"加载数据: {file_path}")
        
        data = pd.read_csv(file_path)
        
        # 检查列名，寻找时间列
        print(f"数据列名: {list(data.columns)}")
        
        # 尝试找到时间列
        time_columns = [col for col in data.columns if 'time' in col.lower() or 'date' in col.lower()]
        print(f"发现的时间相关列: {time_columns}")
        
        if time_columns:
            time_col = time_columns[0]
            data['timestamp'] = pd.to_datetime(data[time_col])
            print(f"使用时间列: {time_col}")
        else:
            # 如果没有明显的时间列，检查第一列是否是时间
            first_col = data.columns[0]
            try:
                data['timestamp'] = pd.to_datetime(data[first_col])
                print(f"使用第一列作为时间: {first_col}")
            except:
                print("错误：无法识别时间列！")
                print(f"前5行数据预览:")
                print(data.head())
                raise ValueError("无法识别时间列")
        
        print(f"数据形状: {data.shape}")
        print(f"时间范围: {data['timestamp'].min()} 到 {data['timestamp'].max()}")
        
        return data
    
    def assess_data_availability(self, row):
        """评估单个时间点的数据可用性"""
        # 检查关键风速数据
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
        
        # 数据质量等级判断
        if wind_10m and wind_70m:
            if wind_30m and wind_50m:
                return 'complete', availability
            else:
                return 'sufficient', availability
        elif sum([wind_10m, wind_30m, wind_50m, wind_70m]) >= 2:
            return 'limited', availability
        else:
            return 'insufficient', availability
    
    def calculate_wind_shear_alpha(self, wind_lower, wind_upper, height_lower, height_upper):
        """
        计算风切变参数α
        α = ln(V_upper/V_lower) / ln(h_upper/h_lower)
        """
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
        """
        计算温度变化趋势
        尝试不同的回望时间窗口
        """
        current_temp = temp_series.iloc[current_idx]
        
        if pd.isna(current_temp):
            return np.nan, 'current_missing'
        
        # 尝试不同的时间窗口（以15分钟为单位）
        for hours in lookback_hours:
            lookback_steps = int(hours * 4)  # 每小时4个15分钟
            
            if current_idx >= lookback_steps:
                past_temp = temp_series.iloc[current_idx - lookback_steps]
                if not pd.isna(past_temp):
                    temp_change_rate = (current_temp - past_temp) / hours
                    return temp_change_rate, f'lookback_{hours}h'
        
        return np.nan, 'insufficient_history'
    
    def get_expected_stability_by_time(self, timestamp):
        """基于时间获取预期稳定度"""
        hour = timestamp.hour
        
        if self.daytime_hours[0] <= hour < self.daytime_hours[1]:
            return 'unstable'  # 白天倾向不稳定
        else:
            return 'stable'    # 夜间倾向稳定
    
    def get_expected_stability_by_temp(self, temp_change_rate):
        """基于温度变化获取预期稳定度"""
        if pd.isna(temp_change_rate):
            return 'unknown'
        elif temp_change_rate > self.temp_change_thresholds['rapid_warming']:
            return 'unstable'  # 快速增温
        elif temp_change_rate < self.temp_change_thresholds['rapid_cooling']:
            return 'stable'    # 快速降温
        else:
            return 'neutral'   # 温度相对稳定
    
    def check_physical_consistency(self, stability_alpha, stability_time, stability_temp, alpha_value):
        """检查物理一致性"""
        consistencies = []
        
        # 与时间预期的一致性
        if stability_alpha == stability_time:
            consistencies.append('time_consistent')
        else:
            consistencies.append('time_inconsistent')
        
        # 与温度预期的一致性
        if stability_temp != 'unknown':
            if stability_alpha == stability_temp:
                consistencies.append('temp_consistent')
            else:
                consistencies.append('temp_inconsistent')
        
        # 检查极端α值
        if not pd.isna(alpha_value) and abs(alpha_value) > 1.0:
            consistencies.append('extreme_shear')
        
        # 综合判断
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
    
    def calculate_stability_single_point(self, row, temp_series, row_idx):
        """计算单个时间点的稳定度 - 包含详细计算过程"""
        result = {
            'timestamp': row['timestamp'],
            'stability_final': 'unknown',
            'confidence_final': 0.0,
            'alpha_main': np.nan,
            'data_quality': 'insufficient',
            'quality_flags': [],
            # 详细计算过程变量
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
        
        # 1. 评估数据可用性
        availability_level, availability_detail = self.assess_data_availability(row)
        result['data_availability'] = availability_level
        
        if availability_level == 'insufficient':
            result['quality_flags'].append('insufficient_data')
            return result
        
        # 2. 计算各层α值并记录详细过程
        print(f"\n--- 时间点 {row['timestamp']} 稳定度计算 ---")
        print(f"风速数据: 10m={result['wind_10m']:.2f}, 30m={result['wind_30m']:.2f}, "
              f"50m={result['wind_50m']:.2f}, 70m={result['wind_70m']:.2f}")
        print(f"温度数据: 10m={result['temp_10m']:.2f}°C")
        
        # 10m-30m α值
        if availability_detail['wind_10m'] and availability_detail['wind_30m']:
            result['alpha_10_30'] = self.calculate_wind_shear_alpha(
                result['wind_10m'], result['wind_30m'], 10, 30
            )
            result['stability_10_30'] = self.classify_stability_by_alpha(result['alpha_10_30'])
            print(f"α(10-30m) = ln({result['wind_30m']:.2f}/{result['wind_10m']:.2f})/ln(3) = {result['alpha_10_30']:.3f} → {result['stability_10_30']}")
        
        # 30m-50m α值
        if availability_detail['wind_30m'] and availability_detail['wind_50m']:
            result['alpha_30_50'] = self.calculate_wind_shear_alpha(
                result['wind_30m'], result['wind_50m'], 30, 50
            )
            result['stability_30_50'] = self.classify_stability_by_alpha(result['alpha_30_50'])
            print(f"α(30-50m) = ln({result['wind_50m']:.2f}/{result['wind_30m']:.2f})/ln({50/30:.2f}) = {result['alpha_30_50']:.3f} → {result['stability_30_50']}")
        
        # 50m-70m α值
        if availability_detail['wind_50m'] and availability_detail['wind_70m']:
            result['alpha_50_70'] = self.calculate_wind_shear_alpha(
                result['wind_50m'], result['wind_70m'], 50, 70
            )
            result['stability_50_70'] = self.classify_stability_by_alpha(result['alpha_50_70'])
            print(f"α(50-70m) = ln({result['wind_70m']:.2f}/{result['wind_50m']:.2f})/ln({70/50:.2f}) = {result['alpha_50_70']:.3f} → {result['stability_50_70']}")
        
        # 10m-70m α值（主要指标）
        if availability_detail['wind_10m'] and availability_detail['wind_70m']:
            result['alpha_10_70'] = self.calculate_wind_shear_alpha(
                result['wind_10m'], result['wind_70m'], 10, 70
            )
            result['stability_10_70'] = self.classify_stability_by_alpha(result['alpha_10_70'])
            result['alpha_main'] = result['alpha_10_70']
            print(f"★ α(10-70m) = ln({result['wind_70m']:.2f}/{result['wind_10m']:.2f})/ln(7) = {result['alpha_10_70']:.3f} → {result['stability_10_70']} (主要指标)")
        
        # 如果主要α值无法计算，使用备选方案
        if pd.isna(result['alpha_main']):
            for key in ['alpha_10_30', 'alpha_30_50', 'alpha_50_70']:
                if not pd.isna(result[key]):
                    result['alpha_main'] = result[key]
                    result['quality_flags'].append(f'alternative_alpha_{key}')
                    print(f"使用备选α值: {key} = {result[key]:.3f}")
                    break
        
        if pd.isna(result['alpha_main']):
            result['quality_flags'].append('no_valid_alpha')
            print("无法计算有效的α值")
            return result
        
        # 3. 基于α值的主要分类
        stability_alpha = self.classify_stability_by_alpha(result['alpha_main'])
        result['stability_primary'] = stability_alpha
        print(f"主要稳定度分类: {stability_alpha}")
        
        # 4. 多层一致性检验
        valid_classifications = []
        alpha_keys = ['alpha_10_30', 'alpha_30_50', 'alpha_50_70', 'alpha_10_70']
        stability_keys = ['stability_10_30', 'stability_30_50', 'stability_50_70', 'stability_10_70']
        
        for i, alpha_key in enumerate(alpha_keys):
            if not pd.isna(result[alpha_key]):
                classification = result[stability_keys[i]]
                valid_classifications.append(classification)
        
        if len(valid_classifications) > 1:
            consistency_score = valid_classifications.count(stability_alpha) / len(valid_classifications)
            result['consistency_score'] = consistency_score
            print(f"多层一致性: {valid_classifications} → 一致性评分: {consistency_score:.2f}")
            
            if consistency_score >= 0.8:
                result['quality_flags'].append('high_consistency')
            elif consistency_score < 0.5:
                result['quality_flags'].append('low_consistency')
        else:
            result['consistency_score'] = 1.0
            print("只有一个有效分类，一致性评分: 1.0")
        
        # 5. 时间辅助判断
        stability_time = self.get_expected_stability_by_time(row['timestamp'])
        result['expected_stability_time'] = stability_time
        print(f"时间预期稳定度: {stability_time} (时间: {result['hour']}:xx, {'白天' if result['is_daytime'] else '夜间'})")
        
        # 6. 温度变化辅助判断
        temp_change_rate, temp_status = self.calculate_temp_trend(temp_series, row_idx)
        result['temp_change_rate'] = temp_change_rate
        result['temp_status'] = temp_status
        
        stability_temp = self.get_expected_stability_by_temp(temp_change_rate)
        result['expected_stability_temp'] = stability_temp
        
        if not pd.isna(temp_change_rate):
            print(f"温度变化率: {temp_change_rate:.2f}°C/h ({temp_status}) → 预期稳定度: {stability_temp}")
        else:
            print(f"温度变化率: 无法计算 ({temp_status})")
        
        # 7. 物理一致性检验
        physical_consistency = self.check_physical_consistency(
            stability_alpha, stability_time, stability_temp, result['alpha_main']
        )
        result['physical_consistency'] = physical_consistency
        print(f"物理一致性检验: {physical_consistency}")
        
        # 8. 最终稳定度和置信度计算
        result['stability_final'] = stability_alpha
        
        # 详细置信度计算过程
        confidence = 0.7  # 基础置信度
        confidence_adjustments = []
        
        # 一致性调整
        if result['consistency_score'] >= 0.8:
            adj = 0.15
            confidence += adj
            confidence_adjustments.append(f"高一致性 +{adj}")
        elif result['consistency_score'] < 0.5:
            adj = -0.1
            confidence += adj
            confidence_adjustments.append(f"低一致性 {adj}")
        
        # 物理一致性调整
        if physical_consistency == 'consistent':
            adj = 0.1
            confidence += adj
            confidence_adjustments.append(f"物理一致 +{adj}")
        elif physical_consistency == 'questionable':
            adj = -0.05
            confidence += adj
            confidence_adjustments.append(f"物理可疑 {adj}")
        elif physical_consistency == 'inconsistent':
            adj = -0.15
            confidence += adj
            confidence_adjustments.append(f"物理不一致 {adj}")
        
        # 低风速调整
        if (result['wind_10m'] < self.low_wind_threshold or 
            result['wind_70m'] < self.low_wind_threshold):
            adj = -0.1
            confidence += adj
            confidence_adjustments.append(f"低风速 {adj}")
            result['quality_flags'].append('low_wind_speed')
        
        # 极端α值调整
        if not pd.isna(result['alpha_main']) and abs(result['alpha_main']) > 1.0:
            adj = -0.1
            confidence += adj
            confidence_adjustments.append(f"极端切变 {adj}")
            result['quality_flags'].append('extreme_shear')
        
        result['confidence_final'] = max(min(confidence, 0.95), 0.1)
        result['confidence_adjustments'] = '; '.join(confidence_adjustments)
        
        print(f"置信度计算: 0.70(基础) {' '.join(confidence_adjustments)} = {result['confidence_final']:.3f}")
        
        # 数据质量等级
        if result['confidence_final'] >= 0.8:
            result['data_quality'] = 'high'
        elif result['confidence_final'] >= 0.6:
            result['data_quality'] = 'medium'
        else:
            result['data_quality'] = 'low'
        
        print(f"最终结果: 稳定度={result['stability_final']}, 置信度={result['confidence_final']:.3f}, 质量={result['data_quality']}")
        print(f"质量标记: {result['quality_flags']}")
        
        return result
    
    def calculate_stability_single_point_quiet(self, row, temp_series, row_idx):
        """计算单个时间点的稳定度 - 静默模式（不输出详细过程）"""
        result = {
            'timestamp': row['timestamp'],
            'stability_final': 'unknown',
            'confidence_final': 0.0,
            'alpha_main': np.nan,
            'data_quality': 'insufficient',
            'quality_flags': [],
            # 详细计算过程变量
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
        
        # 数据可用性评估
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
        confidence_adjustments = []
        
        if result['consistency_score'] >= 0.8:
            confidence += 0.15
            confidence_adjustments.append('+0.15')
        elif result['consistency_score'] < 0.5:
            confidence -= 0.1
            confidence_adjustments.append('-0.10')
        
        if physical_consistency == 'consistent':
            confidence += 0.1
            confidence_adjustments.append('+0.10')
        elif physical_consistency == 'questionable':
            confidence -= 0.05
            confidence_adjustments.append('-0.05')
        elif physical_consistency == 'inconsistent':
            confidence -= 0.15
            confidence_adjustments.append('-0.15')
        
        if (result['wind_10m'] < self.low_wind_threshold or 
            result['wind_70m'] < self.low_wind_threshold):
            confidence -= 0.1
            confidence_adjustments.append('-0.10(low_wind)')
            result['quality_flags'].append('low_wind_speed')
        
        if not pd.isna(result['alpha_main']) and abs(result['alpha_main']) > 1.0:
            confidence -= 0.1
            confidence_adjustments.append('-0.10(extreme)')
            result['quality_flags'].append('extreme_shear')
        
        result['confidence_final'] = max(min(confidence, 0.95), 0.1)
        result['confidence_adjustments'] = '; '.join(confidence_adjustments)
        
        # 数据质量等级
        if result['confidence_final'] >= 0.8:
            result['data_quality'] = 'high'
        elif result['confidence_final'] >= 0.6:
            result['data_quality'] = 'medium'
        else:
            result['data_quality'] = 'low'
        
        return result
    
    def analyze_stability(self, data, max_detailed_output=50):
        """批量分析稳定度 - 包含详细输出控制"""
        print("开始稳定度分析...")
        print(f"注意: 为避免输出过多，只显示前{max_detailed_output}个时间点的详细计算过程")
        
        results = []
        temp_series = data['obs_temperature_10m']
        total_points = len(data)
        
        for idx, row in data.iterrows():
            # 控制详细输出数量
            if idx < max_detailed_output:
                result = self.calculate_stability_single_point(row, temp_series, idx)
            else:
                # 简化输出模式
                result = self.calculate_stability_single_point_quiet(row, temp_series, idx)
            
            results.append(result)
            
            # 进度提示
            if idx > 0 and idx % 5000 == 0:
                print(f"\n进度: {idx}/{total_points} ({idx/total_points*100:.1f}%)")
                # 显示最近的统计
                recent_results = results[-1000:]
                recent_stability = [r['stability_final'] for r in recent_results if r['stability_final'] != 'unknown']
                if recent_stability:
                    recent_dist = Counter(recent_stability)
                    print(f"最近1000个点的稳定度分布: {dict(recent_dist)}")
        
        results_df = pd.DataFrame(results)
        print(f"\n分析完成！共处理 {len(results_df)} 个时间点")
        
        return results_df
    
    def generate_summary_statistics(self, results_df):
        """生成统计摘要"""
        print("\n=== 昌马站稳定度分析统计摘要 ===")
        
        # 基本统计
        total_points = len(results_df)
        valid_results = results_df[results_df['stability_final'] != 'unknown']
        valid_count = len(valid_results)
        
        print(f"总时间点数: {total_points}")
        print(f"有效分析点数: {valid_count} ({valid_count/total_points*100:.1f}%)")
        
        # 稳定度分布
        if len(valid_results) > 0:
            stability_dist = valid_results['stability_final'].value_counts()
            print(f"\n稳定度分布:")
            for stability, count in stability_dist.items():
                percentage = count / len(valid_results) * 100
                print(f"  {stability}: {count} ({percentage:.1f}%)")
        
        # 数据质量分布
        quality_dist = results_df['data_quality'].value_counts()
        print(f"\n数据质量分布:")
        for quality, count in quality_dist.items():
            percentage = count / total_points * 100
            print(f"  {quality}: {count} ({percentage:.1f}%)")
        
        # α值统计
        valid_alpha = results_df[~pd.isna(results_df['alpha_main'])]
        if len(valid_alpha) > 0:
            print(f"\nα值统计:")
            print(f"  平均值: {valid_alpha['alpha_main'].mean():.3f}")
            print(f"  标准差: {valid_alpha['alpha_main'].std():.3f}")
            print(f"  最小值: {valid_alpha['alpha_main'].min():.3f}")
            print(f"  最大值: {valid_alpha['alpha_main'].max():.3f}")
        
        # 置信度统计
        valid_confidence = results_df[results_df['confidence_final'] > 0]
        if len(valid_confidence) > 0:
            print(f"\n置信度统计:")
            print(f"  平均置信度: {valid_confidence['confidence_final'].mean():.3f}")
            print(f"  高置信度(>0.8): {sum(valid_confidence['confidence_final'] > 0.8)} "
                  f"({sum(valid_confidence['confidence_final'] > 0.8)/len(valid_confidence)*100:.1f}%)")
        
        # 季节分布
        if len(valid_results) > 0:
            season_dist = valid_results['season'].value_counts()
            print(f"\n季节分布:")
            for season, count in season_dist.items():
                percentage = count / len(valid_results) * 100
                print(f"  {season}: {count} ({percentage:.1f}%)")
        
        # 日夜分布
        if len(valid_results) > 0:
            day_night_dist = valid_results['is_daytime'].value_counts()
            print(f"\n日夜分布:")
            for is_day, count in day_night_dist.items():
                day_night = "白天" if is_day else "夜间"
                percentage = count / len(valid_results) * 100
                print(f"  {day_night}: {count} ({percentage:.1f}%)")
        
        return {
            'total_points': total_points,
            'valid_points': valid_count,
            'stability_distribution': stability_dist.to_dict() if len(valid_results) > 0 else {},
            'quality_distribution': quality_dist.to_dict(),
            'alpha_stats': {
                'mean': valid_alpha['alpha_main'].mean() if len(valid_alpha) > 0 else np.nan,
                'std': valid_alpha['alpha_main'].std() if len(valid_alpha) > 0 else np.nan,
                'min': valid_alpha['alpha_main'].min() if len(valid_alpha) > 0 else np.nan,
                'max': valid_alpha['alpha_main'].max() if len(valid_alpha) > 0 else np.nan
            },
            'season_distribution': season_dist.to_dict() if len(valid_results) > 0 else {},
            'day_night_distribution': day_night_dist.to_dict() if len(valid_results) > 0 else {}
        }

def main():
    """主执行函数"""
    # 设置路径
    data_path = "/Users/xiaxin/work/WindForecast_Project/01_Data/processed/matched_data/changma_matched.csv"
    output_dir = "/Users/xiaxin/work/WindForecast_Project/03_Results/stability_analysis"
    
    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)
    
    # 初始化分析器
    analyzer = ChangmaStabilityAnalyzer()
    
    try:
        # 加载数据
        data = analyzer.load_data(data_path)
        
        # 进行稳定度分析
        results_df = analyzer.analyze_stability(data)
        
        # 生成统计摘要
        summary_stats = analyzer.generate_summary_statistics(results_df)
        
        # 保存结果
        results_output_path = os.path.join(output_dir, "changma_stability_results.csv")
        results_df.to_csv(results_output_path, index=False)
        print(f"\n稳定度分析结果已保存到: {results_output_path}")
        
        # 保存统计摘要
        summary_output_path = os.path.join(output_dir, "changma_stability_summary.txt")
        with open(summary_output_path, 'w', encoding='utf-8') as f:
            f.write("昌马站大气稳定度分析统计摘要\n")
            f.write("="*50 + "\n\n")
            f.write(f"分析时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"数据文件: {data_path}\n\n")
            
            f.write(f"总时间点数: {summary_stats['total_points']}\n")
            f.write(f"有效分析点数: {summary_stats['valid_points']}\n\n")
            
            f.write("稳定度分布:\n")
            for stability, count in summary_stats['stability_distribution'].items():
                percentage = count / summary_stats['valid_points'] * 100
                f.write(f"  {stability}: {count} ({percentage:.1f}%)\n")
            
            f.write(f"\n数据质量分布:\n")
            for quality, count in summary_stats['quality_distribution'].items():
                percentage = count / summary_stats['total_points'] * 100
                f.write(f"  {quality}: {count} ({percentage:.1f}%)\n")
            
            f.write(f"\nα值统计:\n")
            alpha_stats = summary_stats['alpha_stats']
            f.write(f"  平均值: {alpha_stats['mean']:.3f}\n")
            f.write(f"  标准差: {alpha_stats['std']:.3f}\n")
            f.write(f"  范围: [{alpha_stats['min']:.3f}, {alpha_stats['max']:.3f}]\n")
            
            f.write(f"\n季节分布:\n")
            for season, count in summary_stats['season_distribution'].items():
                percentage = count / summary_stats['valid_points'] * 100
                f.write(f"  {season}: {count} ({percentage:.1f}%)\n")
            
            f.write(f"\n日夜分布:\n")
            for is_day, count in summary_stats['day_night_distribution'].items():
                day_night = "白天" if is_day else "夜间"
                percentage = count / summary_stats['valid_points'] * 100
                f.write(f"  {day_night}: {count} ({percentage:.1f}%)\n")
        
        print(f"统计摘要已保存到: {summary_output_path}")
        
        # 生成基础可视化图表
        print(f"\n正在生成基础可视化图表...")
        create_basic_plots(results_df, output_dir)
        
        print(f"\n=== 分析完成 ===")
        print(f"输出文件:")
        print(f"  - 详细结果: {results_output_path}")
        print(f"  - 统计摘要: {summary_output_path}")
        print(f"  - 可视化图表: {output_dir}/plots/")
        print(f"\n下一步可以进行:")
        print(f"1. 稳定度与EC/GFS预报误差的关联分析")
        print(f"2. 日变化和季节变化的详细分析")
        print(f"3. 典型案例的深入分析")
        
    except Exception as e:
        print(f"分析过程中出现错误: {str(e)}")
        import traceback
        traceback.print_exc()
        raise

def create_basic_plots(results_df, output_dir):
    """创建基础可视化图表"""
    plots_dir = os.path.join(output_dir, "plots")
    os.makedirs(plots_dir, exist_ok=True)
    
    # 过滤有效数据
    valid_data = results_df[results_df['stability_final'] != 'unknown'].copy()
    
    if len(valid_data) == 0:
        print("警告: 没有有效数据用于可视化")
        return
    
    # 设置图表样式
    plt.style.use('default')
    fig_size = (12, 8)
    
    # 为了避免中文显示问题，统一使用英文标题和标签
    print("使用英文标题和标签以确保兼容性")
    
    # 1. 稳定度分布饼图
    plt.figure(figsize=fig_size)
    stability_counts = valid_data['stability_final'].value_counts()
    colors = ['#ff6b6b', '#4ecdc4', '#45b7d1']  # 更美观的颜色
    
    # 标签映射
    labels_map = {'stable': 'Stable', 'neutral': 'Neutral', 'unstable': 'Unstable'}
    plot_labels = [labels_map.get(label, label) for label in stability_counts.index]
    
    wedges, texts, autotexts = plt.pie(stability_counts.values, labels=plot_labels, autopct='%1.1f%%', 
                                       colors=colors, startangle=90, textprops={'fontsize': 12})
    
    # 美化饼图
    for autotext in autotexts:
        autotext.set_color('white')
        autotext.set_fontweight('bold')
    
    plt.title('Changma Station Atmospheric Stability Distribution', fontsize=16, fontweight='bold', pad=20)
    plt.axis('equal')
    plt.savefig(os.path.join(plots_dir, 'stability_distribution.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    # 2. α值分布直方图
    plt.figure(figsize=fig_size)
    alpha_data = valid_data[~pd.isna(valid_data['alpha_main'])]['alpha_main']
    
    # 创建更美观的直方图
    n, bins, patches = plt.hist(alpha_data, bins=50, alpha=0.7, color='steelblue', edgecolor='black', linewidth=0.5)
    
    # 根据α值给直方图条着色
    for i, patch in enumerate(patches):
        bin_center = (bins[i] + bins[i+1]) / 2
        if bin_center < 0.1:
            patch.set_facecolor('#ff6b6b')  # 不稳定 - 红色
        elif bin_center < 0.3:
            patch.set_facecolor('#4ecdc4')  # 中性 - 青色
        else:
            patch.set_facecolor('#45b7d1')  # 稳定 - 蓝色
    
    # 添加边界线
    plt.axvline(0.1, color='red', linestyle='--', linewidth=2, alpha=0.8, 
                label='Unstable/Neutral boundary (α=0.1)')
    plt.axvline(0.3, color='orange', linestyle='--', linewidth=2, alpha=0.8,
                label='Neutral/Stable boundary (α=0.3)')
    
    plt.xlabel('Wind Shear Parameter α', fontsize=14, fontweight='bold')
    plt.ylabel('Frequency', fontsize=14, fontweight='bold')
    plt.title('Changma Station Wind Shear Parameter α Distribution', fontsize=16, fontweight='bold', pad=20)
    plt.legend(fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, 'alpha_distribution.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    # 3. 稳定度日变化
    plt.figure(figsize=(14, 8))
    hourly_stability = valid_data.groupby(['hour', 'stability_final']).size().unstack(fill_value=0)
    hourly_stability_pct = hourly_stability.div(hourly_stability.sum(axis=1), axis=0) * 100
    
    # 重命名列标签
    column_rename = {'stable': 'Stable', 'neutral': 'Neutral', 'unstable': 'Unstable'}
    hourly_stability_pct = hourly_stability_pct.rename(columns=column_rename)
    
    # 创建堆叠柱状图
    ax = hourly_stability_pct.plot(kind='bar', stacked=True, 
                                   color=['#ff6b6b', '#4ecdc4', '#45b7d1'],
                                   width=0.8, figsize=(14, 8))
    
    plt.xlabel('Hour of Day', fontsize=14, fontweight='bold')
    plt.ylabel('Percentage (%)', fontsize=14, fontweight='bold')
    plt.title('Changma Station Stability Daily Variation', fontsize=16, fontweight='bold', pad=20)
    plt.legend(title='Stability Classification', bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=12)
    plt.xticks(rotation=0)  # 水平显示小时标签
    plt.grid(True, alpha=0.3, axis='y')
    
    # 添加日夜分界线
    plt.axvline(5.5, color='gray', linestyle=':', alpha=0.7, label='Sunrise (~6:00)')
    plt.axvline(18.5, color='gray', linestyle=':', alpha=0.7, label='Sunset (~18:00)')
    
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, 'stability_daily_variation.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    # 4. 置信度分布
    plt.figure(figsize=fig_size)
    confidence_data = valid_data[valid_data['confidence_final'] > 0]['confidence_final']
    
    plt.hist(confidence_data, bins=30, alpha=0.7, color='mediumseagreen', edgecolor='black', linewidth=0.5)
    plt.axvline(confidence_data.mean(), color='red', linestyle='--', linewidth=2,
                label=f'Mean Confidence: {confidence_data.mean():.3f}')
    plt.axvline(confidence_data.median(), color='orange', linestyle='--', linewidth=2,
                label=f'Median Confidence: {confidence_data.median():.3f}')
    
    plt.xlabel('Confidence Score', fontsize=14, fontweight='bold')
    plt.ylabel('Frequency', fontsize=14, fontweight='bold')
    plt.title('Changma Station Stability Analysis Confidence Distribution', fontsize=16, fontweight='bold', pad=20)
    plt.legend(fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, 'confidence_distribution.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    # 5. 季节-稳定度关系
    plt.figure(figsize=fig_size)
    seasonal_stability = valid_data.groupby(['season', 'stability_final']).size().unstack(fill_value=0)
    seasonal_stability_pct = seasonal_stability.div(seasonal_stability.sum(axis=1), axis=0) * 100
    
    # 重命名标签
    seasonal_stability_pct = seasonal_stability_pct.rename(columns=column_rename)
    season_rename = {'spring': 'Spring', 'summer': 'Summer', 'autumn': 'Autumn', 'winter': 'Winter'}
    seasonal_stability_pct = seasonal_stability_pct.rename(index=season_rename)
    
    ax = seasonal_stability_pct.plot(kind='bar', color=['#ff6b6b', '#4ecdc4', '#45b7d1'], 
                                     width=0.7, figsize=fig_size)
    plt.xlabel('Season', fontsize=14, fontweight='bold')
    plt.ylabel('Percentage (%)', fontsize=14, fontweight='bold')
    plt.title('Changma Station Stability Seasonal Variation', fontsize=16, fontweight='bold', pad=20)
    plt.legend(title='Stability Classification', bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=12)
    plt.xticks(rotation=45)
    plt.grid(True, alpha=0.3, axis='y')
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, 'stability_seasonal_variation.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    # 6. α值时间序列图（采样显示）
    plt.figure(figsize=(16, 8))
    # 为了图表清晰，每50个点取一个
    sample_data = valid_data.iloc[::50].copy()
    
    colors_map = {'stable': '#ff6b6b', 'neutral': '#4ecdc4', 'unstable': '#45b7d1'}
    
    for stability in ['stable', 'neutral', 'unstable']:
        mask = sample_data['stability_final'] == stability
        if mask.any():
            plt.scatter(sample_data[mask]['timestamp'], sample_data[mask]['alpha_main'], 
                       c=colors_map[stability], label=stability.capitalize(), alpha=0.7, s=15)
    
    # 添加边界线
    plt.axhline(0.1, color='red', linestyle='--', alpha=0.8, linewidth=2,
                label='Unstable/Neutral boundary (α=0.1)')
    plt.axhline(0.3, color='orange', linestyle='--', alpha=0.8, linewidth=2,
                label='Neutral/Stable boundary (α=0.3)')
    plt.axhline(0, color='black', linestyle='-', alpha=0.5, linewidth=1)
    
    plt.xlabel('Time', fontsize=14, fontweight='bold')
    plt.ylabel('Wind Shear Parameter α', fontsize=14, fontweight='bold')
    plt.title('Changma Station Wind Shear Parameter Time Series', fontsize=16, fontweight='bold', pad=20)
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, 'alpha_timeseries.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    # 7. 新增：风速剖面图（不同稳定度下的典型风速剖面）
    plt.figure(figsize=(12, 10))
    
    heights = [10, 30, 50, 70]
    
    # 计算不同稳定度下的平均风速剖面
    for i, stability in enumerate(['unstable', 'neutral', 'stable']):
        stability_data = valid_data[valid_data['stability_final'] == stability]
        if len(stability_data) > 0:
            mean_winds = []
            for height in heights:
                col_name = f'wind_{height}m'
                if col_name in stability_data.columns:
                    mean_winds.append(stability_data[col_name].mean())
                else:
                    mean_winds.append(np.nan)
            
            plt.plot(mean_winds, heights, 'o-', linewidth=3, markersize=8, 
                    color=colors_map[stability], label=f'{stability.capitalize()} (n={len(stability_data)})')
    
    plt.xlabel('Wind Speed (m/s)', fontsize=14, fontweight='bold')
    plt.ylabel('Height (m)', fontsize=14, fontweight='bold')
    plt.title('Changma Station Mean Wind Speed Profiles by Stability Class', 
              fontsize=16, fontweight='bold', pad=20)
    plt.legend(fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.ylim(0, 80)
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, 'wind_profiles_by_stability.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"已生成7个可视化图表，保存在: {plots_dir}")
    print("图表说明:")
    print("  1. stability_distribution.png - 稳定度分布饼图")
    print("  2. alpha_distribution.png - α值分布直方图（按稳定度着色）")
    print("  3. stability_daily_variation.png - 稳定度日变化（含日出日落线）")
    print("  4. confidence_distribution.png - 置信度分布")
    print("  5. stability_seasonal_variation.png - 稳定度季节变化")
    print("  6. alpha_timeseries.png - α值时间序列图")
    print("  7. wind_profiles_by_stability.png - 不同稳定度下的风速剖面图")

if __name__ == "__main__":
    main()