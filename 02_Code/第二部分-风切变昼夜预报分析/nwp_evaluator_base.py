#!/usr/bin/env python3
"""
数值预报评估器 - 基础模块
包含数据加载、预处理和评估指标计算
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from scipy import stats
import warnings
import os
warnings.filterwarnings('ignore')

# 设置中文显示
plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial Unicode MS', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

class NWPEvaluatorBase:
    def __init__(self, data_path, save_path):
        self.data_path = data_path
        self.save_path = save_path
        self.data = None
        self.evaluation_results = {}
        
        # 创建保存目录
        os.makedirs(save_path, exist_ok=True)
        
        # 三级风切变阈值
        self.shear_thresholds = {
            'weak_upper': 0.2,
            'moderate_upper': 0.3,
        }
        
        # 重点评估的变量
        self.key_variables = {
            'wind_speed_10m': {
                'obs': 'obs_wind_speed_10m', 
                'ec': 'ec_wind_speed_10m', 
                'gfs': 'gfs_wind_speed_10m', 
                'name': '10m风速',
                'unit': 'm/s'
            },
            'wind_speed_70m': {
                'obs': 'obs_wind_speed_70m', 
                'ec': 'ec_wind_speed_70m', 
                'gfs': 'gfs_wind_speed_70m', 
                'name': '70m风速',
                'unit': 'm/s'
            },
            'temperature_10m': {
                'obs': 'obs_temperature_10m', 
                'ec': 'ec_temperature_10m', 
                'gfs': 'gfs_temperature_10m', 
                'name': '10m温度',
                'unit': '°C'
            }
        }
        
        # 评估指标的中文名称
        self.metric_names = {
            'R2': 'R²决定系数',
            'RMSE': '均方根误差',
            'MAE': '平均绝对误差',
            'BIAS': '平均偏差',
            'CORR': '相关系数'
        }
        
    def load_and_prepare_data(self):
        """加载和预处理数据"""
        print("📊 加载和预处理数据...")
        
        try:
            self.data = pd.read_csv(self.data_path)
            print(f"原始数据形状: {self.data.shape}")
        except Exception as e:
            print(f"❌ 数据加载失败: {e}")
            return None
        
        # 转换datetime列
        if 'datetime' in self.data.columns:
            self.data['datetime'] = pd.to_datetime(self.data['datetime'])
        else:
            print("❌ 未找到datetime列")
            return None
        
        # 检查和清理关键变量
        key_columns = []
        for var_info in self.key_variables.values():
            key_columns.extend([var_info['obs'], var_info['ec'], var_info['gfs']])
        
        # 检查列是否存在
        missing_columns = [col for col in key_columns if col not in self.data.columns]
        if missing_columns:
            print(f"❌ 缺少必要列: {missing_columns}")
            return None
        
        # 添加power列检查
        if 'power' in self.data.columns:
            key_columns.append('power')
        
        # 移除缺失值
        before_clean = len(self.data)
        self.data = self.data.dropna(subset=key_columns)
        
        if 'power' in self.data.columns:
            self.data = self.data[self.data['power'] >= 0]
        
        after_clean = len(self.data)
        
        print(f"清理后数据: {after_clean} 行 (移除了 {before_clean - after_clean} 行)")
        
        if after_clean == 0:
            print("❌ 清理后无有效数据")
            return None
        
        # 计算风切变分类
        return self._calculate_wind_shear_classification()
    
    def _calculate_wind_shear_classification(self):
        """计算风切变并进行分类"""
        print("🌪️ 计算风切变系数并分类...")
        
        # 计算风切变系数
        v1 = self.data['obs_wind_speed_10m']
        v2 = self.data['obs_wind_speed_70m']
        h1, h2 = 10, 70
        
        # 过滤有效数据
        valid_mask = (v1 > 0.5) & (v2 > 0.5)
        valid_count = valid_mask.sum()
        
        if valid_count == 0:
            print("❌ 无有效的风速数据计算风切变")
            return None
        
        self.data = self.data[valid_mask].copy()
        v1, v2 = v1[valid_mask], v2[valid_mask]
        
        print(f"有效风速数据: {valid_count} 条")
        
        # 计算风切变系数
        self.data['wind_shear_alpha'] = np.log(v2 / v1) / np.log(h2 / h1)
        
        # 三级风切变分类
        alpha = self.data['wind_shear_alpha']
        conditions = [
            alpha < self.shear_thresholds['weak_upper'],
            (alpha >= self.shear_thresholds['weak_upper']) & (alpha < self.shear_thresholds['moderate_upper']),
            alpha >= self.shear_thresholds['moderate_upper']
        ]
        choices = ['weak', 'moderate', 'strong']
        self.data['shear_group'] = np.select(conditions, choices, default='unknown')
        
        # 昼夜分类
        self.data['hour'] = self.data['datetime'].dt.hour
        self.data['is_daytime'] = ((self.data['hour'] >= 6) & (self.data['hour'] < 18))
        
        # 组合分类
        self.data['shear_diurnal_class'] = self.data['shear_group'].astype(str) + '_' + \
                                         self.data['is_daytime'].map({True: 'day', False: 'night'})
        
        # 统计分类
        class_counts = self.data['shear_diurnal_class'].value_counts()
        print(f"✓ 风切变分类完成:")
        for class_name, count in class_counts.items():
            if 'unknown' not in class_name:
                percentage = count / len(self.data) * 100
                print(f"  {class_name}: {count} 条 ({percentage:.1f}%)")
        
        return class_counts
    
    def calculate_metrics(self, obs, forecast):
        """计算评估指标，增强鲁棒性"""
        obs = np.array(obs)
        forecast = np.array(forecast)
        
        # 移除缺失值
        valid_mask = ~(np.isnan(obs) | np.isnan(forecast) | np.isinf(obs) | np.isinf(forecast))
        obs_clean = obs[valid_mask]
        forecast_clean = forecast[valid_mask]
        
        if len(obs_clean) < 20:  # 确保足够的样本
            return {
                'RMSE': np.nan, 'MAE': np.nan, 'R2': np.nan, 
                'BIAS': np.nan, 'CORR': np.nan, 'COUNT': len(obs_clean)
            }
        
        try:
            rmse = np.sqrt(mean_squared_error(obs_clean, forecast_clean))
            mae = mean_absolute_error(obs_clean, forecast_clean)
            r2 = r2_score(obs_clean, forecast_clean)
            bias = np.mean(forecast_clean - obs_clean)
            
            # 计算相关系数
            if len(obs_clean) > 1 and np.std(obs_clean) > 0 and np.std(forecast_clean) > 0:
                corr = np.corrcoef(obs_clean, forecast_clean)[0, 1]
            else:
                corr = np.nan
            
            return {
                'RMSE': rmse,
                'MAE': mae,
                'R2': r2,
                'BIAS': bias,
                'CORR': corr,
                'COUNT': len(obs_clean)
            }
        except Exception as e:
            print(f"指标计算错误: {e}")
            return {
                'RMSE': np.nan, 'MAE': np.nan, 'R2': np.nan, 
                'BIAS': np.nan, 'CORR': np.nan, 'COUNT': len(obs_clean)
            }
    
    def evaluate_by_classification(self):
        """按分类评估预报性能"""
        print("📈 按分类评估数值预报性能...")
        
        self.evaluation_results = {}
        unique_classes = [cls for cls in self.data['shear_diurnal_class'].unique() if 'unknown' not in cls]
        
        for class_name in unique_classes:
            class_data = self.data[self.data['shear_diurnal_class'] == class_name]
            
            if len(class_data) < 50:  # 确保足够样本
                print(f"跳过 {class_name}: 样本数不足 ({len(class_data)} < 50)")
                continue
                
            print(f"评估 {class_name}: {len(class_data)} 条样本")
            
            self.evaluation_results[class_name] = {}
            
            for var_name, var_info in self.key_variables.items():
                obs_col = var_info['obs']
                ec_col = var_info['ec']
                gfs_col = var_info['gfs']
                
                if all(col in class_data.columns for col in [obs_col, ec_col, gfs_col]):
                    ec_metrics = self.calculate_metrics(class_data[obs_col], class_data[ec_col])
                    gfs_metrics = self.calculate_metrics(class_data[obs_col], class_data[gfs_col])
                    
                    self.evaluation_results[class_name][var_name] = {
                        'EC': ec_metrics,
                        'GFS': gfs_metrics
                    }
                else:
                    print(f"  警告: {var_name} 缺少必要列")
        
        print(f"✓ 完成 {len(self.evaluation_results)} 个分类的评估")
        return self.evaluation_results
    
    def check_data_quality(self):
        """检查数据质量"""
        print("🔍 检查数据质量...")
        
        if self.data is None:
            print("❌ 数据未加载")
            return False
        
        # 检查基本信息
        print(f"数据形状: {self.data.shape}")
        print(f"时间范围: {self.data['datetime'].min()} 到 {self.data['datetime'].max()}")
        
        # 检查关键变量的覆盖率
        for var_name, var_info in self.key_variables.items():
            obs_col = var_info['obs']
            ec_col = var_info['ec']
            gfs_col = var_info['gfs']
            
            obs_valid = (~self.data[obs_col].isna()).sum()
            ec_valid = (~self.data[ec_col].isna()).sum()
            gfs_valid = (~self.data[gfs_col].isna()).sum()
            
            total = len(self.data)
            print(f"{var_info['name']}:")
            print(f"  观测: {obs_valid}/{total} ({obs_valid/total*100:.1f}%)")
            print(f"  EC: {ec_valid}/{total} ({ec_valid/total*100:.1f}%)")
            print(f"  GFS: {gfs_valid}/{total} ({gfs_valid/total*100:.1f}%)")
        
        # 检查分类覆盖
        if 'shear_diurnal_class' in self.data.columns:
            class_counts = self.data['shear_diurnal_class'].value_counts()
            print(f"分类分布:")
            for class_name, count in class_counts.items():
                print(f"  {class_name}: {count} 条")
        
        return True

if __name__ == "__main__":
    # 测试基础功能
    DATA_PATH = "/Users/xiaxin/work/WindForecast_Project/01_Data/processed/imputed_data/changma_imputed_complete.csv"
    SAVE_PATH = "/Users/xiaxin/work/WindForecast_Project/03_Results/clear_nwp_evaluation_results"
    
    # 创建评估器
    evaluator = NWPEvaluatorBase(DATA_PATH, SAVE_PATH)
    
    # 加载数据
    if evaluator.load_and_prepare_data() is not None:
        print("✓ 数据加载成功")
        
        # 检查数据质量
        evaluator.check_data_quality()
        
        # 评估性能
        evaluator.evaluate_by_classification()
        print("✓ 基础评估完成")
    else:
        print("❌ 数据加载失败")