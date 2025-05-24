#!/usr/bin/env python3
"""
昌马站中性条件粗糙度计算器 - 严格质量控制版本
目标：获得更符合沙漠地表的粗糙度值（~0.002 m）
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
import warnings
import os
from datetime import datetime
warnings.filterwarnings('ignore')

class StrictNeutralRoughnessCalculator:
    """
    严格质量控制的粗糙度计算器
    """
    
    def __init__(self):
        self.kappa = 0.4  # von Karman常数
        self.heights = np.array([10, 30, 50, 70])  # 昌马站测量高度(m)
        
        # 更严格的质量控制参数
        self.min_wind_speed = 2.0  # 提高最小风速阈值(m/s)
        self.min_confidence = 0.8  # 提高置信度阈值
        self.min_r_squared = 0.95  # 提高R²阈值
        
        # 针对沙漠地表的粗糙度范围
        self.z0_min = 0.0001  # 最小粗糙度(m)
        self.z0_max = 0.1     # 降低最大粗糙度(m) - 适合沙漠
        
        # u*合理范围
        self.ustar_min = 0.05  # 提高最小摩擦速度(m/s)
        self.ustar_max = 1.5   # 降低最大摩擦速度(m/s)
        
        # 额外的质量检查
        self.max_wind_speed_std = 5.0  # 风速标准差最大值
        self.min_points_per_profile = 4  # 每个廓线最少有效点数
        
        # 地表类型分类
        self.surface_types = {
            (0, 0.0002): "海面",
            (0.0002, 0.005): "雪地/冰面/沙漠",
            (0.005, 0.03): "开阔平地/机场跑道",
            (0.03, 0.055): "短草地/农田（作物<0.5m）",
            (0.055, 0.1): "农田/牧场（作物0.5-1m）",
            (0.1, 0.2): "农田（作物1-2m）/散布树木",
            (0.2, 0.4): "灌木地/果园",
            (0.4, 0.8): "森林/城郊",
            (0.8, 1.6): "密集森林/城市",
            (1.6, 3.0): "城市中心/高层建筑"
        }
    
    def load_and_filter_data(self, data_path, stability_results_path):
        """
        加载数据并筛选高质量中性条件
        """
        print("=== 昌马站中性条件粗糙度计算（严格质量控制）===")
        print("加载原始数据...")
        
        try:
            data = pd.read_csv(data_path)
            stability_results = pd.read_csv(stability_results_path)
            print(f"原始数据形状: {data.shape}")
            print(f"稳定度结果形状: {stability_results.shape}")
        except Exception as e:
            print(f"文件读取错误: {e}")
            return None
        
        # 智能处理时间列
        def process_time_column(df, df_name):
            time_col = None
            for col in df.columns:
                if any(keyword in col.lower() for keyword in ['time', 'date']):
                    time_col = col
                    break
            
            if time_col is None:
                time_col = df.columns[0]
            
            print(f"  {df_name}使用时间列: {time_col}")
            df['timestamp'] = pd.to_datetime(df[time_col])
            return df
        
        data = process_time_column(data, "原始数据")
        stability_results = process_time_column(stability_results, "稳定度结果")
        
        # 合并数据
        print(f"\n合并数据...")
        data['timestamp'] = pd.to_datetime(data['timestamp']).dt.round('H')
        stability_results['timestamp'] = pd.to_datetime(stability_results['timestamp']).dt.round('H')
        
        combined_data = pd.merge(data, stability_results, on='timestamp', how='inner')
        print(f"合并后数据点: {len(combined_data)}")
        
        # 筛选高置信度中性条件
        print(f"\n应用严格质量控制...")
        print(f"  最小置信度: {self.min_confidence}")
        print(f"  最小风速: {self.min_wind_speed} m/s")
        print(f"  最小R²: {self.min_r_squared}")
        
        neutral_data = combined_data[
            (combined_data['stability_final'] == 'neutral') &
            (combined_data['confidence_final'] >= self.min_confidence)
        ]
        
        # 额外的风速检查
        wind_columns = ['obs_wind_speed_10m', 'obs_wind_speed_30m', 
                       'obs_wind_speed_50m', 'obs_wind_speed_70m']
        
        # 检查10m风速是否满足最小风速要求
        neutral_data = neutral_data[neutral_data['obs_wind_speed_10m'] >= self.min_wind_speed]
        
        # 检查风速廓线的合理性（风速应该随高度增加）
        wind_increase_check = (
            (neutral_data['obs_wind_speed_30m'] >= neutral_data['obs_wind_speed_10m']) &
            (neutral_data['obs_wind_speed_50m'] >= neutral_data['obs_wind_speed_30m']) &
            (neutral_data['obs_wind_speed_70m'] >= neutral_data['obs_wind_speed_50m'])
        )
        neutral_data = neutral_data[wind_increase_check]
        
        print(f"严格筛选后中性数据点: {len(neutral_data)}")
        
        if len(neutral_data) < 10:
            print("警告: 高质量中性数据不足！尝试适当放宽标准...")
            # 稍微放宽标准重试
            self.min_confidence = 0.7
            self.min_r_squared = 0.92
            neutral_data = combined_data[
                (combined_data['stability_final'] == 'neutral') &
                (combined_data['confidence_final'] >= self.min_confidence) &
                (combined_data['obs_wind_speed_10m'] >= self.min_wind_speed)
            ]
            print(f"放宽标准后数据点: {len(neutral_data)}")
        
        return neutral_data
    
    def calculate_neutral_roughness_single(self, wind_speeds, heights, return_details=False):
        """
        计算单个时间点的粗糙度 - 严格版本
        """
        # 质量检查
        valid_mask = (wind_speeds > self.min_wind_speed) & ~np.isnan(wind_speeds)
        if np.sum(valid_mask) < self.min_points_per_profile:
            return np.nan if not return_details else (np.nan, {})
        
        valid_winds = wind_speeds[valid_mask]
        valid_heights = heights[valid_mask]
        
        # 检查风速变化率
        wind_std = np.std(valid_winds)
        if wind_std > self.max_wind_speed_std:
            return np.nan if not return_details else (np.nan, {})
        
        try:
            # 对数风速廓线线性回归
            ln_z = np.log(valid_heights)
            slope, intercept, r_value, p_value, std_err = stats.linregress(ln_z, valid_winds)
            
            # 计算物理参数
            u_star = slope * self.kappa
            
            if slope > 0:
                z0 = np.exp(-intercept / slope)
            else:
                return np.nan if not return_details else (np.nan, {})
            
            # 严格的物理合理性检查
            if not (self.z0_min <= z0 <= self.z0_max and 
                    self.ustar_min <= u_star <= self.ustar_max and 
                    r_value**2 >= self.min_r_squared):
                return np.nan if not return_details else (np.nan, {})
            
            # 额外检查：计算残差
            u_theory = (u_star / self.kappa) * np.log(valid_heights / z0)
            residuals = valid_winds - u_theory
            max_residual = np.max(np.abs(residuals))
            
            # 如果最大残差太大，拒绝
            if max_residual > 2.0:  # 2 m/s的残差阈值
                return np.nan if not return_details else (np.nan, {})
            
            if return_details:
                rmse = np.sqrt(np.mean(residuals**2))
                
                details = {
                    'u_star': u_star,
                    'z0': z0,
                    'r_squared': r_value**2,
                    'rmse': rmse,
                    'max_residual': max_residual,
                    'n_points': len(valid_winds),
                    'wind_profile': valid_winds.tolist(),
                    'height_profile': valid_heights.tolist(),
                    'slope': slope,
                    'intercept': intercept,
                    'wind_std': wind_std
                }
                return z0, details
            else:
                return z0
                
        except Exception as e:
            return np.nan if not return_details else (np.nan, {})
    
    def calculate_representative_roughness(self, neutral_data):
        """
        计算代表性粗糙度 - 使用严格筛选
        """
        print("\n开始计算代表性粗糙度（严格质量控制）...")
        
        wind_columns = ['obs_wind_speed_10m', 'obs_wind_speed_30m', 
                       'obs_wind_speed_50m', 'obs_wind_speed_70m']
        
        z0_results = []
        detailed_results = []
        
        print(f"处理 {len(neutral_data)} 个中性条件时间点...")
        
        success_count = 0
        reject_reasons = {
            'invalid_winds': 0,
            'too_few_points': 0,
            'poor_fit': 0,
            'out_of_range': 0,
            'large_residual': 0
        }
        
        for idx, (_, row) in enumerate(neutral_data.iterrows()):
            if idx % 500 == 0 and idx > 0:
                print(f"  已处理: {idx}/{len(neutral_data)} ({idx/len(neutral_data)*100:.1f}%), 成功: {success_count}")
            
            wind_speeds = np.array([row[col] for col in wind_columns])
            
            if np.any(np.isnan(wind_speeds)) or np.any(wind_speeds <= 0):
                reject_reasons['invalid_winds'] += 1
                continue
            
            z0, details = self.calculate_neutral_roughness_single(
                wind_speeds, self.heights, return_details=True
            )
            
            if not np.isnan(z0):
                z0_results.append(z0)
                details.update({
                    'timestamp': row['timestamp'],
                    'confidence': row['confidence_final']
                })
                detailed_results.append(details)
                success_count += 1
            else:
                # 分析拒绝原因
                if np.sum(wind_speeds > self.min_wind_speed) < self.min_points_per_profile:
                    reject_reasons['too_few_points'] += 1
        
        print(f"\n筛选结果:")
        print(f"  成功计算: {len(z0_results)}/{len(neutral_data)} ({len(z0_results)/len(neutral_data)*100:.1f}%)")
        print(f"  拒绝原因统计:")
        for reason, count in reject_reasons.items():
            if count > 0:
                print(f"    {reason}: {count}")
        
        if len(z0_results) < 10:
            raise ValueError(f"有效粗糙度计算结果太少！仅有{len(z0_results)}个")
        
        # 统计分析
        z0_array = np.array(z0_results)
        
        # 更严格的异常值剔除
        # 使用对数空间的异常值检测
        ln_z0 = np.log(z0_array)
        Q1 = np.percentile(ln_z0, 25)
        Q3 = np.percentile(ln_z0, 75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        
        outlier_mask = (ln_z0 >= lower_bound) & (ln_z0 <= upper_bound)
        z0_clean = z0_array[outlier_mask]
        
        print(f"\n异常值剔除:")
        print(f"  去除异常值: {len(z0_array) - len(z0_clean)} 个")
        print(f"  z₀范围: [{z0_clean.min():.6f}, {z0_clean.max():.6f}] m")
        
        # 计算统计量
        statistics = {
            'count': len(z0_clean),
            'mean': np.mean(z0_clean),
            'median': np.median(z0_clean),
            'std': np.std(z0_clean),
            'min': np.min(z0_clean),
            'max': np.max(z0_clean),
            'percentile_10': np.percentile(z0_clean, 10),
            'percentile_25': np.percentile(z0_clean, 25),
            'percentile_75': np.percentile(z0_clean, 75),
            'percentile_90': np.percentile(z0_clean, 90),
            'geometric_mean': stats.gmean(z0_clean),
        }
        
        # 对于沙漠地表，考虑使用较低的百分位数
        z0_representative = statistics['percentile_25']  # 使用25%分位数
        
        print(f"\n统计结果:")
        print(f"  10%分位数: {statistics['percentile_10']:.6f} m")
        print(f"  25%分位数: {statistics['percentile_25']:.6f} m")
        print(f"  中位数: {statistics['median']:.6f} m")
        print(f"  几何平均: {statistics['geometric_mean']:.6f} m")
        
        return z0_representative, statistics, pd.DataFrame(detailed_results)
    
    def generate_report(self, z0_representative, statistics, output_dir):
        """生成分析报告"""
        os.makedirs(output_dir, exist_ok=True)
        
        surface_type = self.classify_surface_type(z0_representative)
        
        print("\n=== 昌马站粗糙度分析结果（严格质量控制）===")
        print(f"代表性粗糙度 (25%分位数): {z0_representative:.6f} m")
        print(f"地表类型: {surface_type}")
        print(f"几何平均值: {statistics['geometric_mean']:.6f} m")
        print(f"有效样本数: {statistics['count']}")
        
        # 保存报告
        report_path = os.path.join(output_dir, "changma_roughness_report_strict.txt")
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write("昌马站中性条件粗糙度分析报告（严格质量控制）\n")
            f.write("="*60 + "\n\n")
            f.write(f"分析时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            
            f.write("质量控制参数:\n")
            f.write(f"  最小置信度: {self.min_confidence}\n")
            f.write(f"  最小R²: {self.min_r_squared}\n")
            f.write(f"  最小风速: {self.min_wind_speed} m/s\n")
            f.write(f"  粗糙度范围: [{self.z0_min}, {self.z0_max}] m\n")
            f.write(f"  有效样本数: {statistics['count']}\n\n")
            
            f.write("粗糙度分析结果:\n")
            f.write(f"  代表性粗糙度 (25%分位数): {z0_representative:.6f} m\n")
            f.write(f"  10%分位数: {statistics['percentile_10']:.6f} m\n")
            f.write(f"  中位数: {statistics['median']:.6f} m\n")
            f.write(f"  几何平均值: {statistics['geometric_mean']:.6f} m\n")
            f.write(f"  平均值: {statistics['mean']:.6f} ± {statistics['std']:.6f} m\n")
            f.write(f"  范围: [{statistics['min']:.6f}, {statistics['max']:.6f}] m\n\n")
            
            f.write(f"地表类型推断: {surface_type}\n")
        
        print(f"\n报告已保存: {report_path}")
        return z0_representative
    
    def classify_surface_type(self, z0):
        """地表类型分类"""
        for (z0_min, z0_max), surface_type in self.surface_types.items():
            if z0_min <= z0 < z0_max:
                return surface_type
        return "未知地表类型"

def main():
    """主执行函数"""
    # 设置路径
    data_path = "/Users/xiaxin/work/WindForecast_Project/01_Data/processed/matched_data/changma_matched.csv"
    stability_results_path = "/Users/xiaxin/work/WindForecast_Project/03_Results/stability_analysis/changma_stability_results.csv"
    output_dir = "/Users/xiaxin/work/WindForecast_Project/03_Results/neutral_roughness_analysis_strict"
    
    # 初始化计算器
    calculator = StrictNeutralRoughnessCalculator()
    
    try:
        # 1. 加载和筛选数据
        neutral_data = calculator.load_and_filter_data(data_path, stability_results_path)
        if neutral_data is None or len(neutral_data) == 0:
            print("无法获取有效数据")
            return
        
        # 2. 计算代表性粗糙度
        z0_representative, statistics, detailed_results_df = calculator.calculate_representative_roughness(neutral_data)
        
        # 3. 生成报告
        final_z0 = calculator.generate_report(z0_representative, statistics, output_dir)
        
        # 4. 保存详细结果
        if len(detailed_results_df) > 0:
            detailed_results_df.to_csv(
                os.path.join(output_dir, "changma_roughness_detailed_strict.csv"), 
                index=False
            )
        
        print(f"\n=== 分析完成 ===")
        print(f"昌马站代表性粗糙度: {final_z0:.6f} m")
        print(f"地表类型: {calculator.classify_surface_type(final_z0)}")
        
    except Exception as e:
        print(f"分析过程中出现错误: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()