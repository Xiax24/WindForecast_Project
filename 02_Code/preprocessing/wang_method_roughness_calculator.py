#!/usr/bin/env python3
"""
基于Wang et al. (2024) GRL文章方法的昌马站粗糙度计算器

主要改进：
1. 采用d = 2/3 × z₀的零平面位移高度关系
2. 通过最小化RMSE优化ln(z₀)
3. 增加摩擦速度一致性检查
4. 更严格的质量控制标准
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
from scipy.optimize import minimize_scalar
import warnings
import os
from datetime import datetime
warnings.filterwarnings('ignore')

class WangMethodRoughnessCalculator:
    """
    基于Wang et al. (2024)方法的粗糙度计算器
    """
    
    def __init__(self):
        self.kappa = 0.4  # von Karman常数
        self.heights = np.array([10, 30, 50, 70])  # 昌马站测量高度(m)
        
        # 质量控制参数（基于Wang et al. 2024）
        self.min_wind_speed = 0.4   # 最小风速阈值(m/s)
        self.min_confidence = 0.8   # 最小置信度阈值
        self.max_ustar_std = 0.05   # u*标准差最大值（5%）
        
        # 粗糙度合理范围
        self.ln_z0_min = -8.0  # ln(z₀)最小值
        self.ln_z0_max = 1.0   # ln(z₀)最大值
        
        # 地表类型分类（来自文章）
        self.surface_types = {
            (0, 0.0002): "海面/冰面",
            (0.0002, 0.005): "雪地/沙漠",
            (0.005, 0.03): "开阔平地",
            (0.03, 0.055): "短草地",
            (0.055, 0.1): "农田/牧场",
            (0.1, 0.2): "高作物/散布树木",
            (0.2, 0.4): "灌木地/果园",
            (0.4, 0.8): "森林/城郊",
            (0.8, 1.6): "密集森林/城市",
            (1.6, 3.0): "城市中心"
        }
    
    def calculate_neutral_roughness_wang_method(self, wind_speeds, heights):
        """
        使用Wang et al. (2024)方法计算单个时间点的粗糙度
        
        关键改进：
        1. 使用d = 2/3 × z₀关系
        2. 通过最小化RMSE优化ln(z₀)
        3. 计算u*的一致性
        """
        # 质量检查
        valid_mask = (wind_speeds > self.min_wind_speed) & ~np.isnan(wind_speeds)
        if np.sum(valid_mask) < 3:
            return np.nan, {}
        
        valid_winds = wind_speeds[valid_mask]
        valid_heights = heights[valid_mask]
        
        def calculate_rmse_for_ln_z0(ln_z0):
            """计算给定ln(z₀)的RMSE"""
            z0 = np.exp(ln_z0)
            d = 2/3 * z0  # Wang et al.使用的关系
            
            # 确保所有高度都大于d
            if np.any(valid_heights <= d):
                return 1e10
            
            # 线性回归计算u*
            ln_z_minus_d = np.log(valid_heights - d)
            slope, intercept, _, _, _ = stats.linregress(ln_z_minus_d, valid_winds)
            
            if slope <= 0:  # 物理上不合理
                return 1e10
            
            u_star = slope * self.kappa
            
            # 计算预测风速
            u_pred = (u_star / self.kappa) * ln_z_minus_d
            
            # 计算RMSE
            rmse = np.sqrt(np.mean((valid_winds - u_pred)**2))
            return rmse
        
        # 优化ln(z₀)以最小化RMSE
        result = minimize_scalar(
            calculate_rmse_for_ln_z0,
            bounds=(self.ln_z0_min, self.ln_z0_max),
            method='bounded'
        )
        
        if not result.success:
            return np.nan, {}
        
        optimal_ln_z0 = result.x
        z0 = np.exp(optimal_ln_z0)
        d = 2/3 * z0
        
        # 检查物理合理性
        if np.any(valid_heights <= d):
            return np.nan, {}
        
        # 计算最终参数
        ln_z_minus_d = np.log(valid_heights - d)
        slope, intercept, r_value, _, _ = stats.linregress(ln_z_minus_d, valid_winds)
        u_star = slope * self.kappa
        
        # 计算每个高度的u*并检查一致性
        u_star_array = np.zeros(len(valid_heights))
        for i, (h, u) in enumerate(zip(valid_heights, valid_winds)):
            if h > d:
                u_star_array[i] = (u * self.kappa) / np.log((h - d) / z0)
        
        u_star_std = np.std(u_star_array) / np.mean(u_star_array)
        
        # 质量检查：u*的相对标准差应小于5%
        if u_star_std > self.max_ustar_std:
            return np.nan, {}
        
        # 返回结果
        details = {
            'z0': z0,
            'd': d,
            'u_star': u_star,
            'u_star_std': u_star_std,
            'r_squared': r_value**2,
            'rmse': result.fun,
            'n_points': len(valid_winds),
            'wind_profile': valid_winds.tolist(),
            'height_profile': valid_heights.tolist()
        }
        
        return z0, details
    
    def load_and_process_data(self, data_path, stability_results_path):
        """加载并处理数据"""
        print("=== 基于Wang et al. (2024)方法的昌马站粗糙度计算 ===")
        print("加载数据...")
        
        try:
            # 加载数据
            data = pd.read_csv(data_path)
            stability_results = pd.read_csv(stability_results_path)
            
            print(f"原始数据列: {data.columns.tolist()[:10]}...")
            print(f"稳定度数据列: {stability_results.columns.tolist()[:10]}...")
            
            # 智能处理时间列
            def find_and_process_time_column(df, df_name):
                time_candidates = ['timestamp', 'time', 'datetime', 'date', 'Time', 'Timestamp']
                time_col = None
                
                # 查找时间列
                for col in time_candidates:
                    if col in df.columns:
                        time_col = col
                        break
                
                # 如果没找到，查找包含time的列
                if time_col is None:
                    for col in df.columns:
                        if 'time' in col.lower() or 'date' in col.lower():
                            time_col = col
                            break
                
                # 如果还是没找到，使用第一列
                if time_col is None:
                    time_col = df.columns[0]
                    print(f"  警告: {df_name}使用第一列作为时间: {time_col}")
                else:
                    print(f"  {df_name}使用时间列: {time_col}")
                
                # 转换为datetime并标准化列名
                df['timestamp'] = pd.to_datetime(df[time_col]).dt.round('H')
                return df
            
            # 处理两个数据框的时间列
            data = find_and_process_time_column(data, "原始数据")
            stability_results = find_and_process_time_column(stability_results, "稳定度结果")
            
            # 合并数据
            print("\n合并数据...")
            combined_data = pd.merge(data, stability_results, on='timestamp', how='inner')
            print(f"合并后数据点: {len(combined_data)}")
            
            # 筛选高置信度中性条件
            neutral_data = combined_data[
                (combined_data['stability_final'] == 'neutral') &
                (combined_data['confidence_final'] >= self.min_confidence)
            ]
            
            print(f"中性条件数据点: {len(neutral_data)}")
            
            if len(neutral_data) < 10:
                print("警告: 中性数据较少，尝试降低置信度阈值...")
                neutral_data = combined_data[
                    (combined_data['stability_final'] == 'neutral') &
                    (combined_data['confidence_final'] >= 0.6)
                ]
                print(f"降低阈值后中性数据点: {len(neutral_data)}")
                
                if len(neutral_data) < 10:
                    raise ValueError("中性数据仍然不足！")
            
            return neutral_data
            
        except Exception as e:
            print(f"数据处理错误: {e}")
            import traceback
            traceback.print_exc()
            return None
    
    def calculate_roughness_statistics(self, neutral_data):
        """计算粗糙度统计结果"""
        print("\n计算粗糙度...")
        
        wind_columns = ['obs_wind_speed_10m', 'obs_wind_speed_30m', 
                       'obs_wind_speed_50m', 'obs_wind_speed_70m']
        
        z0_results = []
        detailed_results = []
        
        total_samples = len(neutral_data)
        print(f"处理 {total_samples} 个中性条件样本...")
        
        for idx, (_, row) in enumerate(neutral_data.iterrows()):
            if idx % 500 == 0:
                print(f"  进度: {idx}/{total_samples} ({idx/total_samples*100:.1f}%)")
            
            wind_speeds = np.array([row[col] for col in wind_columns])
            
            if np.any(np.isnan(wind_speeds)) or np.any(wind_speeds <= 0):
                continue
            
            z0, details = self.calculate_neutral_roughness_wang_method(
                wind_speeds, self.heights
            )
            
            if not np.isnan(z0):
                z0_results.append(z0)
                details.update({
                    'timestamp': row['timestamp'],
                    'confidence': row['confidence_final']
                })
                detailed_results.append(details)
        
        print(f"\n成功计算: {len(z0_results)}/{total_samples} ({len(z0_results)/total_samples*100:.1f}%)")
        
        if len(z0_results) < 10:
            raise ValueError("有效计算结果太少！")
        
        # 统计分析
        z0_array = np.array(z0_results)
        
        # 使用Wang et al.方法：中位数作为代表值
        z0_representative = np.median(z0_array)
        
        statistics = {
            'count': len(z0_array),
            'median': z0_representative,
            'mean': np.mean(z0_array),
            'std': np.std(z0_array),
            'min': np.min(z0_array),
            'max': np.max(z0_array),
            'percentile_25': np.percentile(z0_array, 25),
            'percentile_75': np.percentile(z0_array, 75),
            'geometric_mean': stats.gmean(z0_array),
        }
        
        return z0_representative, statistics, pd.DataFrame(detailed_results)
    
    def create_comparison_plots(self, detailed_results_df, z0_representative, output_dir):
        """创建结果对比图"""
        plots_dir = os.path.join(output_dir, "wang_method_plots")
        os.makedirs(plots_dir, exist_ok=True)
        
        # 1. 粗糙度分布直方图
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.hist(detailed_results_df['z0'], bins=50, alpha=0.7, color='blue', 
                edgecolor='black', density=True)
        ax.axvline(z0_representative, color='red', linestyle='--', linewidth=2,
                   label=f'Median z₀: {z0_representative:.4f} m')
        ax.set_xlabel('Roughness Length z₀ (m)', fontsize=12)
        ax.set_ylabel('Probability Density', fontsize=12)
        ax.set_title('Changma Station z₀ Distribution (Wang et al. 2024 Method)', fontsize=14)
        ax.legend()
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(plots_dir, 'z0_distribution_wang_method.png'), dpi=300)
        plt.close()
        
        # 2. u*相对标准差分布
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.hist(detailed_results_df['u_star_std'] * 100, bins=30, alpha=0.7, 
                color='green', edgecolor='black')
        ax.axvline(5, color='red', linestyle='--', linewidth=2,
                   label='5% Threshold (Wang et al.)')
        ax.set_xlabel('Relative Standard Deviation of u* (%)', fontsize=12)
        ax.set_ylabel('Count', fontsize=12)
        ax.set_title('Quality Control: u* Consistency Check', fontsize=14)
        ax.legend()
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(plots_dir, 'ustar_consistency_check.png'), dpi=300)
        plt.close()
        
        # 3. RMSE分布
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.hist(detailed_results_df['rmse'], bins=30, alpha=0.7, 
                color='purple', edgecolor='black')
        ax.set_xlabel('RMSE (m/s)', fontsize=12)
        ax.set_ylabel('Count', fontsize=12)
        ax.set_title('Wind Profile Fitting RMSE Distribution', fontsize=14)
        ax.grid(True, alpha=0.3)
        
        # 添加统计信息
        rmse_text = f'Mean RMSE: {detailed_results_df["rmse"].mean():.3f} m/s\n'
        rmse_text += f'Median RMSE: {detailed_results_df["rmse"].median():.3f} m/s'
        ax.text(0.98, 0.98, rmse_text, transform=ax.transAxes,
                verticalalignment='top', horizontalalignment='right',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
        
        plt.tight_layout()
        plt.savefig(os.path.join(plots_dir, 'rmse_distribution.png'), dpi=300)
        plt.close()
    
    def generate_report(self, z0_representative, statistics, output_dir):
        """生成分析报告"""
        surface_type = self.classify_surface_type(z0_representative)
        
        print("\n=== 分析结果（Wang et al. 2024方法）===")
        print(f"代表性粗糙度 (中位数): {z0_representative:.6f} m")
        print(f"地表类型: {surface_type}")
        print(f"平均值: {statistics['mean']:.6f} m")
        print(f"标准差: {statistics['std']:.6f} m")
        print(f"有效样本数: {statistics['count']}")
        
        # 保存详细报告
        report_path = os.path.join(output_dir, "wang_method_roughness_report.txt")
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write("昌马站粗糙度分析报告 - Wang et al. (2024)方法\n")
            f.write("="*60 + "\n\n")
            f.write(f"分析时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            
            f.write("方法说明:\n")
            f.write("- 基于Wang et al. (2024) GRL文章方法\n")
            f.write("- 使用零平面位移高度 d = 2/3 × z₀\n")
            f.write("- 通过最小化RMSE优化ln(z₀)\n")
            f.write("- u*相对标准差 < 5%的质量控制\n\n")
            
            f.write("分析结果:\n")
            f.write(f"代表性粗糙度 (中位数): {z0_representative:.6f} m\n")
            f.write(f"地表类型推断: {surface_type}\n")
            f.write(f"平均值: {statistics['mean']:.6f} ± {statistics['std']:.6f} m\n")
            f.write(f"几何平均值: {statistics['geometric_mean']:.6f} m\n")
            f.write(f"范围: [{statistics['min']:.6f}, {statistics['max']:.6f}] m\n")
            f.write(f"四分位数: Q1={statistics['percentile_25']:.6f}, Q3={statistics['percentile_75']:.6f}\n")
            f.write(f"有效样本数: {statistics['count']}\n")
        
        print(f"\n报告已保存: {report_path}")
        return z0_representative
    
    def classify_surface_type(self, z0):
        """地表类型分类"""
        for (z0_min, z0_max), surface_type in self.surface_types.items():
            if z0_min <= z0 < z0_max:
                return surface_type
        return "未知地表类型"

def compare_methods(wang_z0, original_z0):
    """比较两种方法的结果"""
    print("\n=== 方法对比 ===")
    print(f"原方法粗糙度: {original_z0:.6f} m")
    print(f"Wang方法粗糙度: {wang_z0:.6f} m")
    print(f"相对差异: {(wang_z0 - original_z0) / original_z0 * 100:.1f}%")
    
    if abs(wang_z0 - original_z0) / original_z0 < 0.1:
        print("结论: 两种方法结果接近（差异<10%）")
    else:
        print("结论: 两种方法存在显著差异")

def main():
    """主执行函数"""
    # 设置路径
    data_path = "/Users/xiaxin/work/WindForecast_Project/01_Data/processed/matched_data/changma_matched.csv"
    stability_results_path = "/Users/xiaxin/work/WindForecast_Project/03_Results/stability_analysis/changma_stability_results.csv"
    output_dir = "/Users/xiaxin/work/WindForecast_Project/03_Results/wang_method_roughness_analysis"
    
    # 原方法的结果（从您之前的计算中获得）
    original_z0 = 0.002423  # 您之前计算的代表性粗糙度
    
    # 使用Wang方法计算
    calculator = WangMethodRoughnessCalculator()
    
    try:
        # 1. 加载数据
        neutral_data = calculator.load_and_process_data(data_path, stability_results_path)
        if neutral_data is None:
            return
        
        # 2. 计算粗糙度
        z0_representative, statistics, detailed_results_df = calculator.calculate_roughness_statistics(neutral_data)
        
        # 3. 创建对比图
        calculator.create_comparison_plots(detailed_results_df, z0_representative, output_dir)
        
        # 4. 生成报告
        wang_z0 = calculator.generate_report(z0_representative, statistics, output_dir)
        
        # 5. 保存详细结果
        detailed_results_df.to_csv(
            os.path.join(output_dir, "wang_method_detailed_results.csv"), 
            index=False
        )
        
        # 6. 方法对比
        compare_methods(wang_z0, original_z0)
        
    except Exception as e:
        print(f"分析错误: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()