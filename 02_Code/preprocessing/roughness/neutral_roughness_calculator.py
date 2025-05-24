#!/usr/bin/env python3
"""
修复版昌马站中性条件粗糙度计算器
修复了线性回归公式错误和风速剖面绘图问题

主要修复：
1. 正确的对数风速剖面线性回归实现
2. 修复z₀和u*的计算公式
3. 确保理论线与观测点正确匹配
4. 增强的调试和验证功能
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
import warnings
import os
from datetime import datetime
warnings.filterwarnings('ignore')

class FixedNeutralRoughnessCalculator:
    """
    修复版粗糙度计算器
    """
    
    def __init__(self):
        self.kappa = 0.4  # von Karman常数
        self.heights = np.array([10, 30, 50, 70])  # 昌马站测量高度(m)
        
        # 质量控制参数
        self.min_wind_speed = 1.0  # 最小风速阈值(m/s)
        self.min_confidence = 0.7  # 最小置信度阈值
        self.min_r_squared = 0.7   # 最小R²阈值
        
        # 粗糙度合理范围 - 调整为更现实的范围
        self.z0_min = 0.0001  # 最小粗糙度(m) - 增大下限
        self.z0_max = 1.0     # 最大粗糙度(m)
        
        # u*合理范围
        self.ustar_min = 0.01  # 最小摩擦速度(m/s)
        self.ustar_max = 2.0   # 最大摩擦速度(m/s)
        
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
        加载数据并筛选高置信度中性条件
        """
        print("=== 修复版昌马站中性条件粗糙度计算 ===")
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
            print(f"\n处理{df_name}的时间列...")
            time_candidates = ['timestamp', 'time', 'datetime', 'date']
            
            time_col = None
            for col in time_candidates:
                if col in df.columns:
                    time_col = col
                    break
            
            if time_col is None:
                time_cols = [col for col in df.columns if 
                           any(keyword in col.lower() for keyword in ['time', 'date'])]
                if time_cols:
                    time_col = time_cols[0]
                else:
                    time_col = df.columns[0]
                    print(f"  警告: 使用第一列作为时间: {time_col}")
            
            print(f"  使用时间列: {time_col}")
            df['timestamp'] = pd.to_datetime(df[time_col])
            return df
        
        data = process_time_column(data, "原始数据")
        stability_results = process_time_column(stability_results, "稳定度结果")
        
        # 检查必要的列
        required_wind_cols = ['obs_wind_speed_10m', 'obs_wind_speed_30m', 
                             'obs_wind_speed_50m', 'obs_wind_speed_70m']
        missing_wind_cols = [col for col in required_wind_cols if col not in data.columns]
        if missing_wind_cols:
            print(f"警告: 缺少风速列: {missing_wind_cols}")
            return None
        
        required_stability_cols = ['stability_final', 'confidence_final']
        missing_stability_cols = [col for col in required_stability_cols if col not in stability_results.columns]
        if missing_stability_cols:
            print(f"错误: 缺少稳定度列: {missing_stability_cols}")
            return None
        
        # 合并数据
        print(f"\n合并数据...")
        data['timestamp'] = pd.to_datetime(data['timestamp']).dt.round('H')
        stability_results['timestamp'] = pd.to_datetime(stability_results['timestamp']).dt.round('H')
        
        combined_data = pd.merge(data, stability_results, on='timestamp', how='inner')
        print(f"合并后数据点: {len(combined_data)}")
        
        if len(combined_data) == 0:
            print("错误: 合并后无数据！")
            return None
        
        # 筛选高置信度中性条件
        print(f"\n筛选中性条件...")
        neutral_data = combined_data[
            (combined_data['stability_final'] == 'neutral') &
            (combined_data['confidence_final'] >= self.min_confidence)
        ]
        
        print(f"高置信度中性数据点: {len(neutral_data)}")
        
        if len(neutral_data) < 10:
            raise ValueError("高置信度中性数据不足！")
        
        return neutral_data
    
    def calculate_neutral_roughness_single_fixed(self, wind_speeds, heights, return_details=False):
        """
        修复版：计算单个时间点的粗糙度
        
        正确的对数风速剖面公式：
        U(z) = (u*/κ) × ln(z/z₀)
        
        线性回归形式：
        U(z) = A × ln(z) + B
        其中：A = u*/κ, B = -(u*/κ) × ln(z₀)
        
        因此：u* = A × κ, z₀ = exp(-B/A)
        """
        # 质量检查
        valid_mask = (wind_speeds > self.min_wind_speed) & ~np.isnan(wind_speeds)
        if np.sum(valid_mask) < 3:
            return np.nan if not return_details else (np.nan, {})
        
        valid_winds = wind_speeds[valid_mask]
        valid_heights = heights[valid_mask]
        
        if np.any(valid_winds <= 0) or np.any(valid_heights <= 0):
            return np.nan if not return_details else (np.nan, {})
        
        try:
            # 关键修复：直接对风速和对数高度进行线性回归
            ln_z = np.log(valid_heights)
            
            # 线性回归: U = A * ln(z) + B
            slope, intercept, r_value, p_value, std_err = stats.linregress(ln_z, valid_winds)
            
            # 正确的参数提取
            A = slope          # A = u*/κ
            B = intercept      # B = -(u*/κ) * ln(z₀)
            
            # 计算物理参数
            u_star = A * self.kappa                    # u* = A × κ
            z0 = np.exp(-B / A) if A != 0 else np.nan  # z₀ = exp(-B/A)
            
            # 物理合理性检查
            if (self.z0_min <= z0 <= self.z0_max and 
                self.ustar_min <= u_star <= self.ustar_max and 
                r_value**2 >= self.min_r_squared):
                
                if return_details:
                    # 验证计算：重新计算理论风速
                    u_theory_check = (u_star / self.kappa) * np.log(valid_heights / z0)
                    rmse = np.sqrt(np.mean((valid_winds - u_theory_check)**2))
                    
                    details = {
                        'u_star': u_star,
                        'z0': z0,
                        'r_squared': r_value**2,
                        'rmse': rmse,
                        'n_points': len(valid_winds),
                        'wind_profile': valid_winds.tolist(),
                        'height_profile': valid_heights.tolist(),
                        'slope_A': A,
                        'intercept_B': B
                    }
                    return z0, details
                else:
                    return z0
            else:
                return np.nan if not return_details else (np.nan, {})
                
        except Exception as e:
            print(f"计算错误: {e}")
            return np.nan if not return_details else (np.nan, {})
    
    def calculate_representative_roughness(self, neutral_data):
        """
        计算代表性粗糙度
        """
        print("\n开始计算代表性粗糙度...")
        
        wind_columns = ['obs_wind_speed_10m', 'obs_wind_speed_30m', 
                       'obs_wind_speed_50m', 'obs_wind_speed_70m']
        
        z0_results = []
        detailed_results = []
        
        print(f"处理 {len(neutral_data)} 个中性条件时间点...")
        
        # 调试：显示前几个样本
        print("\n调试信息 - 前5个样本:")
        for idx, (_, row) in enumerate(neutral_data.head(5).iterrows()):
            wind_speeds = np.array([row[col] for col in wind_columns])
            print(f"样本 {idx+1}: 风速 = {wind_speeds}, 时间 = {row['timestamp']}")
        
        success_count = 0
        for idx, (_, row) in enumerate(neutral_data.iterrows()):
            if idx % 2000 == 0 and idx > 0:
                print(f"  已处理: {idx}/{len(neutral_data)}, 成功: {success_count}")
            
            try:
                wind_speeds = np.array([row[col] for col in wind_columns])
            except Exception as e:
                continue
            
            if np.any(np.isnan(wind_speeds)) or np.any(wind_speeds <= 0):
                continue
            
            # 使用修复版计算函数
            z0, details = self.calculate_neutral_roughness_single_fixed(
                wind_speeds, self.heights, return_details=True
            )
            
            if not np.isnan(z0):
                z0_results.append(z0)
                details.update({
                    'timestamp': row['timestamp'],
                    'confidence': row['confidence_final'],
                    'alpha_main': row.get('alpha_main', np.nan)
                })
                detailed_results.append(details)
                success_count += 1
        
        print(f"成功计算: {len(z0_results)}/{len(neutral_data)} ({len(z0_results)/len(neutral_data)*100:.1f}%)")
        
        if len(z0_results) < 10:
            raise ValueError(f"有效粗糙度计算结果太少！仅有{len(z0_results)}个")
        
        # 统计分析
        z0_array = np.array(z0_results)
        
        # 去除异常值
        Q1 = np.percentile(z0_array, 25)
        Q3 = np.percentile(z0_array, 75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        
        outlier_mask = (z0_array >= lower_bound) & (z0_array <= upper_bound)
        z0_clean = z0_array[outlier_mask]
        
        print(f"去除异常值: {len(z0_array) - len(z0_clean)} 个")
        print(f"z₀范围: [{z0_clean.min():.6f}, {z0_clean.max():.6f}] m")
        print(f"z₀中位数: {np.median(z0_clean):.6f} m")
        
        # 计算统计量
        statistics = {
            'count': len(z0_clean),
            'mean': np.mean(z0_clean),
            'median': np.median(z0_clean),
            'std': np.std(z0_clean),
            'min': np.min(z0_clean),
            'max': np.max(z0_clean),
            'percentile_25': np.percentile(z0_clean, 25),
            'percentile_75': np.percentile(z0_clean, 75),
            'geometric_mean': stats.gmean(z0_clean),
        }
        
        z0_representative = statistics['median']
        
        return z0_representative, statistics, pd.DataFrame(detailed_results)
    
    def seasonal_analysis(self, detailed_results_df):
        """季节变化分析"""
        if len(detailed_results_df) == 0:
            return {}
        
        detailed_results_df['timestamp'] = pd.to_datetime(detailed_results_df['timestamp'])
        detailed_results_df['month'] = detailed_results_df['timestamp'].dt.month
        detailed_results_df['season'] = detailed_results_df['month'].map({
            12: 'winter', 1: 'winter', 2: 'winter',
            3: 'spring', 4: 'spring', 5: 'spring',
            6: 'summer', 7: 'summer', 8: 'summer',
            9: 'autumn', 10: 'autumn', 11: 'autumn'
        })
        
        seasonal_stats = {}
        for season in ['spring', 'summer', 'autumn', 'winter']:
            season_data = detailed_results_df[detailed_results_df['season'] == season]
            if len(season_data) > 5:
                seasonal_stats[season] = {
                    'count': len(season_data),
                    'median_z0': np.median(season_data['z0']),
                    'mean_z0': np.mean(season_data['z0']),
                    'std_z0': np.std(season_data['z0'])
                }
        
        return seasonal_stats
    
    def classify_surface_type(self, z0):
        """地表类型分类"""
        for (z0_min, z0_max), surface_type in self.surface_types.items():
            if z0_min <= z0 < z0_max:
                return surface_type
        return "未知地表类型"
    
    def generate_report(self, z0_representative, statistics, seasonal_stats, output_dir):
        """生成分析报告"""
        os.makedirs(output_dir, exist_ok=True)
        
        print("\n=== 修复版昌马站粗糙度分析结果 ===")
        print(f"分析时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"有效样本数: {statistics['count']}")
        print(f"\n粗糙度统计:")
        print(f"  代表性粗糙度 (中位数): {z0_representative:.6f} m")
        print(f"  平均值: {statistics['mean']:.6f} m")
        print(f"  几何平均值: {statistics['geometric_mean']:.6f} m")
        print(f"  标准差: {statistics['std']:.6f} m")
        print(f"  范围: [{statistics['min']:.6f}, {statistics['max']:.6f}] m")
        
        surface_type = self.classify_surface_type(z0_representative)
        print(f"\n地表类型分类: {surface_type}")
        
        if seasonal_stats:
            print(f"\n季节变化分析:")
            for season, stats in seasonal_stats.items():
                print(f"  {season}: {stats['median_z0']:.6f} ± {stats['std_z0']:.6f} m (n={stats['count']})")
        
        # 保存报告
        report_path = os.path.join(output_dir, "changma_fixed_roughness_report.txt")
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write("昌马站中性条件粗糙度分析报告 - 修复版\n")
            f.write("="*50 + "\n\n")
            f.write(f"分析时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"修复内容: 正确的对数风速剖面线性回归实现\n")
            f.write(f"理论基础: U(z) = (u*/κ) × ln(z/z₀)\n\n")
            
            f.write("数据质量控制:\n")
            f.write(f"  最小置信度: {self.min_confidence}\n")
            f.write(f"  最小R²: {self.min_r_squared}\n")
            f.write(f"  有效样本数: {statistics['count']}\n\n")
            
            f.write("粗糙度分析结果:\n")
            f.write(f"  代表性粗糙度: {z0_representative:.6f} m\n")
            f.write(f"  平均值: {statistics['mean']:.6f} ± {statistics['std']:.6f} m\n")
            f.write(f"  中位数: {statistics['median']:.6f} m\n")
            f.write(f"  几何平均值: {statistics['geometric_mean']:.6f} m\n")
            f.write(f"  范围: [{statistics['min']:.6f}, {statistics['max']:.6f}] m\n\n")
            
            f.write(f"地表类型推断: {surface_type}\n\n")
            
            if seasonal_stats:
                f.write("季节变化:\n")
                for season, stats in seasonal_stats.items():
                    f.write(f"  {season}: {stats['median_z0']:.6f} ± {stats['std_z0']:.6f} m (n={stats['count']})\n")
        
        print(f"\n详细报告已保存: {report_path}")
        return z0_representative
    
    def create_all_plots(self, detailed_results_df, z0_representative, output_dir):
        """
        创建完整的6副分析图表，修复图例重叠问题
        """
        if len(detailed_results_df) == 0:
            print("无数据用于可视化")
            return
            
        plots_dir = os.path.join(output_dir, "plots")
        os.makedirs(plots_dir, exist_ok=True)
        
        # 1. 风速剖面验证图
        print("1. 创建风速剖面验证图...")
        self._create_wind_profile_verification(detailed_results_df, plots_dir)
        
        # 2. 粗糙度分布图
        print("2. 创建粗糙度分布图...")
        self._create_roughness_distribution(detailed_results_df, z0_representative, plots_dir)
        
        # 3. 拟合质量分布图
        print("3. 创建拟合质量分布图...")
        self._create_fit_quality_distribution(detailed_results_df, plots_dir)
        
        # 4. 季节变化图
        print("4. 创建季节变化图...")
        self._create_seasonal_variation(detailed_results_df, z0_representative, plots_dir)
        
        # 5. 摩擦速度分布图
        print("5. 创建摩擦速度分布图...")
        self._create_friction_velocity_distribution(detailed_results_df, plots_dir)
        
        # 6. 粗糙度vs摩擦速度关系图
        print("6. 创建粗糙度vs摩擦速度关系图...")
        self._create_roughness_vs_friction(detailed_results_df, z0_representative, plots_dir)
        
        print(f"\n所有图表已保存到: {plots_dir}")
        print("图表说明:")
        print("1. wind_profile_verification.png - 风速剖面验证（理论vs观测）")
        print("2. roughness_distribution.png - 粗糙度分布")
        print("3. fit_quality_distribution.png - 拟合质量分布")
        print("4. seasonal_variation.png - 季节变化趋势")
        print("5. friction_velocity_distribution.png - 摩擦速度分布")
        print("6. roughness_vs_friction.png - 粗糙度与摩擦速度关系")
    
    def _create_wind_profile_verification(self, detailed_results_df, plots_dir):
        """创建风速剖面验证图"""
        high_quality = detailed_results_df[detailed_results_df['r_squared'] > 0.9].head(6)
        
        if len(high_quality) == 0:
            print("无高质量样本用于验证")
            return
        
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        axes = axes.flatten()
        
        for idx, (_, row) in enumerate(high_quality.iterrows()):
            if idx >= 6:
                break
                
            ax = axes[idx]
            
            try:
                wind_profile = np.array(row['wind_profile'])
                height_profile = np.array(row['height_profile'])
                u_star = row['u_star']
                z0 = row['z0']
                r_squared = row['r_squared']
                
                # 计算理论拟合线
                z_theory = np.linspace(height_profile.min(), height_profile.max(), 100)
                u_theory = (u_star / self.kappa) * np.log(z_theory / z0)
                
                # 绘制观测点和理论线
                ax.plot(wind_profile, height_profile, 'ro', markersize=8, alpha=0.8,
                       label='Observations')
                ax.plot(u_theory, z_theory, 'b-', linewidth=2, alpha=0.8,
                       label='Theory')
                
                ax.set_xlabel('Wind Speed (m/s)', fontsize=10)
                ax.set_ylabel('Height (m)', fontsize=10)
                ax.set_title(f'Profile {idx+1}\nz₀={z0:.4f}m, u*={u_star:.3f}m/s, R²={r_squared:.3f}', 
                           fontsize=11)
                
                # 修复图例重叠 - 放在右下角
                ax.legend(fontsize=9, loc='lower right')
                ax.grid(True, alpha=0.3)
                
                # 验证匹配度 - 放在左上角
                u_theory_obs = (u_star / self.kappa) * np.log(height_profile / z0)
                match_error = np.mean(np.abs(wind_profile - u_theory_obs))
                ax.text(0.02, 0.98, f'{match_error:.3f} m/s', 
                       transform=ax.transAxes, verticalalignment='top',
                       bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.7, pad=0.3),
                       fontsize=9)
                
            except Exception as e:
                ax.text(0.5, 0.5, f'Error: {str(e)}', ha='center', va='center', 
                       transform=ax.transAxes)
                ax.set_title(f'Profile {idx+1} (Error)')
        
        # 隐藏多余的子图
        for i in range(len(high_quality), 6):
            axes[i].set_visible(False)
        
        plt.suptitle('Wind Speed Profile Verification\n(Red Points: Observations, Blue Lines: Theory)', 
                     fontsize=16)
        plt.tight_layout()
        plt.savefig(os.path.join(plots_dir, '1_wind_profile_verification.png'), 
                   dpi=300, bbox_inches='tight')
        plt.close()
    
    def _create_roughness_distribution(self, detailed_results_df, z0_representative, plots_dir):
        """创建粗糙度分布图"""
        fig, ax = plt.subplots(figsize=(12, 7))
        
        ax.hist(detailed_results_df['z0'], bins=50, alpha=0.7, color='steelblue', 
                edgecolor='black', density=True)
        ax.axvline(z0_representative, color='red', linestyle='--', linewidth=3,
                   label=f'Representative z₀: {z0_representative:.4f} m')
        ax.axvline(detailed_results_df['z0'].mean(), color='orange', linestyle=':', linewidth=2,
                   label=f'Mean z₀: {detailed_results_df["z0"].mean():.4f} m')
        
        ax.set_xlabel('Roughness Length z₀ (m)', fontsize=12)
        ax.set_ylabel('Probability Density', fontsize=12)
        ax.set_title('Changma Station Roughness Length Distribution (Fixed Version)', fontsize=14)
        
        # 修复图例重叠 - 放在右上角，添加边框
        ax.legend(fontsize=11, loc='upper right', frameon=True, fancybox=True, shadow=True)
        ax.grid(True, alpha=0.3)
        
        # 统计信息放在左上角
        stats_text = f'Sample Size: {len(detailed_results_df)}\n'
        stats_text += f'Median: {np.median(detailed_results_df["z0"]):.4f} m\n'
        stats_text += f'Range: [{detailed_results_df["z0"].min():.4f}, {detailed_results_df["z0"].max():.4f}] m'
        ax.text(0.02, 0.98, stats_text, transform=ax.transAxes,
                verticalalignment='top', horizontalalignment='left',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8, pad=0.5))
        
        plt.tight_layout()
        plt.savefig(os.path.join(plots_dir, '2_roughness_distribution.png'), 
                   dpi=300, bbox_inches='tight')
        plt.close()
    
    def _create_fit_quality_distribution(self, detailed_results_df, plots_dir):
        """创建拟合质量分布图"""
        fig, ax = plt.subplots(figsize=(12, 7))
        
        ax.hist(detailed_results_df['r_squared'], bins=30, alpha=0.7, color='green', 
                edgecolor='black', density=True)
        ax.axvline(detailed_results_df['r_squared'].mean(), color='red', linestyle='--', linewidth=2,
                   label=f'Mean R²: {detailed_results_df["r_squared"].mean():.3f}')
        ax.axvline(0.7, color='orange', linestyle=':', linewidth=2,
                   label='Minimum R² Threshold: 0.7')
        
        ax.set_xlabel('R² (Goodness of Fit)', fontsize=12)
        ax.set_ylabel('Probability Density', fontsize=12)
        ax.set_title('Quality of Logarithmic Wind Profile Fits (Fixed Version)', fontsize=14)
        
        # 图例放在左上角
        ax.legend(fontsize=11, loc='upper left', frameon=True, fancybox=True, shadow=True)
        ax.grid(True, alpha=0.3)
        
        # 质量评估放在右上角
        high_quality = len(detailed_results_df[detailed_results_df['r_squared'] > 0.9])
        quality_text = f'R² > 0.9: {high_quality}/{len(detailed_results_df)} ({high_quality/len(detailed_results_df)*100:.1f}%)\n'
        quality_text += f'R² > 0.8: {len(detailed_results_df[detailed_results_df["r_squared"] > 0.8])}/{len(detailed_results_df)}\n'
        quality_text += f'Mean R²: {detailed_results_df["r_squared"].mean():.3f}'
        ax.text(0.98, 0.98, quality_text, transform=ax.transAxes,
                verticalalignment='top', horizontalalignment='right',
                bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.8, pad=0.5))
        
        plt.tight_layout()
        plt.savefig(os.path.join(plots_dir, '3_fit_quality_distribution.png'), 
                   dpi=300, bbox_inches='tight')
        plt.close()
    
    def _create_seasonal_variation(self, detailed_results_df, z0_representative, plots_dir):
        """创建季节变化图"""
        fig, ax = plt.subplots(figsize=(14, 7))
        
        detailed_results_df['timestamp'] = pd.to_datetime(detailed_results_df['timestamp'])
        
        if len(detailed_results_df) > 100:
            monthly_z0 = detailed_results_df.groupby(detailed_results_df['timestamp'].dt.to_period('M'))['z0'].agg(['median', 'mean', 'std', 'count'])
            monthly_z0 = monthly_z0[monthly_z0['count'] >= 5]
            
            if len(monthly_z0) > 1:
                x_data = monthly_z0.index.to_timestamp()
                
                # 绘制数据
                ax.plot(x_data, monthly_z0['median'], 'o-', linewidth=2, markersize=6, 
                       color='blue', label='Monthly Median')
                ax.fill_between(x_data, 
                               monthly_z0['median'] - monthly_z0['std']/np.sqrt(monthly_z0['count']),
                               monthly_z0['median'] + monthly_z0['std']/np.sqrt(monthly_z0['count']),
                               alpha=0.3, color='blue', label='Standard Error')
                
                ax.axhline(z0_representative, color='red', linestyle='--', alpha=0.7, linewidth=2,
                          label=f'Overall Representative z₀: {z0_representative:.4f} m')
                
                ax.set_ylabel('Roughness Length z₀ (m)', fontsize=12)
                ax.set_xlabel('Time', fontsize=12)
                ax.set_title('Seasonal Variation of Surface Roughness Length (Fixed Version)', fontsize=14)
                
                # 图例放在左上角，避免与数据线重叠
                ax.legend(fontsize=11, loc='upper left', frameon=True, fancybox=True, shadow=True)
                ax.grid(True, alpha=0.3)
                ax.tick_params(axis='x', rotation=45)
            else:
                ax.text(0.5, 0.5, 'Insufficient data for time series analysis', 
                       ha='center', va='center', transform=ax.transAxes, fontsize=14)
                ax.set_title('Seasonal Variation (Insufficient Data)')
        else:
            ax.text(0.5, 0.5, 'Insufficient data for time series analysis', 
                   ha='center', va='center', transform=ax.transAxes, fontsize=14)
            ax.set_title('Seasonal Variation (Insufficient Data)')
        
        plt.tight_layout()
        plt.savefig(os.path.join(plots_dir, '4_seasonal_variation.png'), 
                   dpi=300, bbox_inches='tight')
        plt.close()
    
    def _create_friction_velocity_distribution(self, detailed_results_df, plots_dir):
        """创建摩擦速度分布图"""
        fig, ax = plt.subplots(figsize=(12, 7))
        
        u_star_data = detailed_results_df['u_star']
        ax.hist(u_star_data, bins=30, alpha=0.7, color='purple', 
                edgecolor='black', density=True)
        ax.axvline(u_star_data.mean(), color='red', linestyle='--', linewidth=2,
                   label=f'Mean u*: {u_star_data.mean():.3f} m/s')
        ax.axvline(u_star_data.median(), color='orange', linestyle=':', linewidth=2,
                   label=f'Median u*: {u_star_data.median():.3f} m/s')
        
        ax.set_xlabel('Friction Velocity u* (m/s)', fontsize=12)
        ax.set_ylabel('Probability Density', fontsize=12)
        ax.set_title('Distribution of Friction Velocity u* (Fixed Version)', fontsize=14)
        
        # 图例放在右上角
        ax.legend(fontsize=11, loc='upper right', frameon=True, fancybox=True, shadow=True)
        ax.grid(True, alpha=0.3)
        
        # 统计信息放在左上角
        ustar_text = f'Sample Size: {len(u_star_data)}\n'
        ustar_text += f'Mean: {u_star_data.mean():.3f} ± {u_star_data.std():.3f} m/s\n'
        ustar_text += f'Range: [{u_star_data.min():.3f}, {u_star_data.max():.3f}] m/s'
        ax.text(0.02, 0.98, ustar_text, transform=ax.transAxes,
                verticalalignment='top', horizontalalignment='left',
                bbox=dict(boxstyle='round', facecolor='lavender', alpha=0.8, pad=0.5))
        
        plt.tight_layout()
        plt.savefig(os.path.join(plots_dir, '5_friction_velocity_distribution.png'), 
                   dpi=300, bbox_inches='tight')
        plt.close()
    
    def _create_roughness_vs_friction(self, detailed_results_df, z0_representative, plots_dir):
        """创建粗糙度vs摩擦速度关系图"""
        fig, ax = plt.subplots(figsize=(12, 8))
        
        scatter = ax.scatter(detailed_results_df['u_star'], detailed_results_df['z0'], 
                           c=detailed_results_df['r_squared'], cmap='viridis', 
                           alpha=0.6, s=20)
        
        # 颜色条
        cbar = plt.colorbar(scatter, ax=ax)
        cbar.set_label('R² (Goodness of Fit)', fontsize=11)
        
        ax.set_xlabel('Friction Velocity u* (m/s)', fontsize=12)
        ax.set_ylabel('Roughness Length z₀ (m)', fontsize=12)
        ax.set_title('Relationship between Friction Velocity and Roughness Length (Fixed Version)', fontsize=14)
        ax.set_yscale('log')
        ax.grid(True, alpha=0.3)
        
        # 代表性粗糙度线
        ax.axhline(z0_representative, color='red', linestyle='--', linewidth=2,
                   label=f'Representative z₀: {z0_representative:.4f} m')
        
        # 图例放在左下角，避免与散点重叠
        ax.legend(fontsize=11, loc='lower left', frameon=True, fancybox=True, shadow=True)
        
        plt.tight_layout()
        plt.savefig(os.path.join(plots_dir, '6_roughness_vs_friction.png'), 
                   dpi=300, bbox_inches='tight')
        plt.close()

def main():
    """主执行函数"""
    # 设置路径
    data_path = "/Users/xiaxin/work/WindForecast_Project/01_Data/processed/matched_data/changma_matched.csv"
    stability_results_path = "/Users/xiaxin/work/WindForecast_Project/03_Results/stability_analysis/changma_stability_results.csv"
    output_dir = "/Users/xiaxin/work/WindForecast_Project/03_Results/fixed_neutral_roughness_analysis"
    
    # 初始化修复版计算器
    calculator = FixedNeutralRoughnessCalculator()
    
    try:
        print("开始修复版粗糙度计算...")
        
        # 1. 加载和筛选数据
        neutral_data = calculator.load_and_filter_data(data_path, stability_results_path)
        if neutral_data is None or len(neutral_data) == 0:
            print("无法获取有效数据")
            return
        
        # 2. 计算代表性粗糙度
        result = calculator.calculate_representative_roughness(neutral_data)
        if result[0] is None:
            print("粗糙度计算失败")
            return
            
        z0_representative, statistics, detailed_results_df = result
        
        # 3. 季节分析
        seasonal_stats = calculator.seasonal_analysis(detailed_results_df)
        
        # 4. 生成报告
        final_z0 = calculator.generate_report(z0_representative, statistics, seasonal_stats, output_dir)
        
        # 5. 创建完整的6副图表
        calculator.create_all_plots(detailed_results_df, z0_representative, output_dir)
        
        # 6. 保存详细结果
        if len(detailed_results_df) > 0:
            detailed_results_df.to_csv(
                os.path.join(output_dir, "changma_fixed_roughness_detailed.csv"), 
                index=False
            )
        
        print(f"\n=== 修复版分析完成 ===")
        print(f"昌马站代表性粗糙度: {final_z0:.6f} m")
        print(f"地表类型: {calculator.classify_surface_type(final_z0)}")
        print(f"结果保存在: {output_dir}")
        
        # 验证计算示例
        print(f"\n=== 计算验证示例 ===")
        if len(detailed_results_df) > 0:
            sample = detailed_results_df.iloc[0]
            print(f"样本u*: {sample['u_star']:.4f} m/s")
            print(f"样本z₀: {sample['z0']:.6f} m")
            print(f"观测风速: {sample['wind_profile']}")
            
            # 重新计算理论风速
            heights = np.array(sample['height_profile'])
            u_theory = (sample['u_star'] / 0.4) * np.log(heights / sample['z0'])
            print(f"理论风速: {u_theory.tolist()}")
            print(f"差异: {np.array(sample['wind_profile']) - u_theory}")
        
    except Exception as e:
        print(f"分析过程中出现错误: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()