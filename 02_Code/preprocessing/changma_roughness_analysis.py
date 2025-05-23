#!/usr/bin/env python3
"""
昌马站粗糙度长度迭代计算
基于Monin-Obukhov相似理论和风廓线迭代方法

参考文献:
- Monin, A.S. & Obukhov, A.M. (1954). Basic laws of turbulent mixing
- Businger, J.A. et al. (1971). Flux-profile relationships in the atmospheric surface layer
- Dyer, A.J. (1974). A review of flux-profile relationships
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import warnings
from scipy.optimize import minimize_scalar, minimize
from scipy import stats
import os
warnings.filterwarnings('ignore')

class RoughnessCalculator:
    """
    基于风廓线迭代计算粗糙度长度
    """
    
    def __init__(self):
        self.kappa = 0.4  # von Karman常数
        self.heights = np.array([10, 30, 50, 70])  # 昌马站测量高度(m)
        
        # 迭代参数
        self.max_iterations = 100
        self.tolerance = 1e-6
        self.min_wind_speed = 1.0  # 最小风速阈值(m/s)
        
        # 粗糙度范围限制
        self.z0_min = 1e-4  # 最小粗糙度(m)
        self.z0_max = 2.0   # 最大粗糙度(m)
        
    def stability_correction_unstable(self, zeta):
        """
        不稳定条件下的稳定度修正函数 (Businger-Dyer关系)
        zeta = z/L < 0 (不稳定)
        """
        x = (1 - 16 * zeta) ** 0.25
        psi_m = (2 * np.log((1 + x) / 2) + 
                 np.log((1 + x**2) / 2) - 
                 2 * np.arctan(x) + 
                 np.pi / 2)
        return psi_m
    
    def stability_correction_stable(self, zeta):
        """
        稳定条件下的稳定度修正函数
        zeta = z/L > 0 (稳定)
        """
        return -5 * zeta
    
    def wind_profile_theoretical(self, z, u_star, z0, L=None):
        """
        计算理论风速剖面
        U(z) = (u*/κ) × [ln(z/z0) + ψm(z/L)]
        """
        if z <= z0:
            return 0.0
            
        # 基础对数项
        ln_term = np.log(z / z0)
        
        # 稳定度修正
        if L is None or abs(L) > 1e6:  # 中性条件
            psi_m = 0
        else:
            zeta = z / L
            if zeta < 0:  # 不稳定
                psi_m = self.stability_correction_unstable(zeta)
            elif zeta > 0:  # 稳定
                psi_m = self.stability_correction_stable(zeta)
            else:
                psi_m = 0
        
        # 理论风速
        u_theoretical = (u_star / self.kappa) * (ln_term + psi_m)
        return max(u_theoretical, 0.1)  # 避免负风速
    
    def calculate_roughness_neutral(self, wind_speeds, heights):
        """
        中性条件下的粗糙度计算（简化方法）
        """
        # 过滤有效数据
        valid_mask = (wind_speeds > self.min_wind_speed) & ~np.isnan(wind_speeds)
        if np.sum(valid_mask) < 2:
            return np.nan, np.nan, 0
        
        valid_winds = wind_speeds[valid_mask]
        valid_heights = heights[valid_mask]
        
        # 对数变换进行线性回归
        ln_z = np.log(valid_heights)
        ln_u = np.log(valid_winds)
        
        try:
            # 线性回归: ln(U) = a*ln(z) + b
            # 其中 a = 1/κ * u*, b = -a*ln(z0)
            slope, intercept, r_value, p_value, std_err = stats.linregress(ln_z, ln_u)
            
            # 计算参数
            u_star = slope * self.kappa
            z0 = np.exp(-intercept / slope)
            
            # 检查合理性
            if (self.z0_min <= z0 <= self.z0_max and 
                0.1 <= u_star <= 2.0 and 
                r_value**2 > 0.7):
                return z0, u_star, r_value**2
            else:
                return np.nan, np.nan, 0
                
        except:
            return np.nan, np.nan, 0
    
    def calculate_roughness_iterative(self, wind_speeds, heights, stability_class='neutral', L=None):
        """
        迭代方法计算粗糙度
        """
        # 数据质量检查
        valid_mask = (wind_speeds > self.min_wind_speed) & ~np.isnan(wind_speeds)
        if np.sum(valid_mask) < 3:  # 至少需要3个有效高度
            return {
                'z0': np.nan,
                'u_star': np.nan,
                'L': np.nan,
                'rmse': np.nan,
                'r_squared': np.nan,
                'iterations': 0,
                'converged': False,
                'quality': 'insufficient_data'
            }
        
        valid_winds = wind_speeds[valid_mask]
        valid_heights = heights[valid_mask]
        
        # 初始估计（基于中性条件）
        z0_init, u_star_init, r2_init = self.calculate_roughness_neutral(valid_winds, valid_heights)
        
        if np.isnan(z0_init):
            z0_init = 0.05  # 默认初值
            u_star_init = 0.3
        
        # 定义目标函数
        def objective_function(params):
            z0, u_star = params
            
            # 参数范围检查
            if z0 < self.z0_min or z0 > self.z0_max or u_star < 0.1 or u_star > 2.0:
                return 1e6
            
            # 计算理论风速
            u_theoretical = np.array([
                self.wind_profile_theoretical(z, u_star, z0, L) 
                for z in valid_heights
            ])
            
            # 计算RMSE
            rmse = np.sqrt(np.mean((valid_winds - u_theoretical)**2))
            return rmse
        
        # 优化求解
        try:
            from scipy.optimize import minimize
            
            result = minimize(
                objective_function,
                x0=[z0_init, u_star_init],
                method='Nelder-Mead',
                options={
                    'maxiter': self.max_iterations,
                    'xatol': self.tolerance,
                    'fatol': self.tolerance
                }
            )
            
            z0_opt, u_star_opt = result.x
            converged = result.success
            iterations = result.nit
            
        except:
            z0_opt, u_star_opt = z0_init, u_star_init
            converged = False
            iterations = 0
        
        # 计算最终统计量
        u_theoretical = np.array([
            self.wind_profile_theoretical(z, u_star_opt, z0_opt, L) 
            for z in valid_heights
        ])
        
        rmse = np.sqrt(np.mean((valid_winds - u_theoretical)**2))
        
        # 计算R²
        ss_res = np.sum((valid_winds - u_theoretical)**2)
        ss_tot = np.sum((valid_winds - np.mean(valid_winds))**2)
        r_squared = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0
        
        # 质量评估
        if r_squared > 0.8 and rmse < 1.0 and converged:
            quality = 'high'
        elif r_squared > 0.6 and rmse < 2.0:
            quality = 'medium'
        else:
            quality = 'low'
        
        return {
            'z0': z0_opt,
            'u_star': u_star_opt,
            'L': L,
            'rmse': rmse,
            'r_squared': r_squared,
            'iterations': iterations,
            'converged': converged,
            'quality': quality,
            'n_points': len(valid_winds)
        }
    
    def load_and_prepare_data(self, data_path, stability_results_path):
        """
        加载并准备数据，智能识别时间列
        """
        print("加载原始数据...")
        data = pd.read_csv(data_path)
        print(f"原始数据列名: {list(data.columns)}")
        
        print("加载稳定度分析结果...")
        stability_results = pd.read_csv(stability_results_path)
        print(f"稳定度结果列名: {list(stability_results.columns)}")
        
        # 智能识别时间列
        def find_time_column(df, data_name):
            time_candidates = [col for col in df.columns if 
                             any(time_word in col.lower() for time_word in ['time', 'date', 'timestamp'])]
            
            print(f"{data_name}中发现的时间相关列: {time_candidates}")
            
            if time_candidates:
                time_col = time_candidates[0]
                print(f"使用时间列: {time_col}")
                return time_col
            else:
                # 检查第一列是否可能是时间
                first_col = df.columns[0]
                print(f"尝试将第一列作为时间: {first_col}")
                try:
                    pd.to_datetime(df[first_col].iloc[:5])  # 测试前5行
                    print(f"第一列确认为时间列: {first_col}")
                    return first_col
                except:
                    print(f"第一列不是时间格式")
                    return None
        
        # 为原始数据找时间列
        data_time_col = find_time_column(data, "原始数据")
        if data_time_col:
            data['timestamp'] = pd.to_datetime(data[data_time_col])
        else:
            raise ValueError("无法在原始数据中找到时间列！")
        
        # 为稳定度结果找时间列
        stability_time_col = find_time_column(stability_results, "稳定度结果")
        if stability_time_col and stability_time_col != 'timestamp':
            stability_results['timestamp'] = pd.to_datetime(stability_results[stability_time_col])
        elif 'timestamp' not in stability_results.columns:
            raise ValueError("无法在稳定度结果中找到时间列！")
        else:
            stability_results['timestamp'] = pd.to_datetime(stability_results['timestamp'])
        
        return data, stability_results
    
    def analyze_changma_roughness(self, data_path, stability_results_path, output_dir):
        """
        分析昌马站的粗糙度特征
        """
        print("开始昌马站粗糙度分析...")
        
        # 创建输出目录
        os.makedirs(output_dir, exist_ok=True)
        
        # 加载和准备数据
        try:
            data, stability_results = self.load_and_prepare_data(data_path, stability_results_path)
        except Exception as e:
            print(f"数据加载失败: {e}")
            return None
        
        # 合并数据
        print("合并数据...")
        
        # 选择需要的稳定度列
        stability_cols = ['timestamp', 'stability_final', 'confidence_final']
        if 'alpha_main' in stability_results.columns:
            stability_cols.append('alpha_main')
        
        combined_data = pd.merge(
            data, stability_results[stability_cols],
            on='timestamp', how='inner'
        )
        
        print(f"合并后数据点数: {len(combined_data)}")
        
        # 检查风速列是否存在
        wind_columns = ['obs_wind_speed_10m', 'obs_wind_speed_30m', 
                       'obs_wind_speed_50m', 'obs_wind_speed_70m']
        
        missing_cols = [col for col in wind_columns if col not in combined_data.columns]
        if missing_cols:
            print(f"警告: 缺少风速列 {missing_cols}")
            print(f"可用的列: {list(combined_data.columns)}")
            
            # 尝试找到对应的列
            available_wind_cols = [col for col in combined_data.columns if 'wind' in col.lower()]
            print(f"发现的风速相关列: {available_wind_cols}")
            
            if len(available_wind_cols) < 4:
                raise ValueError("数据中风速列不足，无法进行粗糙度计算！")
        
        results = []
        
        # 逐点计算粗糙度
        print("开始逐点计算粗糙度...")
        
        for idx, row in combined_data.iterrows():
            if idx % 5000 == 0:
                print(f"进度: {idx}/{len(combined_data)} ({idx/len(combined_data)*100:.1f}%)")
            
            # 提取风速数据
            try:
                wind_speeds = np.array([row[col] for col in wind_columns])
            except KeyError as e:
                print(f"第{idx}行数据提取失败: {e}")
                continue
            
            # 基于稳定度选择计算方法
            stability = row['stability_final']
            confidence = row['confidence_final']
            
            # 高置信度数据才进行复杂计算
            if confidence > 0.6 and stability in ['stable', 'neutral', 'unstable']:
                if stability == 'neutral':
                    # 中性条件：简化计算
                    result = self.calculate_roughness_iterative(
                        wind_speeds, self.heights, 'neutral', L=None
                    )
                else:
                    # 稳定/不稳定条件：使用alpha_main估算L
                    alpha = row.get('alpha_main', np.nan)
                    if not np.isnan(alpha):
                        # 粗略估算Obukhov长度
                        if stability == 'stable':
                            L = 50.0  # 正值，稳定
                        else:  # unstable
                            L = -50.0  # 负值，不稳定
                    else:
                        L = None
                    
                    result = self.calculate_roughness_iterative(
                        wind_speeds, self.heights, stability, L
                    )
            else:
                # 低置信度数据：仅使用简化方法
                z0, u_star, r2 = self.calculate_roughness_neutral(wind_speeds, self.heights)
                result = {
                    'z0': z0,
                    'u_star': u_star,
                    'L': np.nan,
                    'rmse': np.nan,
                    'r_squared': r2,
                    'iterations': 0,
                    'converged': False,
                    'quality': 'low_confidence',
                    'n_points': 4
                }
            
            # 添加元数据
            result.update({
                'timestamp': row['timestamp'],
                'stability': stability,
                'confidence': confidence,
                'alpha_main': row.get('alpha_main', np.nan),
                'wind_10m': wind_speeds[0],
                'wind_30m': wind_speeds[1],
                'wind_50m': wind_speeds[2],
                'wind_70m': wind_speeds[3]
            })
            
            results.append(result)
        
        # 转换为DataFrame
        results_df = pd.DataFrame(results)
        
        # 保存详细结果
        results_output_path = os.path.join(output_dir, "changma_roughness_results.csv")
        results_df.to_csv(results_output_path, index=False)
        print(f"详细结果已保存到: {results_output_path}")
        
        # 统计分析
        self.generate_roughness_statistics(results_df, output_dir)
        
        # 生成可视化
        self.create_roughness_plots(results_df, output_dir)
        
        return results_df
    
    def generate_roughness_statistics(self, results_df, output_dir):
        """
        生成粗糙度统计分析
        """
        print("\n=== 昌马站粗糙度分析统计摘要 ===")
        
        # 过滤有效结果
        valid_results = results_df[~pd.isna(results_df['z0']) & (results_df['z0'] > 0)]
        print(f"有效计算点数: {len(valid_results)}/{len(results_df)} ({len(valid_results)/len(results_df)*100:.1f}%)")
        
        if len(valid_results) == 0:
            print("警告: 没有有效的粗糙度计算结果！")
            return
        
        # 整体统计
        print(f"\n整体粗糙度统计:")
        print(f"  平均值: {valid_results['z0'].mean():.4f} m")
        print(f"  中位数: {valid_results['z0'].median():.4f} m")
        print(f"  标准差: {valid_results['z0'].std():.4f} m")
        print(f"  范围: [{valid_results['z0'].min():.4f}, {valid_results['z0'].max():.4f}] m")
        
        # 按稳定度分类统计
        print(f"\n按稳定度分类:")
        for stability in ['unstable', 'neutral', 'stable']:
            subset = valid_results[valid_results['stability'] == stability]
            if len(subset) > 0:
                print(f"  {stability}:")
                print(f"    样本数: {len(subset)}")
                print(f"    平均z0: {subset['z0'].mean():.4f} ± {subset['z0'].std():.4f} m")
                print(f"    平均u*: {subset['u_star'].mean():.3f} ± {subset['u_star'].std():.3f} m/s")
        
        # 按数据质量分类
        print(f"\n按数据质量分类:")
        for quality in ['high', 'medium', 'low']:
            subset = valid_results[valid_results['quality'] == quality]
            if len(subset) > 0:
                print(f"  {quality}: {len(subset)} ({len(subset)/len(valid_results)*100:.1f}%)")
                print(f"    平均z0: {subset['z0'].mean():.4f} m")
        
        # 地表类型推断
        z0_mean = valid_results['z0'].mean()
        print(f"\n地表类型推断 (基于平均z0 = {z0_mean:.4f} m):")
        if z0_mean < 0.001:
            surface_type = "非常光滑表面 (雪地/冰面)"
        elif z0_mean < 0.01:
            surface_type = "光滑表面 (短草地)"
        elif z0_mean < 0.05:
            surface_type = "农田/牧场"
        elif z0_mean < 0.1:
            surface_type = "农田/低矮植被"
        elif z0_mean < 0.5:
            surface_type = "灌木地/粗糙农田"
        else:
            surface_type = "森林/建筑区"
        
        print(f"  推断地表类型: {surface_type}")
        
        # 保存统计摘要
        summary_path = os.path.join(output_dir, "changma_roughness_summary.txt")
        with open(summary_path, 'w', encoding='utf-8') as f:
            f.write("昌马站粗糙度分析统计摘要\n")
            f.write("="*50 + "\n\n")
            f.write(f"分析时间: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"有效计算点数: {len(valid_results)}/{len(results_df)}\n\n")
            
            f.write("整体粗糙度统计:\n")
            f.write(f"  平均值: {valid_results['z0'].mean():.4f} m\n")
            f.write(f"  中位数: {valid_results['z0'].median():.4f} m\n")
            f.write(f"  标准差: {valid_results['z0'].std():.4f} m\n")
            f.write(f"  范围: [{valid_results['z0'].min():.4f}, {valid_results['z0'].max():.4f}] m\n\n")
            
            f.write(f"推断地表类型: {surface_type}\n")
        
        print(f"统计摘要已保存到: {summary_path}")
    
    def create_roughness_plots(self, results_df, output_dir):
        """
        创建粗糙度分析图表
        """
        plots_dir = os.path.join(output_dir, "plots")
        os.makedirs(plots_dir, exist_ok=True)
        
        # 过滤有效数据
        valid_results = results_df[~pd.isna(results_df['z0']) & (results_df['z0'] > 0)]
        
        if len(valid_results) == 0:
            print("警告: 没有有效数据用于绘图")
            return
        
        # 设置绘图样式
        plt.style.use('default')
        fig_size = (12, 8)
        
        # 1. 粗糙度分布直方图
        plt.figure(figsize=fig_size)
        plt.hist(valid_results['z0'], bins=50, alpha=0.7, color='steelblue', edgecolor='black')
        plt.axvline(valid_results['z0'].mean(), color='red', linestyle='--', 
                   label=f'Mean: {valid_results["z0"].mean():.4f} m')
        plt.axvline(valid_results['z0'].median(), color='orange', linestyle='--',
                   label=f'Median: {valid_results["z0"].median():.4f} m')
        plt.xlabel('Roughness Length z0 (m)', fontsize=12, fontweight='bold')
        plt.ylabel('Frequency', fontsize=12, fontweight='bold')
        plt.title('Changma Station Roughness Length Distribution', fontsize=14, fontweight='bold')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.yscale('log')
        plt.xscale('log')
        plt.tight_layout()
        plt.savefig(os.path.join(plots_dir, 'roughness_distribution.png'), dpi=300, bbox_inches='tight')
        plt.close()
        
        # 2. 按稳定度分类的粗糙度箱线图
        plt.figure(figsize=fig_size)
        stability_data = []
        stability_labels = []
        
        for stability in ['unstable', 'neutral', 'stable']:
            subset = valid_results[valid_results['stability'] == stability]
            if len(subset) > 10:  # 至少10个样本才绘制
                stability_data.append(subset['z0'])
                stability_labels.append(f'{stability.capitalize()}\n(n={len(subset)})')
        
        if stability_data:
            plt.boxplot(stability_data, labels=stability_labels)
            plt.ylabel('Roughness Length z0 (m)', fontsize=12, fontweight='bold')
            plt.xlabel('Stability Class', fontsize=12, fontweight='bold')
            plt.title('Changma Station Roughness Length by Stability Class', fontsize=14, fontweight='bold')
            plt.yscale('log')
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            plt.savefig(os.path.join(plots_dir, 'roughness_by_stability.png'), dpi=300, bbox_inches='tight')
        plt.close()
        
        # 3. 粗糙度时间序列
        plt.figure(figsize=(15, 6))
        valid_results['timestamp'] = pd.to_datetime(valid_results['timestamp'])
        
        # 按月平均
        monthly_z0 = valid_results.groupby(valid_results['timestamp'].dt.to_period('M'))['z0'].mean()
        
        plt.plot(monthly_z0.index.to_timestamp(), monthly_z0.values, 'o-', linewidth=2, markersize=6)
        plt.ylabel('Monthly Mean Roughness Length z0 (m)', fontsize=12, fontweight='bold')
        plt.xlabel('Time', fontsize=12, fontweight='bold')
        plt.title('Changma Station Roughness Length Time Series', fontsize=14, fontweight='bold')
        plt.yscale('log')
        plt.grid(True, alpha=0.3)
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig(os.path.join(plots_dir, 'roughness_timeseries.png'), dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"已生成3个粗糙度分析图表，保存在: {plots_dir}")

def main():
    """
    主执行函数
    """
    # 设置路径
    data_path = "/Users/xiaxin/work/WindForecast_Project/01_Data/processed/matched_data/changma_matched.csv"
    stability_results_path = "/Users/xiaxin/work/WindForecast_Project/03_Results/stability_analysis/changma_stability_results.csv"
    output_dir = "/Users/xiaxin/work/WindForecast_Project/03_Results/roughness_analysis"
    
    # 初始化计算器
    calculator = RoughnessCalculator()
    
    try:
        # 执行粗糙度分析
        results_df = calculator.analyze_changma_roughness(
            data_path, stability_results_path, output_dir
        )
        
        print(f"\n=== 分析完成 ===")
        print(f"结果保存在: {output_dir}")
        print(f"下一步可以:")
        print(f"1. 分析粗糙度与风速预报误差的关系")
        print(f"2. 比较不同稳定度下的粗糙度特征")
        print(f"3. 研究粗糙度对EC vs GFS预报的影响")
        
    except Exception as e:
        print(f"分析过程中出现错误: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()