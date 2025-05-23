#!/usr/bin/env python3
"""
诊断版粗糙度计算器
用于找出原版和修复版的问题所在，获得正确的0.0008m同时保证风速剖面匹配

目标：
1. 验证不同计算方法
2. 对比原版、修复版和理论正确版
3. 确保物理合理性和数学正确性
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

class DiagnosticRoughnessCalculator:
    """
    诊断版粗糙度计算器
    测试多种计算方法，找出正确的实现
    """
    
    def __init__(self):
        self.kappa = 0.4  # von Karman常数
        self.heights = np.array([10, 30, 50, 70])  # 测量高度
    
    def method_original(self, wind_speeds, heights):
        """
        原版方法：对ln(U)和ln(z)进行回归
        """
        valid_mask = (wind_speeds > 1.0) & ~np.isnan(wind_speeds)
        if np.sum(valid_mask) < 3:
            return np.nan, np.nan, np.nan, {}
        
        valid_winds = wind_speeds[valid_mask]
        valid_heights = heights[valid_mask]
        
        try:
            # 对数变换
            ln_z = np.log(valid_heights)
            ln_u = np.log(valid_winds)
            
            # 线性回归: ln(U) = a*ln(z) + b
            slope, intercept, r_value, p_value, std_err = stats.linregress(ln_z, ln_u)
            
            # 参数提取（原版方法）
            u_star = slope * self.kappa
            z0 = np.exp(-intercept / slope)
            
            return z0, u_star, r_value**2, {
                'method': 'original',
                'slope': slope,
                'intercept': intercept,
                'formula': 'ln(U) = slope*ln(z) + intercept'
            }
        except:
            return np.nan, np.nan, np.nan, {}
    
    def method_fixed(self, wind_speeds, heights):
        """
        修复版方法：对U和ln(z)进行回归
        """
        valid_mask = (wind_speeds > 1.0) & ~np.isnan(wind_speeds)
        if np.sum(valid_mask) < 3:
            return np.nan, np.nan, np.nan, {}
        
        valid_winds = wind_speeds[valid_mask]
        valid_heights = heights[valid_mask]
        
        try:
            # 对数变换
            ln_z = np.log(valid_heights)
            
            # 线性回归: U = A*ln(z) + B
            slope, intercept, r_value, p_value, std_err = stats.linregress(ln_z, valid_winds)
            
            # 参数提取（修复版方法）
            A = slope          # A = u*/κ
            B = intercept      # B = -(u*/κ) * ln(z₀)
            
            u_star = A * self.kappa
            z0 = np.exp(-B / A) if A != 0 else np.nan
            
            return z0, u_star, r_value**2, {
                'method': 'fixed',
                'slope': slope,
                'intercept': intercept,
                'formula': 'U = slope*ln(z) + intercept'
            }
        except:
            return np.nan, np.nan, np.nan, {}
    
    def method_theoretical(self, wind_speeds, heights):
        """
        理论正确方法：基于对数风速剖面的严格推导
        
        理论：U(z) = (u*/κ) × ln(z/z₀)
        重写：U(z) = (u*/κ) × ln(z) - (u*/κ) × ln(z₀)
        即：U(z) = A × ln(z) + B
        其中：A = u*/κ, B = -(u*/κ) × ln(z₀)
        
        因此：u* = A × κ, z₀ = exp(-B/A)
        """
        valid_mask = (wind_speeds > 1.0) & ~np.isnan(wind_speeds)
        if np.sum(valid_mask) < 3:
            return np.nan, np.nan, np.nan, {}
        
        valid_winds = wind_speeds[valid_mask]
        valid_heights = heights[valid_mask]
        
        try:
            # 线性回归: U = A*ln(z) + B
            ln_z = np.log(valid_heights)
            slope, intercept, r_value, p_value, std_err = stats.linregress(ln_z, valid_winds)
            
            # 理论正确的参数提取
            A = slope          # A = u*/κ
            B = intercept      # B = -(u*/κ) × ln(z₀)
            
            u_star = A * self.kappa
            z0 = np.exp(-B / A) if A != 0 else np.nan
            
            return z0, u_star, r_value**2, {
                'method': 'theoretical',
                'slope': slope,
                'intercept': intercept,
                'A': A,
                'B': B,
                'formula': 'U = (u*/κ)*ln(z) - (u*/κ)*ln(z₀)'
            }
        except:
            return np.nan, np.nan, np.nan, {}
    
    def method_alternative(self, wind_speeds, heights):
        """
        替代方法：使用最小二乘法直接拟合对数剖面
        """
        valid_mask = (wind_speeds > 1.0) & ~np.isnan(wind_speeds)
        if np.sum(valid_mask) < 3:
            return np.nan, np.nan, np.nan, {}
        
        valid_winds = wind_speeds[valid_mask]
        valid_heights = heights[valid_mask]
        
        try:
            # 使用非线性优化拟合 U = (u*/κ) × ln(z/z₀)
            from scipy.optimize import curve_fit
            
            def log_profile(z, u_star, z0):
                return (u_star / self.kappa) * np.log(z / z0)
            
            # 初始猜测
            popt, pcov = curve_fit(log_profile, valid_heights, valid_winds, 
                                 p0=[0.1, 0.001], bounds=([0.01, 1e-6], [2.0, 1.0]))
            
            u_star_fit, z0_fit = popt
            
            # 计算R²
            u_pred = log_profile(valid_heights, u_star_fit, z0_fit)
            ss_res = np.sum((valid_winds - u_pred) ** 2)
            ss_tot = np.sum((valid_winds - np.mean(valid_winds)) ** 2)
            r_squared = 1 - (ss_res / ss_tot)
            
            return z0_fit, u_star_fit, r_squared, {
                'method': 'nonlinear_fit',
                'formula': 'U = (u*/κ) × ln(z/z₀) - direct fit'
            }
        except:
            return np.nan, np.nan, np.nan, {}
    
    def test_single_profile(self, wind_speeds, heights):
        """
        测试单个风速剖面的不同计算方法
        """
        print("=== 单个风速剖面测试 ===")
        print(f"观测数据:")
        print(f"  高度: {heights}")
        print(f"  风速: {wind_speeds}")
        
        methods = [
            ('原版方法', self.method_original),
            ('修复版方法', self.method_fixed),
            ('理论正确方法', self.method_theoretical),
            ('非线性拟合方法', self.method_alternative)
        ]
        
        results = {}
        
        for name, method in methods:
            z0, u_star, r2, details = method(wind_speeds, heights)
            results[name] = {
                'z0': z0,
                'u_star': u_star,
                'r_squared': r2,
                'details': details
            }
            
            print(f"\n{name}:")
            print(f"  z₀: {z0:.6f} m")
            print(f"  u*: {u_star:.4f} m/s")
            print(f"  R²: {r2:.4f}")
            
            # 验证：重新计算理论风速
            if not np.isnan(z0) and not np.isnan(u_star):
                u_theory = (u_star / self.kappa) * np.log(heights / z0)
                error = np.mean(np.abs(wind_speeds - u_theory))
                print(f"  理论风速: {u_theory}")
                print(f"  匹配误差: {error:.4f} m/s")
        
        return results
    
    def create_comparison_plot(self, wind_speeds, heights, results):
        """
        创建对比图
        """
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        axes = axes.flatten()
        
        colors = ['red', 'blue', 'green', 'orange']
        methods = ['原版方法', '修复版方法', '理论正确方法', '非线性拟合方法']
        
        for i, (method_name, color) in enumerate(zip(methods, colors)):
            ax = axes[i]
            
            # 观测点
            ax.plot(wind_speeds, heights, 'ko', markersize=8, label='观测数据')
            
            # 理论线
            result = results[method_name]
            if not np.isnan(result['z0']) and not np.isnan(result['u_star']):
                z_theory = np.linspace(heights.min(), heights.max(), 100)
                u_theory = (result['u_star'] / self.kappa) * np.log(z_theory / result['z0'])
                ax.plot(u_theory, z_theory, color=color, linewidth=2, label='理论线')
                
                # 计算匹配误差
                u_obs_theory = (result['u_star'] / self.kappa) * np.log(heights / result['z0'])
                error = np.mean(np.abs(wind_speeds - u_obs_theory))
                
                ax.set_title(f'{method_name}\nz₀={result["z0"]:.6f}m, u*={result["u_star"]:.4f}m/s\n'
                           f'R²={result["r_squared"]:.4f}, 误差={error:.4f}m/s')
            else:
                ax.set_title(f'{method_name}\n计算失败')
            
            ax.set_xlabel('风速 (m/s)')
            ax.set_ylabel('高度 (m)')
            ax.legend()
            ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()
        
        return fig

def main():
    """
    主函数：测试不同方法
    """
    print("=== 诊断版粗糙度计算器 ===")
    
    # 使用您数据中的一个典型样本
    wind_speeds = np.array([4.31, 5.2, 5.3, 5.45])  # 来自您的Excel数据
    heights = np.array([10, 30, 50, 70])
    
    calculator = DiagnosticRoughnessCalculator()
    
    # 测试所有方法
    results = calculator.test_single_profile(wind_speeds, heights)
    
    # 创建对比图
    # fig = calculator.create_comparison_plot(wind_speeds, heights, results)
    
    print("\n=== 结论 ===")
    print("哪种方法能同时满足:")
    print("1. 合理的粗糙度值 (0.0008m)")
    print("2. 正确的风速剖面匹配")
    
    # 分析哪种方法最合理
    for method_name, result in results.items():
        z0 = result['z0']
        if not np.isnan(z0):
            if 0.0001 <= z0 <= 0.01:  # 合理范围
                print(f"\n{method_name} 在合理范围内: z₀ = {z0:.6f} m")

if __name__ == "__main__":
    main()