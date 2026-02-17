#!/usr/bin/env python3
"""
增强版按风向分类的SHAP分析 - 包含风切变指数计算
通过4层梯度(10m, 30m, 50m, 70m)用最小二乘法拟合每个时刻的风切变指数
将风切变指数作为新特征输入模型,分析其对功率预测的贡献

Wind Shear Exponent (α) 从幂律公式推导: v(z) = v_ref * (z/z_ref)^α
取对数: ln(v) = ln(v_ref) + α * ln(z/z_ref)
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from scipy.optimize import least_squares
from scipy.stats import linregress
import lightgbm as lgb
import shap
import warnings
import os
warnings.filterwarnings('ignore')

# 设置绘图样式
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

# 设置中文字体支持
plt.rcParams['font.sans-serif'] = ['Arial Unicode MS', 'SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False


class WindShearCalculator:
    """风切变指数计算器"""
    
    def __init__(self):
        # 定义测风高度 (米)
        self.heights = np.array([10, 30, 50, 70])
        self.height_names = ['10m', '30m', '50m', '70m']
        
    def calculate_wind_shear_least_squares(self, wind_speeds):
        """
        使用最小二乘法计算风切变指数
        
        幂律公式: v(z) = v_ref * (z/z_ref)^α
        取对数变换为线性: ln(v) = ln(v_ref) + α * ln(z/z_ref)
        
        参数:
            wind_speeds: array-like, shape (4,) 
                        包含10m, 30m, 50m, 70m的风速
        
        返回:
            alpha: float, 风切变指数
            r_squared: float, 拟合优度
            wind_shear_quality: str, 拟合质量评级
        """
        # 检查输入有效性
        if len(wind_speeds) != 4:
            return np.nan, np.nan, 'invalid'
        
        # 过滤无效数据
        valid_mask = (wind_speeds > 0) & (~np.isnan(wind_speeds))
        if valid_mask.sum() < 3:  # 至少需要3个有效点
            return np.nan, np.nan, 'insufficient_data'
        
        valid_heights = self.heights[valid_mask]
        valid_speeds = wind_speeds[valid_mask]
        
        # 使用10m作为参考高度
        z_ref = 10.0
        
        # 对数变换
        ln_z_ratio = np.log(valid_heights / z_ref)
        ln_v = np.log(valid_speeds)
        
        # 最小二乘线性回归
        # ln(v) = intercept + slope * ln(z/z_ref)
        # 其中 slope = α (风切变指数)
        try:
            slope, intercept, r_value, p_value, std_err = linregress(ln_z_ratio, ln_v)
            
            alpha = slope
            r_squared = r_value ** 2
            
            # 评估拟合质量
            if r_squared > 0.95:
                quality = 'excellent'
            elif r_squared > 0.85:
                quality = 'good'
            elif r_squared > 0.70:
                quality = 'fair'
            else:
                quality = 'poor'
            
            # 物理合理性检查 (风切变指数通常在0.05到0.5之间)
            if alpha < 0 or alpha > 1.0:
                quality = 'unrealistic'
            
            return alpha, r_squared, quality
            
        except Exception as e:
            return np.nan, np.nan, 'calculation_error'
    
    def calculate_for_dataframe(self, df):
        """
        为整个数据框计算风切变指数
        
        参数:
            df: DataFrame, 必须包含obs_wind_speed_10m, 30m, 50m, 70m列
        
        返回:
            df: DataFrame, 添加了wind_shear_alpha, wind_shear_r2, wind_shear_quality列
        """
        print("="*60)
        print("开始计算风切变指数 (Wind Shear Exponent)")
        print("="*60)
        print("方法: 最小二乘法拟合幂律风廓线")
        print(f"使用高度: {self.heights} m")
        print(f"幂律公式: v(z) = v_ref * (z/z_ref)^α")
        print(f"参考高度: z_ref = 10 m")
        print()
        
        # 检查必需的列
        required_cols = [f'obs_wind_speed_{h}' for h in self.height_names]
        missing_cols = [col for col in required_cols if col not in df.columns]
        if missing_cols:
            raise ValueError(f"缺少必需的列: {missing_cols}")
        
        # 准备风速数组
        wind_speed_matrix = df[required_cols].values
        
        # 计算每一行的风切变指数
        results = []
        for i, wind_speeds in enumerate(wind_speed_matrix):
            if i % 10000 == 0 and i > 0:
                print(f"  已处理 {i}/{len(wind_speed_matrix)} 条记录...")
            
            alpha, r2, quality = self.calculate_wind_shear_least_squares(wind_speeds)
            results.append({
                'wind_shear_alpha': alpha,
                'wind_shear_r2': r2,
                'wind_shear_quality': quality
            })
        
        # 转换为DataFrame并合并
        results_df = pd.DataFrame(results)
        df = pd.concat([df, results_df], axis=1)
        
        # 统计信息
        print(f"\n计算完成! 总共处理 {len(df)} 条记录")
        print("\n风切变指数统计:")
        print(f"  平均值: {df['wind_shear_alpha'].mean():.4f}")
        print(f"  中位数: {df['wind_shear_alpha'].median():.4f}")
        print(f"  标准差: {df['wind_shear_alpha'].std():.4f}")
        print(f"  最小值: {df['wind_shear_alpha'].min():.4f}")
        print(f"  最大值: {df['wind_shear_alpha'].max():.4f}")
        
        print("\n拟合质量分布:")
        quality_counts = df['wind_shear_quality'].value_counts()
        for quality, count in quality_counts.items():
            percentage = count / len(df) * 100
            print(f"  {quality}: {count} ({percentage:.2f}%)")
        
        print("\nR²统计 (拟合优度):")
        print(f"  平均 R²: {df['wind_shear_r2'].mean():.4f}")
        print(f"  R² > 0.95: {(df['wind_shear_r2'] > 0.95).sum()} ({(df['wind_shear_r2'] > 0.95).sum()/len(df)*100:.2f}%)")
        print(f"  R² > 0.85: {(df['wind_shear_r2'] > 0.85).sum()} ({(df['wind_shear_r2'] > 0.85).sum()/len(df)*100:.2f}%)")
        
        return df
    
    def visualize_wind_shear_distribution(self, df, save_path):
        """可视化风切变指数分布"""
        print("\n生成风切变指数分布图...")
        
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle('Wind Shear Exponent Analysis', fontsize=16, fontweight='bold')
        
        # 1. 风切变指数分布直方图
        ax1 = axes[0, 0]
        valid_alpha = df['wind_shear_alpha'].dropna()
        valid_alpha = valid_alpha[(valid_alpha > 0) & (valid_alpha < 1)]  # 物理合理范围
        
        ax1.hist(valid_alpha, bins=50, alpha=0.7, color='#3498db', edgecolor='black')
        ax1.axvline(valid_alpha.mean(), color='red', linestyle='--', linewidth=2, 
                   label=f'Mean = {valid_alpha.mean():.3f}')
        ax1.axvline(valid_alpha.median(), color='green', linestyle='--', linewidth=2,
                   label=f'Median = {valid_alpha.median():.3f}')
        ax1.set_xlabel('Wind Shear Exponent (α)', fontweight='bold', fontsize=12)
        ax1.set_ylabel('Frequency', fontweight='bold', fontsize=12)
        ax1.set_title('(A) Distribution of Wind Shear Exponent', fontweight='bold', fontsize=13)
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # 2. R²分布直方图
        ax2 = axes[0, 1]
        valid_r2 = df['wind_shear_r2'].dropna()
        
        ax2.hist(valid_r2, bins=50, alpha=0.7, color='#e74c3c', edgecolor='black')
        ax2.axvline(valid_r2.mean(), color='blue', linestyle='--', linewidth=2,
                   label=f'Mean R² = {valid_r2.mean():.3f}')
        ax2.set_xlabel('R² (Goodness of Fit)', fontweight='bold', fontsize=12)
        ax2.set_ylabel('Frequency', fontweight='bold', fontsize=12)
        ax2.set_title('(B) Distribution of Fitting Quality (R²)', fontweight='bold', fontsize=13)
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # 3. 风切变指数 vs 10m风速
        ax3 = axes[1, 0]
        sample_size = min(5000, len(df))
        sample_df = df.sample(n=sample_size, random_state=42)
        
        scatter = ax3.scatter(sample_df['obs_wind_speed_10m'], 
                             sample_df['wind_shear_alpha'],
                             c=sample_df['wind_shear_r2'], 
                             cmap='RdYlGn', alpha=0.6, s=20, edgecolors='none')
        ax3.set_xlabel('10m Wind Speed (m/s)', fontweight='bold', fontsize=12)
        ax3.set_ylabel('Wind Shear Exponent (α)', fontweight='bold', fontsize=12)
        ax3.set_title('(C) Wind Shear vs Wind Speed', fontweight='bold', fontsize=13)
        ax3.set_ylim([0, 0.8])
        cbar = plt.colorbar(scatter, ax=ax3)
        cbar.set_label('R² (Fit Quality)', fontweight='bold')
        ax3.grid(True, alpha=0.3)
        
        # 4. 风切变指数 vs 功率
        ax4 = axes[1, 1]
        ax4.scatter(sample_df['wind_shear_alpha'], 
                   sample_df['power'],
                   c=sample_df['obs_wind_speed_10m'],
                   cmap='viridis', alpha=0.6, s=20, edgecolors='none')
        ax4.set_xlabel('Wind Shear Exponent (α)', fontweight='bold', fontsize=12)
        ax4.set_ylabel('Power (kW)', fontweight='bold', fontsize=12)
        ax4.set_title('(D) Wind Shear vs Power Output', fontweight='bold', fontsize=13)
        ax4.set_xlim([0, 0.8])
        cbar = plt.colorbar(scatter, ax=ax4)
        cbar.set_label('10m Wind Speed (m/s)', fontweight='bold')
        ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(os.path.join(save_path, 'wind_shear_distribution.png'), 
                   dpi=300, bbox_inches='tight', facecolor='white')
        plt.savefig(os.path.join(save_path, 'wind_shear_distribution.pdf'),
                   dpi=300, bbox_inches='tight', facecolor='white')
        plt.show()
        
        print(f"风切变分布图已保存至: {save_path}")


class WindDirectionSHAPAnalyzer:
    """增强版风向分类SHAP分析器 - 包含风切变指数"""
    
    def __init__(self, data_path, results_path):
        self.data_path = data_path
        self.results_path = results_path
        self.raw_data = None
        self.east_wind_data = None
        self.west_wind_data = None
        
        # 风切变计算器
        self.wind_shear_calculator = WindShearCalculator()
        
        # 为东风和西风分别创建分析器
        self.east_analyzer = None
        self.west_analyzer = None
        
    def load_and_calculate_wind_shear(self):
        """加载数据并计算风切变指数"""
        print("=== 第1步: 加载数据 ===")
        
        # 加载原始数据
        self.raw_data = pd.read_csv(self.data_path)
        print(f"原始数据形状: {self.raw_data.shape}")
        
        # 计算风切变指数
        self.raw_data = self.wind_shear_calculator.calculate_for_dataframe(self.raw_data)
        
        # 可视化风切变分布
        self.wind_shear_calculator.visualize_wind_shear_distribution(
            self.raw_data, self.results_path
        )
        
        return self.raw_data
    
    def load_and_classify_data(self):
        """按风向分类数据"""
        print("\n=== 第2步: 按风向分类数据 ===")
        
        # 检查wind_direction列
        wind_dir_col10 = 'obs_wind_direction_10m'
        wind_dir_col30 = 'obs_wind_direction_30m'
        wind_dir_col50 = 'obs_wind_direction_50m'
        wind_dir_col70 = 'obs_wind_direction_70m'
        
        if wind_dir_col70 not in self.raw_data.columns:
            raise ValueError(f"未找到列: {wind_dir_col70}")
        
        print(f"风向数据范围: {self.raw_data[wind_dir_col70].min():.2f}° - {self.raw_data[wind_dir_col70].max():.2f}°")
        
        # 东风区间：45° 到 135°（确保所有高度都在此区间）
        east_mask = (
            (self.raw_data[wind_dir_col10] >= 45) & (self.raw_data[wind_dir_col10] <= 135) &
            (self.raw_data[wind_dir_col30] >= 45) & (self.raw_data[wind_dir_col30] <= 135) &
            (self.raw_data[wind_dir_col50] >= 45) & (self.raw_data[wind_dir_col50] <= 135) &
            (self.raw_data[wind_dir_col70] >= 45) & (self.raw_data[wind_dir_col70] <= 135)
        )
        self.east_wind_data = self.raw_data[east_mask].copy()
        
        # 西风区间：225° 到 315°
        west_mask = (
            (self.raw_data[wind_dir_col10] >= 225) & (self.raw_data[wind_dir_col10] <= 315) &
            (self.raw_data[wind_dir_col30] >= 225) & (self.raw_data[wind_dir_col30] <= 315) &
            (self.raw_data[wind_dir_col50] >= 225) & (self.raw_data[wind_dir_col50] <= 315) &
            (self.raw_data[wind_dir_col70] >= 225) & (self.raw_data[wind_dir_col70] <= 315)
        )
        self.west_wind_data = self.raw_data[west_mask].copy()
        
        # 统计信息
        excluded_count = len(self.raw_data) - len(self.east_wind_data) - len(self.west_wind_data)
        
        print(f"\n风向分类结果:")
        print(f"  东风区间 (45°-135°): {len(self.east_wind_data)} 条数据")
        print(f"  西风区间 (225°-315°): {len(self.west_wind_data)} 条数据")
        print(f"  南北风向 (排除): {excluded_count} 条数据")
        
        # 显示功率和风切变统计
        print(f"\n东风区间统计:")
        print(f"  平均功率: {self.east_wind_data['power'].mean():.2f} kW")
        print(f"  平均风切变指数: {self.east_wind_data['wind_shear_alpha'].mean():.4f}")
        
        print(f"\n西风区间统计:")
        print(f"  平均功率: {self.west_wind_data['power'].mean():.2f} kW")
        print(f"  平均风切变指数: {self.west_wind_data['wind_shear_alpha'].mean():.4f}")
        
        return self.east_wind_data, self.west_wind_data
    
    def run_analysis(self):
        """运行完整的风向分类SHAP分析（包含风切变指数）"""
        print("="*70)
        print("增强版按风向分类的SHAP分析 - 包含风切变指数")
        print("="*70)
        print()
        
        # 1. 加载数据并计算风切变指数
        self.load_and_calculate_wind_shear()
        
        # 2. 数据分类
        self.load_and_classify_data()
        
        # 3. 为东风区间创建分析器
        print("\n" + "="*60)
        print("第3步: 分析东风区间数据 (45°-135°)")
        print("="*60)
        
        east_results_path = os.path.join(self.results_path, "east_wind")
        os.makedirs(east_results_path, exist_ok=True)
        
        # 保存东风数据
        east_data_path = os.path.join(east_results_path, "east_wind_data_with_shear.csv")
        self.east_wind_data.to_csv(east_data_path, index=False)
        
        self.east_analyzer = CustomSHAPVisualizer(
            data=self.east_wind_data,
            results_path=east_results_path,
            wind_type="East Wind (45°-135°)"
        )
        east_model = self.east_analyzer.run_analysis()
        
        # 4. 为西风区间创建分析器
        print("\n" + "="*60)
        print("第4步: 分析西风区间数据 (225°-315°)")
        print("="*60)
        
        west_results_path = os.path.join(self.results_path, "west_wind")
        os.makedirs(west_results_path, exist_ok=True)
        
        # 保存西风数据
        west_data_path = os.path.join(west_results_path, "west_wind_data_with_shear.csv")
        self.west_wind_data.to_csv(west_data_path, index=False)
        
        self.west_analyzer = CustomSHAPVisualizer(
            data=self.west_wind_data,
            results_path=west_results_path,
            wind_type="West Wind (225°-315°)"
        )
        west_model = self.west_analyzer.run_analysis()
        
        # 5. 生成对比分析
        self.create_comparison_analysis()
        
        # 6. 生成风切变重要性对比分析
        self.create_wind_shear_importance_comparison()
        
        print("\n" + "="*70)
        print("增强版按风向分类的SHAP分析完成!")
        print("="*70)
        print("生成的文件:")
        print(f"  风切变分布: {self.results_path}/wind_shear_distribution.png")
        print(f"  东风区间结果: {east_results_path}/")
        print(f"  西风区间结果: {west_results_path}/")
        print(f"  对比分析: {self.results_path}/wind_direction_comparison.png")
        print(f"  风切变重要性: {self.results_path}/wind_shear_importance_comparison.png")
        
        return east_model, west_model
    
    def create_comparison_analysis(self):
        """创建东西风区间的对比分析"""
        print("\n生成风向对比分析...")
        
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle('Wind Direction Analysis Comparison: East vs West (With Wind Shear)', 
                    fontsize=16, fontweight='bold')
        
        # 1. 功率分布对比
        ax1 = axes[0, 0]
        east_power = self.east_wind_data['power'].dropna()
        west_power = self.west_wind_data['power'].dropna()
        
        ax1.hist(east_power, bins=30, alpha=0.7, label='East Wind (45°-135°)', 
                color='#FF6B6B', density=True)
        ax1.hist(west_power, bins=30, alpha=0.7, label='West Wind (225°-315°)', 
                color='#4ECDC4', density=True)
        ax1.set_xlabel('Power (kW)', fontweight='bold')
        ax1.set_ylabel('Density', fontweight='bold')
        ax1.set_title('(A) Power Distribution Comparison', fontweight='bold')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # 2. 风切变指数分布对比
        ax2 = axes[0, 1]
        east_alpha = self.east_wind_data['wind_shear_alpha'].dropna()
        east_alpha = east_alpha[(east_alpha > 0) & (east_alpha < 1)]
        west_alpha = self.west_wind_data['wind_shear_alpha'].dropna()
        west_alpha = west_alpha[(west_alpha > 0) & (west_alpha < 1)]
        
        ax2.hist(east_alpha, bins=30, alpha=0.7, label='East Wind', 
                color='#FF6B6B', density=True)
        ax2.hist(west_alpha, bins=30, alpha=0.7, label='West Wind', 
                color='#4ECDC4', density=True)
        ax2.set_xlabel('Wind Shear Exponent (α)', fontweight='bold')
        ax2.set_ylabel('Density', fontweight='bold')
        ax2.set_title('(B) Wind Shear Distribution Comparison', fontweight='bold')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # 3. 数据统计对比
        ax3 = axes[1, 0]
        stats_data = {
            'East Wind': [
                len(self.east_wind_data),
                self.east_wind_data['power'].mean(),
                self.east_wind_data['obs_wind_speed_10m'].mean(),
                self.east_wind_data['wind_shear_alpha'].mean() * 100  # 放大100倍以便显示
            ],
            'West Wind': [
                len(self.west_wind_data),
                self.west_wind_data['power'].mean(),
                self.west_wind_data['obs_wind_speed_10m'].mean(),
                self.west_wind_data['wind_shear_alpha'].mean() * 100
            ]
        }
        
        stats_labels = ['Sample Count', 'Mean Power (kW)', 
                       'Mean Wind Speed (m/s)', 'Mean α (×100)']
        x = np.arange(len(stats_labels))
        width = 0.35
        
        bars1 = ax3.bar(x - width/2, stats_data['East Wind'], width, 
                       label='East Wind', color='#FF6B6B', alpha=0.8)
        bars2 = ax3.bar(x + width/2, stats_data['West Wind'], width, 
                       label='West Wind', color='#4ECDC4', alpha=0.8)
        
        ax3.set_xlabel('Statistics', fontweight='bold')
        ax3.set_ylabel('Value', fontweight='bold')
        ax3.set_title('(C) Statistical Comparison', fontweight='bold')
        ax3.set_xticks(x)
        ax3.set_xticklabels(stats_labels, rotation=45, ha='right')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        
        # 添加数值标签
        for bars in [bars1, bars2]:
            for bar in bars:
                height = bar.get_height()
                ax3.text(bar.get_x() + bar.get_width()/2., height + height*0.01,
                        f'{height:.1f}', ha='center', va='bottom', fontsize=8)
        
        # 4. 模型性能对比
        ax4 = axes[1, 1]
        if hasattr(self.east_analyzer, 'model') and hasattr(self.west_analyzer, 'model'):
            east_r2 = getattr(self.east_analyzer, 'test_r2', 0)
            west_r2 = getattr(self.west_analyzer, 'test_r2', 0)
            east_rmse = getattr(self.east_analyzer, 'test_rmse', 0)
            west_rmse = getattr(self.west_analyzer, 'test_rmse', 0)
            
            metrics = ['R² Score', 'RMSE (kW)']
            east_metrics = [east_r2, east_rmse]
            west_metrics = [west_r2, west_rmse]
            
            x = np.arange(len(metrics))
            width = 0.35
            
            ax4.bar(x - width/2, east_metrics, width, label='East Wind', 
                   color='#FF6B6B', alpha=0.8)
            ax4.bar(x + width/2, west_metrics, width, label='West Wind', 
                   color='#4ECDC4', alpha=0.8)
            
            ax4.set_xlabel('Metrics', fontweight='bold')
            ax4.set_ylabel('Value', fontweight='bold')
            ax4.set_title('(D) Model Performance Comparison', fontweight='bold')
            ax4.set_xticks(x)
            ax4.set_xticklabels(metrics)
            ax4.legend()
            ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        # 保存对比图
        comparison_path = os.path.join(self.results_path, 'wind_direction_comparison.png')
        plt.savefig(comparison_path, dpi=300, bbox_inches='tight', facecolor='white')
        plt.savefig(comparison_path.replace('.png', '.pdf'), 
                   dpi=300, bbox_inches='tight', facecolor='white')
        plt.show()
    
    def create_wind_shear_importance_comparison(self):
        """创建风切变指数重要性对比分析"""
        print("\n生成风切变重要性对比分析...")
        
        if not hasattr(self.east_analyzer, 'shap_values') or not hasattr(self.west_analyzer, 'shap_values'):
            print("警告: SHAP分析未完成，跳过风切变重要性对比")
            return
        
        fig, axes = plt.subplots(1, 2, figsize=(18, 7))
        fig.suptitle('Wind Shear Exponent Importance in SHAP Analysis', 
                    fontsize=16, fontweight='bold')
        
        # 东风区间
        ax1 = axes[0]
        self._plot_wind_shear_importance(
            ax1, 
            self.east_analyzer.shap_values,
            self.east_analyzer.feature_names,
            "East Wind (45°-135°)",
            '#FF6B6B'
        )
        
        # 西风区间
        ax2 = axes[1]
        self._plot_wind_shear_importance(
            ax2,
            self.west_analyzer.shap_values,
            self.west_analyzer.feature_names,
            "West Wind (225°-315°)",
            '#4ECDC4'
        )
        
        plt.tight_layout()
        
        # 保存图形
        save_path = os.path.join(self.results_path, 'wind_shear_importance_comparison.png')
        plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
        plt.savefig(save_path.replace('.png', '.pdf'), 
                   dpi=300, bbox_inches='tight', facecolor='white')
        plt.show()
        
        print(f"风切变重要性对比图已保存至: {save_path}")
    
    def _plot_wind_shear_importance(self, ax, shap_values, feature_names, title, color):
        """绘制单个风向区间的风切变重要性"""
        # 计算特征重要性
        importance = np.abs(shap_values).mean(0)
        
        # 找到风切变指数的索引
        wind_shear_idx = None
        for i, name in enumerate(feature_names):
            if 'wind_shear_alpha' in name:
                wind_shear_idx = i
                break
        
        # 创建显示名称
        display_names = []
        for name in feature_names:
            if 'wind_shear_alpha' in name:
                display_names.append('Wind Shear α ★')
            else:
                display_names.append(name.replace('obs_', '').replace('_', ' ').title())
        
        # 选择前15个最重要的特征
        top_indices = np.argsort(importance)[-15:]
        top_importance = importance[top_indices]
        top_names = [display_names[i] for i in top_indices]
        
        # 确定颜色（高亮风切变指数）
        colors = []
        for idx in top_indices:
            if idx == wind_shear_idx:
                colors.append('#FF6B35')  # 橙红色高亮
            else:
                colors.append(color)
        
        # 绘制水平条形图
        bars = ax.barh(range(len(top_indices)), top_importance, color=colors, alpha=0.8)
        
        # 设置y轴标签
        ax.set_yticks(range(len(top_indices)))
        ax.set_yticklabels(top_names, fontsize=10)
        
        # 高亮风切变指数的标签
        for i, (idx, label) in enumerate(zip(top_indices, ax.get_yticklabels())):
            if idx == wind_shear_idx:
                label.set_fontweight('bold')
                label.set_fontsize(11)
                label.set_color('#FF6B35')
        
        # 设置标题和标签
        ax.set_xlabel('Mean |SHAP Value|', fontsize=11, fontweight='bold')
        ax.set_title(title, fontsize=13, fontweight='bold', pad=15)
        
        # 添加数值标签
        for i, (bar, value) in enumerate(zip(bars, top_importance)):
            width = bar.get_width()
            ax.text(width + width*0.02, bar.get_y() + bar.get_height()/2,
                   f'{value:.3f}', ha='left', va='center', fontsize=9, fontweight='bold')
        
        # 如果风切变指数在前15名，添加排名标注
        if wind_shear_idx is not None and wind_shear_idx in top_indices:
            rank = len(top_indices) - list(top_indices).index(wind_shear_idx)
            ax.text(0.98, 0.98, f'Wind Shear α\nRank: #{rank}', 
                   transform=ax.transAxes, fontsize=11, fontweight='bold',
                   ha='right', va='top', color='#FF6B35',
                   bbox=dict(boxstyle='round,pad=0.5', facecolor='white', 
                           edgecolor='#FF6B35', linewidth=2, alpha=0.9))
        
        # 美化图表
        ax.grid(True, alpha=0.3, axis='x')
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)


class CustomSHAPVisualizer:
    """SHAP分析器 - 增强版（包含风切变指数）"""
    
    def __init__(self, data, results_path, wind_type=""):
        self.data = data
        self.results_path = results_path
        self.wind_type = wind_type
        self.features = None
        self.target = None
        self.feature_names = None
        self.model = None
        self.X_test = None
        self.y_test = None
        self.shap_values = None
        self.X_sample = None
        self.explainer = None
        
        # 存储性能指标
        self.test_r2 = 0
        self.test_rmse = 0
        self.test_mae = 0
        
    def load_and_prepare_data(self):
        """加载和预处理数据"""
        print(f"预处理{self.wind_type}数据...")
        
        print(f"输入数据形状: {self.data.shape}")
        
        # 选择观测数据列 + 风切变指数
        obs_columns = [col for col in self.data.columns if col.startswith('obs_')]
        obs_columns += ['datetime', 'power', 'wind_shear_alpha', 'wind_shear_r2']
        
        # 移除密度和湿度变量
        obs_columns = [col for col in obs_columns if 'density' not in col and 'humidity' not in col]
        
        self.data = self.data[obs_columns].copy()
        print(f"选择列后的数据形状: {self.data.shape}")
        
        # 清理数据
        self.data = self.data.dropna()
        self.data = self.data[self.data['power'] >= 0]
        
        # 过滤风切变指数：保留物理合理的值
        self.data = self.data[
            (self.data['wind_shear_alpha'] > 0) & 
            (self.data['wind_shear_alpha'] < 1) &
            (self.data['wind_shear_r2'] > 0.7)  # 保留拟合较好的数据
        ]
        
        print(f"最终数据形状: {self.data.shape}")
        print(f"风切变指数范围: {self.data['wind_shear_alpha'].min():.4f} - {self.data['wind_shear_alpha'].max():.4f}")
        
        return self.data
    
    def process_wind_direction(self):
        """处理风向变量为sin/cos分量"""
        print("处理风向变量...")
        
        wind_dir_cols = [col for col in self.data.columns if 'wind_direction' in col]
        print(f"发现{len(wind_dir_cols)}个风向变量: {wind_dir_cols}")
        
        for col in wind_dir_cols:
            # 转换为弧度
            wind_dir_rad = np.deg2rad(self.data[col])
            
            # 创建sin/cos分量
            sin_col = col.replace('wind_direction', 'wind_dir_sin')
            cos_col = col.replace('wind_direction', 'wind_dir_cos')
            
            self.data[sin_col] = np.sin(wind_dir_rad)
            self.data[cos_col] = np.cos(wind_dir_rad)
            
            print(f"  已创建: {sin_col}, {cos_col}")
        
        # 移除原始风向列
        self.data = self.data.drop(columns=wind_dir_cols)
        print(f"已移除原始风向列")
    
    def create_features(self):
        """创建特征矩阵"""
        print("创建特征矩阵...")
        
        # 处理风向
        self.process_wind_direction()
        
        # 选择特征列（包括风切变指数）
        feature_cols = [col for col in self.data.columns 
                       if col not in ['datetime', 'power', 'wind_shear_r2']]
        
        print(f"使用{len(feature_cols)}个特征")
        print(f"特征列表: {feature_cols}")
        
        # 创建特征矩阵
        self.features = self.data[feature_cols].values
        self.target = self.data['power'].values
        self.feature_names = feature_cols
        
        print(f"特征矩阵形状: {self.features.shape}")
        
        # 检查风切变指数是否在特征中
        if 'wind_shear_alpha' in feature_cols:
            print("✓ 风切变指数已成功添加为特征")
        else:
            print("✗ 警告: 风切变指数未被添加为特征")
        
        return feature_cols
    
    def train_lightgbm(self):
        """训练LightGBM模型"""
        print(f"训练{self.wind_type} LightGBM模型...")
        
        # 检查数据量
        if len(self.features) < 100:
            print(f"警告: 数据量较少 ({len(self.features)} 样本)")
        
        # 分割数据
        X_train, X_test, y_train, y_test = train_test_split(
            self.features, self.target, test_size=0.2, random_state=42
        )
        
        # 保存测试数据
        self.X_test = X_test
        self.y_test = y_test
        
        # LightGBM参数
        lgb_params = {
            'objective': 'regression',
            'metric': 'rmse',
            'boosting_type': 'gbdt',
            'num_leaves': min(31, max(10, len(self.features) // 20)),
            'learning_rate': 0.1,
            'feature_fraction': 0.8,
            'bagging_fraction': 0.8,
            'reg_alpha': 0.1,
            'reg_lambda': 0.1,
            'min_child_samples': max(10, len(X_train) // 100),
            'n_estimators': 100,
            'random_state': 42,
            'verbose': -1
        }
        
        # 训练模型
        self.model = lgb.LGBMRegressor(**lgb_params)
        self.model.fit(X_train, y_train)
        
        # 评估性能
        y_pred_test = self.model.predict(X_test)
        self.test_r2 = r2_score(y_test, y_pred_test)
        self.test_rmse = np.sqrt(mean_squared_error(y_test, y_pred_test))
        self.test_mae = mean_absolute_error(y_test, y_pred_test)
        
        print(f"{self.wind_type} 模型性能:")
        print(f"  测试集 R²: {self.test_r2:.3f}")
        print(f"  测试集 RMSE: {self.test_rmse:.3f}")
        print(f"  测试集 MAE: {self.test_mae:.3f}")
        
        # 输出风切变指数的特征重要性
        if 'wind_shear_alpha' in self.feature_names:
            wind_shear_idx = self.feature_names.index('wind_shear_alpha')
            feature_importance = self.model.feature_importances_
            wind_shear_importance = feature_importance[wind_shear_idx]
            rank = np.argsort(feature_importance)[::-1].tolist().index(wind_shear_idx) + 1
            
            print(f"\n风切变指数特征重要性:")
            print(f"  重要性得分: {wind_shear_importance:.4f}")
            print(f"  排名: #{rank}/{len(feature_importance)}")
        
        return self.model
    
    def calculate_shap_values(self):
        """计算SHAP值"""
        print(f"计算{self.wind_type} SHAP值...")
        
        # 创建SHAP解释器
        self.explainer = shap.TreeExplainer(self.model)
        
        # 使用测试数据样本
        sample_size = min(500, len(self.X_test))
        indices = np.random.choice(len(self.X_test), sample_size, replace=False)
        self.X_sample = self.X_test[indices]
        
        print(f"计算{sample_size}个样本的SHAP值...")
        self.shap_values = self.explainer.shap_values(self.X_sample)
        
        # 输出风切变指数的SHAP重要性
        if 'wind_shear_alpha' in self.feature_names:
            wind_shear_idx = self.feature_names.index('wind_shear_alpha')
            shap_importance = np.abs(self.shap_values[:, wind_shear_idx]).mean()
            all_shap_importance = np.abs(self.shap_values).mean(0)
            rank = np.argsort(all_shap_importance)[::-1].tolist().index(wind_shear_idx) + 1
            
            print(f"\n风切变指数SHAP重要性:")
            print(f"  平均|SHAP值|: {shap_importance:.4f}")
            print(f"  排名: #{rank}/{len(self.feature_names)}")
        
        return self.shap_values, self.X_sample
    
    def plot_combined_visualization(self):
        """绘制组合的SHAP可视化"""
        print(f"绘制{self.wind_type} SHAP可视化...")
        
        fig = plt.figure(figsize=(16, 12))
        gs = fig.add_gridspec(2, 2, height_ratios=[1, 1.2], hspace=0.3, wspace=0.3)
        
        # 子图1: 特征重要性
        ax1 = fig.add_subplot(gs[0, 0])
        self._plot_feature_importance_subplot(ax1)
        
        # 子图2: 特征影响分布
        ax2 = fig.add_subplot(gs[0, 1])
        self._plot_feature_impact_distribution_subplot(ax2)
        
        # 子图3: 瀑布图
        ax3 = fig.add_subplot(gs[1, :])
        self._plot_waterfall_subplot(ax3)
        
        # 添加总标题
        title = f'SHAP Analysis Results - {self.wind_type} (with Wind Shear α)' if self.wind_type else 'SHAP Analysis Results'
        fig.suptitle(title, fontsize=18, fontweight='bold', y=0.98)
        
        # 保存图形
        plt.savefig(f'{self.results_path}/combined_shap_analysis.png',
                   dpi=300, bbox_inches='tight', facecolor='white')
        plt.savefig(f'{self.results_path}/combined_shap_analysis.pdf',
                   dpi=300, bbox_inches='tight', facecolor='white')
        plt.show()
    
    def _plot_feature_importance_subplot(self, ax):
        """绘制特征重要性（高亮风切变指数）"""
        importance = np.abs(self.shap_values).mean(0)
        
        # 找到风切变指数
        wind_shear_idx = None
        for i, name in enumerate(self.feature_names):
            if 'wind_shear_alpha' in name:
                wind_shear_idx = i
                break
        
        # 创建显示名称
        display_names = []
        for name in self.feature_names:
            if 'wind_shear_alpha' in name:
                display_names.append('Wind Shear α ★')
            else:
                display_names.append(name.replace('obs_', '').replace('_', ' ').title())
        
        # 选择前10个
        top_indices = np.argsort(importance)[-10:]
        top_importance = importance[top_indices]
        top_names = [display_names[i] for i in top_indices]
        
        # 颜色（高亮风切变）
        colors = []
        for idx in top_indices:
            if idx == wind_shear_idx:
                colors.append('#FF6B35')
            else:
                colors.append(plt.cm.viridis(0.5))
        
        # 绘制
        bars = ax.barh(range(len(top_indices)), top_importance, color=colors, alpha=0.8)
        
        ax.set_yticks(range(len(top_indices)))
        ax.set_yticklabels(top_names, fontsize=9)
        
        # 高亮标签
        for i, (idx, label) in enumerate(zip(top_indices, ax.get_yticklabels())):
            if idx == wind_shear_idx:
                label.set_fontweight('bold')
                label.set_fontsize(10)
                label.set_color('#FF6B35')
        
        ax.set_xlabel('Mean |SHAP Value|', fontsize=10, fontweight='bold')
        ax.set_title('(A) Feature Importance', fontsize=12, fontweight='bold', pad=15)
        
        # 数值标签
        for bar, value in zip(bars, top_importance):
            width = bar.get_width()
            ax.text(width + width*0.02, bar.get_y() + bar.get_height()/2,
                   f'{value:.3f}', ha='left', va='center', fontsize=8, fontweight='bold')
        
        ax.grid(True, alpha=0.3, axis='x')
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
    
    def _plot_feature_impact_distribution_subplot(self, ax):
        """绘制特征影响分布"""
        importance = np.abs(self.shap_values).mean(0)
        top_indices = np.argsort(importance)[-10:]
        
        display_names = [self.feature_names[i].replace('obs_', '').replace('_', ' ').title()
                        for i in top_indices]
        
        # 用星号标记风切变
        for i, idx in enumerate(top_indices):
            if 'wind_shear_alpha' in self.feature_names[idx]:
                display_names[i] = 'Wind Shear α ★'
        
        shap_colors = ['#008bfb', '#ff0051']
        from matplotlib.colors import LinearSegmentedColormap
        shap_cmap = LinearSegmentedColormap.from_list('shap_classic', shap_colors, N=256)
        
        y_positions = []
        for i, feat_idx in enumerate(top_indices):
            shap_vals = self.shap_values[:, feat_idx]
            feature_vals = self.X_sample[:, feat_idx]
            
            norm_feature_vals = (feature_vals - feature_vals.min()) / (feature_vals.max() - feature_vals.min() + 1e-8)
            
            y_pos = np.full_like(shap_vals, i) + np.random.normal(0, 0.08, len(shap_vals))
            y_positions.append(i)
            
            scatter = ax.scatter(shap_vals, y_pos, c=norm_feature_vals,
                               cmap=shap_cmap, alpha=0.7, s=18, edgecolors='white', linewidth=0.4)
        
        ax.set_yticks(y_positions)
        ax.set_yticklabels(display_names, fontsize=9)
        
        # 高亮风切变标签
        for label in ax.get_yticklabels():
            if '★' in label.get_text():
                label.set_fontweight('bold')
                label.set_color('#FF6B35')
        
        ax.set_xlabel('SHAP Value (Impact on Model Output)', fontsize=10, fontweight='bold')
        ax.set_title('(B) Feature Impact Distribution', fontsize=12, fontweight='bold', pad=15)
        
        cbar = plt.colorbar(scatter, ax=ax, pad=0.02, aspect=20)
        cbar.set_label('Feature Value', fontsize=9, fontweight='bold')
        
        ax.axvline(x=0, color='#333333', linestyle='-', alpha=0.8, linewidth=1.5)
        ax.grid(True, alpha=0.3, axis='x', linestyle='--')
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
    
    def _plot_waterfall_subplot(self, ax):
        """绘制瀑布图"""
        y_pred_sample = self.model.predict(self.X_sample)
        median_idx = np.argsort(np.abs(y_pred_sample - np.median(y_pred_sample)))[0]
        
        sample_shap = self.shap_values[median_idx]
        sample_features = self.X_sample[median_idx]
        base_value = self.explainer.expected_value
        prediction = y_pred_sample[median_idx]
        
        top_indices = np.argsort(np.abs(sample_shap))[-10:]
        
        feature_names = []
        for i in top_indices:
            name = self.feature_names[i].replace('obs_', '').replace('_', ' ').title()
            if 'wind_shear_alpha' in self.feature_names[i]:
                name = 'Wind Shear α ★'
            feature_names.append(name)
        
        shap_values_subset = sample_shap[top_indices]
        feature_values_subset = sample_features[top_indices]
        
        sorted_indices = np.argsort(shap_values_subset)
        feature_names = [feature_names[i] for i in sorted_indices]
        shap_values_subset = shap_values_subset[sorted_indices]
        feature_values_subset = feature_values_subset[sorted_indices]
        
        ax.set_facecolor('#f9f9f9')
        
        cumulative = base_value
        bar_height = 0.6
        
        x_min = min(base_value, prediction) - 5
        x_max = max(base_value, prediction) + 2
        feature_label_x = x_min + 0.5
        
        for i, (name, feat_val, shap_val) in enumerate(zip(feature_names, feature_values_subset, shap_values_subset)):
            y = len(feature_names) - 1 - i
            
            if shap_val >= 0:
                start_x = cumulative
                width = shap_val
                color = '#ff6b6b'
                label_text = f'+{shap_val:.2f}'
            else:
                width = abs(shap_val)
                start_x = cumulative + shap_val
                color = '#4ecdc4'
                label_text = f'{shap_val:.2f}'
            
            # 高亮风切变条形
            if '★' in name:
                color = '#FF6B35'
                edgecolor = '#FF6B35'
                linewidth = 2.5
            else:
                edgecolor = 'white'
                linewidth = 1.5
            
            bar = ax.barh(y, width, left=start_x, height=bar_height,
                         color=color, alpha=0.85, edgecolor=edgecolor, linewidth=linewidth)
            
            text_x = start_x + width/2
            ax.text(text_x, y, label_text, ha='center', va='center',
                   fontweight='bold', color='white', fontsize=9)
            
            # 特征名称（高亮风切变）
            if '★' in name:
                ax.text(feature_label_x, y, name, ha='left', va='center',
                       fontsize=11, fontweight='bold', color='#FF6B35')
            else:
                ax.text(feature_label_x, y, name, ha='left', va='center',
                       fontsize=10, fontweight='bold', color='#2c3e50')
            
            ax.text(feature_label_x, y - 0.35, f'value = {feat_val:.3f}',
                   ha='left', va='center', fontsize=8, color='#7f8c8d', style='italic')
            
            cumulative += shap_val
        
        y_min = -0.8
        y_max = len(feature_names) - 0.2
        
        ax.axvline(x=base_value, color='#34495e', linestyle='-', alpha=0.8, linewidth=2.5)
        ax.text(base_value, y_max + 0.4, f'Baseline\n{base_value:.2f}',
               ha='center', va='bottom', fontsize=10, fontweight='bold',
               bbox=dict(boxstyle="round,pad=0.4", facecolor='#ecf0f1',
                        edgecolor='#34495e', alpha=0.9))
        
        ax.axvline(x=prediction, color='#e74c3c', linestyle='-', alpha=0.8, linewidth=2.5)
        ax.text(prediction, y_max + 0.4, f'Prediction\n{prediction:.2f}',
               ha='center', va='bottom', fontsize=10, fontweight='bold',
               bbox=dict(boxstyle="round,pad=0.4", facecolor='#fadbd8',
                        edgecolor='#e74c3c', alpha=0.9))
        
        ax.set_xlim(x_min, x_max)
        ax.set_ylim(y_min, y_max + 1)
        ax.set_yticks([])
        
        ax.set_xlabel('Model Output (kW)', fontsize=11, fontweight='bold', color='#2c3e50')
        ax.set_title(f'(C) Feature Contributions for Single Prediction ({prediction:.1f} kW)',
                    fontsize=12, fontweight='bold', pad=20, color='#2c3e50')
        
        ax.grid(True, alpha=0.4, axis='x', linestyle='--', linewidth=0.8)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['left'].set_visible(False)
        ax.spines['bottom'].set_linewidth(1.5)
        ax.spines['bottom'].set_color('#34495e')
    
    def run_analysis(self):
        """运行完整分析"""
        print(f"=== {self.wind_type} SHAP可视化分析 (含风切变指数) ===")
        
        self.load_and_prepare_data()
        self.create_features()
        self.train_lightgbm()
        self.calculate_shap_values()
        self.plot_combined_visualization()
        
        print(f"\n{self.wind_type} SHAP可视化完成!")
        print("生成的文件:")
        print(f"  - {self.results_path}/combined_shap_analysis.png")
        print(f"  - {self.results_path}/combined_shap_analysis.pdf")
        
        return self.model


if __name__ == "__main__":
    # 配置路径
    DATA_PATH = "/Users/xiaxin/work/WindForecast_Project/01_Data/processed/imputed_data/changma_imputed_complete.csv"
    RESULTS_PATH = "/Users/xiaxin/work/WindForecast_Project/03_Results/figures/3-10/wind_shear_shap_analysis"
    
    # 创建结果目录
    os.makedirs(RESULTS_PATH, exist_ok=True)
    
    print("="*70)
    print("增强版风向分类SHAP分析 - 包含风切变指数计算")
    print("="*70)
    print("研究目标: 验证风切变指数是否是风电功率预测的关键贡献因子")
    print("方法: 最小二乘法拟合4层梯度风廓线 (10m, 30m, 50m, 70m)")
    print("="*70)
    print()
    
    # 运行分析
    analyzer = WindDirectionSHAPAnalyzer(DATA_PATH, RESULTS_PATH)
    east_model, west_model = analyzer.run_analysis()
    
    print("\n" + "="*70)
    print("分析完成!")
    print("="*70)