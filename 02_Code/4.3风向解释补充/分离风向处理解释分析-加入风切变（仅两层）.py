#!/usr/bin/env python3
"""
增强版按风向分类的SHAP分析 - 自适应风切变指数计算
- 东风区间 (45°-135°): 使用10m和30m计算风切变 (受地形影响的近地层)
- 西风区间 (225°-315°): 使用10m和70m计算风切变 (自由流条件)
将风切变指数作为新特征输入LightGBM，分析其对功率预测的贡献
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
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


class AdaptiveWindShearCalculator:
    """自适应风切变指数计算器 - 根据风向选择不同高度层"""
    
    def __init__(self):
        pass
    
    def calculate_wind_shear_two_heights(self, v1, z1, v2, z2):
        """
        使用两个高度的风速计算风切变指数
        
        幂律公式: v2/v1 = (z2/z1)^α
        取对数: α = ln(v2/v1) / ln(z2/z1)
        
        参数:
            v1: 参考高度风速 (m/s)
            z1: 参考高度 (m)
            v2: 目标高度风速 (m/s)
            z2: 目标高度 (m)
        
        返回:
            alpha: float, 风切变指数
        """
        # 检查输入有效性
        if pd.isna(v1) or pd.isna(v2) or v1 <= 0 or v2 <= 0:
            return np.nan
        
        if z1 <= 0 or z2 <= 0 or z1 == z2:
            return np.nan
        
        try:
            # 计算风切变指数
            # α = ln(v2/v1) / ln(z2/z1)
            alpha = np.log(v2 / v1) / np.log(z2 / z1)
            
            # 物理合理性检查 (风切变指数通常在-0.2到0.6之间)
            # 负值表示逆风切变（高度越高风速越小），在稳定层结时可能出现
            if alpha < -0.5 or alpha > 1.0:
                return np.nan
            
            return alpha
            
        except Exception as e:
            return np.nan
    
    def calculate_for_east_wind(self, df):
        """
        为东风区间计算风切变指数 (使用10m和30m)
        
        参数:
            df: DataFrame, 必须包含obs_wind_speed_10m和obs_wind_speed_30m
        
        返回:
            df: DataFrame, 添加了wind_shear_alpha列
        """
        print("\n" + "="*60)
        print("计算东风区间风切变指数 (East Wind)")
        print("="*60)
        print("使用高度: 10m (参考) → 30m (目标)")
        print("原因: 东风来流受地形影响，近地层风切变最显著")
        print(f"幂律公式: v(30m) = v(10m) × (30/10)^α")
        print()
        
        # 检查必需的列
        required_cols = ['obs_wind_speed_10m', 'obs_wind_speed_30m']
        missing_cols = [col for col in required_cols if col not in df.columns]
        if missing_cols:
            raise ValueError(f"缺少必需的列: {missing_cols}")
        
        print(f"✓ 找到所有必需的列")
        
        # 计算风切变指数
        z1, z2 = 10.0, 30.0
        df['wind_shear_alpha'] = df.apply(
            lambda row: self.calculate_wind_shear_two_heights(
                row['obs_wind_speed_10m'], z1,
                row['obs_wind_speed_30m'], z2
            ), axis=1
        )
        
        # 统计信息
        valid_alpha = df['wind_shear_alpha'].dropna()
        print(f"\n计算完成! 总共处理 {len(df)} 条记录")
        print(f"有效数据: {len(valid_alpha)}/{len(df)} ({len(valid_alpha)/len(df)*100:.2f}%)")
        
        if len(valid_alpha) > 0:
            print("\n风切变指数统计 (10m→30m):")
            print(f"  平均值: {valid_alpha.mean():.4f}")
            print(f"  中位数: {valid_alpha.median():.4f}")
            print(f"  标准差: {valid_alpha.std():.4f}")
            print(f"  最小值: {valid_alpha.min():.4f}")
            print(f"  最大值: {valid_alpha.max():.4f}")
            
            # 物理解释
            print("\n物理解释:")
            if valid_alpha.mean() < 0:
                print("  → 平均为负值，表示逆风切变（高度越高风速越小）")
            elif valid_alpha.mean() < 0.15:
                print("  → 弱风切变，接近中性层结")
            elif valid_alpha.mean() < 0.25:
                print("  → 中等风切变，典型的不稳定层结")
            else:
                print("  → 强风切变，稳定层结")
        
        return df
    
    def calculate_for_west_wind(self, df):
        """
        为西风区间计算风切变指数 (使用10m和70m)
        
        参数:
            df: DataFrame, 必须包含obs_wind_speed_10m和obs_wind_speed_70m
        
        返回:
            df: DataFrame, 添加了wind_shear_alpha列
        """
        print("\n" + "="*60)
        print("计算西风区间风切变指数 (West Wind)")
        print("="*60)
        print("使用高度: 10m (参考) → 70m (目标)")
        print("原因: 西风为自由流，使用全高度范围更能反映风廓线特征")
        print(f"幂律公式: v(70m) = v(10m) × (70/10)^α")
        print()
        
        # 检查必需的列
        required_cols = ['obs_wind_speed_10m', 'obs_wind_speed_70m']
        missing_cols = [col for col in required_cols if col not in df.columns]
        if missing_cols:
            raise ValueError(f"缺少必需的列: {missing_cols}")
        
        print(f"✓ 找到所有必需的列")
        
        # 计算风切变指数
        z1, z2 = 10.0, 70.0
        df['wind_shear_alpha'] = df.apply(
            lambda row: self.calculate_wind_shear_two_heights(
                row['obs_wind_speed_10m'], z1,
                row['obs_wind_speed_70m'], z2
            ), axis=1
        )
        
        # 统计信息
        valid_alpha = df['wind_shear_alpha'].dropna()
        print(f"\n计算完成! 总共处理 {len(df)} 条记录")
        print(f"有效数据: {len(valid_alpha)}/{len(df)} ({len(valid_alpha)/len(df)*100:.2f}%)")
        
        if len(valid_alpha) > 0:
            print("\n风切变指数统计 (10m→70m):")
            print(f"  平均值: {valid_alpha.mean():.4f}")
            print(f"  中位数: {valid_alpha.median():.4f}")
            print(f"  标准差: {valid_alpha.std():.4f}")
            print(f"  最小值: {valid_alpha.min():.4f}")
            print(f"  最大值: {valid_alpha.max():.4f}")
            
            # 物理解释
            print("\n物理解释:")
            if valid_alpha.mean() < 0:
                print("  → 平均为负值，表示逆风切变（罕见）")
            elif valid_alpha.mean() < 0.15:
                print("  → 弱风切变，接近中性层结")
            elif valid_alpha.mean() < 0.25:
                print("  → 中等风切变，典型的不稳定层结")
            else:
                print("  → 强风切变，稳定层结")
        
        return df
    
    def visualize_wind_shear_comparison(self, east_df, west_df, save_path):
        """可视化东西风区间的风切变对比"""
        print("\n生成风切变对比分析图...")
        
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle('Adaptive Wind Shear Analysis: East (10m-30m) vs West (10m-70m)', 
                    fontsize=16, fontweight='bold')
        
        # 1. 风切变指数分布对比
        ax1 = axes[0, 0]
        east_alpha = east_df['wind_shear_alpha'].dropna()
        east_alpha = east_alpha[(east_alpha > -0.5) & (east_alpha < 1)]
        west_alpha = west_df['wind_shear_alpha'].dropna()
        west_alpha = west_alpha[(west_alpha > -0.5) & (west_alpha < 1)]
        
        if len(east_alpha) > 0:
            ax1.hist(east_alpha, bins=40, alpha=0.7, label=f'East Wind (10m→30m)\nMean={east_alpha.mean():.3f}', 
                    color='#FF6B6B', density=True, edgecolor='black', linewidth=0.5)
        if len(west_alpha) > 0:
            ax1.hist(west_alpha, bins=40, alpha=0.7, label=f'West Wind (10m→70m)\nMean={west_alpha.mean():.3f}', 
                    color='#4ECDC4', density=True, edgecolor='black', linewidth=0.5)
        
        ax1.axvline(0, color='gray', linestyle='--', linewidth=1, alpha=0.5)
        ax1.set_xlabel('Wind Shear Exponent (α)', fontweight='bold', fontsize=12)
        ax1.set_ylabel('Density', fontweight='bold', fontsize=12)
        ax1.set_title('(A) Wind Shear Distribution Comparison', fontweight='bold', fontsize=13)
        ax1.legend(fontsize=10)
        ax1.grid(True, alpha=0.3)
        
        # 2. 风切变 vs 10m风速
        ax2 = axes[0, 1]
        
        # 东风
        if len(east_alpha) > 0:
            sample_east = east_df.sample(n=min(2000, len(east_df)), random_state=42)
            sample_east = sample_east.dropna(subset=['obs_wind_speed_10m', 'wind_shear_alpha'])
            ax2.scatter(sample_east['obs_wind_speed_10m'], sample_east['wind_shear_alpha'],
                       alpha=0.4, s=15, color='#FF6B6B', label='East Wind', edgecolors='none')
        
        # 西风
        if len(west_alpha) > 0:
            sample_west = west_df.sample(n=min(2000, len(west_df)), random_state=42)
            sample_west = sample_west.dropna(subset=['obs_wind_speed_10m', 'wind_shear_alpha'])
            ax2.scatter(sample_west['obs_wind_speed_10m'], sample_west['wind_shear_alpha'],
                       alpha=0.4, s=15, color='#4ECDC4', label='West Wind', edgecolors='none')
        
        ax2.axhline(0, color='gray', linestyle='--', linewidth=1, alpha=0.5)
        ax2.set_xlabel('10m Wind Speed (m/s)', fontweight='bold', fontsize=12)
        ax2.set_ylabel('Wind Shear Exponent (α)', fontweight='bold', fontsize=12)
        ax2.set_title('(B) Wind Shear vs Wind Speed', fontweight='bold', fontsize=13)
        ax2.set_ylim([-0.3, 0.6])
        ax2.legend(fontsize=10)
        ax2.grid(True, alpha=0.3)
        
        # 3. 风切变 vs 功率
        ax3 = axes[1, 0]
        
        # 东风
        if len(east_alpha) > 0 and 'power' in east_df.columns:
            sample_east = east_df.sample(n=min(2000, len(east_df)), random_state=42)
            sample_east = sample_east.dropna(subset=['wind_shear_alpha', 'power'])
            scatter1 = ax3.scatter(sample_east['wind_shear_alpha'], sample_east['power'],
                                  c=sample_east['obs_wind_speed_10m'], cmap='Reds',
                                  alpha=0.5, s=20, vmin=0, vmax=15, edgecolors='none')
        
        # 西风
        if len(west_alpha) > 0 and 'power' in west_df.columns:
            sample_west = west_df.sample(n=min(2000, len(west_df)), random_state=42)
            sample_west = sample_west.dropna(subset=['wind_shear_alpha', 'power'])
            scatter2 = ax3.scatter(sample_west['wind_shear_alpha'], sample_west['power'],
                                  c=sample_west['obs_wind_speed_10m'], cmap='Blues',
                                  alpha=0.5, s=20, vmin=0, vmax=15, edgecolors='none')
        
        ax3.axvline(0, color='gray', linestyle='--', linewidth=1, alpha=0.5)
        ax3.set_xlabel('Wind Shear Exponent (α)', fontweight='bold', fontsize=12)
        ax3.set_ylabel('Power (kW)', fontweight='bold', fontsize=12)
        ax3.set_title('(C) Wind Shear vs Power Output', fontweight='bold', fontsize=13)
        ax3.set_xlim([-0.3, 0.6])
        
        # 添加颜色条
        if len(east_alpha) > 0:
            cbar1 = plt.colorbar(scatter1, ax=ax3, pad=0.02, aspect=15)
            cbar1.set_label('10m Wind Speed (m/s)', fontsize=9, fontweight='bold')
        
        ax3.grid(True, alpha=0.3)
        
        # 4. 统计对比
        ax4 = axes[1, 1]
        
        stats_data = []
        labels = []
        colors_bar = []
        
        if len(east_alpha) > 0:
            stats_data.append([
                east_alpha.mean(),
                east_alpha.std(),
                east_alpha.min(),
                east_alpha.max()
            ])
            labels.append('East Wind\n(10m→30m)')
            colors_bar.append('#FF6B6B')
        
        if len(west_alpha) > 0:
            stats_data.append([
                west_alpha.mean(),
                west_alpha.std(),
                west_alpha.min(),
                west_alpha.max()
            ])
            labels.append('West Wind\n(10m→70m)')
            colors_bar.append('#4ECDC4')
        
        if stats_data:
            stats_data = np.array(stats_data).T
            stat_labels = ['Mean α', 'Std α', 'Min α', 'Max α']
            x = np.arange(len(stat_labels))
            width = 0.35
            
            for i, (data, label, color) in enumerate(zip(stats_data.T, labels, colors_bar)):
                offset = (i - 0.5) * width
                bars = ax4.bar(x + offset, data, width, label=label, color=color, alpha=0.8)
                
                # 添加数值标签
                for bar, value in zip(bars, data):
                    height = bar.get_height()
                    ax4.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                            f'{value:.3f}', ha='center', va='bottom', fontsize=8)
            
            ax4.set_xlabel('Statistics', fontweight='bold', fontsize=12)
            ax4.set_ylabel('Value', fontweight='bold', fontsize=12)
            ax4.set_title('(D) Statistical Comparison', fontweight='bold', fontsize=13)
            ax4.set_xticks(x)
            ax4.set_xticklabels(stat_labels)
            ax4.legend(fontsize=10)
            ax4.grid(True, alpha=0.3, axis='y')
            ax4.axhline(0, color='gray', linestyle='-', linewidth=1, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(os.path.join(save_path, 'adaptive_wind_shear_comparison.png'),
                   dpi=300, bbox_inches='tight', facecolor='white')
        plt.savefig(os.path.join(save_path, 'adaptive_wind_shear_comparison.pdf'),
                   dpi=300, bbox_inches='tight', facecolor='white')
        plt.show()
        
        print(f"风切变对比图已保存至: {save_path}")


class WindDirectionSHAPAnalyzer:
    """增强版风向分类SHAP分析器 - 使用自适应风切变指数"""
    
    def __init__(self, data_path, results_path):
        self.data_path = data_path
        self.results_path = results_path
        self.raw_data = None
        self.east_wind_data = None
        self.west_wind_data = None
        
        # 风切变计算器
        self.wind_shear_calculator = AdaptiveWindShearCalculator()
        
        # 为东风和西风分别创建分析器
        self.east_analyzer = None
        self.west_analyzer = None
        
    def load_and_classify_data(self):
        """加载数据并按风向分类"""
        print("="*70)
        print("第1步: 加载数据并按风向分类")
        print("="*70)
        
        # 加载原始数据
        self.raw_data = pd.read_csv(self.data_path)
        print(f"原始数据形状: {self.raw_data.shape}")
        
        # 检查wind_direction列
        wind_dir_col10 = 'obs_wind_direction_10m'
        wind_dir_col30 = 'obs_wind_direction_30m'
        wind_dir_col50 = 'obs_wind_direction_50m'
        wind_dir_col70 = 'obs_wind_direction_70m'
        
        if wind_dir_col70 not in self.raw_data.columns:
            raise ValueError(f"未找到列: {wind_dir_col70}")
        
        print(f"风向数据范围: {self.raw_data[wind_dir_col70].min():.2f}° - {self.raw_data[wind_dir_col70].max():.2f}°")
        
        # 东风区间：45° 到 135°
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
        
        return self.east_wind_data, self.west_wind_data
    
    def calculate_adaptive_wind_shear(self):
        """为东西风区间分别计算自适应风切变指数"""
        print("\n" + "="*70)
        print("第2步: 计算自适应风切变指数")
        print("="*70)
        
        # 东风: 使用10m和30m
        self.east_wind_data = self.wind_shear_calculator.calculate_for_east_wind(self.east_wind_data)
        
        # 西风: 使用10m和70m
        self.west_wind_data = self.wind_shear_calculator.calculate_for_west_wind(self.west_wind_data)
        
        # 生成对比可视化
        self.wind_shear_calculator.visualize_wind_shear_comparison(
            self.east_wind_data, self.west_wind_data, self.results_path
        )
        
        return self.east_wind_data, self.west_wind_data
    
    def run_analysis(self):
        """运行完整的风向分类SHAP分析（包含自适应风切变指数）"""
        print("="*70)
        print("自适应风切变 + 风向分类SHAP分析")
        print("="*70)
        print()
        
        # 1. 加载数据并分类
        self.load_and_classify_data()
        
        # 2. 计算自适应风切变指数
        self.calculate_adaptive_wind_shear()
        
        # 3. 为东风区间创建分析器
        print("\n" + "="*70)
        print("第3步: 分析东风区间数据 (45°-135°, 风切变: 10m→30m)")
        print("="*70)
        
        east_results_path = os.path.join(self.results_path, "east_wind")
        os.makedirs(east_results_path, exist_ok=True)
        
        # 保存东风数据
        east_data_path = os.path.join(east_results_path, "east_wind_data_with_shear.csv")
        self.east_wind_data.to_csv(east_data_path, index=False)
        print(f"东风数据已保存: {east_data_path}")
        
        self.east_analyzer = CustomSHAPVisualizer(
            data=self.east_wind_data,
            results_path=east_results_path,
            wind_type="East Wind (45°-135°)",
            shear_method="10m→30m"
        )
        east_model = self.east_analyzer.run_analysis()
        
        # 4. 为西风区间创建分析器
        print("\n" + "="*70)
        print("第4步: 分析西风区间数据 (225°-315°, 风切变: 10m→70m)")
        print("="*70)
        
        west_results_path = os.path.join(self.results_path, "west_wind")
        os.makedirs(west_results_path, exist_ok=True)
        
        # 保存西风数据
        west_data_path = os.path.join(west_results_path, "west_wind_data_with_shear.csv")
        self.west_wind_data.to_csv(west_data_path, index=False)
        print(f"西风数据已保存: {west_data_path}")
        
        self.west_analyzer = CustomSHAPVisualizer(
            data=self.west_wind_data,
            results_path=west_results_path,
            wind_type="West Wind (225°-315°)",
            shear_method="10m→70m"
        )
        west_model = self.west_analyzer.run_analysis()
        
        # 5. 生成综合对比分析
        self.create_comprehensive_comparison()
        
        print("\n" + "="*70)
        print("自适应风切变SHAP分析完成!")
        print("="*70)
        print("生成的文件:")
        print(f"  风切变对比: {self.results_path}/adaptive_wind_shear_comparison.png")
        print(f"  东风区间结果: {east_results_path}/")
        print(f"  西风区间结果: {west_results_path}/")
        print(f"  综合对比: {self.results_path}/comprehensive_comparison.png")
        
        return east_model, west_model
    
    def create_comprehensive_comparison(self):
        """创建综合对比分析"""
        print("\n生成综合对比分析...")
        
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle('Comprehensive Analysis: East (10m→30m) vs West (10m→70m)', 
                    fontsize=16, fontweight='bold')
        
        # 1. 功率分布对比
        ax1 = axes[0, 0]
        east_power = self.east_wind_data['power'].dropna()
        west_power = self.west_wind_data['power'].dropna()
        
        ax1.hist(east_power, bins=30, alpha=0.7, label='East Wind', 
                color='#FF6B6B', density=True, edgecolor='black', linewidth=0.5)
        ax1.hist(west_power, bins=30, alpha=0.7, label='West Wind', 
                color='#4ECDC4', density=True, edgecolor='black', linewidth=0.5)
        ax1.set_xlabel('Power (kW)', fontweight='bold', fontsize=12)
        ax1.set_ylabel('Density', fontweight='bold', fontsize=12)
        ax1.set_title('(A) Power Distribution', fontweight='bold', fontsize=13)
        ax1.legend(fontsize=11)
        ax1.grid(True, alpha=0.3)
        
        # 2. 风切变指数分布对比
        ax2 = axes[0, 1]
        east_alpha = self.east_wind_data['wind_shear_alpha'].dropna()
        east_alpha = east_alpha[(east_alpha > -0.5) & (east_alpha < 1)]
        west_alpha = self.west_wind_data['wind_shear_alpha'].dropna()
        west_alpha = west_alpha[(west_alpha > -0.5) & (west_alpha < 1)]
        
        ax2.hist(east_alpha, bins=30, alpha=0.7, 
                label=f'East (10m→30m)\nμ={east_alpha.mean():.3f}', 
                color='#FF6B6B', density=True, edgecolor='black', linewidth=0.5)
        ax2.hist(west_alpha, bins=30, alpha=0.7, 
                label=f'West (10m→70m)\nμ={west_alpha.mean():.3f}', 
                color='#4ECDC4', density=True, edgecolor='black', linewidth=0.5)
        ax2.axvline(0, color='gray', linestyle='--', linewidth=1, alpha=0.5)
        ax2.set_xlabel('Wind Shear Exponent (α)', fontweight='bold', fontsize=12)
        ax2.set_ylabel('Density', fontweight='bold', fontsize=12)
        ax2.set_title('(B) Wind Shear Distribution', fontweight='bold', fontsize=13)
        ax2.legend(fontsize=11)
        ax2.grid(True, alpha=0.3)
        
        # 3. 统计对比
        ax3 = axes[1, 0]
        stats_data = {
            'East Wind': [
                len(self.east_wind_data),
                self.east_wind_data['power'].mean(),
                self.east_wind_data['obs_wind_speed_10m'].mean(),
                east_alpha.mean() if len(east_alpha) > 0 else 0
            ],
            'West Wind': [
                len(self.west_wind_data),
                self.west_wind_data['power'].mean(),
                self.west_wind_data['obs_wind_speed_10m'].mean(),
                west_alpha.mean() if len(west_alpha) > 0 else 0
            ]
        }
        
        stats_labels = ['Sample Count', 'Mean Power\n(kW)', 
                       'Mean Wind Speed\n(m/s)', 'Mean α']
        x = np.arange(len(stats_labels))
        width = 0.35
        
        bars1 = ax3.bar(x - width/2, stats_data['East Wind'], width, 
                       label='East Wind', color='#FF6B6B', alpha=0.8, edgecolor='black', linewidth=0.5)
        bars2 = ax3.bar(x + width/2, stats_data['West Wind'], width, 
                       label='West Wind', color='#4ECDC4', alpha=0.8, edgecolor='black', linewidth=0.5)
        
        ax3.set_ylabel('Value', fontweight='bold', fontsize=12)
        ax3.set_title('(C) Statistical Comparison', fontweight='bold', fontsize=13)
        ax3.set_xticks(x)
        ax3.set_xticklabels(stats_labels, fontsize=10)
        ax3.legend(fontsize=11)
        ax3.grid(True, alpha=0.3, axis='y')
        
        # 添加数值标签
        for bars in [bars1, bars2]:
            for bar in bars:
                height = bar.get_height()
                ax3.text(bar.get_x() + bar.get_width()/2., height + height*0.02,
                        f'{height:.1f}', ha='center', va='bottom', fontsize=9)
        
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
            
            bars1 = ax4.bar(x - width/2, east_metrics, width, label='East Wind', 
                           color='#FF6B6B', alpha=0.8, edgecolor='black', linewidth=0.5)
            bars2 = ax4.bar(x + width/2, west_metrics, width, label='West Wind', 
                           color='#4ECDC4', alpha=0.8, edgecolor='black', linewidth=0.5)
            
            ax4.set_ylabel('Value', fontweight='bold', fontsize=12)
            ax4.set_title('(D) Model Performance (with Wind Shear α)', 
                         fontweight='bold', fontsize=13)
            ax4.set_xticks(x)
            ax4.set_xticklabels(metrics, fontsize=11)
            ax4.legend(fontsize=11)
            ax4.grid(True, alpha=0.3, axis='y')
            
            # 添加数值标签
            for bars in [bars1, bars2]:
                for bar in bars:
                    height = bar.get_height()
                    ax4.text(bar.get_x() + bar.get_width()/2., height + height*0.02,
                            f'{height:.3f}', ha='center', va='bottom', fontsize=9)
        else:
            ax4.text(0.5, 0.5, 'Model Performance\n(Run analysis first)', 
                    ha='center', va='center', transform=ax4.transAxes, fontsize=12)
            ax4.set_title('(D) Model Performance', fontweight='bold', fontsize=13)
        
        plt.tight_layout()
        
        # 保存图形
        comparison_path = os.path.join(self.results_path, 'comprehensive_comparison.png')
        plt.savefig(comparison_path, dpi=300, bbox_inches='tight', facecolor='white')
        plt.savefig(comparison_path.replace('.png', '.pdf'), 
                   dpi=300, bbox_inches='tight', facecolor='white')
        plt.show()


class CustomSHAPVisualizer:
    """SHAP分析器 - 包含自适应风切变指数"""
    
    def __init__(self, data, results_path, wind_type="", shear_method=""):
        self.data = data
        self.results_path = results_path
        self.wind_type = wind_type
        self.shear_method = shear_method
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
        print(f"预处理{self.wind_type}数据 (风切变方法: {self.shear_method})...")
        
        print(f"输入数据形状: {self.data.shape}")
        
        # 选择观测数据列 + 风切变指数
        obs_columns = [col for col in self.data.columns if col.startswith('obs_')]
        obs_columns += ['datetime', 'power', 'wind_shear_alpha']
        
        # 移除密度和湿度变量
        obs_columns = [col for col in obs_columns if 'density' not in col and 'humidity' not in col]
        
        # 确保列存在
        obs_columns = [col for col in obs_columns if col in self.data.columns]
        
        self.data = self.data[obs_columns].copy()
        print(f"选择列后的数据形状: {self.data.shape}")
        
        # 清理数据
        self.data = self.data.dropna()
        self.data = self.data[self.data['power'] >= 0]
        
        # 过滤风切变指数：保留物理合理的值
        if 'wind_shear_alpha' in self.data.columns:
            before_filter = len(self.data)
            self.data = self.data[
                (self.data['wind_shear_alpha'] > -0.5) & 
                (self.data['wind_shear_alpha'] < 1.0)
            ]
            after_filter = len(self.data)
            print(f"风切变过滤: {before_filter} → {after_filter} ({after_filter/before_filter*100:.1f}%)")
        
        print(f"最终数据形状: {self.data.shape}")
        if 'wind_shear_alpha' in self.data.columns:
            alpha_stats = self.data['wind_shear_alpha']
            print(f"风切变指数范围: {alpha_stats.min():.4f} - {alpha_stats.max():.4f} (mean={alpha_stats.mean():.4f})")
        
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
                       if col not in ['datetime', 'power']]
        
        print(f"使用{len(feature_cols)}个特征")
        
        # 创建特征矩阵
        self.features = self.data[feature_cols].values
        self.target = self.data['power'].values
        self.feature_names = feature_cols
        
        print(f"特征矩阵形状: {self.features.shape}")
        
        # 检查风切变指数是否在特征中
        if 'wind_shear_alpha' in feature_cols:
            print(f"✓ 风切变指数已成功添加为特征 (方法: {self.shear_method})")
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
        
        print(f"\n{self.wind_type} 模型性能:")
        print(f"  测试集 R²: {self.test_r2:.4f}")
        print(f"  测试集 RMSE: {self.test_rmse:.3f} kW")
        print(f"  测试集 MAE: {self.test_mae:.3f} kW")
        
        # 输出风切变指数的特征重要性
        if 'wind_shear_alpha' in self.feature_names:
            wind_shear_idx = self.feature_names.index('wind_shear_alpha')
            feature_importance = self.model.feature_importances_
            wind_shear_importance = feature_importance[wind_shear_idx]
            rank = np.argsort(feature_importance)[::-1].tolist().index(wind_shear_idx) + 1
            
            print(f"\n风切变指数 ({self.shear_method}) 特征重要性:")
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
            
            print(f"\n风切变指数 ({self.shear_method}) SHAP重要性:")
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
        title = f'SHAP Analysis - {self.wind_type}\nWind Shear Method: {self.shear_method}'
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
                display_names.append(f'Wind Shear α ★\n({self.shear_method})')
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
        bars = ax.barh(range(len(top_indices)), top_importance, color=colors, alpha=0.8,
                      edgecolor='black', linewidth=0.5)
        
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
        
        display_names = []
        for i in top_indices:
            name = self.feature_names[i].replace('obs_', '').replace('_', ' ').title()
            if 'wind_shear_alpha' in self.feature_names[i]:
                name = f'Wind Shear α ★'
            display_names.append(name)
        
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
                name = f'Wind Shear α ★\n({self.shear_method})'
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
            
            # 特征名称
            if '★' in name:
                ax.text(feature_label_x, y, name, ha='left', va='center',
                       fontsize=10, fontweight='bold', color='#FF6B35')
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
        print(f"=== {self.wind_type} SHAP分析 (风切变: {self.shear_method}) ===")
        
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
    RESULTS_PATH = "/Users/xiaxin/work/WindForecast_Project/03_Results/figures/3-10/adaptive_wind_shear_shap"
    
    # 创建结果目录
    os.makedirs(RESULTS_PATH, exist_ok=True)
    
    print("="*70)
    print("自适应风切变 + 风向分类SHAP分析")
    print("="*70)
    print("研究设计:")
    print("  东风区间 (45°-135°): 风切变计算使用 10m → 30m (近地层)")
    print("  西风区间 (225°-315°): 风切变计算使用 10m → 70m (全高度)")
    print("研究目标: 验证风切变指数对风电功率预测的贡献")
    print("="*70)
    print()
    
    # 运行分析
    analyzer = WindDirectionSHAPAnalyzer(DATA_PATH, RESULTS_PATH)
    east_model, west_model = analyzer.run_analysis()
    
    print("\n" + "="*70)
    print("分析完成!")
    print("="*70)