#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Monthly Wind Speed Time Series Analysis
月度风速时间序列分析：为每个高度、每个月生成obs, ec, gfs的时间序列图
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import warnings
from datetime import datetime
warnings.filterwarnings('ignore')

# 设置matplotlib参数
plt.rcParams['font.sans-serif'] = ['Arial', 'DejaVu Sans', 'Liberation Sans']
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['figure.dpi'] = 300

class MonthlyWindSeriesAnalyzer:
    """月度风速时间序列分析器"""
    
    def __init__(self, data_path, results_dir):
        """初始化分析器"""
        self.data_path = data_path
        self.results_dir = results_dir
        
        # 创建主结果目录
        self.sequence_dir = os.path.join(results_dir, 'monthly_sequences')
        os.makedirs(self.sequence_dir, exist_ok=True)
        
        # 颜色方案
        self.colors = {
            'obs': '#2C3E50',      # 深蓝灰 - 观测值
            'ec': '#E74C3C',       # 红色 - EC模型
            'gfs': '#3498DB'       # 蓝色 - GFS模型
        }
        
        # 线型样式
        self.styles = {
            'obs': {'linestyle': '-', 'linewidth': 2.5, 'alpha': 0.9},
            'ec': {'linestyle': '-', 'linewidth': 2, 'alpha': 0.8},
            'gfs': {'linestyle': '-', 'linewidth': 2, 'alpha': 0.8}
        }
        
        # 月份信息
        self.month_names = {
            1: 'January', 2: 'February', 3: 'March', 4: 'April',
            5: 'May', 6: 'June', 7: 'July', 8: 'August',
            9: 'September', 10: 'October', 11: 'November', 12: 'December'
        }
        
        # 高度列表
        self.heights = ['10m', '30m', '50m', '70m']
        
        self.load_data()
    
    def load_data(self):
        """加载和预处理数据"""
        print("📊 Loading wind speed data for monthly sequence analysis...")
        
        # 读取数据
        self.data = pd.read_csv(self.data_path)
        self.data['datetime'] = pd.to_datetime(self.data['datetime'])
        
        # 添加时间信息
        self.data['year'] = self.data['datetime'].dt.year
        self.data['month'] = self.data['datetime'].dt.month
        self.data['day'] = self.data['datetime'].dt.day
        self.data['hour'] = self.data['datetime'].dt.hour
        
        # 按时间排序
        self.data = self.data.sort_values('datetime').reset_index(drop=True)
        
        print(f"✅ Loaded {len(self.data):,} records")
        print(f"   Date range: {self.data['datetime'].min().date()} to {self.data['datetime'].max().date()}")
        
        # 检查可用的风速变量
        self.available_heights = []
        for height in self.heights:
            obs_col = f'obs_wind_speed_{height}'
            ec_col = f'ec_wind_speed_{height}'
            gfs_col = f'gfs_wind_speed_{height}'
            
            if all(col in self.data.columns for col in [obs_col, ec_col, gfs_col]):
                self.available_heights.append(height)
                print(f"   ✓ Found variables for {height}")
            else:
                print(f"   ✗ Missing variables for {height}")
        
        if not self.available_heights:
            raise ValueError("No complete wind speed variables found!")
        
        # 检查每个月的数据量
        monthly_counts = self.data.groupby(['year', 'month']).size().reset_index(name='count')
        print(f"\n   Monthly data distribution:")
        for _, row in monthly_counts.iterrows():
            year, month, count = row['year'], row['month'], row['count']
            print(f"     {year}-{self.month_names[month]:>9}: {count:,} records")
    
    def get_monthly_data(self):
        """获取分月数据"""
        print("\n📅 Organizing data by months...")
        
        self.monthly_data = {}
        
        # 按年月分组
        for (year, month), group in self.data.groupby(['year', 'month']):
            month_key = f"{year}-{month:02d}"
            month_name = f"{self.month_names[month]} {year}"
            
            self.monthly_data[month_key] = {
                'year': year,
                'month': month,
                'month_name': month_name,
                'data': group.sort_values('datetime').reset_index(drop=True),
                'count': len(group)
            }
            
            print(f"   ✓ {month_name}: {len(group):,} records")
        
        print(f"✅ Organized data into {len(self.monthly_data)} monthly datasets")
    
    def calculate_rmse(self, actual, predicted):
        """计算RMSE"""
        return np.sqrt(np.mean((actual - predicted) ** 2))
    
    def create_height_directories(self):
        """为每个高度创建目录"""
        self.height_dirs = {}
        for height in self.available_heights:
            height_dir = os.path.join(self.sequence_dir, f'wind_speed_{height}')
            os.makedirs(height_dir, exist_ok=True)
            self.height_dirs[height] = height_dir
            print(f"   📁 Created directory for {height}: {height_dir}")
    
    def create_monthly_series_plots(self):
        """为每个高度、每个月创建时间序列图"""
        print("\n🎨 Creating monthly time series plots...")
        
        # 创建高度目录
        self.create_height_directories()
        
        # 存储所有RMSE结果
        self.rmse_results = []
        
        for height in self.available_heights:
            print(f"\n🔧 Processing {height}...")
            
            for month_key in sorted(self.monthly_data.keys()):
                self._create_single_month_height_plot(height, month_key)
        
        print(f"\n✅ All monthly plots created")
    
    def _create_single_month_height_plot(self, height, month_key):
        """创建单个月份、单个高度的时间序列图"""
        month_info = self.monthly_data[month_key]
        month_data = month_info['data']
        month_name = month_info['month_name']
        
        # 获取变量列名
        obs_col = f'obs_wind_speed_{height}'
        ec_col = f'ec_wind_speed_{height}'
        gfs_col = f'gfs_wind_speed_{height}'
        
        # 检查数据完整性
        valid_mask = (
            ~month_data[obs_col].isna() & 
            ~month_data[ec_col].isna() & 
            ~month_data[gfs_col].isna()
        )
        
        if valid_mask.sum() == 0:
            print(f"   ⚠️  Skipping {month_name} {height} - no valid data")
            return
        
        valid_data = month_data[valid_mask].copy()
        
        # 计算RMSE
        obs_values = valid_data[obs_col].values
        ec_values = valid_data[ec_col].values
        gfs_values = valid_data[gfs_col].values
        
        rmse_obs_ec = self.calculate_rmse(obs_values, ec_values)
        rmse_obs_gfs = self.calculate_rmse(obs_values, gfs_values)
        
        # 计算EC相对于GFS的变化
        if rmse_obs_gfs != 0:
            rmse_change = ((rmse_obs_gfs - rmse_obs_ec) / rmse_obs_gfs) * 100
            change_type = "improvement" if rmse_change > 0 else "degradation"
        else:
            rmse_change = 0
            change_type = "no change"
        
        # 保存RMSE结果
        self.rmse_results.append({
            'height': height,
            'year_month': month_key,
            'month_name': month_name,
            'valid_count': len(valid_data),
            'rmse_obs_ec': rmse_obs_ec,
            'rmse_obs_gfs': rmse_obs_gfs,
            'rmse_change_percent': rmse_change,
            'change_type': change_type
        })
        
        # 打印RMSE信息
        print(f"   📊 {month_name} {height} RMSE:")
        print(f"      obs vs ec:  {rmse_obs_ec:.3f} m/s")
        print(f"      obs vs gfs: {rmse_obs_gfs:.3f} m/s")
        print(f"      EC vs GFS:  {rmse_change:+.1f}% ({change_type})")
        
        # 创建图表
        fig, ax = plt.subplots(1, 1, figsize=(16, 10))
        
        # 绘制时间序列
        datetime_values = valid_data['datetime']
        
        # 观测值
        ax.plot(datetime_values, obs_values,
               color=self.colors['obs'],
               linestyle=self.styles['obs']['linestyle'],
               linewidth=self.styles['obs']['linewidth'],
               alpha=self.styles['obs']['alpha'],
               label='Observed')
        
        # EC模型
        ax.plot(datetime_values, ec_values,
               color=self.colors['ec'],
               linestyle=self.styles['ec']['linestyle'],
               linewidth=self.styles['ec']['linewidth'],
               alpha=self.styles['ec']['alpha'],
               label='EC Model')
        
        # GFS模型
        ax.plot(datetime_values, gfs_values,
               color=self.colors['gfs'],
               linestyle=self.styles['gfs']['linestyle'],
               linewidth=self.styles['gfs']['linewidth'],
               alpha=self.styles['gfs']['alpha'],
               label='GFS Model')
        
        # 设置图表属性
        ax.set_xlabel('Date', fontsize=14, fontweight='bold')
        ax.set_ylabel('Wind Speed (m/s)', fontsize=14, fontweight='bold')
        
        # 设置标题
        title = f'{month_name} Wind Speed {height} Time Series'
        subtitle = f'Valid samples: {len(valid_data):,} | EC vs GFS: {rmse_change:+.1f}% RMSE {change_type}'
        ax.set_title(f'{title}\n{subtitle}', fontsize=16, fontweight='bold', pad=20)
        
        # 设置x轴格式
        ax.tick_params(axis='x', labelsize=12, rotation=45)
        ax.tick_params(axis='y', labelsize=12)
        
        # 添加网格
        ax.grid(True, alpha=0.3, linestyle='--', linewidth=0.8)
        ax.set_axisbelow(True)
        
        # 设置图例
        legend = ax.legend(loc='upper right', 
                          fontsize=12,
                          framealpha=0.9,
                          edgecolor='black',
                          title='Data Sources',
                          title_fontsize=12)
        legend.get_title().set_fontweight('bold')
        
        # 添加背景色
        ax.set_facecolor('#FAFAFA')
        
        # 添加统计信息文本框
        stats_text = self._generate_stats_text(valid_data, obs_col, ec_col, gfs_col, 
                                              rmse_obs_ec, rmse_obs_gfs, rmse_change)
        ax.text(0.02, 0.98, stats_text,
               transform=ax.transAxes,
               fontsize=10,
               verticalalignment='top',
               bbox=dict(boxstyle='round,pad=0.5', 
                        facecolor='white', 
                        alpha=0.9,
                        edgecolor='gray'))
        
        # 添加季节标识
        season = self._get_season(month_info['month'])
        season_colors = {
            'Spring': '#90EE90',
            'Summer': '#FFB6C1', 
            'Autumn': '#DEB887',
            'Winter': '#B0C4DE'
        }
        
        ax.text(0.98, 0.98, f'{season}',
               transform=ax.transAxes,
               fontsize=14,
               fontweight='bold',
               horizontalalignment='right',
               verticalalignment='top',
               bbox=dict(boxstyle='round,pad=0.3',
                        facecolor=season_colors[season],
                        alpha=0.8,
                        edgecolor='black'))
        
        # 保存图表
        plt.tight_layout()
        filename = f'{month_key}_{self.month_names[month_info["month"]].lower()}_wind_{height}.png'
        output_path = os.path.join(self.height_dirs[height], filename)
        plt.savefig(output_path, dpi=300, bbox_inches='tight', 
                   facecolor='white', edgecolor='none')
        plt.close()
        
        print(f"   ✓ Created: {filename}")
    
    def _generate_stats_text(self, data, obs_col, ec_col, gfs_col, 
                           rmse_obs_ec, rmse_obs_gfs, rmse_change):
        """生成统计信息文本"""
        stats_text = f"Statistics:\n"
        
        # 基本统计
        for var_name, col in [('Observed', obs_col), ('EC Model', ec_col), ('GFS Model', gfs_col)]:
            values = data[col]
            stats_text += f"• {var_name}:\n"
            stats_text += f"  Mean: {values.mean():.2f} m/s\n"
            stats_text += f"  Std: {values.std():.2f} m/s\n"
            stats_text += f"  Range: {values.min():.2f}-{values.max():.2f} m/s\n"
        
        # RMSE分析
        stats_text += f"\nRMSE Analysis:\n"
        stats_text += f"• obs vs EC: {rmse_obs_ec:.3f} m/s\n"
        stats_text += f"• obs vs GFS: {rmse_obs_gfs:.3f} m/s\n"
        stats_text += f"• EC vs GFS: {rmse_change:+.1f}%\n"
        
        return stats_text.strip()
    
    def _get_season(self, month):
        """根据月份获取季节"""
        if month in [12, 1, 2]:
            return 'Winter'
        elif month in [3, 4, 5]:
            return 'Spring'
        elif month in [6, 7, 8]:
            return 'Summer'
        else:
            return 'Autumn'
    
    def save_rmse_analysis(self):
        """保存RMSE分析结果"""
        print("\n💾 Saving RMSE analysis results...")
        
        # 转换为DataFrame
        rmse_df = pd.DataFrame(self.rmse_results)
        
        # 保存详细结果
        rmse_file = os.path.join(self.sequence_dir, 'monthly_rmse_analysis.csv')
        rmse_df.to_csv(rmse_file, index=False)
        
        # 生成汇总统计
        summary_stats = []
        
        for height in self.available_heights:
            height_data = rmse_df[rmse_df['height'] == height]
            
            summary_stats.append({
                'height': height,
                'total_months': len(height_data),
                'avg_rmse_obs_ec': height_data['rmse_obs_ec'].mean(),
                'avg_rmse_obs_gfs': height_data['rmse_obs_gfs'].mean(),
                'avg_rmse_change': height_data['rmse_change_percent'].mean(),
                'months_ec_better': (height_data['rmse_change_percent'] > 0).sum(),
                'months_gfs_better': (height_data['rmse_change_percent'] < 0).sum(),
                'best_ec_month': height_data.loc[height_data['rmse_change_percent'].idxmax(), 'month_name'],
                'best_ec_improvement': height_data['rmse_change_percent'].max(),
                'worst_ec_month': height_data.loc[height_data['rmse_change_percent'].idxmin(), 'month_name'],
                'worst_ec_change': height_data['rmse_change_percent'].min()
            })
        
        summary_df = pd.DataFrame(summary_stats)
        summary_file = os.path.join(self.sequence_dir, 'rmse_summary_by_height.csv')
        summary_df.to_csv(summary_file, index=False)
        
        print(f"   ✓ Saved: monthly_rmse_analysis.csv")
        print(f"   ✓ Saved: rmse_summary_by_height.csv")
        print(f"   📁 Files saved to: {self.sequence_dir}")
    
    def generate_analysis_report(self):
        """生成分析报告"""
        print("\n" + "="*80)
        print("📊 MONTHLY WIND SPEED TIME SERIES ANALYSIS REPORT")
        print("="*80)
        
        # 数据概览
        print(f"\n🗓️  Data Overview:")
        print(f"   Total Records: {len(self.data):,}")
        print(f"   Date Range: {self.data['datetime'].min().date()} to {self.data['datetime'].max().date()}")
        print(f"   Available Heights: {', '.join(self.available_heights)}")
        print(f"   Monthly Datasets: {len(self.monthly_data)}")
        
        # RMSE分析汇总
        if hasattr(self, 'rmse_results') and self.rmse_results:
            rmse_df = pd.DataFrame(self.rmse_results)
            
            print(f"\n📊 RMSE Analysis Summary:")
            
            for height in self.available_heights:
                height_data = rmse_df[rmse_df['height'] == height]
                
                print(f"\n   {height}:")
                print(f"     Average RMSE (obs vs EC):  {height_data['rmse_obs_ec'].mean():.3f} m/s")
                print(f"     Average RMSE (obs vs GFS): {height_data['rmse_obs_gfs'].mean():.3f} m/s")
                print(f"     Average EC vs GFS change:  {height_data['rmse_change_percent'].mean():+.1f}%")
                
                ec_better = (height_data['rmse_change_percent'] > 0).sum()
                total_months = len(height_data)
                print(f"     Months EC better than GFS: {ec_better}/{total_months} ({ec_better/total_months*100:.1f}%)")
                
                if not height_data.empty:
                    best_month = height_data.loc[height_data['rmse_change_percent'].idxmax()]
                    worst_month = height_data.loc[height_data['rmse_change_percent'].idxmin()]
                    
                    print(f"     Best EC performance: {best_month['month_name']} ({best_month['rmse_change_percent']:+.1f}%)")
                    print(f"     Worst EC performance: {worst_month['month_name']} ({worst_month['rmse_change_percent']:+.1f}%)")
        
        print(f"\n📁 Output Structure:")
        print(f"   Main directory: {self.sequence_dir}")
        for height in self.available_heights:
            print(f"   {height} plots: {self.height_dirs[height]}")
        
        print(f"\n🎉 Monthly wind speed time series analysis completed!")
        print("="*80)
    
    def run_analysis(self):
        """运行完整的月度风速时间序列分析"""
        print("🌪️  Starting Monthly Wind Speed Time Series Analysis...")
        print("="*70)
        
        # 1. 组织月度数据
        self.get_monthly_data()
        
        # 2. 创建月度时间序列图
        self.create_monthly_series_plots()
        
        # 3. 保存RMSE分析
        self.save_rmse_analysis()
        
        # 4. 生成报告
        self.generate_analysis_report()


def main():
    """主函数"""
    # 配置路径
    DATA_PATH = '/Users/xiaxin/work/WindForecast_Project/01_Data/processed/imputed_data/changma_imputed_complete.csv'
    RESULTS_DIR = '/Users/xiaxin/work/WindForecast_Project/03_Results/'
    
    # 检查文件是否存在
    if not os.path.exists(DATA_PATH):
        print(f"❌ Error: Data file not found at {DATA_PATH}")
        return
    
    # 创建分析器并运行分析
    analyzer = MonthlyWindSeriesAnalyzer(DATA_PATH, RESULTS_DIR)
    analyzer.run_analysis()


if __name__ == "__main__":
    main()