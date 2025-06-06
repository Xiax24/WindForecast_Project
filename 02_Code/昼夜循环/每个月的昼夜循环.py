#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Wind Forecast Diurnal Bar Chart Analysis
风速预报日变化柱状图分析 - obs vs ec vs gfs
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# 设置matplotlib参数
plt.rcParams['font.sans-serif'] = ['Arial', 'DejaVu Sans', 'Liberation Sans']
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['figure.dpi'] = 300

class WindForecastDiurnalBarAnalyzer:
    """风速预报日变化柱状图分析器"""
    
    def __init__(self, data_path, output_path):
        """初始化分析器"""
        self.data_path = data_path
        self.output_path = output_path
        self.colors = {
            'obs': '#2E8B57',        # 深海绿
            'ec': '#4169E1',         # 皇家蓝
            'gfs': '#DC143C',        # 深红色
            'ec_bias': '#FF6347',    # 番茄红
            'gfs_bias': '#FF8C00'    # 深橙色
        }
        self.load_data()
        self.get_all_months()
    
    def load_data(self):
        """加载和预处理数据"""
        print("📊 Loading data for wind forecast diurnal bar analysis...")
        
        # 读取数据
        self.data = pd.read_csv(self.data_path)
        self.data['datetime'] = pd.to_datetime(self.data['datetime'])
        
        # 添加时间信息
        self.data['year'] = self.data['datetime'].dt.year
        self.data['month'] = self.data['datetime'].dt.month
        self.data['day'] = self.data['datetime'].dt.day
        self.data['hour'] = self.data['datetime'].dt.hour
        self.data['month_name'] = self.data['datetime'].dt.strftime('%Y-%m')
        
        # 定义高度层
        self.heights = ['10m', '30m', '50m', '70m']
        
        # 计算bias
        for height in self.heights:
            obs_col = f'obs_wind_speed_{height}'
            ec_col = f'ec_wind_speed_{height}'
            gfs_col = f'gfs_wind_speed_{height}'
            
            if all(col in self.data.columns for col in [obs_col, ec_col, gfs_col]):
                self.data[f'ec_bias_{height}'] = self.data[ec_col] - self.data[obs_col]
                self.data[f'gfs_bias_{height}'] = self.data[gfs_col] - self.data[obs_col]
        
        print(f"✅ Loaded {len(self.data):,} records from {self.data['datetime'].min().date()} to {self.data['datetime'].max().date()}")
        print(f"✅ Available heights: {self.heights}")
    
    def get_all_months(self):
        """获取所有月份的数据"""
        # 获取所有年月组合
        year_months = self.data.groupby(['year', 'month']).size().reset_index(name='count')
        year_months = year_months[year_months['count'] > 100]  # 至少要有100条记录
        
        print(f"📅 Available months with sufficient data:")
        
        self.months_info = []
        month_names = {
            1: '一月', 2: '二月', 3: '三月', 4: '四月', 5: '五月', 6: '六月',
            7: '七月', 8: '八月', 9: '九月', 10: '十月', 11: '十一月', 12: '十二月'
        }
        
        for _, row in year_months.iterrows():
            year = int(row['year'])
            month = int(row['month'])
            count = int(row['count'])
            
            month_data = self.data[
                (self.data['year'] == year) & 
                (self.data['month'] == month)
            ]
            
            if len(month_data) > 0:
                self.months_info.append({
                    'year': year,
                    'month': month,
                    'month_name': f'{year}-{month:02d}',
                    'display_name': f'{year}-{month}',
                    'chinese_name': month_names[month],
                    'data_count': len(month_data),
                    'sort_key': year * 100 + month
                })
                
                print(f"   • {year}年{month}月: {len(month_data):,} records")
        
        # 按时间顺序排序
        self.months_info.sort(key=lambda x: x['sort_key'])
        
        print(f"✅ Total months to analyze: {len(self.months_info)}")
        print(f"✅ Total records: {len(self.data):,}")
    
    def calculate_monthly_diurnal_stats(self):
        """计算每月每高度的日变化统计"""
        print("🕐 Calculating monthly diurnal statistics for all heights...")
        
        self.monthly_stats = {}
        
        for month_info in self.months_info:
            year = month_info['year']
            month = month_info['month']
            month_name = month_info['month_name']
            
            # 筛选当月数据
            month_data = self.data[
                (self.data['year'] == year) &
                (self.data['month'] == month)
            ]
            
            self.monthly_stats[month_name] = {
                'info': month_info,
                'heights_stats': {}
            }
            
            # 为每个高度层计算统计
            for height in self.heights:
                obs_col = f'obs_wind_speed_{height}'
                ec_col = f'ec_wind_speed_{height}'
                gfs_col = f'gfs_wind_speed_{height}'
                ec_bias_col = f'ec_bias_{height}'
                gfs_bias_col = f'gfs_bias_{height}'
                
                # 检查列是否存在
                if not all(col in month_data.columns for col in [obs_col, ec_col, gfs_col]):
                    continue
                
                hourly_stats = {}
                for hour in range(24):
                    hour_data = month_data[month_data['hour'] == hour]
                    
                    if len(hour_data) > 0:
                        hourly_stats[hour] = {
                            'hour': hour,
                            'count': len(hour_data),
                            # 风速数据
                            'obs_mean': float(np.mean(hour_data[obs_col])),
                            'obs_std': float(np.std(hour_data[obs_col])),
                            'ec_mean': float(np.mean(hour_data[ec_col])),
                            'ec_std': float(np.std(hour_data[ec_col])),
                            'gfs_mean': float(np.mean(hour_data[gfs_col])),
                            'gfs_std': float(np.std(hour_data[gfs_col])),
                            # Bias数据
                            'ec_bias_mean': float(np.mean(hour_data[ec_bias_col])),
                            'ec_bias_std': float(np.std(hour_data[ec_bias_col])),
                            'gfs_bias_mean': float(np.mean(hour_data[gfs_bias_col])),
                            'gfs_bias_std': float(np.std(hour_data[gfs_bias_col]))
                        }
                
                self.monthly_stats[month_name]['heights_stats'][height] = hourly_stats
        
        print("✅ Monthly diurnal statistics calculation completed")
    
    def create_monthly_plots(self):
        """创建每月每高度的柱状图"""
        print("🎨 Creating monthly diurnal bar plots...")
        
        # 创建输出文件夹
        plots_path = os.path.join(self.output_path, 'wind_forecast_diurnal_plots')
        os.makedirs(plots_path, exist_ok=True)
        
        # 为每个月每个高度创建图
        for month_name, month_data in self.monthly_stats.items():
            month_info = month_data['info']
            
            for height in self.heights:
                if height in month_data['heights_stats']:
                    hourly_stats = month_data['heights_stats'][height]
                    
                    # 创建风速对比图
                    self._create_wind_speed_bar_plot(month_info, height, hourly_stats, plots_path)
                    
                    # 创建bias对比图
                    self._create_bias_bar_plot(month_info, height, hourly_stats, plots_path)
        
        # 创建综合对比图
        self._create_comprehensive_comparison(plots_path)
        
        # 创建高度层对比图
        self._create_height_comparison(plots_path)
        
        print(f"✅ All plots saved to {plots_path}")
    
    def _create_wind_speed_bar_plot(self, month_info, height, hourly_stats, output_path):
        """创建风速对比柱状图"""
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle(f'{month_info["display_name"]} {height} Wind Speed Diurnal Variation\n(Sample Count: {month_info["data_count"]:,})', 
                    fontsize=16, fontweight='bold')
        
        hours = sorted(hourly_stats.keys())
        
        # 提取数据
        obs_means = [hourly_stats[h]['obs_mean'] for h in hours]
        obs_stds = [hourly_stats[h]['obs_std'] for h in hours]
        ec_means = [hourly_stats[h]['ec_mean'] for h in hours]
        ec_stds = [hourly_stats[h]['ec_std'] for h in hours]
        gfs_means = [hourly_stats[h]['gfs_mean'] for h in hours]
        gfs_stds = [hourly_stats[h]['gfs_std'] for h in hours]
        
        # 主图：三组并排柱状图
        ax1 = axes[0, 0]
        x = np.array(hours)
        width = 0.25
        
        bars1 = ax1.bar(x - width, obs_means, width, 
                       label='Observed', color=self.colors['obs'], 
                       alpha=0.8, edgecolor='black', linewidth=0.5)
        bars2 = ax1.bar(x, ec_means, width,
                       label='EC Forecast', color=self.colors['ec'], 
                       alpha=0.8, edgecolor='black', linewidth=0.5)
        bars3 = ax1.bar(x + width, gfs_means, width,
                       label='GFS Forecast', color=self.colors['gfs'], 
                       alpha=0.8, edgecolor='black', linewidth=0.5)
        
        # 添加误差条
        ax1.errorbar(x - width, obs_means, yerr=obs_stds, 
                    fmt='none', color='black', alpha=0.6, capsize=2)
        ax1.errorbar(x, ec_means, yerr=ec_stds, 
                    fmt='none', color='black', alpha=0.6, capsize=2)
        ax1.errorbar(x + width, gfs_means, yerr=gfs_stds, 
                    fmt='none', color='black', alpha=0.6, capsize=2)
        
        ax1.set_xlabel('Hour of Day')
        ax1.set_ylabel('Wind Speed (m/s)')
        ax1.set_title('Hourly Wind Speed Comparison (Mean ± Std)')
        ax1.legend()
        ax1.grid(True, alpha=0.3, axis='y')
        ax1.set_xticks(hours[::2])
        
        # 仅观测数据
        ax2 = axes[0, 1]
        bars_obs = ax2.bar(hours, obs_means, color=self.colors['obs'], 
                          alpha=0.8, edgecolor='black', linewidth=0.5)
        ax2.errorbar(hours, obs_means, yerr=obs_stds, 
                    fmt='none', color='black', alpha=0.6, capsize=2)
        
        ax2.set_xlabel('Hour of Day')
        ax2.set_ylabel('Wind Speed (m/s)')
        ax2.set_title('Observed Wind Speed Diurnal Pattern')
        ax2.grid(True, alpha=0.3, axis='y')
        ax2.set_xticks(hours[::2])
        
        # 添加数值标签
        for i, (hour, speed) in enumerate(zip(hours, obs_means)):
            if i % 3 == 0:  # 每3小时显示一个标签
                ax2.text(hour, speed + 0.1, f'{speed:.1f}', 
                        ha='center', va='bottom', fontsize=8, fontweight='bold')
        
        # EC预报
        ax3 = axes[1, 0]
        bars_ec = ax3.bar(hours, ec_means, color=self.colors['ec'], 
                         alpha=0.8, edgecolor='black', linewidth=0.5)
        ax3.errorbar(hours, ec_means, yerr=ec_stds, 
                    fmt='none', color='black', alpha=0.6, capsize=2)
        
        ax3.set_xlabel('Hour of Day')
        ax3.set_ylabel('Wind Speed (m/s)')
        ax3.set_title('EC Forecast Diurnal Pattern')
        ax3.grid(True, alpha=0.3, axis='y')
        ax3.set_xticks(hours[::2])
        
        # GFS预报
        ax4 = axes[1, 1]
        bars_gfs = ax4.bar(hours, gfs_means, color=self.colors['gfs'], 
                          alpha=0.8, edgecolor='black', linewidth=0.5)
        ax4.errorbar(hours, gfs_means, yerr=gfs_stds, 
                    fmt='none', color='black', alpha=0.6, capsize=2)
        
        ax4.set_xlabel('Hour of Day')
        ax4.set_ylabel('Wind Speed (m/s)')
        ax4.set_title('GFS Forecast Diurnal Pattern')
        ax4.grid(True, alpha=0.3, axis='y')
        ax4.set_xticks(hours[::2])
        
        plt.tight_layout()
        
        # 保存图片
        filename = f"{month_info['month_name']}_{height}_wind_speed_diurnal.png"
        plt.savefig(os.path.join(output_path, filename), dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"  ✅ Saved: {filename}")
    
    def _create_bias_bar_plot(self, month_info, height, hourly_stats, output_path):
        """创建bias对比柱状图"""
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle(f'{month_info["display_name"]} {height} Bias Diurnal Variation\n(Sample Count: {month_info["data_count"]:,})', 
                    fontsize=16, fontweight='bold')
        
        hours = sorted(hourly_stats.keys())
        
        # 提取数据
        ec_bias_means = [hourly_stats[h]['ec_bias_mean'] for h in hours]
        ec_bias_stds = [hourly_stats[h]['ec_bias_std'] for h in hours]
        gfs_bias_means = [hourly_stats[h]['gfs_bias_mean'] for h in hours]
        gfs_bias_stds = [hourly_stats[h]['gfs_bias_std'] for h in hours]
        
        # 主图：并排柱状图
        ax1 = axes[0, 0]
        x = np.array(hours)
        width = 0.35
        
        bars1 = ax1.bar(x - width/2, ec_bias_means, width, 
                       label='EC Bias', color=self.colors['ec_bias'], 
                       alpha=0.8, edgecolor='black', linewidth=0.5)
        bars2 = ax1.bar(x + width/2, gfs_bias_means, width,
                       label='GFS Bias', color=self.colors['gfs_bias'], 
                       alpha=0.8, edgecolor='black', linewidth=0.5)
        
        # 添加误差条
        ax1.errorbar(x - width/2, ec_bias_means, yerr=ec_bias_stds, 
                    fmt='none', color='black', alpha=0.6, capsize=2)
        ax1.errorbar(x + width/2, gfs_bias_means, yerr=gfs_bias_stds, 
                    fmt='none', color='black', alpha=0.6, capsize=2)
        
        ax1.axhline(y=0, color='black', linestyle='-', alpha=0.3)
        ax1.set_xlabel('Hour of Day')
        ax1.set_ylabel('Bias (m/s)')
        ax1.set_title('Hourly Bias Comparison (Mean ± Std)')
        ax1.legend()
        ax1.grid(True, alpha=0.3, axis='y')
        ax1.set_xticks(hours[::2])
        
        # 仅EC bias
        ax2 = axes[0, 1]
        bars_ec = ax2.bar(hours, ec_bias_means, color=self.colors['ec_bias'], 
                         alpha=0.8, edgecolor='black', linewidth=0.5)
        ax2.errorbar(hours, ec_bias_means, yerr=ec_bias_stds, 
                    fmt='none', color='black', alpha=0.6, capsize=2)
        
        ax2.axhline(y=0, color='black', linestyle='-', alpha=0.3)
        ax2.set_xlabel('Hour of Day')
        ax2.set_ylabel('EC Bias (m/s)')
        ax2.set_title('EC Bias Diurnal Pattern')
        ax2.grid(True, alpha=0.3, axis='y')
        ax2.set_xticks(hours[::2])
        
        # 添加数值标签（仅显示绝对值较大的）
        for i, (hour, bias) in enumerate(zip(hours, ec_bias_means)):
            if abs(bias) > 0.1:
                ax2.text(hour, bias + 0.02 if bias > 0 else bias - 0.05, 
                        f'{bias:.2f}', ha='center', va='bottom' if bias > 0 else 'top', 
                        fontsize=8, fontweight='bold')
        
        # 仅GFS bias
        ax3 = axes[1, 0]
        bars_gfs = ax3.bar(hours, gfs_bias_means, color=self.colors['gfs_bias'], 
                          alpha=0.8, edgecolor='black', linewidth=0.5)
        ax3.errorbar(hours, gfs_bias_means, yerr=gfs_bias_stds, 
                    fmt='none', color='black', alpha=0.6, capsize=2)
        
        ax3.axhline(y=0, color='black', linestyle='-', alpha=0.3)
        ax3.set_xlabel('Hour of Day')
        ax3.set_ylabel('GFS Bias (m/s)')
        ax3.set_title('GFS Bias Diurnal Pattern')
        ax3.grid(True, alpha=0.3, axis='y')
        ax3.set_xticks(hours[::2])
        
        # 添加数值标签（仅显示绝对值较大的）
        for i, (hour, bias) in enumerate(zip(hours, gfs_bias_means)):
            if abs(bias) > 0.1:
                ax3.text(hour, bias + 0.02 if bias > 0 else bias - 0.05, 
                        f'{bias:.2f}', ha='center', va='bottom' if bias > 0 else 'top', 
                        fontsize=8, fontweight='bold')
        
        # 统计汇总
        ax4 = axes[1, 1]
        
        # 计算统计
        overall_ec_bias = np.mean(ec_bias_means)
        overall_gfs_bias = np.mean(gfs_bias_means)
        max_ec_bias = max(ec_bias_means, key=abs)
        max_gfs_bias = max(gfs_bias_means, key=abs)
        
        # 创建统计文本
        stats_text = f"""
Statistics Summary:

EC Bias:
  • Mean: {overall_ec_bias:+.3f} m/s
  • Max: {max_ec_bias:+.3f} m/s
  • Range: {max(ec_bias_means) - min(ec_bias_means):.3f} m/s

GFS Bias:
  • Mean: {overall_gfs_bias:+.3f} m/s  
  • Max: {max_gfs_bias:+.3f} m/s
  • Range: {max(gfs_bias_means) - min(gfs_bias_means):.3f} m/s

Peak Hours:
  • EC Max+: {hours[np.argmax(ec_bias_means)]:02d}h
  • EC Max-: {hours[np.argmin(ec_bias_means)]:02d}h
  • GFS Max+: {hours[np.argmax(gfs_bias_means)]:02d}h
  • GFS Max-: {hours[np.argmin(gfs_bias_means)]:02d}h
        """
        
        ax4.text(0.05, 0.95, stats_text, transform=ax4.transAxes, 
                fontsize=10, verticalalignment='top', fontfamily='monospace',
                bbox=dict(boxstyle='round,pad=0.5', facecolor='lightgray', alpha=0.8))
        ax4.set_title('Statistics Summary')
        ax4.axis('off')
        
        plt.tight_layout()
        
        # 保存图片
        filename = f"{month_info['month_name']}_{height}_bias_diurnal.png"
        plt.savefig(os.path.join(output_path, filename), dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"  ✅ Saved: {filename}")
    
    def _create_comprehensive_comparison(self, output_path):
        """创建综合对比图"""
        print("  Creating comprehensive comparison plots...")
        
        # 按年份分组月份
        years_data = {}
        for month_name, month_data in self.monthly_stats.items():
            year = month_data['info']['year']
            if year not in years_data:
                years_data[year] = []
            years_data[year].append((month_name, month_data))
        
        # 为每年创建对比图
        for year, year_months in years_data.items():
            self._create_yearly_comparison(output_path, year_months, year)
    
    def _create_yearly_comparison(self, output_path, months_data, year):
        """创建年度对比图"""
        months_count = len(months_data)
        
        # 动态确定子图布局
        if months_count <= 3:
            rows, cols = 2, months_count
            figsize = (6 * months_count, 10)
        elif months_count <= 6:
            rows, cols = 2, 3
            figsize = (18, 10)
        elif months_count <= 12:
            rows, cols = 4, 3
            figsize = (18, 16)
        else:
            rows, cols = 4, 4
            figsize = (20, 16)
            months_data = months_data[:16]
        
        # 为每个高度层创建年度对比图
        for height in self.heights:
            fig, axes = plt.subplots(rows, cols, figsize=figsize)
            if months_count == 1:
                axes = [axes]
            elif rows == 1 or cols == 1:
                axes = axes.flatten()
            else:
                axes = axes.flatten()
            
            fig.suptitle(f'{year}年 {height} Monthly Bias Comparison', 
                        fontsize=16, fontweight='bold')
            
            for i, (month_name, month_data) in enumerate(months_data):
                if i >= len(axes):
                    break
                
                month_info = month_data['info']
                
                if height in month_data['heights_stats']:
                    hourly_stats = month_data['heights_stats'][height]
                    ax = axes[i]
                    
                    hours = sorted(hourly_stats.keys())
                    ec_bias_means = [hourly_stats[h]['ec_bias_mean'] for h in hours]
                    gfs_bias_means = [hourly_stats[h]['gfs_bias_mean'] for h in hours]
                    
                    # 并排柱状图
                    x = np.array(hours)
                    width = 0.35
                    
                    ax.bar(x - width/2, ec_bias_means, width, 
                          label='EC' if i == 0 else "", 
                          color=self.colors['ec_bias'], alpha=0.8)
                    ax.bar(x + width/2, gfs_bias_means, width,
                          label='GFS' if i == 0 else "", 
                          color=self.colors['gfs_bias'], alpha=0.8)
                    
                    ax.axhline(y=0, color='black', linestyle='-', alpha=0.3)
                    ax.set_title(f'{month_info["month"]:02d}月\n({month_info["data_count"]:,} samples)', 
                                fontsize=10)
                    ax.grid(True, alpha=0.3, axis='y')
                    ax.set_xticks(hours[::4])
                    
                    if i == 0:
                        ax.legend(fontsize=8)
            
            # 隐藏多余的子图
            for i in range(len(months_data), len(axes)):
                axes[i].axis('off')
            
            plt.tight_layout()
            plt.savefig(os.path.join(output_path, f'{year}_{height}_monthly_bias_comparison.png'), 
                       dpi=300, bbox_inches='tight')
            plt.close()
            
            print(f"  ✅ Saved: {year}_{height}_monthly_bias_comparison.png")
    
    def _create_height_comparison(self, output_path):
        """创建高度层对比图"""
        print("  Creating height level comparison plots...")
        
        # 为每个月创建高度层对比
        for month_name, month_data in self.monthly_stats.items():
            month_info = month_data['info']
            
            # 风速对比
            self._create_height_wind_speed_comparison(month_info, month_data, output_path)
            
            # Bias对比
            self._create_height_bias_comparison(month_info, month_data, output_path)
    
    def _create_height_wind_speed_comparison(self, month_info, month_data, output_path):
        """创建高度层风速对比图"""
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle(f'{month_info["display_name"]} Wind Speed Height Comparison', 
                    fontsize=16, fontweight='bold')
        
        heights_stats = month_data['heights_stats']
        
        # 观测数据对比
        ax1 = axes[0, 0]
        for i, height in enumerate(self.heights):
            if height in heights_stats:
                hourly_stats = heights_stats[height]
                hours = sorted(hourly_stats.keys())
                obs_means = [hourly_stats[h]['obs_mean'] for h in hours]
                
                color = plt.cm.Greens(0.3 + i * 0.2)
                ax1.bar([h + i*0.2 for h in hours], obs_means, width=0.18, 
                       label=height, color=color, alpha=0.8)
        
        ax1.set_xlabel('Hour of Day')
        ax1.set_ylabel('Wind Speed (m/s)')
        ax1.set_title('Observed Wind Speed')
        ax1.legend()
        ax1.grid(True, alpha=0.3, axis='y')
        
        # EC预报对比
        ax2 = axes[0, 1]
        for i, height in enumerate(self.heights):
            if height in heights_stats:
                hourly_stats = heights_stats[height]
                hours = sorted(hourly_stats.keys())
                ec_means = [hourly_stats[h]['ec_mean'] for h in hours]
                
                color = plt.cm.Blues(0.3 + i * 0.2)
                ax2.bar([h + i*0.2 for h in hours], ec_means, width=0.18, 
                       label=height, color=color, alpha=0.8)
        
        ax2.set_xlabel('Hour of Day')
        ax2.set_ylabel('Wind Speed (m/s)')
        ax2.set_title('EC Forecast')
        ax2.legend()
        ax2.grid(True, alpha=0.3, axis='y')
        
        # GFS预报对比
        ax3 = axes[1, 0]
        for i, height in enumerate(self.heights):
            if height in heights_stats:
                hourly_stats = heights_stats[height]
                hours = sorted(hourly_stats.keys())
                gfs_means = [hourly_stats[h]['gfs_mean'] for h in hours]
                
                color = plt.cm.Reds(0.3 + i * 0.2)
                ax3.bar([h + i*0.2 for h in hours], gfs_means, width=0.18, 
                       label=height, color=color, alpha=0.8)
        
        ax3.set_xlabel('Hour of Day')
        ax3.set_ylabel('Wind Speed (m/s)')
        ax3.set_title('GFS Forecast')
        ax3.legend()
        ax3.grid(True, alpha=0.3, axis='y')
        
        # 相关系数对比
        ax4 = axes[1, 1]
        height_labels = []
        ec_corrs = []
        gfs_corrs = []
        
        for height in self.heights:
            if height in heights_stats:
                hourly_stats = heights_stats[height]
                hours = sorted(hourly_stats.keys())
                
                obs_values = [hourly_stats[h]['obs_mean'] for h in hours]
                ec_values = [hourly_stats[h]['ec_mean'] for h in hours]
                gfs_values = [hourly_stats[h]['gfs_mean'] for h in hours]
                
                if len(obs_values) > 5:
                    ec_corr = np.corrcoef(obs_values, ec_values)[0, 1]
                    gfs_corr = np.corrcoef(obs_values, gfs_values)[0, 1]
                    
                    height_labels.append(height)
                    ec_corrs.append(ec_corr)
                    gfs_corrs.append(gfs_corr)
        
        if height_labels:
            x = np.arange(len(height_labels))
            width = 0.35
            
            bars1 = ax4.bar(x - width/2, ec_corrs, width, label='EC', 
                           color=self.colors['ec'], alpha=0.7)
            bars2 = ax4.bar(x + width/2, gfs_corrs, width, label='GFS', 
                           color=self.colors['gfs'], alpha=0.7)
            
            ax4.set_ylabel('Correlation Coefficient')
            ax4.set_title('Forecast Accuracy (Correlation)')
            ax4.set_xticks(x)
            ax4.set_xticklabels(height_labels)
            ax4.legend()
            ax4.grid(True, alpha=0.3, axis='y')
            ax4.set_ylim(0, 1)
            
            # 添加数值标签
            for bar, corr in zip(bars1, ec_corrs):
                ax4.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                        f'{corr:.3f}', ha='center', va='bottom', fontsize=9, fontweight='bold')
            
            for bar, corr in zip(bars2, gfs_corrs):
                ax4.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                        f'{corr:.3f}', ha='center', va='bottom', fontsize=9, fontweight='bold')
        
        plt.tight_layout()
        filename = f"{month_info['month_name']}_height_wind_speed_comparison.png"
        plt.savefig(os.path.join(output_path, filename), dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"  ✅ Saved: {filename}")
    
    def _create_height_bias_comparison(self, month_info, month_data, output_path):
        """创建高度层bias对比图"""
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle(f'{month_info["display_name"]} Bias Height Comparison', 
                    fontsize=16, fontweight='bold')
        
        heights_stats = month_data['heights_stats']
        
        # EC bias对比
        ax1 = axes[0, 0]
        for i, height in enumerate(self.heights):
            if height in heights_stats:
                hourly_stats = heights_stats[height]
                hours = sorted(hourly_stats.keys())
                ec_bias_means = [hourly_stats[h]['ec_bias_mean'] for h in hours]
                
                color = plt.cm.Oranges(0.4 + i * 0.15)
                ax1.bar([h + i*0.2 for h in hours], ec_bias_means, width=0.18, 
                       label=height, color=color, alpha=0.8)
        
        ax1.axhline(y=0, color='black', linestyle='-', alpha=0.3)
        ax1.set_xlabel('Hour of Day')
        ax1.set_ylabel('EC Bias (m/s)')
        ax1.set_title('EC Bias by Height')
        ax1.legend()
        ax1.grid(True, alpha=0.3, axis='y')
        
        # GFS bias对比
        ax2 = axes[0, 1]
        for i, height in enumerate(self.heights):
            if height in heights_stats:
                hourly_stats = heights_stats[height]
                hours = sorted(hourly_stats.keys())
                gfs_bias_means = [hourly_stats[h]['gfs_bias_mean'] for h in hours]
                
                color = plt.cm.Reds(0.4 + i * 0.15)
                ax2.bar([h + i*0.2 for h in hours], gfs_bias_means, width=0.18, 
                       label=height, color=color, alpha=0.8)
        
        ax2.axhline(y=0, color='black', linestyle='-', alpha=0.3)
        ax2.set_xlabel('Hour of Day')
        ax2.set_ylabel('GFS Bias (m/s)')
        ax2.set_title('GFS Bias by Height')
        ax2.legend()
        ax2.grid(True, alpha=0.3, axis='y')
        
        # 平均bias对比
        ax3 = axes[1, 0]
        height_labels = []
        ec_mean_bias = []
        gfs_mean_bias = []
        
        for height in self.heights:
            if height in heights_stats:
                hourly_stats = heights_stats[height]
                hours = sorted(hourly_stats.keys())
                
                ec_bias_values = [hourly_stats[h]['ec_bias_mean'] for h in hours]
                gfs_bias_values = [hourly_stats[h]['gfs_bias_mean'] for h in hours]
                
                height_labels.append(height)
                ec_mean_bias.append(np.mean(ec_bias_values))
                gfs_mean_bias.append(np.mean(gfs_bias_values))
        
        if height_labels:
            x = np.arange(len(height_labels))
            width = 0.35
            
            bars1 = ax3.bar(x - width/2, ec_mean_bias, width, label='EC Bias', 
                           color=self.colors['ec_bias'], alpha=0.7)
            bars2 = ax3.bar(x + width/2, gfs_mean_bias, width, label='GFS Bias', 
                           color=self.colors['gfs_bias'], alpha=0.7)
            
            ax3.axhline(y=0, color='black', linestyle='-', alpha=0.4)
            ax3.set_ylabel('Mean Bias (m/s)')
            ax3.set_title('Average Bias by Height')
            ax3.set_xticks(x)
            ax3.set_xticklabels(height_labels)
            ax3.legend()
            ax3.grid(True, alpha=0.3, axis='y')
            
            # 添加数值标签
            for bar, bias in zip(bars1, ec_mean_bias):
                ax3.text(bar.get_x() + bar.get_width()/2, 
                        bias + 0.02 if bias > 0 else bias - 0.05,
                        f'{bias:+.3f}', ha='center', 
                        va='bottom' if bias > 0 else 'top', fontsize=9, fontweight='bold')
            
            for bar, bias in zip(bars2, gfs_mean_bias):
                ax3.text(bar.get_x() + bar.get_width()/2, 
                        bias + 0.02 if bias > 0 else bias - 0.05,
                        f'{bias:+.3f}', ha='center', 
                        va='bottom' if bias > 0 else 'top', fontsize=9, fontweight='bold')
        
        # 统计摘要
        ax4 = axes[1, 1]
        
        # 创建统计文本
        stats_text = f"""
Height Level Statistics:

Best EC Performance:
"""
        
        if height_labels and ec_mean_bias:
            best_ec_idx = np.argmin([abs(bias) for bias in ec_mean_bias])
            best_gfs_idx = np.argmin([abs(bias) for bias in gfs_mean_bias])
            
            stats_text += f"  • {height_labels[best_ec_idx]} ({ec_mean_bias[best_ec_idx]:+.3f} m/s)\n\n"
            stats_text += f"Best GFS Performance:\n"
            stats_text += f"  • {height_labels[best_gfs_idx]} ({gfs_mean_bias[best_gfs_idx]:+.3f} m/s)\n\n"
            
            # 找出bias最大的高度
            worst_ec_idx = np.argmax([abs(bias) for bias in ec_mean_bias])
            worst_gfs_idx = np.argmax([abs(bias) for bias in gfs_mean_bias])
            
            stats_text += f"Largest EC Bias:\n"
            stats_text += f"  • {height_labels[worst_ec_idx]} ({ec_mean_bias[worst_ec_idx]:+.3f} m/s)\n\n"
            stats_text += f"Largest GFS Bias:\n"
            stats_text += f"  • {height_labels[worst_gfs_idx]} ({gfs_mean_bias[worst_gfs_idx]:+.3f} m/s)"
        
        ax4.text(0.05, 0.95, stats_text, transform=ax4.transAxes, 
                fontsize=10, verticalalignment='top', fontfamily='monospace',
                bbox=dict(boxstyle='round,pad=0.5', facecolor='lightgray', alpha=0.8))
        ax4.set_title('Height Performance Summary')
        ax4.axis('off')
        
        plt.tight_layout()
        filename = f"{month_info['month_name']}_height_bias_comparison.png"
        plt.savefig(os.path.join(output_path, filename), dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"  ✅ Saved: {filename}")
    
    def save_analysis_data(self):
        """保存分析数据"""
        print("💾 Saving analysis data...")
        
        # 创建输出文件夹
        data_path = os.path.join(self.output_path, 'wind_forecast_analysis_data')
        os.makedirs(data_path, exist_ok=True)
        
        # 保存详细数据
        all_data = []
        for month_name, month_data in self.monthly_stats.items():
            month_info = month_data['info']
            
            for height in self.heights:
                if height in month_data['heights_stats']:
                    hourly_stats = month_data['heights_stats'][height]
                    
                    for hour, stats in hourly_stats.items():
                        all_data.append({
                            'Year': month_info['year'],
                            'Month': month_info['month'],
                            'Month_Name': month_name,
                            'Display_Name': month_info['display_name'],
                            'Height': height,
                            'Hour': hour,
                            'Sample_Count': stats['count'],
                            'Obs_Mean': stats['obs_mean'],
                            'Obs_Std': stats['obs_std'],
                            'EC_Mean': stats['ec_mean'],
                            'EC_Std': stats['ec_std'],
                            'GFS_Mean': stats['gfs_mean'],
                            'GFS_Std': stats['gfs_std'],
                            'EC_Bias_Mean': stats['ec_bias_mean'],
                            'EC_Bias_Std': stats['ec_bias_std'],
                            'GFS_Bias_Mean': stats['gfs_bias_mean'],
                            'GFS_Bias_Std': stats['gfs_bias_std']
                        })
        
        analysis_df = pd.DataFrame(all_data)
        analysis_df.to_csv(os.path.join(data_path, 'wind_forecast_analysis_data.csv'), index=False)
        
        # 保存汇总统计
        summary_stats = {}
        for month_name, month_data in self.monthly_stats.items():
            month_info = month_data['info']
            
            month_info_serializable = {
                'year': int(month_info['year']),
                'month': int(month_info['month']),
                'month_name': str(month_info['month_name']),
                'display_name': str(month_info['display_name']),
                'data_count': int(month_info['data_count'])
            }
            
            summary_stats[month_name] = {
                'month_info': month_info_serializable,
                'heights_summary': {}
            }
            
            for height in self.heights:
                if height in month_data['heights_stats']:
                    hourly_stats = month_data['heights_stats'][height]
                    hours = sorted(hourly_stats.keys())
                    
                    ec_bias_means = [hourly_stats[h]['ec_bias_mean'] for h in hours]
                    gfs_bias_means = [hourly_stats[h]['gfs_bias_mean'] for h in hours]
                    
                    summary_stats[month_name]['heights_summary'][height] = {
                        'ec_bias': {
                            'overall_mean': float(np.mean(ec_bias_means)),
                            'overall_std': float(np.std(ec_bias_means)),
                            'max_positive': float(max(ec_bias_means)),
                            'max_negative': float(min(ec_bias_means)),
                            'range': float(max(ec_bias_means) - min(ec_bias_means))
                        },
                        'gfs_bias': {
                            'overall_mean': float(np.mean(gfs_bias_means)),
                            'overall_std': float(np.std(gfs_bias_means)),
                            'max_positive': float(max(gfs_bias_means)),
                            'max_negative': float(min(gfs_bias_means)),
                            'range': float(max(gfs_bias_means) - min(gfs_bias_means))
                        }
                    }
        
        import json
        with open(os.path.join(data_path, 'wind_forecast_summary.json'), 'w') as f:
            json.dump(summary_stats, f, indent=4, ensure_ascii=False)
        
        print(f"✅ Analysis data saved to {data_path}")
        return summary_stats
    
    def generate_report(self, summary_stats):
        """生成分析报告"""
        print("\n" + "="*70)
        print("📊 WIND FORECAST DIURNAL BAR CHART ANALYSIS REPORT")
        print("="*70)
        
        print(f"\n🌟 Analysis Overview:")
        print(f"   • Total Months Analyzed: {len(self.months_info)}")
        print(f"   • Total Records: {len(self.data):,}")
        print(f"   • Date Range: {self.data['datetime'].min().date()} to {self.data['datetime'].max().date()}")
        print(f"   • Height Levels: {', '.join(self.heights)}")
        
        print(f"\n📊 Monthly and Height Summary:")
        
        # 按年份组织显示
        years_data = {}
        for month_name, stats in summary_stats.items():
            year = stats['month_info']['year']
            if year not in years_data:
                years_data[year] = []
            years_data[year].append((month_name, stats))
        
        for year in sorted(years_data.keys()):
            print(f"\n   {year}年:")
            year_months = sorted(years_data[year], key=lambda x: x[1]['month_info']['month'])
            
            for month_name, stats in year_months:
                month_info = stats['month_info']
                print(f"     {month_info['month']:2d}月 ({month_info['data_count']:,} samples):")
                
                for height in self.heights:
                    if height in stats['heights_summary']:
                        height_stats = stats['heights_summary'][height]
                        ec_bias = height_stats['ec_bias']['overall_mean']
                        gfs_bias = height_stats['gfs_bias']['overall_mean']
                        print(f"       {height}: EC {ec_bias:+.3f}, GFS {gfs_bias:+.3f} m/s")
        
        print(f"\n🔍 Key Findings:")
        
        # 找出最佳预报性能
        best_ec_performance = {}
        best_gfs_performance = {}
        
        for month_name, stats in summary_stats.items():
            for height in self.heights:
                if height in stats['heights_summary']:
                    height_stats = stats['heights_summary'][height]
                    ec_abs_bias = abs(height_stats['ec_bias']['overall_mean'])
                    gfs_abs_bias = abs(height_stats['gfs_bias']['overall_mean'])
                    
                    if height not in best_ec_performance or ec_abs_bias < best_ec_performance[height]['bias']:
                        best_ec_performance[height] = {
                            'bias': ec_abs_bias,
                            'month': stats['month_info']['display_name'],
                            'value': height_stats['ec_bias']['overall_mean']
                        }
                    
                    if height not in best_gfs_performance or gfs_abs_bias < best_gfs_performance[height]['bias']:
                        best_gfs_performance[height] = {
                            'bias': gfs_abs_bias,
                            'month': stats['month_info']['display_name'],
                            'value': height_stats['gfs_bias']['overall_mean']
                        }
        
        print("   最佳预报性能 (各高度层最小bias):")
        for height in self.heights:
            if height in best_ec_performance and height in best_gfs_performance:
                ec_best = best_ec_performance[height]
                gfs_best = best_gfs_performance[height]
                print(f"   • {height}: EC最佳 {ec_best['value']:+.3f} m/s ({ec_best['month']})")
                print(f"             GFS最佳 {gfs_best['value']:+.3f} m/s ({gfs_best['month']})")
        
        print(f"\n📁 Output Files Generated:")
        print(f"   • Individual monthly plots: wind_forecast_diurnal_plots/")
        print(f"   • Wind speed comparisons: *_wind_speed_diurnal.png")
        print(f"   • Bias comparisons: *_bias_diurnal.png")
        print(f"   • Height level comparisons: *_height_*_comparison.png")
        print(f"   • Yearly comparisons: *_monthly_bias_comparison.png")
        print(f"   • Data files: wind_forecast_analysis_data/")
        
        print("\n🎉 Wind forecast diurnal bar chart analysis completed!")
        print("="*70)
    
    def run_analysis(self):
        """运行完整分析"""
        print("🌪️ Starting Wind Forecast Diurnal Bar Chart Analysis...")
        print("="*60)
        
        # 1. 计算统计数据
        self.calculate_monthly_diurnal_stats()
        
        # 2. 创建可视化
        self.create_monthly_plots()
        
        # 3. 保存数据
        summary_stats = self.save_analysis_data()
        
        # 4. 生成报告
        self.generate_report(summary_stats)


def main():
    """主函数"""
    # 数据和结果路径
    data_path = '/Users/xiaxin/work/WindForecast_Project/01_Data/processed/imputed_data/changma_imputed_complete.csv'
    results_dir = '/Users/xiaxin/work/WindForecast_Project/03_Results/7everydiurnal_sequence/'
    
    # 检查文件是否存在
    if not os.path.exists(data_path):
        print(f"❌ Error: Data file not found at {data_path}")
        return
    
    # 创建分析器并运行分析
    analyzer = WindForecastDiurnalBarAnalyzer(data_path, results_dir)
    analyzer.run_analysis()


if __name__ == "__main__":
    main()