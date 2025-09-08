#!/usr/bin/env python3
"""
数值预报评估器 - 数据分布可视化模块
包含数据分布概览图表
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

class NWPDistributionPlots:
    def __init__(self, evaluator):
        self.evaluator = evaluator
        self.data = evaluator.data
        self.save_path = evaluator.save_path
        
    def plot_data_distribution(self):
        """绘制数据分布概览的"""
        print("📊 绘制数据分布概览...")
        
        if self.data is None:
            print("❌ 无数据可绘制")
            return
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('数据分布概览', fontsize=16, fontweight='bold')
        
        # 1. 风切变分类分布
        ax1 = axes[0, 0]
        try:
            shear_counts = self.data['shear_group'].value_counts()
            colors = ['lightblue', 'orange', 'lightcoral']
            
            if len(shear_counts) > 0:
                wedges, texts, autotexts = ax1.pie(
                    shear_counts.values, 
                    labels=[f'{k}\n({v}条)' for k, v in shear_counts.items()], 
                    autopct='%1.1f%%', 
                    colors=colors[:len(shear_counts)], 
                    startangle=90
                )
                ax1.set_title('风切变强度分布')
            else:
                ax1.text(0.5, 0.5, '无风切变数据', ha='center', va='center', transform=ax1.transAxes)
                ax1.set_title('风切变强度分布')
        except Exception as e:
            print(f"风切变分布图绘制错误: {e}")
            ax1.text(0.5, 0.5, '绘制失败', ha='center', va='center', transform=ax1.transAxes)
            ax1.set_title('风切变强度分布')
        
        # 2. 昼夜分布
        ax2 = axes[0, 1]
        try:
            diurnal_counts = self.data['is_daytime'].value_counts()
            diurnal_labels = ['夜间' if not k else '白天' for k in diurnal_counts.index]
            colors = ['darkblue', 'gold']
            
            if len(diurnal_counts) > 0:
                wedges, texts, autotexts = ax2.pie(
                    diurnal_counts.values, 
                    labels=[f'{label}\n({count}条)' for label, count in zip(diurnal_labels, diurnal_counts.values)], 
                    autopct='%1.1f%%', 
                    colors=colors[:len(diurnal_counts)], 
                    startangle=90
                )
                ax2.set_title('昼夜时段分布')
            else:
                ax2.text(0.5, 0.5, '无昼夜数据', ha='center', va='center', transform=ax2.transAxes)
                ax2.set_title('昼夜时段分布')
        except Exception as e:
            print(f"昼夜分布图绘制错误: {e}")
            ax2.text(0.5, 0.5, '绘制失败', ha='center', va='center', transform=ax2.transAxes)
            ax2.set_title('昼夜时段分布')
        
        # 3. 综合分类分布
        ax3 = axes[1, 0]
        try:
            class_counts = self.data['shear_diurnal_class'].value_counts()
            
            if len(class_counts) > 0:
                bars = ax3.bar(range(len(class_counts)), class_counts.values, color='steelblue')
                ax3.set_xticks(range(len(class_counts)))
                ax3.set_xticklabels([c.replace('_', '\n') for c in class_counts.index], rotation=45, ha='right')
                ax3.set_ylabel('样本数量')
                ax3.set_title('风切变-昼夜综合分类分布')
                ax3.grid(True, alpha=0.3)
                
                # 添加数值标签
                for bar in bars:
                    height = bar.get_height()
                    ax3.text(bar.get_x() + bar.get_width()/2., height + height*0.01,
                            f'{int(height)}', ha='center', va='bottom', fontweight='bold')
            else:
                ax3.text(0.5, 0.5, '无分类数据', ha='center', va='center', transform=ax3.transAxes)
                ax3.set_title('风切变-昼夜综合分类分布')
        except Exception as e:
            print(f"综合分类分布图绘制错误: {e}")
            ax3.text(0.5, 0.5, '绘制失败', ha='center', va='center', transform=ax3.transAxes)
            ax3.set_title('风切变-昼夜综合分类分布')
        
        # 4. 风切变系数分布
        ax4 = axes[1, 1]
        try:
            if 'wind_shear_alpha' in self.data.columns:
                alpha_values = self.data['wind_shear_alpha']
                alpha_values = alpha_values[~np.isnan(alpha_values)]  # 移除NaN值
                
                if len(alpha_values) > 0:
                    ax4.hist(alpha_values, bins=30, color='lightgreen', alpha=0.7, edgecolor='black')
                    ax4.axvline(self.evaluator.shear_thresholds['weak_upper'], color='red', linestyle='--', 
                               label=f'弱切变阈值 ({self.evaluator.shear_thresholds["weak_upper"]})')
                    ax4.axvline(self.evaluator.shear_thresholds['moderate_upper'], color='orange', linestyle='--', 
                               label=f'中等切变阈值 ({self.evaluator.shear_thresholds["moderate_upper"]})')
                    ax4.set_xlabel('风切变系数 α')
                    ax4.set_ylabel('频数')
                    ax4.set_title('风切变系数分布')
                    ax4.legend()
                    ax4.grid(True, alpha=0.3)
                    
                    # 添加统计信息
                    ax4.text(0.02, 0.98, f'样本数: {len(alpha_values)}\n均值: {np.mean(alpha_values):.3f}\n标准差: {np.std(alpha_values):.3f}', 
                            transform=ax4.transAxes, va='top', fontsize=10,
                            bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
                else:
                    ax4.text(0.5, 0.5, '无有效风切变数据', ha='center', va='center', transform=ax4.transAxes)
                    ax4.set_title('风切变系数分布')
            else:
                ax4.text(0.5, 0.5, '无风切变系数', ha='center', va='center', transform=ax4.transAxes)
                ax4.set_title('风切变系数分布')
        except Exception as e:
            print(f"风切变系数分布图绘制错误: {e}")
            ax4.text(0.5, 0.5, '绘制失败', ha='center', va='center', transform=ax4.transAxes)
            ax4.set_title('风切变系数分布')
        
        plt.tight_layout()
        
        try:
            plt.savefig(f"{self.save_path}/01_data_distribution.png", dpi=300, bbox_inches='tight')
            print("✓ 数据分布图已保存")
        except Exception as e:
            print(f"保存数据分布图失败: {e}")
        
        plt.show()
    
    def plot_time_series_overview(self):
        """绘制时间序列概览"""
        print("📊 绘制时间序列概览...")
        
        if self.data is None or 'datetime' not in self.data.columns:
            print("❌ 无时间序列数据可绘制")
            return
        
        fig, axes = plt.subplots(3, 1, figsize=(15, 12))
        fig.suptitle('时间序列数据概览', fontsize=16, fontweight='bold')
        
        # 按月份采样，避免数据过多
        sample_data = self.data.copy()
        if len(sample_data) > 10000:
            sample_data = sample_data.sample(n=10000, random_state=42).sort_values('datetime')
        
        try:
            # 1. 风速时间序列
            ax1 = axes[0]
            for var_name in ['wind_speed_10m', 'wind_speed_70m']:
                if var_name in self.evaluator.key_variables:
                    var_info = self.evaluator.key_variables[var_name]
                    obs_col = var_info['obs']
                    if obs_col in sample_data.columns:
                        ax1.plot(sample_data['datetime'], sample_data[obs_col], 
                                label=var_info['name'], alpha=0.7, linewidth=0.8)
            
            ax1.set_ylabel('风速 (m/s)')
            ax1.set_title('风速时间序列')
            ax1.legend()
            ax1.grid(True, alpha=0.3)
            
            # 2. 温度时间序列
            ax2 = axes[1]
            if 'temperature_10m' in self.evaluator.key_variables:
                var_info = self.evaluator.key_variables['temperature_10m']
                obs_col = var_info['obs']
                if obs_col in sample_data.columns:
                    ax2.plot(sample_data['datetime'], sample_data[obs_col], 
                            color='red', alpha=0.7, linewidth=0.8, label=var_info['name'])
            
            ax2.set_ylabel('温度 (°C)')
            ax2.set_title('温度时间序列')
            ax2.legend()
            ax2.grid(True, alpha=0.3)
            
            # 3. 风切变系数时间序列
            ax3 = axes[2]
            if 'wind_shear_alpha' in sample_data.columns:
                valid_alpha = sample_data['wind_shear_alpha'].dropna()
                valid_dates = sample_data.loc[valid_alpha.index, 'datetime']
                
                ax3.plot(valid_dates, valid_alpha, color='green', alpha=0.7, linewidth=0.8)
                ax3.axhline(self.evaluator.shear_thresholds['weak_upper'], color='red', linestyle='--', 
                           label=f'弱切变阈值 ({self.evaluator.shear_thresholds["weak_upper"]})')
                ax3.axhline(self.evaluator.shear_thresholds['moderate_upper'], color='orange', linestyle='--', 
                           label=f'中等切变阈值 ({self.evaluator.shear_thresholds["moderate_upper"]})')
            
            ax3.set_xlabel('时间')
            ax3.set_ylabel('风切变系数 α')
            ax3.set_title('风切变系数时间序列')
            ax3.legend()
            ax3.grid(True, alpha=0.3)
            
            # 格式化x轴
            for ax in axes:
                ax.tick_params(axis='x', rotation=45)
            
        except Exception as e:
            print(f"时间序列图绘制错误: {e}")
            for i, ax in enumerate(axes):
                ax.text(0.5, 0.5, f'第{i+1}个图绘制失败', ha='center', va='center', transform=ax.transAxes)
        
        plt.tight_layout()
        
        try:
            plt.savefig(f"{self.save_path}/01b_time_series_overview.png", dpi=300, bbox_inches='tight')
            print("✓ 时间序列概览图已保存")
        except Exception as e:
            print(f"保存时间序列图失败: {e}")
        
        plt.show()

if __name__ == "__main__":
    # 测试分布图功能
    from nwp_evaluator_base import NWPEvaluatorBase
    
    DATA_PATH = "/Users/xiaxin/work/WindForecast_Project/01_Data/processed/imputed_data/changma_imputed_complete.csv"
    SAVE_PATH = "/Users/xiaxin/work/WindForecast_Project/03_Results/clear_nwp_evaluation_results"
    
    # 创建评估器
    evaluator = NWPEvaluatorBase(DATA_PATH, SAVE_PATH)
    
    # 加载数据
    if evaluator.load_and_prepare_data() is not None:
        # 创建分布图绘制器
        plotter = NWPDistributionPlots(evaluator)
        
        # 绘制分布图
        plotter.plot_data_distribution()
        plotter.plot_time_series_overview()
        print("✓ 分布图绘制完成")
    else:
        print("❌ 数据加载失败")