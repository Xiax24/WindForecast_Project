#!/usr/bin/env python3
"""
数值预报评估器 - 分类对比分析模块
包含风切变分类和昼夜对比分析
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

class NWPClassificationAnalysis:
    def __init__(self, evaluator):
        self.evaluator = evaluator
        self.data = evaluator.data
        self.save_path = evaluator.save_path
        self.evaluation_results = evaluator.evaluation_results
        self.key_variables = evaluator.key_variables
        
    def plot_shear_classification_comparison(self):
        """绘制风切变分类对比分析"""
        print("📊 绘制风切变分类对比分析...")
        
        if not self.evaluation_results:
            print("❌ 无评估结果可绘制")
            return
        
        fig, axes = plt.subplots(1, 3, figsize=(18, 6))
        fig.suptitle('不同风切变强度下的预报性能对比', fontsize=16, fontweight='bold')
        
        shear_types = ['weak', 'moderate', 'strong']
        shear_names = ['弱切变 (α<0.2)', '中等切变 (0.2≤α<0.3)', '强切变 (α≥0.3)']
        
        for i, (shear_type, shear_name) in enumerate(zip(shear_types, shear_names)):
            ax = axes[i]
            try:
                self._plot_shear_specific_analysis(ax, shear_type, shear_name)
            except Exception as e:
                print(f"  ❌ {shear_name} 绘制失败: {e}")
                ax.text(0.5, 0.5, f'{shear_name}\n绘制失败', ha='center', va='center', 
                       transform=ax.transAxes, fontsize=12)
                ax.set_title(shear_name)
        
        plt.tight_layout()
        plt.savefig(f"{self.save_path}/03_shear_classification_comparison.png", dpi=300, bbox_inches='tight')
        plt.show()
    
    def _plot_shear_specific_analysis(self, ax, shear_type, shear_name):
        """绘制特定切变类型的分析"""
        target_classes = [cls for cls in self.evaluation_results.keys() if cls.startswith(shear_type)]
        
        if not target_classes:
            ax.text(0.5, 0.5, f'{shear_name}\n暂无数据', ha='center', va='center', 
                   transform=ax.transAxes, fontsize=12)
            ax.set_title(shear_name)
            return
        
        # 收集数据
        variables = []
        ec_r2 = []
        gfs_r2 = []
        periods = []
        
        for cls in target_classes:
            period = '白天' if 'day' in cls else '夜间'
            
            for var_name, var_info in self.key_variables.items():
                if var_name in self.evaluation_results[cls]:
                    ec_val = self.evaluation_results[cls][var_name]['EC']['R2']
                    gfs_val = self.evaluation_results[cls][var_name]['GFS']['R2']
                    
                    if not (np.isnan(ec_val) or np.isnan(gfs_val)):
                        variables.append(f"{var_info['name']}\n({period})")
                        ec_r2.append(ec_val)
                        gfs_r2.append(gfs_val)
                        periods.append(period)
        
        if not variables:
            ax.text(0.5, 0.5, f'{shear_name}\n数据不足', ha='center', va='center', 
                   transform=ax.transAxes, fontsize=12)
            ax.set_title(shear_name)
            return
        
        # 绘制对比
        x = np.arange(len(variables))
        width = 0.35
        
        bars1 = ax.bar(x - width/2, ec_r2, width, label='EC', color='blue', alpha=0.8)
        bars2 = ax.bar(x + width/2, gfs_r2, width, label='GFS', color='red', alpha=0.8)
        
        ax.set_xlabel('变量')
        ax.set_ylabel('R² Score')
        ax.set_title(shear_name)
        ax.set_xticks(x)
        ax.set_xticklabels(variables, rotation=45, ha='right')
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.set_ylim(0, 1)
        
        # 添加数值标签
        for bars in [bars1, bars2]:
            for bar in bars:
                height = bar.get_height()
                if not np.isnan(height):
                    ax.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                           f'{height:.3f}', ha='center', va='bottom', fontsize=8)
    
    def plot_diurnal_comparison(self):
        """绘制昼夜对比分析"""
        print("📊 绘制昼夜对比分析...")
        
        if not self.evaluation_results:
            print("❌ 无评估结果可绘制")
            return
        
        fig, axes = plt.subplots(1, 2, figsize=(15, 6))
        fig.suptitle('昼夜预报性能对比', fontsize=16, fontweight='bold')
        
        periods = ['day', 'night']
        period_names = ['白天 (6:00-18:00)', '夜间 (18:00-6:00)']
        
        for i, (period, period_name) in enumerate(zip(periods, period_names)):
            ax = axes[i]
            try:
                self._plot_period_specific_analysis(ax, period, period_name)
            except Exception as e:
                print(f"  ❌ {period_name} 绘制失败: {e}")
                ax.text(0.5, 0.5, f'{period_name}\n绘制失败', ha='center', va='center', 
                       transform=ax.transAxes, fontsize=12)
                ax.set_title(period_name)
        
        plt.tight_layout()
        plt.savefig(f"{self.save_path}/04_diurnal_comparison.png", dpi=300, bbox_inches='tight')
        plt.show()
    
    def _plot_period_specific_analysis(self, ax, period, period_name):
        """绘制特定时段的分析"""
        target_classes = [cls for cls in self.evaluation_results.keys() if cls.endswith(period)]
        
        if not target_classes:
            ax.text(0.5, 0.5, f'{period_name}\n暂无数据', ha='center', va='center', 
                   transform=ax.transAxes, fontsize=12)
            ax.set_title(period_name)
            return
        
        variables = []
        ec_r2 = []
        gfs_r2 = []
        shear_types = []
        
        for cls in target_classes:
            shear_type = cls.split('_')[0]
            shear_name = {'weak': '弱切变', 'moderate': '中等切变', 'strong': '强切变'}.get(shear_type, shear_type)
            
            for var_name, var_info in self.key_variables.items():
                if var_name in self.evaluation_results[cls]:
                    ec_val = self.evaluation_results[cls][var_name]['EC']['R2']
                    gfs_val = self.evaluation_results[cls][var_name]['GFS']['R2']
                    
                    if not (np.isnan(ec_val) or np.isnan(gfs_val)):
                        variables.append(f"{var_info['name']}\n({shear_name})")
                        ec_r2.append(ec_val)
                        gfs_r2.append(gfs_val)
                        shear_types.append(shear_type)
        
        if not variables:
            ax.text(0.5, 0.5, f'{period_name}\n数据不足', ha='center', va='center', 
                   transform=ax.transAxes, fontsize=12)
            ax.set_title(period_name)
            return
        
        # 绘制对比
        x = np.arange(len(variables))
        width = 0.35
        
        bars1 = ax.bar(x - width/2, ec_r2, width, label='EC', color='blue', alpha=0.8)
        bars2 = ax.bar(x + width/2, gfs_r2, width, label='GFS', color='red', alpha=0.8)
        
        ax.set_xlabel('变量')
        ax.set_ylabel('R² Score')
        ax.set_title(period_name)
        ax.set_xticks(x)
        ax.set_xticklabels(variables, rotation=45, ha='right')
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.set_ylim(0, 1)
        
        # 添加数值标签
        for bars in [bars1, bars2]:
            for bar in bars:
                height = bar.get_height()
                if not np.isnan(height):
                    ax.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                           f'{height:.3f}', ha='center', va='bottom', fontsize=8)

if __name__ == "__main__":
    print("分类对比分析模块测试")
    print("请通过主运行脚本使用此模块")