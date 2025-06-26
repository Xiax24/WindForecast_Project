#!/usr/bin/env python3
"""
数值预报评估器 - 变量性能分析模块
为每个变量创建详细的性能对比图
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

class NWPVariablePerformance:
    def __init__(self, evaluator):
        self.evaluator = evaluator
        self.data = evaluator.data
        self.save_path = evaluator.save_path
        self.evaluation_results = evaluator.evaluation_results
        self.key_variables = evaluator.key_variables
        
    def plot_variable_performance_detailed(self):
        """为每个变量绘制详细的性能对比图"""
        print("📊 绘制各变量详细性能对比...")
        
        if not self.evaluation_results:
            print("❌ 无评估结果可绘制")
            return
        
        for var_name, var_info in self.key_variables.items():
            print(f"  绘制 {var_info['name']} 性能图...")
            
            try:
                self._plot_single_variable_performance(var_name, var_info)
            except Exception as e:
                print(f"  ❌ {var_info['name']} 绘制失败: {e}")
                continue
    
    def _plot_single_variable_performance(self, var_name, var_info):
        """绘制单个变量的详细性能分析"""
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle(f'{var_info["name"]} 预报性能详细分析', fontsize=16, fontweight='bold')
        
        # 收集数据
        classes = list(self.evaluation_results.keys())
        ec_metrics = {metric: [] for metric in ['R2', 'RMSE', 'MAE', 'BIAS', 'CORR']}
        gfs_metrics = {metric: [] for metric in ['R2', 'RMSE', 'MAE', 'BIAS', 'CORR']}
        valid_classes = []
        
        for cls in classes:
            if var_name in self.evaluation_results[cls]:
                valid_data = True
                temp_ec_metrics = {}
                temp_gfs_metrics = {}
                
                for metric in ['R2', 'RMSE', 'MAE', 'BIAS', 'CORR']:
                    ec_val = self.evaluation_results[cls][var_name]['EC'][metric]
                    gfs_val = self.evaluation_results[cls][var_name]['GFS'][metric]
                    
                    if np.isnan(ec_val) or np.isnan(gfs_val):
                        valid_data = False
                        break
                    
                    temp_ec_metrics[metric] = ec_val
                    temp_gfs_metrics[metric] = gfs_val
                
                if valid_data:
                    for metric in ['R2', 'RMSE', 'MAE', 'BIAS', 'CORR']:
                        ec_metrics[metric].append(temp_ec_metrics[metric])
                        gfs_metrics[metric].append(temp_gfs_metrics[metric])
                    valid_classes.append(cls.replace('_', '\n'))
        
        if not valid_classes:
            fig.text(0.5, 0.5, f'{var_info["name"]}暂无有效数据', 
                    ha='center', va='center', fontsize=20)
            plt.savefig(f"{self.save_path}/02_{var_name}_performance.png", dpi=300, bbox_inches='tight')
            plt.show()
            return
        
        # 1. R²对比
        self._plot_metric_comparison(axes[0, 0], valid_classes, ec_metrics['R2'], gfs_metrics['R2'], 
                                   'R²决定系数对比', 'R² Score', ylim=(0, 1))
        
        # 2. RMSE对比
        self._plot_metric_comparison(axes[0, 1], valid_classes, ec_metrics['RMSE'], gfs_metrics['RMSE'], 
                                   f'均方根误差对比 ({var_info["unit"]})', 'RMSE')
        
        # 3. MAE对比
        self._plot_metric_comparison(axes[0, 2], valid_classes, ec_metrics['MAE'], gfs_metrics['MAE'], 
                                   f'平均绝对误差对比 ({var_info["unit"]})', 'MAE')
        
        # 4. 偏差对比
        self._plot_metric_comparison(axes[1, 0], valid_classes, ec_metrics['BIAS'], gfs_metrics['BIAS'], 
                                   f'平均偏差对比 ({var_info["unit"]})', 'BIAS', add_zero_line=True)
        
        # 5. 相关系数对比
        self._plot_metric_comparison(axes[1, 1], valid_classes, ec_metrics['CORR'], gfs_metrics['CORR'], 
                                   '相关系数对比', '相关系数', ylim=(0, 1))
        
        # 6. 综合性能差异
        self._plot_performance_difference(axes[1, 2], valid_classes, ec_metrics, gfs_metrics, var_info['name'])
        
        plt.tight_layout()
        plt.savefig(f"{self.save_path}/02_{var_name}_performance.png", dpi=300, bbox_inches='tight')
        plt.show()
    
    def _plot_metric_comparison(self, ax, classes, ec_values, gfs_values, title, ylabel, ylim=None, add_zero_line=False):
        """绘制指标对比条形图"""
        x = np.arange(len(classes))
        width = 0.35
        
        bars1 = ax.bar(x - width/2, ec_values, width, label='EC', color='blue', alpha=0.8)
        bars2 = ax.bar(x + width/2, gfs_values, width, label='GFS', color='red', alpha=0.8)
        
        ax.set_xlabel('分类')
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.set_xticks(x)
        ax.set_xticklabels(classes, rotation=45, ha='right')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        if ylim:
            ax.set_ylim(ylim)
        
        if add_zero_line:
            ax.axhline(y=0, color='black', linestyle='-', alpha=0.5)
        
        # 添加数值标签
        for bars in [bars1, bars2]:
            for bar in bars:
                height = bar.get_height()
                if not np.isnan(height):
                    ax.text(bar.get_x() + bar.get_width()/2., height + (ax.get_ylim()[1] - ax.get_ylim()[0])*0.01,
                           f'{height:.3f}', ha='center', va='bottom', fontsize=8)
    
    def _plot_performance_difference(self, ax, classes, ec_metrics, gfs_metrics, var_name):
        """绘制综合性能差异图"""
        metrics_diff = []
        metric_labels = ['R²', 'RMSE', 'MAE', '偏差', '相关系数']
        
        for metric in ['R2', 'RMSE', 'MAE', 'BIAS', 'CORR']:
            if ec_metrics[metric] and gfs_metrics[metric]:
                if metric in ['RMSE', 'MAE']:  # 对于误差指标，值越小越好
                    diff = np.mean(gfs_metrics[metric]) - np.mean(ec_metrics[metric])
                elif metric == 'BIAS':  # 对于偏差，绝对值越小越好
                    diff = np.mean(np.abs(gfs_metrics[metric])) - np.mean(np.abs(ec_metrics[metric]))
                else:  # 对于R²和相关系数，值越大越好
                    diff = np.mean(ec_metrics[metric]) - np.mean(gfs_metrics[metric])
                metrics_diff.append(diff)
            else:
                metrics_diff.append(0)
        
        # 绘制条形图显示性能差异
        colors = ['green' if x > 0 else 'red' for x in metrics_diff]
        bars = ax.bar(metric_labels, metrics_diff, color=colors, alpha=0.7)
        ax.set_ylabel('性能差异 (EC相对GFS)')
        ax.set_title(f'{var_name}平均性能差异\n(正值=EC更好)')
        ax.axhline(y=0, color='black', linestyle='-', alpha=0.5)
        ax.grid(True, alpha=0.3)
        
        # 添加数值标签
        for bar, value in zip(bars, metrics_diff):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., 
                   height + (0.001 if height > 0 else -0.001),
                   f'{value:.3f}', ha='center', 
                   va='bottom' if height > 0 else 'top', 
                   fontsize=9, fontweight='bold')
    
    def plot_variable_scatter_analysis(self):
        """绘制变量散点图分析"""
        print("📊 绘制变量散点图分析...")
        
        if not self.evaluation_results:
            print("❌ 无评估结果可绘制")
            return
        
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle('变量预报性能散点图分析', fontsize=16, fontweight='bold')
        
        # 1. EC vs GFS R²散点图
        ax1 = axes[0, 0]
        self._plot_r2_scatter(ax1, 'EC vs GFS R²性能对比')
        
        # 2. RMSE vs R²关系
        ax2 = axes[0, 1]
        self._plot_rmse_r2_relationship(ax2, 'RMSE与R²关系')
        
        # 3. 偏差分布
        ax3 = axes[1, 0]
        self._plot_bias_distribution(ax3, '预报偏差分布')
        
        # 4. 相关系数箱线图
        ax4 = axes[1, 1]
        self._plot_correlation_boxplot(ax4, '相关系数分布对比')
        
        plt.tight_layout()
        plt.savefig(f"{self.save_path}/02_variable_scatter_analysis.png", dpi=300, bbox_inches='tight')
        plt.show()
    
    def _plot_r2_scatter(self, ax, title):
        """绘制EC vs GFS R²散点图"""
        ec_r2 = []
        gfs_r2 = []
        labels = []
        
        for cls in self.evaluation_results.keys():
            for var in self.evaluation_results[cls]:
                ec_val = self.evaluation_results[cls][var]['EC']['R2']
                gfs_val = self.evaluation_results[cls][var]['GFS']['R2']
                
                if not (np.isnan(ec_val) or np.isnan(gfs_val)):
                    ec_r2.append(ec_val)
                    gfs_r2.append(gfs_val)
                    var_name = self.key_variables[var]['name'] if var in self.key_variables else var
                    labels.append(f"{cls}_{var_name}")
        
        if ec_r2 and gfs_r2:
            scatter = ax.scatter(ec_r2, gfs_r2, alpha=0.7, s=60)
            
            # 添加对角线
            min_r2 = min(min(ec_r2), min(gfs_r2))
            max_r2 = max(max(ec_r2), max(gfs_r2))
            ax.plot([min_r2, max_r2], [min_r2, max_r2], 'r--', alpha=0.5, label='y=x线')
            
            ax.set_xlabel('EC R²')
            ax.set_ylabel('GFS R²')
            ax.set_title(title)
            ax.grid(True, alpha=0.3)
            ax.legend()
            
            # 添加统计信息
            better_ec = sum(1 for i in range(len(ec_r2)) if ec_r2[i] > gfs_r2[i])
            ax.text(0.02, 0.98, f'EC更好: {better_ec}/{len(ec_r2)}', 
                   transform=ax.transAxes, va='top', fontsize=10,
                   bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        else:
            ax.text(0.5, 0.5, '无R²数据', ha='center', va='center', transform=ax.transAxes)
            ax.set_title(title)
    
    def _plot_rmse_r2_relationship(self, ax, title):
        """绘制RMSE与R²关系图"""
        rmse_values = []
        r2_values = []
        model_types = []
        
        for cls in self.evaluation_results.keys():
            for var in self.evaluation_results[cls]:
                for model in ['EC', 'GFS']:
                    rmse = self.evaluation_results[cls][var][model]['RMSE']
                    r2 = self.evaluation_results[cls][var][model]['R2']
                    
                    if not (np.isnan(rmse) or np.isnan(r2)):
                        rmse_values.append(rmse)
                        r2_values.append(r2)
                        model_types.append(model)
        
        if rmse_values and r2_values:
            # 按模型类型绘制
            ec_mask = [m == 'EC' for m in model_types]
            gfs_mask = [m == 'GFS' for m in model_types]
            
            if any(ec_mask):
                ax.scatter([r2_values[i] for i in range(len(r2_values)) if ec_mask[i]], 
                          [rmse_values[i] for i in range(len(rmse_values)) if ec_mask[i]], 
                          c='blue', alpha=0.7, label='EC', s=50)
            
            if any(gfs_mask):
                ax.scatter([r2_values[i] for i in range(len(r2_values)) if gfs_mask[i]], 
                          [rmse_values[i] for i in range(len(rmse_values)) if gfs_mask[i]], 
                          c='red', alpha=0.7, label='GFS', s=50)
            
            ax.set_xlabel('R² Score')
            ax.set_ylabel('RMSE')
            ax.set_title(title)
            ax.legend()
            ax.grid(True, alpha=0.3)
        else:
            ax.text(0.5, 0.5, '无RMSE-R²数据', ha='center', va='center', transform=ax.transAxes)
            ax.set_title(title)
    
    def _plot_bias_distribution(self, ax, title):
        """绘制偏差分布图"""
        ec_bias = []
        gfs_bias = []
        
        for cls in self.evaluation_results.keys():
            for var in self.evaluation_results[cls]:
                ec_b = self.evaluation_results[cls][var]['EC']['BIAS']
                gfs_b = self.evaluation_results[cls][var]['GFS']['BIAS']
                
                if not np.isnan(ec_b):
                    ec_bias.append(ec_b)
                if not np.isnan(gfs_b):
                    gfs_bias.append(gfs_b)
        
        if ec_bias or gfs_bias:
            if ec_bias:
                ax.hist(ec_bias, bins=15, alpha=0.7, label='EC', color='blue', density=True)
            if gfs_bias:
                ax.hist(gfs_bias, bins=15, alpha=0.7, label='GFS', color='red', density=True)
            
            ax.axvline(0, color='black', linestyle='--', alpha=0.5, label='无偏差线')
            ax.set_xlabel('预报偏差')
            ax.set_ylabel('密度')
            ax.set_title(title)
            ax.legend()
            ax.grid(True, alpha=0.3)
            
            # 添加统计信息
            if ec_bias and gfs_bias:
                ax.text(0.02, 0.98, f'EC均值: {np.mean(ec_bias):.3f}\nGFS均值: {np.mean(gfs_bias):.3f}', 
                       transform=ax.transAxes, va='top', fontsize=10,
                       bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        else:
            ax.text(0.5, 0.5, '无偏差数据', ha='center', va='center', transform=ax.transAxes)
            ax.set_title(title)
    
    def _plot_correlation_boxplot(self, ax, title):
        """绘制相关系数箱线图"""
        ec_corr = []
        gfs_corr = []
        
        for cls in self.evaluation_results.keys():
            for var in self.evaluation_results[cls]:
                ec_c = self.evaluation_results[cls][var]['EC']['CORR']
                gfs_c = self.evaluation_results[cls][var]['GFS']['CORR']
                
                if not np.isnan(ec_c):
                    ec_corr.append(ec_c)
                if not np.isnan(gfs_c):
                    gfs_corr.append(gfs_c)
        
        if ec_corr or gfs_corr:
            data_to_plot = []
            labels = []
            
            if ec_corr:
                data_to_plot.append(ec_corr)
                labels.append('EC')
            if gfs_corr:
                data_to_plot.append(gfs_corr)
                labels.append('GFS')
            
            bp = ax.boxplot(data_to_plot, labels=labels, patch_artist=True)
            
            # 设置颜色
            colors = ['lightblue', 'lightcoral']
            for patch, color in zip(bp['boxes'], colors[:len(bp['boxes'])]):
                patch.set_facecolor(color)
            
            ax.set_ylabel('相关系数')
            ax.set_title(title)
            ax.grid(True, alpha=0.3)
            ax.set_ylim(0, 1)
            
            # 添加统计信息
            if ec_corr and gfs_corr:
                ax.text(0.02, 0.98, f'EC均值: {np.mean(ec_corr):.3f}\nGFS均值: {np.mean(gfs_corr):.3f}', 
                       transform=ax.transAxes, va='top', fontsize=10,
                       bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        else:
            ax.text(0.5, 0.5, '无相关性数据', ha='center', va='center', transform=ax.transAxes)
            ax.set_title(title)

if __name__ == "__main__":
    # 测试变量性能分析功能
    import sys
    import os
    sys.path.append(os.path.dirname(os.path.abspath(__file__)))
    
    from nwp_evaluator_base import NWPEvaluatorBase
    
    DATA_PATH = "/Users/xiaxin/work/WindForecast_Project/01_Data/processed/imputed_data/changma_imputed_complete.csv"
    SAVE_PATH = "/Users/xiaxin/work/WindForecast_Project/03_Results/clear_nwp_evaluation_results"
    
    # 创建评估器
    evaluator = NWPEvaluatorBase(DATA_PATH, SAVE_PATH)
    
    # 加载数据和评估
    if evaluator.load_and_prepare_data() is not None:
        evaluator.evaluate_by_classification()
        
        if evaluator.evaluation_results:
            # 创建变量性能分析器
            plotter = NWPVariablePerformance(evaluator)
            
            # 绘制变量性能图
            plotter.plot_variable_performance_detailed()
            plotter.plot_variable_scatter_analysis()
            print("✓ 变量性能分析完成")
        else:
            print("❌ 无评估结果")
    else:
        print("❌ 数据加载失败")