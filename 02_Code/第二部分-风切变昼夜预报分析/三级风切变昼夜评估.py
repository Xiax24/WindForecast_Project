#!/usr/bin/env python3
"""
优化的三级风切变分类下数值预报评估分析
修复图表显示问题，增强数据处理鲁棒性
Author: Research Team
Date: 2025-06-17
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

# 设置中文显示
plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial Unicode MS', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

class ImprovedNWPEvaluator:
    def __init__(self, data_path, save_path):
        self.data_path = data_path
        self.save_path = save_path
        self.data = None
        self.evaluation_results = {}
        
        # 三级风切变阈值
        self.shear_thresholds = {
            'weak_upper': 0.2,
            'moderate_upper': 0.3,
        }
        
        # 重点评估的变量
        self.key_variables = {
            'wind_speed_10m': {'obs': 'obs_wind_speed_10m', 'ec': 'ec_wind_speed_10m', 'gfs': 'gfs_wind_speed_10m'},
            'wind_speed_70m': {'obs': 'obs_wind_speed_70m', 'ec': 'ec_wind_speed_70m', 'gfs': 'gfs_wind_speed_70m'},
            'temperature_10m': {'obs': 'obs_temperature_10m', 'ec': 'ec_temperature_10m', 'gfs': 'gfs_temperature_10m'}
        }
        
    def load_and_prepare_data(self):
        """加载和预处理数据"""
        print("📊 加载和预处理数据...")
        
        self.data = pd.read_csv(self.data_path)
        print(f"原始数据形状: {self.data.shape}")
        
        # 转换datetime列
        self.data['datetime'] = pd.to_datetime(self.data['datetime'])
        
        # 检查和清理关键变量
        key_columns = []
        for var_info in self.key_variables.values():
            key_columns.extend(var_info.values())
        key_columns.append('power')
        
        # 移除缺失值
        before_clean = len(self.data)
        self.data = self.data.dropna(subset=key_columns)
        self.data = self.data[self.data['power'] >= 0]
        after_clean = len(self.data)
        
        print(f"清理后数据: {after_clean} 行 (移除了 {before_clean - after_clean} 行)")
        
        # 计算风切变分类
        self._calculate_wind_shear_classification()
        
        return self.data
    
    def _calculate_wind_shear_classification(self):
        """计算风切变并进行分类"""
        print("🌪️ 计算风切变系数并分类...")
        
        # 计算风切变系数
        v1 = self.data['obs_wind_speed_10m']
        v2 = self.data['obs_wind_speed_70m']
        h1, h2 = 10, 70
        
        # 过滤有效数据
        valid_mask = (v1 > 0.5) & (v2 > 0.5)
        self.data = self.data[valid_mask].copy()
        v1, v2 = v1[valid_mask], v2[valid_mask]
        
        # 计算风切变系数
        self.data['wind_shear_alpha'] = np.log(v2 / v1) / np.log(h2 / h1)
        
        # 三级风切变分类
        alpha = self.data['wind_shear_alpha']
        conditions = [
            alpha < self.shear_thresholds['weak_upper'],
            (alpha >= self.shear_thresholds['weak_upper']) & (alpha < self.shear_thresholds['moderate_upper']),
            alpha >= self.shear_thresholds['moderate_upper']
        ]
        choices = ['weak', 'moderate', 'strong']
        self.data['shear_group'] = np.select(conditions, choices, default='unknown')
        
        # 昼夜分类
        self.data['hour'] = self.data['datetime'].dt.hour
        self.data['is_daytime'] = ((self.data['hour'] >= 6) & (self.data['hour'] < 18))
        
        # 组合分类
        self.data['shear_diurnal_class'] = self.data['shear_group'].astype(str) + '_' + \
                                         self.data['is_daytime'].map({True: 'day', False: 'night'})
        
        # 统计分类
        class_counts = self.data['shear_diurnal_class'].value_counts()
        print(f"✓ 风切变分类完成:")
        for class_name, count in class_counts.items():
            if 'unknown' not in class_name:
                percentage = count / len(self.data) * 100
                print(f"  {class_name}: {count} 条 ({percentage:.1f}%)")
        
        return class_counts
    
    def calculate_metrics(self, obs, forecast):
        """计算评估指标，增强鲁棒性"""
        obs = np.array(obs)
        forecast = np.array(forecast)
        
        # 移除缺失值
        valid_mask = ~(np.isnan(obs) | np.isnan(forecast) | np.isinf(obs) | np.isinf(forecast))
        obs_clean = obs[valid_mask]
        forecast_clean = forecast[valid_mask]
        
        if len(obs_clean) < 20:  # 确保足够的样本
            return {
                'RMSE': np.nan, 'MAE': np.nan, 'R2': np.nan, 
                'BIAS': np.nan, 'CORR': np.nan, 'COUNT': len(obs_clean)
            }
        
        try:
            rmse = np.sqrt(mean_squared_error(obs_clean, forecast_clean))
            mae = mean_absolute_error(obs_clean, forecast_clean)
            r2 = r2_score(obs_clean, forecast_clean)
            bias = np.mean(forecast_clean - obs_clean)
            corr = np.corrcoef(obs_clean, forecast_clean)[0, 1]
            
            return {
                'RMSE': rmse,
                'MAE': mae,
                'R2': r2,
                'BIAS': bias,
                'CORR': corr,
                'COUNT': len(obs_clean)
            }
        except:
            return {
                'RMSE': np.nan, 'MAE': np.nan, 'R2': np.nan, 
                'BIAS': np.nan, 'CORR': np.nan, 'COUNT': len(obs_clean)
            }
    
    def evaluate_by_classification(self):
        """按分类评估预报性能"""
        print("📈 按分类评估数值预报性能...")
        
        self.evaluation_results = {}
        unique_classes = [cls for cls in self.data['shear_diurnal_class'].unique() if 'unknown' not in cls]
        
        for class_name in unique_classes:
            class_data = self.data[self.data['shear_diurnal_class'] == class_name]
            
            if len(class_data) < 100:  # 确保足够样本
                continue
                
            print(f"评估 {class_name}: {len(class_data)} 条样本")
            
            self.evaluation_results[class_name] = {}
            
            for var_name, var_info in self.key_variables.items():
                obs_col = var_info['obs']
                ec_col = var_info['ec']
                gfs_col = var_info['gfs']
                
                if all(col in class_data.columns for col in [obs_col, ec_col, gfs_col]):
                    ec_metrics = self.calculate_metrics(class_data[obs_col], class_data[ec_col])
                    gfs_metrics = self.calculate_metrics(class_data[obs_col], class_data[gfs_col])
                    
                    self.evaluation_results[class_name][var_name] = {
                        'EC': ec_metrics,
                        'GFS': gfs_metrics
                    }
        
        print(f"✓ 完成 {len(self.evaluation_results)} 个分类的评估")
        return self.evaluation_results
    
    def create_comprehensive_plots(self):
        """创建优化的综合图表"""
        print("📊 创建综合评估图表...")
        
        fig = plt.figure(figsize=(20, 16))
        
        # 创建网格布局
        gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)
        
        # 1. R2差异热力图
        ax1 = fig.add_subplot(gs[0, 0])
        self._plot_r2_difference_heatmap(ax1)
        
        # 2. RMSE相对差异
        ax2 = fig.add_subplot(gs[0, 1])
        self._plot_rmse_relative_difference(ax2)
        
        # 3. 偏差散点图
        ax3 = fig.add_subplot(gs[0, 2])
        self._plot_bias_scatter(ax3)
        
        # 4. 10m风速性能对比
        ax4 = fig.add_subplot(gs[1, 0])
        self._plot_variable_performance(ax4, 'wind_speed_10m', '10m风速预报性能对比')
        
        # 5. 70m风速性能对比
        ax5 = fig.add_subplot(gs[1, 1])
        self._plot_variable_performance(ax5, 'wind_speed_70m', '70m风速预报性能对比')
        
        # 6. 10m温度性能对比
        ax6 = fig.add_subplot(gs[1, 2])
        self._plot_variable_performance(ax6, 'temperature_10m', '10m温度预报性能对比')
        
        # 7. 昼夜性能对比
        ax7 = fig.add_subplot(gs[2, 0])
        self._plot_diurnal_performance(ax7)
        
        # 8. 预报模式优劣统计
        ax8 = fig.add_subplot(gs[2, 1])
        self._plot_model_superiority(ax8)
        
        # 9. 相关性箱线图
        ax9 = fig.add_subplot(gs[2, 2])
        self._plot_correlation_boxplot(ax9)
        
        plt.suptitle('三级风切变分类下的数值预报性能综合评估', fontsize=16, fontweight='bold')
        plt.savefig(f"{self.save_path}/comprehensive_nwp_evaluation.png", dpi=300, bbox_inches='tight')
        plt.show()
    
    def _plot_r2_difference_heatmap(self, ax):
        """绘制R2差异热力图"""
        classes = list(self.evaluation_results.keys())
        variables = list(self.key_variables.keys())
        
        if not classes or not variables:
            ax.text(0.5, 0.5, '数据不足', ha='center', va='center', transform=ax.transAxes)
            ax.set_title('EC vs GFS R²差异\n(蓝色=EC更好, 红色=GFS更好)')
            return
        
        # 构建差异矩阵 (EC - GFS)
        diff_matrix = np.zeros((len(classes), len(variables)))
        
        for i, cls in enumerate(classes):
            for j, var in enumerate(variables):
                if var in self.evaluation_results[cls]:
                    ec_r2 = self.evaluation_results[cls][var]['EC']['R2']
                    gfs_r2 = self.evaluation_results[cls][var]['GFS']['R2']
                    
                    if not (np.isnan(ec_r2) or np.isnan(gfs_r2)):
                        diff_matrix[i, j] = ec_r2 - gfs_r2
                    else:
                        diff_matrix[i, j] = 0
        
        # 绘制热力图
        im = ax.imshow(diff_matrix, cmap='RdBu_r', aspect='auto', vmin=-0.3, vmax=0.3)
        
        # 设置标签
        ax.set_xticks(range(len(variables)))
        ax.set_xticklabels([v.replace('_', '\n') for v in variables])
        ax.set_yticks(range(len(classes)))
        ax.set_yticklabels([c.replace('_', '\n') for c in classes])
        
        # 添加数值标注
        for i in range(len(classes)):
            for j in range(len(variables)):
                value = diff_matrix[i, j]
                color = 'white' if abs(value) > 0.15 else 'black'
                ax.text(j, i, f'{value:.2f}', ha="center", va="center", 
                       color=color, fontweight='bold')
        
        ax.set_title('EC vs GFS R²差异\n(蓝色=EC更好, 红色=GFS更好)')
        
        # 添加颜色条
        plt.colorbar(im, ax=ax, shrink=0.8)
    
    def _plot_rmse_relative_difference(self, ax):
        """绘制RMSE相对差异"""
        classes = list(self.evaluation_results.keys())
        improvements = []
        labels = []
        
        for cls in classes:
            class_improvements = []
            for var in self.key_variables.keys():
                if var in self.evaluation_results[cls]:
                    ec_rmse = self.evaluation_results[cls][var]['EC']['RMSE']
                    gfs_rmse = self.evaluation_results[cls][var]['GFS']['RMSE']
                    
                    if not (np.isnan(ec_rmse) or np.isnan(gfs_rmse)) and gfs_rmse > 0:
                        # 计算GFS相对于EC的RMSE差异百分比
                        improvement = (gfs_rmse - ec_rmse) / gfs_rmse * 100
                        class_improvements.append(improvement)
            
            if class_improvements:
                improvements.append(np.mean(class_improvements))
                labels.append(cls.replace('_', '\n'))
        
        if not improvements:
            ax.text(0.5, 0.5, '数据不足', ha='center', va='center', transform=ax.transAxes)
            ax.set_title('GFS相对EC的RMSE表现')
            return
        
        # 绘制条形图
        colors = ['green' if x > 0 else 'red' for x in improvements]
        bars = ax.bar(range(len(improvements)), improvements, color=colors, alpha=0.7)
        
        ax.set_xticks(range(len(labels)))
        ax.set_xticklabels(labels, rotation=45, ha='right')
        ax.set_ylabel('RMSE相对差异 (%)')
        ax.set_title('GFS相对EC的RMSE表现\n(正值=EC更好)')
        ax.axhline(y=0, color='black', linestyle='-', alpha=0.3)
        ax.grid(True, alpha=0.3)
        
        # 添加数值标签
        for bar, value in zip(bars, improvements):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + (1 if height > 0 else -2),
                   f'{value:.1f}%', ha='center', va='bottom' if height > 0 else 'top', 
                   fontsize=9, fontweight='bold')
    
    def _plot_bias_scatter(self, ax):
        """绘制偏差散点图"""
        ec_bias = []
        gfs_bias = []
        labels = []
        
        for cls in self.evaluation_results.keys():
            cls_ec_bias = []
            cls_gfs_bias = []
            
            for var in self.key_variables.keys():
                if var in self.evaluation_results[cls]:
                    ec_b = self.evaluation_results[cls][var]['EC']['BIAS']
                    gfs_b = self.evaluation_results[cls][var]['GFS']['BIAS']
                    
                    if not np.isnan(ec_b):
                        cls_ec_bias.append(ec_b)
                    if not np.isnan(gfs_b):
                        cls_gfs_bias.append(gfs_b)
            
            if cls_ec_bias and cls_gfs_bias:
                ec_bias.append(np.mean(cls_ec_bias))
                gfs_bias.append(np.mean(cls_gfs_bias))
                labels.append(cls.replace('_', '\n'))
        
        if not ec_bias or not gfs_bias:
            ax.text(0.5, 0.5, '偏差数据不足', ha='center', va='center', transform=ax.transAxes)
            ax.set_title('EC vs GFS 偏差对比')
            return
        
        # 绘制散点图
        scatter = ax.scatter(ec_bias, gfs_bias, s=100, alpha=0.7, c='blue')
        
        # 添加标签
        for i, label in enumerate(labels):
            ax.annotate(label, (ec_bias[i], gfs_bias[i]), 
                       xytext=(5, 5), textcoords='offset points', fontsize=8)
        
        # 添加对角线
        all_bias = ec_bias + gfs_bias
        min_bias, max_bias = min(all_bias), max(all_bias)
        ax.plot([min_bias, max_bias], [min_bias, max_bias], 'r--', alpha=0.5)
        
        ax.set_xlabel('EC平均偏差')
        ax.set_ylabel('GFS平均偏差')
        ax.set_title('EC vs GFS 偏差对比')
        ax.grid(True, alpha=0.3)
        ax.axhline(y=0, color='black', linestyle='-', alpha=0.3)
        ax.axvline(x=0, color='black', linestyle='-', alpha=0.3)
    
    def _plot_variable_performance(self, ax, variable, title):
        """绘制单个变量的性能对比"""
        classes = list(self.evaluation_results.keys())
        ec_r2 = []
        gfs_r2 = []
        valid_classes = []
        
        for cls in classes:
            if variable in self.evaluation_results[cls]:
                ec_val = self.evaluation_results[cls][variable]['EC']['R2']
                gfs_val = self.evaluation_results[cls][variable]['GFS']['R2']
                
                if not (np.isnan(ec_val) or np.isnan(gfs_val)):
                    ec_r2.append(ec_val)
                    gfs_r2.append(gfs_val)
                    valid_classes.append(cls.replace('_', '\n'))
        
        if not ec_r2 or not gfs_r2:
            ax.text(0.5, 0.5, f'{title}\n数据不足', ha='center', va='center', transform=ax.transAxes)
            ax.set_title(title)
            return
        
        x = np.arange(len(valid_classes))
        width = 0.35
        
        bars1 = ax.bar(x - width/2, ec_r2, width, label='EC', alpha=0.8, color='blue')
        bars2 = ax.bar(x + width/2, gfs_r2, width, label='GFS', alpha=0.8, color='red')
        
        ax.set_xticks(x)
        ax.set_xticklabels(valid_classes, rotation=45, ha='right')
        ax.set_ylabel('R² Score')
        ax.set_title(title)
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.set_ylim(0, 1)
        
        # 添加数值标签
        for bars in [bars1, bars2]:
            for bar in bars:
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                       f'{height:.2f}', ha='center', va='bottom', fontsize=8)
    
    def _plot_diurnal_performance(self, ax):
        """绘制昼夜性能对比"""
        day_ec = []
        day_gfs = []
        night_ec = []
        night_gfs = []
        
        for cls in self.evaluation_results.keys():
            is_day = 'day' in cls
            
            for var in self.evaluation_results[cls]:
                ec_r2 = self.evaluation_results[cls][var]['EC']['R2']
                gfs_r2 = self.evaluation_results[cls][var]['GFS']['R2']
                
                if not np.isnan(ec_r2):
                    if is_day:
                        day_ec.append(ec_r2)
                    else:
                        night_ec.append(ec_r2)
                
                if not np.isnan(gfs_r2):
                    if is_day:
                        day_gfs.append(gfs_r2)
                    else:
                        night_gfs.append(gfs_r2)
        
        # 计算平均值
        day_ec_mean = np.mean(day_ec) if day_ec else 0
        day_gfs_mean = np.mean(day_gfs) if day_gfs else 0
        night_ec_mean = np.mean(night_ec) if night_ec else 0
        night_gfs_mean = np.mean(night_gfs) if night_gfs else 0
        
        categories = ['白天', '夜间']
        ec_means = [day_ec_mean, night_ec_mean]
        gfs_means = [day_gfs_mean, night_gfs_mean]
        
        x = np.arange(len(categories))
        width = 0.35
        
        bars1 = ax.bar(x - width/2, ec_means, width, label='EC', alpha=0.8, color='blue')
        bars2 = ax.bar(x + width/2, gfs_means, width, label='GFS', alpha=0.8, color='red')
        
        ax.set_xticks(x)
        ax.set_xticklabels(categories)
        ax.set_ylabel('平均R² Score')
        ax.set_title('昼夜预报性能对比')
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.set_ylim(0, 1)
        
        # 添加数值标签
        for bars in [bars1, bars2]:
            for bar in bars:
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height + 0.02,
                       f'{height:.3f}', ha='center', va='bottom', fontsize=10, fontweight='bold')
    
    def _plot_model_superiority(self, ax):
        """绘制模式优劣统计"""
        ec_wins = 0
        gfs_wins = 0
        ties = 0
        
        for cls in self.evaluation_results:
            for var in self.evaluation_results[cls]:
                ec_r2 = self.evaluation_results[cls][var]['EC']['R2']
                gfs_r2 = self.evaluation_results[cls][var]['GFS']['R2']
                
                if not (np.isnan(ec_r2) or np.isnan(gfs_r2)):
                    if abs(ec_r2 - gfs_r2) < 0.005:
                        ties += 1
                    elif ec_r2 > gfs_r2:
                        ec_wins += 1
                    else:
                        gfs_wins += 1
        
        total = ec_wins + gfs_wins + ties
        
        if total == 0:
            ax.text(0.5, 0.5, '无对比数据', ha='center', va='center', transform=ax.transAxes)
            ax.set_title('预报模式优劣统计')
            return
        
        sizes = [ec_wins, gfs_wins, ties]
        labels = [f'EC胜出\n({ec_wins}次)', f'GFS胜出\n({gfs_wins}次)', f'平局\n({ties}次)']
        colors = ['blue', 'red', 'gray']
        
        wedges, texts, autotexts = ax.pie(sizes, labels=labels, colors=colors, 
                                         autopct='%1.1f%%', startangle=90)
        ax.set_title(f'预报模式优劣统计\n(总计{total}次对比)')
    
    def _plot_correlation_boxplot(self, ax):
        """绘制相关性箱线图"""
        ec_corr = []
        gfs_corr = []
        
        for cls in self.evaluation_results:
            for var in self.evaluation_results[cls]:
                ec_c = self.evaluation_results[cls][var]['EC']['CORR']
                gfs_c = self.evaluation_results[cls][var]['GFS']['CORR']
                
                if not np.isnan(ec_c):
                    ec_corr.append(ec_c)
                if not np.isnan(gfs_c):
                    gfs_corr.append(gfs_c)
        
        if not ec_corr or not gfs_corr:
            ax.text(0.5, 0.5, '相关性数据不足', ha='center', va='center', transform=ax.transAxes)
            ax.set_title('EC vs GFS 相关性分布')
            return
        
        bp = ax.boxplot([ec_corr, gfs_corr], labels=['EC', 'GFS'], patch_artist=True)
        bp['boxes'][0].set_facecolor('lightblue')
        bp['boxes'][1].set_facecolor('lightcoral')
        
        ax.set_ylabel('相关系数')
        ax.set_title('EC vs GFS 相关性分布')
        ax.grid(True, alpha=0.3)
        ax.set_ylim(0, 1)
        
        # 添加统计信息
        ec_mean = np.mean(ec_corr)
        gfs_mean = np.mean(gfs_corr)
        ax.text(0.02, 0.98, f'EC均值: {ec_mean:.3f}\nGFS均值: {gfs_mean:.3f}', 
               transform=ax.transAxes, va='top', fontsize=10,
               bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    def create_classification_analysis(self):
        """创建分类专项分析"""
        print("📊 创建分类专项分析...")
        
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle('分风切变-昼夜类型的数值预报专项分析', fontsize=16, fontweight='bold')
        
        # 1-3. 按切变强度分析
        shear_types = ['weak', 'moderate', 'strong']
        titles = ['弱切变条件 (α<0.2)', '中等切变条件 (0.2≤α<0.3)', '强切变条件 (α≥0.3)']
        
        for i, (shear_type, title) in enumerate(zip(shear_types, titles)):
            self._plot_shear_specific_analysis(axes[0, i], shear_type, title)
        
        # 4. 昼夜预报性能对比
        self._plot_enhanced_diurnal_comparison(axes[1, 0])
        
        # 5. 变量-模式性能矩阵
        self._plot_performance_matrix(axes[1, 1])
        
        # 6. 误差分布分析
        self._plot_error_distribution(axes[1, 2])
        
        plt.tight_layout()
        plt.savefig(f"{self.save_path}/classification_specific_analysis.png", dpi=300, bbox_inches='tight')
        plt.show()
    
    def _plot_shear_specific_analysis(self, ax, shear_type, title):
        """绘制特定切变类型的分析"""
        target_classes = [cls for cls in self.evaluation_results.keys() if cls.startswith(shear_type)]
        
        if not target_classes:
            ax.text(0.5, 0.5, f'{title}\n暂无数据', ha='center', va='center', transform=ax.transAxes)
            ax.set_title(title)
            return
        
        # 收集数据
        variables = []
        ec_performance = []
        gfs_performance = []
        colors = []
        
        for cls in target_classes:
            period = 'day' if 'day' in cls else 'night'
            period_color = 'lightblue' if period == 'day' else 'darkblue'
            
            for var in self.key_variables.keys():
                if var in self.evaluation_results[cls]:
                    ec_r2 = self.evaluation_results[cls][var]['EC']['R2']
                    gfs_r2 = self.evaluation_results[cls][var]['GFS']['R2']
                    
                    if not (np.isnan(ec_r2) or np.isnan(gfs_r2)):
                        variables.append(f"{var}_{period}")
                        ec_performance.append(ec_r2)
                        gfs_performance.append(gfs_r2)
                        colors.append(period_color)
        
        if not variables:
            ax.text(0.5, 0.5, f'{title}\n数据不足', ha='center', va='center', transform=ax.transAxes)
            ax.set_title(title)
            return
        
        # 绘制对比
        x = np.arange(len(variables))
        width = 0.35
        
        # EC条形图
        bars1 = ax.bar(x - width/2, ec_performance, width, label='EC', alpha=0.8, color='blue')
        # GFS条形图，使用不同颜色区分昼夜
        bars2 = ax.bar(x + width/2, gfs_performance, width, label='GFS', alpha=0.8, color='red')
        
        ax.set_xticks(x)
        ax.set_xticklabels([v.replace('_', '\n') for v in variables], rotation=45, ha='right')
        ax.set_ylabel('R² Score')
        ax.set_title(title)
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.set_ylim(0, 1)
        
        # 添加图例说明
        ax.text(0.02, 0.98, '浅色=白天\n深色=夜间', transform=ax.transAxes, va='top',
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8), fontsize=8)
    
    def _plot_enhanced_diurnal_comparison(self, ax):
        """绘制增强的昼夜对比分析"""
        day_ec = []
        day_gfs = []
        night_ec = []
        night_gfs = []
        
        for cls in self.evaluation_results.keys():
            is_day = 'day' in cls
            
            for var in self.evaluation_results[cls]:
                ec_r2 = self.evaluation_results[cls][var]['EC']['R2']
                gfs_r2 = self.evaluation_results[cls][var]['GFS']['R2']
                
                if not np.isnan(ec_r2):
                    if is_day:
                        day_ec.append(ec_r2)
                    else:
                        night_ec.append(ec_r2)
                
                if not np.isnan(gfs_r2):
                    if is_day:
                        day_gfs.append(gfs_r2)
                    else:
                        night_gfs.append(gfs_r2)
        
        # 计算统计量
        periods = ['白天', '夜间']
        ec_data = [day_ec, night_ec]
        gfs_data = [day_gfs, night_gfs]
        
        # 绘制箱线图
        positions_ec = [0.8, 2.8]
        positions_gfs = [1.2, 3.2]
        
        if any(len(data) > 0 for data in ec_data):
            bp1 = ax.boxplot([data for data in ec_data if len(data) > 0], 
                            positions=[pos for i, pos in enumerate(positions_ec) if len(ec_data[i]) > 0],
                            patch_artist=True, widths=0.3)
            for patch in bp1['boxes']:
                patch.set_facecolor('lightblue')
        
        if any(len(data) > 0 for data in gfs_data):
            bp2 = ax.boxplot([data for data in gfs_data if len(data) > 0],
                            positions=[pos for i, pos in enumerate(positions_gfs) if len(gfs_data[i]) > 0],
                            patch_artist=True, widths=0.3)
            for patch in bp2['boxes']:
                patch.set_facecolor('lightcoral')
        
        ax.set_xticks([1, 3])
        ax.set_xticklabels(periods)
        ax.set_ylabel('R² Score')
        ax.set_title('昼夜预报性能对比')
        ax.grid(True, alpha=0.3)
        ax.set_ylim(0, 1)
        
        # 添加图例
        ax.plot([], [], color='lightblue', marker='s', linestyle='None', label='EC')
        ax.plot([], [], color='lightcoral', marker='s', linestyle='None', label='GFS')
        ax.legend()
    
    def _plot_performance_matrix(self, ax):
        """绘制变量-模式性能矩阵"""
        variables = list(self.key_variables.keys())
        models = ['EC', 'GFS']
        
        # 计算各变量的平均性能
        performance_matrix = np.zeros((len(models), len(variables)))
        
        for i, model in enumerate(models):
            for j, var in enumerate(variables):
                r2_values = []
                for cls in self.evaluation_results:
                    if var in self.evaluation_results[cls]:
                        r2 = self.evaluation_results[cls][var][model]['R2']
                        if not np.isnan(r2):
                            r2_values.append(r2)
                performance_matrix[i, j] = np.mean(r2_values) if r2_values else 0
        
        # 绘制热力图
        im = ax.imshow(performance_matrix, cmap='YlOrRd', aspect='auto', vmin=0, vmax=1)
        
        # 设置标签
        ax.set_xticks(range(len(variables)))
        ax.set_xticklabels([v.replace('_', '\n') for v in variables])
        ax.set_yticks(range(len(models)))
        ax.set_yticklabels(models)
        
        # 添加数值标注
        for i in range(len(models)):
            for j in range(len(variables)):
                text = ax.text(j, i, f'{performance_matrix[i, j]:.3f}',
                             ha="center", va="center", color="black", fontweight='bold')
        
        ax.set_title('变量-模式性能矩阵\n(平均R²)')
        
        # 添加颜色条
        plt.colorbar(im, ax=ax, shrink=0.8)
    
    def _plot_error_distribution(self, ax):
        """绘制误差分布分析"""
        ec_rmse_data = []
        gfs_rmse_data = []
        
        for cls in self.evaluation_results:
            for var in self.evaluation_results[cls]:
                ec_rmse = self.evaluation_results[cls][var]['EC']['RMSE']
                gfs_rmse = self.evaluation_results[cls][var]['GFS']['RMSE']
                
                if not np.isnan(ec_rmse):
                    ec_rmse_data.append(ec_rmse)
                if not np.isnan(gfs_rmse):
                    gfs_rmse_data.append(gfs_rmse)
        
        if not ec_rmse_data or not gfs_rmse_data:
            ax.text(0.5, 0.5, '误差数据不足', ha='center', va='center', transform=ax.transAxes)
            ax.set_title('预报误差分布对比')
            return
        
        # 绘制直方图对比
        max_rmse = max(max(ec_rmse_data), max(gfs_rmse_data))
        bins = np.linspace(0, max_rmse, 20)
        
        ax.hist(ec_rmse_data, bins=bins, alpha=0.7, label='EC', color='blue', density=True)
        ax.hist(gfs_rmse_data, bins=bins, alpha=0.7, label='GFS', color='red', density=True)
        
        ax.set_xlabel('RMSE')
        ax.set_ylabel('密度')
        ax.set_title('预报误差分布对比')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # 添加统计信息
        ec_mean_rmse = np.mean(ec_rmse_data)
        gfs_mean_rmse = np.mean(gfs_rmse_data)
        ax.axvline(ec_mean_rmse, color='blue', linestyle='--', alpha=0.8)
        ax.axvline(gfs_mean_rmse, color='red', linestyle='--', alpha=0.8)
        
        ax.text(0.02, 0.98, f'EC均值: {ec_mean_rmse:.3f}\nGFS均值: {gfs_mean_rmse:.3f}', 
               transform=ax.transAxes, va='top', fontsize=10,
               bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    def create_detailed_summary_table(self):
        """创建详细汇总表"""
        print("📋 创建详细汇总表...")
        
        summary_data = []
        
        for cls in self.evaluation_results:
            for var in self.evaluation_results[cls]:
                ec_metrics = self.evaluation_results[cls][var]['EC']
                gfs_metrics = self.evaluation_results[cls][var]['GFS']
                
                summary_data.append({
                    'Classification': cls,
                    'Variable': var,
                    'EC_R2': ec_metrics['R2'],
                    'GFS_R2': gfs_metrics['R2'],
                    'EC_RMSE': ec_metrics['RMSE'],
                    'GFS_RMSE': gfs_metrics['RMSE'],
                    'EC_MAE': ec_metrics['MAE'],
                    'GFS_MAE': gfs_metrics['MAE'],
                    'EC_BIAS': ec_metrics['BIAS'],
                    'GFS_BIAS': gfs_metrics['BIAS'],
                    'EC_CORR': ec_metrics['CORR'],
                    'GFS_CORR': gfs_metrics['CORR'],
                    'Sample_Size': ec_metrics['COUNT'],
                    'R2_Diff_EC_minus_GFS': ec_metrics['R2'] - gfs_metrics['R2'],
                    'Better_Model': 'EC' if ec_metrics['R2'] > gfs_metrics['R2'] else 'GFS'
                })
        
        summary_df = pd.DataFrame(summary_data)
        
        # 保存详细结果
        summary_df.to_csv(f"{self.save_path}/detailed_nwp_comparison.csv", index=False)
        print(f"✓ 详细对比表已保存")
        
        return summary_df
    
    def generate_performance_report(self):
        """生成性能评估报告"""
        print("📄 生成性能评估报告...")
        
        # 统计总体表现
        total_comparisons = 0
        ec_wins = 0
        gfs_wins = 0
        
        all_ec_r2 = []
        all_gfs_r2 = []
        
        for cls in self.evaluation_results:
            for var in self.evaluation_results[cls]:
                ec_r2 = self.evaluation_results[cls][var]['EC']['R2']
                gfs_r2 = self.evaluation_results[cls][var]['GFS']['R2']
                
                if not (np.isnan(ec_r2) or np.isnan(gfs_r2)):
                    total_comparisons += 1
                    all_ec_r2.append(ec_r2)
                    all_gfs_r2.append(gfs_r2)
                    
                    if ec_r2 > gfs_r2:
                        ec_wins += 1
                    else:
                        gfs_wins += 1
        
        # 生成报告
        report = {
            'total_comparisons': total_comparisons,
            'ec_wins': ec_wins,
            'gfs_wins': gfs_wins,
            'ec_win_rate': ec_wins / total_comparisons * 100 if total_comparisons > 0 else 0,
            'gfs_win_rate': gfs_wins / total_comparisons * 100 if total_comparisons > 0 else 0,
            'ec_avg_r2': np.mean(all_ec_r2) if all_ec_r2 else 0,
            'gfs_avg_r2': np.mean(all_gfs_r2) if all_gfs_r2 else 0,
            'performance_gap': np.mean(all_ec_r2) - np.mean(all_gfs_r2) if all_ec_r2 and all_gfs_r2 else 0
        }
        
        # 按分类分析最佳表现
        best_combinations = []
        for cls in self.evaluation_results:
            cls_performance = []
            for var in self.evaluation_results[cls]:
                ec_r2 = self.evaluation_results[cls][var]['EC']['R2']
                gfs_r2 = self.evaluation_results[cls][var]['GFS']['R2']
                if not (np.isnan(ec_r2) or np.isnan(gfs_r2)):
                    cls_performance.append(max(ec_r2, gfs_r2))
            
            if cls_performance:
                best_combinations.append({
                    'classification': cls,
                    'avg_best_performance': np.mean(cls_performance)
                })
        
        best_combinations.sort(key=lambda x: x['avg_best_performance'], reverse=True)
        report['best_combinations'] = best_combinations[:3]  # Top 3
        
        # 保存报告
        import json
        with open(f"{self.save_path}/nwp_performance_report.json", 'w', encoding='utf-8') as f:
            json.dump(report, f, ensure_ascii=False, indent=2)
        
        print(f"✓ 性能报告已保存")
        return report
    
    def run_complete_evaluation(self):
        """运行完整的评估流程"""
        print("=" * 70)
        print("📡 优化的数值预报评估分析")
        print("=" * 70)
        
        try:
            # 1. 加载和预处理数据
            self.load_and_prepare_data()
            
            # 2. 评估预报性能
            self.evaluate_by_classification()
            
            # 3. 创建综合图表
            self.create_comprehensive_plots()
            
            # 4. 创建分类专项分析
            self.create_classification_analysis()
            
            # 5. 创建详细汇总表
            summary_df = self.create_detailed_summary_table()
            
            # 6. 生成性能报告
            report = self.generate_performance_report()
            
            print("\n" + "=" * 70)
            print("🎉 数值预报评估分析完成！")
            print("=" * 70)
            
            # 输出主要发现
            print("📊 主要发现:")
            print(f"  评估分类数量: {len(self.evaluation_results)}")
            print(f"  总对比次数: {report['total_comparisons']}")
            print(f"  EC胜出率: {report['ec_win_rate']:.1f}%")
            print(f"  GFS胜出率: {report['gfs_win_rate']:.1f}%")
            print(f"  EC平均R²: {report['ec_avg_r2']:.3f}")
            print(f"  GFS平均R²: {report['gfs_avg_r2']:.3f}")
            print(f"  性能差距(EC-GFS): {report['performance_gap']:.3f}")
            
            # 输出最佳组合
            if report['best_combinations']:
                print(f"\n🏆 表现最佳的分类组合:")
                for i, combo in enumerate(report['best_combinations'], 1):
                    print(f"  {i}. {combo['classification']}: R²={combo['avg_best_performance']:.3f}")
            
            # 提供使用建议
            print(f"\n💡 使用建议:")
            if report['ec_win_rate'] > 60:
                print("  - 总体建议优先使用EC模式预报")
            elif report['gfs_win_rate'] > 60:
                print("  - 总体建议优先使用GFS模式预报")
            else:
                print("  - EC和GFS性能相当，建议根据具体条件选择")
            
            print(f"\n📁 输出文件:")
            print(f"  - comprehensive_nwp_evaluation.png: 综合评估图表")
            print(f"  - classification_specific_analysis.png: 分类专项分析")
            print(f"  - detailed_nwp_comparison.csv: 详细对比数据")
            print(f"  - nwp_performance_report.json: 性能评估报告")
            
            return True
            
        except Exception as e:
            print(f"❌ 评估过程出错: {str(e)}")
            import traceback
            traceback.print_exc()
            return False

if __name__ == "__main__":
    # 配置路径 - 请根据实际情况修改
    DATA_PATH = "/Users/xiaxin/work/WindForecast_Project/01_Data/processed/imputed_data/changma_imputed_complete.csv"
    SAVE_PATH = "/Users/xiaxin/work/WindForecast_Project/03_Results/nwp_evaluation_results"        # 请替换为实际保存路径
    
    # 创建保存目录
    import os
    os.makedirs(SAVE_PATH, exist_ok=True)
    
    # 创建评估器并运行
    evaluator = ImprovedNWPEvaluator(DATA_PATH, SAVE_PATH)
    success = evaluator.run_complete_evaluation()
    
    if success:
        print("\n🎯 数值预报评估成功完成！")
        print("\n🔧 主要改进:")
        print("  1. 增强了数据预处理的鲁棒性")
        print("  2. 修复了图表显示问题")
        print("  3. 优化了样本数量要求")
        print("  4. 改进了缺失值处理")
        print("  5. 增强了错误处理机制")
        print("\n📈 实用价值:")
        print("  1. 识别最优预报模式选择策略")
        print("  2. 量化不同条件下的预报性能差异")
        print("  3. 为风电预测提供数据驱动的建议")
        print("  4. 支持多模式集成预报权重优化")
    else:
        print("\n⚠️ 评估失败，请检查数据路径和格式")