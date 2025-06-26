#!/usr/bin/env python3
"""
实用版数值预报评估器 - 增加显著性检验
重点关注：相关系数、偏差、RMSE + 样本数 + 显著性检验
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, mean_absolute_error
from scipy import stats
from scipy.stats import pearsonr, ttest_ind, ttest_rel, mannwhitneyu, wilcoxon
import os
import json
import warnings
warnings.filterwarnings('ignore')

# 设置英文显示
plt.rcParams['font.family'] = ['Arial', 'DejaVu Sans', 'Liberation Sans']
plt.rcParams['axes.unicode_minus'] = False

def practical_nwp_evaluation_with_significance(data_path, save_path):
    """实用版数值预报评估 - 增加显著性检验"""
    print("=" * 80)
    print("📡 实用版数值预报评估分析 - 含显著性检验")
    print("重点指标：相关系数、偏差、RMSE + 样本数 + 显著性检验")
    print("=" * 80)
    
    os.makedirs(save_path, exist_ok=True)
    
    # 配置
    shear_thresholds = {'weak_upper': 0.1, 'moderate_upper': 0.3}
    min_samples = 50
    min_valid_samples = 20
    alpha_level = 0.05  # 显著性水平
    
    key_variables = {
        'wind_speed_10m': {
            'obs': 'obs_wind_speed_10m', 'ec': 'ec_wind_speed_10m', 'gfs': 'gfs_wind_speed_10m', 
            'name': '10m Wind Speed', 'unit': 'm/s'
        },
        'wind_speed_70m': {
            'obs': 'obs_wind_speed_70m', 'ec': 'ec_wind_speed_70m', 'gfs': 'gfs_wind_speed_70m', 
            'name': '70m Wind Speed', 'unit': 'm/s'
        },
        'temperature_10m': {
            'obs': 'obs_temperature_10m', 'ec': 'ec_temperature_10m', 'gfs': 'gfs_temperature_10m', 
            'name': '10m Temperature', 'unit': '°C'
        }
    }
    
    def calculate_significance_tests(obs, ec_forecast, gfs_forecast):
        """计算显著性检验"""
        obs = np.array(obs)
        ec_forecast = np.array(ec_forecast)
        gfs_forecast = np.array(gfs_forecast)
        
        # 移除缺失值
        valid_mask = ~(np.isnan(obs) | np.isnan(ec_forecast) | np.isnan(gfs_forecast) | 
                      np.isinf(obs) | np.isinf(ec_forecast) | np.isinf(gfs_forecast))
        obs_clean = obs[valid_mask]
        ec_clean = ec_forecast[valid_mask]
        gfs_clean = gfs_forecast[valid_mask]
        
        if len(obs_clean) < min_valid_samples:
            return {
                'corr_ec_pvalue': np.nan, 'corr_gfs_pvalue': np.nan, 'corr_diff_pvalue': np.nan,
                'bias_ttest_pvalue': np.nan, 'rmse_diff_pvalue': np.nan,
                'ec_corr_significant': False, 'gfs_corr_significant': False,
                'corr_diff_significant': False, 'bias_diff_significant': False,
                'rmse_diff_significant': False
            }
        
        try:
            # 1. 相关系数的显著性检验
            ec_corr, ec_corr_pvalue = pearsonr(obs_clean, ec_clean)
            gfs_corr, gfs_corr_pvalue = pearsonr(obs_clean, gfs_clean)
            
            # 2. 相关系数差异的显著性检验 (Fisher's z-transformation)
            def fisher_z_test(r1, r2, n1, n2):
                """Fisher's z-transformation test for correlation difference"""
                if abs(r1) >= 1 or abs(r2) >= 1 or n1 < 3 or n2 < 3:
                    return np.nan
                
                z1 = 0.5 * np.log((1 + r1) / (1 - r1))
                z2 = 0.5 * np.log((1 + r2) / (1 - r2))
                se_diff = np.sqrt(1/(n1-3) + 1/(n2-3))
                z_stat = (z1 - z2) / se_diff
                p_value = 2 * (1 - stats.norm.cdf(abs(z_stat)))
                return p_value
            
            corr_diff_pvalue = fisher_z_test(ec_corr, gfs_corr, len(obs_clean), len(obs_clean))
            
            # 3. 偏差差异的显著性检验 (配对t检验)
            ec_errors = ec_clean - obs_clean
            gfs_errors = gfs_clean - obs_clean
            
            # 检验EC和GFS的误差是否有显著差异
            try:
                _, bias_diff_pvalue = ttest_rel(ec_errors, gfs_errors)
            except:
                bias_diff_pvalue = np.nan
            
            # 4. RMSE差异的显著性检验 (bootstrap方法)
            def bootstrap_rmse_test(obs, pred1, pred2, n_bootstrap=1000):
                """Bootstrap test for RMSE difference"""
                n_samples = len(obs)
                rmse1_orig = np.sqrt(mean_squared_error(obs, pred1))
                rmse2_orig = np.sqrt(mean_squared_error(obs, pred2))
                diff_orig = rmse1_orig - rmse2_orig
                
                diff_bootstrap = []
                for _ in range(n_bootstrap):
                    indices = np.random.choice(n_samples, n_samples, replace=True)
                    obs_boot = obs[indices]
                    pred1_boot = pred1[indices]
                    pred2_boot = pred2[indices]
                    
                    rmse1_boot = np.sqrt(mean_squared_error(obs_boot, pred1_boot))
                    rmse2_boot = np.sqrt(mean_squared_error(obs_boot, pred2_boot))
                    diff_bootstrap.append(rmse1_boot - rmse2_boot)
                
                # 计算p值 (双尾检验)
                diff_bootstrap = np.array(diff_bootstrap)
                p_value = 2 * min(np.mean(diff_bootstrap >= 0), np.mean(diff_bootstrap <= 0))
                return p_value
            
            rmse_diff_pvalue = bootstrap_rmse_test(obs_clean, ec_clean, gfs_clean)
            
            # 判断显著性
            results = {
                'corr_ec_pvalue': ec_corr_pvalue,
                'corr_gfs_pvalue': gfs_corr_pvalue,
                'corr_diff_pvalue': corr_diff_pvalue,
                'bias_ttest_pvalue': bias_diff_pvalue,
                'rmse_diff_pvalue': rmse_diff_pvalue,
                'ec_corr_significant': ec_corr_pvalue < alpha_level,
                'gfs_corr_significant': gfs_corr_pvalue < alpha_level,
                'corr_diff_significant': corr_diff_pvalue < alpha_level,
                'bias_diff_significant': bias_diff_pvalue < alpha_level,
                'rmse_diff_significant': rmse_diff_pvalue < alpha_level
            }
            
            return results
            
        except Exception as e:
            print(f"    显著性检验计算错误: {e}")
            return {
                'corr_ec_pvalue': np.nan, 'corr_gfs_pvalue': np.nan, 'corr_diff_pvalue': np.nan,
                'bias_ttest_pvalue': np.nan, 'rmse_diff_pvalue': np.nan,
                'ec_corr_significant': False, 'gfs_corr_significant': False,
                'corr_diff_significant': False, 'bias_diff_significant': False,
                'rmse_diff_significant': False
            }
    
    try:
        # 1. 数据加载和预处理（与原代码相同）
        print("\n🔄 步骤1: 加载数据")
        data = pd.read_csv(data_path)
        data['datetime'] = pd.to_datetime(data['datetime'])
        print(f"原始数据形状: {data.shape}")
        
        # 数据清理
        for var_name, var_info in key_variables.items():
            for col_type in ['obs', 'ec', 'gfs']:
                col = var_info[col_type]
                if col in data.columns:
                    if 'wind_speed' in col:
                        data[col] = data[col].where((data[col] >= 0) & (data[col] <= 50))
                    elif 'temperature' in col:
                        data[col] = data[col].where((data[col] >= -50) & (data[col] <= 60))
        
        # 风切变计算和分类
        print("\n🔄 步骤2: 风切变计算和分类")
        v1 = data['obs_wind_speed_10m']
        v2 = data['obs_wind_speed_70m']
        
        valid_wind_mask = (v1 > 0.5) & (v2 > 0.5) & (~v1.isna()) & (~v2.isna())
        data = data[valid_wind_mask].copy()
        
        data['wind_shear_alpha'] = np.log(data['obs_wind_speed_70m'] / data['obs_wind_speed_10m']) / np.log(70 / 10)
        
        alpha = data['wind_shear_alpha']
        valid_alpha = (~np.isnan(alpha)) & (~np.isinf(alpha)) & (alpha > -1) & (alpha < 2)
        data = data[valid_alpha].copy()
        
        # 分类
        alpha = data['wind_shear_alpha']
        conditions = [
            alpha < shear_thresholds['weak_upper'],
            (alpha >= shear_thresholds['weak_upper']) & (alpha < shear_thresholds['moderate_upper']),
            alpha >= shear_thresholds['moderate_upper']
        ]
        choices = ['weak', 'moderate', 'strong']
        data['shear_group'] = np.select(conditions, choices, default='unknown')
        
        data['hour'] = data['datetime'].dt.hour
        data['is_daytime'] = ((data['hour'] >= 6) & (data['hour'] < 18))
        data['shear_diurnal_class'] = data['shear_group'].astype(str) + '_' + \
                                     data['is_daytime'].map({True: 'day', False: 'night'})
        
        class_counts = data['shear_diurnal_class'].value_counts()
        print(f"Wind shear classification distribution:")
        for class_name, count in class_counts.items():
            if 'unknown' not in class_name:
                percentage = count / len(data) * 100
                print(f"  {class_name}: {count} samples ({percentage:.1f}%)")
        
        # 3. 计算实用评估指标（增强版）
        print("\n🔄 步骤3: 计算实用评估指标 + 显著性检验")
        
        def calculate_practical_metrics(obs, forecast):
            """计算实用指标：相关系数、偏差、RMSE"""
            obs = np.array(obs)
            forecast = np.array(forecast)
            valid_mask = ~(np.isnan(obs) | np.isnan(forecast) | np.isinf(obs) | np.isinf(forecast))
            obs_clean = obs[valid_mask]
            forecast_clean = forecast[valid_mask]
            
            if len(obs_clean) < min_valid_samples:
                return {'CORR': np.nan, 'BIAS': np.nan, 'RMSE': np.nan, 'MAE': np.nan, 'COUNT': len(obs_clean)}
            
            try:
                # 相关系数
                if len(obs_clean) > 1 and np.std(obs_clean) > 1e-10 and np.std(forecast_clean) > 1e-10:
                    corr = np.corrcoef(obs_clean, forecast_clean)[0, 1]
                else:
                    corr = np.nan
                
                # 偏差
                bias = np.mean(forecast_clean - obs_clean)
                
                # RMSE
                rmse = np.sqrt(mean_squared_error(obs_clean, forecast_clean))
                
                # MAE
                mae = mean_absolute_error(obs_clean, forecast_clean)
                
                return {'CORR': corr, 'BIAS': bias, 'RMSE': rmse, 'MAE': mae, 'COUNT': len(obs_clean)}
            except Exception as e:
                print(f"    计算指标错误: {e}")
                return {'CORR': np.nan, 'BIAS': np.nan, 'RMSE': np.nan, 'MAE': np.nan, 'COUNT': len(obs_clean)}
        
        evaluation_results = {}
        unique_classes = [cls for cls in data['shear_diurnal_class'].unique() if 'unknown' not in cls]
        
        print(f"Starting evaluation of {len(unique_classes)} classifications...")
        
        for class_name in unique_classes:
            class_data = data[data['shear_diurnal_class'] == class_name]
            
            if len(class_data) < min_samples:
                print(f"Skip {class_name}: insufficient samples ({len(class_data)} < {min_samples})")
                continue
                
            print(f"\nEvaluating {class_name}: {len(class_data)} samples")
            evaluation_results[class_name] = {}
            
            for var_name, var_info in key_variables.items():
                obs_col = var_info['obs']
                ec_col = var_info['ec']
                gfs_col = var_info['gfs']
                
                if all(col in class_data.columns for col in [obs_col, ec_col, gfs_col]):
                    var_valid = (~class_data[obs_col].isna()) & (~class_data[ec_col].isna()) & (~class_data[gfs_col].isna())
                    valid_count = var_valid.sum()
                    
                    if valid_count >= min_valid_samples:
                        # 计算基础指标
                        ec_metrics = calculate_practical_metrics(class_data[obs_col], class_data[ec_col])
                        gfs_metrics = calculate_practical_metrics(class_data[obs_col], class_data[gfs_col])
                        
                        # 计算显著性检验
                        sig_tests = calculate_significance_tests(
                            class_data[obs_col], class_data[ec_col], class_data[gfs_col]
                        )
                        
                        evaluation_results[class_name][var_name] = {
                            'EC': ec_metrics,
                            'GFS': gfs_metrics,
                            'SIGNIFICANCE': sig_tests
                        }
                        
                        # 显示关键指标 + 显著性
                        print(f"  {var_info['name']} (N={valid_count}):")
                        print(f"    EC:  CORR={ec_metrics['CORR']:.3f}{'*' if sig_tests['ec_corr_significant'] else ''} "
                              f"BIAS={ec_metrics['BIAS']:+.3f} RMSE={ec_metrics['RMSE']:.3f}")
                        print(f"    GFS: CORR={gfs_metrics['CORR']:.3f}{'*' if sig_tests['gfs_corr_significant'] else ''} "
                              f"BIAS={gfs_metrics['BIAS']:+.3f} RMSE={gfs_metrics['RMSE']:.3f}")
                        
                        # 显著性检验结果
                        sig_symbols = []
                        if sig_tests['corr_diff_significant']:
                            sig_symbols.append("CORR_DIFF*")
                        if sig_tests['bias_diff_significant']:
                            sig_symbols.append("BIAS_DIFF*")
                        if sig_tests['rmse_diff_significant']:
                            sig_symbols.append("RMSE_DIFF*")
                        
                        if sig_symbols:
                            print(f"    Significant differences: {', '.join(sig_symbols)}")
                        else:
                            print(f"    No significant differences detected")
                    else:
                        print(f"  {var_info['name']}: insufficient valid samples ({valid_count} < {min_valid_samples})")
        
        # 4. 创建增强的可视化图表
        print("\n🔄 步骤4: 创建增强可视化图表")
        
        # 设置整体样式
        plt.style.use('default')
        plt.rcParams['axes.facecolor'] = '#F8F9FA'
        plt.rcParams['figure.facecolor'] = 'white'
        plt.rcParams['axes.edgecolor'] = '#6C757D'
        plt.rcParams['axes.linewidth'] = 0.8
        plt.rcParams['grid.color'] = '#DEE2E6'
        plt.rcParams['grid.alpha'] = 0.7
        
        for var_name, var_info in key_variables.items():
            print(f"  Creating {var_info['name']} enhanced analysis chart...")
            
            fig, axes = plt.subplots(2, 2, figsize=(18, 12))
            fig.suptitle(f'{var_info["name"]} Forecast Performance Analysis (with Significance Tests)', 
                        fontsize=16, fontweight='bold')
            
            # 收集数据
            classes = []
            ec_corr = []
            gfs_corr = []
            ec_bias = []
            gfs_bias = []
            ec_rmse = []
            gfs_rmse = []
            sample_counts = []
            sig_corr_diff = []
            sig_bias_diff = []
            sig_rmse_diff = []
            
            for cls in evaluation_results.keys():
                if var_name in evaluation_results[cls]:
                    ec_metrics = evaluation_results[cls][var_name]['EC']
                    gfs_metrics = evaluation_results[cls][var_name]['GFS']
                    sig_tests = evaluation_results[cls][var_name]['SIGNIFICANCE']
                    
                    if not (np.isnan(ec_metrics['CORR']) or np.isnan(gfs_metrics['CORR'])):
                        classes.append(cls.replace('_', '\n'))
                        ec_corr.append(ec_metrics['CORR'])
                        gfs_corr.append(gfs_metrics['CORR'])
                        ec_bias.append(ec_metrics['BIAS'])
                        gfs_bias.append(gfs_metrics['BIAS'])
                        ec_rmse.append(ec_metrics['RMSE'])
                        gfs_rmse.append(gfs_metrics['RMSE'])
                        sample_counts.append(ec_metrics['COUNT'])
                        sig_corr_diff.append(sig_tests['corr_diff_significant'])
                        sig_bias_diff.append(sig_tests['bias_diff_significant'])
                        sig_rmse_diff.append(sig_tests['rmse_diff_significant'])
            
            if classes:
                x = np.arange(len(classes))
                width = 0.35
                
                # 1. 相关系数对比 + 显著性标记
                bars1 = axes[0, 0].bar(x - width/2, ec_corr, width, label='EC', color="#5F46FF", alpha=0.8)
                bars2 = axes[0, 0].bar(x + width/2, gfs_corr, width, label='GFS', color="#E38989", alpha=0.8)
                axes[0, 0].set_xlabel('Classification')
                axes[0, 0].set_ylabel('Correlation Coefficient')
                axes[0, 0].set_title('Correlation Coefficient Comparison')
                axes[0, 0].set_xticks(x)
                axes[0, 0].set_xticklabels(classes, rotation=45, ha='right')
                axes[0, 0].legend()
                axes[0, 0].grid(True, alpha=0.3)
                axes[0, 0].set_ylim(0, 1)
                
                # 添加数值标签、样本数和显著性标记
                for i, (bar1, bar2, count, sig) in enumerate(zip(bars1, bars2, sample_counts, sig_corr_diff)):
                    # 相关系数标签
                    axes[0, 0].text(bar1.get_x() + bar1.get_width()/2., bar1.get_height() + 0.02,
                                   f'{ec_corr[i]:.3f}', ha='center', va='bottom', fontsize=8, fontweight='bold')
                    axes[0, 0].text(bar2.get_x() + bar2.get_width()/2., bar2.get_height() + 0.02,
                                   f'{gfs_corr[i]:.3f}', ha='center', va='bottom', fontsize=8, fontweight='bold')
                    # 样本数标签
                    axes[0, 0].text(i, 0.05, f'N={count}', ha='center', va='bottom', 
                                   fontsize=9, fontweight='bold', color='white',
                                   bbox=dict(boxstyle='round,pad=0.3', facecolor="#0165F1FF", alpha=0.9))
                    # 显著性标记
                    if sig:
                        axes[0, 0].text(i, 0.95, '**', ha='center', va='center', 
                                       fontsize=16, fontweight='bold', color='red')
                
                # 2. 偏差对比 + 显著性标记
                bars1 = axes[0, 1].bar(x - width/2, ec_bias, width, label='EC', color="#30C1FF", alpha=0.8)
                bars2 = axes[0, 1].bar(x + width/2, gfs_bias, width, label='GFS', color="#CE6D7F", alpha=0.8)
                axes[0, 1].set_xlabel('Classification')
                axes[0, 1].set_ylabel(f'Bias ({var_info["unit"]})')
                axes[0, 1].set_title('Forecast Bias Comparison')
                axes[0, 1].set_xticks(x)
                axes[0, 1].set_xticklabels(classes, rotation=45, ha='right')
                axes[0, 1].legend()
                axes[0, 1].grid(True, alpha=0.3)
                axes[0, 1].axhline(y=0, color='black', linestyle='-', alpha=0.5)
                
                # 添加偏差数值标签和显著性标记
                y_range = max(ec_bias + gfs_bias) - min(ec_bias + gfs_bias)
                for i, (bar, value, sig) in enumerate(zip(bars1, ec_bias, sig_bias_diff)):
                    y_pos = value + y_range * 0.02 if value >= 0 else value - y_range * 0.02
                    axes[0, 1].text(bar.get_x() + bar.get_width()/2., y_pos,
                                   f'{value:+.2f}', ha='center', va='bottom' if value >= 0 else 'top', 
                                   fontsize=8, fontweight='bold')
                for i, (bar, value, sig) in enumerate(zip(bars2, gfs_bias, sig_bias_diff)):
                    y_pos = value + y_range * 0.02 if value >= 0 else value - y_range * 0.02
                    axes[0, 1].text(bar.get_x() + bar.get_width()/2., y_pos,
                                   f'{value:+.2f}', ha='center', va='bottom' if value >= 0 else 'top', 
                                   fontsize=8, fontweight='bold')
                    # 显著性标记
                    if sig:
                        max_val = max(max(ec_bias), max(gfs_bias))
                        axes[0, 1].text(i, max_val + y_range * 0.1, '**', ha='center', va='center', 
                                       fontsize=16, fontweight='bold', color='red')
                
                # 3. RMSE对比 + 显著性标记
                bars1 = axes[1, 0].bar(x - width/2, ec_rmse, width, label='EC', color="#F1F148B8", alpha=0.8)
                bars2 = axes[1, 0].bar(x + width/2, gfs_rmse, width, label='GFS', color="#BC89E3", alpha=0.8)
                axes[1, 0].set_xlabel('Classification')
                axes[1, 0].set_ylabel(f'RMSE ({var_info["unit"]})')
                axes[1, 0].set_title('Root Mean Square Error Comparison')
                axes[1, 0].set_xticks(x)
                axes[1, 0].set_xticklabels(classes, rotation=45, ha='right')
                axes[1, 0].legend()
                axes[1, 0].grid(True, alpha=0.3)
                
                # 添加RMSE数值标签和显著性标记
                max_rmse = max(ec_rmse + gfs_rmse)
                for i, (bar, value, sig) in enumerate(zip(bars1, ec_rmse, sig_rmse_diff)):
                    axes[1, 0].text(bar.get_x() + bar.get_width()/2., bar.get_height() + max_rmse * 0.01,
                                   f'{value:.2f}', ha='center', va='bottom', fontsize=8, fontweight='bold')
                    # 显著性标记
                    if sig:
                        axes[1, 0].text(i, max_rmse * 1.1, '**', ha='center', va='center', 
                                       fontsize=16, fontweight='bold', color='red')
                for bar, value in zip(bars2, gfs_rmse):
                    axes[1, 0].text(bar.get_x() + bar.get_width()/2., bar.get_height() + max_rmse * 0.01,
                                   f'{value:.2f}', ha='center', va='bottom', fontsize=8, fontweight='bold')
                
                # 4. 综合性能散点图
                axes[1, 1].scatter(ec_corr, ec_rmse, s=120, alpha=0.8, color='#46C7FF', label='EC', edgecolors='white', linewidth=1)
                axes[1, 1].scatter(gfs_corr, gfs_rmse, s=120, alpha=0.8, color='#E389D9', label='GFS', edgecolors='white', linewidth=1)
                
                # 添加分类标签
                for i, cls in enumerate(classes):
                    axes[1, 1].annotate(f'EC_{cls.replace(chr(10), "_")}', (ec_corr[i], ec_rmse[i]), 
                                       xytext=(5, 5), textcoords='offset points', fontsize=7, color="#46C7FF")
                    axes[1, 1].annotate(f'GFS_{cls.replace(chr(10), "_")}', (gfs_corr[i], gfs_rmse[i]), 
                                       xytext=(-5, -5), textcoords='offset points', fontsize=7, color="#E389D9")
                
                axes[1, 1].set_xlabel('Correlation Coefficient')
                axes[1, 1].set_ylabel(f'RMSE ({var_info["unit"]})')
                axes[1, 1].set_title('Forecast Quality Distribution\n(Bottom-right = High Corr & Low RMSE = Good)')
                axes[1, 1].legend()
                axes[1, 1].grid(True, alpha=0.3)
                axes[1, 1].set_xlim(0, 1)
                
                # 添加理想区域标注
                min_rmse = min(ec_rmse + gfs_rmse)
                axes[1, 1].axhspan(0, min_rmse, xmin=0.7, xmax=1.0, alpha=0.15, color='#C1FF72')
                axes[1, 1].text(0.85, min_rmse * 0.5, 'Ideal Region\nHigh Corr\nLow RMSE', 
                               ha='center', va='center', fontsize=10, 
                               bbox=dict(boxstyle='round', facecolor='#C1FF72', alpha=0.8, edgecolor='#7FB069'))
                
                # 添加显著性说明
                fig.text(0.02, 0.02, '** = Significant difference (p < 0.05)', 
                        fontsize=10, color='red', fontweight='bold')
            else:
                fig.text(0.5, 0.5, f'{var_info["name"]}: No valid evaluation results', 
                        ha='center', va='center', fontsize=20)
            
            plt.tight_layout()
            plt.savefig(f"{save_path}/enhanced_{var_name}_analysis.png", dpi=300, bbox_inches='tight')
            plt.show()
        
        # 5. 生成增强汇总报告
        print("\n🔄 步骤5: 生成增强汇总报告")
        
        summary_data = []
        for cls in evaluation_results:
            for var in evaluation_results[cls]:
                ec_metrics = evaluation_results[cls][var]['EC']
                gfs_metrics = evaluation_results[cls][var]['GFS']
                sig_tests = evaluation_results[cls][var]['SIGNIFICANCE']
                
                summary_data.append({
                    'Classification': cls,
                    'Variable': var,
                    'Variable_Name': key_variables[var]['name'],
                    'Sample_Size': ec_metrics['COUNT'],
                    'EC_CORR': ec_metrics['CORR'],
                    'GFS_CORR': gfs_metrics['CORR'],
                    'EC_BIAS': ec_metrics['BIAS'],
                    'GFS_BIAS': gfs_metrics['BIAS'],
                    'EC_RMSE': ec_metrics['RMSE'],
                    'GFS_RMSE': gfs_metrics['RMSE'],
                    'EC_MAE': ec_metrics['MAE'],
                    'GFS_MAE': gfs_metrics['MAE'],
                    'CORR_Diff_EC_minus_GFS': ec_metrics['CORR'] - gfs_metrics['CORR'],
                    'RMSE_Diff_GFS_minus_EC': gfs_metrics['RMSE'] - ec_metrics['RMSE'],
                    'Better_CORR': 'EC' if ec_metrics['CORR'] > gfs_metrics['CORR'] else 'GFS',
                    'Better_RMSE': 'EC' if ec_metrics['RMSE'] < gfs_metrics['RMSE'] else 'GFS',
                    # 显著性检验结果
                    'EC_CORR_pvalue': sig_tests['corr_ec_pvalue'],
                    'GFS_CORR_pvalue': sig_tests['corr_gfs_pvalue'],
                    'CORR_Diff_pvalue': sig_tests['corr_diff_pvalue'],
                    'BIAS_Diff_pvalue': sig_tests['bias_ttest_pvalue'],
                    'RMSE_Diff_pvalue': sig_tests['rmse_diff_pvalue'],
                    'EC_CORR_Significant': sig_tests['ec_corr_significant'],
                    'GFS_CORR_Significant': sig_tests['gfs_corr_significant'],
                    'CORR_Diff_Significant': sig_tests['corr_diff_significant'],
                    'BIAS_Diff_Significant': sig_tests['bias_diff_significant'],
                    'RMSE_Diff_Significant': sig_tests['rmse_diff_significant']
                })
        
        summary_df = pd.DataFrame(summary_data)
        summary_df.to_csv(f"{save_path}/enhanced_nwp_evaluation_with_significance.csv", index=False, encoding='utf-8-sig')
        
        # 创建显著性检验汇总表
        significance_summary = []
        for var_name, var_info in key_variables.items():
            var_data = summary_df[summary_df['Variable'] == var_name]
            if len(var_data) > 0:
                total_comparisons = len(var_data)
                
                # 统计显著性结果
                ec_corr_sig_count = var_data['EC_CORR_Significant'].sum()
                gfs_corr_sig_count = var_data['GFS_CORR_Significant'].sum()
                corr_diff_sig_count = var_data['CORR_Diff_Significant'].sum()
                bias_diff_sig_count = var_data['BIAS_Diff_Significant'].sum()
                rmse_diff_sig_count = var_data['RMSE_Diff_Significant'].sum()
                
                significance_summary.append({
                    'Variable': var_info['name'],
                    'Total_Classifications': total_comparisons,
                    'EC_CORR_Significant_Count': ec_corr_sig_count,
                    'GFS_CORR_Significant_Count': gfs_corr_sig_count,
                    'CORR_Diff_Significant_Count': corr_diff_sig_count,
                    'BIAS_Diff_Significant_Count': bias_diff_sig_count,
                    'RMSE_Diff_Significant_Count': rmse_diff_sig_count,
                    'EC_CORR_Significant_Percent': ec_corr_sig_count / total_comparisons * 100,
                    'GFS_CORR_Significant_Percent': gfs_corr_sig_count / total_comparisons * 100,
                    'CORR_Diff_Significant_Percent': corr_diff_sig_count / total_comparisons * 100,
                    'BIAS_Diff_Significant_Percent': bias_diff_sig_count / total_comparisons * 100,
                    'RMSE_Diff_Significant_Percent': rmse_diff_sig_count / total_comparisons * 100
                })
        
        sig_summary_df = pd.DataFrame(significance_summary)
        sig_summary_df.to_csv(f"{save_path}/significance_test_summary.csv", index=False, encoding='utf-8-sig')
        
        print("\n" + "=" * 80)
        print("🎉 Enhanced NWP Evaluation with Significance Tests Completed!")
        print("=" * 80)
        
        # 输出增强统计结果
        print("\n📊 Enhanced Statistical Results with Significance Tests:")
        for var_name, var_info in key_variables.items():
            var_data = summary_df[summary_df['Variable'] == var_name]
            if len(var_data) > 0:
                print(f"\n{var_info['name']}:")
                print(f"  Average sample size: {var_data['Sample_Size'].mean():.0f}")
                print(f"  EC average correlation: {var_data['EC_CORR'].mean():.3f}")
                print(f"  GFS average correlation: {var_data['GFS_CORR'].mean():.3f}")
                print(f"  EC average bias: {var_data['EC_BIAS'].mean():+.3f} {var_info['unit']}")
                print(f"  GFS average bias: {var_data['GFS_BIAS'].mean():+.3f} {var_info['unit']}")
                print(f"  EC average RMSE: {var_data['EC_RMSE'].mean():.3f} {var_info['unit']}")
                print(f"  GFS average RMSE: {var_data['GFS_RMSE'].mean():.3f} {var_info['unit']}")
                
                # 计算优势统计
                ec_corr_wins = (var_data['Better_CORR'] == 'EC').sum()
                ec_rmse_wins = (var_data['Better_RMSE'] == 'EC').sum()
                total_comparisons = len(var_data)
                
                print(f"  EC better correlation: {ec_corr_wins}/{total_comparisons} ({ec_corr_wins/total_comparisons*100:.1f}%)")
                print(f"  EC lower error: {ec_rmse_wins}/{total_comparisons} ({ec_rmse_wins/total_comparisons*100:.1f}%)")
                
                # 显著性统计
                corr_diff_sig = var_data['CORR_Diff_Significant'].sum()
                bias_diff_sig = var_data['BIAS_Diff_Significant'].sum()
                rmse_diff_sig = var_data['RMSE_Diff_Significant'].sum()
                
                print(f"  🔍 Significance Tests (α = {alpha_level}):")
                print(f"    Significant correlation differences: {corr_diff_sig}/{total_comparisons} ({corr_diff_sig/total_comparisons*100:.1f}%)")
                print(f"    Significant bias differences: {bias_diff_sig}/{total_comparisons} ({bias_diff_sig/total_comparisons*100:.1f}%)")
                print(f"    Significant RMSE differences: {rmse_diff_sig}/{total_comparisons} ({rmse_diff_sig/total_comparisons*100:.1f}%)")
        
        print(f"\n📁 Output files:")
        print(f"  - enhanced_[variable]_analysis.png: Enhanced analysis charts with significance marks")
        print(f"  - enhanced_nwp_evaluation_with_significance.csv: Complete evaluation data with significance tests")
        print(f"  - significance_test_summary.csv: Summary of significance test results")
        
        print(f"\n🔬 Significance Test Methods Used:")
        print(f"  - Correlation significance: Pearson correlation p-value")
        print(f"  - Correlation difference: Fisher's z-transformation test")
        print(f"  - Bias difference: Paired t-test")
        print(f"  - RMSE difference: Bootstrap test (1000 iterations)")
        print(f"  - Significance level: α = {alpha_level}")
        
        return True
        
    except Exception as e:
        print(f"❌ 评估过程出错: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    # 配置路径
    DATA_PATH = "/Users/xiaxin/work/WindForecast_Project/01_Data/processed/imputed_data/changma_imputed_complete.csv"
    SAVE_PATH = "/Users/xiaxin/work/WindForecast_Project/03_Results/enhanced_nwp_evaluation_results"
    
    success = practical_nwp_evaluation_with_significance(DATA_PATH, SAVE_PATH)
    
    if success:
        print("\n🎯 Enhanced evaluation with significance tests completed!")
        print("\n💡 Key improvements:")
        print("1. ✅ Focus on correlation, bias, RMSE")
        print("2. ✅ Sample size displayed for each classification")
        print("3. ✅ Specific values labeled on charts")
        print("4. ✅ Statistical significance testing")
        print("5. ✅ Significance marks (**) on charts")
        print("6. ✅ All text in English")
        print("7. ✅ Professional color scheme")
        print("\n📈 Practical value:")
        print("• Correlation coefficient → Forecast skill level")
        print("• Bias → Systematic error, can be corrected")
        print("• RMSE → Overall forecast accuracy")
        print("• Sample size → Statistical significance")
        print("• Significance tests → Statistical confidence")
        print("\n🔬 Statistical rigor:")
        print("• Pearson correlation significance")
        print("• Fisher's z-test for correlation differences")
        print("• Paired t-test for bias differences")
        print("• Bootstrap test for RMSE differences")
    else:
        print("\n⚠️ Evaluation failed")