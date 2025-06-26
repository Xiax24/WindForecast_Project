#!/usr/bin/env python3
"""
实用版数值预报评估器
重点关注：相关系数、偏差、RMSE + 样本数显示
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, mean_absolute_error
import os
import json
import warnings
warnings.filterwarnings('ignore')

# 设置英文显示
plt.rcParams['font.family'] = ['Arial', 'DejaVu Sans', 'Liberation Sans']
plt.rcParams['axes.unicode_minus'] = False

def practical_nwp_evaluation(data_path, save_path):
    """实用版数值预报评估 - 重点关注相关系数、偏差、RMSE"""
    print("=" * 80)
    print("📡 实用版数值预报评估分析")
    print("重点指标：相关系数、偏差、RMSE + 样本数")
    print("=" * 80)
    
    os.makedirs(save_path, exist_ok=True)
    
    # 配置
    shear_thresholds = {'weak_upper': 0.1, 'moderate_upper': 0.3}
    min_samples = 50
    min_valid_samples = 20
    
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
    
    try:
        # 1. 数据加载和预处理（与之前相同）
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
        
        # 3. 计算实用评估指标
        print("\n🔄 步骤3: 计算实用评估指标")
        
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
                        ec_metrics = calculate_practical_metrics(class_data[obs_col], class_data[ec_col])
                        gfs_metrics = calculate_practical_metrics(class_data[obs_col], class_data[gfs_col])
                        
                        evaluation_results[class_name][var_name] = {
                            'EC': ec_metrics,
                            'GFS': gfs_metrics
                        }
                        
                        # 显示关键指标
                        print(f"  {var_info['name']} (N={valid_count}):")
                        print(f"    EC:  CORR={ec_metrics['CORR']:.3f}, BIAS={ec_metrics['BIAS']:+.3f}, RMSE={ec_metrics['RMSE']:.3f}")
                        print(f"    GFS: CORR={gfs_metrics['CORR']:.3f}, BIAS={gfs_metrics['BIAS']:+.3f}, RMSE={gfs_metrics['RMSE']:.3f}")
                    else:
                        print(f"  {var_info['name']}: insufficient valid samples ({valid_count} < {min_valid_samples})")
        
        # 4. 创建实用的可视化图表
        print("\n🔄 步骤4: 创建实用可视化图表")
        
        # 设置整体样式
        plt.style.use('default')
        plt.rcParams['axes.facecolor'] = '#F8F9FA'
        plt.rcParams['figure.facecolor'] = 'white'
        plt.rcParams['axes.edgecolor'] = '#6C757D'
        plt.rcParams['axes.linewidth'] = 0.8
        plt.rcParams['grid.color'] = '#DEE2E6'
        plt.rcParams['grid.alpha'] = 0.7
        
        for var_name, var_info in key_variables.items():
            print(f"  Creating {var_info['name']} practical analysis chart...")
            
            fig, axes = plt.subplots(2, 2, figsize=(16, 12))
            fig.suptitle(f'{var_info["name"]} Forecast Performance Analysis', fontsize=16, fontweight='bold')
            
            # 收集数据
            classes = []
            ec_corr = []
            gfs_corr = []
            ec_bias = []
            gfs_bias = []
            ec_rmse = []
            gfs_rmse = []
            sample_counts = []
            
            for cls in evaluation_results.keys():
                if var_name in evaluation_results[cls]:
                    ec_metrics = evaluation_results[cls][var_name]['EC']
                    gfs_metrics = evaluation_results[cls][var_name]['GFS']
                    
                    # 只要相关系数不是NaN就包含
                    if not (np.isnan(ec_metrics['CORR']) or np.isnan(gfs_metrics['CORR'])):
                        classes.append(cls.replace('_', '\n'))
                        ec_corr.append(ec_metrics['CORR'])
                        gfs_corr.append(gfs_metrics['CORR'])
                        ec_bias.append(ec_metrics['BIAS'])
                        gfs_bias.append(gfs_metrics['BIAS'])
                        ec_rmse.append(ec_metrics['RMSE'])
                        gfs_rmse.append(gfs_metrics['RMSE'])
                        sample_counts.append(ec_metrics['COUNT'])
            
            if classes:
                x = np.arange(len(classes))
                width = 0.35
                
                # 1. 相关系数对比
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
                
                # 添加数值标签和样本数
                for i, (bar1, bar2, count) in enumerate(zip(bars1, bars2, sample_counts)):
                    # 相关系数标签
                    axes[0, 0].text(bar1.get_x() + bar1.get_width()/2., bar1.get_height() + 0.02,
                                   f'{ec_corr[i]:.3f}', ha='center', va='bottom', fontsize=8, fontweight='bold')
                    axes[0, 0].text(bar2.get_x() + bar2.get_width()/2., bar2.get_height() + 0.02,
                                   f'{gfs_corr[i]:.3f}', ha='center', va='bottom', fontsize=8, fontweight='bold')
                    # 样本数标签
                    axes[0, 0].text(i, 0.05, f'N={count}', ha='center', va='bottom', 
                                   fontsize=9, fontweight='bold', color='white',
                                   bbox=dict(boxstyle='round,pad=0.3', facecolor="#0165F1FF", alpha=0.9))
                
                # 2. 偏差对比
                bars1 = axes[0, 1].bar(x - width/2, ec_bias, width, label='EC', color="#30C1FF", alpha=0.8)
                bars2 = axes[0, 1].bar(x + width/2, gfs_bias, width, label='GFS', color="#CE6D7F", alpha=0.8)
                axes[0, 1].set_xlabel('Classification')
                axes[0, 1].set_ylabel(f'Bias ({var_info["unit"]})')
                axes[0, 1].set_title('Forecast Bias Comparison (Positive = Overestimate)')
                axes[0, 1].set_xticks(x)
                axes[0, 1].set_xticklabels(classes, rotation=45, ha='right')
                axes[0, 1].legend()
                axes[0, 1].grid(True, alpha=0.3)
                axes[0, 1].axhline(y=0, color='black', linestyle='-', alpha=0.5)
                
                # 添加偏差数值标签
                for bar, value in zip(bars1, ec_bias):
                    y_pos = value + (max(ec_bias + gfs_bias) - min(ec_bias + gfs_bias)) * 0.02 if value >= 0 else value - (max(ec_bias + gfs_bias) - min(ec_bias + gfs_bias)) * 0.02
                    axes[0, 1].text(bar.get_x() + bar.get_width()/2., y_pos,
                                   f'{value:+.2f}', ha='center', va='bottom' if value >= 0 else 'top', 
                                   fontsize=8, fontweight='bold')
                for bar, value in zip(bars2, gfs_bias):
                    y_pos = value + (max(ec_bias + gfs_bias) - min(ec_bias + gfs_bias)) * 0.02 if value >= 0 else value - (max(ec_bias + gfs_bias) - min(ec_bias + gfs_bias)) * 0.02
                    axes[0, 1].text(bar.get_x() + bar.get_width()/2., y_pos,
                                   f'{value:+.2f}', ha='center', va='bottom' if value >= 0 else 'top', 
                                   fontsize=8, fontweight='bold')
                
                # 3. RMSE对比
                bars1 = axes[1, 0].bar(x - width/2, ec_rmse, width, label='EC', color="#F1F148B8", alpha=0.8)
                bars2 = axes[1, 0].bar(x + width/2, gfs_rmse, width, label='GFS', color="#BC89E3", alpha=0.8)
                axes[1, 0].set_xlabel('Classification')
                axes[1, 0].set_ylabel(f'RMSE ({var_info["unit"]})')
                axes[1, 0].set_title('Root Mean Square Error Comparison')
                axes[1, 0].set_xticks(x)
                axes[1, 0].set_xticklabels(classes, rotation=45, ha='right')
                axes[1, 0].legend()
                axes[1, 0].grid(True, alpha=0.3)
                
                # 添加RMSE数值标签
                for bar, value in zip(bars1, ec_rmse):
                    axes[1, 0].text(bar.get_x() + bar.get_width()/2., bar.get_height() + max(ec_rmse + gfs_rmse) * 0.01,
                                   f'{value:.2f}', ha='center', va='bottom', fontsize=8, fontweight='bold')
                for bar, value in zip(bars2, gfs_rmse):
                    axes[1, 0].text(bar.get_x() + bar.get_width()/2., bar.get_height() + max(ec_rmse + gfs_rmse) * 0.01,
                                   f'{value:.2f}', ha='center', va='bottom', fontsize=8, fontweight='bold')
                
                # 4. 综合性能散点图 (相关系数 vs RMSE)
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
                axes[1, 1].set_title('Forecast Quality Distribution\n(Bottom-right = High Corr & Low RMSE = Good Forecast)')
                axes[1, 1].legend()
                axes[1, 1].grid(True, alpha=0.3)
                axes[1, 1].set_xlim(0, 1)
                
                # 添加理想区域标注
                axes[1, 1].axhspan(0, min(ec_rmse + gfs_rmse), xmin=0.7, xmax=1.0, alpha=0.15, color='#C1FF72')
                axes[1, 1].text(0.85, min(ec_rmse + gfs_rmse) * 0.5, 'Ideal Region\nHigh Corr\nLow RMSE', 
                               ha='center', va='center', fontsize=10, 
                               bbox=dict(boxstyle='round', facecolor='#C1FF72', alpha=0.8, edgecolor='#7FB069'))
            else:
                fig.text(0.5, 0.5, f'{var_info["name"]}: No valid evaluation results', 
                        ha='center', va='center', fontsize=20)
            
            plt.tight_layout()
            plt.savefig(f"{save_path}/practical_{var_name}_analysis.png", dpi=300, bbox_inches='tight')
            plt.show()
        
        # 5. 生成实用汇总报告
        print("\n🔄 步骤5: 生成实用汇总报告")
        
        summary_data = []
        for cls in evaluation_results:
            for var in evaluation_results[cls]:
                ec_metrics = evaluation_results[cls][var]['EC']
                gfs_metrics = evaluation_results[cls][var]['GFS']
                
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
                    'RMSE_Diff_GFS_minus_EC': gfs_metrics['RMSE'] - ec_metrics['RMSE'],  # 正值表示EC更好
                    'Better_CORR': 'EC' if ec_metrics['CORR'] > gfs_metrics['CORR'] else 'GFS',
                    'Better_RMSE': 'EC' if ec_metrics['RMSE'] < gfs_metrics['RMSE'] else 'GFS'
                })
        
        summary_df = pd.DataFrame(summary_data)
        summary_df.to_csv(f"{save_path}/practical_nwp_evaluation.csv", index=False, encoding='utf-8-sig')
        
        print("\n" + "=" * 80)
        print("🎉 Practical NWP Evaluation Analysis Completed!")
        print("=" * 80)
        
        # 输出实用统计结果
        print("\n📊 Practical Statistical Results:")
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
        
        print(f"\n📁 Output files:")
        print(f"  - practical_[variable]_analysis.png: Variable-specific analysis charts")
        print(f"  - practical_nwp_evaluation.csv: Practical evaluation data table")
        
        return True
        
    except Exception as e:
        print(f"❌ 评估过程出错: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    # 配置路径
    DATA_PATH = "/Users/xiaxin/work/WindForecast_Project/01_Data/processed/imputed_data/changma_imputed_complete.csv"
    SAVE_PATH = "/Users/xiaxin/work/WindForecast_Project/03_Results/practical_nwp_evaluation_results"
    
    success = practical_nwp_evaluation(DATA_PATH, SAVE_PATH)
    
    if success:
        print("\n🎯 Practical version evaluation completed!")
        print("\n💡 Key improvements:")
        print("1. ✅ Focus on correlation, bias, RMSE")
        print("2. ✅ Sample size displayed for each classification")
        print("3. ✅ Specific values labeled on charts")
        print("4. ✅ All text in English")
        print("5. ✅ Professional color scheme")
        print("\n📈 Practical value:")
        print("• Correlation coefficient → Forecast skill level")
        print("• Bias → Systematic error, can be corrected")
        print("• RMSE → Overall forecast accuracy")
        print("• Sample size → Statistical significance")
    else:
        print("\n⚠️ Evaluation failed")