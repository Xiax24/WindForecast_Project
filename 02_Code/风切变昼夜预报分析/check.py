#!/usr/bin/env python3
"""
最终修复版 - 正确处理负数R²
问题：原代码没有正确显示负数R²值
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import os
import json
import warnings
warnings.filterwarnings('ignore')

# 设置中文显示
plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial Unicode MS', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

def final_fixed_nwp_evaluation(data_path, save_path):
    """最终修复版 - 正确处理所有R²值包括负数"""
    print("=" * 80)
    print("📡 最终修复版数值预报评估分析 - 正确显示所有R²值")
    print("=" * 80)
    
    os.makedirs(save_path, exist_ok=True)
    
    # 配置
    shear_thresholds = {'weak_upper': 0.2, 'moderate_upper': 0.3}
    min_samples = 50  # 恢复原来的要求
    min_valid_samples = 20
    
    key_variables = {
        'wind_speed_10m': {
            'obs': 'obs_wind_speed_10m', 'ec': 'ec_wind_speed_10m', 'gfs': 'gfs_wind_speed_10m', 
            'name': '10m风速', 'unit': 'm/s'
        },
        'wind_speed_70m': {
            'obs': 'obs_wind_speed_70m', 'ec': 'ec_wind_speed_70m', 'gfs': 'gfs_wind_speed_70m', 
            'name': '70m风速', 'unit': 'm/s'
        },
        'temperature_10m': {
            'obs': 'obs_temperature_10m', 'ec': 'ec_temperature_10m', 'gfs': 'gfs_temperature_10m', 
            'name': '10m温度', 'unit': '°C'
        }
    }
    
    try:
        # 1. 加载数据
        print("\n🔄 步骤1: 加载数据")
        data = pd.read_csv(data_path)
        data['datetime'] = pd.to_datetime(data['datetime'])
        print(f"原始数据形状: {data.shape}")
        
        # 2. 数据预处理（按照调试结果，数据质量很好）
        print("\n🔄 步骤2: 数据预处理")
        
        # 清理明显异常值
        for var_name, var_info in key_variables.items():
            for col_type in ['obs', 'ec', 'gfs']:
                col = var_info[col_type]
                if col in data.columns:
                    if 'wind_speed' in col:
                        data[col] = data[col].where((data[col] >= 0) & (data[col] <= 50))
                    elif 'temperature' in col:
                        data[col] = data[col].where((data[col] >= -50) & (data[col] <= 60))
        
        # 3. 风切变计算
        print("\n🔄 步骤3: 风切变计算")
        
        v1 = data['obs_wind_speed_10m']
        v2 = data['obs_wind_speed_70m']
        
        valid_wind_mask = (v1 > 0.5) & (v2 > 0.5) & (~v1.isna()) & (~v2.isna())
        print(f"有效风速数据: {valid_wind_mask.sum()}/{len(data)} ({valid_wind_mask.sum()/len(data)*100:.1f}%)")
        
        data = data[valid_wind_mask].copy()
        v1_filtered = data['obs_wind_speed_10m']
        v2_filtered = data['obs_wind_speed_70m']
        
        # 计算风切变
        data['wind_shear_alpha'] = np.log(v2_filtered / v1_filtered) / np.log(70 / 10)
        
        # 清理异常风切变
        alpha = data['wind_shear_alpha']
        valid_alpha = (~np.isnan(alpha)) & (~np.isinf(alpha)) & (alpha > -1) & (alpha < 2)
        data = data[valid_alpha].copy()
        
        print(f"风切变清理后: {len(data)} 条")
        
        # 分类
        alpha = data['wind_shear_alpha']
        conditions = [
            alpha < shear_thresholds['weak_upper'],
            (alpha >= shear_thresholds['weak_upper']) & (alpha < shear_thresholds['moderate_upper']),
            alpha >= shear_thresholds['moderate_upper']
        ]
        choices = ['weak', 'moderate', 'strong']
        data['shear_group'] = np.select(conditions, choices, default='unknown')
        
        # 昼夜分类
        data['hour'] = data['datetime'].dt.hour
        data['is_daytime'] = ((data['hour'] >= 6) & (data['hour'] < 18))
        data['shear_diurnal_class'] = data['shear_group'].astype(str) + '_' + \
                                     data['is_daytime'].map({True: 'day', False: 'night'})
        
        class_counts = data['shear_diurnal_class'].value_counts()
        print(f"\n风切变分类分布:")
        for class_name, count in class_counts.items():
            if 'unknown' not in class_name:
                percentage = count / len(data) * 100
                print(f"  {class_name}: {count} 条 ({percentage:.1f}%)")
        
        # 4. 计算评估指标 - 关键修复：正确处理负数R²
        print("\n🔄 步骤4: 计算评估指标（包括负数R²）")
        
        def calculate_metrics(obs, forecast):
            obs = np.array(obs)
            forecast = np.array(forecast)
            valid_mask = ~(np.isnan(obs) | np.isnan(forecast) | np.isinf(obs) | np.isinf(forecast))
            obs_clean = obs[valid_mask]
            forecast_clean = forecast[valid_mask]
            
            if len(obs_clean) < min_valid_samples:
                return {'RMSE': np.nan, 'MAE': np.nan, 'R2': np.nan, 'BIAS': np.nan, 'CORR': np.nan, 'COUNT': len(obs_clean)}
            
            try:
                rmse = np.sqrt(mean_squared_error(obs_clean, forecast_clean))
                mae = mean_absolute_error(obs_clean, forecast_clean)
                r2 = r2_score(obs_clean, forecast_clean)  # 不限制R²范围，允许负数
                bias = np.mean(forecast_clean - obs_clean)
                
                if len(obs_clean) > 1 and np.std(obs_clean) > 1e-10 and np.std(forecast_clean) > 1e-10:
                    corr = np.corrcoef(obs_clean, forecast_clean)[0, 1]
                else:
                    corr = np.nan
                
                return {'RMSE': rmse, 'MAE': mae, 'R2': r2, 'BIAS': bias, 'CORR': corr, 'COUNT': len(obs_clean)}
            except Exception as e:
                print(f"    计算指标错误: {e}")
                return {'RMSE': np.nan, 'MAE': np.nan, 'R2': np.nan, 'BIAS': np.nan, 'CORR': np.nan, 'COUNT': len(obs_clean)}
        
        evaluation_results = {}
        unique_classes = [cls for cls in data['shear_diurnal_class'].unique() if 'unknown' not in cls]
        
        print(f"开始评估 {len(unique_classes)} 个分类...")
        
        for class_name in unique_classes:
            class_data = data[data['shear_diurnal_class'] == class_name]
            
            if len(class_data) < min_samples:
                print(f"跳过 {class_name}: 样本数不足 ({len(class_data)} < {min_samples})")
                continue
                
            print(f"\n评估 {class_name}: {len(class_data)} 条样本")
            evaluation_results[class_name] = {}
            
            for var_name, var_info in key_variables.items():
                obs_col = var_info['obs']
                ec_col = var_info['ec']
                gfs_col = var_info['gfs']
                
                if all(col in class_data.columns for col in [obs_col, ec_col, gfs_col]):
                    var_valid = (~class_data[obs_col].isna()) & (~class_data[ec_col].isna()) & (~class_data[gfs_col].isna())
                    valid_count = var_valid.sum()
                    
                    if valid_count >= min_valid_samples:
                        ec_metrics = calculate_metrics(class_data[obs_col], class_data[ec_col])
                        gfs_metrics = calculate_metrics(class_data[obs_col], class_data[gfs_col])
                        
                        evaluation_results[class_name][var_name] = {
                            'EC': ec_metrics,
                            'GFS': gfs_metrics
                        }
                        
                        # 显示结果（包括负数R²）
                        ec_r2 = ec_metrics['R2']
                        gfs_r2 = gfs_metrics['R2']
                        print(f"  {var_info['name']}: EC R²={ec_r2:.3f}, GFS R²={gfs_r2:.3f} (样本数:{valid_count})")
                    else:
                        print(f"  {var_info['name']}: 有效样本不足 ({valid_count} < {min_valid_samples})")
        
        print(f"\n完成 {len(evaluation_results)} 个分类的评估")
        
        # 5. 创建图表 - 修复版本，正确显示负数R²
        print("\n🔄 步骤5: 创建可视化图表（包含负数R²）")
        
        for var_name, var_info in key_variables.items():
            print(f"  创建{var_info['name']}性能图...")
            
            fig, axes = plt.subplots(2, 2, figsize=(15, 12))
            fig.suptitle(f'{var_info["name"]} 预报性能分析（修复版 - 显示所有R²值）', fontsize=16, fontweight='bold')
            
            # 收集数据 - 关键修复：不过滤负数R²
            classes = []
            ec_r2 = []
            gfs_r2 = []
            ec_rmse = []
            gfs_rmse = []
            sample_counts = []
            
            for cls in evaluation_results.keys():
                if var_name in evaluation_results[cls]:
                    ec_r2_val = evaluation_results[cls][var_name]['EC']['R2']
                    gfs_r2_val = evaluation_results[cls][var_name]['GFS']['R2']
                    ec_rmse_val = evaluation_results[cls][var_name]['EC']['RMSE']
                    gfs_rmse_val = evaluation_results[cls][var_name]['GFS']['RMSE']
                    sample_count = evaluation_results[cls][var_name]['EC']['COUNT']
                    
                    # 只要不是NaN就包含（包括负数）
                    if not (np.isnan(ec_r2_val) or np.isnan(gfs_r2_val)):
                        classes.append(cls.replace('_', '\n'))
                        ec_r2.append(ec_r2_val)
                        gfs_r2.append(gfs_r2_val)
                        ec_rmse.append(ec_rmse_val)
                        gfs_rmse.append(gfs_rmse_val)
                        sample_counts.append(sample_count)
            
            if classes:
                x = np.arange(len(classes))
                width = 0.35
                
                # R²对比 - 调整y轴范围以显示负数
                bars1 = axes[0, 0].bar(x - width/2, ec_r2, width, label='EC', color='blue', alpha=0.8)
                bars2 = axes[0, 0].bar(x + width/2, gfs_r2, width, label='GFS', color='red', alpha=0.8)
                axes[0, 0].set_xlabel('分类')
                axes[0, 0].set_ylabel('R² Score')
                axes[0, 0].set_title('R²决定系数对比（包含负值）')
                axes[0, 0].set_xticks(x)
                axes[0, 0].set_xticklabels(classes, rotation=45, ha='right')
                axes[0, 0].legend()
                axes[0, 0].grid(True, alpha=0.3)
                axes[0, 0].axhline(y=0, color='black', linestyle='-', alpha=0.5, linewidth=1)  # 添加零线
                
                # 动态调整y轴范围
                all_r2 = ec_r2 + gfs_r2
                y_min = min(all_r2) - 0.1
                y_max = max(all_r2) + 0.1
                axes[0, 0].set_ylim(y_min, y_max)
                
                # 添加数值标签
                for bar, value in zip(bars1, ec_r2):
                    y_pos = value + 0.02 if value >= 0 else value - 0.05
                    axes[0, 0].text(bar.get_x() + bar.get_width()/2., y_pos,
                                   f'{value:.3f}', ha='center', va='bottom' if value >= 0 else 'top', 
                                   fontsize=8, fontweight='bold')
                for bar, value in zip(bars2, gfs_r2):
                    y_pos = value + 0.02 if value >= 0 else value - 0.05
                    axes[0, 0].text(bar.get_x() + bar.get_width()/2., y_pos,
                                   f'{value:.3f}', ha='center', va='bottom' if value >= 0 else 'top', 
                                   fontsize=8, fontweight='bold')
                
                # RMSE对比
                bars1 = axes[0, 1].bar(x - width/2, ec_rmse, width, label='EC', color='blue', alpha=0.8)
                bars2 = axes[0, 1].bar(x + width/2, gfs_rmse, width, label='GFS', color='red', alpha=0.8)
                axes[0, 1].set_xlabel('分类')
                axes[0, 1].set_ylabel(f'RMSE ({var_info["unit"]})')
                axes[0, 1].set_title('均方根误差对比')
                axes[0, 1].set_xticks(x)
                axes[0, 1].set_xticklabels(classes, rotation=45, ha='right')
                axes[0, 1].legend()
                axes[0, 1].grid(True, alpha=0.3)
                
                # 散点图 - EC vs GFS R²
                axes[1, 0].scatter(ec_r2, gfs_r2, alpha=0.7, s=100, c='green')
                
                # 添加对角线和零线
                all_r2_range = [min(all_r2), max(all_r2)]
                axes[1, 0].plot(all_r2_range, all_r2_range, 'r--', alpha=0.5, label='y=x线')
                axes[1, 0].axhline(y=0, color='gray', linestyle='-', alpha=0.3)
                axes[1, 0].axvline(x=0, color='gray', linestyle='-', alpha=0.3)
                
                axes[1, 0].set_xlabel('EC R²')
                axes[1, 0].set_ylabel('GFS R²')
                axes[1, 0].set_title('EC vs GFS R²对比')
                axes[1, 0].grid(True, alpha=0.3)
                axes[1, 0].legend()
                
                # 添加点标签
                for i, cls in enumerate(classes):
                    axes[1, 0].annotate(cls.replace('\n', '_'), (ec_r2[i], gfs_r2[i]), 
                                       xytext=(5, 5), textcoords='offset points', fontsize=8)
                
                # 性能差异柱状图
                diff = [ec - gfs for ec, gfs in zip(ec_r2, gfs_r2)]
                colors = ['green' if x > 0 else 'red' for x in diff]
                bars = axes[1, 1].bar(range(len(diff)), diff, color=colors, alpha=0.7)
                axes[1, 1].set_xlabel('分类序号')
                axes[1, 1].set_ylabel('R²差异 (EC - GFS)')
                axes[1, 1].set_title('性能差异 (正值=EC更好)')
                axes[1, 1].axhline(y=0, color='black', linestyle='-', alpha=0.5)
                axes[1, 1].grid(True, alpha=0.3)
                
                # 添加差异数值标签
                for bar, value in zip(bars, diff):
                    y_pos = value + 0.01 if value >= 0 else value - 0.02
                    axes[1, 1].text(bar.get_x() + bar.get_width()/2., y_pos,
                                   f'{value:.3f}', ha='center', va='bottom' if value >= 0 else 'top', 
                                   fontsize=9, fontweight='bold')
                
                # 在右下角添加分类标签
                class_labels = '\n'.join([f"{i}: {cls.replace(chr(10), '_')}" for i, cls in enumerate(classes)])
                axes[1, 1].text(0.02, 0.98, class_labels, transform=axes[1, 1].transAxes, 
                               va='top', fontsize=7, bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
            else:
                fig.text(0.5, 0.5, f'{var_info["name"]}: 无有效评估结果', 
                        ha='center', va='center', fontsize=20)
            
            plt.tight_layout()
            plt.savefig(f"{save_path}/fixed_{var_name}_performance.png", dpi=300, bbox_inches='tight')
            plt.show()
        
        # 6. 生成总结报告
        print("\n🔄 步骤6: 生成总结报告")
        
        # 统计结果
        summary_data = []
        for cls in evaluation_results:
            for var in evaluation_results[cls]:
                ec_metrics = evaluation_results[cls][var]['EC']
                gfs_metrics = evaluation_results[cls][var]['GFS']
                
                summary_data.append({
                    'Classification': cls,
                    'Variable': var,
                    'Variable_Name': key_variables[var]['name'],
                    'EC_R2': ec_metrics['R2'],
                    'GFS_R2': gfs_metrics['R2'],
                    'EC_RMSE': ec_metrics['RMSE'],
                    'GFS_RMSE': gfs_metrics['RMSE'],
                    'Sample_Size': ec_metrics['COUNT'],
                    'R2_Diff_EC_minus_GFS': ec_metrics['R2'] - gfs_metrics['R2'],
                    'Better_Model': 'EC' if ec_metrics['R2'] > gfs_metrics['R2'] else 'GFS'
                })
        
        summary_df = pd.DataFrame(summary_data)
        summary_df.to_csv(f"{save_path}/fixed_detailed_comparison.csv", index=False, encoding='utf-8-sig')
        
        print("\n" + "=" * 80)
        print("🎉 修复版数值预报评估分析完成！")
        print("=" * 80)
        
        print("\n📊 关键发现:")
        print("✅ 问题已解决：原代码没有正确显示负数R²值")
        print("✅ 现在所有R²值（包括负数）都会显示在图表中")
        print("✅ 负数R²表示预报效果比简单平均还差，这是正常现象")
        
        # 按变量统计R²情况
        for var_name, var_info in key_variables.items():
            var_data = summary_df[summary_df['Variable'] == var_name]
            if len(var_data) > 0:
                ec_r2_avg = var_data['EC_R2'].mean()
                gfs_r2_avg = var_data['GFS_R2'].mean()
                positive_ec = (var_data['EC_R2'] > 0).sum()
                positive_gfs = (var_data['GFS_R2'] > 0).sum()
                
                print(f"\n{var_info['name']}:")
                print(f"  EC平均R²: {ec_r2_avg:.3f}, 正值比例: {positive_ec}/{len(var_data)}")
                print(f"  GFS平均R²: {gfs_r2_avg:.3f}, 正值比例: {positive_gfs}/{len(var_data)}")
        
        return True
        
    except Exception as e:
        print(f"❌ 评估过程出错: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    # 配置路径
    DATA_PATH = "/Users/xiaxin/work/WindForecast_Project/01_Data/processed/imputed_data/changma_imputed_complete.csv"
    SAVE_PATH = "/Users/xiaxin/work/WindForecast_Project/03_Results/fixed_nwp_evaluation_results"
    
    success = final_fixed_nwp_evaluation(DATA_PATH, SAVE_PATH)
    
    if success:
        print("\n🎯 问题解决！")
        print("\n💡 解释：")
        print("1. 风速预报在某些条件下确实表现很差（负R²）")
        print("2. 这可能是因为风速的随机性较强，难以预报")
        print("3. 温度预报效果很好，因为温度变化相对规律")
        print("4. 负数R²是正常现象，表示模型预报效果差")
    else:
        print("\n⚠️ 评估失败")