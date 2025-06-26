#!/usr/bin/env python3
"""
预报切变与真实切变关系分析器
目标：验证EC、GFS、EC+GFS平均的预报切变是否可以作为真实切变的代理变量
重点：找出最接近真实切变的预报切变，用于权重分配策略
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, mean_absolute_error
from scipy.stats import pearsonr, spearmanr
import seaborn as sns
import os
import warnings
warnings.filterwarnings('ignore')

# 设置英文显示
plt.rcParams['font.family'] = ['Arial', 'DejaVu Sans', 'Liberation Sans']
plt.rcParams['axes.unicode_minus'] = False

def analyze_forecast_vs_observed_shear(data_path, save_path):
    """分析预报切变与观测切变的关系"""
    print("=" * 80)
    print("🌬️ 预报切变与真实切变关系分析")
    print("目标：找出最接近真实切变的预报切变，用作权重分配的桥梁变量")
    print("对比：EC预报切变 vs GFS预报切变 vs EC+GFS平均切变")
    print("=" * 80)
    
    os.makedirs(save_path, exist_ok=True)
    
    # 1. 数据加载
    print("\n🔄 步骤1: 数据加载")
    data = pd.read_csv(data_path)
    data['datetime'] = pd.to_datetime(data['datetime'])
    print(f"原始数据形状: {data.shape}")
    
    # 2. 检查必需列
    required_cols = [
        'obs_wind_speed_10m', 'obs_wind_speed_70m',
        'ec_wind_speed_10m', 'ec_wind_speed_70m', 
        'gfs_wind_speed_10m', 'gfs_wind_speed_70m'
    ]
    
    missing_cols = [col for col in required_cols if col not in data.columns]
    if missing_cols:
        print(f"❌ 缺少必需列: {missing_cols}")
        return False
    
    print("✅ 所有必需列都存在")
    
    # 3. 数据清理
    print("\n🔄 步骤2: 数据清理")
    
    # 清理异常值
    for col in required_cols:
        data[col] = data[col].where((data[col] >= 0) & (data[col] <= 50))
    
    # 筛选有效数据
    valid_obs = (data['obs_wind_speed_10m'] > 0.5) & (data['obs_wind_speed_70m'] > 0.5)
    valid_ec = (data['ec_wind_speed_10m'] > 0.5) & (data['ec_wind_speed_70m'] > 0.5)
    valid_gfs = (data['gfs_wind_speed_10m'] > 0.5) & (data['gfs_wind_speed_70m'] > 0.5)
    
    # 去除NaN
    valid_obs = valid_obs & (~data['obs_wind_speed_10m'].isna()) & (~data['obs_wind_speed_70m'].isna())
    valid_ec = valid_ec & (~data['ec_wind_speed_10m'].isna()) & (~data['ec_wind_speed_70m'].isna())
    valid_gfs = valid_gfs & (~data['gfs_wind_speed_10m'].isna()) & (~data['gfs_wind_speed_70m'].isna())
    
    # 只保留所有数据都有效的样本
    valid_all = valid_obs & valid_ec & valid_gfs
    data = data[valid_all].copy()
    
    print(f"有效数据点: {len(data)} (清理后)")
    
    # 4. 计算风切变
    print("\n🔄 步骤3: 计算风切变")
    
    # 观测风切变 (alpha法 - 对数风廓线)
    data['obs_shear_alpha'] = np.log(data['obs_wind_speed_70m'] / data['obs_wind_speed_10m']) / np.log(70 / 10)
    
    # EC预报风切变
    data['ec_shear_alpha'] = np.log(data['ec_wind_speed_70m'] / data['ec_wind_speed_10m']) / np.log(70 / 10)
    
    # GFS预报风切变
    data['gfs_shear_alpha'] = np.log(data['gfs_wind_speed_70m'] / data['gfs_wind_speed_10m']) / np.log(70 / 10)
    
    # EC+GFS平均风切变
    avg_10m = (data['ec_wind_speed_10m'] + data['gfs_wind_speed_10m']) / 2
    avg_70m = (data['ec_wind_speed_70m'] + data['gfs_wind_speed_70m']) / 2
    data['avg_shear_alpha'] = np.log(avg_70m / avg_10m) / np.log(70 / 10)
    
    # 另一种平均方法：直接对切变求平均
    data['avg_shear_alpha_direct'] = (data['ec_shear_alpha'] + data['gfs_shear_alpha']) / 2
    
    # 过滤异常切变值
    shear_cols = ['obs_shear_alpha', 'ec_shear_alpha', 'gfs_shear_alpha', 'avg_shear_alpha', 'avg_shear_alpha_direct']
    for col in shear_cols:
        valid_shear = (~np.isnan(data[col])) & (~np.isinf(data[col])) & (data[col] > -1) & (data[col] < 2)
        data = data[valid_shear].copy()
    
    print(f"最终有效数据点: {len(data)}")
    
    # 添加时间信息
    data['hour'] = data['datetime'].dt.hour
    data['is_daytime'] = (data['hour'] >= 6) & (data['hour'] < 18)
    
    # 5. 相关性分析
    print("\n🔄 步骤4: 相关性分析")
    
    obs_shear = data['obs_shear_alpha']
    ec_shear = data['ec_shear_alpha']
    gfs_shear = data['gfs_shear_alpha']
    avg_shear = data['avg_shear_alpha']
    avg_shear_direct = data['avg_shear_alpha_direct']
    
    # 计算相关系数
    ec_corr, ec_p = pearsonr(obs_shear, ec_shear)
    gfs_corr, gfs_p = pearsonr(obs_shear, gfs_shear)
    avg_corr, avg_p = pearsonr(obs_shear, avg_shear)
    avg_direct_corr, avg_direct_p = pearsonr(obs_shear, avg_shear_direct)
    
    print(f"预报切变与观测切变的相关性:")
    print(f"  EC预报切变:      r = {ec_corr:.4f}, p = {ec_p:.6f}")
    print(f"  GFS预报切变:     r = {gfs_corr:.4f}, p = {gfs_p:.6f}")
    print(f"  EC+GFS平均切变:  r = {avg_corr:.4f}, p = {avg_p:.6f}")
    print(f"  切变直接平均:    r = {avg_direct_corr:.4f}, p = {avg_direct_p:.6f}")
    
    # 6. 预报精度分析
    print("\n🔄 步骤5: 预报精度分析")
    
    # RMSE
    ec_rmse = np.sqrt(mean_squared_error(obs_shear, ec_shear))
    gfs_rmse = np.sqrt(mean_squared_error(obs_shear, gfs_shear))
    avg_rmse = np.sqrt(mean_squared_error(obs_shear, avg_shear))
    avg_direct_rmse = np.sqrt(mean_squared_error(obs_shear, avg_shear_direct))
    
    # MAE
    ec_mae = mean_absolute_error(obs_shear, ec_shear)
    gfs_mae = mean_absolute_error(obs_shear, gfs_shear)
    avg_mae = mean_absolute_error(obs_shear, avg_shear)
    avg_direct_mae = mean_absolute_error(obs_shear, avg_shear_direct)
    
    # 偏差
    ec_bias = np.mean(ec_shear - obs_shear)
    gfs_bias = np.mean(gfs_shear - obs_shear)
    avg_bias = np.mean(avg_shear - obs_shear)
    avg_direct_bias = np.mean(avg_shear_direct - obs_shear)
    
    print(f"预报切变精度指标 (越小越好):")
    print(f"                    RMSE     MAE      偏差")
    print(f"  EC预报切变:      {ec_rmse:.4f}  {ec_mae:.4f}  {ec_bias:+.4f}")
    print(f"  GFS预报切变:     {gfs_rmse:.4f}  {gfs_mae:.4f}  {gfs_bias:+.4f}")
    print(f"  EC+GFS平均切变:  {avg_rmse:.4f}  {avg_mae:.4f}  {avg_bias:+.4f}")
    print(f"  切变直接平均:    {avg_direct_rmse:.4f}  {avg_direct_mae:.4f}  {avg_direct_bias:+.4f}")
    
    # 7. 找出最优预报切变
    print("\n🔄 步骤6: 最优预报切变评估")
    
    correlations = {'EC': ec_corr, 'GFS': gfs_corr, 'EC+GFS平均': avg_corr, '切变直接平均': avg_direct_corr}
    rmses = {'EC': ec_rmse, 'GFS': gfs_rmse, 'EC+GFS平均': avg_rmse, '切变直接平均': avg_direct_rmse}
    
    best_corr_model = max(correlations, key=correlations.get)
    best_rmse_model = min(rmses, key=rmses.get)
    
    print(f"🏆 最高相关性: {best_corr_model} (r = {correlations[best_corr_model]:.4f})")
    print(f"🏆 最小误差:   {best_rmse_model} (RMSE = {rmses[best_rmse_model]:.4f})")
    
    # 综合评分 (相关性权重0.6，精度权重0.4)
    scores = {}
    for model in correlations.keys():
        # 标准化分数
        corr_score = correlations[model]  # 相关性越高越好
        rmse_score = 1 / (1 + rmses[model])  # RMSE越小越好
        scores[model] = 0.6 * corr_score + 0.4 * rmse_score
    
    best_overall_model = max(scores, key=scores.get)
    print(f"🏆 综合最优:   {best_overall_model} (综合得分 = {scores[best_overall_model]:.4f})")
    
    # 8. 分时段分析
    print("\n🔄 步骤7: 分时段分析")
    
    time_periods = {'白天(6-18h)': True, '夜间(18-6h)': False}
    time_results = {}
    
    for period_name, is_day in time_periods.items():
        period_data = data[data['is_daytime'] == is_day]
        if len(period_data) > 50:
            obs_period = period_data['obs_shear_alpha']
            ec_period = period_data['ec_shear_alpha']
            gfs_period = period_data['gfs_shear_alpha']
            avg_period = period_data['avg_shear_alpha']
            
            ec_corr_period, _ = pearsonr(obs_period, ec_period)
            gfs_corr_period, _ = pearsonr(obs_period, gfs_period)
            avg_corr_period, _ = pearsonr(obs_period, avg_period)
            
            time_results[period_name] = {
                'EC': ec_corr_period,
                'GFS': gfs_corr_period,
                'EC+GFS平均': avg_corr_period,
                'sample_size': len(period_data)
            }
            
            print(f"  {period_name} (N={len(period_data)}):")
            print(f"    EC: {ec_corr_period:.3f}, GFS: {gfs_corr_period:.3f}, 平均: {avg_corr_period:.3f}")
    
    # 9. 可视化分析
    print("\n🔄 步骤8: 创建可视化")
    
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    
    # 1. EC vs 观测散点图
    ax1 = axes[0, 0]
    ax1.scatter(obs_shear, ec_shear, alpha=0.6, s=15, c='blue')
    ax1.plot([-1, 2], [-1, 2], 'r--', alpha=0.8, label='完美预报线')
    ax1.set_xlabel('观测风切变 Alpha')
    ax1.set_ylabel('EC预报风切变 Alpha')
    ax1.set_title(f'EC预报 vs 观测\nr={ec_corr:.3f}, RMSE={ec_rmse:.3f}')
    ax1.grid(True, alpha=0.3)
    ax1.legend()
    
    # 2. GFS vs 观测散点图
    ax2 = axes[0, 1]
    ax2.scatter(obs_shear, gfs_shear, alpha=0.6, s=15, c='green')
    ax2.plot([-1, 2], [-1, 2], 'r--', alpha=0.8, label='完美预报线')
    ax2.set_xlabel('观测风切变 Alpha')
    ax2.set_ylabel('GFS预报风切变 Alpha')
    ax2.set_title(f'GFS预报 vs 观测\nr={gfs_corr:.3f}, RMSE={gfs_rmse:.3f}')
    ax2.grid(True, alpha=0.3)
    ax2.legend()
    
    # 3. 平均 vs 观测散点图
    ax3 = axes[0, 2]
    ax3.scatter(obs_shear, avg_shear, alpha=0.6, s=15, c='orange')
    ax3.plot([-1, 2], [-1, 2], 'r--', alpha=0.8, label='完美预报线')
    ax3.set_xlabel('观测风切变 Alpha')
    ax3.set_ylabel('EC+GFS平均风切变 Alpha')
    ax3.set_title(f'EC+GFS平均 vs 观测\nr={avg_corr:.3f}, RMSE={avg_rmse:.3f}')
    ax3.grid(True, alpha=0.3)
    ax3.legend()
    
    # 4. 相关性对比柱状图
    ax4 = axes[1, 0]
    models = ['EC', 'GFS', 'EC+GFS\n平均', '切变直接\n平均']
    corrs = [ec_corr, gfs_corr, avg_corr, avg_direct_corr]
    colors = ['blue', 'green', 'orange', 'red']
    
    bars = ax4.bar(models, corrs, color=colors, alpha=0.7)
    ax4.set_ylabel('相关系数')
    ax4.set_title('预报切变相关性对比')
    ax4.grid(True, alpha=0.3)
    ax4.set_ylim(0, 1)
    
    # 添加数值标签
    for bar, corr in zip(bars, corrs):
        ax4.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                f'{corr:.3f}', ha='center', va='bottom', fontweight='bold')
    
    # 5. RMSE对比柱状图
    ax5 = axes[1, 1]
    rmses_list = [ec_rmse, gfs_rmse, avg_rmse, avg_direct_rmse]
    
    bars2 = ax5.bar(models, rmses_list, color=colors, alpha=0.7)
    ax5.set_ylabel('RMSE')
    ax5.set_title('预报切变误差对比')
    ax5.grid(True, alpha=0.3)
    
    # 添加数值标签
    for bar, rmse in zip(bars2, rmses_list):
        ax5.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.002,
                f'{rmse:.3f}', ha='center', va='bottom', fontweight='bold')
    
    # 6. 偏差分布图
    ax6 = axes[1, 2]
    biases = [ec_shear - obs_shear, gfs_shear - obs_shear, avg_shear - obs_shear]
    labels = ['EC偏差', 'GFS偏差', '平均偏差']
    colors_hist = ['blue', 'green', 'orange']
    
    ax6.hist(biases, bins=30, alpha=0.6, label=labels, color=colors_hist)
    ax6.axvline(x=0, color='red', linestyle='--', alpha=0.8, label='零偏差线')
    ax6.set_xlabel('预报偏差 (预报值 - 观测值)')
    ax6.set_ylabel('频次')
    ax6.set_title('预报切变偏差分布')
    ax6.legend()
    ax6.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f"{save_path}/forecast_shear_analysis.png", dpi=300, bbox_inches='tight')
    plt.show()
    
    # 10. 结论和建议
    print("\n" + "=" * 80)
    print("📋 分析结论和建议")
    print("=" * 80)
    
    best_corr_value = correlations[best_corr_model]
    best_rmse_value = rmses[best_rmse_model]
    
    print(f"\n🔍 主要发现:")
    print(f"  相关性最高: {best_corr_model} (r = {best_corr_value:.4f})")
    print(f"  误差最小:   {best_rmse_model} (RMSE = {best_rmse_value:.4f})")
    print(f"  综合最优:   {best_overall_model}")
    
    # 桥梁变量评估
    print(f"\n🌉 桥梁变量评估:")
    if best_corr_value > 0.8:
        bridge_quality = "✅ 优秀桥梁变量"
        recommendation = f"强烈建议使用{best_corr_model}预报切变直接指导权重分配"
        usable = True
    elif best_corr_value > 0.6:
        bridge_quality = "✅ 良好桥梁变量"
        recommendation = f"建议使用{best_corr_model}预报切变指导权重分配"
        usable = True
    elif best_corr_value > 0.4:
        bridge_quality = "⚠️ 中等桥梁变量"
        recommendation = f"可以尝试使用{best_corr_model}预报切变，但需要谨慎"
        usable = True
    elif best_corr_value > 0.2:
        bridge_quality = "⚠️ 较弱桥梁变量"
        recommendation = "建议作为辅助参考，不要完全依赖"
        usable = False
    else:
        bridge_quality = "❌ 不适合作桥梁变量"
        recommendation = "寻找其他更可靠的方法"
        usable = False
    
    print(f"  {bridge_quality}")
    print(f"  建议: {recommendation}")
    
    # 具体权重策略
    if usable and best_corr_value > 0.4:
        print(f"\n🔧 具体权重分配策略 (基于{best_overall_model}预报切变):")
        print("```python")
        print("def calculate_weights_by_forecast_shear(forecast_shear_alpha):")
        print("    # 基于预报切变动态调整10m和70m权重")
        print("    if forecast_shear_alpha < 0.05:")
        print("        # 极弱切变 - 10m风速更重要")
        print("        w_10m, w_70m = 0.85, 0.15")
        print("    elif forecast_shear_alpha < 0.15:")
        print("        # 弱切变 - 10m风速主导")
        print("        w_10m, w_70m = 0.75, 0.25")
        print("    elif forecast_shear_alpha < 0.25:")
        print("        # 中等切变 - 均衡权重")
        print("        w_10m, w_70m = 0.60, 0.40")
        print("    elif forecast_shear_alpha < 0.35:")
        print("        # 较强切变 - 70m重要性增加")
        print("        w_10m, w_70m = 0.45, 0.55")
        print("    else:")
        print("        # 强切变 - 70m风速更重要")
        print("        w_10m, w_70m = 0.35, 0.65")
        print("    return w_10m, w_70m")
        print("```")
        print("\n原理: 切变越强，表示高度间风速差异越大，轮毂高度(70m)风速越重要")
    
    # 11. 保存结果
    print(f"\n📁 保存结果")
    
    # 详细数据
    results_df = data[['datetime', 'obs_shear_alpha', 'ec_shear_alpha', 'gfs_shear_alpha', 
                      'avg_shear_alpha', 'avg_shear_alpha_direct', 'hour', 'is_daytime']].copy()
    results_df['ec_bias'] = results_df['ec_shear_alpha'] - results_df['obs_shear_alpha']
    results_df['gfs_bias'] = results_df['gfs_shear_alpha'] - results_df['obs_shear_alpha']
    results_df['avg_bias'] = results_df['avg_shear_alpha'] - results_df['obs_shear_alpha']
    
    results_df.to_csv(f"{save_path}/forecast_shear_detailed_data.csv", index=False)
    
    # 汇总报告
    summary_report = {
        'analysis_summary': {
            'sample_size': len(data),
            'best_correlation_model': best_corr_model,
            'best_correlation_value': best_corr_value,
            'best_rmse_model': best_rmse_model,
            'best_rmse_value': best_rmse_value,
            'best_overall_model': best_overall_model,
            'bridge_quality': bridge_quality,
            'recommendation': recommendation,
            'usable_as_bridge': usable
        },
        'correlation_results': correlations,
        'accuracy_results': {
            'RMSE': rmses,
            'MAE': {'EC': ec_mae, 'GFS': gfs_mae, 'EC+GFS平均': avg_mae, '切变直接平均': avg_direct_mae},
            'BIAS': {'EC': ec_bias, 'GFS': gfs_bias, 'EC+GFS平均': avg_bias, '切变直接平均': avg_direct_bias}
        },
        'time_period_results': time_results
    }
    
    import json
    with open(f"{save_path}/forecast_shear_analysis_report.json", 'w', encoding='utf-8') as f:
        json.dump(summary_report, f, indent=2, ensure_ascii=False, default=str)
    
    print(f"  ✅ 详细数据: forecast_shear_detailed_data.csv")
    print(f"  ✅ 分析图表: forecast_shear_analysis.png")
    print(f"  ✅ 汇总报告: forecast_shear_analysis_report.json")
    
    return True

if __name__ == "__main__":
    # 配置路径
    DATA_PATH = "/Users/xiaxin/work/WindForecast_Project/01_Data/processed/imputed_data/changma_imputed_complete.csv"
    SAVE_PATH = "/Users/xiaxin/work/WindForecast_Project/03_Results/forecast_shear_bridge_analysis"
    
    success = analyze_forecast_vs_observed_shear(DATA_PATH, SAVE_PATH)
    
    if success:
        print("\n🎉 预报切变桥梁变量分析完成!")
        print("\n💡 如果分析显示预报切变可以作为桥梁变量，")
        print("   您就可以在第三部分试验中使用预报切变来动态调整权重了！")
    else:
        print("\n⚠️ 分析失败")