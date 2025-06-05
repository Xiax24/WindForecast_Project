#!/usr/bin/env python3
"""
误差传播分析：明确的下一步行动指南
从你现有的数据和模型开始，一步步完成误差分析
"""

import numpy as np
import pandas as pd
import joblib
import matplotlib.pyplot as plt

def clear_step_by_step_guide():
    """清晰的分步指南"""
    print("=" * 80)
    print("🎯 误差传播分析：你的下一步行动计划")
    print("=" * 80)
    
    print("📋 你现在的状况:")
    print("✅ 训练好的LightGBM模型")
    print("✅ obs_features (观测数据)")
    print("✅ ecmwf_features (ECMWF预报)")
    print("✅ gfs_features (GFS预报)")
    print("✅ SHAP重要特征列表")
    
    print("\n🎯 下一步要做的事情（按顺序）：")
    
    step1_basic_error_decomposition()
    step2_detailed_sensitivity()
    step3_monte_carlo_validation()
    step4_comprehensive_analysis()

def step1_basic_error_decomposition():
    """第一步：基础误差分解"""
    print("\n" + "="*60)
    print("🔥 第一步：基础误差分解（最重要！先做这个）")
    print("="*60)
    
    print("\n💡 这一步的目的：")
    print("回答问题：当我们用预报数据而不是观测数据时，功率预测误差增加了多少？")
    
    print("\n📝 具体要做的事：")
    print("""
def step1_basic_error_decomposition():
    '''第一步：基础误差分解 - 立即可执行'''
    
    # 1. 加载你的数据和模型
    model = joblib.load('your_lightgbm_model.pkl')
    obs_features = np.load('obs_features.npy')
    ecmwf_features = np.load('ecmwf_features.npy') 
    gfs_features = np.load('gfs_features.npy')
    actual_power = np.load('actual_power.npy')  # 真实功率
    
    print("📊 数据概况:")
    print(f"样本数量: {len(obs_features)}")
    print(f"特征数量: {obs_features.shape[1]}")
    
    # 2. 三次功率预测
    print("\\n🔮 进行三次功率预测...")
    P_obs = model.predict(obs_features)      # 用观测数据预测
    P_ecmwf = model.predict(ecmwf_features)  # 用ECMWF数据预测  
    P_gfs = model.predict(gfs_features)      # 用GFS数据预测
    
    # 3. 误差分解计算
    print("\\n📐 计算各种误差...")
    
    # 建模误差：即使用观测数据，模型也有误差
    modeling_error = P_obs - actual_power
    modeling_rmse = np.sqrt(np.mean(modeling_error**2))
    
    # 输入误差传播：预报数据与观测数据的差异导致的额外误差
    ecmwf_propagation = P_ecmwf - P_obs
    ecmwf_prop_rmse = np.sqrt(np.mean(ecmwf_propagation**2))
    
    gfs_propagation = P_gfs - P_obs  
    gfs_prop_rmse = np.sqrt(np.mean(gfs_propagation**2))
    
    # 总误差
    ecmwf_total_error = P_ecmwf - actual_power
    ecmwf_total_rmse = np.sqrt(np.mean(ecmwf_total_error**2))
    
    gfs_total_error = P_gfs - actual_power
    gfs_total_rmse = np.sqrt(np.mean(gfs_total_error**2))
    
    # 4. 结果输出
    print("\\n🎯 误差分解结果:")
    print("="*50)
    print(f"建模误差 RMSE:        {modeling_rmse:.1f} kW")
    print(f"ECMWF传播误差 RMSE:   {ecmwf_prop_rmse:.1f} kW") 
    print(f"GFS传播误差 RMSE:     {gfs_prop_rmse:.1f} kW")
    print(f"ECMWF总误差 RMSE:     {ecmwf_total_rmse:.1f} kW")
    print(f"GFS总误差 RMSE:       {gfs_total_rmse:.1f} kW")
    
    print(f"\\n💡 关键发现:")
    print(f"• 输入误差传播占总误差的 {ecmwf_prop_rmse/ecmwf_total_rmse*100:.0f}%")
    print(f"• ECMWF vs GFS: {ecmwf_prop_rmse:.1f} vs {gfs_prop_rmse:.1f} kW")
    
    # 5. 验证误差分解的正确性
    print(f"\\n✅ 验证误差分解:")
    reconstructed = np.sqrt(np.mean((ecmwf_propagation + modeling_error)**2))
    print(f"总误差: {ecmwf_total_rmse:.3f} kW")
    print(f"重构误差: {reconstructed:.3f} kW") 
    print(f"差异: {abs(ecmwf_total_rmse - reconstructed):.6f} kW")
    
    return {
        'modeling_rmse': modeling_rmse,
        'ecmwf_prop_rmse': ecmwf_prop_rmse,
        'gfs_prop_rmse': gfs_prop_rmse,
        'ecmwf_total_rmse': ecmwf_total_rmse,
        'gfs_total_rmse': gfs_total_rmse
    }
    """)
    
    print("\n🎯 第一步完成后你就知道：")
    print("✓ 气象预报误差对功率预测的具体影响（多少kW）")
    print("✓ ECMWF和GFS哪个更好") 
    print("✓ 建模误差 vs 输入误差的相对重要性")
    print("✓ 这些数字可以直接写到论文里！")

def step2_detailed_sensitivity():
    """第二步：详细敏感性分析"""
    print("\n" + "="*60)
    print("🔍 第二步：详细敏感性分析（深入理解）")
    print("="*60)
    
    print("\n💡 这一步的目的：")
    print("回答问题：哪个气象变量的预报误差对功率预测影响最大？")
    
    print("\n📝 具体要做的事：")
    print("""
def step2_sensitivity_analysis(model, obs_features, important_features):
    '''第二步：针对SHAP重要特征进行敏感性分析'''
    
    # 1. 选择要分析的特征（使用SHAP结果）
    # important_features = ['obs_wind_speed_70m', 'obs_temperature_10m', ...]
    
    print(f"🔍 分析 {len(important_features)} 个重要特征的敏感性")
    
    sensitivities = {}
    
    for i, feature_name in enumerate(important_features):
        print(f"\\n计算 {feature_name} 的敏感性...")
        
        # 找到特征在数组中的索引
        feature_idx = feature_names.index(feature_name)
        
        # 确定合理的扰动量
        if 'wind_speed' in feature_name:
            delta = 0.1  # 0.1 m/s
            unit_name = "m/s"
        elif 'temperature' in feature_name:
            delta = 0.1  # 0.1°C  
            unit_name = "°C"
        else:
            delta = 0.01
            unit_name = "units"
        
        # 计算敏感性（中心差分）
        features_plus = obs_features.copy()
        features_plus[:, feature_idx] += delta
        
        features_minus = obs_features.copy()
        features_minus[:, feature_idx] -= delta
        
        pred_plus = model.predict(features_plus)
        pred_minus = model.predict(features_minus)
        
        sensitivity = (pred_plus - pred_minus) / (2 * delta)
        mean_sensitivity = np.mean(sensitivity)
        
        sensitivities[feature_name] = {
            'sensitivity': mean_sensitivity,
            'unit': unit_name,
            'abs_sensitivity': abs(mean_sensitivity)
        }
        
        print(f"  敏感性: {mean_sensitivity:.3f} kW/{unit_name}")
    
    # 2. 特征重要性排序
    print(f"\\n🏆 敏感性排序:")
    sorted_features = sorted(important_features, 
                           key=lambda x: sensitivities[x]['abs_sensitivity'], 
                           reverse=True)
    
    for i, feature in enumerate(sorted_features):
        sens = sensitivities[feature]['sensitivity']
        unit = sensitivities[feature]['unit']
        print(f"  {i+1}. {feature}: {abs(sens):.2f} kW/{unit}")
    
    return sensitivities
    """)
    
    print("\n🎯 第二步完成后你就知道：")
    print("✓ 风速、温度等变量的敏感性排序")
    print("✓ 每个变量变化1个单位对功率的影响")
    print("✓ 哪些变量需要重点提高预报精度")

def step3_monte_carlo_validation():
    """第三步：蒙特卡洛验证"""
    print("\n" + "="*60)
    print("🎲 第三步：蒙特卡洛验证（可选，验证理论）")
    print("="*60)
    
    print("\n💡 这一步的目的：")
    print("验证前面的敏感性分析是否准确，处理模型的非线性")
    
    print("\n📝 具体要做的事：")
    print("""
def step3_monte_carlo_validation(model, obs_features, ecmwf_features):
    '''第三步：蒙特卡洛验证敏感性分析'''
    
    print("🎲 蒙特卡洛验证...")
    
    # 1. 随机采样进行验证
    n_samples = 2000
    indices = np.random.choice(len(obs_features), n_samples, replace=False)
    
    sample_obs = obs_features[indices]
    sample_ecmwf = ecmwf_features[indices]
    
    # 2. 计算实际的误差传播
    P_obs_sample = model.predict(sample_obs)
    P_ecmwf_sample = model.predict(sample_ecmwf)
    
    actual_propagation = P_ecmwf_sample - P_obs_sample
    actual_rmse = np.sqrt(np.mean(actual_propagation**2))
    
    print(f"实际误差传播 RMSE: {actual_rmse:.2f} kW")
    
    # 3. 分析误差传播的分布特征
    print(f"误差传播统计:")
    print(f"  均值: {np.mean(actual_propagation):+.2f} kW")
    print(f"  标准差: {np.std(actual_propagation):.2f} kW")
    print(f"  25分位: {np.percentile(actual_propagation, 25):+.2f} kW")
    print(f"  75分位: {np.percentile(actual_propagation, 75):+.2f} kW")
    
    return {
        'actual_rmse': actual_rmse,
        'propagation_stats': {
            'mean': np.mean(actual_propagation),
            'std': np.std(actual_propagation),
            'q25': np.percentile(actual_propagation, 25),
            'q75': np.percentile(actual_propagation, 75)
        }
    }
    """)
    
    print("\n🎯 第三步完成后你就知道：")
    print("✓ 误差传播的实际分布特征")
    print("✓ 敏感性分析的准确性")
    print("✓ 模型非线性的影响程度")

def step4_comprehensive_analysis():
    """第四步：综合分析和应用指导"""
    print("\n" + "="*60)
    print("📊 第四步：综合分析和应用指导（论文写作）")
    print("="*60)
    
    print("\n💡 这一步的目的：")
    print("整合所有结果，为实际应用和论文写作提供指导")
    
    print("\n📝 具体要做的事：")
    print("""
def step4_comprehensive_analysis(step1_results, step2_results, step3_results):
    '''第四步：综合分析和应用指导'''
    
    print("📊 综合分析报告")
    print("="*50)
    
    # 1. 误差贡献分析
    print("\\n1️⃣ 误差来源分析:")
    modeling_pct = step1_results['modeling_rmse'] / step1_results['ecmwf_total_rmse'] * 100
    propagation_pct = step1_results['ecmwf_prop_rmse'] / step1_results['ecmwf_total_rmse'] * 100
    
    print(f"  建模误差贡献: {modeling_pct:.0f}%")
    print(f"  输入误差传播: {propagation_pct:.0f}%") 
    print(f"  结论: {'气象预报误差' if propagation_pct > 50 else '建模误差'}是主要问题")
    
    # 2. 数据源比较
    print("\\n2️⃣ 气象数据源比较:")
    ecmwf_rmse = step1_results['ecmwf_prop_rmse']
    gfs_rmse = step1_results['gfs_prop_rmse']
    
    better_source = 'ECMWF' if ecmwf_rmse < gfs_rmse else 'GFS'
    improvement = abs(ecmwf_rmse - gfs_rmse) / max(ecmwf_rmse, gfs_rmse) * 100
    
    print(f"  ECMWF传播误差: {ecmwf_rmse:.1f} kW")
    print(f"  GFS传播误差:   {gfs_rmse:.1f} kW")
    print(f"  推荐使用: {better_source} (优势: {improvement:.0f}%)")
    
    # 3. 敏感变量分析
    print("\\n3️⃣ 关键敏感变量:")
    # 从step2_results中提取最敏感的3个变量
    top_3_vars = list(step2_results.keys())[:3]  # 假设已排序
    
    for i, var in enumerate(top_3_vars):
        sens = step2_results[var]['sensitivity']
        unit = step2_results[var]['unit']
        print(f"  {i+1}. {var}: {abs(sens):.2f} kW/{unit}")
    
    print(f"  建议: 重点提高{top_3_vars[0]}的预报精度")
    
    # 4. 实际应用指导
    print("\\n4️⃣ 实际应用指导:")
    print(f"  功率预测不确定性: ±{step1_results['ecmwf_prop_rmse']:.0f} kW")
    print(f"  建议安全裕度: {step1_results['ecmwf_prop_rmse'] * 1.5:.0f} kW")
    print(f"  重点改进方向: {'提高气象预报精度' if propagation_pct > 50 else '改进功率预测模型'}")
    
    # 5. 论文写作要点
    print("\\n5️⃣ 论文写作要点:")
    print("  ✓ 量化了输入误差传播的具体影响")
    print("  ✓ 比较了不同气象数据源的性能")
    print("  ✓ 识别了关键的敏感变量")
    print("  ✓ 为实际应用提供了定量指导")
    
    return {
        'summary': {
            'main_error_source': '气象预报误差' if propagation_pct > 50 else '建模误差',
            'best_forecast_source': better_source,
            'uncertainty_range': step1_results['ecmwf_prop_rmse'],
            'top_sensitive_vars': top_3_vars
        }
    }
    """)
    
    print("\n🎯 第四步完成后你就有：")
    print("✓ 完整的误差传播分析报告")
    print("✓ 实际应用的定量指导")
    print("✓ 论文的核心结果和结论")
    print("✓ 改进方向的明确建议")

def immediate_action_plan():
    """立即行动计划"""
    print("\n" + "="*80)
    print("🚀 立即行动计划")
    print("="*80)
    
    print("\n📅 时间安排建议:")
    print("第1天: 完成第一步（基础误差分解）")
    print("第2天: 完成第二步（敏感性分析）") 
    print("第3天: 完成第三步（蒙特卡洛验证）")
    print("第4天: 完成第四步（综合分析）")
    
    print("\n💻 今天就开始写代码:")
    print("""
# 立即可以执行的代码框架
import numpy as np
import joblib

def main():
    # 第一步：基础误差分解
    print("开始第一步：基础误差分解")
    
    # 加载你的数据
    model = joblib.load('你的模型路径')
    obs_features = np.load('obs_features.npy')
    ecmwf_features = np.load('ecmwf_features.npy')
    gfs_features = np.load('gfs_features.npy') 
    actual_power = np.load('actual_power.npy')
    
    # 三次预测
    P_obs = model.predict(obs_features)
    P_ecmwf = model.predict(ecmwf_features)
    P_gfs = model.predict(gfs_features)
    
    # 误差计算
    modeling_rmse = np.sqrt(np.mean((P_obs - actual_power)**2))
    ecmwf_prop_rmse = np.sqrt(np.mean((P_ecmwf - P_obs)**2))
    gfs_prop_rmse = np.sqrt(np.mean((P_gfs - P_obs)**2))
    
    print(f"建模误差: {modeling_rmse:.1f} kW")
    print(f"ECMWF传播误差: {ecmwf_prop_rmse:.1f} kW")
    print(f"GFS传播误差: {gfs_prop_rmse:.1f} kW")
    
    print("第一步完成！你已经有了核心结果！")

if __name__ == "__main__":
    main()
    """)
    
    print("\n🎯 关键提醒:")
    print("• 不要想太复杂，先把第一步做完")
    print("• 第一步的结果就足够写论文了")
    print("• 每一步都有明确的目标和输出")
    print("• 按顺序做，不要跳跃")

if __name__ == "__main__":
    clear_step_by_step_guide()
    immediate_action_plan()