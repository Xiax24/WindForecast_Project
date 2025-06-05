#!/usr/bin/env python3
"""
第二步：敏感性分析 - 基于你的SHAP结果
分析每个重要特征的敏感性，找出哪个变量的预报误差影响最大
"""

import numpy as np
import pandas as pd
import joblib
import matplotlib.pyplot as plt

def step2_sensitivity_analysis():
    """第二步：敏感性分析 - 立即可执行"""
    print("=" * 80)
    print("🔍 第二步：敏感性分析")
    print("=" * 80)
    
    # 你的SHAP重要特征（按重要性排序）
    important_features = [
        'obs_wind_speed_10m',  # 最重要：10m风速 
        'obs_wind_speed_70m',  # 70m风速
        'obs_temperature_10m', # 温度影响
        'obs_wind_speed_30m',  # 30m风速
        'obs_wind_speed_50m',  # 50m风速
        'obs_wind_dir_sin_50m', # 50m风向sin
        'obs_wind_dir_cos_50m', # 50m风向cos
        'obs_wind_dir_sin_70m', # 70m风向sin
        'obs_wind_dir_cos_70m', # 70m风向cos
        'obs_wind_dir_sin_10m'  # 10m风向sin
    ]
    
    print(f"📊 分析 {len(important_features)} 个SHAP重要特征的敏感性")
    
    # 加载数据（使用你的实际路径）
    model = load_your_model()  # 替换为你的模型加载代码
    obs_features, feature_names = load_your_data()  # 替换为你的数据加载代码
    
    # 选择分析样本（减少计算量）
    n_samples = 3000
    if len(obs_features) > n_samples:
        indices = np.random.choice(len(obs_features), n_samples, replace=False)
        sample_features = obs_features[indices]
    else:
        sample_features = obs_features
    
    print(f"使用 {len(sample_features)} 个样本进行敏感性分析")
    
    # 敏感性分析
    sensitivities = {}
    
    for i, feature_name in enumerate(important_features):
        print(f"\n🔍 计算 {feature_name} 的敏感性... ({i+1}/{len(important_features)})")
        
        # 找到特征索引
        try:
            feature_idx = feature_names.index(feature_name)
        except ValueError:
            print(f"  ⚠️ 特征 {feature_name} 未找到，跳过")
            continue
        
        # 确定扰动量和单位
        if 'wind_speed' in feature_name:
            delta = 0.1  # 0.1 m/s
            unit_name = "m/s"
            unit_delta = 1.0  # 1 m/s 的敏感性
        elif 'temperature' in feature_name:
            delta = 0.1  # 0.1°C
            unit_name = "°C"
            unit_delta = 1.0  # 1°C 的敏感性
        elif 'wind_dir' in feature_name:
            delta = 0.01  # 0.01 units (sin/cos值)
            unit_name = "0.01 units"
            unit_delta = 0.1  # 0.1 units 的敏感性
        else:
            delta = 0.01
            unit_name = "units"
            unit_delta = 0.1
        
        # 计算小扰动敏感性（用于梯度估计）
        features_plus = sample_features.copy()
        features_plus[:, feature_idx] += delta
        
        features_minus = sample_features.copy()
        features_minus[:, feature_idx] -= delta
        
        pred_plus = model.predict(features_plus)
        pred_minus = model.predict(features_minus)
        
        # 梯度计算
        gradient = (pred_plus - pred_minus) / (2 * delta)
        mean_gradient = np.mean(gradient)
        std_gradient = np.std(gradient)
        
        # 计算实用的单位敏感性
        features_unit_plus = sample_features.copy()
        features_unit_plus[:, feature_idx] += unit_delta
        
        pred_unit = model.predict(features_unit_plus)
        pred_baseline = model.predict(sample_features)
        
        unit_sensitivity = np.mean(pred_unit - pred_baseline) / unit_delta
        
        # 存储结果
        sensitivities[feature_name] = {
            'gradient': mean_gradient,
            'gradient_std': std_gradient,
            'unit_sensitivity': unit_sensitivity,
            'abs_unit_sensitivity': abs(unit_sensitivity),
            'unit_name': unit_name,
            'delta_used': delta,
            'unit_delta': unit_delta
        }
        
        print(f"  梯度: {mean_gradient:.3f} ± {std_gradient:.3f} kW/{unit_name}")
        print(f"  单位敏感性: {unit_sensitivity:.3f} kW/{unit_name.replace('0.01 ', '').replace('0.1 ', '')}")
    
    # 结果分析和排序
    analyze_sensitivity_results(sensitivities, important_features)
    
    return sensitivities

def analyze_sensitivity_results(sensitivities, important_features):
    """分析敏感性结果"""
    print("\n" + "="*60)
    print("📊 敏感性分析结果")
    print("="*60)
    
    # 按单位敏感性排序
    valid_features = [f for f in important_features if f in sensitivities]
    sorted_features = sorted(valid_features, 
                           key=lambda x: sensitivities[x]['abs_unit_sensitivity'], 
                           reverse=True)
    
    print(f"\n🏆 特征敏感性排序（按影响大小）:")
    print("-" * 50)
    
    for i, feature in enumerate(sorted_features):
        sens_data = sensitivities[feature]
        unit_sens = sens_data['unit_sensitivity']
        unit_name = sens_data['unit_name'].replace('0.01 ', '').replace('0.1 ', '')
        
        # 特征类型
        if 'wind_speed' in feature:
            feature_type = "🌪️ 风速"
            height = feature.split('_')[-1]
            display_name = f"{feature_type} ({height})"
        elif 'temperature' in feature:
            feature_type = "🌡️ 温度"
            display_name = feature_type
        elif 'wind_dir' in feature:
            feature_type = "🧭 风向"
            height = feature.split('_')[-1]
            direction = 'sin' if 'sin' in feature else 'cos'
            display_name = f"{feature_type} ({direction}, {height})"
        else:
            display_name = feature
        
        print(f"  {i+1:2d}. {display_name:<25} {abs(unit_sens):6.2f} kW/{unit_name}")
    
    # 分类分析
    print(f"\n📈 分类敏感性分析:")
    print("-" * 50)
    
    wind_speed_sens = []
    temperature_sens = []
    wind_dir_sens = []
    
    for feature, data in sensitivities.items():
        if 'wind_speed' in feature:
            wind_speed_sens.append(abs(data['unit_sensitivity']))
        elif 'temperature' in feature:
            temperature_sens.append(abs(data['unit_sensitivity']))
        elif 'wind_dir' in feature:
            wind_dir_sens.append(abs(data['unit_sensitivity']))
    
    if wind_speed_sens:
        print(f"🌪️ 风速敏感性:")
        print(f"    平均: {np.mean(wind_speed_sens):.2f} kW/(m/s)")
        print(f"    最大: {np.max(wind_speed_sens):.2f} kW/(m/s)")
        print(f"    最小: {np.min(wind_speed_sens):.2f} kW/(m/s)")
    
    if temperature_sens:
        print(f"🌡️ 温度敏感性:")
        print(f"    平均: {np.mean(temperature_sens):.2f} kW/°C")
    
    if wind_dir_sens:
        print(f"🧭 风向敏感性:")
        print(f"    平均: {np.mean(wind_dir_sens):.2f} kW/0.1units")
    
    # 关键发现
    print(f"\n💡 关键发现:")
    print("-" * 50)
    
    top_3 = sorted_features[:3]
    print(f"最敏感的3个变量:")
    for i, feature in enumerate(top_3):
        sens = sensitivities[feature]['unit_sensitivity']
        unit = sensitivities[feature]['unit_name'].replace('0.01 ', '').replace('0.1 ', '')
        print(f"  {i+1}. {feature}: {abs(sens):.2f} kW/{unit}")
    
    # 实际应用建议
    print(f"\n📋 实际应用建议:")
    print("-" * 50)
    
    most_sensitive = sorted_features[0]
    most_sens_value = sensitivities[most_sensitive]['abs_unit_sensitivity']
    
    print(f"🎯 重点关注: {most_sensitive}")
    print(f"   影响: {most_sens_value:.1f} kW per unit")
    print(f"   建议: 优先提高该变量的预报精度")
    
    if 'wind_speed' in most_sensitive:
        print(f"   实际含义: 风速预报误差1 m/s → 功率误差{most_sens_value:.1f} kW")
    elif 'temperature' in most_sensitive:
        print(f"   实际含义: 温度预报误差1°C → 功率误差{most_sens_value:.1f} kW")
    
    # 与之前的误差分解结果结合
    print(f"\n🔗 与误差分解结果结合:")
    print("-" * 50)
    print(f"你之前的结果显示:")
    print(f"  ECMWF传播误差: 36.7 kW")
    print(f"  GFS传播误差: 39.1 kW")
    print(f"")
    print(f"现在我们知道这些误差主要来自:")
    for i, feature in enumerate(top_3):
        print(f"  {i+1}. {feature}的预报误差")

def create_sensitivity_visualization(sensitivities):
    """Create sensitivity visualization"""
    print(f"\n📊 Generating sensitivity visualization...")
    
    # Prepare data
    features = list(sensitivities.keys())
    unit_sensitivities = [sensitivities[f]['abs_unit_sensitivity'] for f in features]
    
    # Create figure
    plt.figure(figsize=(12, 8))
    
    # Horizontal bar chart
    bars = plt.barh(range(len(features)), unit_sensitivities, 
                   color=['skyblue' if 'wind_speed' in f 
                         else 'lightcoral' if 'temperature' in f 
                         else 'lightgreen' for f in features])
    
    plt.yticks(range(len(features)), [f.replace('obs_', '').replace('_', ' ') for f in features])
    plt.xlabel('Sensitivity (kW/unit)')
    plt.title('Feature Sensitivity Analysis Results')
    plt.grid(axis='x', alpha=0.3)
    
    # Add value labels
    for i, (feature, value) in enumerate(zip(features, unit_sensitivities)):
        plt.text(value + max(unit_sensitivities)*0.01, i, f'{value:.2f}', 
                va='center', fontsize=9)
    
    # Add legend
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor='skyblue', label='Wind Speed'),
        Patch(facecolor='lightcoral', label='Temperature'), 
        Patch(facecolor='lightgreen', label='Wind Direction')
    ]
    plt.legend(handles=legend_elements, loc='lower right')
    
    plt.tight_layout()
    plt.savefig('sensitivity_analysis.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    print(f"✅ Chart saved as 'sensitivity_analysis.png'")

def load_your_model():
    """加载你的模型 - 替换为实际路径"""
    # 替换为你的实际模型路径
    MODEL_PATH = "/Users/xiaxin/work/WindForecast_Project/03_Results/03saved_models/best_lightgbm_model.pkl"
    return joblib.load(MODEL_PATH)

def load_your_data():
    """加载你的数据 - 替换为实际路径"""
    # 替换为你的实际数据路径
    DATA_PATH = "/Users/xiaxin/work/WindForecast_Project/03_Results/04error_propagation_data"
    
    obs_features = np.load(f"{DATA_PATH}/obs_features.npy")
    feature_mapping = joblib.load(f"{DATA_PATH}/feature_mapping.pkl")
    feature_names = feature_mapping['obs_features']
    
    return obs_features, feature_names

if __name__ == "__main__":
    print("🚀 开始第二步：敏感性分析")
    print("基于你的SHAP结果，分析每个重要特征的敏感性")
    
    try:
        sensitivities = step2_sensitivity_analysis()
        create_sensitivity_visualization(sensitivities)
        
        print("\n🎉 第二步完成！")
        print("现在你知道了:")
        print("✓ 每个特征的具体敏感性数值")
        print("✓ 哪个变量的预报误差影响最大") 
        print("✓ 风速、温度、风向的相对重要性")
        print("✓ 具体的改进建议")
        
    except Exception as e:
        print(f"❌ 执行出错: {e}")
        print("请检查数据路径和模型路径是否正确")