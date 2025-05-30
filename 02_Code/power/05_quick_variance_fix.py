#!/usr/bin/env python3
"""
独立的方差计算诊断脚本
不依赖其他模块，直接从保存的结果文件中分析问题
"""

import numpy as np
import pandas as pd
import joblib
from sklearn.preprocessing import StandardScaler

def load_analysis_data():
    """加载已保存的分析数据"""
    print("📦 加载已保存的分析数据...")
    
    # 路径配置
    DATA_PATH = "/Users/xiaxin/work/WindForecast_Project/03_Results/04error_propagation_data"
    MODEL_PATH = "/Users/xiaxin/work/WindForecast_Project/03_Results/03saved_models"
    RESULTS_PATH = "/Users/xiaxin/work/WindForecast_Project/03_Results/04error_propagation_analysis"
    
    try:
        # 加载预处理数据
        obs_features = np.load(f"{DATA_PATH}/obs_features.npy")
        ecmwf_features = np.load(f"{DATA_PATH}/ecmwf_features.npy")
        gfs_features = np.load(f"{DATA_PATH}/gfs_features.npy")
        actual_power = np.load(f"{DATA_PATH}/actual_power.npy")
        
        # 加载特征名称
        feature_mapping = joblib.load(f"{DATA_PATH}/feature_mapping.pkl")
        feature_names = feature_mapping['obs_features']
        
        # 加载模型
        model = joblib.load(f"{MODEL_PATH}/best_lightgbm_model.pkl")
        
        print(f"✅ 数据加载成功:")
        print(f"  样本数: {len(actual_power)}")
        print(f"  特征数: {len(feature_names)}")
        
        return {
            'obs_features': obs_features,
            'ecmwf_features': ecmwf_features, 
            'gfs_features': gfs_features,
            'actual_power': actual_power,
            'feature_names': feature_names,
            'model': model
        }
        
    except Exception as e:
        print(f"❌ 数据加载失败: {str(e)}")
        return None

def perform_error_decomposition(data):
    """执行误差分解"""
    print("🔬 执行误差分解...")
    
    model = data['model']
    
    # 预测
    P_obs = model.predict(data['obs_features'])
    P_ecmwf = model.predict(data['ecmwf_features'])
    P_gfs = model.predict(data['gfs_features'])
    
    # 误差分解
    modeling_error = P_obs - data['actual_power']
    ecmwf_propagation = P_ecmwf - P_obs
    gfs_propagation = P_gfs - P_obs
    
    print(f"  建模误差 RMSE: {np.sqrt(np.mean(modeling_error**2)):.3f}")
    print(f"  ECMWF传播误差 RMSE: {np.sqrt(np.mean(ecmwf_propagation**2)):.3f}")
    print(f"  GFS传播误差 RMSE: {np.sqrt(np.mean(gfs_propagation**2)):.3f}")
    
    return {
        'P_obs': P_obs,
        'P_ecmwf': P_ecmwf, 
        'P_gfs': P_gfs,
        'modeling_error': modeling_error,
        'ecmwf_propagation': ecmwf_propagation,
        'gfs_propagation': gfs_propagation
    }

def compute_corrected_sensitivity(data, n_samples=3000):
    """重新计算正确的敏感性"""
    print("🔍 重新计算敏感性...")
    
    # 重要特征
    important_features = [
        'obs_wind_speed_70m',
        'obs_wind_speed_50m', 
        'obs_wind_speed_30m',
        'obs_wind_speed_10m',
        'obs_temperature_10m',
        'obs_wind_dir_sin_70m',
        'obs_wind_dir_cos_70m'
    ]
    
    # 过滤存在的特征
    analyzed_features = [f for f in important_features if f in data['feature_names']]
    feature_indices = [data['feature_names'].index(f) for f in analyzed_features]
    
    print(f"  分析特征: {analyzed_features}")
    
    # 采样
    if len(data['obs_features']) > n_samples:
        indices = np.random.choice(len(data['obs_features']), n_samples, replace=False)
        sample_features = data['obs_features'][indices]
    else:
        sample_features = data['obs_features']
    
    # 标准化
    scaler = StandardScaler()
    normalized_features = scaler.fit_transform(sample_features)
    
    # 预测包装器
    def predict_from_normalized(norm_input):
        original_input = scaler.inverse_transform(norm_input)
        return data['model'].predict(original_input)
    
    # 计算敏感性
    sensitivities = {}
    
    for i, feature_name in enumerate(analyzed_features):
        feature_idx = feature_indices[i]
        
        # 扰动计算
        delta = 0.01
        
        features_plus = normalized_features.copy()
        features_plus[:, feature_idx] += delta
        pred_plus = predict_from_normalized(features_plus)
        
        features_minus = normalized_features.copy()
        features_minus[:, feature_idx] -= delta
        pred_minus = predict_from_normalized(features_minus)
        
        # 标准化梯度
        normalized_gradient = (pred_plus - pred_minus) / (2 * delta)
        
        # 特征统计
        feature_std = np.sqrt(scaler.var_[feature_idx])
        feature_mean = scaler.mean_[feature_idx]
        
        # 物理敏感性（特征变化1个标准差的影响）
        physical_sensitivity = np.mean(normalized_gradient) * feature_std
        
        sensitivities[feature_name] = {
            'normalized_gradient': np.mean(normalized_gradient),
            'normalized_gradient_std': np.std(normalized_gradient),
            'physical_sensitivity': physical_sensitivity,
            'abs_physical_sensitivity': abs(physical_sensitivity),
            'feature_std': feature_std,
            'feature_mean': feature_mean
        }
        
        print(f"    {feature_name}:")
        print(f"      标准化梯度: {np.mean(normalized_gradient):.6f}")
        print(f"      物理敏感性: {physical_sensitivity:.6f} kW/std")
    
    return sensitivities, analyzed_features, scaler

def diagnose_variance_calculation(data, errors, sensitivities, analyzed_features):
    """诊断方差计算"""
    print("\n🔍 诊断方差计算...")
    
    feature_indices = [data['feature_names'].index(f) for f in analyzed_features]
    
    # 计算输入误差
    ecmwf_input_errors = data['ecmwf_features'] - data['obs_features']
    gfs_input_errors = data['gfs_features'] - data['obs_features']
    
    # 实际传播误差方差
    actual_vars = {
        'ecmwf': np.var(errors['ecmwf_propagation']),
        'gfs': np.var(errors['gfs_propagation'])
    }
    
    print(f"实际传播误差方差:")
    print(f"  ECMWF: {actual_vars['ecmwf']:.6f}")
    print(f"  GFS: {actual_vars['gfs']:.6f}")
    
    # 方法1：错误的方法（可能导致之前的问题）
    print(f"\n方法1：可能错误的计算（重复标准化）")
    
    wrong_theoretical_vars = {}
    for source, input_errors in [('ecmwf', ecmwf_input_errors), ('gfs', gfs_input_errors)]:
        theoretical_var = 0
        
        for i, feature_name in enumerate(analyzed_features):
            feature_idx = feature_indices[i]
            
            # 错误方法：使用物理敏感性的平方（可能重复考虑了标准差）
            phys_sens = sensitivities[feature_name]['physical_sensitivity']
            input_var = np.var(input_errors[:, feature_idx])
            contribution = phys_sens**2 * input_var
            theoretical_var += contribution
        
        wrong_theoretical_vars[source] = theoretical_var
        print(f"  {source.upper()} 错误理论方差: {theoretical_var:.6f}")
    
    # 方法2：正确的方法（使用标准化梯度）
    print(f"\n方法2：正确的计算（标准化梯度）")
    
    correct_theoretical_vars = {}
    for source, input_errors in [('ecmwf', ecmwf_input_errors), ('gfs', gfs_input_errors)]:
        theoretical_var = 0
        
        print(f"  {source.upper()} 详细计算:")
        
        for i, feature_name in enumerate(analyzed_features):
            feature_idx = feature_indices[i]
            
            # 正确方法：直接使用标准化梯度的平方
            norm_grad = sensitivities[feature_name]['normalized_gradient']
            input_var = np.var(input_errors[:, feature_idx])
            contribution = norm_grad**2 * input_var
            theoretical_var += contribution
            
            print(f"    {feature_name}:")
            print(f"      标准化梯度²: {norm_grad**2:.8f}")
            print(f"      输入方差: {input_var:.8f}")
            print(f"      贡献: {contribution:.8f}")
        
        correct_theoretical_vars[source] = theoretical_var
        print(f"    总理论方差: {theoretical_var:.6f}")
    
    # 方法3：蒙特卡洛验证
    print(f"\n方法3：蒙特卡洛验证")
    mc_vars = monte_carlo_variance_check(data, sensitivities, analyzed_features)
    
    # 比较所有方法
    print(f"\n📊 方法比较:")
    print(f"{'方法':<25} {'ECMWF理论':<15} {'ECMWF比值':<10} {'GFS理论':<15} {'GFS比值':<10}")
    print("-" * 80)
    
    methods = [
        ("错误方法（重复标准化）", wrong_theoretical_vars),
        ("正确方法（标准化梯度）", correct_theoretical_vars),
        ("蒙特卡洛验证", mc_vars)
    ]
    
    for method_name, theoretical_vars in methods:
        ecmwf_ratio = actual_vars['ecmwf'] / theoretical_vars['ecmwf']
        gfs_ratio = actual_vars['gfs'] / theoretical_vars['gfs']
        
        print(f"{method_name:<25} {theoretical_vars['ecmwf']:<15.6f} {ecmwf_ratio:<10.4f} {theoretical_vars['gfs']:<15.6f} {gfs_ratio:<10.4f}")
    
    print(f"{'实际方差':<25} {actual_vars['ecmwf']:<15.6f} {'1.0000':<10} {actual_vars['gfs']:<15.6f} {'1.0000':<10}")
    
    # 推荐
    print(f"\n💡 分析结论:")
    
    correct_ecmwf_ratio = actual_vars['ecmwf'] / correct_theoretical_vars['ecmwf']
    correct_gfs_ratio = actual_vars['gfs'] / correct_theoretical_vars['gfs']
    
    if 0.5 <= correct_ecmwf_ratio <= 2.0 and 0.5 <= correct_gfs_ratio <= 2.0:
        print("✅ 使用标准化梯度的方法结果合理")
        print("✅ 问题确实是重复标准化导致的")
        print("✅ 建议更新主程序使用正确的计算方法")
    else:
        print("⚠️  即使修正后比值仍不理想，可能存在其他问题:")
        print("   - 模型非线性程度较高")
        print("   - 变量间存在显著相关性")
        print("   - 需要考虑高阶项或交互项")
    
    return correct_theoretical_vars, actual_vars

def monte_carlo_variance_check(data, sensitivities, analyzed_features, n_samples=1000):
    """蒙特卡洛方差验证"""
    print("  🎲 蒙特卡洛验证...")
    
    feature_indices = [data['feature_names'].index(f) for f in analyzed_features]
    
    # 采样
    if len(data['obs_features']) > n_samples:
        sample_indices = np.random.choice(len(data['obs_features']), n_samples, replace=False)
        sample_obs = data['obs_features'][sample_indices]
        sample_ecmwf = data['ecmwf_features'][sample_indices]
        sample_gfs = data['gfs_features'][sample_indices]
    else:
        sample_obs = data['obs_features']
        sample_ecmwf = data['ecmwf_features']
        sample_gfs = data['gfs_features']
    
    mc_vars = {}
    
    for source, sample_data in [('ecmwf', sample_ecmwf), ('gfs', sample_gfs)]:
        # 使用线性近似计算每个样本的误差
        linear_errors = []
        
        for j in range(len(sample_obs)):
            linear_error = 0
            
            for i, feature_name in enumerate(analyzed_features):
                feature_idx = feature_indices[i]
                
                # 输入差异
                input_diff = sample_data[j, feature_idx] - sample_obs[j, feature_idx]
                
                # 物理敏感性
                sensitivity = sensitivities[feature_name]['physical_sensitivity']
                
                # 线性近似
                linear_error += sensitivity * input_diff
            
            linear_errors.append(linear_error)
        
        mc_var = np.var(linear_errors)
        mc_vars[source] = mc_var
        
        print(f"    {source.upper()} 蒙特卡洛方差: {mc_var:.6f}")
    
    return mc_vars

def main():
    """主函数"""
    print("🔧 独立方差计算诊断工具")
    print("=" * 60)
    
    # 加载数据
    data = load_analysis_data()
    if data is None:
        return
    
    # 误差分解
    errors = perform_error_decomposition(data)
    
    # 重新计算敏感性
    sensitivities, analyzed_features, scaler = compute_corrected_sensitivity(data)
    
    # 诊断方差计算
    correct_vars, actual_vars = diagnose_variance_calculation(data, errors, sensitivities, analyzed_features)
    
    # 创建修正报告
    print(f"\n📋 修正后的方差分析报告:")
    print("=" * 50)
    
    for source in ['ecmwf', 'gfs']:
        theoretical = correct_vars[source]
        actual = actual_vars[source]
        ratio = actual / theoretical
        
        print(f"{source.upper()}:")
        print(f"  修正后理论方差: {theoretical:.6f}")
        print(f"  实际方差: {actual:.6f}")
        print(f"  比值 (实际/理论): {ratio:.4f}")
        
        if 0.5 <= ratio <= 2.0:
            print(f"  状态: ✅ 良好")
        elif ratio < 0.5:
            print(f"  状态: ⚠️ 理论偏高")
        else:
            print(f"  状态: ⚠️ 理论偏低")
        print()

if __name__ == "__main__":
    main()