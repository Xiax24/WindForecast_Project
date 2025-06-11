
# ===== 基于稳定度的风电预测使用示例 =====
import joblib
import numpy as np
import pandas as pd

# 1. 加载所有稳定度模型
models = {}
stabilities = ['unstable', 'neutral', 'stable']

for stability in stabilities:
    model_path = "/Users/xiaxin/work/WindForecast_Project/03_Results/stability_based_prediction/lightgbm_model_{stability}.pkl"
    models[stability] = joblib.load(model_path)

# 加载特征名称和结果信息
feature_names = joblib.load("/Users/xiaxin/work/WindForecast_Project/03_Results/stability_based_prediction/feature_names.pkl")
results_summary = joblib.load("/Users/xiaxin/work/WindForecast_Project/03_Results/stability_based_prediction/stability_results_summary.pkl")

print("已加载的稳定度模型:", list(models.keys()))
print("特征数量:", len(feature_names))

# 2. 统一的稳定度预测函数
def predict_by_stability(input_data, stability_labels, feature_names):
    """
    基于稳定度进行风电功率预测
    
    Parameters:
    -----------
    input_data : pd.DataFrame
        输入特征数据
    stability_labels : list or array
        对应每条数据的稳定度标签
    feature_names : list
        特征名称列表
    
    Returns:
    --------
    predictions : np.array
        预测结果
    used_models : list
        实际使用的模型列表
    """
    predictions = np.zeros(len(input_data))
    used_models = []
    
    # 准备特征矩阵
    X = input_data[feature_names].values
    
    for stability in set(stability_labels):
        if stability in models:
            # 找到对应稳定度的数据索引
            mask = np.array(stability_labels) == stability
            if np.any(mask):
                # 使用对应模型预测
                predictions[mask] = models[stability].predict(X[mask])
                used_models.append(stability)
                print(f"使用 {stability} 模型预测了 {np.sum(mask)} 条数据")
        else:
            print(f"警告: 没有找到 {stability} 稳定度的模型")
    
    return predictions, used_models

# 3. 处理新数据的完整流程示例
def process_new_data_with_stability(wind_data_path, stability_data_path):
    """
    处理新数据并进行稳定度分类预测的完整流程
    """
    # 加载数据
    wind_data = pd.read_csv(wind_data_path)
    stability_data = pd.read_csv(stability_data_path)
    
    # 时间对齐和合并（根据你的数据结构调整）
    wind_data['datetime'] = pd.to_datetime(wind_data['datetime'])
    stability_data['datetime'] = pd.to_datetime(stability_data['timestamp'])
    
    merged_data = pd.merge(wind_data, stability_data[['datetime', 'stability_final']], 
                          on='datetime', how='inner')
    
    # 处理风向（与训练时保持一致）
    wind_dir_cols = [col for col in merged_data.columns if 'wind_direction' in col]
    for col in wind_dir_cols:
        math_angle = (90 - merged_data[col] + 360) % 360
        wind_dir_rad = np.deg2rad(math_angle)
        
        sin_col = col.replace('wind_direction', 'wind_dir_sin')
        cos_col = col.replace('wind_direction', 'wind_dir_cos')
        merged_data[sin_col] = np.sin(wind_dir_rad)
        merged_data[cos_col] = np.cos(wind_dir_rad)
    
    merged_data = merged_data.drop(columns=wind_dir_cols)
    
    # 进行预测
    predictions, used_models = predict_by_stability(
        merged_data, 
        merged_data['stability_final'].values,
        feature_names
    )
    
    # 添加预测结果
    merged_data['predicted_power'] = predictions
    
    return merged_data, used_models

# 4. 模型性能对比
print("\n各稳定度模型性能:")
for stability, summary in results_summary.items():
    print(f"  {stability}: R²={summary['r2_test']:.3f}, "
          f"RMSE={summary['rmse_test']:.1f}MW, "
          f"样本={summary['sample_count']}")

# 5. 使用建议
print("\n使用建议:")
print("1. 确保输入数据包含所有训练特征")
print("2. 风向数据需要按照训练时的方式处理（sin/cos分量）")
print("3. 稳定度标签必须与训练时的类别一致")
print("4. 对于未见过的稳定度类型，建议使用最相近的稳定度模型")
print("5. 可以根据置信度对预测结果进行加权")

# ===== 误差传播分析扩展 =====
def analyze_stability_error_propagation(obs_data, forecast_data, stability_labels):
    """
    分析不同稳定度条件下的误差传播特性
    """
    results = {}
    
    for stability in set(stability_labels):
        mask = np.array(stability_labels) == stability
        if np.any(mask) and stability in models:
            # 分别用观测和预报数据预测
            P_obs = models[stability].predict(obs_data[mask])
            P_forecast = models[stability].predict(forecast_data[mask])
            
            # 计算误差传播
            propagation_error = P_forecast - P_obs
            
            results[stability] = {
                'rmse_propagation': np.sqrt(np.mean(propagation_error**2)),
                'mean_propagation': np.mean(propagation_error),
                'std_propagation': np.std(propagation_error),
                'sample_count': np.sum(mask)
            }
    
    return results
        