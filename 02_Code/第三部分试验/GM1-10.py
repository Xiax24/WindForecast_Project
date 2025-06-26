#!/usr/bin/env python3
"""
G-M1-10m 简化基准试验
使用GFS-WRF原始10m风速进行功率预测
简化版本：LightGBM + RMSE + 相关系数
"""

import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import os
import json
import warnings
warnings.filterwarnings('ignore')

# 设置随机种子
RANDOM_STATE = 42

def prepare_simple_features(data, wind_col):
    """准备简单特征"""
    features = pd.DataFrame()
    
    # 主要风速特征
    features['wind_speed'] = data[wind_col]
    features['wind_speed_2'] = data[wind_col] ** 2
    features['wind_speed_3'] = data[wind_col] ** 3
    
    # 基础时间特征
    features['hour'] = data['datetime'].dt.hour
    features['month'] = data['datetime'].dt.month
    features['is_daytime'] = ((data['datetime'].dt.hour >= 6) & 
                             (data['datetime'].dt.hour < 18)).astype(int)
    
    # 简单滞后特征
    features['wind_lag_1h'] = data[wind_col].shift(1)
    features['wind_lag_24h'] = data[wind_col].shift(24)
    
    # 填充NaN
    features = features.fillna(method='bfill').fillna(method='ffill')
    
    return features

def create_train_test_split(data, test_size=0.2, indices_path=None):
    """创建或加载训练测试集划分"""
    
    if indices_path and os.path.exists(indices_path):
        # 加载已有划分
        with open(indices_path, 'r') as f:
            indices = json.load(f)
        train_indices = indices['train_indices']
        test_indices = indices['test_indices']
        print(f"加载已有划分: 训练集{len(train_indices)}, 测试集{len(test_indices)}")
    else:
        # 创建新划分（时间序列划分）
        data_sorted = data.sort_values('datetime').reset_index(drop=True)
        n_samples = len(data_sorted)
        split_idx = int(n_samples * (1 - test_size))
        
        train_indices = list(range(split_idx))
        test_indices = list(range(split_idx, n_samples))
        
        # 保存划分
        if indices_path:
            os.makedirs(os.path.dirname(indices_path), exist_ok=True)
            indices_data = {
                'train_indices': train_indices,
                'test_indices': test_indices,
                'split_date': data_sorted.iloc[split_idx]['datetime'].isoformat(),
                'train_size': len(train_indices),
                'test_size': len(test_indices)
            }
            with open(indices_path, 'w') as f:
                json.dump(indices_data, f, indent=2)
            print(f"新建划分已保存: 训练集{len(train_indices)}, 测试集{len(test_indices)}")
    
    return train_indices, test_indices

def train_lightgbm_model(X_train, y_train, X_test, y_test):
    """训练LightGBM模型"""
    
    # LightGBM参数
    params = {
        'objective': 'regression',
        'metric': 'rmse',
        'boosting_type': 'gbdt',
        'num_leaves': 31,
        'learning_rate': 0.1,
        'feature_fraction': 0.9,
        'bagging_fraction': 0.8,
        'bagging_freq': 5,
        'verbose': -1,
        'random_state': RANDOM_STATE
    }
    
    # 创建数据集
    train_data = lgb.Dataset(X_train, label=y_train)
    valid_data = lgb.Dataset(X_test, label=y_test, reference=train_data)
    
    # 训练模型
    model = lgb.train(
        params,
        train_data,
        valid_sets=[valid_data],
        num_boost_round=100,
        callbacks=[lgb.early_stopping(stopping_rounds=10), lgb.log_evaluation(0)]
    )
    
    return model

def evaluate_model(model, X_test, y_test):
    """评估模型"""
    y_pred = model.predict(X_test, num_iteration=model.best_iteration)
    
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    correlation = np.corrcoef(y_test, y_pred)[0, 1]
    
    return {
        'RMSE': rmse,
        'Correlation': correlation
    }, y_pred

def run_gm1_10m_experiment(data_path, save_dir, indices_path=None):
    """运行G-M1-10m简化试验"""
    
    print("=" * 60)
    print("🌬️ G-M1-10m 简化试验 (LightGBM)")
    print("=" * 60)
    
    os.makedirs(save_dir, exist_ok=True)
    
    # 1. 数据加载
    print("📂 加载数据...")
    data = pd.read_csv(data_path)
    data['datetime'] = pd.to_datetime(data['datetime'])
    
    # 数据清理
    data = data.dropna(subset=['gfs_wind_speed_10m', 'power'])
    data = data[(data['gfs_wind_speed_10m'] >= 0) & (data['gfs_wind_speed_10m'] <= 50)]
    data = data[(data['power'] >= 0)]
    data = data.sort_values('datetime').reset_index(drop=True)
    
    print(f"清理后数据: {len(data)} 样本")
    
    # 2. 特征准备
    print("🔧 准备特征...")
    features = prepare_simple_features(data, 'gfs_wind_speed_10m')
    target = data['power'].values
    
    print(f"特征数量: {features.shape[1]}")
    
    # 3. 划分数据集
    print("✂️ 划分训练测试集...")
    train_indices, test_indices = create_train_test_split(data, indices_path=indices_path)
    
    X_train = features.iloc[train_indices]
    X_test = features.iloc[test_indices]
    y_train = target[train_indices]
    y_test = target[test_indices]
    
    # 4. 训练模型
    print("🚀 训练LightGBM模型...")
    model = train_lightgbm_model(X_train, y_train, X_test, y_test)
    
    # 5. 评估模型
    print("📊 评估模型...")
    metrics, y_pred = evaluate_model(model, X_test, y_test)
    
    print(f"RMSE: {metrics['RMSE']:.4f}")
    print(f"相关系数: {metrics['Correlation']:.4f}")
    
    # 6. 保存结果
    print("💾 保存结果...")
    
    # 保存预测结果
    results_df = pd.DataFrame({
        'datetime': data.iloc[test_indices]['datetime'].values,
        'actual_power': y_test,
        'predicted_power': y_pred,
        'wind_speed': data.iloc[test_indices]['gfs_wind_speed_10m'].values
    })
    results_df.to_csv(os.path.join(save_dir, 'G-M1-10m_results.csv'), index=False)
    
    # 保存指标
    with open(os.path.join(save_dir, 'G-M1-10m_metrics.json'), 'w') as f:
        json.dump(metrics, f, indent=2)
    
    # 保存模型
    model.save_model(os.path.join(save_dir, 'G-M1-10m_model.txt'))
    
    print("✅ G-M1-10m试验完成!")
    print("=" * 60)
    
    return metrics, results_df

# 为后续试验准备的通用函数
def run_wind_power_experiment(data_path, save_dir, indices_path, 
                             wind_col, model_name, 
                             use_corrected=False, weights=None):
    """通用风电试验函数，可用于所有14个试验"""
    
    print(f"🌬️ {model_name} 试验")
    print(f"风速列: {wind_col}")
    if weights:
        print(f"权重: {weights}")
    
    os.makedirs(save_dir, exist_ok=True)
    
    # 加载数据
    data = pd.read_csv(data_path)
    data['datetime'] = pd.to_datetime(data['datetime'])
    
    # 数据清理
    required_cols = [col for col in [wind_col] if isinstance(wind_col, str)]
    if isinstance(wind_col, list):
        required_cols = wind_col
    
    data = data.dropna(subset=required_cols + ['power'])
    for col in required_cols:
        data = data[(data[col] >= 0) & (data[col] <= 50)]
    data = data[data['power'] >= 0]
    data = data.sort_values('datetime').reset_index(drop=True)
    
    # 处理权重融合
    if weights and isinstance(wind_col, list):
        # 多风速加权融合
        wind_series = np.zeros(len(data))
        for i, col in enumerate(wind_col):
            wind_series += weights[i] * data[col].values
        data['fused_wind'] = wind_series
        feature_wind_col = 'fused_wind'
    else:
        feature_wind_col = wind_col
    
    # 特征准备
    features = prepare_simple_features(data, feature_wind_col)
    target = data['power'].values
    
    # 划分数据集
    train_indices, test_indices = create_train_test_split(data, indices_path=indices_path)
    
    X_train = features.iloc[train_indices]
    X_test = features.iloc[test_indices]
    y_train = target[train_indices]
    y_test = target[test_indices]
    
    # 训练和评估
    model = train_lightgbm_model(X_train, y_train, X_test, y_test)
    metrics, y_pred = evaluate_model(model, X_test, y_test)
    
    print(f"RMSE: {metrics['RMSE']:.4f}")
    print(f"相关系数: {metrics['Correlation']:.4f}")
    
    # 保存结果
    results_df = pd.DataFrame({
        'datetime': data.iloc[test_indices]['datetime'].values,
        'actual_power': y_test,
        'predicted_power': y_pred
    })
    results_df.to_csv(os.path.join(save_dir, f'{model_name}_results.csv'), index=False)
    
    with open(os.path.join(save_dir, f'{model_name}_metrics.json'), 'w') as f:
        json.dump(metrics, f, indent=2)
    
    model.save_model(os.path.join(save_dir, f'{model_name}_model.txt'))
    
    return metrics, results_df

if __name__ == "__main__":
    # 配置路径
    DATA_PATH = "/Users/xiaxin/work/WindForecast_Project/01_Data/processed/imputed_data/changma_imputed_complete.csv"
    SAVE_DIR = "/Users/xiaxin/work/WindForecast_Project/03_Results/third_part_experiments/G-M1-10m"
    INDICES_PATH = "/Users/xiaxin/work/WindForecast_Project/03_Results/third_part_experiments/train_test_split.json"
    
    # 运行G-M1-10m基准试验
    metrics, results = run_gm1_10m_experiment(
        data_path=DATA_PATH,
        save_dir=SAVE_DIR,
        indices_path=INDICES_PATH
    )
    
    print(f"\n💡 基准试验完成!")
    print(f"📁 结果保存在: {SAVE_DIR}")
    print(f"🔗 训练测试集划分: {INDICES_PATH}")
    print(f"📊 G-M1-10m指标: RMSE={metrics['RMSE']:.4f}, Corr={metrics['Correlation']:.4f}")
    
    print(f"\n接下来可以用 run_wind_power_experiment 函数运行其他13个试验!")
    
    # 示例：如何运行其他试验
    print(f"\n示例：")
    print(f"# G-M1-70m:")
    print(f"run_wind_power_experiment(DATA_PATH, 'G-M1-70m路径', INDICES_PATH, 'gfs_wind_speed_70m', 'G-M1-70m')")
    print(f"# G-M3固定权重:")
    print(f"run_wind_power_experiment(DATA_PATH, 'G-M3路径', INDICES_PATH, ['gfs_wind_speed_10m', 'gfs_wind_speed_70m'], 'G-M3-Fixed', weights=[0.65, 0.35])")