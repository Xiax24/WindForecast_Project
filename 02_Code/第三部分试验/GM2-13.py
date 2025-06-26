#!/usr/bin/env python3
"""
批量运行第三部分的14个试验
完整版本，包含所有必需函数，无外部依赖
"""

import pandas as pd
import numpy as np
import lightgbm as lgb
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

def calculate_time_adaptive_weights(hour):
    """基于SHAP分析的昼夜动态权重
    
    基于SHAP重要性分析：
    - 白天: 10m≈15.5, 70m≈12.5 → 权重比 0.55:0.45  
    - 夜间: 10m≈16.5, 70m≈10.5 → 权重比 0.61:0.39
    """
    is_daytime = (6 <= hour < 18)
    
    if is_daytime:
        # 白天：基于SHAP分析的权重比例
        return [0.55, 0.45]  # [10m权重, 70m权重]
    else:
        # 夜间：10m重要性更突出
        return [0.61, 0.39]

def apply_bias_correction(forecast_data, bias_correction_factor):
    """应用偏差校正"""
    return forecast_data * bias_correction_factor

def run_single_experiment(data_path, save_dir, indices_path, exp_config):
    """运行单个试验"""
    
    os.makedirs(save_dir, exist_ok=True)
    
    # 加载数据
    data = pd.read_csv(data_path)
    data['datetime'] = pd.to_datetime(data['datetime'])
    
    # 数据清理
    data = data.dropna(subset=['power'])
    data = data[data['power'] >= 0]
    
    # 处理不同的试验类型
    if 'wind_col' in exp_config:
        # 单风速试验
        wind_col = exp_config['wind_col']
        data = data.dropna(subset=[wind_col])
        data = data[(data[wind_col] >= 0) & (data[wind_col] <= 50)]
        
        # 应用校正
        if exp_config.get('use_correction', False):
            factor = exp_config['correction_factor']
            data[wind_col] = apply_bias_correction(data[wind_col], factor)
        
        feature_wind_col = wind_col
        
    elif 'wind_cols' in exp_config:
        # 多风速融合试验
        wind_cols = exp_config['wind_cols']
        data = data.dropna(subset=wind_cols)
        for col in wind_cols:
            data = data[(data[col] >= 0) & (data[col] <= 50)]
        
        # 处理不同融合策略
        if exp_config.get('adaptive_weights', False):
            # 昼夜动态权重
            fused_values = []
            for idx, row in data.iterrows():
                weights = calculate_time_adaptive_weights(row['datetime'].hour)
                fused_value = sum(weights[i] * row[wind_cols[i]] for i in range(len(wind_cols)))
                fused_values.append(fused_value)
            data['fused_wind'] = fused_values
                
        elif exp_config.get('weights'):
            # 固定权重
            weights = exp_config['weights']
            data['fused_wind'] = sum(weights[i] * data[wind_cols[i]] for i in range(len(weights)))
            
        elif exp_config.get('fusion_strategy') == 'corrected_average':
            # 联合校正后平均
            ec_10m = apply_bias_correction(data['ec_wind_speed_10m'], 1.03)
            ec_70m = apply_bias_correction(data['ec_wind_speed_70m'], 1.01)
            gfs_10m = apply_bias_correction(data['gfs_wind_speed_10m'], 1.05)
            gfs_70m = apply_bias_correction(data['gfs_wind_speed_70m'], 0.98)
            
            # 加权平均 (EC权重更高，10m权重更高)
            data['fused_wind'] = (0.4 * ec_10m + 0.2 * ec_70m + 
                                 0.3 * gfs_10m + 0.1 * gfs_70m)
            
        elif exp_config.get('fusion_strategy') == 'optimal_adaptive':
            # 基于试验结果的最优动态融合策略
            fused_values = []
            for idx, row in data.iterrows():
                hour = row['datetime'].hour
                is_day = 6 <= hour < 18
                
                # 基于排名结果：EC >> GFS, 10m > 70m
                if is_day:
                    # 白天：基于SHAP昼夜分析，EC主导
                    weights = [0.45, 0.35, 0.12, 0.08]  # EC-10m(0.55*0.8), EC-70m(0.45*0.8), GFS权重较小
                else:
                    # 夜间：10m更重要，EC主导
                    weights = [0.50, 0.25, 0.15, 0.10]  # 更偏向EC-10m
                
                fused_value = sum(weights[i] * row[wind_cols[i]] for i in range(len(wind_cols)))
                fused_values.append(fused_value)
            data['fused_wind'] = fused_values
        
        feature_wind_col = 'fused_wind'
    
    data = data.sort_values('datetime').reset_index(drop=True)
    
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
    
    # 保存结果
    results_df = pd.DataFrame({
        'datetime': data.iloc[test_indices]['datetime'].values,
        'actual_power': y_test,
        'predicted_power': y_pred
    })
    results_df.to_csv(os.path.join(save_dir, f'{exp_config["name"]}_results.csv'), index=False)
    
    with open(os.path.join(save_dir, f'{exp_config["name"]}_metrics.json'), 'w') as f:
        json.dump(metrics, f, indent=2)
    
    model.save_model(os.path.join(save_dir, f'{exp_config["name"]}_model.txt'))
    
    return metrics

def run_all_experiments(data_path, base_save_dir, indices_path):
    """运行所有14个试验"""
    
    print("=" * 80)
    print("🚀 开始批量运行第三部分试验 (共14个)")
    print("=" * 80)
    
    # 试验配置
    experiments = [
        # GFS试验
        {
            'name': 'G-M1-10m',
            'wind_col': 'gfs_wind_speed_10m',
            'description': 'GFS原始10m风速'
        },
        {
            'name': 'G-M2-10m', 
            'wind_col': 'gfs_wind_speed_10m',
            'use_correction': True,
            'correction_factor': 1.05,
            'description': 'GFS校正后10m风速'
        },
        {
            'name': 'G-M1-70m',
            'wind_col': 'gfs_wind_speed_70m', 
            'description': 'GFS原始70m风速'
        },
        {
            'name': 'G-M2-70m',
            'wind_col': 'gfs_wind_speed_70m',
            'use_correction': True,
            'correction_factor': 0.98,
            'description': 'GFS校正后70m风速'
        },
        {
            'name': 'G-M3-Fixed',
            'wind_cols': ['gfs_wind_speed_10m', 'gfs_wind_speed_70m'],
            'weights': [0.55, 0.45],  # 基于SHAP总体权重比例
            'description': 'GFS固定权重融合(基于SHAP总体比例)'
        },
        {
            'name': 'G-M3-TimeAdaptive',
            'wind_cols': ['gfs_wind_speed_10m', 'gfs_wind_speed_70m'],
            'adaptive_weights': True,
            'description': 'GFS昼夜动态权重融合(基于SHAP昼夜分析)'
        },
        
        # EC试验
        {
            'name': 'E-M1-10m',
            'wind_col': 'ec_wind_speed_10m',
            'description': 'EC原始10m风速'
        },
        {
            'name': 'E-M2-10m',
            'wind_col': 'ec_wind_speed_10m',
            'use_correction': True,
            'correction_factor': 1.03,
            'description': 'EC校正后10m风速'
        },
        {
            'name': 'E-M1-70m',
            'wind_col': 'ec_wind_speed_70m',
            'description': 'EC原始70m风速'
        },
        {
            'name': 'E-M2-70m',
            'wind_col': 'ec_wind_speed_70m',
            'use_correction': True,
            'correction_factor': 1.01,
            'description': 'EC校正后70m风速'
        },
        {
            'name': 'E-M3-Fixed',
            'wind_cols': ['ec_wind_speed_10m', 'ec_wind_speed_70m'],
            'weights': [0.55, 0.45],  # 基于SHAP总体权重比例
            'description': 'EC固定权重融合(基于SHAP总体比例)'
        },
        {
            'name': 'E-M3-TimeAdaptive',
            'wind_cols': ['ec_wind_speed_10m', 'ec_wind_speed_70m'],
            'adaptive_weights': True,
            'description': 'EC昼夜动态权重融合(基于SHAP昼夜分析)'
        },
        
        # 融合试验
        {
            'name': 'Fusion-M1',
            'wind_cols': ['ec_wind_speed_10m', 'ec_wind_speed_70m', 'gfs_wind_speed_10m', 'gfs_wind_speed_70m'],
            'fusion_strategy': 'corrected_average',
            'description': '联合校正后融合'
        },
        {
            'name': 'Fusion-M2',
            'wind_cols': ['ec_wind_speed_10m', 'ec_wind_speed_70m', 'gfs_wind_speed_10m', 'gfs_wind_speed_70m'],
            'fusion_strategy': 'optimal_adaptive',
            'description': '基于试验结果的动态权重融合(EC主导+SHAP昼夜权重)'
        }
    ]
    
    # 存储所有结果
    all_results = []
    
    for i, exp_config in enumerate(experiments, 1):
        print(f"\n{'='*60}")
        print(f"🔬 试验 {i}/14: {exp_config['name']}")
        print(f"📝 描述: {exp_config['description']}")
        print(f"{'='*60}")
        
        try:
            # 运行单个试验
            save_dir = os.path.join(base_save_dir, exp_config['name'])
            metrics = run_single_experiment(data_path, save_dir, indices_path, exp_config)
            
            # 记录结果
            result = {
                'experiment': exp_config['name'],
                'description': exp_config['description'],
                'RMSE': metrics['RMSE'],
                'Correlation': metrics['Correlation']
            }
            all_results.append(result)
            
            print(f"✅ {exp_config['name']} 完成")
            print(f"   RMSE: {metrics['RMSE']:.4f}")
            print(f"   相关系数: {metrics['Correlation']:.4f}")
            
        except Exception as e:
            print(f"❌ {exp_config['name']} 失败: {str(e)}")
            result = {
                'experiment': exp_config['name'],
                'description': exp_config['description'],
                'RMSE': None,
                'Correlation': None,
                'error': str(e)
            }
            all_results.append(result)
    
    # 生成汇总报告
    create_summary_report(all_results, base_save_dir)
    
    print(f"\n{'='*80}")
    print(f"🎉 所有试验完成!")
    print(f"📁 结果保存在: {base_save_dir}")
    print(f"📊 汇总报告: {base_save_dir}/experiment_summary.csv")
    print(f"{'='*80}")
    
    return all_results

def create_summary_report(all_results, base_save_dir):
    """创建汇总报告"""
    
    # 转换为DataFrame
    df = pd.DataFrame(all_results)
    
    # 按RMSE排序
    df_valid = df[df['RMSE'].notna()].copy()
    df_valid = df_valid.sort_values('RMSE')
    
    # 保存详细结果
    df.to_csv(os.path.join(base_save_dir, 'experiment_summary.csv'), index=False)
    
    # 创建排名报告
    print(f"\n📊 试验结果排名 (按RMSE排序):")
    print(f"{'排名':<4} {'试验名称':<20} {'RMSE':<10} {'相关系数':<10} {'描述'}")
    print(f"-" * 80)
    
    for i, (_, row) in enumerate(df_valid.iterrows(), 1):
        print(f"{i:<4} {row['experiment']:<20} {row['RMSE']:<10.4f} {row['Correlation']:<10.4f} {row['description']}")
    
    # 保存排名
    with open(os.path.join(base_save_dir, 'ranking_summary.txt'), 'w') as f:
        f.write("试验结果排名 (按RMSE排序):\n")
        f.write("="*80 + "\n")
        for i, (_, row) in enumerate(df_valid.iterrows(), 1):
            f.write(f"{i}. {row['experiment']}: RMSE={row['RMSE']:.4f}, Corr={row['Correlation']:.4f}\n")
    
    # 分析最佳策略
    if len(df_valid) > 0:
        best_result = df_valid.iloc[0]
        print(f"\n🏆 最佳试验: {best_result['experiment']}")
        print(f"   RMSE: {best_result['RMSE']:.4f}")
        print(f"   相关系数: {best_result['Correlation']:.4f}")
        print(f"   策略: {best_result['description']}")

if __name__ == "__main__":
    # 配置路径
    DATA_PATH = "/Users/xiaxin/work/WindForecast_Project/01_Data/processed/imputed_data/changma_imputed_complete.csv"
    BASE_SAVE_DIR = "/Users/xiaxin/work/WindForecast_Project/03_Results/third_part_experiments"
    INDICES_PATH = "/Users/xiaxin/work/WindForecast_Project/03_Results/third_part_experiments/train_test_split.json"
    
    # 运行所有试验
    results = run_all_experiments(DATA_PATH, BASE_SAVE_DIR, INDICES_PATH)
    
    print(f"\n💡 试验完成！")
    print(f"现在可以分析不同策略的效果，找出最优的风速融合方案！")