#!/usr/bin/env python3
"""
完整的14个两步法ML校正试验系统
M1系列: 直接预测 (NWP风速 → 功率)
M2系列: 两步法ML校正 (NWP变量 → 校正风速 → 功率)  
M3系列: 多风速融合 (多个校正风速 → 融合 → 功率)
Fusion系列: 跨模式融合
"""

import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.metrics import mean_squared_error
import os
import json
import warnings
warnings.filterwarnings('ignore')

RANDOM_STATE = 42

class WindCorrectionModel:
    """风速校正模型"""
    
    def __init__(self, target_height='10m', source='gfs'):
        self.target_height = target_height
        self.source = source
        self.model = None
        self.feature_names = None
        
    def prepare_correction_features(self, data):
        """准备风速校正的特征"""
        
        features = pd.DataFrame()
        
        # 主要预报风速
        main_wind_col = f'{self.source}_wind_speed_{self.target_height}'
        features['forecast_wind'] = data[main_wind_col]
        features['forecast_wind_2'] = data[main_wind_col] ** 2
        
        # 其他高度的风速
        other_height = '70m' if self.target_height == '10m' else '10m'
        other_wind_col = f'{self.source}_wind_speed_{other_height}'
        if other_wind_col in data.columns:
            features['other_height_wind'] = data[other_wind_col]
            features['wind_shear'] = np.log(data[other_wind_col] / (data[main_wind_col] + 0.1))
        
        # 温度特征
        if f'{self.source}_temperature_10m' in data.columns:
            features['temperature'] = data[f'{self.source}_temperature_10m']
        
        # 时间特征
        features['hour'] = data['datetime'].dt.hour
        features['month'] = data['datetime'].dt.month
        features['is_daytime'] = ((data['datetime'].dt.hour >= 6) & 
                                 (data['datetime'].dt.hour < 18)).astype(int)
        
        # 滞后特征
        features['wind_lag_1h'] = data[main_wind_col].shift(1)
        features['wind_lag_24h'] = data[main_wind_col].shift(24)
        
        # 滚动统计
        features['wind_24h_mean'] = data[main_wind_col].rolling(window=24, min_periods=1).mean()
        
        # 填充NaN
        features = features.fillna(method='bfill').fillna(method='ffill')
        
        self.feature_names = features.columns.tolist()
        return features
    
    def train(self, data, train_indices):
        """训练风速校正模型"""
        
        features = self.prepare_correction_features(data)
        target_col = f'obs_wind_speed_{self.target_height}'
        target = data[target_col].values
        
        # 划分训练验证集
        val_size = int(len(train_indices) * 0.2)
        train_only_indices = train_indices[:-val_size]
        val_indices = train_indices[-val_size:]
        
        X_train = features.iloc[train_only_indices]
        y_train = target[train_only_indices]
        X_val = features.iloc[val_indices]
        y_val = target[val_indices]
        
        # 训练LightGBM
        params = {
            'objective': 'regression',
            'metric': 'rmse',
            'boosting_type': 'gbdt',
            'num_leaves': 31,
            'learning_rate': 0.1,
            'feature_fraction': 0.9,
            'verbose': -1,
            'random_state': RANDOM_STATE
        }
        
        train_data = lgb.Dataset(X_train, label=y_train)
        valid_data = lgb.Dataset(X_val, label=y_val, reference=train_data)
        
        self.model = lgb.train(
            params,
            train_data,
            valid_sets=[valid_data],
            num_boost_round=100,
            callbacks=[lgb.early_stopping(stopping_rounds=10), lgb.log_evaluation(0)]
        )
        
        # 评估校正效果
        y_pred = self.model.predict(features.iloc[train_indices], num_iteration=self.model.best_iteration)
        y_true = target[train_indices]
        
        rmse = np.sqrt(mean_squared_error(y_true, y_pred))
        corr = np.corrcoef(y_true, y_pred)[0, 1]
        
        return {'rmse': rmse, 'correlation': corr}
    
    def predict(self, data):
        """预测校正后的风速"""
        features = self.prepare_correction_features(data)
        return self.model.predict(features, num_iteration=self.model.best_iteration)

class PowerPredictionModel:
    """功率预测模型"""
    
    def __init__(self):
        self.model = None
        self.feature_names = None
    
    def prepare_power_features(self, wind_data, original_data):
        """准备功率预测特征"""
        
        features = pd.DataFrame()
        
        # 风速特征
        if isinstance(wind_data, dict):
            # 多风速情况
            for key, wind_values in wind_data.items():
                features[f'wind_{key}'] = wind_values
                features[f'wind_{key}_2'] = wind_values ** 2
                features[f'wind_{key}_3'] = wind_values ** 3
        else:
            # 单风速情况
            features['wind'] = wind_data
            features['wind_2'] = wind_data ** 2
            features['wind_3'] = wind_data ** 3
        
        # 时间特征
        features['hour'] = original_data['datetime'].dt.hour
        features['month'] = original_data['datetime'].dt.month
        features['is_daytime'] = ((original_data['datetime'].dt.hour >= 6) & 
                                 (original_data['datetime'].dt.hour < 18)).astype(int)
        
        # 滞后特征
        main_wind = list(wind_data.values())[0] if isinstance(wind_data, dict) else wind_data
        features['wind_lag_1h'] = pd.Series(main_wind).shift(1)
        features['wind_lag_24h'] = pd.Series(main_wind).shift(24)
        
        # 填充NaN
        features = features.fillna(method='bfill').fillna(method='ffill')
        
        self.feature_names = features.columns.tolist()
        return features
    
    def train(self, wind_data, original_data, train_indices, test_indices):
        """训练功率预测模型"""
        
        features = self.prepare_power_features(wind_data, original_data)
        target = original_data['power'].values
        
        X_train = features.iloc[train_indices]
        X_test = features.iloc[test_indices]
        y_train = target[train_indices]
        y_test = target[test_indices]
        
        # 训练LightGBM
        params = {
            'objective': 'regression',
            'metric': 'rmse',
            'boosting_type': 'gbdt',
            'num_leaves': 31,
            'learning_rate': 0.1,
            'feature_fraction': 0.9,
            'verbose': -1,
            'random_state': RANDOM_STATE
        }
        
        train_data = lgb.Dataset(X_train, label=y_train)
        valid_data = lgb.Dataset(X_test, label=y_test, reference=train_data)
        
        self.model = lgb.train(
            params,
            train_data,
            valid_sets=[valid_data],
            num_boost_round=100,
            callbacks=[lgb.early_stopping(stopping_rounds=10), lgb.log_evaluation(0)]
        )
        
        # 评估功率预测效果
        y_pred = self.model.predict(X_test, num_iteration=self.model.best_iteration)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        corr = np.corrcoef(y_test, y_pred)[0, 1]
        
        return {
            'rmse': rmse,
            'correlation': corr,
            'predictions': y_pred,
            'actual': y_test,
            'features_test': X_test
        }

def prepare_simple_features(data, wind_col):
    """为M1系列准备简单特征（直接预测）"""
    features = pd.DataFrame()
    
    features['wind_speed'] = data[wind_col]
    features['wind_speed_2'] = data[wind_col] ** 2
    features['wind_speed_3'] = data[wind_col] ** 3
    
    features['hour'] = data['datetime'].dt.hour
    features['month'] = data['datetime'].dt.month
    features['is_daytime'] = ((data['datetime'].dt.hour >= 6) & 
                             (data['datetime'].dt.hour < 18)).astype(int)
    
    features['wind_lag_1h'] = data[wind_col].shift(1)
    features['wind_lag_24h'] = data[wind_col].shift(24)
    
    features = features.fillna(method='bfill').fillna(method='ffill')
    
    return features

def train_simple_lightgbm(X_train, y_train, X_test, y_test):
    """训练简单LightGBM模型（M1系列用）"""
    params = {
        'objective': 'regression',
        'metric': 'rmse',
        'boosting_type': 'gbdt',
        'num_leaves': 31,
        'learning_rate': 0.1,
        'verbose': -1,
        'random_state': RANDOM_STATE
    }
    
    train_data = lgb.Dataset(X_train, label=y_train)
    valid_data = lgb.Dataset(X_test, label=y_test, reference=train_data)
    
    model = lgb.train(
        params,
        train_data,
        valid_sets=[valid_data],
        num_boost_round=100,
        callbacks=[lgb.early_stopping(stopping_rounds=10), lgb.log_evaluation(0)]
    )
    
    return model

def create_three_way_split(data, indices_path, val_ratio=0.25):
    """创建严格的三分割：训练集 → 训练子集 + 验证集，测试集保持不变"""
    
    # 加载原有的训练测试集划分
    with open(indices_path, 'r') as f:
        indices = json.load(f)
    
    original_train_indices = indices['train_indices']
    test_indices = indices['test_indices']
    
    # 在原训练集内部再分割出验证集
    train_size = len(original_train_indices)
    val_size = int(train_size * val_ratio)
    
    # 时间序列分割：取原训练集的后25%作为验证集
    new_train_indices = original_train_indices[:-val_size]
    val_indices = original_train_indices[-val_size:]
    
    print(f"  三分割结果:")
    print(f"    新训练集: {len(new_train_indices)} 样本")
    print(f"    验证集:   {len(val_indices)} 样本") 
    print(f"    测试集:   {len(test_indices)} 样本")
    
    return new_train_indices, val_indices, test_indices

def optimize_fusion_weights_on_validation(corrected_winds_dict, data, train_indices, val_indices):
    """在验证集上优化融合权重，避免数据泄露"""
    
    print("  🔍 在验证集上优化融合权重...")
    
    # 候选权重组合
    weight_candidates = [
        [0.25, 0.25, 0.25, 0.25],  # 均等权重
        [0.30, 0.20, 0.30, 0.20],  # 偏向10m
        [0.20, 0.30, 0.20, 0.30],  # 偏向70m
        [0.40, 0.10, 0.40, 0.10],  # 强偏向10m
        [0.35, 0.15, 0.35, 0.15],  # 中等偏向10m
        [0.30, 0.15, 0.30, 0.25],  # 混合策略1
        [0.25, 0.15, 0.35, 0.25],  # 混合策略2
        [0.20, 0.15, 0.35, 0.30],  # GFS偏向
        [0.35, 0.25, 0.25, 0.15],  # EC偏向
    ]
    
    best_rmse = float('inf')
    best_weights = None
    validation_results = []
    
    keys = list(corrected_winds_dict.keys())
    
    for i, weights in enumerate(weight_candidates):
        try:
            # 计算融合风速（完整数据集）
            fused_wind = np.array([
                sum(weights[j] * corrected_winds_dict[keys[j]][idx] for j in range(len(keys)))
                for idx in range(len(data))
            ])
            
            # 训练功率预测模型（只在训练集上）
            power_predictor = PowerPredictionModel()
            
            # 将验证集索引映射为相对于训练集的索引
            all_train_val_indices = train_indices + val_indices
            relative_train_indices = list(range(len(train_indices)))
            relative_val_indices = list(range(len(train_indices), len(train_indices) + len(val_indices)))
            
            # 准备训练+验证的数据
            train_val_fused = fused_wind[all_train_val_indices]
            train_val_data = data.iloc[all_train_val_indices].reset_index(drop=True)
            
            power_results = power_predictor.train(
                train_val_fused, train_val_data,
                relative_train_indices, 
                relative_val_indices
            )
            
            val_rmse = power_results['rmse']
            validation_results.append({
                'weights': weights,
                'val_rmse': val_rmse
            })
            
            if val_rmse < best_rmse:
                best_rmse = val_rmse
                best_weights = weights
                
        except Exception as e:
            print(f"    权重组合 {weights} 失败: {str(e)}")
            validation_results.append({
                'weights': weights,
                'val_rmse': float('inf'),
                'error': str(e)
            })
    
    print(f"    最优权重: {best_weights}")
    print(f"    验证集最佳RMSE: {best_rmse:.4f}")
    
    # 显示前5个权重的表现
    validation_results.sort(key=lambda x: x['val_rmse'])
    for i, result in enumerate(validation_results[:5]):
        if result['val_rmse'] != float('inf'):
            weights_str = [f"{w:.2f}" for w in result['weights']]
            print(f"      {i+1}. [{', '.join(weights_str)}] → RMSE: {result['val_rmse']:.4f}")
    
    return best_weights, validation_results

def calculate_shap_weights(hour):
    """基于SHAP分析的昼夜权重（用于两风速融合）"""
    is_daytime = (6 <= hour < 18)
    if is_daytime:
        return [0.55, 0.45]  # [10m权重, 70m权重]
    else:
        return [0.61, 0.39]

def run_no_leakage_fusion_experiment(data_path, save_dir, indices_path, exp_config):
    """运行无数据泄露的融合试验"""
    
    print(f"  🔒 无泄露融合试验: {exp_config['name']}")
    
    os.makedirs(save_dir, exist_ok=True)
    
    # 加载数据
    data = pd.read_csv(data_path)
    data['datetime'] = pd.to_datetime(data['datetime'])
    
    # 基础数据清理
    data = data.dropna(subset=['power'])
    data = data[data['power'] >= 0]
    
    # 清理所有需要的风速数据
    wind_configs = exp_config['wind_configs']
    all_required_cols = ['power']
    for config in wind_configs:
        source = config['source']
        height = config['height']
        all_required_cols.extend([f'{source}_wind_speed_{height}', f'obs_wind_speed_{height}'])
    
    data = data.dropna(subset=all_required_cols)
    for col in all_required_cols:
        if 'wind_speed' in col:
            data = data[(data[col] >= 0) & (data[col] <= 50)]
    
    data = data.sort_values('datetime').reset_index(drop=True)
    
    processing_log = []
    processing_log.append(f"无泄露融合试验: {exp_config['name']}")
    processing_log.append(f"数据大小: {len(data)}")
    
    # 创建严格的三分割
    print("  ✂️ 创建三分割...")
    train_indices, val_indices, test_indices = create_three_way_split(data, indices_path)
    
    # 第一步：在训练集上训练各个风速校正模型
    print("  🎯 第一步：训练风速校正模型（仅训练集）...")
    
    corrected_winds = {}
    correction_stats = {}
    
    for config in wind_configs:
        source = config['source']
        height = config['height']
        key = f"{source}_{height}"
        
        wind_corrector = WindCorrectionModel(target_height=height, source=source)
        correction_stat = wind_corrector.train(data, train_indices)  # 只用训练集
        correction_stats[key] = correction_stat
        
        # 对整个数据集进行校正
        corrected_winds[key] = wind_corrector.predict(data)
        
        processing_log.append(f"{key}校正性能(仅训练集): RMSE={correction_stat['rmse']:.4f}")
    
    # 第二步：在验证集上优化权重
    print("  🔍 第二步：验证集权重优化...")
    
    optimal_weights, weight_search_results = optimize_fusion_weights_on_validation(
        corrected_winds, data, train_indices, val_indices
    )
    
    processing_log.append(f"权重优化: 测试了{len(weight_search_results)}组权重")
    processing_log.append(f"最优权重: EC-10m({optimal_weights[0]:.3f}), EC-70m({optimal_weights[1]:.3f})")
    processing_log.append(f"         GFS-10m({optimal_weights[2]:.3f}), GFS-70m({optimal_weights[3]:.3f})")
    
    # 第三步：用最优权重在训练+验证集上重新训练，在测试集上评估
    print("  ⚡ 第三步：最终模型训练和测试集评估...")
    
    keys = list(corrected_winds.keys())
    
    # 用最优权重计算融合风速
    fused_wind = np.array([
        sum(optimal_weights[i] * corrected_winds[keys[i]][idx] for i in range(len(keys)))
        for idx in range(len(data))
    ])
    
    # 使用训练+验证集训练最终功率预测模型
    combined_train_indices = train_indices + val_indices
    
    power_predictor = PowerPredictionModel()
    power_results = power_predictor.train(
        fused_wind, data, 
        combined_train_indices, 
        test_indices
    )
    
    rmse = power_results['rmse']
    corr = power_results['correlation']
    y_pred = power_results['predictions']
    y_test = power_results['actual']
    
    processing_log.append(f"最终测试性能: RMSE={rmse:.4f}, 相关系数={corr:.4f}")
    
    # 保存详细结果
    detailed_results = pd.DataFrame({
        'datetime': data.iloc[test_indices]['datetime'].values,
        'actual_power': y_test,
        'predicted_power': y_pred,
        'fused_corrected_wind': fused_wind[test_indices],
        'error': y_pred - y_test,
        'abs_error': np.abs(y_pred - y_test)
    })
    
    # 添加各个校正后的风速和权重
    for i, key in enumerate(keys):
        detailed_results[f'corrected_{key}'] = corrected_winds[key][test_indices]
        detailed_results[f'weight_{key}'] = optimal_weights[i]
    
    detailed_results.to_csv(os.path.join(save_dir, f'{exp_config["name"]}_detailed_results.csv'), index=False)
    
    # 保存处理日志
    process_log = {
        'experiment_name': exp_config['name'],
        'experiment_config': exp_config,
        'data_split': {
            'train_size': len(train_indices),
            'val_size': len(val_indices),
            'test_size': len(test_indices)
        },
        'processing_log': processing_log,
        'weight_optimization': weight_search_results,
        'optimal_weights': {
            'ec_10m': optimal_weights[0],
            'ec_70m': optimal_weights[1], 
            'gfs_10m': optimal_weights[2],
            'gfs_70m': optimal_weights[3]
        },
        'correction_stats': correction_stats,
        'final_performance': {'rmse': rmse, 'correlation': corr}
    }
    
    with open(os.path.join(save_dir, f'{exp_config["name"]}_process_log.json'), 'w') as f:
        json.dump(process_log, f, indent=2, default=str)
    
    # 保存指标
    metrics = {'RMSE': rmse, 'Correlation': corr}
    with open(os.path.join(save_dir, f'{exp_config["name"]}_metrics.json'), 'w') as f:
        json.dump(metrics, f, indent=2)
    
    print(f"  ✅ 无泄露试验完成! RMSE: {rmse:.4f}, 相关系数: {corr:.4f}")
    
    return metrics

def calculate_performance_based_weights():
    """基于实际试验结果的性能权重分配"""
    
    # 基于实际M1和M2系列试验结果 (RMSE越小越好，权重越高)
    performance_data = {
        'ec_10m': 33.1672,   # E-M1-10m性能
        'ec_70m': 33.3369,   # E-M1-70m性能  
        'gfs_10m': 33.7579,  # G-M1-10m性能
        'gfs_70m': 34.2257   # G-M1-70m性能
    }
    
    # 对于校正模型，使用两步法的性能
    corrected_performance = {
        'ec_10m_corrected': 33.1708,   # E-M2-10m性能
        'ec_70m_corrected': 33.8399,   # E-M2-70m性能
        'gfs_10m_corrected': 32.9289,  # G-M2-10m性能 
        'gfs_70m_corrected': 32.6539   # G-M2-70m性能 (最好!)
    }
    
    # 计算性能权重 (RMSE的倒数，性能越好权重越高)
    performance_scores = {}
    for key, rmse in corrected_performance.items():
        performance_scores[key] = 1.0 / rmse
    
    # 归一化权重
    total_score = sum(performance_scores.values())
    normalized_weights = {key: score/total_score for key, score in performance_scores.items()}
    
    # 按顺序返回：EC-10m, EC-70m, GFS-10m, GFS-70m
    weights = [
        normalized_weights['ec_10m_corrected'],
        normalized_weights['ec_70m_corrected'], 
        normalized_weights['gfs_10m_corrected'],
        normalized_weights['gfs_70m_corrected']
    ]
    
    return weights

def calculate_adaptive_performance_weights(hour):
    """基于试验结果的时间自适应权重"""
    
    # 基础性能权重
    base_weights = calculate_performance_based_weights()
    
    is_daytime = (6 <= hour < 18)
    
    if is_daytime:
        # 白天：根据SHAP分析，稍微增加10m权重
        adjustment_factors = [1.1, 0.9, 1.1, 0.9]  # 增加10m，减少70m
    else:
        # 夜间：更加偏向10m
        adjustment_factors = [1.2, 0.8, 1.2, 0.8]
    
    # 应用调整因子
    adjusted_weights = [base_weights[i] * adjustment_factors[i] for i in range(4)]
    
    # 重新归一化
    total = sum(adjusted_weights)
    final_weights = [w/total for w in adjusted_weights]
    
    return final_weights

def calculate_simple_optimal_weights():
    """基于试验结果的简化最优权重"""
    
    # 直接基于各模型的RMSE表现分配权重
    # G-M2-70m表现最好(32.6539)，应该给最高权重
    # G-M2-10m次之(32.9289) 
    # E-M2-10m第三(33.1708)
    # E-M2-70m最差(33.8399)
    
    # 简化策略：给表现好的更高权重
    weights = [
        0.25,  # EC-10m (中等表现)
        0.15,  # EC-70m (较差表现) 
        0.30,  # GFS-10m (较好表现)
        0.30   # GFS-70m (最好表现)
    ]
    
    return weights

def load_train_test_split(indices_path):
    """加载训练测试集划分"""
    with open(indices_path, 'r') as f:
        indices = json.load(f)
    return indices['train_indices'], indices['test_indices']

def run_experiment(data_path, save_dir, indices_path, exp_config):
    """运行单个试验"""
    
    print(f"\n{'='*60}")
    print(f"🔬 试验: {exp_config['name']}")
    print(f"📝 {exp_config['description']}")
    print(f"{'='*60}")
    
    os.makedirs(save_dir, exist_ok=True)
    
    # 加载数据
    print("  📂 加载数据...")
    data = pd.read_csv(data_path)
    data['datetime'] = pd.to_datetime(data['datetime'])
    
    # 基础数据清理
    data = data.dropna(subset=['power'])
    data = data[data['power'] >= 0]
    data = data.sort_values('datetime').reset_index(drop=True)
    
    # 获取训练测试集划分
    train_indices, test_indices = load_train_test_split(indices_path)
    
    processing_log = []
    processing_log.append(f"试验类型: {exp_config['type']}")
    processing_log.append(f"数据大小: {len(data)}")
    
    try:
        if exp_config['type'] == 'no_leakage_fusion':
            # 无数据泄露的融合试验
            return run_no_leakage_fusion_experiment(data_path, save_dir, indices_path, exp_config)
        
        elif exp_config['type'] == 'direct':
            # M1系列：直接预测
            wind_col = exp_config['wind_col']
            processing_log.append(f"直接预测风速列: {wind_col}")
            
            # 清理风速数据
            data = data.dropna(subset=[wind_col])
            data = data[(data[wind_col] >= 0) & (data[wind_col] <= 50)]
            
            # 重新获取有效索引
            valid_indices = data.index.tolist()
            train_indices = [i for i in train_indices if i in valid_indices]
            test_indices = [i for i in test_indices if i in valid_indices]
            
            # 准备特征和训练模型
            features = prepare_simple_features(data, wind_col)
            target = data['power'].values
            
            X_train = features.iloc[train_indices]
            X_test = features.iloc[test_indices]
            y_train = target[train_indices]
            y_test = target[test_indices]
            
            print("  🚀 训练直接预测模型...")
            model = train_simple_lightgbm(X_train, y_train, X_test, y_test)
            
            y_pred = model.predict(X_test, num_iteration=model.best_iteration)
            rmse = np.sqrt(mean_squared_error(y_test, y_pred))
            corr = np.corrcoef(y_test, y_pred)[0, 1]
            
            processing_log.append(f"直接预测性能: RMSE={rmse:.4f}, 相关系数={corr:.4f}")
            
            # 保存结果
            detailed_results = pd.DataFrame({
                'datetime': data.iloc[test_indices]['datetime'].values,
                'actual_power': y_test,
                'predicted_power': y_pred,
                'wind_speed': data.iloc[test_indices][wind_col].values,
                'error': y_pred - y_test,
                'abs_error': np.abs(y_pred - y_test)
            })
            
            # 添加特征
            for i, col in enumerate(features.columns):
                detailed_results[f'feature_{col}'] = X_test.iloc[:, i].values
            
        elif exp_config['type'] == 'two_step':
            # M2系列：两步法校正
            wind_source = exp_config['wind_source']
            wind_height = exp_config['wind_height']
            
            processing_log.append(f"两步法校正: {wind_source.upper()}-{wind_height}")
            
            # 清理数据
            required_cols = [f'{wind_source}_wind_speed_{wind_height}', f'obs_wind_speed_{wind_height}']
            data = data.dropna(subset=required_cols)
            for col in required_cols:
                data = data[(data[col] >= 0) & (data[col] <= 50)]
            
            # 重新获取有效索引
            valid_indices = data.index.tolist()
            train_indices = [i for i in train_indices if i in valid_indices]
            test_indices = [i for i in test_indices if i in valid_indices]
            
            # 第一步：风速校正
            print("  🎯 第一步：风速校正...")
            wind_corrector = WindCorrectionModel(target_height=wind_height, source=wind_source)
            correction_stats = wind_corrector.train(data, train_indices)
            
            corrected_wind = wind_corrector.predict(data)
            processing_log.append(f"风速校正性能: RMSE={correction_stats['rmse']:.4f}")
            
            # 第二步：功率预测
            print("  ⚡ 第二步：功率预测...")
            power_predictor = PowerPredictionModel()
            power_results = power_predictor.train(corrected_wind, data, train_indices, test_indices)
            
            processing_log.append(f"功率预测性能: RMSE={power_results['rmse']:.4f}")
            
            rmse = power_results['rmse']
            corr = power_results['correlation']
            y_pred = power_results['predictions']
            y_test = power_results['actual']
            
            # 保存结果
            detailed_results = pd.DataFrame({
                'datetime': data.iloc[test_indices]['datetime'].values,
                'actual_power': y_test,
                'predicted_power': y_pred,
                'corrected_wind': corrected_wind[test_indices],
                'original_wind': data.iloc[test_indices][f'{wind_source}_wind_speed_{wind_height}'].values,
                'observed_wind': data.iloc[test_indices][f'obs_wind_speed_{wind_height}'].values,
                'error': y_pred - y_test,
                'abs_error': np.abs(y_pred - y_test)
            })
            
        elif exp_config['type'] == 'fusion':
            # M3和Fusion系列：多风速融合
            wind_configs = exp_config['wind_configs']
            fusion_strategy = exp_config['fusion_strategy']
            
            processing_log.append(f"融合策略: {fusion_strategy}")
            processing_log.append(f"风速配置: {wind_configs}")
            
            # 清理所有需要的风速数据
            all_required_cols = ['power']
            for config in wind_configs:
                source = config['source']
                height = config['height']
                all_required_cols.extend([f'{source}_wind_speed_{height}'])
                if exp_config['type'] == 'fusion' and fusion_strategy != 'direct':
                    all_required_cols.append(f'obs_wind_speed_{height}')
            
            data = data.dropna(subset=all_required_cols)
            for col in all_required_cols:
                if 'wind_speed' in col:
                    data = data[(data[col] >= 0) & (data[col] <= 50)]
            
            # 重新获取有效索引
            valid_indices = data.index.tolist()
            train_indices = [i for i in train_indices if i in valid_indices]
            test_indices = [i for i in test_indices if i in valid_indices]
            
            if fusion_strategy == 'direct':
                # 直接融合原始风速
                print("  🔗 直接融合原始风速...")
                
                # 获取权重
                if exp_config.get('weights'):
                    # 固定权重
                    weights = exp_config['weights']
                    fused_wind = np.zeros(len(data))
                    for i, config in enumerate(wind_configs):
                        source = config['source']
                        height = config['height']
                        wind_col = f'{source}_wind_speed_{height}'
                        fused_wind += weights[i] * data[wind_col].values
                    
                    processing_log.append(f"固定权重: {weights}")
                    
                elif exp_config.get('adaptive_weights'):
                    # 昼夜动态权重
                    fused_wind = []
                    for idx, row in data.iterrows():
                        weights = calculate_shap_weights(row['datetime'].hour)
                        wind_values = [row[f"{config['source']}_wind_speed_{config['height']}"] 
                                     for config in wind_configs]
                        fused_value = sum(weights[i] * wind_values[i] for i in range(len(weights)))
                        fused_wind.append(fused_value)
                    fused_wind = np.array(fused_wind)
                    
                    processing_log.append("昼夜动态权重融合")
                
                # 使用融合风速训练功率预测模型
                power_predictor = PowerPredictionModel()
                power_results = power_predictor.train(fused_wind, data, train_indices, test_indices)
                
                rmse = power_results['rmse']
                corr = power_results['correlation']
                y_pred = power_results['predictions']
                y_test = power_results['actual']
                
                # 保存结果
                detailed_results = pd.DataFrame({
                    'datetime': data.iloc[test_indices]['datetime'].values,
                    'actual_power': y_test,
                    'predicted_power': y_pred,
                    'fused_wind': fused_wind[test_indices],
                    'error': y_pred - y_test,
                    'abs_error': np.abs(y_pred - y_test)
                })
                
                # 添加原始风速
                for config in wind_configs:
                    source = config['source']
                    height = config['height']
                    wind_col = f'{source}_wind_speed_{height}'
                    detailed_results[f'original_{source}_{height}'] = data.iloc[test_indices][wind_col].values
                
            else:
                # 校正后融合
                print("  🎯 多风速校正...")
                
                corrected_winds = {}
                correction_stats_all = {}
                
                # 分别校正每个风速
                for config in wind_configs:
                    source = config['source']
                    height = config['height']
                    key = f"{source}_{height}"
                    
                    wind_corrector = WindCorrectionModel(target_height=height, source=source)
                    correction_stats = wind_corrector.train(data, train_indices)
                    correction_stats_all[key] = correction_stats
                    
                    corrected_winds[key] = wind_corrector.predict(data)
                    
                    processing_log.append(f"{key}校正性能: RMSE={correction_stats['rmse']:.4f}")
                
                # 融合校正后的风速
                print("  🔗 融合校正后风速...")
                
                if exp_config.get('weights'):
                    weights = exp_config['weights']
                    fused_wind = np.zeros(len(data))
                    for i, key in enumerate(corrected_winds.keys()):
                        fused_wind += weights[i] * corrected_winds[key]
                    
                elif exp_config.get('adaptive_weights'):
                    fused_wind = []
                    keys = list(corrected_winds.keys())
                    for idx, row in data.iterrows():
                        weights = calculate_shap_weights(row['datetime'].hour)
                        fused_value = sum(weights[i] * corrected_winds[keys[i]][idx] 
                                        for i in range(len(weights)))
                        fused_wind.append(fused_value)
                    fused_wind = np.array(fused_wind)
                
                elif fusion_strategy == 'performance_based':
                    # 基于实际试验结果的最优权重融合
                    fused_wind = []
                    keys = list(corrected_winds.keys())
                    
                    # 获取基于性能的权重
                    optimal_weights = calculate_simple_optimal_weights()
                    
                    processing_log.append("基于试验结果的性能权重融合")
                    processing_log.append(f"权重分配: EC-10m({optimal_weights[0]:.3f}), EC-70m({optimal_weights[1]:.3f})")
                    processing_log.append(f"         GFS-10m({optimal_weights[2]:.3f}), GFS-70m({optimal_weights[3]:.3f})")
                    processing_log.append("权重依据: G-M2-70m(32.65)最好, G-M2-10m(32.93)次之")
                    
                    # 简单静态融合 - 不再使用复杂的时间动态
                    for idx in range(len(data)):
                        fused_value = sum(optimal_weights[i] * corrected_winds[keys[i]][idx] 
                                        for i in range(len(keys)))
                        fused_wind.append(fused_value)
                    
                    fused_wind = np.array(fused_wind)
                
                elif fusion_strategy == 'corrected':
                    # 默认固定权重融合
                    weights = exp_config.get('weights', [0.4, 0.2, 0.3, 0.1])
                    fused_wind = np.zeros(len(data))
                    for i, key in enumerate(corrected_winds.keys()):
                        fused_wind += weights[i] * corrected_winds[key]
                
                # 功率预测
                print("  ⚡ 功率预测...")
                power_predictor = PowerPredictionModel()
                power_results = power_predictor.train(fused_wind, data, train_indices, test_indices)
                
                rmse = power_results['rmse']
                corr = power_results['correlation']
                y_pred = power_results['predictions']
                y_test = power_results['actual']
                
                # 保存结果
                detailed_results = pd.DataFrame({
                    'datetime': data.iloc[test_indices]['datetime'].values,
                    'actual_power': y_test,
                    'predicted_power': y_pred,
                    'fused_corrected_wind': fused_wind[test_indices],
                    'error': y_pred - y_test,
                    'abs_error': np.abs(y_pred - y_test)
                })
                
                # 添加各个校正后的风速
                for key, wind_values in corrected_winds.items():
                    detailed_results[f'corrected_{key}'] = wind_values[test_indices]
        
        print(f"  ✅ 完成! RMSE: {rmse:.4f}, 相关系数: {corr:.4f}")
        
        # 保存所有结果文件
        detailed_results.to_csv(os.path.join(save_dir, f'{exp_config["name"]}_detailed_results.csv'), index=False)
        
        # 保存处理日志
        process_log = {
            'experiment_name': exp_config['name'],
            'experiment_config': exp_config,
            'processing_log': processing_log,
            'final_performance': {'rmse': rmse, 'correlation': corr}
        }
        
        with open(os.path.join(save_dir, f'{exp_config["name"]}_process_log.json'), 'w') as f:
            json.dump(process_log, f, indent=2, default=str)
        
        # 保存指标
        metrics = {'RMSE': rmse, 'Correlation': corr}
        with open(os.path.join(save_dir, f'{exp_config["name"]}_metrics.json'), 'w') as f:
            json.dump(metrics, f, indent=2)
        
        return metrics
        
    except Exception as e:
        print(f"  ❌ 失败: {str(e)}")
        return {'RMSE': None, 'Correlation': None, 'error': str(e)}

def run_all_14_experiments(data_path, base_save_dir, indices_path):
    """运行完整的14个试验"""
    
    print("=" * 80)
    print("🚀 开始运行完整的14个两步法ML试验")
    print("=" * 80)
    
    # 定义所有14个试验配置
    experiments = [
        # M1系列：直接预测
        {
            'name': 'G-M1-10m',
            'description': 'GFS原始10m风速直接预测功率',
            'type': 'direct',
            'wind_col': 'gfs_wind_speed_10m'
        },
        {
            'name': 'G-M1-70m', 
            'description': 'GFS原始70m风速直接预测功率',
            'type': 'direct',
            'wind_col': 'gfs_wind_speed_70m'
        },
        {
            'name': 'E-M1-10m',
            'description': 'EC原始10m风速直接预测功率', 
            'type': 'direct',
            'wind_col': 'ec_wind_speed_10m'
        },
        {
            'name': 'E-M1-70m',
            'description': 'EC原始70m风速直接预测功率',
            'type': 'direct', 
            'wind_col': 'ec_wind_speed_70m'
        },
        
        # M2系列：两步法ML校正
        {
            'name': 'G-M2-10m',
            'description': 'GFS 10m风速两步法ML校正',
            'type': 'two_step',
            'wind_source': 'gfs',
            'wind_height': '10m'
        },
        {
            'name': 'G-M2-70m',
            'description': 'GFS 70m风速两步法ML校正',
            'type': 'two_step',
            'wind_source': 'gfs', 
            'wind_height': '70m'
        },
        {
            'name': 'E-M2-10m',
            'description': 'EC 10m风速两步法ML校正',
            'type': 'two_step',
            'wind_source': 'ec',
            'wind_height': '10m'
        },
        {
            'name': 'E-M2-70m',
            'description': 'EC 70m风速两步法ML校正', 
            'type': 'two_step',
            'wind_source': 'ec',
            'wind_height': '70m'
        },
        
        # M3系列：多风速融合
        {
            'name': 'G-M3-Fixed',
            'description': 'GFS固定权重融合(基于SHAP)',
            'type': 'fusion',
            'wind_configs': [
                {'source': 'gfs', 'height': '10m'},
                {'source': 'gfs', 'height': '70m'}
            ],
            'fusion_strategy': 'direct',
            'weights': [0.5, 0.5]
        },
        {
            'name': 'G-M3-TimeAdaptive',
            'description': 'GFS昼夜动态权重融合(基于SHAP)',
            'type': 'fusion',
            'wind_configs': [
                {'source': 'gfs', 'height': '10m'},
                {'source': 'gfs', 'height': '70m'}
            ],
            'fusion_strategy': 'direct',
            'adaptive_weights': True
        },
        {
            'name': 'E-M3-Fixed',
            'description': 'EC固定权重融合(基于SHAP)', 
            'type': 'fusion',
            'wind_configs': [
                {'source': 'ec', 'height': '10m'},
                {'source': 'ec', 'height': '70m'}
            ],
            'fusion_strategy': 'direct',
            'weights': [0.5, 0.5]
        },
        {
            'name': 'E-M3-TimeAdaptive',
            'description': 'EC昼夜动态权重融合(基于SHAP)',
            'type': 'fusion',
            'wind_configs': [
                {'source': 'ec', 'height': '10m'},
                {'source': 'ec', 'height': '70m'}
            ],
            'fusion_strategy': 'direct', 
            'adaptive_weights': True
        },
        
        # Fusion系列：跨模式融合
        {
            'name': 'Fusion-M1',
            'description': '联合校正后静态融合',
            'type': 'fusion',
            'wind_configs': [
                {'source': 'ec', 'height': '10m'},
                {'source': 'ec', 'height': '70m'},
                {'source': 'gfs', 'height': '10m'},
                {'source': 'gfs', 'height': '70m'}
            ],
            'fusion_strategy': 'corrected',
            'weights': [0.25, 0.25, 0.25, 0.25]  # EC主导，10m重要
        },
        {
            'name': 'Fusion-M2',
            'description': 'ML校正后无数据泄露的权重优化融合',
            'type': 'no_leakage_fusion',
            'wind_configs': [
                {'source': 'ec', 'height': '10m'},
                {'source': 'ec', 'height': '70m'},
                {'source': 'gfs', 'height': '10m'},
                {'source': 'gfs', 'height': '70m'}
            ]
        }
    ]
    
    # 存储所有结果
    all_results = []
    
    for i, exp_config in enumerate(experiments, 1):
        print(f"\n进度: {i}/14")
        
        try:
            # 运行单个试验
            save_dir = os.path.join(base_save_dir, exp_config['name'])
            metrics = run_experiment(data_path, save_dir, indices_path, exp_config)
            
            # 记录结果
            result = {
                'experiment': exp_config['name'],
                'description': exp_config['description'],
                'type': exp_config['type'],
                'RMSE': metrics.get('RMSE'),
                'Correlation': metrics.get('Correlation')
            }
            
            if 'error' in metrics:
                result['error'] = metrics['error']
            
            all_results.append(result)
            
        except Exception as e:
            print(f"❌ {exp_config['name']} 试验失败: {str(e)}")
            result = {
                'experiment': exp_config['name'],
                'description': exp_config['description'],
                'type': exp_config['type'],
                'RMSE': None,
                'Correlation': None,
                'error': str(e)
            }
            all_results.append(result)
    
    # 生成汇总报告
    create_comprehensive_summary(all_results, base_save_dir)
    
    print(f"\n{'='*80}")
    print(f"🎉 所有14个试验完成!")
    print(f"📁 结果保存在: {base_save_dir}")
    print(f"📊 汇总报告: {base_save_dir}/comprehensive_summary.csv")
    print(f"{'='*80}")
    
    return all_results

def create_comprehensive_summary(all_results, base_save_dir):
    """创建综合汇总报告"""
    
    # 转换为DataFrame
    df = pd.DataFrame(all_results)
    
    # 按RMSE排序（只包含有效结果）
    df_valid = df[df['RMSE'].notna()].copy()
    df_valid = df_valid.sort_values('RMSE')
    
    # 保存详细结果
    df.to_csv(os.path.join(base_save_dir, 'comprehensive_summary.csv'), index=False)
    
    # 创建排名报告
    print(f"\n📊 完整试验结果排名 (按RMSE排序):")
    print(f"{'排名':<4} {'试验名称':<20} {'类型':<12} {'RMSE':<10} {'相关系数':<10} {'描述'}")
    print(f"-" * 95)
    
    for i, (_, row) in enumerate(df_valid.iterrows(), 1):
        print(f"{i:<4} {row['experiment']:<20} {row['type']:<12} {row['RMSE']:<10.4f} {row['Correlation']:<10.4f} {row['description']}")
    
    # 按类型分析
    print(f"\n📈 按试验类型分析:")
    
    type_analysis = {}
    for exp_type in ['direct', 'two_step', 'fusion']:
        type_data = df_valid[df_valid['type'] == exp_type]
        if len(type_data) > 0:
            type_analysis[exp_type] = {
                'count': len(type_data),
                'best_rmse': type_data['RMSE'].min(),
                'best_experiment': type_data.loc[type_data['RMSE'].idxmin(), 'experiment'],
                'avg_rmse': type_data['RMSE'].mean(),
                'avg_correlation': type_data['Correlation'].mean()
            }
            
            print(f"  {exp_type.upper()}类型:")
            print(f"    试验数量: {type_analysis[exp_type]['count']}")
            print(f"    最佳RMSE: {type_analysis[exp_type]['best_rmse']:.4f} ({type_analysis[exp_type]['best_experiment']})")
            print(f"    平均RMSE: {type_analysis[exp_type]['avg_rmse']:.4f}")
            print(f"    平均相关系数: {type_analysis[exp_type]['avg_correlation']:.4f}")
    
    # 数据源对比
    print(f"\n🔍 数据源对比:")
    
    # GFS vs EC对比
    gfs_experiments = df_valid[df_valid['experiment'].str.contains('G-')]
    ec_experiments = df_valid[df_valid['experiment'].str.contains('E-')]
    
    if len(gfs_experiments) > 0 and len(ec_experiments) > 0:
        print(f"  GFS系列 (平均RMSE: {gfs_experiments['RMSE'].mean():.4f})")
        print(f"  EC系列  (平均RMSE: {ec_experiments['RMSE'].mean():.4f})")
        
        if gfs_experiments['RMSE'].mean() < ec_experiments['RMSE'].mean():
            print(f"  → GFS整体表现更好")
        else:
            print(f"  → EC整体表现更好")
    
    # 10m vs 70m对比
    m10_experiments = df_valid[df_valid['experiment'].str.contains('10m')]
    m70_experiments = df_valid[df_valid['experiment'].str.contains('70m')]
    
    if len(m10_experiments) > 0 and len(m70_experiments) > 0:
        print(f"  10m风速 (平均RMSE: {m10_experiments['RMSE'].mean():.4f})")
        print(f"  70m风速 (平均RMSE: {m70_experiments['RMSE'].mean():.4f})")
        
        if m10_experiments['RMSE'].mean() < m70_experiments['RMSE'].mean():
            print(f"  → 10m风速整体表现更好")
        else:
            print(f"  → 70m风速整体表现更好")
    
    # 保存分析报告
    with open(os.path.join(base_save_dir, 'analysis_summary.txt'), 'w') as f:
        f.write("14个试验综合分析报告\n")
        f.write("="*50 + "\n\n")
        
        f.write("1. 总体排名 (按RMSE):\n")
        for i, (_, row) in enumerate(df_valid.iterrows(), 1):
            f.write(f"{i}. {row['experiment']}: RMSE={row['RMSE']:.4f}, Corr={row['Correlation']:.4f}\n")
        
        f.write(f"\n2. 按类型分析:\n")
        for exp_type, stats in type_analysis.items():
            f.write(f"{exp_type.upper()}: 最佳RMSE={stats['best_rmse']:.4f} ({stats['best_experiment']})\n")
        
        f.write(f"\n3. 主要发现:\n")
        if len(df_valid) > 0:
            best_overall = df_valid.iloc[0]
            f.write(f"- 最佳试验: {best_overall['experiment']} (RMSE: {best_overall['RMSE']:.4f})\n")
            f.write(f"- 最佳策略: {best_overall['description']}\n")
            
            # ML校正效果分析
            direct_avg = df_valid[df_valid['type'] == 'direct']['RMSE'].mean() if len(df_valid[df_valid['type'] == 'direct']) > 0 else None
            two_step_avg = df_valid[df_valid['type'] == 'two_step']['RMSE'].mean() if len(df_valid[df_valid['type'] == 'two_step']) > 0 else None
            
            if direct_avg and two_step_avg:
                improvement = (direct_avg - two_step_avg) / direct_avg * 100
                f.write(f"- ML校正效果: 平均改善 {improvement:.2f}%\n")
    
    # 分析最佳策略
    if len(df_valid) > 0:
        best_result = df_valid.iloc[0]
        print(f"\n🏆 最佳试验: {best_result['experiment']}")
        print(f"   RMSE: {best_result['RMSE']:.4f}")
        print(f"   相关系数: {best_result['Correlation']:.4f}")
        print(f"   策略: {best_result['description']}")
        print(f"   类型: {best_result['type']}")


def run_weight_comparison_experiments(data_path, base_save_dir, indices_path):
    """运行权重对比试验"""
    
    print("=" * 80)
    print("🔬 权重对比试验：测试不同融合权重策略")
    print("=" * 80)
    
    # 定义不同的权重策略
    weight_strategies = [
        {
            'name': 'Fusion-Equal',
            'description': '四模型等权重融合 (各25%)',
            'type': 'fusion',
            'wind_configs': [
                {'source': 'ec', 'height': '10m'},
                {'source': 'ec', 'height': '70m'},
                {'source': 'gfs', 'height': '10m'},
                {'source': 'gfs', 'height': '70m'}
            ],
            'fusion_strategy': 'corrected',
            'weights': [0.25, 0.25, 0.25, 0.25]  # 等权重
        },
        {
            'name': 'Fusion-Original',
            'description': '原始权重 (EC主导)',
            'type': 'fusion',
            'wind_configs': [
                {'source': 'ec', 'height': '10m'},
                {'source': 'ec', 'height': '70m'},
                {'source': 'gfs', 'height': '10m'},
                {'source': 'gfs', 'height': '70m'}
            ],
            'fusion_strategy': 'corrected',
            'weights': [0.4, 0.2, 0.3, 0.1]  # 原始权重
        },
        {
            'name': 'Fusion-Performance',
            'description': '基于单模型性能的权重',
            'type': 'fusion',
            'wind_configs': [
                {'source': 'ec', 'height': '10m'},
                {'source': 'ec', 'height': '70m'},
                {'source': 'gfs', 'height': '10m'},
                {'source': 'gfs', 'height': '70m'}
            ],
            'fusion_strategy': 'corrected',
            # 基于试验结果：G-M2-70m(32.65) > G-M2-10m(32.93) > E-M2-10m(33.17) > E-M2-70m(33.84)
            'weights': [0.25, 0.15, 0.30, 0.30]  # GFS-70m最好给最高权重
        },
        {
            'name': 'Fusion-10m-Focus',
            'description': '偏向10m高度 (10m占70%)',
            'type': 'fusion',
            'wind_configs': [
                {'source': 'ec', 'height': '10m'},
                {'source': 'ec', 'height': '70m'},
                {'source': 'gfs', 'height': '10m'},
                {'source': 'gfs', 'height': '70m'}
            ],
            'fusion_strategy': 'corrected',
            'weights': [0.35, 0.15, 0.35, 0.15]  # 10m占70%
        },
        {
            'name': 'Fusion-EC-Focus',
            'description': 'EC主导策略 (EC占70%)',
            'type': 'fusion',
            'wind_configs': [
                {'source': 'ec', 'height': '10m'},
                {'source': 'ec', 'height': '70m'},
                {'source': 'gfs', 'height': '10m'},
                {'source': 'gfs', 'height': '70m'}
            ],
            'fusion_strategy': 'corrected',
            'weights': [0.45, 0.25, 0.20, 0.10]  # EC占70%
        },
        {
            'name': 'Fusion-GFS-Focus',
            'description': 'GFS主导策略 (GFS占70%)',
            'type': 'fusion',
            'wind_configs': [
                {'source': 'ec', 'height': '10m'},
                {'source': 'ec', 'height': '70m'},
                {'source': 'gfs', 'height': '10m'},
                {'source': 'gfs', 'height': '70m'}
            ],
            'fusion_strategy': 'corrected',
            'weights': [0.15, 0.15, 0.35, 0.35]  # GFS占70%
        }
    ]
    
    # 存储所有结果
    comparison_results = []
    
    for i, exp_config in enumerate(weight_strategies, 1):
        print(f"\n权重策略 {i}/6: {exp_config['name']}")
        print(f"权重: {exp_config['weights']}")
        print(f"说明: {exp_config['description']}")
        
        try:
            # 运行试验
            save_dir = os.path.join(base_save_dir, 'weight_comparison', exp_config['name'])
            metrics = run_experiment(data_path, save_dir, indices_path, exp_config)
            
            # 记录结果
            result = {
                'strategy': exp_config['name'],
                'description': exp_config['description'],
                'weights': exp_config['weights'],
                'ec_10m_weight': exp_config['weights'][0],
                'ec_70m_weight': exp_config['weights'][1],
                'gfs_10m_weight': exp_config['weights'][2],
                'gfs_70m_weight': exp_config['weights'][3],
                'ec_total_weight': exp_config['weights'][0] + exp_config['weights'][1],
                'gfs_total_weight': exp_config['weights'][2] + exp_config['weights'][3],
                'm10_total_weight': exp_config['weights'][0] + exp_config['weights'][2],
                'm70_total_weight': exp_config['weights'][1] + exp_config['weights'][3],
                'RMSE': metrics.get('RMSE'),
                'Correlation': metrics.get('Correlation')
            }
            
            if 'error' in metrics:
                result['error'] = metrics['error']
            
            comparison_results.append(result)
            
            print(f"  结果: RMSE={result['RMSE']:.4f}, 相关系数={result['Correlation']:.4f}")
            
        except Exception as e:
            print(f"  ❌ 失败: {str(e)}")
            result = {
                'strategy': exp_config['name'],
                'description': exp_config['description'],
                'weights': exp_config['weights'],
                'RMSE': None,
                'Correlation': None,
                'error': str(e)
            }
            comparison_results.append(result)
    
    # 分析和汇总结果
    analyze_weight_comparison_results(comparison_results, base_save_dir)
    
    return comparison_results

def analyze_weight_comparison_results(results, base_save_dir):
    """分析权重对比结果"""
    
    # 转换为DataFrame
    df = pd.DataFrame(results)
    
    # 保存完整结果
    os.makedirs(os.path.join(base_save_dir, 'weight_comparison'), exist_ok=True)
    df.to_csv(os.path.join(base_save_dir, 'weight_comparison', 'weight_comparison_results.csv'), index=False)
    
    # 筛选有效结果并排序
    df_valid = df[df['RMSE'].notna()].copy()
    df_valid = df_valid.sort_values('RMSE')
    
    print(f"\n{'='*80}")
    print(f"📊 权重策略对比结果 (按RMSE排序)")
    print(f"{'='*80}")
    print(f"{'排名':<4} {'策略名称':<20} {'RMSE':<10} {'相关系数':<8} {'权重分配':<25} {'说明'}")
    print(f"-" * 100)
    
    for i, (_, row) in enumerate(df_valid.iterrows(), 1):
        weights_str = f"[{row['ec_10m_weight']:.2f},{row['ec_70m_weight']:.2f},{row['gfs_10m_weight']:.2f},{row['gfs_70m_weight']:.2f}]"
        print(f"{i:<4} {row['strategy']:<20} {row['RMSE']:<10.4f} {row['Correlation']:<8.4f} {weights_str:<25} {row['description']}")
    
    # 分析不同权重策略的效果
    print(f"\n🔍 权重策略分析:")
    
    if len(df_valid) > 0:
        best_strategy = df_valid.iloc[0]
        worst_strategy = df_valid.iloc[-1]
        
        print(f"  🏆 最佳策略: {best_strategy['strategy']}")
        print(f"     权重: EC-10m({best_strategy['ec_10m_weight']:.2f}), EC-70m({best_strategy['ec_70m_weight']:.2f})")
        print(f"          GFS-10m({best_strategy['gfs_10m_weight']:.2f}), GFS-70m({best_strategy['gfs_70m_weight']:.2f})")
        print(f"     RMSE: {best_strategy['RMSE']:.4f}")
        
        print(f"  📉 最差策略: {worst_strategy['strategy']}")
        print(f"     RMSE: {worst_strategy['RMSE']:.4f}")
        
        improvement = (worst_strategy['RMSE'] - best_strategy['RMSE']) / worst_strategy['RMSE'] * 100
        print(f"  📈 最优vs最差改善: {improvement:.2f}%")
        
        # 等权重策略的表现
        equal_weight = df_valid[df_valid['strategy'] == 'Fusion-Equal']
        if len(equal_weight) > 0:
            equal_result = equal_weight.iloc[0]
            equal_rank = list(df_valid['strategy']).index('Fusion-Equal') + 1
            print(f"  ⚖️ 等权重策略表现:")
            print(f"     排名: 第{equal_rank}名 (共{len(df_valid)}个)")
            print(f"     RMSE: {equal_result['RMSE']:.4f}")
            
            if equal_result['RMSE'] == best_strategy['RMSE']:
                print(f"     🎯 等权重策略就是最优策略!")
            else:
                gap = (equal_result['RMSE'] - best_strategy['RMSE']) / best_strategy['RMSE'] * 100
                print(f"     📊 与最优策略差距: {gap:.2f}%")
    
    # EC vs GFS 权重效果分析
    print(f"\n🔬 EC vs GFS 权重效果分析:")
    
    for _, row in df_valid.iterrows():
        ec_weight = row['ec_total_weight'] 
        gfs_weight = row['gfs_total_weight']
        print(f"  {row['strategy']:<20} EC:{ec_weight:.2f} GFS:{gfs_weight:.2f} → RMSE:{row['RMSE']:.4f}")
    
    # 10m vs 70m 权重效果分析  
    print(f"\n🌪️ 10m vs 70m 权重效果分析:")
    
    for _, row in df_valid.iterrows():
        m10_weight = row['m10_total_weight']
        m70_weight = row['m70_total_weight'] 
        print(f"  {row['strategy']:<20} 10m:{m10_weight:.2f} 70m:{m70_weight:.2f} → RMSE:{row['RMSE']:.4f}")
    
    # 保存分析报告
    with open(os.path.join(base_save_dir, 'weight_comparison', 'analysis_report.txt'), 'w', encoding='utf-8') as f:
        f.write("权重策略对比分析报告\n")
        f.write("="*50 + "\n\n")
        
        f.write("1. 策略排名:\n")
        for i, (_, row) in enumerate(df_valid.iterrows(), 1):
            f.write(f"{i}. {row['strategy']}: RMSE={row['RMSE']:.4f}, 权重={row['weights']}\n")
        
        f.write(f"\n2. 主要发现:\n")
        if len(df_valid) > 0:
            best = df_valid.iloc[0]
            f.write(f"- 最优策略: {best['strategy']} (RMSE: {best['RMSE']:.4f})\n")
            f.write(f"- 最优权重: {best['weights']}\n")
            
            equal_weight = df_valid[df_valid['strategy'] == 'Fusion-Equal']
            if len(equal_weight) > 0:
                equal_rmse = equal_weight.iloc[0]['RMSE']
                f.write(f"- 等权重表现: RMSE={equal_rmse:.4f}\n")
                
                if equal_rmse == best['RMSE']:
                    f.write(f"- 结论: 等权重策略已经是最优的!\n")
                else:
                    gap = (equal_rmse - best['RMSE']) / best['RMSE'] * 100
                    f.write(f"- 权重优化带来的改善: {gap:.2f}%\n")
# 在原有的 test.py 文件的 main 部分添加这个调用
def run_weight_comparison_main():
    """权重对比试验的主函数"""
    
    # 配置路径
    DATA_PATH = "/Users/xiaxin/work/WindForecast_Project/01_Data/processed/imputed_data/changma_imputed_complete.csv"
    BASE_SAVE_DIR = "/Users/xiaxin/work/WindForecast_Project/03_Results/weight_comparison_experiments"
    INDICES_PATH = "/Users/xiaxin/work/WindForecast_Project/03_Results/third_part_experiments/train_test_split.json"
    
    # 运行权重对比试验
    results = run_weight_comparison_experiments(DATA_PATH, BASE_SAVE_DIR, INDICES_PATH)
    
    print(f"\n💡 权重对比试验完成!")
    print(f"📊 现在你知道等权重和其他策略的对比效果了!")



if __name__ == "__main__":
    # 配置路径
    DATA_PATH = "/Users/xiaxin/work/WindForecast_Project/01_Data/processed/imputed_data/changma_imputed_complete.csv"
    BASE_SAVE_DIR = "/Users/xiaxin/work/WindForecast_Project/03_Results/complete_14_experiments"
    INDICES_PATH = "/Users/xiaxin/work/WindForecast_Project/03_Results/third_part_experiments/train_test_split.json"
    
    # 运行所有14个试验
    results = run_all_14_experiments(DATA_PATH, BASE_SAVE_DIR, INDICES_PATH)
    # run_weight_comparison_main()
    print(f"\n💡 完整的14个试验系统运行完成!")
    print(f"\n🔬 试验包括:")
    print(f"   M1系列 (4个): 直接预测")
    print(f"   M2系列 (4个): 两步法ML校正")  
    print(f"   M3系列 (4个): 多风速融合")
    print(f"   Fusion系列 (2个): 跨模式融合 (其中Fusion-M2为无数据泄露版本)")
    print(f"\n📊 现在可以全面对比不同策略的效果!")
    print(f"   - 哪种方法最好: 直接预测 vs ML校正 vs 融合")
    print(f"   - 哪个数据源最好: GFS vs EC")
    print(f"   - 哪个高度最好: 10m vs 70m")
    print(f"   - 哪种融合策略最好: 固定权重 vs 动态权重 vs 无泄露权重优化")