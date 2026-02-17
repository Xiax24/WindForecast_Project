#!/usr/bin/env python3
"""
基于风向分类的Fusion-M1改进建模系统
根据四层风向（10m, 30m, 50m, 70m）的情况，动态选择变量参与建模

风向分类规则：
- 东风区间：45° - 135°（东北风至东南风）
- 西风区间：225° - 315°（西北风至西南风）

建模规则：
1. 四层均为东风 → 10m风速 + 30m风速 + 10m温度
2. 四层均为西风 → 10m风速 + 70m风速 + 10m温度
3. 其他情况 → 仅10m风速
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

def classify_wind_direction(direction):
    """
    分类风向
    
    Args:
        direction: 风向角度 (0-360度)
    
    Returns:
        'east': 东风区间 (45°-135°)
        'west': 西风区间 (225°-315°)
        'other': 其他
    """
    if pd.isna(direction):
        return 'other'
    
    # 确保角度在0-360范围内
    direction = direction % 360
    
    # 东风区间：45° 到 135°
    if 45 <= direction <= 135:
        return 'east'
    # 西风区间：225° 到 315°
    elif 225 <= direction <= 315:
        return 'west'
    else:
        return 'other'

def get_wind_direction_category(row):
    """
    判断样本的风向类别
    要求四层风向（10m, 30m, 50m, 70m）都在同一区间
    
    Returns:
        'east': 四层都是东风
        'west': 四层都是西风
        'other': 其他情况
    """
    # 获取四层风向
    dir_10m = row['ec_wind_direction_10m']
    dir_30m = row['ec_wind_direction_30m']
    dir_50m = row['ec_wind_direction_50m']
    dir_70m = row['ec_wind_direction_70m']
    
    # 分类每层风向
    cat_10m = classify_wind_direction(dir_10m)
    cat_30m = classify_wind_direction(dir_30m)
    cat_50m = classify_wind_direction(dir_50m)
    cat_70m = classify_wind_direction(dir_70m)
    
    # 只有四层都是同一类别才返回该类别
    if cat_10m == 'east' and cat_30m == 'east' and cat_50m == 'east' and cat_70m == 'east':
        return 'east'
    elif cat_10m == 'west' and cat_30m == 'west' and cat_50m == 'west' and cat_70m == 'west':
        return 'west'
    else:
        return 'other'

def create_train_test_split_if_needed(data_path, indices_path, test_ratio=0.2):
    """如果划分文件不存在，则创建它"""
    
    if os.path.exists(indices_path):
        print(f"  📋 使用已存在的划分文件: {indices_path}")
        return
    
    print(f"  📋 创建训练测试集划分文件...")
    
    # 加载数据
    data = pd.read_csv(data_path)
    data['datetime'] = pd.to_datetime(data['datetime'])
    
    # 基础清理
    data = data.dropna(subset=['power'])
    data = data[data['power'] >= 0]
    data = data.sort_values('datetime').reset_index(drop=True)
    
    # 按时间顺序划分
    total_samples = len(data)
    test_size = int(total_samples * test_ratio)
    train_size = total_samples - test_size
    
    # 训练集：前80%，测试集：后20%
    train_indices = list(range(train_size))
    test_indices = list(range(train_size, total_samples))
    
    # 保存划分
    split_data = {
        'train_indices': train_indices,
        'test_indices': test_indices,
        'total_samples': total_samples,
        'train_size': len(train_indices),
        'test_size': len(test_indices),
        'test_ratio': test_ratio,
        'split_method': 'time_based'
    }
    
    # 创建目录
    os.makedirs(os.path.dirname(indices_path), exist_ok=True)
    
    # 保存
    with open(indices_path, 'w') as f:
        json.dump(split_data, f, indent=2)
    
    print(f"  ✅ 划分文件创建完成: {len(train_indices)} 训练, {len(test_indices)} 测试")

def load_train_test_split(indices_path):
    """加载训练测试集划分"""
    with open(indices_path, 'r') as f:
        indices = json.load(f)
    return indices['train_indices'], indices['test_indices']

class WindCorrectionModel:
    """风速校正模型"""
    
    def __init__(self, wind_source, wind_height):
        """
        Args:
            wind_source: 'gfs' or 'ec'
            wind_height: '10m', '30m', '70m' etc.
        """
        self.wind_source = wind_source
        self.wind_height = wind_height
        self.model = None
        self.feature_names = None
        
    def prepare_correction_features(self, data, temperature_col=None):
        """准备风速校正的特征"""
        
        features = pd.DataFrame()
        
        # 主要预报风速
        main_wind_col = f'{self.wind_source}_wind_speed_{self.wind_height}'
        features['forecast_wind'] = data[main_wind_col]
        features['forecast_wind_2'] = data[main_wind_col] ** 2
        
        # 温度特征（如果提供）
        if temperature_col and temperature_col in data.columns:
            features['temperature'] = data[temperature_col]
        
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
    
    def train(self, data, train_indices, temperature_col=None):
        """训练风速校正模型"""
        
        features = self.prepare_correction_features(data, temperature_col)
        
        # 目标：观测风速
        target_col = f'obs_wind_speed_{self.wind_height}'
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
    
    def predict(self, data, temperature_col=None):
        """预测校正后的风速"""
        features = self.prepare_correction_features(data, temperature_col)
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

def run_wind_direction_based_fusion_model(data_path, save_dir, indices_path):
    """
    运行基于风向分类的Fusion-M1改进模型
    """
    
    print("=" * 80)
    print("🚀 基于风向分类的Fusion-M1改进建模系统")
    print("=" * 80)
    
    os.makedirs(save_dir, exist_ok=True)
    
    # 确保训练测试集划分文件存在
    create_train_test_split_if_needed(data_path, indices_path)
    
    # 加载数据
    print("\n📂 加载数据...")
    data = pd.read_csv(data_path)
    data['datetime'] = pd.to_datetime(data['datetime'])
    
    # 基础数据清理
    data = data.dropna(subset=['power'])
    data = data[data['power'] >= 0]
    data = data.sort_values('datetime').reset_index(drop=True)
    
    print(f"   数据大小: {len(data)}")
    
    # 获取训练测试集划分
    train_indices, test_indices = load_train_test_split(indices_path)
    
    # 第一步：风向分类
    print("\n🧭 第一步：风向分类...")
    print("   分类规则:")
    print("   - 东风: 45° - 135° (四层都需满足)")
    print("   - 西风: 225° - 315° (四层都需满足)")
    print("   - 其他: 不满足上述条件")
    
    # 检查必需的风向列
    required_dir_cols = [
        'ec_wind_direction_10m',
        'ec_wind_direction_30m', 
        'ec_wind_direction_50m',
        'ec_wind_direction_70m'
    ]
    
    missing_cols = [col for col in required_dir_cols if col not in data.columns]
    if missing_cols:
        print(f"   ❌ 缺少风向列: {missing_cols}")
        return None
    
    # 对每个样本进行风向分类
    data['wind_category'] = data.apply(get_wind_direction_category, axis=1)
    
    # 统计各类别样本数
    category_counts = data['wind_category'].value_counts()
    print(f"\n   风向分类统计:")
    print(f"   - 东风样本: {category_counts.get('east', 0)} ({category_counts.get('east', 0)/len(data)*100:.2f}%)")
    print(f"   - 西风样本: {category_counts.get('west', 0)} ({category_counts.get('west', 0)/len(data)*100:.2f}%)")
    print(f"   - 其他样本: {category_counts.get('other', 0)} ({category_counts.get('other', 0)/len(data)*100:.2f}%)")
    
    # 第二步：根据风向类别准备不同的建模配置
    print("\n🔧 第二步：根据风向类别准备建模配置...")
    
    # 为每种风向类别定义不同的风速配置
    wind_configs_by_category = {
        'east': {
            'description': '东风情况：10m + 30m风速 + 10m温度',
            'wind_configs': [
                {'source': 'gfs', 'height': '10m'},
                {'source': 'ec', 'height': '10m'},
                {'source': 'gfs', 'height': '30m'},
                {'source': 'ec', 'height': '30m'}
            ],
            'temperature_col': 'ec_temperature_10m',
            'weights': [0.25, 0.25, 0.25, 0.25]
        },
        'west': {
            'description': '西风情况：10m + 70m风速 + 10m温度',
            'wind_configs': [
                {'source': 'gfs', 'height': '10m'},
                {'source': 'ec', 'height': '10m'},
                {'source': 'gfs', 'height': '70m'},
                {'source': 'ec', 'height': '70m'}
            ],
            'temperature_col': 'ec_temperature_10m',
            'weights': [0.25, 0.25, 0.25, 0.25]
        },
        'other': {
            'description': '其他情况：仅10m风速',
            'wind_configs': [
                {'source': 'gfs', 'height': '10m'},
                {'source': 'ec', 'height': '10m'}
            ],
            'temperature_col': None,
            'weights': [0.5, 0.5]
        }
    }
    
    # 第三步：分别对每个类别建模
    print("\n⚙️ 第三步：分别对每个类别建模...")
    
    category_results = {}
    all_predictions = np.zeros(len(data))
    processing_logs = []
    
    for category in ['east', 'west', 'other']:
        if category not in category_counts or category_counts[category] == 0:
            print(f"\n   跳过 {category} (无样本)")
            continue
        
        print(f"\n   {'='*60}")
        print(f"   处理类别: {category.upper()}")
        print(f"   {wind_configs_by_category[category]['description']}")
        print(f"   {'='*60}")
        
        # 筛选该类别的数据
        category_mask = data['wind_category'] == category
        category_data = data[category_mask].copy()
        category_data = category_data.reset_index(drop=True)
        
        # 建立原索引到新索引的映射
        original_indices = data[category_mask].index.tolist()
        index_mapping = {new_idx: orig_idx for new_idx, orig_idx in enumerate(original_indices)}
        
        # 筛选该类别的训练测试索引
        category_train_indices = [i for i, orig_idx in enumerate(original_indices) if orig_idx in train_indices]
        category_test_indices = [i for i, orig_idx in enumerate(original_indices) if orig_idx in test_indices]
        
        print(f"   该类别样本数: {len(category_data)}")
        print(f"   训练样本: {len(category_train_indices)}, 测试样本: {len(category_test_indices)}")
        
        if len(category_test_indices) == 0:
            print(f"   ⚠️ 该类别无测试样本，跳过")
            continue
        
        # 准备该类别的风速配置
        wind_configs = wind_configs_by_category[category]['wind_configs']
        temperature_col = wind_configs_by_category[category]['temperature_col']
        weights = wind_configs_by_category[category]['weights']
        
        # 数据清理
        required_cols = ['power']
        for config in wind_configs:
            source = config['source']
            height = config['height']
            required_cols.append(f'{source}_wind_speed_{height}')
            required_cols.append(f'obs_wind_speed_{height}')
        
        if temperature_col:
            required_cols.append(temperature_col)
        
        # 清理缺失值和异常值
        category_data = category_data.dropna(subset=required_cols)
        for col in required_cols:
            if 'wind_speed' in col:
                category_data = category_data[(category_data[col] >= 0) & (category_data[col] <= 50)]
        
        category_data = category_data.reset_index(drop=True)
        
        # 更新索引映射
        original_indices_cleaned = category_data.index.tolist()
        category_train_indices_cleaned = [i for i in range(len(category_data)) 
                                         if index_mapping.get(i) in train_indices]
        category_test_indices_cleaned = [i for i in range(len(category_data)) 
                                        if index_mapping.get(i) in test_indices]
        
        print(f"   清理后样本数: {len(category_data)}")
        print(f"   清理后训练样本: {len(category_train_indices_cleaned)}, 测试样本: {len(category_test_indices_cleaned)}")
        
        if len(category_test_indices_cleaned) == 0:
            print(f"   ⚠️ 清理后无测试样本，跳过")
            continue
        
        # 第一步：分别校正每个风速
        print(f"\n   🎯 校正风速...")
        corrected_winds = {}
        correction_stats = {}
        
        for config in wind_configs:
            source = config['source']
            height = config['height']
            key = f"{source}_{height}"
            
            print(f"      校正 {key}...")
            
            wind_corrector = WindCorrectionModel(wind_source=source, wind_height=height)
            correction_stat = wind_corrector.train(
                category_data, 
                category_train_indices_cleaned,
                temperature_col=temperature_col
            )
            correction_stats[key] = correction_stat
            
            corrected_winds[key] = wind_corrector.predict(category_data, temperature_col=temperature_col)
            
            print(f"      {key} 校正RMSE: {correction_stat['rmse']:.4f}")
        
        # 第二步：融合校正后的风速
        print(f"\n   🔗 融合校正后风速...")
        fused_corrected_wind = np.zeros(len(category_data))
        
        for i, key in enumerate(corrected_winds.keys()):
            fused_corrected_wind += weights[i] * corrected_winds[key]
            print(f"      {key} 权重: {weights[i]}")
        
        # 第三步：用融合风速预测功率
        print(f"\n   ⚡ 功率预测...")
        
        power_predictor = PowerPredictionModel()
        power_results = power_predictor.train(
            fused_corrected_wind, 
            category_data,
            category_train_indices_cleaned,
            category_test_indices_cleaned
        )
        
        rmse = power_results['rmse']
        corr = power_results['correlation']
        y_pred = power_results['predictions']
        y_test = power_results['actual']
        
        print(f"\n   ✅ {category.upper()} 类别完成!")
        print(f"      RMSE: {rmse:.4f}")
        print(f"      相关系数: {corr:.4f}")
        
        # 保存该类别的结果
        category_results[category] = {
            'rmse': rmse,
            'correlation': corr,
            'sample_count': len(category_data),
            'test_count': len(category_test_indices_cleaned),
            'correction_stats': correction_stats,
            'description': wind_configs_by_category[category]['description']
        }
        
        # 将预测结果填充到全局预测数组
        for i, pred in enumerate(y_pred):
            # 找到原始数据的索引
            orig_idx = original_indices[category_test_indices_cleaned[i]]
            all_predictions[orig_idx] = pred
        
        # 记录处理日志
        processing_logs.append({
            'category': category,
            'description': wind_configs_by_category[category]['description'],
            'sample_count': len(category_data),
            'test_count': len(category_test_indices_cleaned),
            'rmse': rmse,
            'correlation': corr,
            'correction_stats': correction_stats
        })
    
    # 第四步：计算整体性能
    print(f"\n{'='*80}")
    print(f"📊 第四步：计算整体性能...")
    print(f"{'='*80}")
    
    # 获取测试集的实际功率和预测功率
    test_actual = data.iloc[test_indices]['power'].values
    test_predicted = all_predictions[test_indices]
    
    # 只计算有预测值的样本（非零）
    valid_mask = test_predicted != 0
    
    if valid_mask.sum() > 0:
        overall_rmse = np.sqrt(mean_squared_error(test_actual[valid_mask], test_predicted[valid_mask]))
        overall_corr = np.corrcoef(test_actual[valid_mask], test_predicted[valid_mask])[0, 1]
        
        print(f"\n整体测试性能:")
        print(f"   RMSE: {overall_rmse:.4f}")
        print(f"   相关系数: {overall_corr:.4f}")
        print(f"   有效预测样本数: {valid_mask.sum()} / {len(test_indices)}")
    else:
        print(f"\n   ⚠️ 无有效预测样本")
        overall_rmse = None
        overall_corr = None
    
    # 第五步：保存结果
    print(f"\n💾 第五步：保存结果...")
    
    # 保存详细结果
    detailed_results = pd.DataFrame({
        'datetime': data.iloc[test_indices]['datetime'].values,
        'actual_power': test_actual,
        'predicted_power': test_predicted,
        'wind_category': data.iloc[test_indices]['wind_category'].values,
        'error': test_predicted - test_actual,
        'abs_error': np.abs(test_predicted - test_actual)
    })
    
    detailed_results.to_csv(os.path.join(save_dir, 'detailed_results.csv'), index=False)
    print(f"   ✅ 详细结果已保存")
    
    # 保存各类别结果
    with open(os.path.join(save_dir, 'category_results.json'), 'w', encoding='utf-8') as f:
        json.dump(category_results, f, indent=2, ensure_ascii=False)
    print(f"   ✅ 各类别结果已保存")
    
    # 保存整体指标
    overall_metrics = {
        'overall_rmse': overall_rmse,
        'overall_correlation': overall_corr,
        'valid_predictions': int(valid_mask.sum()) if valid_mask.sum() > 0 else 0,
        'total_test_samples': len(test_indices),
        'category_results': category_results
    }
    
    with open(os.path.join(save_dir, 'overall_metrics.json'), 'w', encoding='utf-8') as f:
        json.dump(overall_metrics, f, indent=2, ensure_ascii=False)
    print(f"   ✅ 整体指标已保存")
    
    # 保存处理日志
    process_log = {
        'model_description': '基于风向分类的Fusion-M1改进模型',
        'wind_direction_rules': {
            'east': '45° - 135° (四层都需满足)',
            'west': '225° - 315° (四层都需满足)',
            'other': '不满足上述条件'
        },
        'modeling_strategies': wind_configs_by_category,
        'category_statistics': {
            'east': int(category_counts.get('east', 0)),
            'west': int(category_counts.get('west', 0)),
            'other': int(category_counts.get('other', 0))
        },
        'processing_logs': processing_logs,
        'overall_performance': {
            'rmse': overall_rmse,
            'correlation': overall_corr
        }
    }
    
    with open(os.path.join(save_dir, 'process_log.json'), 'w', encoding='utf-8') as f:
        json.dump(process_log, f, indent=2, ensure_ascii=False, default=str)
    print(f"   ✅ 处理日志已保存")
    
    # 生成分类统计报告
    print(f"\n📈 生成分类统计报告...")
    
    summary_lines = []
    summary_lines.append("=" * 80)
    summary_lines.append("基于风向分类的Fusion-M1改进模型 - 结果汇总")
    summary_lines.append("=" * 80)
    summary_lines.append("")
    summary_lines.append("1. 风向分类统计:")
    summary_lines.append(f"   - 东风样本: {category_counts.get('east', 0)} ({category_counts.get('east', 0)/len(data)*100:.2f}%)")
    summary_lines.append(f"   - 西风样本: {category_counts.get('west', 0)} ({category_counts.get('west', 0)/len(data)*100:.2f}%)")
    summary_lines.append(f"   - 其他样本: {category_counts.get('other', 0)} ({category_counts.get('other', 0)/len(data)*100:.2f}%)")
    summary_lines.append("")
    summary_lines.append("2. 各类别建模策略:")
    for category, config in wind_configs_by_category.items():
        summary_lines.append(f"   {category.upper()}: {config['description']}")
    summary_lines.append("")
    summary_lines.append("3. 各类别性能:")
    for category, result in category_results.items():
        summary_lines.append(f"   {category.upper()}:")
        summary_lines.append(f"      RMSE: {result['rmse']:.4f}")
        summary_lines.append(f"      相关系数: {result['correlation']:.4f}")
        summary_lines.append(f"      测试样本数: {result['test_count']}")
    summary_lines.append("")
    summary_lines.append("4. 整体性能:")
    if overall_rmse is not None:
        summary_lines.append(f"   RMSE: {overall_rmse:.4f}")
        summary_lines.append(f"   相关系数: {overall_corr:.4f}")
        summary_lines.append(f"   有效预测样本: {valid_mask.sum()} / {len(test_indices)}")
    else:
        summary_lines.append("   无有效预测")
    summary_lines.append("")
    summary_lines.append("=" * 80)
    
    summary_text = "\n".join(summary_lines)
    print(summary_text)
    
    with open(os.path.join(save_dir, 'summary_report.txt'), 'w', encoding='utf-8') as f:
        f.write(summary_text)
    print(f"   ✅ 汇总报告已保存")
    
    print(f"\n{'='*80}")
    print(f"🎉 基于风向分类的Fusion-M1改进建模完成!")
    print(f"📁 结果保存在: {save_dir}")
    print(f"{'='*80}")
    
    return overall_metrics

if __name__ == "__main__":
    # 配置路径
    DATA_PATH = "/Users/xiaxin/work/WindForecast_Project/01_Data/processed/imputed_data/changma_imputed_complete.csv"
    SAVE_DIR = "/Users/xiaxin/work/WindForecast_Project/03_Results/建模试验/wind_direction_based_fusion"
    INDICES_PATH = "/Users/xiaxin/work/WindForecast_Project/03_Results/建模试验/third_part_experiments/train_test_split.json"
    
    # 运行基于风向分类的Fusion-M1改进模型
    results = run_wind_direction_based_fusion_model(DATA_PATH, SAVE_DIR, INDICES_PATH)
    
    print("\n💡 建模完成!")
    print("\n🎯 主要创新点:")
    print("   1. 根据四层风向（10m, 30m, 50m, 70m）自动分类")
    print("   2. 东风情况：使用10m+30m风速+温度")
    print("   3. 西风情况：使用10m+70m风速+温度")
    print("   4. 其他情况：仅使用10m风速")
    print("   5. 每种情况都采用Fusion-M1的校正融合策略")