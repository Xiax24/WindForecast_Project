#!/usr/bin/env python3
"""
重新训练并保存LightGBM模型 - 用于误差传播分析
快速版本：专门为保存模型而设计
Author: Research Team
Date: 2025-05-30
"""

import pandas as pd
import numpy as np
import joblib
import lightgbm as lgb
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score
import warnings
warnings.filterwarnings('ignore')

class LightGBMTrainerSaver:
    def __init__(self, data_path, save_path):
        self.data_path = data_path
        self.save_path = save_path
        self.data = None
        self.features = None
        self.target = None
        self.feature_names = None
        self.model = None
        self.scaler = None
        
    def load_and_prepare_data(self):
        """加载和预处理数据（与之前保持一致）"""
        print("📊 加载和预处理数据...")
        
        # 加载数据
        self.data = pd.read_csv(self.data_path)
        print(f"原始数据形状: {self.data.shape}")
        
        # 选择观测数据列
        obs_columns = [col for col in self.data.columns if col.startswith('obs_')]
        obs_columns += ['datetime', 'power']
        
        # 移除密度和湿度（与之前保持一致）
        obs_columns = [col for col in obs_columns if 'density' not in col and 'humidity' not in col]
        
        self.data = self.data[obs_columns].copy()
        print(f"选择列数: {len(obs_columns)-2}")  # 除去datetime和power
        
        # 移除缺失值和负功率
        initial_shape = self.data.shape[0]
        self.data = self.data.dropna()
        self.data = self.data[self.data['power'] >= 0]
        final_shape = self.data.shape[0]
        print(f"清理后数据: {final_shape} 行 (移除了 {initial_shape - final_shape} 行)")
        
        return self.data
    
    def process_wind_direction(self):
        """处理风向变量为sin/cos分量（正确的气象角度转换）"""
        print("🧭 处理风向变量...")
        
        # 找到风向列
        wind_dir_cols = [col for col in self.data.columns if 'wind_direction' in col]
        print(f"发现 {len(wind_dir_cols)} 个风向列: {wind_dir_cols}")
        
        # 处理每个风向列
        wind_dir_processed = {}
        for col in wind_dir_cols:
            print(f"  处理 {col}...")
            
            # 气象角度转换为数学角度
            # 气象学：0°=北，顺时针；数学：0°=东，逆时针
            math_angle = (90 - self.data[col] + 360) % 360
            wind_dir_rad = np.deg2rad(math_angle)
            
            # 创建sin/cos分量
            sin_col = col.replace('wind_direction', 'wind_dir_sin')  # 南北分量
            cos_col = col.replace('wind_direction', 'wind_dir_cos')  # 东西分量
            
            self.data[sin_col] = np.sin(wind_dir_rad)  # 南北分量
            self.data[cos_col] = np.cos(wind_dir_rad)  # 东西分量
            
            wind_dir_processed[col] = {'sin': sin_col, 'cos': cos_col}
            print(f"    转换 {col} → {sin_col} (南北分量), {cos_col} (东西分量)")
        
        # 移除原始风向列
        self.data = self.data.drop(columns=wind_dir_cols)
        print(f"✓ 移除原始风向列，添加了 {len(wind_dir_cols)*2} 个sin/cos列")
        print("✓ 使用正确的气象角度转换（0°=北，顺时针 → 0°=东，逆时针）")
        
        return wind_dir_processed
    
    def create_features(self):
        """创建特征矩阵"""
        print("🔧 创建特征矩阵...")
        
        # 处理风向
        self.process_wind_direction()
        
        # 选择所有观测变量（除了datetime和power）
        feature_cols = [col for col in self.data.columns 
                       if col not in ['datetime', 'power']]
        
        print(f"✓ 使用 {len(feature_cols)} 个特征:")
        
        # 按类型分组显示
        wind_speed_cols = [col for col in feature_cols if 'wind_speed' in col]
        wind_dir_cols = [col for col in feature_cols if 'wind_dir' in col]
        temp_cols = [col for col in feature_cols if 'temperature' in col]
        
        print(f"  - 风速变量 ({len(wind_speed_cols)}): {wind_speed_cols}")
        print(f"  - 风向变量 ({len(wind_dir_cols)}): {wind_dir_cols}")
        print(f"  - 温度变量 ({len(temp_cols)}): {temp_cols}")
        
        # 创建特征矩阵和目标向量
        self.features = self.data[feature_cols].values
        self.target = self.data['power'].values
        self.feature_names = feature_cols
        
        print(f"✓ 特征矩阵形状: {self.features.shape}")
        print(f"✓ 目标向量形状: {self.target.shape}")
        
        return feature_cols
    
    def train_lightgbm(self):
        """训练LightGBM模型（基于之前最佳参数）"""
        print("🚀 训练LightGBM模型...")
        
        # 数据分割（使用相同的random_state保证一致性）
        X_train, X_test, y_train, y_test = train_test_split(
            self.features, self.target, test_size=0.2, random_state=42
        )
        
        print(f"训练集: {X_train.shape}, 测试集: {X_test.shape}")
        
        # LightGBM参数网格（简化版，基于经验选择较好的参数范围）
        param_grid = {
            'num_leaves': [31, 63],
            'learning_rate': [0.05, 0.1],
            'reg_alpha': [0.1, 0.5],
            'reg_lambda': [0.1, 0.5],
            'min_child_samples': [20, 30]
        }
        
        # 创建LightGBM模型
        lgb_model = lgb.LGBMRegressor(
            objective='regression',
            metric='rmse',
            boosting_type='gbdt',
            n_estimators=200,
            random_state=42,
            verbose=-1,
            n_jobs=-1
        )
        
        print("🔍 执行网格搜索...")
        # 网格搜索
        grid_search = GridSearchCV(
            lgb_model, param_grid, cv=3,
            scoring='neg_mean_squared_error', 
            n_jobs=-1, 
            verbose=1
        )
        
        # 训练
        grid_search.fit(X_train, y_train)
        
        # 获取最佳模型
        self.model = grid_search.best_estimator_
        best_params = grid_search.best_params_
        
        print(f"✓ 最佳参数: {best_params}")
        
        # 评估模型性能
        y_pred_train = self.model.predict(X_train)
        y_pred_test = self.model.predict(X_test)
        
        train_r2 = r2_score(y_train, y_pred_train)
        test_r2 = r2_score(y_test, y_pred_test)
        train_rmse = np.sqrt(mean_squared_error(y_train, y_pred_train))
        test_rmse = np.sqrt(mean_squared_error(y_test, y_pred_test))
        
        print(f"📊 模型性能:")
        print(f"  训练集 R²: {train_r2:.4f}, RMSE: {train_rmse:.2f}")
        print(f"  测试集 R²: {test_r2:.4f}, RMSE: {test_rmse:.2f}")
        print(f"  过拟合差距: {train_r2 - test_r2:.4f}")
        
        # 存储训练数据用于后续分析
        self.train_test_data = {
            'X_train': X_train,
            'X_test': X_test,
            'y_train': y_train,
            'y_test': y_test,
            'y_pred_train': y_pred_train,
            'y_pred_test': y_pred_test,
            'best_params': best_params
        }
        
        return self.model
    
    def save_model_and_components(self):
        """保存模型和相关组件"""
        print("💾 保存模型和组件...")
        
        # 1. 保存LightGBM模型
        model_path = f"{self.save_path}/best_lightgbm_model.pkl"
        joblib.dump(self.model, model_path)
        print(f"✓ 模型已保存: {model_path}")
        
        # 2. 保存特征名称
        feature_names_path = f"{self.save_path}/feature_names.pkl"
        joblib.dump(self.feature_names, feature_names_path)
        print(f"✓ 特征名称已保存: {feature_names_path}")
        
        # 3. 保存模型元信息
        model_info = {
            'feature_names': self.feature_names,
            'n_features': len(self.feature_names),
            'model_type': 'LightGBM',
            'train_test_split_random_state': 42,
            'best_params': self.train_test_data['best_params'],
            'performance': {
                'train_r2': r2_score(self.train_test_data['y_train'], self.train_test_data['y_pred_train']),
                'test_r2': r2_score(self.train_test_data['y_test'], self.train_test_data['y_pred_test']),
                'train_rmse': np.sqrt(mean_squared_error(self.train_test_data['y_train'], self.train_test_data['y_pred_train'])),
                'test_rmse': np.sqrt(mean_squared_error(self.train_test_data['y_test'], self.train_test_data['y_pred_test']))
            }
        }
        
        info_path = f"{self.save_path}/model_info.pkl"
        joblib.dump(model_info, info_path)
        print(f"✓ 模型信息已保存: {info_path}")
        
        # 4. 只保存必要的组件（不保存函数，避免pickle问题）
        print("✓ 预测函数请直接使用模型：model.predict(input_data)")
        
        return {
            'model_path': model_path,
            'feature_names_path': feature_names_path,
            'info_path': info_path
        }
    
    def create_usage_example(self):
        """创建使用示例代码"""
        example_code = f'''
# ===== LightGBM模型使用示例 =====
import joblib
import numpy as np
import pandas as pd

# 1. 加载模型
model = joblib.load("{self.save_path}/best_lightgbm_model.pkl")
feature_names = joblib.load("{self.save_path}/feature_names.pkl")
model_info = joblib.load("{self.save_path}/model_info.pkl")

print("模型信息:", model_info['performance'])
print("特征列表:", feature_names)

# 2. 统一的预测函数（用于误差传播分析）
def fx_predict(input_data):
    \"\"\"
    统一的预测函数，用于误差传播分析
    input_data: numpy array, shape (n_samples, n_features)
    返回: numpy array, shape (n_samples,)
    \"\"\"
    if input_data.shape[1] != len(feature_names):
        raise ValueError(f"输入特征数量 {{input_data.shape[1]}} 与期望的 {{len(feature_names)}} 不匹配")
    
    return model.predict(input_data)

# 3. 准备输入数据（示例）
# 注意：输入数据必须包含这些特征，且顺序要一致！
required_features = {feature_names}

# 从你的DataFrame中选择特征（示例）
# df = pd.read_csv("your_data.csv")
# 
# # 处理风向（与训练时保持一致）
# wind_dir_cols = [col for col in df.columns if 'wind_direction' in col]
# for col in wind_dir_cols:
#     # 气象角度转换为数学角度
#     math_angle = (90 - df[col] + 360) % 360
#     wind_dir_rad = np.deg2rad(math_angle)
#     
#     sin_col = col.replace('wind_direction', 'wind_dir_sin')
#     cos_col = col.replace('wind_direction', 'wind_dir_cos')
#     df[sin_col] = np.sin(wind_dir_rad)
#     df[cos_col] = np.cos(wind_dir_rad)
# 
# # 移除原始风向列
# df = df.drop(columns=wind_dir_cols)
# 
# # 选择特征（顺序必须与训练时一致！）
# input_features = df[feature_names].values

# 4. 进行预测
# predictions = fx_predict(input_features)

# ===== 误差传播分析使用示例 =====
# 准备三套输入数据：
# obs_features = df[feature_names].values     # 观测数据特征
# ecmwf_features = df_ecmwf[feature_names_ecmwf].values  # ECMWF数据特征
# gfs_features = df_gfs[feature_names_gfs].values        # GFS数据特征
# actual_power = df['power'].values

# 误差分解：
# P_obs = fx_predict(obs_features)      # 用观测数据预测
# P_ecmwf = fx_predict(ecmwf_features)  # 用ECMWF预测数据
# P_gfs = fx_predict(gfs_features)      # 用GFS预测数据
# 
# # 计算误差分量：
# modeling_error = P_obs - actual_power      # 建模误差
# ecmwf_propagation = P_ecmwf - P_obs        # ECMWF输入误差传播
# gfs_propagation = P_gfs - P_obs            # GFS输入误差传播
# 
# # 分析误差统计特性：
# print("建模误差 RMSE:", np.sqrt(np.mean(modeling_error**2)))
# print("ECMWF传播误差 RMSE:", np.sqrt(np.mean(ecmwf_propagation**2)))
# print("GFS传播误差 RMSE:", np.sqrt(np.mean(gfs_propagation**2)))
        '''
        
        example_path = f"{self.save_path}/usage_example.py"
        with open(example_path, 'w', encoding='utf-8') as f:
            f.write(example_code)
        print(f"✓ 使用示例已保存: {example_path}")
        
        return example_path
    
    def run_training_and_save(self):
        """运行完整的训练和保存流程"""
        print("=" * 60)
        print("🎯 LightGBM模型重新训练和保存流程")
        print("=" * 60)
        
        try:
            # 1. 加载和预处理数据
            self.load_and_prepare_data()
            
            # 2. 创建特征
            self.create_features()
            
            # 3. 训练模型
            self.train_lightgbm()
            
            # 4. 保存模型和组件
            saved_paths = self.save_model_and_components()
            
            # 5. 创建使用示例
            self.create_usage_example()
            
            print("\n" + "=" * 60)
            print("🎉 训练和保存完成！")
            print("=" * 60)
            print("保存的文件:")
            for name, path in saved_paths.items():
                print(f"  ✓ {name}: {path}")
            
            print(f"\n📋 特征列表 ({len(self.feature_names)} 个):")
            for i, feature in enumerate(self.feature_names):
                print(f"  {i+1:2d}. {feature}")
            
            print(f"\n📊 最终模型性能:")
            train_r2 = r2_score(self.train_test_data['y_train'], self.train_test_data['y_pred_train'])
            test_r2 = r2_score(self.train_test_data['y_test'], self.train_test_data['y_pred_test'])
            print(f"  测试集 R²: {test_r2:.4f}")
            print(f"  过拟合风险: {'低' if train_r2 - test_r2 < 0.05 else '中' if train_r2 - test_r2 < 0.1 else '高'}")
            
            print(f"\n🚀 下一步：你可以开始误差传播分析了！")
            print(f"   使用保存的模型路径: {saved_paths['model_path']}")
            
            return True
            
        except Exception as e:
            print(f"❌ 训练过程出错: {str(e)}")
            import traceback
            traceback.print_exc()
            return False

if __name__ == "__main__":
    # 配置路径
    DATA_PATH = "/Users/xiaxin/work/WindForecast_Project/01_Data/processed/imputed_data/changma_imputed_complete.csv"
    SAVE_PATH = "/Users/xiaxin/work/WindForecast_Project/03_Results/saved_models"
    
    # 创建保存目录
    import os
    os.makedirs(SAVE_PATH, exist_ok=True)
    
    # 创建训练器并运行
    trainer = LightGBMTrainerSaver(DATA_PATH, SAVE_PATH)
    success = trainer.run_training_and_save()
    
    if success:
        print("\n🎯 现在你可以继续进行误差传播分析了！")
    else:
        print("\n⚠️ 训练失败，请检查错误信息")