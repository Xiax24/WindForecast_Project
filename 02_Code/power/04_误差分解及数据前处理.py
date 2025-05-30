#!/usr/bin/env python3
"""
误差传播分析 - 数据预处理模块
处理观测、ECMWF、GFS数据，使其格式统一，便于误差传播分析
Author: Research Team
Date: 2025-05-30
"""

import pandas as pd
import numpy as np
import joblib
import warnings
warnings.filterwarnings('ignore')

class ErrorPropagationDataPreprocessor:
    def __init__(self, data_path, model_path):
        self.data_path = data_path
        self.model_path = model_path
        self.data = None
        self.model = None
        self.feature_names = None
        self.model_info = None
        
        # 处理后的数据
        self.obs_features = None
        self.ecmwf_features = None
        self.gfs_features = None
        self.actual_power = None
        
    def load_model_info(self):
        """加载训练好的模型和特征信息"""
        print("📦 加载模型和特征信息...")
        
        # 加载模型
        self.model = joblib.load(f"{self.model_path}/best_lightgbm_model.pkl")
        self.feature_names = joblib.load(f"{self.model_path}/feature_names.pkl")
        self.model_info = joblib.load(f"{self.model_path}/model_info.pkl")
        
        print(f"✅ 模型加载成功")
        print(f"📊 模型性能: 测试集 R² = {self.model_info['performance']['test_r2']:.4f}")
        print(f"🔧 特征数量: {len(self.feature_names)}")
        
        # 显示特征列表
        print("📋 模型期望的特征列表:")
        for i, feature in enumerate(self.feature_names):
            print(f"  {i+1:2d}. {feature}")
        
        return True
    
    def load_raw_data(self):
        """加载原始数据"""
        print("📊 加载原始数据...")
        
        self.data = pd.read_csv(self.data_path)
        print(f"✅ 数据加载成功: {self.data.shape}")
        
        # 检查数据完整性
        print("🔍 数据完整性检查:")
        print(f"  时间范围: {self.data['datetime'].min()} 到 {self.data['datetime'].max()}")
        print(f"  总行数: {len(self.data)}")
        
        # 检查缺失值
        missing_summary = self.data.isnull().sum()
        missing_cols = missing_summary[missing_summary > 0]
        if len(missing_cols) > 0:
            print("⚠️  发现缺失值:")
            for col, count in missing_cols.items():
                print(f"    {col}: {count} ({count/len(self.data)*100:.1f}%)")
        else:
            print("✅ 无缺失值")
        
        return self.data
    
    def process_wind_direction_for_source(self, source_prefix):
        """
        处理特定数据源的风向变量
        source_prefix: 'obs', 'ec', 'gfs'
        """
        print(f"🧭 处理 {source_prefix} 风向数据...")
        
        # 找到该数据源的风向列
        wind_dir_cols = [col for col in self.data.columns 
                        if col.startswith(f'{source_prefix}_wind_direction')]
        
        print(f"  发现 {len(wind_dir_cols)} 个风向列: {wind_dir_cols}")
        
        processed_cols = {}
        for col in wind_dir_cols:
            print(f"    处理 {col}...")
            
            # 气象角度转换为数学角度
            # 气象学：0°=北，顺时针；数学：0°=东，逆时针
            math_angle = (90 - self.data[col] + 360) % 360
            wind_dir_rad = np.deg2rad(math_angle)
            
            # 创建sin/cos分量（统一命名格式）
            height = col.split('_')[-1]  # 提取高度信息，如 '10m'
            sin_col = f'{source_prefix}_wind_dir_sin_{height}'
            cos_col = f'{source_prefix}_wind_dir_cos_{height}'
            
            self.data[sin_col] = np.sin(wind_dir_rad)  # 南北分量
            self.data[cos_col] = np.cos(wind_dir_rad)  # 东西分量
            
            processed_cols[col] = {'sin': sin_col, 'cos': cos_col}
            print(f"      → {sin_col} (南北), {cos_col} (东西)")
        
        return processed_cols
    
    def create_feature_mapping(self):
        """创建观测特征到预测特征的映射关系"""
        print("🔗 创建特征映射关系...")
        
        # 建立映射关系
        obs_to_ecmwf = {}
        obs_to_gfs = {}
        
        for obs_feature in self.feature_names:
            # 风速映射
            if 'wind_speed' in obs_feature:
                ecmwf_feature = obs_feature.replace('obs_', 'ec_')
                gfs_feature = obs_feature.replace('obs_', 'gfs_')
                
            # 风向sin/cos映射
            elif 'wind_dir_sin' in obs_feature:
                ecmwf_feature = obs_feature.replace('obs_', 'ec_')
                gfs_feature = obs_feature.replace('obs_', 'gfs_')
                
            elif 'wind_dir_cos' in obs_feature:
                ecmwf_feature = obs_feature.replace('obs_', 'ec_')
                gfs_feature = obs_feature.replace('obs_', 'gfs_')
                
            # 温度映射
            elif 'temperature' in obs_feature:
                ecmwf_feature = obs_feature.replace('obs_', 'ec_')
                gfs_feature = obs_feature.replace('obs_', 'gfs_')
                
            else:
                print(f"⚠️  未知特征类型: {obs_feature}")
                continue
            
            obs_to_ecmwf[obs_feature] = ecmwf_feature
            obs_to_gfs[obs_feature] = gfs_feature
        
        print(f"✅ 特征映射创建完成: {len(obs_to_ecmwf)} 个特征")
        
        # 验证映射的特征是否存在
        print("🔍 验证特征存在性:")
        missing_ecmwf = [f for f in obs_to_ecmwf.values() if f not in self.data.columns]
        missing_gfs = [f for f in obs_to_gfs.values() if f not in self.data.columns]
        
        if missing_ecmwf:
            print(f"❌ ECMWF缺失特征: {missing_ecmwf}")
        else:
            print("✅ ECMWF特征完整")
            
        if missing_gfs:
            print(f"❌ GFS缺失特征: {missing_gfs}")
        else:
            print("✅ GFS特征完整")
        
        return obs_to_ecmwf, obs_to_gfs
    
    def extract_features(self, obs_to_ecmwf, obs_to_gfs):
        """提取三套特征矩阵"""
        print("🔧 提取特征矩阵...")
        
        # 清理数据：移除缺失值和负功率
        print("🧹 数据清理...")
        initial_shape = self.data.shape[0]
        
        # 移除功率缺失或负值的行
        clean_data = self.data.dropna(subset=['power'])
        clean_data = clean_data[clean_data['power'] >= 0]
        
        # 移除观测特征缺失的行
        obs_features_to_check = [f for f in self.feature_names if f in clean_data.columns]
        clean_data = clean_data.dropna(subset=obs_features_to_check)
        
        # 移除ECMWF特征缺失的行
        ecmwf_features_to_check = [f for f in obs_to_ecmwf.values() if f in clean_data.columns]
        clean_data = clean_data.dropna(subset=ecmwf_features_to_check)
        
        # 移除GFS特征缺失的行
        gfs_features_to_check = [f for f in obs_to_gfs.values() if f in clean_data.columns]
        clean_data = clean_data.dropna(subset=gfs_features_to_check)
        
        final_shape = clean_data.shape[0]
        print(f"  清理前: {initial_shape} 行")
        print(f"  清理后: {final_shape} 行")
        print(f"  移除: {initial_shape - final_shape} 行 ({(initial_shape - final_shape)/initial_shape*100:.1f}%)")
        
        # 提取观测特征矩阵
        print("📊 提取观测特征...")
        self.obs_features = clean_data[self.feature_names].values
        print(f"  观测特征矩阵: {self.obs_features.shape}")
        
        # 提取ECMWF特征矩阵
        print("📊 提取ECMWF特征...")
        ecmwf_feature_names = [obs_to_ecmwf[f] for f in self.feature_names]
        self.ecmwf_features = clean_data[ecmwf_feature_names].values
        print(f"  ECMWF特征矩阵: {self.ecmwf_features.shape}")
        
        # 提取GFS特征矩阵
        print("📊 提取GFS特征...")
        gfs_feature_names = [obs_to_gfs[f] for f in self.feature_names]
        self.gfs_features = clean_data[gfs_feature_names].values
        print(f"  GFS特征矩阵: {self.gfs_features.shape}")
        
        # 提取真实功率
        self.actual_power = clean_data['power'].values
        print(f"  真实功率向量: {self.actual_power.shape}")
        
        # 保存清理后的数据和时间信息
        self.clean_datetime = clean_data['datetime'].values
        
        # 数据统计
        print("\n📈 数据统计摘要:")
        print(f"  功率范围: {self.actual_power.min():.1f} - {self.actual_power.max():.1f} kW")
        print(f"  功率均值: {self.actual_power.mean():.1f} kW")
        
        return True
    
    def validate_data_consistency(self):
        """验证三套数据的一致性"""
        print("🔍 验证数据一致性...")
        
        # 检查形状一致性
        shapes = {
            'obs': self.obs_features.shape,
            'ecmwf': self.ecmwf_features.shape,
            'gfs': self.gfs_features.shape,
            'power': self.actual_power.shape
        }
        
        print("📏 数据形状:")
        for name, shape in shapes.items():
            print(f"  {name}: {shape}")
        
        # 检查是否所有数据行数一致
        n_samples = [self.obs_features.shape[0], self.ecmwf_features.shape[0], 
                    self.gfs_features.shape[0], len(self.actual_power)]
        
        if len(set(n_samples)) == 1:
            print("✅ 所有数据样本数一致")
        else:
            print("❌ 数据样本数不一致！")
            return False
        
        # 检查特征数一致性
        n_features = [self.obs_features.shape[1], self.ecmwf_features.shape[1], self.gfs_features.shape[1]]
        if len(set(n_features)) == 1:
            print("✅ 所有数据特征数一致")
        else:
            print("❌ 数据特征数不一致！")
            return False
        
        # 简单的数值范围检查
        print("\n📊 数值范围检查:")
        for i, name in enumerate(['obs', 'ecmwf', 'gfs']):
            data = [self.obs_features, self.ecmwf_features, self.gfs_features][i]
            print(f"  {name}特征范围: {data.min():.2f} - {data.max():.2f}")
            
            # 检查异常值
            if np.any(np.isnan(data)):
                print(f"    ⚠️ {name}包含NaN值")
            if np.any(np.isinf(data)):
                print(f"    ⚠️ {name}包含无穷值")
        
        return True
    
    def create_prediction_function(self):
        """创建统一的预测函数"""
        print("🔧 创建预测函数...")
        
        def fx_predict(input_data, data_source="unknown"):
            """
            统一的预测函数，用于误差传播分析
            
            Parameters:
            input_data: numpy array, shape (n_samples, n_features)
            data_source: str, 数据源标识（用于调试）
            
            Returns:
            predictions: numpy array, shape (n_samples,)
            """
            if input_data.shape[1] != len(self.feature_names):
                raise ValueError(f"输入特征数量 {input_data.shape[1]} 与期望的 {len(self.feature_names)} 不匹配")
            
            # 检查数据有效性
            if np.any(np.isnan(input_data)):
                print(f"⚠️  {data_source}数据包含NaN值")
            
            predictions = self.model.predict(input_data)
            print(f"✅ {data_source}预测完成: {len(predictions)} 个样本")
            
            return predictions
        
        return fx_predict
    
    def save_processed_data(self, save_path):
        """保存预处理后的数据"""
        print("💾 保存预处理数据...")
        
        import os
        os.makedirs(save_path, exist_ok=True)
        
        # 保存特征矩阵
        np.save(f"{save_path}/obs_features.npy", self.obs_features)
        np.save(f"{save_path}/ecmwf_features.npy", self.ecmwf_features)
        np.save(f"{save_path}/gfs_features.npy", self.gfs_features)
        np.save(f"{save_path}/actual_power.npy", self.actual_power)
        
        # 保存时间信息
        pd.Series(self.clean_datetime).to_csv(f"{save_path}/datetime_index.csv", index=False, header=['datetime'])
        
        # 保存特征名称映射
        feature_mapping = {
            'obs_features': self.feature_names,
            'n_samples': len(self.actual_power),
            'n_features': len(self.feature_names),
            'data_range': {
                'start': str(self.clean_datetime[0]),
                'end': str(self.clean_datetime[-1])
            }
        }
        
        joblib.dump(feature_mapping, f"{save_path}/feature_mapping.pkl")
        
        print(f"✅ 数据已保存到: {save_path}")
        print(f"  📁 obs_features.npy: {self.obs_features.shape}")
        print(f"  📁 ecmwf_features.npy: {self.ecmwf_features.shape}")
        print(f"  📁 gfs_features.npy: {self.gfs_features.shape}")
        print(f"  📁 actual_power.npy: {self.actual_power.shape}")
        
        return save_path
    
    def run_preprocessing(self, save_path=None):
        """运行完整的数据预处理流程"""
        print("=" * 60)
        print("🎯 误差传播分析 - 数据预处理流程")
        print("=" * 60)
        
        try:
            # 1. 加载模型信息
            self.load_model_info()
            
            # 2. 加载原始数据
            self.load_raw_data()
            
            # 3. 处理风向数据
            self.process_wind_direction_for_source('obs')
            self.process_wind_direction_for_source('ec')
            self.process_wind_direction_for_source('gfs')
            
            # 4. 创建特征映射
            obs_to_ecmwf, obs_to_gfs = self.create_feature_mapping()
            
            # 5. 提取特征矩阵
            self.extract_features(obs_to_ecmwf, obs_to_gfs)
            
            # 6. 验证数据一致性
            self.validate_data_consistency()
            
            # 7. 创建预测函数
            fx_predict = self.create_prediction_function()
            
            # 8. 保存预处理数据（可选）
            if save_path:
                self.save_processed_data(save_path)
            
            print("\n" + "=" * 60)
            print("🎉 数据预处理完成！")
            print("=" * 60)
            print(f"📊 最终数据摘要:")
            print(f"  样本数: {len(self.actual_power)}")
            print(f"  特征数: {len(self.feature_names)}")
            print(f"  时间范围: {self.clean_datetime[0]} 到 {self.clean_datetime[-1]}")
            
            print(f"\n🚀 现在可以开始误差传播分析:")
            print(f"  P_obs = fx_predict(obs_features)")
            print(f"  P_ecmwf = fx_predict(ecmwf_features)")
            print(f"  P_gfs = fx_predict(gfs_features)")
            
            return {
                'obs_features': self.obs_features,
                'ecmwf_features': self.ecmwf_features,
                'gfs_features': self.gfs_features,
                'actual_power': self.actual_power,
                'datetime': self.clean_datetime,
                'fx_predict': fx_predict,
                'feature_names': self.feature_names
            }
            
        except Exception as e:
            print(f"❌ 预处理过程出错: {str(e)}")
            import traceback
            traceback.print_exc()
            return None

if __name__ == "__main__":
    # 配置路径
    DATA_PATH = "/Users/xiaxin/work/WindForecast_Project/01_Data/processed/imputed_data/changma_imputed_complete.csv"
    MODEL_PATH = "/Users/xiaxin/work/WindForecast_Project/03_Results/03saved_models"
    SAVE_PATH = "/Users/xiaxin/work/WindForecast_Project/03_Results/04error_propagation_data"
    
    # 创建预处理器
    preprocessor = ErrorPropagationDataPreprocessor(DATA_PATH, MODEL_PATH)
    
    # 运行预处理
    results = preprocessor.run_preprocessing(save_path=SAVE_PATH)
    
    if results:
        print("\n🎯 预处理成功！现在可以开始误差传播分析了！")
        
        # 快速验证
        fx_predict = results['fx_predict']
        obs_features = results['obs_features']
        
        print("\n🧪 快速预测测试:")
        test_sample = obs_features[:10]  # 测试前10个样本
        test_predictions = fx_predict(test_sample, "obs_test")
        print(f"  测试预测结果: {test_predictions[:5]}...")
        
    else:
        print("\n⚠️ 预处理失败，请检查错误信息")