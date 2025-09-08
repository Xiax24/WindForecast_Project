#!/usr/bin/env python3
"""
简化版风电功率预测 - 仅LightGBM + SHAP分析
生成3个SHAP可视化图表：特征重要性 + 影响分布 + 力图
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import lightgbm as lgb
import shap
import warnings
warnings.filterwarnings('ignore')

# 设置绘图样式
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

class SimpleLightGBMAnalyzer:
    def __init__(self, data_path, results_path):
        self.data_path = data_path
        self.results_path = results_path
        self.data = None
        self.features = None
        self.target = None
        self.feature_names = None
        self.model = None
        self.X_test = None
        self.y_test = None
        
    def load_and_prepare_data(self):
        """加载和预处理数据"""
        print("加载和预处理数据...")
        
        # 加载数据
        self.data = pd.read_csv(self.data_path)
        print(f"原始数据形状: {self.data.shape}")
        
        # 选择观测数据列
        obs_columns = [col for col in self.data.columns if col.startswith('obs_')]
        obs_columns += ['datetime', 'power']
        
        # 移除密度和湿度变量
        obs_columns = [col for col in obs_columns if 'density' not in col and 'humidity' not in col]
        
        self.data = self.data[obs_columns].copy()
        print(f"选择列后的数据形状: {self.data.shape}")
        
        # 清理数据
        self.data = self.data.dropna()
        self.data = self.data[self.data['power'] >= 0]
        
        print(f"最终数据形状: {self.data.shape}")
        return self.data
    
    def process_wind_direction(self):
        """处理风向变量为sin/cos分量"""
        print("处理风向变量...")
        
        wind_dir_cols = [col for col in self.data.columns if 'wind_direction' in col]
        print(f"发现{len(wind_dir_cols)}个风向变量: {wind_dir_cols}")
        
        for col in wind_dir_cols:
            # 转换为弧度
            wind_dir_rad = np.deg2rad(self.data[col])
            
            # 创建sin/cos分量
            sin_col = col.replace('wind_direction', 'wind_dir_sin')
            cos_col = col.replace('wind_direction', 'wind_dir_cos')
            
            self.data[sin_col] = np.sin(wind_dir_rad)
            self.data[cos_col] = np.cos(wind_dir_rad)
            
            print(f"  已创建: {sin_col}, {cos_col}")
        
        # 移除原始风向列
        self.data = self.data.drop(columns=wind_dir_cols)
        print(f"已移除原始风向列")
    
    def create_features(self):
        """创建特征矩阵"""
        print("创建特征矩阵...")
        
        # 处理风向
        self.process_wind_direction()
        
        # 选择特征列
        feature_cols = [col for col in self.data.columns 
                       if col not in ['datetime', 'power']]
        
        print(f"使用{len(feature_cols)}个特征")
        
        # 创建特征矩阵
        self.features = self.data[feature_cols].values
        self.target = self.data['power'].values
        self.feature_names = feature_cols
        
        print(f"特征矩阵形状: {self.features.shape}")
        
        return feature_cols
    
    def train_lightgbm(self):
        """训练LightGBM模型"""
        print("训练LightGBM模型...")
        
        # 分割数据
        X_train, X_test, y_train, y_test = train_test_split(
            self.features, self.target, test_size=0.2, random_state=42
        )
        
        # 保存测试数据
        self.X_test = X_test
        self.y_test = y_test
        
        # LightGBM参数
        lgb_params = {
            'objective': 'regression',
            'metric': 'rmse',
            'boosting_type': 'gbdt',
            'num_leaves': 63,
            'learning_rate': 0.1,
            'feature_fraction': 0.8,
            'bagging_fraction': 0.8,
            'reg_alpha': 0.1,
            'reg_lambda': 0.1,
            'min_child_samples': 20,
            'n_estimators': 200,
            'random_state': 42,
            'verbose': -1
        }
        
        # 训练模型
        self.model = lgb.LGBMRegressor(**lgb_params)
        self.model.fit(X_train, y_train)
        
        # 评估性能
        y_pred_test = self.model.predict(X_test)
        test_r2 = r2_score(y_test, y_pred_test)
        test_rmse = np.sqrt(mean_squared_error(y_test, y_pred_test))
        test_mae = mean_absolute_error(y_test, y_pred_test)
        
        print(f"LightGBM模型性能:")
        print(f"  测试集 R²: {test_r2:.3f}")
        print(f"  测试集 RMSE: {test_rmse:.3f}")
        print(f"  测试集 MAE: {test_mae:.3f}")
        
        return self.model
    
    def perform_shap_analysis(self):
        """执行SHAP分析并生成3个独立的可视化图表"""
        print("执行SHAP分析...")
        
        if self.model is None:
            raise ValueError("请先训练模型")
        
        # 创建SHAP解释器
        explainer = shap.TreeExplainer(self.model)
        
        # 使用测试数据样本进行SHAP分析
        sample_size = min(1000, len(self.X_test))
        indices = np.random.choice(len(self.X_test), sample_size, replace=False)
        X_sample = self.X_test[indices]
        
        print(f"计算{sample_size}个样本的SHAP值...")
        shap_values = explainer.shap_values(X_sample)
        
        # 创建特征名称（去掉obs_前缀，便于显示）
        display_names = [name.replace('obs_', '').replace('_', ' ') 
                        for name in self.feature_names]
        
        print("生成SHAP可视化图表...")
        
        # 图1: 特征重要性条形图
        plt.figure(figsize=(8, 6))
        shap.summary_plot(shap_values, X_sample, feature_names=display_names, 
                        plot_type="bar", show=False, max_display=10)
        plt.title('Feature Importance (SHAP)', fontsize=14, fontweight='bold', pad=20)
        plt.tight_layout()
        plt.savefig(f'{self.results_path}/shap_feature_importance.png', 
                   dpi=300, bbox_inches='tight', facecolor='white')
        plt.savefig(f'{self.results_path}/shap_feature_importance.pdf', 
                   dpi=300, bbox_inches='tight', facecolor='white')
        plt.show()
        
        # 图2: 特征影响分布图（蜂群图）
        plt.figure(figsize=(8, 6))
        shap.summary_plot(shap_values, X_sample, feature_names=display_names, 
                        show=False, max_display=10)
        plt.title('Feature Impact Distribution', fontsize=14, fontweight='bold', pad=20)
        plt.tight_layout()
        plt.savefig(f'{self.results_path}/shap_impact_distribution.png', 
                   dpi=300, bbox_inches='tight', facecolor='white')
        plt.savefig(f'{self.results_path}/shap_impact_distribution.pdf', 
                   dpi=300, bbox_inches='tight', facecolor='white')
        plt.show()
        
        # 图3: 单个预测解释 - 瀑布图
        print("生成单个预测解释瀑布图...")
        
        # 选择预测值接近中位数的样本作为代表性个例
        y_pred_sample = self.model.predict(X_sample)
        median_idx = np.argsort(np.abs(y_pred_sample - np.median(y_pred_sample)))[0]
        
        plt.figure(figsize=(10, 6))
        try:
            shap.waterfall_plot(
                shap.Explanation(values=shap_values[median_idx], 
                               base_values=explainer.expected_value,
                               data=X_sample[median_idx],
                               feature_names=display_names),
                show=False
            )
            plt.title(f'Single Prediction Example (Predicted: {y_pred_sample[median_idx]:.1f} kW)', 
                     fontsize=14, fontweight='bold', pad=20)
        except Exception as e:
            plt.text(0.5, 0.5, f'Waterfall plot unavailable\n{str(e)[:50]}...', 
                    ha='center', va='center', transform=plt.gca().transAxes, fontsize=12)
            plt.title('Single Prediction Example', fontsize=14, fontweight='bold', pad=20)
        
        plt.tight_layout()
        plt.savefig(f'{self.results_path}/shap_waterfall.png', 
                   dpi=300, bbox_inches='tight', facecolor='white')
        plt.savefig(f'{self.results_path}/shap_waterfall.pdf', 
                   dpi=300, bbox_inches='tight', facecolor='white')
        plt.show()
        
        return shap_values, X_sample
    
    def save_results(self, shap_values, X_sample):
        """保存SHAP分析结果"""
        print("保存SHAP分析结果...")
        
        # 计算特征重要性
        shap_importance = np.abs(shap_values).mean(0)
        
        # 创建特征重要性DataFrame
        importance_df = pd.DataFrame({
            'Feature': self.feature_names,
            'SHAP_Importance': shap_importance,
            'SHAP_Std': np.abs(shap_values).std(0)
        }).sort_values('SHAP_Importance', ascending=False)
        
        # 保存特征重要性
        importance_df.to_csv(f'{self.results_path}/lightgbm_shap_importance.csv', index=False)
        
        # 保存模型性能
        y_pred = self.model.predict(self.X_test)
        performance = {
            'Model': 'LightGBM',
            'Test_R2': r2_score(self.y_test, y_pred),
            'Test_RMSE': np.sqrt(mean_squared_error(self.y_test, y_pred)),
            'Test_MAE': mean_absolute_error(self.y_test, y_pred),
            'Features_Used': len(self.feature_names)
        }
        
        performance_df = pd.DataFrame([performance])
        performance_df.to_csv(f'{self.results_path}/lightgbm_performance.csv', index=False)
        
        print(f"\n模型性能:")
        print(f"  测试集R²: {performance['Test_R2']:.3f}")
        print(f"  测试集RMSE: {performance['Test_RMSE']:.3f}")
        print(f"  使用特征数: {performance['Features_Used']}")
        
        print(f"\n前5个最重要特征:")
        for i, (_, row) in enumerate(importance_df.head().iterrows()):
            print(f"  {i+1}. {row['Feature']}: {row['SHAP_Importance']:.3f}")
    
    def run_analysis(self):
        """运行完整分析"""
        print("=== 简化版LightGBM + SHAP分析 ===")
        
        # 1. 加载和预处理数据
        self.load_and_prepare_data()
        
        # 2. 创建特征
        self.create_features()
        
        # 3. 训练LightGBM模型
        self.train_lightgbm()
        
        # 4. 执行SHAP分析并生成可视化
        shap_values, X_sample = self.perform_shap_analysis()
        
        # 5. 保存结果
        self.save_results(shap_values, X_sample)
        
        print("\n简化版分析完成!")
        print("生成的文件:")
        print(f"  - {self.results_path}/shap_feature_importance.png/pdf (SHAP特征重要性)")
        print(f"  - {self.results_path}/shap_impact_distribution.png/pdf (SHAP影响分布)")
        print(f"  - {self.results_path}/shap_waterfall.png/pdf (瀑布图)")
        print(f"  - {self.results_path}/lightgbm_shap_importance.csv (特征重要性)")
        print(f"  - {self.results_path}/lightgbm_performance.csv (模型性能)")
        
        return self.model

if __name__ == "__main__":
    # 配置路径
    DATA_PATH = "/Users/xiaxin/work/WindForecast_Project/01_Data/processed/imputed_data/changma_imputed_complete.csv"
    RESULTS_PATH = "/Users/xiaxin/work/WindForecast_Project/03_Results/figures/3.1results/lightgbm_shap"
    
    # 创建结果目录
    import os
    os.makedirs(RESULTS_PATH, exist_ok=True)
    
    # 运行简化分析
    analyzer = SimpleLightGBMAnalyzer(DATA_PATH, RESULTS_PATH)
    model = analyzer.run_analysis()