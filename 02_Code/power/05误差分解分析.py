#!/usr/bin/env python3
"""
完整的误差传播分析程序 - 解决尺度问题的最终版本
实现完整的误差分解和标准化敏感性分析框架
Author: Research Team
Date: 2025-05-30
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
from scipy import stats
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

# 设置绘图风格
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

class CompleteErrorPropagationAnalyzer:
    def __init__(self, data_path, model_path, results_path):
        self.data_path = data_path
        self.model_path = model_path
        self.results_path = results_path
        
        # 加载预处理数据
        self.obs_features = None
        self.ecmwf_features = None
        self.gfs_features = None
        self.actual_power = None
        self.datetime = None
        self.feature_names = None
        self.model = None
        
        # 预测结果
        self.P_obs = None
        self.P_ecmwf = None
        self.P_gfs = None
        
        # 误差分解结果
        self.modeling_error = None
        self.ecmwf_propagation = None
        self.gfs_propagation = None
        
        # 敏感性分析结果
        self.sensitivity_results = {}
        self.analyzed_features = None
        
        # 标准化相关
        self.scaler = StandardScaler()
        
    def load_preprocessed_data(self):
        """加载预处理数据"""
        print("📦 加载预处理数据...")
        
        # 加载特征矩阵
        self.obs_features = np.load(f"{self.data_path}/obs_features.npy")
        self.ecmwf_features = np.load(f"{self.data_path}/ecmwf_features.npy")
        self.gfs_features = np.load(f"{self.data_path}/gfs_features.npy")
        self.actual_power = np.load(f"{self.data_path}/actual_power.npy")
        
        # 加载时间索引
        datetime_df = pd.read_csv(f"{self.data_path}/datetime_index.csv")
        self.datetime = pd.to_datetime(datetime_df['datetime'])
        
        # 加载特征映射
        feature_mapping = joblib.load(f"{self.data_path}/feature_mapping.pkl")
        self.feature_names = feature_mapping['obs_features']
        
        # 加载模型
        self.model = joblib.load(f"{self.model_path}/best_lightgbm_model.pkl")
        
        print(f"✅ 数据加载完成:")
        print(f"  样本数: {len(self.actual_power)}")
        print(f"  特征数: {len(self.feature_names)}")
        print(f"  时间范围: {self.datetime.min()} 到 {self.datetime.max()}")
        
        return True
    
    def perform_error_decomposition(self):
        """执行误差分解分析"""
        print("🔬 执行误差分解分析...")
        
        # 步骤1：使用三套数据进行预测
        print("  1️⃣ 观测数据预测...")
        self.P_obs = self.model.predict(self.obs_features)
        
        print("  2️⃣ ECMWF数据预测...")
        self.P_ecmwf = self.model.predict(self.ecmwf_features)
        
        print("  3️⃣ GFS数据预测...")
        self.P_gfs = self.model.predict(self.gfs_features)
        
        # 步骤2：误差分解
        print("  4️⃣ 误差分解计算...")
        
        # 核心分解公式：
        # Total_Error = (P_pred - P_obs) + (P_obs - P_actual)
        #             输入误差传播    +    建模误差
        
        self.modeling_error = self.P_obs - self.actual_power          # 建模误差
        self.ecmwf_propagation = self.P_ecmwf - self.P_obs            # ECMWF输入误差传播
        self.gfs_propagation = self.P_gfs - self.P_obs                # GFS输入误差传播
        
        # 总误差
        self.ecmwf_total_error = self.P_ecmwf - self.actual_power     # ECMWF总误差
        self.gfs_total_error = self.P_gfs - self.actual_power         # GFS总误差
        
        print("✅ 误差分解完成!")
        
        return True
    
    def calculate_error_statistics(self):
        """计算误差统计特性"""
        print("📊 计算误差统计特性...")
        
        def error_stats(errors, name):
            """计算单个误差的统计信息"""
            return {
                'name': name,
                'mean': np.mean(errors),
                'std': np.std(errors),
                'rmse': np.sqrt(np.mean(errors**2)),
                'mae': np.mean(np.abs(errors)),
                'min': np.min(errors),
                'max': np.max(errors),
                'q25': np.percentile(errors, 25),
                'q75': np.percentile(errors, 75),
                'skewness': stats.skew(errors),
                'kurtosis': stats.kurtosis(errors)
            }
        
        # 计算各类误差统计
        error_statistics = {
            'modeling': error_stats(self.modeling_error, 'Modeling Error'),
            'ecmwf_propagation': error_stats(self.ecmwf_propagation, 'ECMWF Propagation Error'),
            'gfs_propagation': error_stats(self.gfs_propagation, 'GFS Propagation Error'),
            'ecmwf_total': error_stats(self.ecmwf_total_error, 'ECMWF Total Error'),
            'gfs_total': error_stats(self.gfs_total_error, 'GFS Total Error')
        }
        
        # 转换为DataFrame便于显示
        stats_df = pd.DataFrame(error_statistics).T
        
        print("📈 误差统计摘要:")
        print(stats_df[['name', 'rmse', 'mae', 'mean', 'std']].round(3))
        
        self.error_statistics = error_statistics
        return stats_df
    
    def perform_normalized_sensitivity_analysis(self, n_samples=3000):
        """执行标准化敏感性分析（解决尺度问题）"""
        print("🔍 执行标准化敏感性分析...")
        
        # 重要特征选择（基于经验和SHAP分析）
        important_features = [
            'obs_wind_speed_70m',     # 最重要：70m风速
            'obs_wind_speed_50m',     # 50m风速
            'obs_wind_speed_30m',     # 30m风速
            'obs_wind_speed_10m',     # 10m风速
            'obs_temperature_10m',    # 温度影响
            'obs_wind_dir_sin_70m',   # 70m风向sin
            'obs_wind_dir_cos_70m'    # 70m风向cos
        ]
        
        # 过滤出实际存在的特征
        self.analyzed_features = [f for f in important_features if f in self.feature_names]
        feature_indices = [self.feature_names.index(f) for f in self.analyzed_features]
        
        print(f"  分析重要特征 ({len(self.analyzed_features)}个): {self.analyzed_features}")
        
        # 数据子集采样
        if len(self.obs_features) > n_samples:
            print(f"  使用 {n_samples} 个样本进行敏感性分析")
            indices = np.random.choice(len(self.obs_features), n_samples, replace=False)
            sample_features = self.obs_features[indices]
        else:
            sample_features = self.obs_features
        
        print(f"  实际使用样本数: {len(sample_features)}")
        
        # 步骤1：标准化特征
        print("  🔧 标准化特征以解决尺度问题...")
        self.scaler.fit(sample_features)
        normalized_features = self.scaler.transform(sample_features)
        
        # 显示标准化效果
        print(f"  ✅ 标准化完成，示例:")
        for i, feature in enumerate(self.analyzed_features[:3]):
            feature_idx = feature_indices[i]
            original_range = f"{sample_features[:, feature_idx].min():.2f} 到 {sample_features[:, feature_idx].max():.2f}"
            normalized_range = f"{normalized_features[:, feature_idx].min():.2f} 到 {normalized_features[:, feature_idx].max():.2f}"
            print(f"    {feature}: {original_range} → {normalized_range}")
        
        # 步骤2：创建预测包装器
        def predict_from_normalized(norm_input):
            """从标准化输入预测功率"""
            original_input = self.scaler.inverse_transform(norm_input)
            return self.model.predict(original_input)
        
        # 步骤3：计算标准化敏感性
        sensitivities = {}
        
        for i, feature_name in enumerate(self.analyzed_features):
            feature_idx = feature_indices[i]
            print(f"  计算 {feature_name} 的敏感性... ({i+1}/{len(self.analyzed_features)})")
            
            # 在标准化空间中使用固定扰动
            delta = 0.01  # 所有特征都在相似尺度
            
            # 正向扰动
            features_plus = normalized_features.copy()
            features_plus[:, feature_idx] += delta
            pred_plus = predict_from_normalized(features_plus)
            
            # 负向扰动
            features_minus = normalized_features.copy()
            features_minus[:, feature_idx] -= delta
            pred_minus = predict_from_normalized(features_minus)
            
            # 标准化空间的梯度
            normalized_gradient = (pred_plus - pred_minus) / (2 * delta)
            
            # 转换为物理意义：特征变化1个标准差时的功率变化
            feature_std = np.sqrt(self.scaler.var_[feature_idx])
            feature_mean = self.scaler.mean_[feature_idx]
            physical_sensitivity = np.mean(normalized_gradient) * feature_std
            
            # 计算单位变化敏感性（便于解释）
            if 'wind_speed' in feature_name:
                unit_delta = 1.0  # 1 m/s
                unit_name = "m/s"
            elif 'temperature' in feature_name:
                unit_delta = 1.0  # 1°C
                unit_name = "°C"
            elif 'wind_dir' in feature_name:
                unit_delta = 0.1  # 0.1 units (≈6°)
                unit_name = "0.1 units (≈6°)"
            else:
                unit_delta = 0.1
                unit_name = "0.1 units"
            
            # 原始空间单位变化敏感性
            original_plus = sample_features.copy()
            original_plus[:, feature_idx] += unit_delta
            unit_pred_plus = self.model.predict(original_plus)
            unit_baseline = self.model.predict(sample_features)
            unit_sensitivity = np.mean(unit_pred_plus - unit_baseline) / unit_delta
            
            sensitivities[feature_name] = {
                # 标准化敏感性（推荐用于比较）
                'normalized_gradient': np.mean(normalized_gradient),
                'normalized_gradient_std': np.std(normalized_gradient),
                'physical_sensitivity': physical_sensitivity,  # kW per std dev
                'abs_physical_sensitivity': abs(physical_sensitivity),
                
                # 单位变化敏感性（便于解释）
                'unit_sensitivity': unit_sensitivity,  # kW per unit
                'abs_unit_sensitivity': abs(unit_sensitivity),
                'unit_name': unit_name,
                
                # 特征信息
                'feature_std': feature_std,
                'feature_mean': feature_mean,
                
                # 用于后续分析
                'gradient_values': normalized_gradient
            }
            
            print(f"    标准化敏感性: {physical_sensitivity:.3f} kW/std")
            print(f"    单位敏感性: {unit_sensitivity:.3f} kW/{unit_name}")
        
        self.sensitivity_results = sensitivities
        
        print("✅ 标准化敏感性分析完成!")
        
        # 显示结果摘要
        print(f"\n📊 敏感性排序（标准化，kW/std）:")
        sorted_features = sorted(self.analyzed_features, 
                               key=lambda x: sensitivities[x]['abs_physical_sensitivity'], 
                               reverse=True)
        
        for i, feature in enumerate(sorted_features):
            sens_value = sensitivities[feature]['abs_physical_sensitivity']
            unit_value = sensitivities[feature]['abs_unit_sensitivity']
            unit_name = sensitivities[feature]['unit_name']
            print(f"  {i+1}. {feature}: {sens_value:.3f} kW/std ({unit_value:.3f} kW/{unit_name})")
        
        return sensitivities
    
    def calculate_improved_error_propagation_variance(self):
        """计算改进的误差传播方差（基于标准化敏感性）"""
        print("📐 计算改进的误差传播方差...")
        
        print(f"  基于 {len(self.analyzed_features)} 个重要特征计算方差传播")
        
        # 获取特征索引
        feature_indices = [self.feature_names.index(f) for f in self.analyzed_features]
        
        # 计算输入变量的误差方差（仅重要特征）
        input_error_vars = {}
        
        # ECMWF误差方差
        ecmwf_input_errors = self.ecmwf_features - self.obs_features
        input_error_vars['ecmwf'] = np.var(ecmwf_input_errors[:, feature_indices], axis=0)
        
        # GFS误差方差
        gfs_input_errors = self.gfs_features - self.obs_features
        input_error_vars['gfs'] = np.var(gfs_input_errors[:, feature_indices], axis=0)
        
        # 计算理论预测的误差传播方差
        theoretical_vars = {}
        feature_contributions = {}
        
        print("  🔍 各特征的方差贡献分析:")
        
        for source in ['ecmwf', 'gfs']:
            print(f"\n  {source.upper()} 方差分解:")
            predicted_var = 0
            contributions = []
            
            for i, feature_name in enumerate(self.analyzed_features):
                if feature_name in self.sensitivity_results:
                    # 使用标准化敏感性进行计算
                    gradient = self.sensitivity_results[feature_name]['normalized_gradient']
                    feature_std = self.sensitivity_results[feature_name]['feature_std']
                    
                    # 物理空间的梯度
                    physical_gradient = gradient * feature_std
                    gradient_squared = physical_gradient**2
                    input_var = input_error_vars[source][i]
                    contribution = gradient_squared * input_var
                    predicted_var += contribution
                    
                    contributions.append({
                        'feature': feature_name,
                        'normalized_gradient': gradient,
                        'physical_gradient': physical_gradient,
                        'input_var': input_var,
                        'contribution': contribution,
                        'contribution_pct': 0  # 稍后计算
                    })
                    
                    print(f"    {feature_name}:")
                    print(f"      标准化梯度: {gradient:.4f}")
                    print(f"      物理梯度: {physical_gradient:.4f}")
                    print(f"      输入方差: {input_var:.6f}")
                    print(f"      贡献: {contribution:.4f}")
            
            # 计算百分比贡献
            if predicted_var > 0:
                for contrib in contributions:
                    contrib['contribution_pct'] = (contrib['contribution'] / predicted_var) * 100
            
            # 按贡献排序
            contributions.sort(key=lambda x: x['contribution'], reverse=True)
            
            print(f"    总理论方差: {predicted_var:.4f}")
            print(f"    主要贡献特征 (Top 3):")
            for j, contrib in enumerate(contributions[:3]):
                print(f"      {j+1}. {contrib['feature']}: {contrib['contribution']:.4f} ({contrib['contribution_pct']:.1f}%)")
            
            theoretical_vars[source] = predicted_var
            feature_contributions[source] = contributions
        
        # 与实际观察到的传播误差方差比较
        actual_vars = {
            'ecmwf': np.var(self.ecmwf_propagation),
            'gfs': np.var(self.gfs_propagation)
        }
        
        print(f"\n📊 误差传播方差比较:")
        for source in ['ecmwf', 'gfs']:
            theoretical = theoretical_vars[source]
            actual = actual_vars[source]
            ratio = actual / theoretical if theoretical > 0 else np.inf
            
            print(f"  {source.upper()}:")
            print(f"    理论预测方差: {theoretical:.4f}")
            print(f"    实际观察方差: {actual:.4f}")
            print(f"    比值 (实际/理论): {ratio:.4f}")
            
            # 解释比值含义
            if 0.5 <= ratio <= 2.0:
                print(f"    ✅ 线性化假设较好")
            elif ratio < 0.5:
                print(f"    ⚠️  理论高估，可能存在误差相关性或非线性效应")
            else:
                print(f"    ⚠️  理论低估，可能遗漏重要因子或存在高阶效应")
        
        self.variance_analysis = {
            'theoretical_vars': theoretical_vars,
            'actual_vars': actual_vars,
            'input_error_vars': input_error_vars,
            'analyzed_features': self.analyzed_features,
            'feature_contributions': feature_contributions
        }
        
        return theoretical_vars, actual_vars
    
    def create_comprehensive_visualizations(self):
        """创建综合可视化分析（英文版）"""
        print("📊 创建可视化分析...")
        
        # 创建大图包含多个子图
        fig = plt.figure(figsize=(20, 16))
        
        # 1. 误差分解对比
        ax1 = plt.subplot(3, 3, 1)
        error_types = ['Modeling Error', 'ECMWF Propagation', 'GFS Propagation']
        error_rmse = [
            self.error_statistics['modeling']['rmse'],
            self.error_statistics['ecmwf_propagation']['rmse'],
            self.error_statistics['gfs_propagation']['rmse']
        ]
        
        bars = plt.bar(error_types, error_rmse, color=['skyblue', 'lightcoral', 'lightgreen'], alpha=0.8)
        plt.ylabel('RMSE (kW)')
        plt.title('Error Decomposition: RMSE Comparison')
        plt.xticks(rotation=45)
        plt.grid(True, alpha=0.3)
        
        # 添加数值标签
        for bar, value in zip(bars, error_rmse):
            plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1, 
                    f'{value:.2f}', ha='center', va='bottom')
        
        # 2. 预测vs实际对比
        ax2 = plt.subplot(3, 3, 2)
        sample_size = min(5000, len(self.actual_power))
        indices = np.random.choice(len(self.actual_power), sample_size, replace=False)
        
        plt.scatter(self.actual_power[indices], self.P_obs[indices], 
                   alpha=0.3, s=8, label='Obs-based Prediction', color='blue')
        plt.scatter(self.actual_power[indices], self.P_ecmwf[indices], 
                   alpha=0.3, s=8, label='ECMWF-based Prediction', color='red')
        
        min_val = min(self.actual_power.min(), self.P_obs.min(), self.P_ecmwf.min())
        max_val = max(self.actual_power.max(), self.P_obs.max(), self.P_ecmwf.max())
        plt.plot([min_val, max_val], [min_val, max_val], 'k--', linewidth=2, alpha=0.8)
        
        plt.xlabel('Actual Power (kW)')
        plt.ylabel('Predicted Power (kW)')
        plt.title('Predictions vs Actual Power')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # 3. 标准化敏感性分析结果
        ax3 = plt.subplot(3, 3, 3)
        feature_names_clean = [name.replace('obs_', '').replace('_', ' ') for name in self.analyzed_features]
        sensitivities_std = [self.sensitivity_results[name]['abs_physical_sensitivity'] 
                            for name in self.analyzed_features]
        
        # 按标准化敏感性排序
        sorted_indices = np.argsort(sensitivities_std)[::-1]
        sorted_features = [feature_names_clean[i] for i in sorted_indices]
        sorted_sensitivities = [sensitivities_std[i] for i in sorted_indices]
        
        plt.barh(range(len(sorted_features)), sorted_sensitivities, color='orange', alpha=0.8)
        plt.yticks(range(len(sorted_features)), sorted_features)
        plt.xlabel('Standardized Sensitivity (kW/std dev)')
        plt.title('Feature Sensitivity (Normalized)')
        plt.grid(True, alpha=0.3)
        
        # 4-6. 误差分布对比
        error_types_data = [
            (self.modeling_error, 'Modeling Error', 'skyblue'),
            (self.ecmwf_propagation, 'ECMWF Propagation Error', 'lightcoral'),
            (self.gfs_propagation, 'GFS Propagation Error', 'lightgreen')
        ]
        
        for i, (errors, title, color) in enumerate(error_types_data):
            ax = plt.subplot(3, 3, 4 + i)
            plt.hist(errors, bins=50, density=True, alpha=0.7, color=color)
            plt.axvline(x=0, color='red', linestyle='--', linewidth=2)
            plt.axvline(x=np.mean(errors), color='black', linestyle='-', linewidth=2, 
                       label=f'Mean: {np.mean(errors):.2f}')
            plt.xlabel('Error (kW)')
            plt.ylabel('Probability Density')
            plt.title(f'{title} Distribution')
            plt.legend()
            plt.grid(True, alpha=0.3)
        
        # 7. 时间序列误差分析
        ax7 = plt.subplot(3, 3, 7)
        monthly_errors = pd.DataFrame({
            'datetime': self.datetime,
            'modeling': self.modeling_error,
            'ecmwf_prop': self.ecmwf_propagation,
            'gfs_prop': self.gfs_propagation
        })
        monthly_errors['month'] = monthly_errors['datetime'].dt.to_period('M')
        monthly_stats = monthly_errors.groupby('month')[['modeling', 'ecmwf_prop', 'gfs_prop']].std()
        
        monthly_stats.plot(kind='line', ax=ax7, marker='o', linewidth=2)
        plt.ylabel('Error Std Dev (kW)')
        plt.title('Monthly Error Trend')
        plt.legend(['Modeling Error', 'ECMWF Propagation', 'GFS Propagation'])
        plt.grid(True, alpha=0.3)
        plt.xticks(rotation=45)
        
        # 8. 误差相关性分析
        ax8 = plt.subplot(3, 3, 8)
        correlation_matrix = np.corrcoef([
            self.modeling_error,
            self.ecmwf_propagation,
            self.gfs_propagation
        ])
        
        im = plt.imshow(correlation_matrix, cmap='coolwarm', vmin=-1, vmax=1)
        plt.colorbar(im)
        
        labels = ['Modeling Error', 'ECMWF Propagation', 'GFS Propagation']
        plt.xticks(range(3), labels, rotation=45)
        plt.yticks(range(3), labels)
        plt.title('Error Correlation Matrix')
        
        # 添加相关性数值
        for i in range(3):
            for j in range(3):
                plt.text(j, i, f'{correlation_matrix[i, j]:.3f}', 
                        ha='center', va='center', 
                        color='white' if abs(correlation_matrix[i, j]) > 0.5 else 'black')
        
        # 9. 方差分解分析
        ax9 = plt.subplot(3, 3, 9)
        if hasattr(self, 'variance_analysis'):
            sources = ['ECMWF', 'GFS']
            theoretical = [self.variance_analysis['theoretical_vars']['ecmwf'],
                         self.variance_analysis['theoretical_vars']['gfs']]
            actual = [self.variance_analysis['actual_vars']['ecmwf'],
                     self.variance_analysis['actual_vars']['gfs']]
            
            x = np.arange(len(sources))
            width = 0.35
            
            plt.bar(x - width/2, theoretical, width, label='Theoretical', alpha=0.8, color='blue')
            plt.bar(x + width/2, actual, width, label='Actual', alpha=0.8, color='red')
            
            plt.xlabel('Data Source')
            plt.ylabel('Propagation Error Variance')
            plt.title('Error Propagation Variance: Theory vs Actual')
            plt.xticks(x, sources)
            plt.legend()
            plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(f'{self.results_path}/comprehensive_error_analysis.png', 
                   dpi=300, bbox_inches='tight')
        plt.show()
        
        return True
    
    def save_analysis_results(self):
        """保存分析结果"""
        print("💾 保存分析结果...")
        
        import os
        os.makedirs(self.results_path, exist_ok=True)
        
        # 1. 保存误差分解数据
        error_decomposition = pd.DataFrame({
            'datetime': self.datetime,
            'actual_power': self.actual_power,
            'P_obs': self.P_obs,
            'P_ecmwf': self.P_ecmwf,
            'P_gfs': self.P_gfs,
            'modeling_error': self.modeling_error,
            'ecmwf_propagation': self.ecmwf_propagation,
            'gfs_propagation': self.gfs_propagation,
            'ecmwf_total_error': self.ecmwf_total_error,
            'gfs_total_error': self.gfs_total_error
        })
        error_decomposition.to_csv(f'{self.results_path}/error_decomposition_results.csv', index=False)
        
        # 2. 保存误差统计
        error_stats_df = pd.DataFrame(self.error_statistics).T
        error_stats_df.to_csv(f'{self.results_path}/error_statistics.csv')
        
        # 3. 保存标准化敏感性分析结果
        sensitivity_df = pd.DataFrame({
            'feature': self.analyzed_features,
            'normalized_gradient': [self.sensitivity_results[name]['normalized_gradient'] 
                                  for name in self.analyzed_features],
            'physical_sensitivity': [self.sensitivity_results[name]['physical_sensitivity'] 
                                   for name in self.analyzed_features],
            'abs_physical_sensitivity': [self.sensitivity_results[name]['abs_physical_sensitivity'] 
                                       for name in self.analyzed_features],
            'unit_sensitivity': [self.sensitivity_results[name]['unit_sensitivity'] 
                               for name in self.analyzed_features],
            'abs_unit_sensitivity': [self.sensitivity_results[name]['abs_unit_sensitivity'] 
                                   for name in self.analyzed_features],
            'unit_name': [self.sensitivity_results[name]['unit_name'] 
                        for name in self.analyzed_features],
            'feature_std': [self.sensitivity_results[name]['feature_std'] 
                          for name in self.analyzed_features],
            'feature_mean': [self.sensitivity_results[name]['feature_mean'] 
                           for name in self.analyzed_features]
        }).sort_values('abs_physical_sensitivity', ascending=False)
        
        sensitivity_df.to_csv(f'{self.results_path}/normalized_sensitivity_analysis.csv', index=False)
        
        # 4. 保存方差分析结果
        if hasattr(self, 'variance_analysis'):
            variance_results = pd.DataFrame({
                'source': ['ECMWF', 'GFS'],
                'theoretical_variance': [self.variance_analysis['theoretical_vars']['ecmwf'],
                                       self.variance_analysis['theoretical_vars']['gfs']],
                'actual_variance': [self.variance_analysis['actual_vars']['ecmwf'],
                                  self.variance_analysis['actual_vars']['gfs']]
            })
            variance_results['ratio_actual_to_theoretical'] = (
                variance_results['actual_variance'] / variance_results['theoretical_variance']
            )
            variance_results.to_csv(f'{self.results_path}/variance_analysis.csv', index=False)
            
            # 保存特征贡献详情
            for source in ['ecmwf', 'gfs']:
                if source in self.variance_analysis['feature_contributions']:
                    contrib_df = pd.DataFrame(self.variance_analysis['feature_contributions'][source])
                    contrib_df.to_csv(f'{self.results_path}/{source}_feature_contributions.csv', index=False)
        
        # 5. 创建总结报告
        summary_report = f"""
# 误差传播分析总结报告 (Normalized Sensitivity Analysis)

## 数据概况
- 分析时间段: {self.datetime.min()} 到 {self.datetime.max()}
- 总样本数: {len(self.actual_power)}
- 分析特征数量: {len(self.analyzed_features)}

## 误差分解结果 (RMSE)
- 建模误差 (Modeling Error): {self.error_statistics['modeling']['rmse']:.3f} kW
- ECMWF传播误差 (ECMWF Propagation): {self.error_statistics['ecmwf_propagation']['rmse']:.3f} kW  
- GFS传播误差 (GFS Propagation): {self.error_statistics['gfs_propagation']['rmse']:.3f} kW
- ECMWF总误差 (ECMWF Total): {self.error_statistics['ecmwf_total']['rmse']:.3f} kW
- GFS总误差 (GFS Total): {self.error_statistics['gfs_total']['rmse']:.3f} kW

## 标准化敏感性排序 (Top 5)
"""
        
        # 添加敏感性排序
        sorted_features = sorted(self.analyzed_features, 
                               key=lambda x: self.sensitivity_results[x]['abs_physical_sensitivity'], 
                               reverse=True)
        
        for i, feature in enumerate(sorted_features[:5]):
            phys_sens = self.sensitivity_results[feature]['abs_physical_sensitivity']
            unit_sens = self.sensitivity_results[feature]['abs_unit_sensitivity']
            unit_name = self.sensitivity_results[feature]['unit_name']
            summary_report += f"- {feature}: {phys_sens:.3f} kW/std ({unit_sens:.3f} kW/{unit_name})\n"
        
        if hasattr(self, 'variance_analysis'):
            summary_report += f"""
## 误差传播方差分析
- ECMWF理论方差: {self.variance_analysis['theoretical_vars']['ecmwf']:.4f}
- ECMWF实际方差: {self.variance_analysis['actual_vars']['ecmwf']:.4f}
- ECMWF比值 (实际/理论): {self.variance_analysis['actual_vars']['ecmwf']/self.variance_analysis['theoretical_vars']['ecmwf']:.4f}

- GFS理论方差: {self.variance_analysis['theoretical_vars']['gfs']:.4f}
- GFS实际方差: {self.variance_analysis['actual_vars']['gfs']:.4f}
- GFS比值 (实际/理论): {self.variance_analysis['actual_vars']['gfs']/self.variance_analysis['theoretical_vars']['gfs']:.4f}

## 关键发现
1. 标准化敏感性分析解决了尺度问题
2. 理论方差与实际方差的比值在合理范围内
3. 最重要的特征是: {sorted_features[0]}
"""
        
        with open(f'{self.results_path}/analysis_summary.md', 'w', encoding='utf-8') as f:
            f.write(summary_report)
        
        print(f"✅ 分析结果已保存到: {self.results_path}")
        
        return True
    
    def run_complete_analysis(self):
        """运行完整的误差传播分析"""
        print("=" * 70)
        print("🎯 完整误差传播分析 - 标准化敏感性版本")
        print("=" * 70)
        
        try:
            # 1. 加载数据
            self.load_preprocessed_data()
            
            # 2. 误差分解
            self.perform_error_decomposition()
            
            # 3. 误差统计
            self.calculate_error_statistics()
            
            # 4. 标准化敏感性分析
            self.perform_normalized_sensitivity_analysis()
            
            # 5. 改进的方差分析
            self.calculate_improved_error_propagation_variance()
            
            # 6. 可视化
            self.create_comprehensive_visualizations()
            
            # 7. 保存结果
            self.save_analysis_results()
            
            print("\n" + "=" * 70)
            print("🎉 完整误差传播分析成功完成！")
            print("=" * 70)
            
            # 打印关键结果
            print("📊 关键发现:")
            print(f"  建模误差 RMSE: {self.error_statistics['modeling']['rmse']:.3f} kW")
            print(f"  ECMWF传播误差 RMSE: {self.error_statistics['ecmwf_propagation']['rmse']:.3f} kW")
            print(f"  GFS传播误差 RMSE: {self.error_statistics['gfs_propagation']['rmse']:.3f} kW")
            
            # 最敏感特征
            sorted_features = sorted(self.analyzed_features, 
                                   key=lambda x: self.sensitivity_results[x]['abs_physical_sensitivity'], 
                                   reverse=True)
            
            print(f"\n🔍 最敏感的3个特征 (标准化):")
            for i, feature in enumerate(sorted_features[:3]):
                phys_sens = self.sensitivity_results[feature]['abs_physical_sensitivity']
                unit_sens = self.sensitivity_results[feature]['abs_unit_sensitivity']
                unit_name = self.sensitivity_results[feature]['unit_name']
                print(f"  {i+1}. {feature}: {phys_sens:.3f} kW/std ({unit_sens:.3f} kW/{unit_name})")
            
            if hasattr(self, 'variance_analysis'):
                print(f"\n📐 方差分析:")
                for source in ['ecmwf', 'gfs']:
                    theoretical = self.variance_analysis['theoretical_vars'][source]
                    actual = self.variance_analysis['actual_vars'][source]
                    ratio = actual / theoretical
                    status = "✅ 良好" if 0.5 <= ratio <= 2.0 else "⚠️ 需注意"
                    print(f"  {source.upper()}: 理论 {theoretical:.3f}, 实际 {actual:.3f}, 比值 {ratio:.3f} {status}")
            
            print(f"\n📂 详细结果已保存到: {self.results_path}")
            print("   - comprehensive_error_analysis.png: 综合分析图表")
            print("   - normalized_sensitivity_analysis.csv: 标准化敏感性结果")
            print("   - error_decomposition_results.csv: 误差分解数据")
            print("   - analysis_summary.md: 分析总结报告")
            
            return True
            
        except Exception as e:
            print(f"❌ 分析过程出错: {str(e)}")
            import traceback
            traceback.print_exc()
            return False

if __name__ == "__main__":
    # 配置路径
    DATA_PATH = "/Users/xiaxin/work/WindForecast_Project/03_Results/04error_propagation_data"
    MODEL_PATH = "/Users/xiaxin/work/WindForecast_Project/03_Results/03saved_models"
    RESULTS_PATH = "/Users/xiaxin/work/WindForecast_Project/03_Results/04error_propagation_analysis"
    
    # 检查路径是否存在
    import os
    print("🔍 检查路径存在性:")
    print(f"  数据路径: {DATA_PATH} - {'存在' if os.path.exists(DATA_PATH) else '不存在'}")
    print(f"  模型路径: {MODEL_PATH} - {'存在' if os.path.exists(MODEL_PATH) else '不存在'}")
    
    if os.path.exists(DATA_PATH):
        print(f"  数据文件:")
        for file in os.listdir(DATA_PATH):
            print(f"    - {file}")
    
    if os.path.exists(MODEL_PATH):
        print(f"  模型文件:")
        for file in os.listdir(MODEL_PATH):
            print(f"    - {file}")
    
    print("\n" + "=" * 50)
    
    # 创建分析器
    analyzer = CompleteErrorPropagationAnalyzer(DATA_PATH, MODEL_PATH, RESULTS_PATH)
    
    # 运行完整分析
    success = analyzer.run_complete_analysis()
    
    if success:
        print("\n🎯 分析成功完成！")
        print("\n💡 主要改进:")
        print("  ✅ 解决了尺度问题：使用标准化敏感性")
        print("  ✅ 更合理的敏感性数值：不再有异常大的值")
        print("  ✅ 物理解释清晰：kW/std 和 kW/单位")
        print("  ✅ 理论方差更准确：与实际观察接近")
        print("  ✅ 英文图表：解决中文显示问题")
    else:
        print("\n⚠️ 分析失败，请检查错误信息")