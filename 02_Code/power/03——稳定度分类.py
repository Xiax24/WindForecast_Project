#!/usr/bin/env python3
"""
基于大气稳定度分类的风电预测与SHAP重要性分析
Author: Research Team
Date: 2025-06-09
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import lightgbm as lgb
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import shap
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# 设置中文显示
plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial Unicode MS', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

class StabilityBasedWindPredictionAnalyzer:
    def __init__(self, wind_data_path, stability_data_path, save_path):
        self.wind_data_path = wind_data_path
        self.stability_data_path = stability_data_path
        self.save_path = save_path
        self.wind_data = None
        self.stability_data = None
        self.merged_data = None
        self.stability_groups = {}
        self.feature_names = None
        self.models = {}
        self.shap_explainers = {}
        self.results = {}
        
    def load_data(self):
        """加载风电数据和稳定度数据"""
        print("📊 加载数据...")
        
        # 加载风电数据
        self.wind_data = pd.read_csv(self.wind_data_path)
        print(f"风电数据形状: {self.wind_data.shape}")
        
        # 加载稳定度数据
        self.stability_data = pd.read_csv(self.stability_data_path)
        print(f"稳定度数据形状: {self.stability_data.shape}")
        
        # 转换时间列
        if 'datetime' in self.wind_data.columns:
            self.wind_data['datetime'] = pd.to_datetime(self.wind_data['datetime'])
        
        if 'timestamp' in self.stability_data.columns:
            self.stability_data['timestamp'] = pd.to_datetime(self.stability_data['timestamp'])
            # 重命名为datetime以便合并
            self.stability_data = self.stability_data.rename(columns={'timestamp': 'datetime'})
        
        return self.wind_data, self.stability_data
    
    def analyze_stability_distribution(self):
        """分析稳定度分布"""
        print("🌀 分析稳定度分布...")
        
        # 稳定度分布统计
        stability_counts = self.stability_data['stability_final'].value_counts()
        confidence_stats = self.stability_data.groupby('stability_final')['confidence_final'].agg(['mean', 'std', 'count'])
        
        print("\n稳定度分布:")
        for stability, count in stability_counts.items():
            percentage = count / len(self.stability_data) * 100
            print(f"  {stability}: {count} 条 ({percentage:.1f}%)")
        
        # 绘制分布图
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('大气稳定度分布分析', fontsize=16, fontweight='bold')
        
        # 1. 稳定度频次分布
        ax1 = axes[0, 0]
        stability_counts.plot(kind='bar', ax=ax1, color='skyblue', alpha=0.7)
        ax1.set_title('稳定度类型分布')
        ax1.set_xlabel('稳定度类型')
        ax1.set_ylabel('频次')
        ax1.tick_params(axis='x', rotation=45)
        
        # 添加百分比标签
        total = len(self.stability_data)
        for i, (stability, count) in enumerate(stability_counts.items()):
            percentage = count / total * 100
            ax1.text(i, count + total*0.01, f'{percentage:.1f}%', 
                    ha='center', va='bottom', fontweight='bold')
        
        # 2. 置信度分布
        ax2 = axes[0, 1]
        self.stability_data.boxplot(column='confidence_final', by='stability_final', ax=ax2)
        ax2.set_title('各稳定度类型的置信度分布')
        ax2.set_xlabel('稳定度类型')
        ax2.set_ylabel('置信度')
        
        # 3. 时间序列分布
        ax3 = axes[1, 0]
        # 按小时统计稳定度分布
        hourly_stability = self.stability_data.groupby(['hour', 'stability_final']).size().unstack(fill_value=0)
        hourly_stability_pct = hourly_stability.div(hourly_stability.sum(axis=1), axis=0) * 100
        
        hourly_stability_pct.plot(kind='area', stacked=True, ax=ax3, alpha=0.7)
        ax3.set_title('稳定度的日变化模式')
        ax3.set_xlabel('小时')
        ax3.set_ylabel('百分比 (%)')
        ax3.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        
        # 4. 季节性分布
        ax4 = axes[1, 1]
        seasonal_stability = self.stability_data.groupby(['season', 'stability_final']).size().unstack(fill_value=0)
        seasonal_stability_pct = seasonal_stability.div(seasonal_stability.sum(axis=1), axis=0) * 100
        
        seasonal_stability_pct.plot(kind='bar', stacked=True, ax=ax4, alpha=0.7)
        ax4.set_title('稳定度的季节变化')
        ax4.set_xlabel('季节')
        ax4.set_ylabel('百分比 (%)')
        ax4.tick_params(axis='x', rotation=0)
        ax4.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        
        plt.tight_layout()
        plt.savefig(f"{self.save_path}/stability_distribution_analysis.png", dpi=300, bbox_inches='tight')
        plt.show()
        
        return stability_counts, confidence_stats
    
    def merge_data_by_time(self):
        """按时间合并风电数据和稳定度数据"""
        print("🔗 合并风电数据和稳定度数据...")
        
        # 选择风电观测数据列
        obs_columns = [col for col in self.wind_data.columns if col.startswith('obs_')]
        obs_columns += ['datetime', 'power']
        
        # 移除密度和湿度
        obs_columns = [col for col in obs_columns if 'density' not in col and 'humidity' not in col]
        wind_data_clean = self.wind_data[obs_columns].copy()
        
        # 选择关键稳定度信息
        stability_columns = ['datetime', 'stability_final', 'confidence_final', 'alpha_main', 
                           'data_quality', 'is_daytime', 'temp_change_rate']
        stability_data_clean = self.stability_data[stability_columns].copy()
        
        # 按时间合并
        self.merged_data = pd.merge(wind_data_clean, stability_data_clean, on='datetime', how='inner')
        
        print(f"合并前风电数据: {len(wind_data_clean)} 条")
        print(f"合并前稳定度数据: {len(stability_data_clean)} 条")
        print(f"合并后数据: {len(self.merged_data)} 条")
        
        # 清理数据
        initial_shape = len(self.merged_data)
        self.merged_data = self.merged_data.dropna()
        self.merged_data = self.merged_data[self.merged_data['power'] >= 0]
        
        # 只保留高质量和置信度数据
        quality_mask = (self.merged_data['data_quality'].isin(['high', 'medium'])) & \
                      (self.merged_data['confidence_final'] >= 0.6)
        self.merged_data = self.merged_data[quality_mask]
        
        final_shape = len(self.merged_data)
        print(f"数据清理: 从 {initial_shape} 条减少到 {final_shape} 条")
        
        # 分析合并后的稳定度分布
        merged_stability_counts = self.merged_data['stability_final'].value_counts()
        print(f"\n合并后稳定度分布:")
        for stability, count in merged_stability_counts.items():
            percentage = count / len(self.merged_data) * 100
            print(f"  {stability}: {count} 条 ({percentage:.1f}%)")
        
        return self.merged_data
    
    def process_wind_direction(self, data):
        """处理风向变量为sin/cos分量"""
        data = data.copy()
        wind_dir_cols = [col for col in data.columns if 'wind_direction' in col]
        
        if wind_dir_cols:
            print(f"处理 {len(wind_dir_cols)} 个风向列...")
            
            for col in wind_dir_cols:
                # 气象角度转换为数学角度
                math_angle = (90 - data[col] + 360) % 360
                wind_dir_rad = np.deg2rad(math_angle)
                
                # 创建sin/cos分量
                sin_col = col.replace('wind_direction', 'wind_dir_sin')
                cos_col = col.replace('wind_direction', 'wind_dir_cos')
                
                data[sin_col] = np.sin(wind_dir_rad)
                data[cos_col] = np.cos(wind_dir_rad)
            
            # 移除原始风向列
            data = data.drop(columns=wind_dir_cols)
        
        return data
    
    def prepare_stability_groups(self, min_samples=500):
        """按稳定度分组数据，确保每组有足够样本"""
        print("📊 按稳定度分组数据...")
        
        stability_counts = self.merged_data['stability_final'].value_counts()
        
        # 只选择样本数足够的稳定度类型
        valid_stabilities = stability_counts[stability_counts >= min_samples].index.tolist()
        
        print(f"样本数足够的稳定度类型 (>={min_samples}): {valid_stabilities}")
        
        for stability in valid_stabilities:
            stability_data = self.merged_data[self.merged_data['stability_final'] == stability].copy()
            self.stability_groups[stability] = stability_data
            print(f"  {stability}: {len(stability_data)} 条样本")
        
        # 如果某些稳定度样本太少，可以考虑合并相似类型
        small_stabilities = stability_counts[stability_counts < min_samples].index.tolist()
        if small_stabilities:
            print(f"\n样本数不足的稳定度类型: {small_stabilities}")
            print("建议：可以考虑将相似稳定度类型合并或调整min_samples参数")
        
        return self.stability_groups
    
    def train_stability_models(self):
        """为每种稳定度训练独立的预测模型"""
        print("🚀 训练稳定度分类模型...")
        
        # LightGBM基础参数
        base_params = {
            'objective': 'regression',
            'metric': 'rmse',
            'boosting_type': 'gbdt',
            'num_leaves': 31,
            'learning_rate': 0.1,
            'n_estimators': 200,
            'reg_alpha': 0.1,
            'reg_lambda': 0.1,
            'min_child_samples': 20,
            'random_state': 42,
            'verbose': -1,
            'n_jobs': -1
        }
        
        for stability, data in self.stability_groups.items():
            print(f"\n训练 {stability} 稳定度模型...")
            print(f"数据量: {len(data)} 条")
            
            # 处理风向和准备特征
            data_processed = self.process_wind_direction(data)
            
            # 选择特征列
            exclude_cols = ['datetime', 'power', 'stability_final', 'confidence_final', 
                          'data_quality', 'is_daytime']
            feature_cols = [col for col in data_processed.columns if col not in exclude_cols]
            
            # 创建特征矩阵
            X = data_processed[feature_cols].values
            y = data_processed['power'].values
            
            # 保存特征名称（所有稳定度使用相同特征）
            if self.feature_names is None:
                self.feature_names = feature_cols
                print(f"  设置特征名称，共 {len(feature_cols)} 个特征")
            
            print(f"  特征数量: {len(feature_cols)}")
            print(f"  功率范围: {y.min():.1f} - {y.max():.1f} MW")
            print(f"  功率均值: {y.mean():.1f} MW")
            
            # 数据分割
            if len(data) >= 100:  # 确保有足够数据进行分割
                X_train, X_test, y_train, y_test = train_test_split(
                    X, y, test_size=0.2, random_state=42
                )
                
                # 训练模型
                model = lgb.LGBMRegressor(**base_params)
                model.fit(X_train, y_train)
                
                # 预测和评估
                y_pred_train = model.predict(X_train)
                y_pred_test = model.predict(X_test)
                
                train_r2 = r2_score(y_train, y_pred_train)
                test_r2 = r2_score(y_test, y_pred_test)
                train_rmse = np.sqrt(mean_squared_error(y_train, y_pred_train))
                test_rmse = np.sqrt(mean_squared_error(y_test, y_pred_test))
                
                # 保存模型和结果
                self.models[stability] = model
                self.results[stability] = {
                    'r2_train': train_r2,
                    'r2_test': test_r2,
                    'rmse_train': train_rmse,
                    'rmse_test': test_rmse,
                    'X_train': X_train,
                    'X_test': X_test,
                    'y_train': y_train,
                    'y_test': y_test,
                    'y_pred_train': y_pred_train,
                    'y_pred_test': y_pred_test,
                    'sample_count': len(data),
                    'power_mean': y.mean(),
                    'power_std': y.std()
                }
                
                print(f"  ✓ 训练完成 - R²: {test_r2:.4f}, RMSE: {test_rmse:.2f} MW")
                print(f"    过拟合检查: 训练R²={train_r2:.4f}, 测试R²={test_r2:.4f}, 差值={train_r2-test_r2:.4f}")
                
            else:
                print(f"  ⚠️ 样本数不足 ({len(data)} < 100)，跳过训练")
        
        print(f"\n✓ 共训练了 {len(self.models)} 个稳定度模型")
        return self.models
    
    def calculate_stability_shap_values(self, n_samples=800):
        """计算各稳定度模型的SHAP值"""
        print("📊 计算稳定度SHAP重要性...")
        
        for stability in self.models.keys():
            print(f"计算 {stability} 的SHAP值...")
            
            # 获取测试数据
            X_test = self.results[stability]['X_test']
            
            # 限制样本数量
            if len(X_test) > n_samples:
                indices = np.random.choice(len(X_test), n_samples, replace=False)
                X_sample = X_test[indices]
            else:
                X_sample = X_test
            
            # 计算SHAP值
            explainer = shap.TreeExplainer(self.models[stability])
            shap_values = explainer.shap_values(X_sample)
            
            # 保存结果
            self.shap_explainers[stability] = explainer
            self.results[stability]['shap_values'] = shap_values
            self.results[stability]['X_shap'] = X_sample
            
            print(f"  ✓ 完成 (样本数: {len(X_sample)})")
    
    def plot_stability_performance_comparison(self):
        """绘制不同稳定度模型的性能对比"""
        print("📈 绘制稳定度模型性能对比...")
        
        if not self.results:
            print("⚠️ 没有训练结果，跳过性能对比")
            return
        
        # 准备数据
        stabilities = list(self.results.keys())
        r2_values = [self.results[s]['r2_test'] for s in stabilities]
        rmse_values = [self.results[s]['rmse_test'] for s in stabilities]
        sample_counts = [self.results[s]['sample_count'] for s in stabilities]
        power_means = [self.results[s]['power_mean'] for s in stabilities]
        
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle('不同稳定度条件下的模型性能对比', fontsize=16, fontweight='bold')
        
        # 1. R² 性能对比
        ax1 = axes[0, 0]
        bars1 = ax1.bar(stabilities, r2_values, color='skyblue', alpha=0.7)
        ax1.set_ylabel('R² Score')
        ax1.set_title('R² 性能对比')
        ax1.set_ylim(0, 1)
        ax1.tick_params(axis='x', rotation=45)
        
        # 添加数值标签
        for bar, r2 in zip(bars1, r2_values):
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                    f'{r2:.3f}', ha='center', va='bottom', fontweight='bold')
        
        # 2. RMSE 对比
        ax2 = axes[0, 1]
        bars2 = ax2.bar(stabilities, rmse_values, color='lightcoral', alpha=0.7)
        ax2.set_ylabel('RMSE (MW)')
        ax2.set_title('RMSE 对比')
        ax2.tick_params(axis='x', rotation=45)
        
        for bar, rmse in zip(bars2, rmse_values):
            height = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width()/2., height + height*0.01,
                    f'{rmse:.1f}', ha='center', va='bottom', fontweight='bold')
        
        # 3. 样本数量对比
        ax3 = axes[0, 2]
        bars3 = ax3.bar(stabilities, sample_counts, color='lightgreen', alpha=0.7)
        ax3.set_ylabel('样本数量')
        ax3.set_title('各稳定度样本分布')
        ax3.tick_params(axis='x', rotation=45)
        
        for bar, count in zip(bars3, sample_counts):
            height = bar.get_height()
            ax3.text(bar.get_x() + bar.get_width()/2., height + height*0.01,
                    f'{count}', ha='center', va='bottom', fontweight='bold')
        
        # 4. 功率均值对比
        ax4 = axes[1, 0]
        bars4 = ax4.bar(stabilities, power_means, color='gold', alpha=0.7)
        ax4.set_ylabel('平均功率 (MW)')
        ax4.set_title('各稳定度平均功率')
        ax4.tick_params(axis='x', rotation=45)
        
        for bar, power in zip(bars4, power_means):
            height = bar.get_height()
            ax4.text(bar.get_x() + bar.get_width()/2., height + height*0.01,
                    f'{power:.1f}', ha='center', va='bottom', fontweight='bold')
        
        # 5. 预测效果散点图（选择最好的模型）
        best_stability = stabilities[np.argmax(r2_values)]
        ax5 = axes[1, 1]
        
        y_test = self.results[best_stability]['y_test']
        y_pred = self.results[best_stability]['y_pred_test']
        
        ax5.scatter(y_test, y_pred, alpha=0.5, s=20)
        min_val = min(y_test.min(), y_pred.min())
        max_val = max(y_test.max(), y_pred.max())
        ax5.plot([min_val, max_val], [min_val, max_val], 'r--', alpha=0.8)
        
        ax5.set_xlabel('实际功率 (MW)')
        ax5.set_ylabel('预测功率 (MW)')
        ax5.set_title(f'最佳模型预测效果 ({best_stability})')
        ax5.grid(True, alpha=0.3)
        
        r2_best = self.results[best_stability]['r2_test']
        ax5.text(0.05, 0.95, f'R² = {r2_best:.3f}', transform=ax5.transAxes,
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        # 6. 性能-样本量关系
        ax6 = axes[1, 2]
        scatter = ax6.scatter(sample_counts, r2_values, c=rmse_values, 
                            cmap='viridis_r', s=100, alpha=0.7)
        
        # 添加稳定度标签
        for i, stability in enumerate(stabilities):
            ax6.annotate(stability, (sample_counts[i], r2_values[i]),
                        xytext=(5, 5), textcoords='offset points', fontsize=8)
        
        ax6.set_xlabel('样本数量')
        ax6.set_ylabel('R² Score')
        ax6.set_title('性能与样本量关系')
        ax6.grid(True, alpha=0.3)
        
        # 添加颜色条
        cbar = plt.colorbar(scatter, ax=ax6)
        cbar.set_label('RMSE (MW)')
        
        plt.tight_layout()
        plt.savefig(f"{self.save_path}/stability_performance_comparison.png", dpi=300, bbox_inches='tight')
        plt.show()
        
        # 输出性能排名
        print("\n📊 性能排名 (按R²降序):")
        performance_df = pd.DataFrame({
            'stability': stabilities,
            'r2': r2_values,
            'rmse': rmse_values,
            'samples': sample_counts,
            'power_mean': power_means
        }).sort_values('r2', ascending=False)
        
        for i, row in performance_df.iterrows():
            print(f"  {i+1}. {row['stability']}: R²={row['r2']:.3f}, "
                  f"RMSE={row['rmse']:.1f}MW, 样本={row['samples']}")
        
        return performance_df
    
    def plot_stability_shap_comparison(self):
        """绘制不同稳定度的SHAP重要性对比"""
        print("📊 绘制稳定度SHAP重要性对比...")
        
        if not self.results or not any('shap_values' in result for result in self.results.values()):
            print("⚠️ 没有SHAP结果，跳过SHAP对比")
            return
        
        # 计算各稳定度的平均SHAP重要性
        shap_importance_df = pd.DataFrame({'feature': self.feature_names})
        
        for stability in self.results.keys():
            if 'shap_values' in self.results[stability]:
                shap_values = self.results[stability]['shap_values']
                importance = np.abs(shap_values).mean(axis=0)
                shap_importance_df[f'{stability}_importance'] = importance
        
        # 计算总体重要性排序
        importance_cols = [col for col in shap_importance_df.columns if 'importance' in col]
        shap_importance_df['avg_importance'] = shap_importance_df[importance_cols].mean(axis=1)
        shap_importance_df = shap_importance_df.sort_values('avg_importance', ascending=False)
        
        # 绘制对比图
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle('不同稳定度条件下的SHAP重要性对比', fontsize=16, fontweight='bold')
        
        # 1. Top特征重要性对比
        top_n = 15
        top_features = shap_importance_df.head(top_n)
        
        ax1 = axes[0, 0]
        stabilities = [col.replace('_importance', '') for col in importance_cols]
        x = np.arange(len(top_features))
        width = 0.8 / len(stabilities)
        
        for i, stability in enumerate(stabilities):
            col = f'{stability}_importance'
            offset = (i - len(stabilities)/2 + 0.5) * width
            ax1.barh(x + offset, top_features[col], width, 
                    label=stability, alpha=0.7)
        
        ax1.set_yticks(x)
        ax1.set_yticklabels(top_features['feature'], fontsize=8)
        ax1.set_xlabel('SHAP重要性')
        ax1.set_title(f'Top {top_n} 特征重要性对比')
        ax1.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        ax1.grid(True, alpha=0.3)
        
        # 2. 特征重要性热力图
        ax2 = axes[0, 1]
        # 选择top特征绘制热力图
        top_20_features = shap_importance_df.head(20)
        heatmap_data = top_20_features[importance_cols].T
        heatmap_data.columns = top_20_features['feature']
        
        sns.heatmap(heatmap_data, annot=True, fmt='.3f', cmap='YlOrRd', 
                   ax=ax2, cbar_kws={'label': 'SHAP重要性'})
        ax2.set_title('Top 20 特征重要性热力图')
        ax2.set_xlabel('特征')
        ax2.set_ylabel('稳定度类型')
        
        # 3. 按变量类型分组的重要性
        ax3 = axes[1, 0]
        
        feature_categories = {
            'wind_speed': [f for f in self.feature_names if 'wind_speed' in f],
            'wind_direction': [f for f in self.feature_names if 'wind_dir' in f],  
            'temperature': [f for f in self.feature_names if 'temperature' in f],
            'alpha': [f for f in self.feature_names if 'alpha' in f],
            'other': [f for f in self.feature_names if not any(keyword in f for keyword in 
                     ['wind_speed', 'wind_dir', 'temperature', 'alpha'])]
        }
        
        category_importance = pd.DataFrame()
        for category, features in feature_categories.items():
            if features:
                cat_data = {}
                cat_data['category'] = category
                for stability in stabilities:
                    col = f'{stability}_importance'
                    if col in shap_importance_df.columns:
                        cat_features = shap_importance_df[shap_importance_df['feature'].isin(features)]
                        cat_data[stability] = cat_features[col].sum()
                    else:
                        cat_data[stability] = 0
                category_importance = pd.concat([category_importance, pd.DataFrame([cat_data])], ignore_index=True)
        
        # 绘制分类重要性对比
        x = np.arange(len(category_importance))
        width = 0.8 / len(stabilities)
        
        for i, stability in enumerate(stabilities):
            offset = (i - len(stabilities)/2 + 0.5) * width
            if stability in category_importance.columns:
                ax3.bar(x + offset, category_importance[stability], width, 
                       label=stability, alpha=0.7)
        
        ax3.set_xticks(x)
        ax3.set_xticklabels(category_importance['category'])
        ax3.set_ylabel('累计SHAP重要性')
        ax3.set_title('按变量类型分组的重要性对比')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        
        # 4. 稳定度间重要性差异分析
        ax4 = axes[1, 1]
        
        if len(stabilities) >= 2:
            # 计算不同稳定度间的重要性差异
            stability1, stability2 = stabilities[0], stabilities[1]
            col1, col2 = f'{stability1}_importance', f'{stability2}_importance'
            
            if col1 in shap_importance_df.columns and col2 in shap_importance_df.columns:
                diff = shap_importance_df[col1] - shap_importance_df[col2]
                shap_importance_df['diff'] = diff
                diff_sorted = shap_importance_df.sort_values('diff', ascending=True)
                
                colors = ['red' if x < 0 else 'blue' for x in diff_sorted['diff']]
                ax4.barh(range(len(diff_sorted)), diff_sorted['diff'], color=colors, alpha=0.6)
                ax4.set_yticks(range(len(diff_sorted)))
                ax4.set_yticklabels(diff_sorted['feature'], fontsize=6)
                ax4.set_xlabel(f'重要性差异 ({stability1} - {stability2})')
                ax4.set_title(f'{stability1} vs {stability2} 重要性差异')
                ax4.axvline(x=0, color='black', linestyle='--', alpha=0.5)
                ax4.grid(True, alpha=0.3)
            else:
                ax4.text(0.5, 0.5, '数据不足\n无法进行差异分析', 
                        ha='center', va='center', transform=ax4.transAxes)
                ax4.set_title('重要性差异分析')
        else:
            ax4.text(0.5, 0.5, '需要至少2种稳定度\n进行差异分析', 
                    ha='center', va='center', transform=ax4.transAxes)
            ax4.set_title('重要性差异分析')
        
        plt.tight_layout()
        plt.savefig(f"{self.save_path}/stability_shap_comparison.png", dpi=300, bbox_inches='tight')
        plt.show()
        
        # 保存重要性对比数据
        shap_importance_df.to_csv(f"{self.save_path}/stability_shap_importance.csv", index=False)
        print("✓ SHAP重要性对比数据已保存")
        
        return shap_importance_df
    
    def analyze_stability_power_characteristics(self):
        """分析不同稳定度下的功率特征"""
        print("⚡ 分析不同稳定度的功率特征...")
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle('不同稳定度条件下的功率特征分析', fontsize=16, fontweight='bold')
        
        stabilities = list(self.stability_groups.keys())
        
        # 1. 功率分布对比
        ax1 = axes[0, 0]
        for stability in stabilities:
            power_data = self.stability_groups[stability]['power']
            ax1.hist(power_data, bins=30, alpha=0.6, label=f'{stability} (μ={power_data.mean():.1f})', 
                    density=True)
        
        ax1.set_xlabel('功率 (MW)')
        ax1.set_ylabel('密度')
        ax1.set_title('功率分布对比')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # 2. 功率统计箱线图
        ax2 = axes[0, 1]
        power_data_list = []
        labels = []
        for stability in stabilities:
            power_data_list.append(self.stability_groups[stability]['power'])
            labels.append(stability)
        
        ax2.boxplot(power_data_list, labels=labels)
        ax2.set_ylabel('功率 (MW)')
        ax2.set_title('功率统计分布')
        ax2.tick_params(axis='x', rotation=45)
        ax2.grid(True, alpha=0.3)
        
        # 3. 功率与风速关系 - 先找到风速列
        ax3 = axes[1, 0]
        # 从第一个稳定度组中找风速列
        sample_data = self.stability_groups[stabilities[0]]
        wind_speed_cols = [col for col in sample_data.columns if 'wind_speed' in col and col.startswith('obs_')]
        
        if wind_speed_cols:
            main_wind_col = wind_speed_cols[0]  # 使用第一个风速列
            print(f"  使用风速列: {main_wind_col}")
            
            for stability in stabilities:
                data = self.stability_groups[stability]
                if main_wind_col in data.columns:
                    ax3.scatter(data[main_wind_col], data['power'], 
                              alpha=0.5, s=10, label=stability)
            
            ax3.set_xlabel(f'{main_wind_col} (m/s)')
            ax3.set_ylabel('功率 (MW)')
            ax3.set_title('功率-风速关系')
            ax3.legend()
            ax3.grid(True, alpha=0.3)
        else:
            ax3.text(0.5, 0.5, '未找到风速数据', ha='center', va='center', transform=ax3.transAxes)
            ax3.set_title('功率-风速关系')
        
        # 4. 稳定度功率统计表
        ax4 = axes[1, 1]
        ax4.axis('tight')
        ax4.axis('off')
        
        # 创建统计表
        stats_data = []
        for stability in stabilities:
            power = self.stability_groups[stability]['power']
            stats_data.append([
                stability,
                f"{len(power)}",
                f"{power.mean():.1f}",
                f"{power.std():.1f}",
                f"{power.min():.1f}",
                f"{power.max():.1f}",
                f"{power.quantile(0.5):.1f}"
            ])
        
        table = ax4.table(cellText=stats_data,
                         colLabels=['稳定度', '样本数', '均值', '标准差', '最小值', '最大值', '中位数'],
                         cellLoc='center',
                         loc='center')
        table.auto_set_font_size(False)
        table.set_fontsize(9)
        table.scale(1.2, 1.5)
        ax4.set_title('功率统计摘要', pad=20)
        
        plt.tight_layout()
        plt.savefig(f"{self.save_path}/stability_power_characteristics.png", dpi=300, bbox_inches='tight')
        plt.show()
        
        return stats_data
    
    def save_stability_models_and_results(self):
        """保存稳定度模型和结果"""
        print("💾 保存稳定度模型和结果...")
        
        # 保存各稳定度模型
        for stability, model in self.models.items():
            model_path = f"{self.save_path}/lightgbm_model_{stability}.pkl"
            joblib.dump(model, model_path)
            print(f"✓ {stability}模型已保存: {model_path}")
        
        # 保存特征名称
        feature_names_path = f"{self.save_path}/feature_names.pkl"
        joblib.dump(self.feature_names, feature_names_path)
        print(f"✓ 特征名称已保存: {feature_names_path}")
        
        # 保存结果摘要
        results_summary = {}
        for stability in self.results:
            results_summary[stability] = {
                'r2_test': self.results[stability]['r2_test'],
                'rmse_test': self.results[stability]['rmse_test'],
                'sample_count': self.results[stability]['sample_count'],
                'power_mean': self.results[stability]['power_mean'],
                'power_std': self.results[stability]['power_std']
            }
        
        summary_path = f"{self.save_path}/stability_results_summary.pkl"
        joblib.dump(results_summary, summary_path)
        print(f"✓ 结果摘要已保存: {summary_path}")
        
        # 保存稳定度分组数据信息
        stability_info = {}
        for stability, data in self.stability_groups.items():
            stability_info[stability] = {
                'sample_count': len(data),
                'power_range': [data['power'].min(), data['power'].max()],
                'power_mean': data['power'].mean(),
                'confidence_mean': data['confidence_final'].mean() if 'confidence_final' in data.columns else None
            }
        
        info_path = f"{self.save_path}/stability_groups_info.pkl"
        joblib.dump(stability_info, info_path)
        print(f"✓ 稳定度分组信息已保存: {info_path}")
        
        return results_summary, stability_info
    
    def create_stability_prediction_function(self):
        """创建基于稳定度的预测函数示例"""
        print("📝 创建稳定度预测函数示例...")
        
        example_code = f'''
# ===== 基于稳定度的风电预测使用示例 =====
import joblib
import numpy as np
import pandas as pd

# 1. 加载所有稳定度模型
models = {{}}
stabilities = {list(self.models.keys())}

for stability in stabilities:
    model_path = "{self.save_path}/lightgbm_model_{{stability}}.pkl"
    models[stability] = joblib.load(model_path)

# 加载特征名称和结果信息
feature_names = joblib.load("{self.save_path}/feature_names.pkl")
results_summary = joblib.load("{self.save_path}/stability_results_summary.pkl")

print("已加载的稳定度模型:", list(models.keys()))
print("特征数量:", len(feature_names))

# 2. 统一的稳定度预测函数
def predict_by_stability(input_data, stability_labels, feature_names):
    \"\"\"
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
    \"\"\"
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
                print(f"使用 {{stability}} 模型预测了 {{np.sum(mask)}} 条数据")
        else:
            print(f"警告: 没有找到 {{stability}} 稳定度的模型")
    
    return predictions, used_models

# 3. 处理新数据的完整流程示例
def process_new_data_with_stability(wind_data_path, stability_data_path):
    \"\"\"
    处理新数据并进行稳定度分类预测的完整流程
    \"\"\"
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
print("\\n各稳定度模型性能:")
for stability, summary in results_summary.items():
    print(f"  {{stability}}: R²={{summary['r2_test']:.3f}}, "
          f"RMSE={{summary['rmse_test']:.1f}}MW, "
          f"样本={{summary['sample_count']}}")

# 5. 使用建议
print("\\n使用建议:")
print("1. 确保输入数据包含所有训练特征")
print("2. 风向数据需要按照训练时的方式处理（sin/cos分量）")
print("3. 稳定度标签必须与训练时的类别一致")
print("4. 对于未见过的稳定度类型，建议使用最相近的稳定度模型")
print("5. 可以根据置信度对预测结果进行加权")

# ===== 误差传播分析扩展 =====
def analyze_stability_error_propagation(obs_data, forecast_data, stability_labels):
    \"\"\"
    分析不同稳定度条件下的误差传播特性
    \"\"\"
    results = {{}}
    
    for stability in set(stability_labels):
        mask = np.array(stability_labels) == stability
        if np.any(mask) and stability in models:
            # 分别用观测和预报数据预测
            P_obs = models[stability].predict(obs_data[mask])
            P_forecast = models[stability].predict(forecast_data[mask])
            
            # 计算误差传播
            propagation_error = P_forecast - P_obs
            
            results[stability] = {{
                'rmse_propagation': np.sqrt(np.mean(propagation_error**2)),
                'mean_propagation': np.mean(propagation_error),
                'std_propagation': np.std(propagation_error),
                'sample_count': np.sum(mask)
            }}
    
    return results
        '''
        
        example_path = f"{self.save_path}/stability_prediction_usage.py"
        with open(example_path, 'w', encoding='utf-8') as f:
            f.write(example_code)
        print(f"✓ 使用示例已保存: {example_path}")
        
        return example_path
    
    def run_full_stability_analysis(self):
        """运行完整的稳定度分析流程"""
        print("=" * 70)
        print("🌀 基于大气稳定度分类的风电预测分析")
        print("=" * 70)
        
        try:
            # 1. 加载数据
            self.load_data()
            
            # 2. 分析稳定度分布
            stability_counts, confidence_stats = self.analyze_stability_distribution()
            
            # 3. 合并数据
            self.merge_data_by_time()
            
            # 4. 按稳定度分组
            self.prepare_stability_groups(min_samples=500)
            
            # 5. 训练稳定度模型（这一步会设置feature_names）
            self.train_stability_models()
            
            # 6. 分析功率特征（现在feature_names已经设置好了）
            power_stats = self.analyze_stability_power_characteristics()
            
            # 7. 计算SHAP值
            self.calculate_stability_shap_values()
            
            # 8. 绘制性能对比
            performance_df = self.plot_stability_performance_comparison()
            
            # 9. 绘制SHAP对比
            shap_comparison = self.plot_stability_shap_comparison()
            
            # 10. 保存模型和结果
            results_summary, stability_info = self.save_stability_models_and_results()
            
            # 11. 创建使用示例
            self.create_stability_prediction_function()
            
            print("\n" + "=" * 70)
            print("🎉 稳定度分析完成！")
            print("=" * 70)
            
            print("📊 主要发现:")
            print(f"  训练的稳定度模型数量: {len(self.models)}")
            print(f"  稳定度类型: {list(self.models.keys())}")
            
            if performance_df is not None and len(performance_df) > 0:
                best_stability = performance_df.iloc[0]['stability']
                best_r2 = performance_df.iloc[0]['r2']
                print(f"  最佳预测性能: {best_stability} (R²={best_r2:.3f})")
                
                worst_stability = performance_df.iloc[-1]['stability']
                worst_r2 = performance_df.iloc[-1]['r2']
                print(f"  最低预测性能: {worst_stability} (R²={worst_r2:.3f})")
                
                r2_range = best_r2 - worst_r2
                print(f"  性能差距: {r2_range:.3f}")
                
                if r2_range > 0.1:
                    print("  → 不同稳定度的预测难度差异较大，分类建模很有价值")
                else:
                    print("  → 不同稳定度的预测性能相近，可考虑统一建模")
            
            print(f"\n📁 结果文件保存在: {self.save_path}")
            print("  - stability_distribution_analysis.png: 稳定度分布分析")
            print("  - stability_power_characteristics.png: 功率特征分析")
            print("  - stability_performance_comparison.png: 模型性能对比")
            print("  - stability_shap_comparison.png: SHAP重要性对比")
            print("  - stability_prediction_usage.py: 使用示例代码")
            
            print(f"\n🔍 关键洞察:")
            
            # 分析稳定度与预测性能的关系
            if len(self.models) >= 2:
                print("  不同稳定度条件下的预测特征:")
                for stability in self.models.keys():
                    r2 = results_summary[stability]['r2_test']
                    samples = results_summary[stability]['sample_count']
                    power_mean = results_summary[stability]['power_mean']
                    
                    if r2 > 0.8:
                        perf_level = "优秀"
                    elif r2 > 0.6:
                        perf_level = "良好"
                    elif r2 > 0.4:
                        perf_level = "一般"
                    else:
                        perf_level = "较差"
                    
                    print(f"    - {stability}: {perf_level}预测性能 (R²={r2:.3f}), "
                          f"平均功率{power_mean:.1f}MW, 样本{samples}条")
            
            return True
            
        except Exception as e:
            print(f"❌ 分析过程出错: {str(e)}")
            import traceback
            traceback.print_exc()
            return False

if __name__ == "__main__":
    # 配置路径
    WIND_DATA_PATH = "/Users/xiaxin/work/WindForecast_Project/01_Data/processed/imputed_data/changma_imputed_complete.csv"
    STABILITY_DATA_PATH = "/Users/xiaxin/work/WindForecast_Project/03_Results/stability_analysis/changma_stability_results.csv"
    SAVE_PATH = "/Users/xiaxin/work/WindForecast_Project/03_Results/stability_based_prediction"
    
    # 创建保存目录
    import os
    os.makedirs(SAVE_PATH, exist_ok=True)
    
    # 创建分析器并运行
    analyzer = StabilityBasedWindPredictionAnalyzer(WIND_DATA_PATH, STABILITY_DATA_PATH, SAVE_PATH)
    success = analyzer.run_full_stability_analysis()
    
    if success:
        print("\n🎯 稳定度分析成功完成！")
        print("\n💡 后续研究建议:")
        print("  1. 深入分析预测性能差异的物理机制")
        print("  2. 结合天气类型进一步细化稳定度分类")
        print("  3. 研究稳定度转换时段的预测策略")
        print("  4. 开发稳定度自适应的混合预测模型")
        print("  5. 分析不同稳定度下的误差传播特性")
    else:
        print("\n⚠️ 分析失败，请检查错误信息和数据路径")