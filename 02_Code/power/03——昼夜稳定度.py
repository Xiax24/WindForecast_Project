#!/usr/bin/env python3
"""
基于物理风切变阈值的昼夜分类风电预测与SHAP重要性分析
使用大气物理学的绝对阈值进行风切变分类
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
from datetime import datetime, time
import warnings
warnings.filterwarnings('ignore')

# 设置中文显示
plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial Unicode MS', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

class PhysicalWindShearDiurnalAnalyzer:
    def __init__(self, data_path, save_path):
        self.data_path = data_path
        self.save_path = save_path
        self.data = None
        self.groups = {}
        self.feature_names = None
        self.models = {}
        self.shap_explainers = {}
        self.results = {}
        
        # 基于大气物理学的风切变阈值
        self.physical_thresholds = {
            'very_stable': 0.40,      # α > 0.40: 极稳定（很少见）
            'stable': 0.25,           # α > 0.25: 稳定边界层
            'near_neutral': 0.15,     # 0.15 < α ≤ 0.25: 接近中性
            'unstable': 0.05,         # 0.05 < α ≤ 0.15: 轻微不稳定
            'very_unstable': 0.05     # α ≤ 0.05: 强不稳定/中性
        }
        
    def load_and_prepare_data(self):
        """加载和预处理数据"""
        print("📊 加载和预处理数据...")
        
        # 加载数据
        self.data = pd.read_csv(self.data_path)
        print(f"原始数据形状: {self.data.shape}")
        
        # 转换datetime列
        self.data['datetime'] = pd.to_datetime(self.data['datetime'])
        
        # 选择观测数据列
        obs_columns = [col for col in self.data.columns if col.startswith('obs_')]
        obs_columns += ['datetime', 'power']
        
        # 移除密度和湿度
        obs_columns = [col for col in obs_columns if 'density' not in col and 'humidity' not in col]
        
        self.data = self.data[obs_columns].copy()
        print(f"选择列数: {len(obs_columns)-2}")
        
        # 移除缺失值和负功率
        initial_shape = self.data.shape[0]
        self.data = self.data.dropna()
        self.data = self.data[self.data['power'] >= 0]
        final_shape = self.data.shape[0]
        print(f"清理后数据: {final_shape} 行 (移除了 {initial_shape - final_shape} 行)")
        
        return self.data
    
    def calculate_wind_shear(self):
        """计算风切变系数"""
        print("🌪️ 计算风切变系数...")
        
        # 找到不同高度的风速列
        wind_speed_cols = [col for col in self.data.columns if 'wind_speed' in col and col.startswith('obs_')]
        wind_speed_cols.sort()
        
        print(f"发现风速列: {wind_speed_cols}")
        
        if len(wind_speed_cols) < 2:
            raise ValueError("需要至少2个高度的风速数据来计算风切变")
        
        # 提取高度信息
        heights = []
        wind_speeds = {}
        
        for col in wind_speed_cols:
            try:
                height_str = col.split('_')[-1].replace('m', '')
                height = float(height_str)
                heights.append(height)
                wind_speeds[height] = self.data[col]
                print(f"  {col} -> {height}m")
            except:
                print(f"  警告: 无法从 {col} 提取高度信息")
        
        if len(heights) < 2:
            raise ValueError("无法提取足够的高度信息")
        
        heights.sort()
        print(f"✓ 可用高度: {heights} m")
        
        # 计算风切变系数（使用最低和最高两个高度）
        h1, h2 = heights[0], heights[-1]
        v1, v2 = wind_speeds[h1], wind_speeds[h2]
        
        # 避免除零和对数错误
        valid_mask = (v1 > 0.5) & (v2 > 0.5)
        
        self.data = self.data[valid_mask].copy()
        v1, v2 = v1[valid_mask], v2[valid_mask]
        
        # 计算风切变系数
        self.data['wind_shear_alpha'] = np.log(v2 / v1) / np.log(h2 / h1)
        
        print(f"✓ 风切变计算完成，使用 {h1}m 和 {h2}m 高度")
        print(f"  有效数据: {len(self.data)} 条")
        print(f"  风切变范围: {self.data['wind_shear_alpha'].min():.3f} ~ {self.data['wind_shear_alpha'].max():.3f}")
        print(f"  风切变均值: {self.data['wind_shear_alpha'].mean():.3f}")
        print(f"  风切变中位数: {self.data['wind_shear_alpha'].median():.3f}")
        
        return h1, h2
    
    def classify_stability_by_physics(self):
        """基于物理阈值分类大气稳定度"""
        print("🔬 基于物理阈值分类大气稳定度...")
        
        alpha = self.data['wind_shear_alpha']
        
        # 定义物理分类条件
        conditions = [
            alpha > self.physical_thresholds['stable'],           # α > 0.25: 稳定
            (alpha > self.physical_thresholds['unstable']) & 
            (alpha <= self.physical_thresholds['stable']),        # 0.05 < α ≤ 0.25: 接近中性
            alpha <= self.physical_thresholds['unstable']         # α ≤ 0.05: 不稳定
        ]
        
        choices = ['stable', 'near_neutral', 'unstable']
        
        self.data['stability_class'] = np.select(conditions, choices, default='unknown')
        
        # 统计各稳定度类别
        stability_counts = self.data['stability_class'].value_counts()
        print(f"\n📊 基于物理阈值的稳定度分类:")
        print(f"  稳定 (α > {self.physical_thresholds['stable']}): {stability_counts.get('stable', 0)} 条")
        print(f"  接近中性 ({self.physical_thresholds['unstable']} < α ≤ {self.physical_thresholds['stable']}): {stability_counts.get('near_neutral', 0)} 条")
        print(f"  不稳定 (α ≤ {self.physical_thresholds['unstable']}): {stability_counts.get('unstable', 0)} 条")
        
        # 分析各类别的风切变统计
        stability_stats = self.data.groupby('stability_class')['wind_shear_alpha'].agg(['count', 'mean', 'std', 'min', 'max'])
        print(f"\n各稳定度类别风切变统计:")
        print(stability_stats.round(3))
        
        return stability_counts, stability_stats
    
    def determine_day_night(self):
        """确定昼夜分类"""
        print("☀️🌙 确定昼夜分类...")
        
        # 提取小时信息
        self.data['hour'] = self.data['datetime'].dt.hour
        
        # 昼夜划分（可根据地理位置调整）
        day_start, day_end = 6, 18
        
        self.data['is_daytime'] = ((self.data['hour'] >= day_start) & 
                                  (self.data['hour'] < day_end))
        
        day_count = self.data['is_daytime'].sum()
        night_count = len(self.data) - day_count
        
        print(f"✓ 昼夜分类完成:")
        print(f"  白天 ({day_start}:00-{day_end}:00): {day_count} 条")
        print(f"  夜间: {night_count} 条")
        
        return day_start, day_end
    
    def create_physics_based_classification(self):
        """创建基于物理原理的稳定度-昼夜组合分类"""
        print("🔄 创建物理稳定度-昼夜组合分类...")
        
        # 创建组合分类
        self.data['physics_class'] = self.data['stability_class'].astype(str) + '_' + \
                                   self.data['is_daytime'].map({True: 'day', False: 'night'})
        
        # 统计各分类
        class_stats = self.data.groupby('physics_class').agg({
            'power': ['count', 'mean', 'std'],
            'wind_shear_alpha': ['mean', 'std'],
            'hour': 'mean'
        }).round(3)
        
        print("\n📊 物理稳定度-昼夜分类统计:")
        print("=" * 80)
        for class_name in class_stats.index:
            count = class_stats.loc[class_name, ('power', 'count')]
            power_mean = class_stats.loc[class_name, ('power', 'mean')]
            power_std = class_stats.loc[class_name, ('power', 'std')]
            shear_mean = class_stats.loc[class_name, ('wind_shear_alpha', 'mean')]
            shear_std = class_stats.loc[class_name, ('wind_shear_alpha', 'std')]
            avg_hour = class_stats.loc[class_name, ('hour', 'mean')]
            percentage = count / len(self.data) * 100
            
            print(f"{class_name}:")
            print(f"  样本数: {count} ({percentage:.1f}%)")
            print(f"  功率: {power_mean:.1f}±{power_std:.1f} MW")
            print(f"  风切变: {shear_mean:.3f}±{shear_std:.3f}")
            print(f"  平均时间: {avg_hour:.1f}时")
            print("-" * 50)
        
        # 分析是否符合物理预期
        self.analyze_physical_consistency()
        
        return class_stats
    
    def analyze_physical_consistency(self):
        """分析分类结果是否符合物理预期"""
        print("🔬 分析物理一致性...")
        
        # 分析各组合的物理合理性
        day_stable = len(self.data[(self.data['stability_class'] == 'stable') & 
                                 (self.data['is_daytime'] == True)])
        day_unstable = len(self.data[(self.data['stability_class'] == 'unstable') & 
                                   (self.data['is_daytime'] == True)])
        night_stable = len(self.data[(self.data['stability_class'] == 'stable') & 
                                   (self.data['is_daytime'] == False)])
        night_unstable = len(self.data[(self.data['stability_class'] == 'unstable') & 
                                     (self.data['is_daytime'] == False)])
        
        print(f"\n物理一致性分析:")
        print(f"  白天稳定 (异常): {day_stable} 条")
        print(f"  白天不稳定 (正常): {day_unstable} 条") 
        print(f"  夜间稳定 (正常): {night_stable} 条")
        print(f"  夜间不稳定 (异常): {night_unstable} 条")
        
        # 计算物理一致性比例
        normal_cases = day_unstable + night_stable
        total_cases = len(self.data)
        consistency_ratio = normal_cases / total_cases * 100
        
        print(f"\n物理一致性比例: {consistency_ratio:.1f}%")
        
        if consistency_ratio > 70:
            print("✓ 物理一致性良好，分类符合边界层理论")
        elif consistency_ratio > 50:
            print("⚠ 物理一致性中等，可能存在特殊地形或气候影响")
        else:
            print("❌ 物理一致性较差，需要检查数据质量或调整阈值")
        
        return consistency_ratio
    
    def visualize_physics_classification(self):
        """可视化物理分类结果"""
        print("📊 可视化物理分类结果...")
        
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle('基于物理阈值的稳定度-昼夜分类分析', fontsize=16, fontweight='bold')
        
        # 1. 风切变分布与物理阈值
        ax1 = axes[0, 0]
        alpha_values = self.data['wind_shear_alpha']
        ax1.hist(alpha_values, bins=50, alpha=0.7, color='skyblue', density=True)
        
        # 标记物理阈值
        ax1.axvline(x=self.physical_thresholds['stable'], color='red', linestyle='--', 
                   linewidth=2, label=f'稳定阈值 (α={self.physical_thresholds["stable"]})')
        ax1.axvline(x=self.physical_thresholds['unstable'], color='blue', linestyle='--',
                   linewidth=2, label=f'不稳定阈值 (α={self.physical_thresholds["unstable"]})')
        
        ax1.set_xlabel('风切变系数 α')
        ax1.set_ylabel('密度')
        ax1.set_title('风切变分布与物理阈值')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # 2. 稳定度日变化
        ax2 = axes[0, 1]
        hourly_stability = self.data.groupby(['hour', 'stability_class']).size().unstack(fill_value=0)
        hourly_stability_pct = hourly_stability.div(hourly_stability.sum(axis=1), axis=0) * 100
        
        if not hourly_stability_pct.empty:
            hourly_stability_pct.plot(kind='area', stacked=True, ax=ax2, alpha=0.7)
            ax2.set_xlabel('小时')
            ax2.set_ylabel('百分比 (%)')
            ax2.set_title('稳定度的日变化模式')
            ax2.legend(title='稳定度类型')
            ax2.grid(True, alpha=0.3)
        
        # 3. 物理分类散点图
        ax3 = axes[0, 2]
        classes = self.data['physics_class'].unique()
        colors = ['red', 'blue', 'orange', 'green', 'purple', 'brown']
        
        for i, class_name in enumerate(classes):
            if 'unknown' not in class_name:
                class_data = self.data[self.data['physics_class'] == class_name]
                ax3.scatter(class_data['wind_shear_alpha'], class_data['power'], 
                           alpha=0.6, s=15, label=class_name, color=colors[i % len(colors)])
        
        ax3.axvline(x=self.physical_thresholds['stable'], color='red', linestyle='--', alpha=0.5)
        ax3.axvline(x=self.physical_thresholds['unstable'], color='blue', linestyle='--', alpha=0.5)
        ax3.set_xlabel('风切变系数 α')
        ax3.set_ylabel('功率 (MW)')
        ax3.set_title('物理分类散点图')
        ax3.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        ax3.grid(True, alpha=0.3)
        
        # 4. 各分类功率箱线图
        ax4 = axes[1, 0]
        power_data_by_class = []
        class_labels = []
        
        for class_name in sorted(classes):
            if 'unknown' not in class_name:
                power_data = self.data[self.data['physics_class'] == class_name]['power']
                if len(power_data) > 0:
                    power_data_by_class.append(power_data)
                    class_labels.append(class_name.replace('_', '\n'))
        
        if power_data_by_class:
            ax4.boxplot(power_data_by_class, labels=class_labels)
            ax4.set_ylabel('功率 (MW)')
            ax4.set_title('各物理分类功率分布')
            ax4.tick_params(axis='x', rotation=0)
            ax4.grid(True, alpha=0.3)
        
        # 5. 物理一致性饼图
        ax5 = axes[1, 1]
        consistency_data = {
            '正常组合\n(白天不稳定+夜间稳定)': len(self.data[
                ((self.data['stability_class'] == 'unstable') & (self.data['is_daytime'] == True)) |
                ((self.data['stability_class'] == 'stable') & (self.data['is_daytime'] == False))
            ]),
            '异常组合\n(白天稳定+夜间不稳定)': len(self.data[
                ((self.data['stability_class'] == 'stable') & (self.data['is_daytime'] == True)) |
                ((self.data['stability_class'] == 'unstable') & (self.data['is_daytime'] == False))
            ]),
            '中性条件': len(self.data[self.data['stability_class'] == 'near_neutral'])
        }
        
        colors_pie = ['lightgreen', 'lightcoral', 'lightyellow']
        ax5.pie(consistency_data.values(), labels=consistency_data.keys(), colors=colors_pie,
                autopct='%1.1f%%', startangle=90)
        ax5.set_title('物理一致性分析')
        
        # 6. 风切变-功率相关性分析
        ax6 = axes[1, 2]
        
        # 按稳定度类别分析相关性
        corr_data = []
        for stability in ['stable', 'near_neutral', 'unstable']:
            for period in ['day', 'night']:
                class_name = f'{stability}_{period}'
                if class_name in self.data['physics_class'].values:
                    subset = self.data[self.data['physics_class'] == class_name]
                    if len(subset) > 10:  # 确保有足够样本
                        corr = subset['wind_shear_alpha'].corr(subset['power'])
                        corr_data.append({
                            'class': class_name,
                            'correlation': corr,
                            'count': len(subset)
                        })
        
        if corr_data:
            corr_df = pd.DataFrame(corr_data)
            bars = ax6.bar(range(len(corr_df)), corr_df['correlation'], 
                          alpha=0.7, color=['red' if x < 0 else 'blue' for x in corr_df['correlation']])
            ax6.set_xticks(range(len(corr_df)))
            ax6.set_xticklabels(corr_df['class'], rotation=45, ha='right')
            ax6.set_ylabel('相关系数')
            ax6.set_title('风切变-功率相关性')
            ax6.axhline(y=0, color='black', linestyle='-', alpha=0.5)
            ax6.grid(True, alpha=0.3)
            
            # 添加样本数标签
            for bar, count in zip(bars, corr_df['count']):
                height = bar.get_height()
                ax6.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                        f'n={count}', ha='center', va='bottom', fontsize=8)
        
        plt.tight_layout()
        plt.savefig(f"{self.save_path}/physics_based_classification.png", dpi=300, bbox_inches='tight')
        plt.show()
    
    def prepare_classification_groups(self, min_samples=200):
        """按物理分类分组数据"""
        print(f"📊 按物理分类分组数据 (最小样本数: {min_samples})...")
        
        class_counts = self.data['physics_class'].value_counts()
        print(f"所有分类样本数: {dict(class_counts)}")
        
        # 只选择样本数足够的分类
        valid_classes = class_counts[class_counts >= min_samples].index.tolist()
        valid_classes = [cls for cls in valid_classes if 'unknown' not in cls]
        
        print(f"样本数足够的分类: {valid_classes}")
        
        for class_name in valid_classes:
            class_data = self.data[self.data['physics_class'] == class_name].copy()
            self.groups[class_name] = class_data
            print(f"  {class_name}: {len(class_data)} 条样本")
        
        return self.groups
    
    def process_wind_direction(self, data):
        """处理风向变量为sin/cos分量"""
        data = data.copy()
        wind_dir_cols = [col for col in data.columns if 'wind_direction' in col]
        
        if wind_dir_cols:
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
    
    def train_physics_models(self):
        """为每种物理分类训练独立的预测模型"""
        print("🚀 训练物理分类模型...")
        
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
        
        for class_name, data in self.groups.items():
            print(f"\n训练 {class_name} 模型...")
            print(f"数据量: {len(data)} 条")
            
            # 处理风向和准备特征
            data_processed = self.process_wind_direction(data)
            
            # 选择特征列
            exclude_cols = ['datetime', 'power', 'hour', 'is_daytime', 'wind_shear_alpha',
                          'stability_class', 'physics_class']
            feature_cols = [col for col in data_processed.columns if col not in exclude_cols]
            
            # 创建特征矩阵
            X = data_processed[feature_cols].values
            y = data_processed['power'].values
            
            # 保存特征名称
            if self.feature_names is None:
                self.feature_names = feature_cols
                print(f"  设置特征名称，共 {len(feature_cols)} 个特征")
            
            print(f"  特征数量: {len(feature_cols)}")
            print(f"  功率范围: {y.min():.1f} - {y.max():.1f} MW")
            print(f"  功率均值: {y.mean():.1f} MW")
            print(f"  风切变范围: {data['wind_shear_alpha'].min():.3f} - {data['wind_shear_alpha'].max():.3f}")
            
            # 数据分割
            if len(data) >= 100:
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
                self.models[class_name] = model
                self.results[class_name] = {
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
                    'power_std': y.std(),
                    'shear_mean': data['wind_shear_alpha'].mean(),
                    'shear_std': data['wind_shear_alpha'].std()
                }
                
                print(f"  ✓ 训练完成 - R²: {test_r2:.4f}, RMSE: {test_rmse:.2f} MW")
                print(f"    过拟合检查: 训练R²={train_r2:.4f}, 测试R²={test_r2:.4f}, 差值={train_r2-test_r2:.4f}")
                
            else:
                print(f"  ⚠️ 样本数不足 ({len(data)} < 100)，跳过训练")
        
        print(f"\n✓ 共训练了 {len(self.models)} 个物理分类模型")
        return self.models
    
    def calculate_shap_values(self, n_samples=800):
        """计算各分类模型的SHAP值"""
        print("📊 计算SHAP重要性...")
        
        for class_name in self.models.keys():
            print(f"计算 {class_name} 的SHAP值...")
            
            # 获取测试数据
            X_test = self.results[class_name]['X_test']
            
            # 限制样本数量
            if len(X_test) > n_samples:
                indices = np.random.choice(len(X_test), n_samples, replace=False)
                X_sample = X_test[indices]
            else:
                X_sample = X_test
            
            # 计算SHAP值
            explainer = shap.TreeExplainer(self.models[class_name])
            shap_values = explainer.shap_values(X_sample)
            
            # 保存结果
            self.shap_explainers[class_name] = explainer
            self.results[class_name]['shap_values'] = shap_values
            self.results[class_name]['X_shap'] = X_sample
            
            print(f"  ✓ 完成 (样本数: {len(X_sample)})")
    
    def plot_performance_comparison(self):
        """绘制模型性能对比"""
        print("📈 绘制模型性能对比...")
        
        if not self.results:
            print("⚠️ 没有训练结果，跳过性能对比")
            return
        
        # 准备数据
        class_names = list(self.results.keys())
        r2_values = [self.results[cls]['r2_test'] for cls in class_names]
        rmse_values = [self.results[cls]['rmse_test'] for cls in class_names]
        sample_counts = [self.results[cls]['sample_count'] for cls in class_names]
        power_means = [self.results[cls]['power_mean'] for cls in class_names]
        shear_means = [self.results[cls]['shear_mean'] for cls in class_names]
        
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle('基于物理阈值的稳定度-昼夜分类模型性能对比', fontsize=16, fontweight='bold')
        
        # 1. R² 性能对比
        ax1 = axes[0, 0]
        # 根据物理类型着色
        colors = []
        for name in class_names:
            if 'stable' in name:
                colors.append('red')
            elif 'unstable' in name:
                colors.append('blue')
            else:
                colors.append('orange')
        
        bars1 = ax1.bar(range(len(class_names)), r2_values, color=colors, alpha=0.7)
        ax1.set_xticks(range(len(class_names)))
        ax1.set_xticklabels([name.replace('_', '\n') for name in class_names], rotation=0)
        ax1.set_ylabel('R² Score')
        ax1.set_title('R² 性能对比')
        ax1.set_ylim(0, 1)
        
        # 添加数值标签
        for bar, r2 in zip(bars1, r2_values):
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                    f'{r2:.3f}', ha='center', va='bottom', fontweight='bold')
        
        # 2. RMSE 对比
        ax2 = axes[0, 1]
        bars2 = ax2.bar(range(len(class_names)), rmse_values, color=colors, alpha=0.7)
        ax2.set_xticks(range(len(class_names)))
        ax2.set_xticklabels([name.replace('_', '\n') for name in class_names], rotation=0)
        ax2.set_ylabel('RMSE (MW)')
        ax2.set_title('RMSE 对比')
        
        for bar, rmse in zip(bars2, rmse_values):
            height = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width()/2., height + height*0.01,
                    f'{rmse:.1f}', ha='center', va='bottom', fontweight='bold')
        
        # 3. 样本数量对比
        ax3 = axes[0, 2]
        bars3 = ax3.bar(range(len(class_names)), sample_counts, color=colors, alpha=0.7)
        ax3.set_xticks(range(len(class_names)))
        ax3.set_xticklabels([name.replace('_', '\n') for name in class_names], rotation=0)
        ax3.set_ylabel('样本数量')
        ax3.set_title('各分类样本分布')
        
        for bar, count in zip(bars3, sample_counts):
            height = bar.get_height()
            ax3.text(bar.get_x() + bar.get_width()/2., height + height*0.01,
                    f'{count}', ha='center', va='bottom', fontweight='bold')
        
        # 4. 昼夜性能对比
        ax4 = axes[1, 0]
        day_classes = [cls for cls in class_names if 'day' in cls]
        night_classes = [cls for cls in class_names if 'night' in cls]
        
        day_r2 = [self.results[cls]['r2_test'] for cls in day_classes]
        night_r2 = [self.results[cls]['r2_test'] for cls in night_classes]
        
        x = np.arange(max(len(day_classes), len(night_classes)))
        width = 0.35
        
        if day_r2:
            ax4.bar(x[:len(day_r2)] - width/2, day_r2, width, label='白天', alpha=0.7, color='orange')
        if night_r2:
            ax4.bar(x[:len(night_r2)] + width/2, night_r2, width, label='夜间', alpha=0.7, color='navy')
        
        ax4.set_xlabel('稳定度类型')
        ax4.set_ylabel('R² Score')
        ax4.set_title('昼夜性能对比')
        ax4.legend()
        ax4.grid(True, alpha=0.3)
        
        # 5. 稳定度性能对比
        ax5 = axes[1, 1]
        stable_classes = [cls for cls in class_names if 'stable' in cls]
        unstable_classes = [cls for cls in class_names if 'unstable' in cls]
        neutral_classes = [cls for cls in class_names if 'neutral' in cls]
        
        stability_data = []
        stability_labels = []
        if stable_classes:
            stability_data.append([self.results[cls]['r2_test'] for cls in stable_classes])
            stability_labels.append('稳定')
        if neutral_classes:
            stability_data.append([self.results[cls]['r2_test'] for cls in neutral_classes])
            stability_labels.append('中性')
        if unstable_classes:
            stability_data.append([self.results[cls]['r2_test'] for cls in unstable_classes])
            stability_labels.append('不稳定')
        
        if stability_data:
            ax5.boxplot(stability_data, labels=stability_labels)
            ax5.set_ylabel('R² Score')
            ax5.set_title('稳定度类型性能对比')
            ax5.grid(True, alpha=0.3)
        
        # 6. 物理阈值验证
        ax6 = axes[1, 2]
        
        # 绘制风切变-性能散点图
        scatter = ax6.scatter(shear_means, r2_values, c=sample_counts, 
                            s=100, cmap='viridis', alpha=0.7)
        
        # 添加物理阈值线
        ax6.axvline(x=self.physical_thresholds['stable'], color='red', linestyle='--', 
                   alpha=0.7, label=f'稳定阈值 (α={self.physical_thresholds["stable"]})')
        ax6.axvline(x=self.physical_thresholds['unstable'], color='blue', linestyle='--',
                   alpha=0.7, label=f'不稳定阈值 (α={self.physical_thresholds["unstable"]})')
        
        # 添加分类标签
        for i, class_name in enumerate(class_names):
            ax6.annotate(class_name.replace('_', '\n'), 
                        (shear_means[i], r2_values[i]),
                        xytext=(5, 5), textcoords='offset points', fontsize=8)
        
        ax6.set_xlabel('平均风切变系数 α')
        ax6.set_ylabel('R² Score')
        ax6.set_title('风切变-性能关系\n(点大小=样本数)')
        ax6.legend()
        ax6.grid(True, alpha=0.3)
        
        # 添加颜色条
        cbar = plt.colorbar(scatter, ax=ax6)
        cbar.set_label('样本数量')
        
        plt.tight_layout()
        plt.savefig(f"{self.save_path}/physics_performance_comparison.png", dpi=300, bbox_inches='tight')
        plt.show()
        
        # 输出性能排名
        print("\n📊 性能排名 (按R²降序):")
        performance_df = pd.DataFrame({
            'classification': class_names,
            'r2': r2_values,
            'rmse': rmse_values,
            'samples': sample_counts,
            'power_mean': power_means,
            'shear_mean': shear_means
        }).sort_values('r2', ascending=False)
        
        for i, row in performance_df.iterrows():
            stability = 'stable' if 'stable' in row['classification'] else \
                       'unstable' if 'unstable' in row['classification'] else 'neutral'
            period = 'day' if 'day' in row['classification'] else 'night'
            
            print(f"  {i+1}. {row['classification']} ({stability}+{period}): "
                  f"R²={row['r2']:.3f}, RMSE={row['rmse']:.1f}MW, "
                  f"样本={row['samples']}, α={row['shear_mean']:.3f}")
        
        return performance_df
    
    def plot_shap_comparison(self):
        """绘制SHAP重要性对比"""
        print("📊 绘制SHAP重要性对比...")
        
        if not self.results or not any('shap_values' in result for result in self.results.values()):
            print("⚠️ 没有SHAP结果，跳过SHAP对比")
            return
        
        # 计算各分类的平均SHAP重要性
        shap_importance_df = pd.DataFrame({'feature': self.feature_names})
        
        for class_name in self.results.keys():
            if 'shap_values' in self.results[class_name]:
                shap_values = self.results[class_name]['shap_values']
                importance = np.abs(shap_values).mean(axis=0)
                shap_importance_df[f'{class_name}_importance'] = importance
        
        # 计算总体重要性排序
        importance_cols = [col for col in shap_importance_df.columns if 'importance' in col]
        shap_importance_df['avg_importance'] = shap_importance_df[importance_cols].mean(axis=1)
        shap_importance_df = shap_importance_df.sort_values('avg_importance', ascending=False)
        
        # 绘制对比图
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle('基于物理阈值的稳定度分类SHAP重要性对比', fontsize=16, fontweight='bold')
        
        # 1. Top特征重要性对比
        top_n = 15
        top_features = shap_importance_df.head(top_n)
        
        ax1 = axes[0, 0]
        class_names = [col.replace('_importance', '') for col in importance_cols]
        x = np.arange(len(top_features))
        width = 0.8 / len(class_names)
        
        # 先打印一下实际的分类名称，用于调试
        print(f"实际的分类名称: {class_names}")
        
        # 设置颜色和填充样式
        for i, class_name in enumerate(class_names):
            col = f'{class_name}_importance'
            offset = (i - len(class_names)/2 + 0.5) * width
            
            print(f"处理分类: {class_name}")  # 调试信息
            
            # 根据稳定度设置颜色 - 使用更明显的颜色对比
            if 'stable' in class_name and 'unstable' not in class_name:  # 只匹配stable，排除unstable
                color = 'gray'  # 深红色
                print(f"  -> 设置为稳定(红色)")
            elif 'unstable' in class_name:
                color = 'royalblue'  # 皇家蓝
                print(f"  -> 设置为不稳定(蓝色)")
            elif 'neutral' in class_name:
                color = 'orange'  # 深橙色
                print(f"  -> 设置为中性(橙色)")
            else:
                color = 'gray'  # 默认颜色
                print(f"  -> 设置为默认(灰色)")
            
            # 根据昼夜设置填充样式
            if 'night' in class_name:
                # 夜间用斜线条纹
                hatch = '//'
                alpha = 0.8
                edgecolor = 'black'
                linewidth = 1.0
                print(f"  -> 夜间(条纹)")
            elif 'day' in class_name:
                # 白天用实心填充
                hatch = None
                alpha = 0.9
                edgecolor = 'darkgray'
                linewidth = 0.8
                print(f"  -> 白天(实心)")
            else:
                # 默认
                hatch = None
                alpha = 0.7
                edgecolor = 'black'
                linewidth = 0.5
                print(f"  -> 默认样式")
            
            ax1.barh(x + offset, top_features[col], width, 
                    label=class_name.replace('_', ' '), 
                    color=color, alpha=alpha, hatch=hatch, 
                    edgecolor=edgecolor, linewidth=linewidth)
        
        ax1.set_yticks(x)
        ax1.set_yticklabels(top_features['feature'], fontsize=8)
        ax1.set_xlabel('SHAP重要性')
        ax1.set_title(f'Top {top_n} 特征重要性对比')
        ax1.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        ax1.grid(True, alpha=0.3)
        
        # 2. 稳定度vs不稳定度对比
        ax2 = axes[0, 1]
        
        # 分离稳定和不稳定类型
        stable_cols = [col for col in importance_cols if 'stable' in col and 'unstable' not in col]
        unstable_cols = [col for col in importance_cols if 'unstable' in col]
        
        if stable_cols and unstable_cols:
            stable_avg = shap_importance_df[stable_cols].mean(axis=1)
            unstable_avg = shap_importance_df[unstable_cols].mean(axis=1)
            
            # 选择top特征进行对比
            top_features_diff = shap_importance_df.head(20)
            stable_top = stable_avg[top_features_diff.index]
            unstable_top = unstable_avg[top_features_diff.index]
            
            x = np.arange(len(top_features_diff))
            width = 0.35
            
            ax2.barh(x - width/2, stable_top, width, label='稳定条件', alpha=0.7, color='red')
            ax2.barh(x + width/2, unstable_top, width, label='不稳定条件', alpha=0.7, color='blue')
            
            ax2.set_yticks(x)
            ax2.set_yticklabels(top_features_diff['feature'], fontsize=8)
            ax2.set_xlabel('平均SHAP重要性')
            ax2.set_title('稳定vs不稳定条件重要性对比')
            ax2.legend()
            ax2.grid(True, alpha=0.3)
        
        # 3. 昼夜对比
        ax3 = axes[1, 0]
        
        day_cols = [col for col in importance_cols if 'day' in col]
        night_cols = [col for col in importance_cols if 'night' in col]
        
        if day_cols and night_cols:
            day_avg = shap_importance_df[day_cols].mean(axis=1)
            night_avg = shap_importance_df[night_cols].mean(axis=1)
            
            top_features_day_night = shap_importance_df.head(20)
            day_top = day_avg[top_features_day_night.index]
            night_top = night_avg[top_features_day_night.index]
            
            x = np.arange(len(top_features_day_night))
            width = 0.35
            
            ax3.barh(x - width/2, day_top, width, label='白天', alpha=0.7, color='orange')
            ax3.barh(x + width/2, night_top, width, label='夜间', alpha=0.7, color='navy')
            
            ax3.set_yticks(x)
            ax3.set_yticklabels(top_features_day_night['feature'], fontsize=8)
            ax3.set_xlabel('平均SHAP重要性')
            ax3.set_title('昼夜条件重要性对比')
            ax3.legend()
            ax3.grid(True, alpha=0.3)
        
        # 4. 物理一致性验证
        ax4 = axes[1, 1]
        
        # 分析10m vs 70m风速在不同条件下的重要性
        wind_10m_col = [f for f in self.feature_names if '10m' in f and 'wind_speed' in f]
        wind_70m_col = [f for f in self.feature_names if '70m' in f and 'wind_speed' in f]
        
        if wind_10m_col and wind_70m_col:
            wind_10m_col = wind_10m_col[0]
            wind_70m_col = wind_70m_col[0]
            
            results_10m = []
            results_70m = []
            labels = []
            
            for class_name in class_names:
                col = f'{class_name}_importance'
                if col in shap_importance_df.columns:
                    feature_10m_idx = shap_importance_df[shap_importance_df['feature'] == wind_10m_col].index
                    feature_70m_idx = shap_importance_df[shap_importance_df['feature'] == wind_70m_col].index
                    
                    if len(feature_10m_idx) > 0 and len(feature_70m_idx) > 0:
                        importance_10m = shap_importance_df.loc[feature_10m_idx[0], col]
                        importance_70m = shap_importance_df.loc[feature_70m_idx[0], col]
                        
                        results_10m.append(importance_10m)
                        results_70m.append(importance_70m)
                        labels.append(class_name.replace('_', '\n'))
            
            if results_10m and results_70m:
                x = np.arange(len(labels))
                width = 0.35
                
                bars1 = ax4.bar(x - width/2, results_10m, width, label='10m风速', alpha=0.7, color='lightblue')
                bars2 = ax4.bar(x + width/2, results_70m, width, label='70m风速', alpha=0.7, color='darkblue')
                
                ax4.set_xticks(x)
                ax4.set_xticklabels(labels, rotation=45, ha='right')
                ax4.set_ylabel('SHAP重要性')
                ax4.set_title('10m vs 70m风速重要性\n(验证物理机制)')
                ax4.legend()
                ax4.grid(True, alpha=0.3)
                
                # 添加数值标签
                for bars in [bars1, bars2]:
                    for bar in bars:
                        height = bar.get_height()
                        ax4.text(bar.get_x() + bar.get_width()/2., height + height*0.01,
                                f'{height:.2f}', ha='center', va='bottom', fontsize=8)
        
        plt.tight_layout()
        plt.savefig(f"{self.save_path}/physics_shap_comparison.png", dpi=300, bbox_inches='tight')
        plt.show()
        
        # 保存重要性对比数据
        shap_importance_df.to_csv(f"{self.save_path}/physics_shap_importance.csv", index=False)
        print("✓ SHAP重要性对比数据已保存")
        
        return shap_importance_df
    
    def save_models_and_results(self):
        """保存模型和结果"""
        print("💾 保存模型和结果...")
        
        # 保存各分类模型
        for class_name, model in self.models.items():
            model_path = f"{self.save_path}/physics_model_{class_name}.pkl"
            joblib.dump(model, model_path)
            print(f"✓ {class_name}模型已保存: {model_path}")
        
        # 保存特征名称
        feature_names_path = f"{self.save_path}/feature_names.pkl"
        joblib.dump(self.feature_names, feature_names_path)
        print(f"✓ 特征名称已保存: {feature_names_path}")
        
        # 保存物理阈值
        thresholds_path = f"{self.save_path}/physical_thresholds.pkl"
        joblib.dump(self.physical_thresholds, thresholds_path)
        print(f"✓ 物理阈值已保存: {thresholds_path}")
        
        # 保存结果摘要
        results_summary = {}
        for class_name in self.results:
            results_summary[class_name] = {
                'r2_test': self.results[class_name]['r2_test'],
                'rmse_test': self.results[class_name]['rmse_test'],
                'sample_count': self.results[class_name]['sample_count'],
                'power_mean': self.results[class_name]['power_mean'],
                'power_std': self.results[class_name]['power_std'],
                'shear_mean': self.results[class_name]['shear_mean'],
                'shear_std': self.results[class_name]['shear_std']
            }
        
        summary_path = f"{self.save_path}/physics_results_summary.pkl"
        joblib.dump(results_summary, summary_path)
        print(f"✓ 结果摘要已保存: {summary_path}")
        
        return results_summary
    
    def run_full_physics_analysis(self):
        """运行完整的物理阈值分析流程"""
        print("=" * 70)
        print("🔬 基于物理阈值的稳定度-昼夜分类风电预测分析")
        print("=" * 70)
        
        try:
            # 1. 加载和预处理数据
            self.load_and_prepare_data()
            
            # 2. 计算风切变系数
            h1, h2 = self.calculate_wind_shear()
            
            # 3. 基于物理阈值分类稳定度
            stability_counts, stability_stats = self.classify_stability_by_physics()
            
            # 4. 确定昼夜分类
            day_start, day_end = self.determine_day_night()
            
            # 5. 创建物理组合分类
            class_stats = self.create_physics_based_classification()
            
            # 6. 可视化分类结果
            self.visualize_physics_classification()
            
            # 7. 按分类分组
            self.prepare_classification_groups(min_samples=200)
            
            # 8. 训练分类模型
            self.train_physics_models()
            
            # 9. 计算SHAP值
            self.calculate_shap_values()
            
            # 10. 绘制性能对比
            performance_df = self.plot_performance_comparison()
            
            # 11. 绘制SHAP对比
            shap_comparison = self.plot_shap_comparison()
            
            # 12. 保存模型和结果
            results_summary = self.save_models_and_results()
            
            print("\n" + "=" * 70)
            print("🎉 物理阈值分析完成！")
            print("=" * 70)
            
            print("📊 主要发现:")
            print(f"  风切变计算: 使用 {h1}m 和 {h2}m 高度数据")
            print(f"  物理阈值: 稳定(α>{self.physical_thresholds['stable']}), "
                  f"不稳定(α≤{self.physical_thresholds['unstable']})")
            print(f"  昼夜划分: {day_start}:00-{day_end}:00为白天")
            print(f"  训练的分类模型数量: {len(self.models)}")
            print(f"  分类类型: {list(self.models.keys())}")
            
            if performance_df is not None and len(performance_df) > 0:
                best_class = performance_df.iloc[0]['classification']
                best_r2 = performance_df.iloc[0]['r2']
                worst_class = performance_df.iloc[-1]['classification']
                worst_r2 = performance_df.iloc[-1]['r2']
                
                print(f"  最佳预测性能: {best_class} (R²={best_r2:.3f})")
                print(f"  最低预测性能: {worst_class} (R²={worst_r2:.3f})")
                
                r2_range = best_r2 - worst_r2
                print(f"  性能差距: {r2_range:.3f}")
                
                if r2_range > 0.1:
                    print("  → 基于物理阈值的分类很有价值，性能差异明显")
                else:
                    print("  → 各分类预测性能相近，物理差异不显著")
            
            # 分析物理机制验证
            print(f"\n🔬 物理机制验证:")
            print("  基于物理阈值的分类结果应该显示:")
            print("  1. 稳定条件下：10m风速重要性更高")
            print("  2. 不稳定条件下：70m风速重要性更高")
            print("  3. 夜间稳定组合：预测性能可能最好")
            print("  4. 白天不稳定组合：需要更复杂的特征")
            
            print(f"\n📁 结果文件保存在: {self.save_path}")
            print("  - physics_based_classification.png: 物理分类分析")
            print("  - physics_performance_comparison.png: 模型性能对比")
            print("  - physics_shap_comparison.png: SHAP重要性对比")
            print("  - physics_shap_importance.csv: 详细重要性数据")
            
            return True
            
        except Exception as e:
            print(f"❌ 分析过程出错: {str(e)}")
            import traceback
            traceback.print_exc()
            return False

if __name__ == "__main__":
    # 配置路径
    DATA_PATH = "/Users/xiaxin/work/WindForecast_Project/01_Data/processed/imputed_data/changma_imputed_complete.csv"
    SAVE_PATH = "/Users/xiaxin/work/WindForecast_Project/03_Results/physics_wind_shear_analysis"
    
    # 创建保存目录
    import os
    os.makedirs(SAVE_PATH, exist_ok=True)
    
    # 创建分析器并运行
    analyzer = PhysicalWindShearDiurnalAnalyzer(DATA_PATH, SAVE_PATH)
    success = analyzer.run_full_physics_analysis()
    
    if success:
        print("\n🎯 基于物理阈值的分析成功完成！")
        print("\n💡 后续研究建议:")
        print("  1. 验证10m vs 70m风速重要性是否符合物理预期")
        print("  2. 分析异常组合(白天稳定/夜间不稳定)的成因")
        print("  3. 优化物理阈值以提升分类效果")
        print("  4. 研究季节性对物理阈值的影响")
        print("  5. 开发自适应的物理约束预测模型")
    else:
        print("\n⚠️ 分析失败，请检查错误信息和数据路径")