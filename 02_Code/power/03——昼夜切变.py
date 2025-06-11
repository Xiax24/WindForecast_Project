#!/usr/bin/env python3
"""
基于风切变-昼夜分类的风电预测与SHAP重要性分析
分类策略：高风切变+白天 / 低风切变+夜间
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

class WindShearDiurnalAnalyzer:
    def __init__(self, data_path, save_path):
        self.data_path = data_path
        self.save_path = save_path
        self.data = None
        self.groups = {}
        self.feature_names = None
        self.models = {}
        self.shap_explainers = {}
        self.results = {}
        self.wind_shear_threshold = None
        
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
        wind_speed_cols.sort()  # 按名称排序，通常包含高度信息
        
        print(f"发现风速列: {wind_speed_cols}")
        
        if len(wind_speed_cols) < 2:
            raise ValueError("需要至少2个高度的风速数据来计算风切变")
        
        # 计算风切变系数 α，使用风速剖面公式: V(z) = V(z_ref) * (z/z_ref)^α
        # α = ln(V2/V1) / ln(z2/z1)
        
        # 提取高度信息（假设列名格式为 obs_wind_speed_XXXm）
        heights = []
        wind_speeds = {}
        
        for col in wind_speed_cols:
            try:
                # 提取高度数字
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
        valid_mask = (v1 > 0.5) & (v2 > 0.5)  # 风速大于0.5m/s
        
        self.data = self.data[valid_mask].copy()
        v1, v2 = v1[valid_mask], v2[valid_mask]
        
        # 计算风切变系数
        self.data['wind_shear_alpha'] = np.log(v2 / v1) / np.log(h2 / h1)
        
        print(f"✓ 风切变计算完成，使用 {h1}m 和 {h2}m 高度")
        print(f"  有效数据: {len(self.data)} 条")
        print(f"  风切变范围: {self.data['wind_shear_alpha'].min():.3f} ~ {self.data['wind_shear_alpha'].max():.3f}")
        print(f"  风切变均值: {self.data['wind_shear_alpha'].mean():.3f}")
        
        return h1, h2
    
    def determine_day_night(self):
        """确定昼夜分类"""
        print("☀️🌙 确定昼夜分类...")
        
        # 提取小时信息
        self.data['hour'] = self.data['datetime'].dt.hour
        
        # 简单的昼夜划分（6:00-18:00为白天）
        # 可以根据实际地理位置和季节调整
        day_start, day_end = 6, 18
        
        self.data['is_daytime'] = ((self.data['hour'] >= day_start) & 
                                  (self.data['hour'] < day_end))
        
        day_count = self.data['is_daytime'].sum()
        night_count = len(self.data) - day_count
        
        print(f"✓ 昼夜分类完成:")
        print(f"  白天 ({day_start}:00-{day_end}:00): {day_count} 条")
        print(f"  夜间: {night_count} 条")
        
        return day_start, day_end
    
    def analyze_wind_shear_diurnal_pattern(self):
        """分析风切变的日变化模式"""
        print("📈 分析风切变日变化模式...")
        
        # 计算每小时的风切变统计
        hourly_shear = self.data.groupby('hour')['wind_shear_alpha'].agg(['mean', 'std', 'count']).reset_index()
        
        # 绘制风切变日变化
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('风切变系数的日变化特征分析', fontsize=16, fontweight='bold')
        
        # 1. 风切变日变化曲线
        ax1 = axes[0, 0]
        ax1.plot(hourly_shear['hour'], hourly_shear['mean'], 'b-', linewidth=2, marker='o')
        ax1.fill_between(hourly_shear['hour'], 
                        hourly_shear['mean'] - hourly_shear['std'],
                        hourly_shear['mean'] + hourly_shear['std'], 
                        alpha=0.3)
        ax1.axhspan(self.data['wind_shear_alpha'].quantile(0.5), 
                   self.data['wind_shear_alpha'].max(), alpha=0.2, color='orange', label='高风切变区间')
        ax1.axhspan(self.data['wind_shear_alpha'].min(),
                   self.data['wind_shear_alpha'].quantile(0.5), alpha=0.2, color='blue', label='低风切变区间')
        
        ax1.set_xlabel('小时')
        ax1.set_ylabel('风切变系数 α')
        ax1.set_title('风切变系数日变化')
        ax1.grid(True, alpha=0.3)
        ax1.legend()
        ax1.set_xticks(range(0, 24, 2))
        
        # 2. 昼夜风切变分布对比
        ax2 = axes[0, 1]
        day_shear = self.data[self.data['is_daytime']]['wind_shear_alpha']
        night_shear = self.data[~self.data['is_daytime']]['wind_shear_alpha']
        
        ax2.hist(day_shear, bins=50, alpha=0.6, label=f'白天 (μ={day_shear.mean():.3f})', color='orange')
        ax2.hist(night_shear, bins=50, alpha=0.6, label=f'夜间 (μ={night_shear.mean():.3f})', color='navy')
        ax2.set_xlabel('风切变系数 α')
        ax2.set_ylabel('频次')
        ax2.set_title('昼夜风切变分布对比')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # 3. 风切变与功率关系
        ax3 = axes[1, 0]
        ax3.scatter(self.data['wind_shear_alpha'], self.data['power'], alpha=0.3, s=10)
        
        # 计算分组平均值
        shear_bins = np.linspace(self.data['wind_shear_alpha'].min(), 
                                self.data['wind_shear_alpha'].max(), 20)
        self.data['shear_bin'] = pd.cut(self.data['wind_shear_alpha'], bins=shear_bins)
        bin_stats = self.data.groupby('shear_bin')['power'].agg(['mean', 'count']).reset_index()
        bin_centers = [(interval.left + interval.right) / 2 for interval in bin_stats['shear_bin']]
        
        ax3.plot(bin_centers, bin_stats['mean'], 'r-', linewidth=2, label='分组平均值')
        ax3.set_xlabel('风切变系数 α')
        ax3.set_ylabel('功率 (MW)')
        ax3.set_title('风切变与功率关系')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        
        # 4. 相关性分析表
        ax4 = axes[1, 1]
        ax4.axis('tight')
        ax4.axis('off')
        
        # 计算相关性统计
        corr_stats = []
        corr_stats.append(['整体', f"{self.data['wind_shear_alpha'].corr(self.data['power']):.3f}"])
        corr_stats.append(['白天', f"{day_shear.corr(self.data[self.data['is_daytime']]['power']):.3f}"])
        corr_stats.append(['夜间', f"{night_shear.corr(self.data[~self.data['is_daytime']]['power']):.3f}"])
        
        # 添加统计信息
        stats_data = [
            ['指标', '白天', '夜间', '差异'],
            ['样本数', f"{len(day_shear)}", f"{len(night_shear)}", f"{len(day_shear)-len(night_shear)}"],
            ['风切变均值', f"{day_shear.mean():.3f}", f"{night_shear.mean():.3f}", f"{day_shear.mean()-night_shear.mean():.3f}"],
            ['风切变标准差', f"{day_shear.std():.3f}", f"{night_shear.std():.3f}", f"{day_shear.std()-night_shear.std():.3f}"],
            ['功率均值(MW)', f"{self.data[self.data['is_daytime']]['power'].mean():.1f}", 
             f"{self.data[~self.data['is_daytime']]['power'].mean():.1f}",
             f"{self.data[self.data['is_daytime']]['power'].mean()-self.data[~self.data['is_daytime']]['power'].mean():.1f}"]
        ]
        
        table = ax4.table(cellText=stats_data,
                         cellLoc='center',
                         loc='center')
        table.auto_set_font_size(False)
        table.set_fontsize(9)
        table.scale(1.2, 1.5)
        ax4.set_title('昼夜风切变统计对比', pad=20)
        
        plt.tight_layout()
        plt.savefig(f"{self.save_path}/wind_shear_diurnal_analysis.png", dpi=300, bbox_inches='tight')
        plt.show()
        
        return hourly_shear, day_shear.mean(), night_shear.mean()
    
    def create_shear_diurnal_classification(self):
        """创建风切变-昼夜组合分类"""
        print("🔄 创建风切变-昼夜组合分类...")
        
        # 确定风切变阈值（使用中位数）
        self.wind_shear_threshold = self.data['wind_shear_alpha'].median()
        print(f"风切变阈值（中位数）: {self.wind_shear_threshold:.3f}")
        
        # 创建高/低风切变标识
        self.data['high_shear'] = self.data['wind_shear_alpha'] > self.wind_shear_threshold
        
        # 创建组合分类
        conditions = [
            self.data['high_shear'] & self.data['is_daytime'],      # 高风切变 + 白天
            ~self.data['high_shear'] & ~self.data['is_daytime'],   # 低风切变 + 夜间
            self.data['high_shear'] & ~self.data['is_daytime'],    # 高风切变 + 夜间
            ~self.data['high_shear'] & self.data['is_daytime']     # 低风切变 + 白天
        ]
        
        choices = [
            'high_shear_day',     # 高风切变白天
            'low_shear_night',    # 低风切变夜间
            'high_shear_night',   # 高风切变夜间
            'low_shear_day'       # 低风切变白天
        ]
        
        self.data['shear_diurnal_class'] = np.select(conditions, choices, default='unknown')
        
        # 统计各分类的数量和特征
        class_stats = self.data.groupby('shear_diurnal_class').agg({
            'power': ['count', 'mean', 'std'],
            'wind_shear_alpha': ['mean', 'std']
        }).round(3)
        
        print("\n📊 风切变-昼夜分类统计:")
        print("=" * 80)
        for class_name in choices:
            if class_name in class_stats.index:
                count = class_stats.loc[class_name, ('power', 'count')]
                power_mean = class_stats.loc[class_name, ('power', 'mean')]
                power_std = class_stats.loc[class_name, ('power', 'std')]
                shear_mean = class_stats.loc[class_name, ('wind_shear_alpha', 'mean')]
                shear_std = class_stats.loc[class_name, ('wind_shear_alpha', 'std')]
                percentage = count / len(self.data) * 100
                
                print(f"{class_name}:")
                print(f"  样本数: {count} ({percentage:.1f}%)")
                print(f"  功率: {power_mean:.1f}±{power_std:.1f} MW")
                print(f"  风切变: {shear_mean:.3f}±{shear_std:.3f}")
                print("-" * 50)
        
        # 可视化分类结果
        self.visualize_classification()
        
        return class_stats
    
    def visualize_classification(self):
        """可视化分类结果"""
        print("📊 可视化分类结果...")
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('风切变-昼夜分类结果可视化', fontsize=16, fontweight='bold')
        
        # 1. 分类散点图
        ax1 = axes[0, 0]
        classes = self.data['shear_diurnal_class'].unique()
        colors = ['red', 'blue', 'orange', 'green']
        
        for i, class_name in enumerate(classes):
            if class_name != 'unknown':
                class_data = self.data[self.data['shear_diurnal_class'] == class_name]
                ax1.scatter(class_data['wind_shear_alpha'], class_data['power'], 
                           alpha=0.6, s=20, label=class_name, color=colors[i % len(colors)])
        
        ax1.axvline(x=self.wind_shear_threshold, color='black', linestyle='--', 
                   alpha=0.7, label=f'风切变阈值={self.wind_shear_threshold:.3f}')
        ax1.set_xlabel('风切变系数 α')
        ax1.set_ylabel('功率 (MW)')
        ax1.set_title('风切变-功率分类散点图')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # 2. 各分类功率分布
        ax2 = axes[0, 1]
        power_data_by_class = []
        class_labels = []
        
        for class_name in ['high_shear_day', 'low_shear_night', 'high_shear_night', 'low_shear_day']:
            if class_name in self.data['shear_diurnal_class'].values:
                power_data = self.data[self.data['shear_diurnal_class'] == class_name]['power']
                power_data_by_class.append(power_data)
                class_labels.append(class_name.replace('_', '\n'))
        
        ax2.boxplot(power_data_by_class, labels=class_labels)
        ax2.set_ylabel('功率 (MW)')
        ax2.set_title('各分类功率分布')
        ax2.tick_params(axis='x', rotation=0)
        ax2.grid(True, alpha=0.3)
        
        # 3. 时间序列模式
        ax3 = axes[1, 0]
        hourly_class = self.data.groupby(['hour', 'shear_diurnal_class']).size().unstack(fill_value=0)
        hourly_class_pct = hourly_class.div(hourly_class.sum(axis=1), axis=0) * 100
        
        if not hourly_class_pct.empty:
            hourly_class_pct.plot(kind='area', stacked=True, ax=ax3, alpha=0.7)
            ax3.set_xlabel('小时')
            ax3.set_ylabel('百分比 (%)')
            ax3.set_title('分类的日变化模式')
            ax3.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
            ax3.grid(True, alpha=0.3)
        
        # 4. 分类特征对比雷达图
        ax4 = axes[1, 1]
        
        # 计算各分类的标准化特征
        class_features = self.data.groupby('shear_diurnal_class').agg({
            'power': 'mean',
            'wind_shear_alpha': 'mean'
        })
        
        # 添加第一个风速列作为参考
        wind_speed_cols = [col for col in self.data.columns if 'wind_speed' in col and col.startswith('obs_')]
        if wind_speed_cols:
            main_wind_col = wind_speed_cols[0]
            class_features[main_wind_col] = self.data.groupby('shear_diurnal_class')[main_wind_col].mean()
        
        # 标准化到0-1范围
        class_features_norm = (class_features - class_features.min()) / (class_features.max() - class_features.min())
        
        # 绘制简化的特征对比
        if not class_features_norm.empty:
            class_features_norm.plot(kind='bar', ax=ax4, alpha=0.7)
            ax4.set_title('各分类标准化特征对比')
            ax4.set_xlabel('分类')
            ax4.set_ylabel('标准化数值')
            ax4.tick_params(axis='x', rotation=45)
            ax4.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
            ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(f"{self.save_path}/shear_diurnal_classification.png", dpi=300, bbox_inches='tight')
        plt.show()
    
    def prepare_classification_groups(self, min_samples=300):
        """按分类分组数据"""
        print(f"📊 按分类分组数据 (最小样本数: {min_samples})...")
        
        class_counts = self.data['shear_diurnal_class'].value_counts()
        print(f"所有分类样本数: {dict(class_counts)}")
        
        # 只选择样本数足够的分类
        valid_classes = class_counts[class_counts >= min_samples].index.tolist()
        valid_classes = [cls for cls in valid_classes if cls != 'unknown']
        
        print(f"样本数足够的分类: {valid_classes}")
        
        for class_name in valid_classes:
            class_data = self.data[self.data['shear_diurnal_class'] == class_name].copy()
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
    
    def train_classification_models(self):
        """为每种分类训练独立的预测模型"""
        print("🚀 训练分类模型...")
        
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
                          'high_shear', 'shear_diurnal_class', 'shear_bin']
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
        
        print(f"\n✓ 共训练了 {len(self.models)} 个分类模型")
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
        fig.suptitle('风切变-昼夜分类模型性能对比', fontsize=16, fontweight='bold')
        
        # 1. R² 性能对比
        ax1 = axes[0, 0]
        bars1 = ax1.bar(range(len(class_names)), r2_values, color='skyblue', alpha=0.7)
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
        bars2 = ax2.bar(range(len(class_names)), rmse_values, color='lightcoral', alpha=0.7)
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
        bars3 = ax3.bar(range(len(class_names)), sample_counts, color='lightgreen', alpha=0.7)
        ax3.set_xticks(range(len(class_names)))
        ax3.set_xticklabels([name.replace('_', '\n') for name in class_names], rotation=0)
        ax3.set_ylabel('样本数量')
        ax3.set_title('各分类样本分布')
        
        for bar, count in zip(bars3, sample_counts):
            height = bar.get_height()
            ax3.text(bar.get_x() + bar.get_width()/2., height + height*0.01,
                    f'{count}', ha='center', va='bottom', fontweight='bold')
        
        # 4. 功率均值对比
        ax4 = axes[1, 0]
        bars4 = ax4.bar(range(len(class_names)), power_means, color='gold', alpha=0.7)
        ax4.set_xticks(range(len(class_names)))
        ax4.set_xticklabels([name.replace('_', '\n') for name in class_names], rotation=0)
        ax4.set_ylabel('平均功率 (MW)')
        ax4.set_title('各分类平均功率')
        
        for bar, power in zip(bars4, power_means):
            height = bar.get_height()
            ax4.text(bar.get_x() + bar.get_width()/2., height + height*0.01,
                    f'{power:.1f}', ha='center', va='bottom', fontweight='bold')
        
        # 5. 预测效果散点图（选择最好的模型）
        best_class = class_names[np.argmax(r2_values)]
        ax5 = axes[1, 1]
        
        y_test = self.results[best_class]['y_test']
        y_pred = self.results[best_class]['y_pred_test']
        
        ax5.scatter(y_test, y_pred, alpha=0.5, s=20)
        min_val = min(y_test.min(), y_pred.min())
        max_val = max(y_test.max(), y_pred.max())
        ax5.plot([min_val, max_val], [min_val, max_val], 'r--', alpha=0.8)
        
        ax5.set_xlabel('实际功率 (MW)')
        ax5.set_ylabel('预测功率 (MW)')
        ax5.set_title(f'最佳模型预测效果\n({best_class.replace("_", " ")})')
        ax5.grid(True, alpha=0.3)
        
        r2_best = self.results[best_class]['r2_test']
        ax5.text(0.05, 0.95, f'R² = {r2_best:.3f}', transform=ax5.transAxes,
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        # 6. 风切变-性能关系
        ax6 = axes[1, 2]
        scatter = ax6.scatter(shear_means, r2_values, c=rmse_values, 
                            s=[count/50 for count in sample_counts], 
                            cmap='viridis_r', alpha=0.7)
        
        # 添加分类标签
        for i, class_name in enumerate(class_names):
            ax6.annotate(class_name.replace('_', '\n'), 
                        (shear_means[i], r2_values[i]),
                        xytext=(5, 5), textcoords='offset points', fontsize=8)
        
        ax6.set_xlabel('平均风切变系数')
        ax6.set_ylabel('R² Score')
        ax6.set_title('风切变与性能关系\n(点大小=样本数)')
        ax6.grid(True, alpha=0.3)
        
        # 添加颜色条
        cbar = plt.colorbar(scatter, ax=ax6)
        cbar.set_label('RMSE (MW)')
        
        plt.tight_layout()
        plt.savefig(f"{self.save_path}/shear_diurnal_performance.png", dpi=300, bbox_inches='tight')
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
            print(f"  {i+1}. {row['classification']}: R²={row['r2']:.3f}, "
                  f"RMSE={row['rmse']:.1f}MW, 样本={row['samples']}, "
                  f"风切变={row['shear_mean']:.3f}")
        
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
        fig.suptitle('风切变-昼夜分类SHAP重要性对比', fontsize=16, fontweight='bold')
        
        # 1. Top特征重要性对比
        top_n = 15
        top_features = shap_importance_df.head(top_n)
        
        ax1 = axes[0, 0]
        class_names = [col.replace('_importance', '') for col in importance_cols]
        x = np.arange(len(top_features))
        width = 0.8 / len(class_names)
        
        colors = ['red', 'blue', 'orange', 'green']
        for i, class_name in enumerate(class_names):
            col = f'{class_name}_importance'
            offset = (i - len(class_names)/2 + 0.5) * width
            ax1.barh(x + offset, top_features[col], width, 
                    label=class_name.replace('_', ' '), alpha=0.7, color=colors[i % len(colors)])
        
        ax1.set_yticks(x)
        ax1.set_yticklabels(top_features['feature'], fontsize=8)
        ax1.set_xlabel('SHAP重要性')
        ax1.set_title(f'Top {top_n} 特征重要性对比')
        ax1.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        ax1.grid(True, alpha=0.3)
        
        # 2. 特征重要性热力图
        ax2 = axes[0, 1]
        top_20_features = shap_importance_df.head(20)
        heatmap_data = top_20_features[importance_cols].T
        heatmap_data.columns = top_20_features['feature']
        heatmap_data.index = [idx.replace('_importance', '').replace('_', ' ') for idx in heatmap_data.index]
        
        sns.heatmap(heatmap_data, annot=True, fmt='.3f', cmap='YlOrRd', 
                   ax=ax2, cbar_kws={'label': 'SHAP重要性'})
        ax2.set_title('Top 20 特征重要性热力图')
        ax2.set_xlabel('特征')
        ax2.set_ylabel('分类')
        
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
                cat_data = {'category': category}
                for class_name in class_names:
                    col = f'{class_name}_importance'
                    if col in shap_importance_df.columns:
                        cat_features = shap_importance_df[shap_importance_df['feature'].isin(features)]
                        cat_data[class_name] = cat_features[col].sum()
                    else:
                        cat_data[class_name] = 0
                category_importance = pd.concat([category_importance, pd.DataFrame([cat_data])], ignore_index=True)
        
        # 绘制分类重要性对比
        x = np.arange(len(category_importance))
        width = 0.8 / len(class_names)
        
        for i, class_name in enumerate(class_names):
            offset = (i - len(class_names)/2 + 0.5) * width
            if class_name in category_importance.columns:
                ax3.bar(x + offset, category_importance[class_name], width, 
                       label=class_name.replace('_', ' '), alpha=0.7, color=colors[i % len(colors)])
        
        ax3.set_xticks(x)
        ax3.set_xticklabels(category_importance['category'])
        ax3.set_ylabel('累计SHAP重要性')
        ax3.set_title('按变量类型分组的重要性对比')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        
        # 4. 分类间重要性差异分析
        ax4 = axes[1, 1]
        
        if len(class_names) >= 2:
            # 比较高风切变白天 vs 低风切变夜间（如果存在）
            target_classes = ['high_shear_day', 'low_shear_night']
            available_classes = [cls for cls in target_classes if f'{cls}_importance' in shap_importance_df.columns]
            
            if len(available_classes) >= 2:
                class1, class2 = available_classes[0], available_classes[1]
                col1, col2 = f'{class1}_importance', f'{class2}_importance'
                
                diff = shap_importance_df[col1] - shap_importance_df[col2]
                shap_importance_df['diff'] = diff
                diff_sorted = shap_importance_df.sort_values('diff', ascending=True)
                
                colors_diff = ['red' if x < 0 else 'blue' for x in diff_sorted['diff']]
                ax4.barh(range(len(diff_sorted)), diff_sorted['diff'], color=colors_diff, alpha=0.6)
                ax4.set_yticks(range(len(diff_sorted)))
                ax4.set_yticklabels(diff_sorted['feature'], fontsize=6)
                ax4.set_xlabel(f'重要性差异\n({class1.replace("_", " ")} - {class2.replace("_", " ")})')
                ax4.set_title('主要分类重要性差异')
                ax4.axvline(x=0, color='black', linestyle='--', alpha=0.5)
                ax4.grid(True, alpha=0.3)
            else:
                ax4.text(0.5, 0.5, '需要至少2种分类\n进行差异分析', 
                        ha='center', va='center', transform=ax4.transAxes)
                ax4.set_title('重要性差异分析')
        else:
            ax4.text(0.5, 0.5, '需要至少2种分类\n进行差异分析', 
                    ha='center', va='center', transform=ax4.transAxes)
            ax4.set_title('重要性差异分析')
        
        plt.tight_layout()
        plt.savefig(f"{self.save_path}/shear_diurnal_shap_comparison.png", dpi=300, bbox_inches='tight')
        plt.show()
        
        # 保存重要性对比数据
        shap_importance_df.to_csv(f"{self.save_path}/shear_diurnal_shap_importance.csv", index=False)
        print("✓ SHAP重要性对比数据已保存")
        
        return shap_importance_df
    
    def save_models_and_results(self):
        """保存模型和结果"""
        print("💾 保存模型和结果...")
        
        # 保存各分类模型
        for class_name, model in self.models.items():
            model_path = f"{self.save_path}/lightgbm_model_{class_name}.pkl"
            joblib.dump(model, model_path)
            print(f"✓ {class_name}模型已保存: {model_path}")
        
        # 保存特征名称
        feature_names_path = f"{self.save_path}/feature_names.pkl"
        joblib.dump(self.feature_names, feature_names_path)
        print(f"✓ 特征名称已保存: {feature_names_path}")
        
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
        
        summary_path = f"{self.save_path}/shear_diurnal_results_summary.pkl"
        joblib.dump(results_summary, summary_path)
        print(f"✓ 结果摘要已保存: {summary_path}")
        
        # 保存风切变阈值
        threshold_info = {
            'wind_shear_threshold': self.wind_shear_threshold,
            'day_start': 6,
            'day_end': 18
        }
        
        threshold_path = f"{self.save_path}/classification_thresholds.pkl"
        joblib.dump(threshold_info, threshold_path)
        print(f"✓ 分类阈值已保存: {threshold_path}")
        
        return results_summary, threshold_info
    
    def run_full_analysis(self):
        """运行完整的风切变-昼夜分析流程"""
        print("=" * 70)
        print("🌪️ 基于风切变-昼夜分类的风电预测分析")
        print("=" * 70)
        
        try:
            # 1. 加载和预处理数据
            self.load_and_prepare_data()
            
            # 2. 计算风切变系数
            h1, h2 = self.calculate_wind_shear()
            
            # 3. 确定昼夜分类
            day_start, day_end = self.determine_day_night()
            
            # 4. 分析风切变日变化模式
            hourly_shear, day_shear_mean, night_shear_mean = self.analyze_wind_shear_diurnal_pattern()
            
            # 5. 创建组合分类
            class_stats = self.create_shear_diurnal_classification()
            
            # 6. 按分类分组
            self.prepare_classification_groups(min_samples=300)
            
            # 7. 训练分类模型
            self.train_classification_models()
            
            # 8. 计算SHAP值
            self.calculate_shap_values()
            
            # 9. 绘制性能对比
            performance_df = self.plot_performance_comparison()
            
            # 10. 绘制SHAP对比
            shap_comparison = self.plot_shap_comparison()
            
            # 11. 保存模型和结果
            results_summary, threshold_info = self.save_models_and_results()
            
            print("\n" + "=" * 70)
            print("🎉 风切变-昼夜分析完成！")
            print("=" * 70)
            
            print("📊 主要发现:")
            print(f"  风切变计算: 使用 {h1}m 和 {h2}m 高度数据")
            print(f"  风切变阈值: {self.wind_shear_threshold:.3f}")
            print(f"  昼夜划分: {day_start}:00-{day_end}:00为白天")
            print(f"  白天平均风切变: {day_shear_mean:.3f}")
            print(f"  夜间平均风切变: {night_shear_mean:.3f}")
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
                    print("  → 风切变-昼夜分类建模很有价值，性能差异明显")
                else:
                    print("  → 各分类预测性能相近，可考虑简化分类策略")
            
            print(f"\n📁 结果文件保存在: {self.save_path}")
            print("  - wind_shear_diurnal_analysis.png: 风切变日变化分析")
            print("  - shear_diurnal_classification.png: 分类结果可视化")
            print("  - shear_diurnal_performance.png: 模型性能对比")
            print("  - shear_diurnal_shap_comparison.png: SHAP重要性对比")
            
            # 分析关键洞察
            print(f"\n🔍 关键洞察:")
            print(f"  风切变与昼夜的关系:")
            if day_shear_mean > night_shear_mean:
                print(f"    - 白天风切变更强 ({day_shear_mean:.3f} vs {night_shear_mean:.3f})")
                print(f"    - 符合边界层理论：白天不稳定导致更强风切变")
            else:
                print(f"    - 夜间风切变更强 ({night_shear_mean:.3f} vs {day_shear_mean:.3f})")
                print(f"    - 可能存在特殊地形或气候影响")
            
            if len(self.models) >= 2:
                print("  不同分类条件下的预测特征:")
                for class_name in self.models.keys():
                    r2 = results_summary[class_name]['r2_test']
                    samples = results_summary[class_name]['sample_count']
                    shear_mean = results_summary[class_name]['shear_mean']
                    
                    if r2 > 0.8:
                        perf_level = "优秀"
                    elif r2 > 0.6:
                        perf_level = "良好"
                    elif r2 > 0.4:
                        perf_level = "一般"
                    else:
                        perf_level = "较差"
                    
                    print(f"    - {class_name}: {perf_level}预测性能 (R²={r2:.3f}), "
                          f"风切变={shear_mean:.3f}, 样本{samples}条")
            
            return True
            
        except Exception as e:
            print(f"❌ 分析过程出错: {str(e)}")
            import traceback
            traceback.print_exc()
            return False

if __name__ == "__main__":
    # 配置路径
    DATA_PATH = "/Users/xiaxin/work/WindForecast_Project/01_Data/processed/imputed_data/changma_imputed_complete.csv"
    SAVE_PATH = "/Users/xiaxin/work/WindForecast_Project/03_Results/wind_shear_diurnal_analysis"
    
    # 创建保存目录
    import os
    os.makedirs(SAVE_PATH, exist_ok=True)
    
    # 创建分析器并运行
    analyzer = WindShearDiurnalAnalyzer(DATA_PATH, SAVE_PATH)
    success = analyzer.run_full_analysis()
    
    if success:
        print("\n🎯 风切变-昼夜分析成功完成！")
        print("\n💡 后续研究建议:")
        print("  1. 对比不同分类策略的优劣")
        print("  2. 研究风切变与稳定度的关系")
        print("  3. 分析季节性对风切变模式的影响")
        print("  4. 开发基于风切变的动态预测模型")
        print("  5. 验证分类策略在不同风电场的适用性")
    else:
        print("\n⚠️ 分析失败，请检查错误信息和数据路径")