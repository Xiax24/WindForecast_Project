#!/usr/bin/env python3
"""
基于三级风切变分组与昼夜结合的风电预测与SHAP重要性分析
分类策略：弱切变(α<0.2) / 中等切变(0.2≤α<0.3) / 强切变(α≥0.3) × 昼夜
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

class ThreeGroupWindShearAnalyzer:
    def __init__(self, data_path, save_path):
        self.data_path = data_path
        self.save_path = save_path
        self.data = None
        self.groups = {}
        self.feature_names = None
        self.models = {}
        self.shap_explainers = {}
        self.results = {}
        
        # 三级风切变阈值（基于大气边界层物理特征）
        self.shear_thresholds = {
            'weak_upper': 0.1,      # α < 0.2: 弱切变
            'moderate_upper': 0.3,  # 0.2 ≤ α < 0.3: 中等切变
            # α ≥ 0.3: 强切变
        }
        
        # 物理机制对应关系
        self.shear_physics = {
            'weak': {
                'description': '弱切变/风速变化小',
                'day_cause': '强混合、不稳定层结',
                'night_cause': '高湍流（如风速大、无逆温）'
            },
            'moderate': {
                'description': '中等切变',
                'day_cause': '常见日间背景状态，偏中性',
                'night_cause': '消弱稳定，或逆温未完全建立'
            },
            'strong': {
                'description': '强切变/层结抑制',
                'day_cause': '非常稳定大气（少见）',
                'night_cause': '夜间逆温显著、摩擦层强层结'
            }
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
    
    def classify_three_group_shear(self):
        """基于三级阈值分类风切变"""
        print("🔄 基于三级阈值分类风切变...")
        
        alpha = self.data['wind_shear_alpha']
        
        # 定义三级分类条件
        conditions = [
            alpha < self.shear_thresholds['weak_upper'],                           # α < 0.2: 弱切变
            (alpha >= self.shear_thresholds['weak_upper']) & 
            (alpha < self.shear_thresholds['moderate_upper']),                     # 0.2 ≤ α < 0.3: 中等切变
            alpha >= self.shear_thresholds['moderate_upper']                       # α ≥ 0.3: 强切变
        ]
        
        choices = ['weak', 'moderate', 'strong']
        
        self.data['shear_group'] = np.select(conditions, choices, default='unknown')
        
        # 统计各切变组别
        shear_counts = self.data['shear_group'].value_counts()
        print(f"\n📊 三级风切变分类统计:")
        print(f"  弱切变 (α < {self.shear_thresholds['weak_upper']}): {shear_counts.get('weak', 0)} 条")
        print(f"  中等切变 ({self.shear_thresholds['weak_upper']} ≤ α < {self.shear_thresholds['moderate_upper']}): {shear_counts.get('moderate', 0)} 条")
        print(f"  强切变 (α ≥ {self.shear_thresholds['moderate_upper']}): {shear_counts.get('strong', 0)} 条")
        
        # 分析各组别的风切变统计
        shear_stats = self.data.groupby('shear_group')['wind_shear_alpha'].agg(['count', 'mean', 'std', 'min', 'max'])
        print(f"\n各切变组别统计:")
        print(shear_stats.round(3))
        
        return shear_counts, shear_stats
    
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
    
    def create_three_group_classification(self):
        """创建三级风切变-昼夜组合分类"""
        print("🔄 创建三级风切变-昼夜组合分类...")
        
        # 创建组合分类
        self.data['three_group_class'] = self.data['shear_group'].astype(str) + '_' + \
                                       self.data['is_daytime'].map({True: 'day', False: 'night'})
        
        # 统计各分类
        class_stats = self.data.groupby('three_group_class').agg({
            'power': ['count', 'mean', 'std'],
            'wind_shear_alpha': ['mean', 'std'],
            'hour': 'mean'
        }).round(3)
        
        print("\n📊 三级风切变-昼夜分类统计:")
        print("=" * 80)
        for class_name in class_stats.index:
            if 'unknown' not in class_name:
                count = class_stats.loc[class_name, ('power', 'count')]
                power_mean = class_stats.loc[class_name, ('power', 'mean')]
                power_std = class_stats.loc[class_name, ('power', 'std')]
                shear_mean = class_stats.loc[class_name, ('wind_shear_alpha', 'mean')]
                shear_std = class_stats.loc[class_name, ('wind_shear_alpha', 'std')]
                avg_hour = class_stats.loc[class_name, ('hour', 'mean')]
                percentage = count / len(self.data) * 100
                
                # 获取物理解释
                shear_type = class_name.split('_')[0]
                period = class_name.split('_')[1]
                
                print(f"{class_name}:")
                print(f"  样本数: {count} ({percentage:.1f}%)")
                print(f"  功率: {power_mean:.1f}±{power_std:.1f} MW")
                print(f"  风切变: {shear_mean:.3f}±{shear_std:.3f}")
                print(f"  平均时间: {avg_hour:.1f}时")
                
                # 添加物理解释
                if shear_type in self.shear_physics:
                    cause_key = f'{period}_cause'
                    if cause_key in self.shear_physics[shear_type]:
                        print(f"  物理成因: {self.shear_physics[shear_type][cause_key]}")
                
                print("-" * 50)
        
        # 分析物理合理性
        self.analyze_three_group_physics()
        
        return class_stats
    
    def analyze_three_group_physics(self):
        """分析三级分类的物理合理性"""
        print("🔬 分析三级分类的物理合理性...")
        
        # 统计各组合的数量
        combinations = {}
        for shear in ['weak', 'moderate', 'strong']:
            for period in ['day', 'night']:
                class_name = f'{shear}_{period}'
                count = len(self.data[self.data['three_group_class'] == class_name])
                combinations[class_name] = count
        
        total = len(self.data)
        
        print(f"\n物理合理性分析:")
        print(f"  弱切变-白天 (强混合): {combinations.get('weak_day', 0)} ({combinations.get('weak_day', 0)/total*100:.1f}%)")
        print(f"  弱切变-夜间 (高湍流): {combinations.get('weak_night', 0)} ({combinations.get('weak_night', 0)/total*100:.1f}%)")
        print(f"  中等切变-白天 (中性): {combinations.get('moderate_day', 0)} ({combinations.get('moderate_day', 0)/total*100:.1f}%)")
        print(f"  中等切变-夜间 (过渡): {combinations.get('moderate_night', 0)} ({combinations.get('moderate_night', 0)/total*100:.1f}%)")
        print(f"  强切变-白天 (异常): {combinations.get('strong_day', 0)} ({combinations.get('strong_day', 0)/total*100:.1f}%)")
        print(f"  强切变-夜间 (层结): {combinations.get('strong_night', 0)} ({combinations.get('strong_night', 0)/total*100:.1f}%)")
        
        # 期望的物理分布
        expected_dominant = ['moderate_day', 'strong_night']  # 最常见的组合
        expected_rare = ['strong_day']  # 最少见的组合
        
        print(f"\n期望物理分布验证:")
        for combo in expected_dominant:
            if combo in combinations:
                pct = combinations[combo] / total * 100
                print(f"  {combo} (期望常见): {pct:.1f}% - {'✓ 符合' if pct > 15 else '⚠ 偏少'}")
        
        for combo in expected_rare:
            if combo in combinations:
                pct = combinations[combo] / total * 100
                print(f"  {combo} (期望罕见): {pct:.1f}% - {'✓ 符合' if pct < 10 else '⚠ 偏多'}")
        
        return combinations
    
    def visualize_three_group_classification(self):
        """可视化三级分类结果"""
        print("📊 可视化三级分类结果...")
        
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle('三级风切变-昼夜分类分析', fontsize=16, fontweight='bold')
        
        # 1. 风切变分布与三级阈值
        ax1 = axes[0, 0]
        alpha_values = self.data['wind_shear_alpha']
        ax1.hist(alpha_values, bins=50, alpha=0.7, color='skyblue', density=True)
        
        # 标记三级阈值
        ax1.axvline(x=self.shear_thresholds['weak_upper'], color='green', linestyle='--', 
                   linewidth=2, label=f'弱切变阈值 (α={self.shear_thresholds["weak_upper"]})')
        ax1.axvline(x=self.shear_thresholds['moderate_upper'], color='orange', linestyle='--',
                   linewidth=2, label=f'强切变阈值 (α={self.shear_thresholds["moderate_upper"]})')
        
        # 添加区域标注
        ax1.axvspan(-0.5, self.shear_thresholds['weak_upper'], alpha=0.2, color='green', label='弱切变区')
        ax1.axvspan(self.shear_thresholds['weak_upper'], self.shear_thresholds['moderate_upper'], 
                   alpha=0.2, color='orange', label='中等切变区')
        ax1.axvspan(self.shear_thresholds['moderate_upper'], 1.0, alpha=0.2, color='red', label='强切变区')
        
        ax1.set_xlabel('风切变系数 α')
        ax1.set_ylabel('密度')
        ax1.set_title('风切变分布与三级阈值')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # 2. 三级切变日变化模式
        ax2 = axes[0, 1]
        hourly_shear = self.data.groupby(['hour', 'shear_group']).size().unstack(fill_value=0)
        hourly_shear_pct = hourly_shear.div(hourly_shear.sum(axis=1), axis=0) * 100
        
        if not hourly_shear_pct.empty:
            colors_shear = {'weak': 'green', 'moderate': 'orange', 'strong': 'red'}
            hourly_shear_pct.plot(kind='area', stacked=True, ax=ax2, alpha=0.7, 
                                 color=[colors_shear.get(col, 'gray') for col in hourly_shear_pct.columns])
            ax2.set_xlabel('小时')
            ax2.set_ylabel('百分比 (%)')
            ax2.set_title('三级切变的日变化模式')
            ax2.legend(title='切变强度')
            ax2.grid(True, alpha=0.3)
        
        # 3. 三级分类散点图
        ax3 = axes[0, 2]
        classes = self.data['three_group_class'].unique()
        colors = {'weak': 'green', 'moderate': 'orange', 'strong': 'red'}
        markers = {'day': 'o', 'night': '^'}
        
        for class_name in classes:
            if 'unknown' not in class_name:
                class_data = self.data[self.data['three_group_class'] == class_name]
                shear_type = class_name.split('_')[0]
                period = class_name.split('_')[1]
                
                color = colors.get(shear_type, 'gray')
                marker = markers.get(period, 'o')
                
                ax3.scatter(class_data['wind_shear_alpha'], class_data['power'], 
                           alpha=0.6, s=20, label=class_name, color=color, marker=marker)
        
        ax3.axvline(x=self.shear_thresholds['weak_upper'], color='green', linestyle='--', alpha=0.5)
        ax3.axvline(x=self.shear_thresholds['moderate_upper'], color='orange', linestyle='--', alpha=0.5)
        ax3.set_xlabel('风切变系数 α')
        ax3.set_ylabel('功率 (MW)')
        ax3.set_title('三级分类散点图\n(圆圈=白天, 三角=夜间)')
        ax3.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        ax3.grid(True, alpha=0.3)
        
        # 4. 各分类功率箱线图
        ax4 = axes[1, 0]
        power_data_by_class = []
        class_labels = []
        
        for shear in ['weak', 'moderate', 'strong']:
            for period in ['day', 'night']:
                class_name = f'{shear}_{period}'
                if class_name in self.data['three_group_class'].values:
                    power_data = self.data[self.data['three_group_class'] == class_name]['power']
                    if len(power_data) > 0:
                        power_data_by_class.append(power_data)
                        class_labels.append(f'{shear}\n{period}')
        
        if power_data_by_class:
            bp = ax4.boxplot(power_data_by_class, labels=class_labels, patch_artist=True)
            
            # 设置箱线图颜色
            for i, patch in enumerate(bp['boxes']):
                shear_type = class_labels[i].split('\n')[0]
                color = colors.get(shear_type, 'gray')
                patch.set_facecolor(color)
                patch.set_alpha(0.7)
                
            ax4.set_ylabel('功率 (MW)')
            ax4.set_title('各三级分类功率分布')
            ax4.tick_params(axis='x', rotation=0)
            ax4.grid(True, alpha=0.3)
        
        # 5. 物理合理性饼图
        ax5 = axes[1, 1]
        
        # 计算合理性分类
        normal_physics = len(self.data[
            (self.data['three_group_class'].isin(['weak_day', 'moderate_day', 'strong_night'])) 
        ])
        transitional = len(self.data[
            (self.data['three_group_class'].isin(['weak_night', 'moderate_night']))
        ])
        unusual = len(self.data[
            (self.data['three_group_class'] == 'strong_day')
        ])
        
        physics_data = {
            '物理常见\n(弱/中-白天, 强-夜间)': normal_physics,
            '过渡状态\n(弱/中-夜间)': transitional,
            '物理异常\n(强-白天)': unusual
        }
        
        colors_pie = ['lightgreen', 'lightyellow', 'lightcoral']
        ax5.pie(physics_data.values(), labels=physics_data.keys(), colors=colors_pie,
                autopct='%1.1f%%', startangle=90)
        ax5.set_title('物理机制合理性分布')
        
        # 6. 三级切变-功率相关性分析
        ax6 = axes[1, 2]
        
        # 计算各分类的风切变-功率相关性
        corr_data = []
        for shear in ['weak', 'moderate', 'strong']:
            for period in ['day', 'night']:
                class_name = f'{shear}_{period}'
                if class_name in self.data['three_group_class'].values:
                    subset = self.data[self.data['three_group_class'] == class_name]
                    if len(subset) > 20:  # 确保有足够样本
                        corr = subset['wind_shear_alpha'].corr(subset['power'])
                        corr_data.append({
                            'class': class_name,
                            'shear_type': shear,
                            'period': period,
                            'correlation': corr,
                            'count': len(subset)
                        })
        
        if corr_data:
            corr_df = pd.DataFrame(corr_data)
            
            # 分组绘制
            x_pos = 0
            for shear in ['weak', 'moderate', 'strong']:
                shear_data = corr_df[corr_df['shear_type'] == shear]
                if len(shear_data) > 0:
                    day_data = shear_data[shear_data['period'] == 'day']
                    night_data = shear_data[shear_data['period'] == 'night']
                    
                    color = colors[shear]
                    if len(day_data) > 0:
                        ax6.bar(x_pos, day_data['correlation'].iloc[0], width=0.4, 
                               color=color, alpha=0.7, label=f'{shear}_day' if x_pos == 0 else "")
                    if len(night_data) > 0:
                        ax6.bar(x_pos + 0.4, night_data['correlation'].iloc[0], width=0.4, 
                               color=color, alpha=0.4, label=f'{shear}_night' if x_pos == 0 else "")
                    
                    x_pos += 1
            
            ax6.set_xticks([0.2, 1.2, 2.2])
            ax6.set_xticklabels(['弱切变', '中等切变', '强切变'])
            ax6.set_ylabel('相关系数')
            ax6.set_title('风切变-功率相关性\n(深色=白天, 浅色=夜间)')
            ax6.axhline(y=0, color='black', linestyle='-', alpha=0.5)
            ax6.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(f"{self.save_path}/three_group_classification.png", dpi=300, bbox_inches='tight')
        plt.show()
    
    def prepare_classification_groups(self, min_samples=200):
        """按三级分类分组数据"""
        print(f"📊 按三级分类分组数据 (最小样本数: {min_samples})...")
        
        class_counts = self.data['three_group_class'].value_counts()
        print(f"所有分类样本数: {dict(class_counts)}")
        
        # 只选择样本数足够的分类
        valid_classes = class_counts[class_counts >= min_samples].index.tolist()
        valid_classes = [cls for cls in valid_classes if 'unknown' not in cls]
        
        print(f"样本数足够的分类: {valid_classes}")
        
        for class_name in valid_classes:
            class_data = self.data[self.data['three_group_class'] == class_name].copy()
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
    
    def train_three_group_models(self):
        """为每种三级分类训练独立的预测模型"""
        print("🚀 训练三级分类模型...")
        
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
                          'shear_group', 'three_group_class']
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
        
        print(f"\n✓ 共训练了 {len(self.models)} 个三级分类模型")
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
        fig.suptitle('三级风切变-昼夜分类模型性能对比', fontsize=16, fontweight='bold')
        
        # 设置颜色映射
        color_map = {'weak': 'green', 'moderate': 'orange', 'strong': 'red'}
        
        # 1. R² 性能对比
        ax1 = axes[0, 0]
        colors = []
        for name in class_names:
            shear_type = name.split('_')[0]
            colors.append(color_map.get(shear_type, 'gray'))
        
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
        
        # 4. 切变强度性能对比
        ax4 = axes[1, 0]
        
        # 按切变强度分组
        shear_performance = {}
        for shear in ['weak', 'moderate', 'strong']:
            day_class = f'{shear}_day'
            night_class = f'{shear}_night'
            
            day_r2 = self.results.get(day_class, {}).get('r2_test', None)
            night_r2 = self.results.get(night_class, {}).get('r2_test', None)
            
            shear_performance[shear] = {'day': day_r2, 'night': night_r2}
        
        x = np.arange(3)
        width = 0.35
        
        day_r2_vals = [shear_performance[s]['day'] for s in ['weak', 'moderate', 'strong']]
        night_r2_vals = [shear_performance[s]['night'] for s in ['weak', 'moderate', 'strong']]
        
        # 只绘制有效数据
        day_r2_clean = [v if v is not None else 0 for v in day_r2_vals]
        night_r2_clean = [v if v is not None else 0 for v in night_r2_vals]
        
        ax4.bar(x - width/2, day_r2_clean, width, label='白天', alpha=0.7, color='orange')
        ax4.bar(x + width/2, night_r2_clean, width, label='夜间', alpha=0.7, color='navy')
        
        ax4.set_xticks(x)
        ax4.set_xticklabels(['弱切变', '中等切变', '强切变'])
        ax4.set_ylabel('R² Score')
        ax4.set_title('切变强度性能对比')
        ax4.legend()
        ax4.grid(True, alpha=0.3)
        
        # 5. 物理机制验证散点图
        ax5 = axes[1, 1]
        
        # 按物理合理性着色
        physical_colors = []
        for name in class_names:
            if name in ['weak_day', 'moderate_day', 'strong_night']:
                physical_colors.append('green')  # 物理合理
            elif name in ['weak_night', 'moderate_night']:
                physical_colors.append('orange')  # 过渡状态
            elif name == 'strong_day':
                physical_colors.append('red')  # 物理异常
            else:
                physical_colors.append('gray')
        
        scatter = ax5.scatter(shear_means, r2_values, c=physical_colors, s=100, alpha=0.7)
        
        # 添加分类标签
        for i, class_name in enumerate(class_names):
            ax5.annotate(class_name.replace('_', '\n'), 
                        (shear_means[i], r2_values[i]),
                        xytext=(5, 5), textcoords='offset points', fontsize=8)
        
        # 添加阈值线
        ax5.axvline(x=self.shear_thresholds['weak_upper'], color='gray', linestyle='--', alpha=0.5)
        ax5.axvline(x=self.shear_thresholds['moderate_upper'], color='gray', linestyle='--', alpha=0.5)
        
        ax5.set_xlabel('平均风切变系数 α')
        ax5.set_ylabel('R² Score')
        ax5.set_title('风切变-性能关系\n(绿=物理合理, 橙=过渡, 红=异常)')
        ax5.grid(True, alpha=0.3)
        
        # 6. 性能稳定性分析
        ax6 = axes[1, 2]
        
        # 计算训练-测试性能差异
        overfitting = []
        for class_name in class_names:
            train_r2 = self.results[class_name]['r2_train']
            test_r2 = self.results[class_name]['r2_test']
            overfitting.append(train_r2 - test_r2)
        
        bars6 = ax6.bar(range(len(class_names)), overfitting, color=colors, alpha=0.7)
        ax6.set_xticks(range(len(class_names)))
        ax6.set_xticklabels([name.replace('_', '\n') for name in class_names], rotation=45, ha='right')
        ax6.set_ylabel('过拟合程度 (训练R² - 测试R²)')
        ax6.set_title('模型稳定性分析')
        ax6.axhline(y=0.1, color='red', linestyle='--', alpha=0.7, label='过拟合警戒线')
        ax6.legend()
        ax6.grid(True, alpha=0.3)
        
        # 添加数值标签
        for bar, overfit in zip(bars6, overfitting):
            height = bar.get_height()
            ax6.text(bar.get_x() + bar.get_width()/2., height + 0.001,
                    f'{overfit:.3f}', ha='center', va='bottom', fontsize=8)
        
        plt.tight_layout()
        plt.savefig(f"{self.save_path}/three_group_performance.png", dpi=300, bbox_inches='tight')
        plt.show()
        
        # 输出性能排名
        print("\n📊 性能排名 (按R²降序):")
        performance_df = pd.DataFrame({
            'classification': class_names,
            'r2': r2_values,
            'rmse': rmse_values,
            'samples': sample_counts,
            'power_mean': power_means,
            'shear_mean': shear_means,
            'overfitting': overfitting
        }).sort_values('r2', ascending=False)
        
        for i, row in performance_df.iterrows():
            shear_type = row['classification'].split('_')[0]
            period = row['classification'].split('_')[1]
            
            # 物理合理性判断
            if row['classification'] in ['weak_day', 'moderate_day', 'strong_night']:
                physics = "合理"
            elif row['classification'] in ['weak_night', 'moderate_night']:
                physics = "过渡"
            elif row['classification'] == 'strong_day':
                physics = "异常"
            else:
                physics = "未知"
            
            print(f"  {i+1}. {row['classification']} ({shear_type}切变+{period}, {physics}): "
                  f"R²={row['r2']:.3f}, RMSE={row['rmse']:.1f}MW, "
                  f"样本={row['samples']}, α={row['shear_mean']:.3f}, "
                  f"过拟合={row['overfitting']:.3f}")
        
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
        fig, axes = plt.subplots(2, 3, figsize=(20, 12))
        fig.suptitle('三级风切变分类SHAP重要性对比', fontsize=16, fontweight='bold')
        
        # 1. Top特征重要性对比
        top_n = 15
        top_features = shap_importance_df.head(top_n)
        
        ax1 = axes[0, 0]
        class_names = [col.replace('_importance', '') for col in importance_cols]
        x = np.arange(len(top_features))
        width = 0.8 / len(class_names)
        
        color_map = {'weak': 'green', 'moderate': 'orange', 'strong': 'red'}
        
        for i, class_name in enumerate(class_names):
            col = f'{class_name}_importance'
            offset = (i - len(class_names)/2 + 0.5) * width
            
            # 根据切变强度设置颜色
            shear_type = class_name.split('_')[0]
            period = class_name.split('_')[1]
            color = color_map.get(shear_type, 'gray')
            
            # 根据昼夜设置填充样式
            if period == 'night':
                alpha = 0.6
                hatch = '//'
            else:
                alpha = 0.9
                hatch = None
            
            ax1.barh(x + offset, top_features[col], width, 
                    label=f'{class_name}', 
                    color=color, alpha=alpha, hatch=hatch)
        
        ax1.set_yticks(x)
        ax1.set_yticklabels(top_features['feature'], fontsize=8)
        ax1.set_xlabel('SHAP重要性')
        ax1.set_title(f'Top {top_n} 特征重要性对比')
        ax1.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        ax1.grid(True, alpha=0.3)
        
        # 2. 切变强度重要性对比
        ax2 = axes[0, 1]
        
        # 按切变强度分组
        shear_groups = {'weak': [], 'moderate': [], 'strong': []}
        for col in importance_cols:
            shear_type = col.replace('_importance', '').split('_')[0]
            if shear_type in shear_groups:
                shear_groups[shear_type].append(col)
        
        top_features_shear = shap_importance_df.head(20)
        x = np.arange(len(top_features_shear))
        width = 0.25
        
        for i, (shear, cols) in enumerate(shear_groups.items()):
            if cols:
                avg_importance = top_features_shear[cols].mean(axis=1)
                ax2.barh(x + i*width - width, avg_importance, width, 
                        label=f'{shear}切变', color=color_map[shear], alpha=0.7)
        
        ax2.set_yticks(x)
        ax2.set_yticklabels(top_features_shear['feature'], fontsize=8)
        ax2.set_xlabel('平均SHAP重要性')
        ax2.set_title('按切变强度分组的重要性')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # 3. 昼夜重要性对比
        ax3 = axes[0, 2]
        
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
        
        # 4. 高度风速重要性验证
        ax4 = axes[1, 0]
        
        # 分析不同高度风速的重要性
        wind_height_features = [f for f in self.feature_names if 'wind_speed' in f]
        
        if len(wind_height_features) >= 2:
            height_importance = {}
            for class_name in class_names:
                col = f'{class_name}_importance'
                if col in shap_importance_df.columns:
                    class_importance = {}
                    for wind_feature in wind_height_features:
                        feature_idx = shap_importance_df[shap_importance_df['feature'] == wind_feature].index
                        if len(feature_idx) > 0:
                            importance = shap_importance_df.loc[feature_idx[0], col]
                            class_importance[wind_feature] = importance
                    height_importance[class_name] = class_importance
            
            # 绘制风速高度重要性
            if height_importance:
                wind_features = list(height_importance[list(height_importance.keys())[0]].keys())
                x = np.arange(len(class_names))
                width = 0.8 / len(wind_features)
                
                for i, wind_feature in enumerate(wind_features):
                    importances = [height_importance[cls].get(wind_feature, 0) for cls in class_names]
                    offset = (i - len(wind_features)/2 + 0.5) * width
                    
                    # 提取高度信息用于标签
                    height_label = wind_feature.split('_')[-1] if '_' in wind_feature else wind_feature
                    
                    ax4.bar(x + offset, importances, width, label=height_label, alpha=0.7)
                
                ax4.set_xticks(x)
                ax4.set_xticklabels([name.replace('_', '\n') for name in class_names], rotation=45, ha='right')
                ax4.set_ylabel('SHAP重要性')
                ax4.set_title('不同高度风速重要性\n(验证切变机制)')
                ax4.legend()
                ax4.grid(True, alpha=0.3)
        
        # 5. 特征类型重要性分布
        ax5 = axes[1, 1]
        
        # 按特征类型分组
        feature_categories = {
            'wind_speed': [f for f in self.feature_names if 'wind_speed' in f],
            'wind_direction': [f for f in self.feature_names if 'wind_dir' in f],
            'temperature': [f for f in self.feature_names if 'temperature' in f],
            'pressure': [f for f in self.feature_names if 'pressure' in f],
            'other': [f for f in self.feature_names if not any(keyword in f for keyword in 
                     ['wind_speed', 'wind_dir', 'temperature', 'pressure'])]
        }
        
        category_importance = {}
        for class_name in class_names:
            col = f'{class_name}_importance'
            if col in shap_importance_df.columns:
                cat_importance = {}
                for category, features in feature_categories.items():
                    if features:
                        cat_features = shap_importance_df[shap_importance_df['feature'].isin(features)]
                        cat_importance[category] = cat_features[col].sum()
                    else:
                        cat_importance[category] = 0
                category_importance[class_name] = cat_importance
        
        # 绘制特征类型重要性
        if category_importance:
            categories = list(feature_categories.keys())
            x = np.arange(len(categories))
            width = 0.8 / len(class_names)
            
            for i, class_name in enumerate(class_names):
                importances = [category_importance[class_name].get(cat, 0) for cat in categories]
                offset = (i - len(class_names)/2 + 0.5) * width
                
                # 设置颜色
                shear_type = class_name.split('_')[0]
                color = color_map.get(shear_type, 'gray')
                
                ax5.bar(x + offset, importances, width, label=class_name, 
                       color=color, alpha=0.7)
            
            ax5.set_xticks(x)
            ax5.set_xticklabels(categories, rotation=45, ha='right')
            ax5.set_ylabel('累计SHAP重要性')
            ax5.set_title('特征类型重要性分布')
            ax5.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
            ax5.grid(True, alpha=0.3)
        
        # 6. 物理机制验证热力图
        ax6 = axes[1, 2]
        
        # 选择关键特征进行热力图
        key_features = shap_importance_df.head(15)['feature'].tolist()
        heatmap_data = []
        heatmap_labels = []
        
        for class_name in class_names:
            col = f'{class_name}_importance'
            if col in shap_importance_df.columns:
                class_importances = []
                for feature in key_features:
                    feature_idx = shap_importance_df[shap_importance_df['feature'] == feature].index
                    if len(feature_idx) > 0:
                        importance = shap_importance_df.loc[feature_idx[0], col]
                        class_importances.append(importance)
                    else:
                        class_importances.append(0)
                heatmap_data.append(class_importances)
                heatmap_labels.append(class_name.replace('_', '\n'))
        
        if heatmap_data:
            heatmap_array = np.array(heatmap_data)
            im = ax6.imshow(heatmap_array, cmap='YlOrRd', aspect='auto')
            
            # 设置标签
            ax6.set_xticks(range(len(key_features)))
            ax6.set_xticklabels([f.split('_')[-1] if '_' in f else f for f in key_features], 
                              rotation=45, ha='right', fontsize=8)
            ax6.set_yticks(range(len(heatmap_labels)))
            ax6.set_yticklabels(heatmap_labels, fontsize=9)
            
            # 添加数值标注
            for i in range(len(heatmap_labels)):
                for j in range(len(key_features)):
                    text = ax6.text(j, i, f'{heatmap_array[i, j]:.2f}',
                                   ha="center", va="center", color="black", fontsize=6)
            
            ax6.set_title('关键特征重要性热力图')
            
            # 添加颜色条
            cbar = plt.colorbar(im, ax=ax6)
            cbar.set_label('SHAP重要性')
        
        plt.tight_layout()
        plt.savefig(f"{self.save_path}/three_group_shap_comparison.png", dpi=300, bbox_inches='tight')
        plt.show()
        
        # 保存重要性对比数据
        shap_importance_df.to_csv(f"{self.save_path}/three_group_shap_importance.csv", index=False)
        print("✓ SHAP重要性对比数据已保存")
        
        return shap_importance_df
    
    def save_models_and_results(self):
        """保存模型和结果"""
        print("💾 保存模型和结果...")
        
        # 保存各分类模型
        for class_name, model in self.models.items():
            model_path = f"{self.save_path}/three_group_model_{class_name}.pkl"
            joblib.dump(model, model_path)
            print(f"✓ {class_name}模型已保存: {model_path}")
        
        # 保存特征名称
        feature_names_path = f"{self.save_path}/feature_names.pkl"
        joblib.dump(self.feature_names, feature_names_path)
        print(f"✓ 特征名称已保存: {feature_names_path}")
        
        # 保存三级阈值
        thresholds_path = f"{self.save_path}/three_group_thresholds.pkl"
        joblib.dump(self.shear_thresholds, thresholds_path)
        print(f"✓ 三级阈值已保存: {thresholds_path}")
        
        # 保存物理机制信息
        physics_path = f"{self.save_path}/shear_physics_info.pkl"
        joblib.dump(self.shear_physics, physics_path)
        print(f"✓ 物理机制信息已保存: {physics_path}")
        
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
        
        summary_path = f"{self.save_path}/three_group_results_summary.pkl"
        joblib.dump(results_summary, summary_path)
        print(f"✓ 结果摘要已保存: {summary_path}")
        
        return results_summary
    
    def generate_physics_insights(self, performance_df):
        """生成基于物理机制的洞察分析"""
        print("🔬 生成物理机制洞察分析...")
        
        insights = {
            'classification_performance': {},
            'physical_consistency': {},
            'feature_mechanisms': {},
            'recommendations': []
        }
        
        # 1. 分类性能分析
        for _, row in performance_df.iterrows():
            class_name = row['classification']
            shear_type = class_name.split('_')[0]
            period = class_name.split('_')[1]
            
            insights['classification_performance'][class_name] = {
                'r2': row['r2'],
                'rmse': row['rmse'],
                'shear_level': shear_type,
                'time_period': period,
                'sample_size': row['samples'],
                'physical_expectation': self.get_physical_expectation(class_name),
                'performance_level': self.classify_performance(row['r2'])
            }
        
        # 2. 物理一致性分析
        best_performing = performance_df.iloc[0]['classification']
        worst_performing = performance_df.iloc[-1]['classification']
        
        insights['physical_consistency'] = {
            'best_class': best_performing,
            'worst_class': worst_performing,
            'performance_gap': performance_df.iloc[0]['r2'] - performance_df.iloc[-1]['r2'],
            'physical_explanation': self.explain_performance_difference(best_performing, worst_performing)
        }
        
        # 3. 特征机制分析
        if hasattr(self, 'results') and self.results:
            insights['feature_mechanisms'] = self.analyze_feature_mechanisms()
        
        # 4. 实用建议
        insights['recommendations'] = self.generate_recommendations(performance_df)
        
        return insights
    
    def get_physical_expectation(self, class_name):
        """获取分类的物理预期"""
        if class_name in ['weak_day', 'moderate_day']:
            return "中高预测性能 - 白天边界层混合充分，湍流特征明显"
        elif class_name == 'strong_night':
            return "高预测性能 - 夜间稳定层结，风剖面规律性强"
        elif class_name in ['weak_night', 'moderate_night']:
            return "中等预测性能 - 过渡状态，稳定度变化"
        elif class_name == 'strong_day':
            return "低预测性能 - 物理异常状态，预测困难"
        else:
            return "未知物理状态"
    
    def classify_performance(self, r2):
        """分类性能水平"""
        if r2 > 0.8:
            return "优秀"
        elif r2 > 0.6:
            return "良好"
        elif r2 > 0.4:
            return "一般"
        else:
            return "较差"
    
    def explain_performance_difference(self, best_class, worst_class):
        """解释性能差异的物理原因"""
        explanations = {
            'strong_night': "夜间强层结条件下风剖面稳定，切变规律性强",
            'moderate_day': "白天中等切变代表典型的中性边界层",
            'weak_day': "白天弱切变表明强混合，但仍有一定规律性",
            'strong_day': "白天强切变为异常状态，可能由特殊天气引起",
            'weak_night': "夜间弱切变可能由残余湍流或特殊地形引起",
            'moderate_night': "夜间中等切变代表稳定层结建立的过渡阶段"
        }
        
        best_explanation = explanations.get(best_class, "未知机制")
        worst_explanation = explanations.get(worst_class, "未知机制")
        
        return {
            'best_mechanism': best_explanation,
            'worst_mechanism': worst_explanation,
            'physical_logic': f"最佳性能({best_class})的物理机制更稳定规律，而最差性能({worst_class})可能涉及复杂的非线性过程"
        }
    
    def analyze_feature_mechanisms(self):
        """分析特征重要性的物理机制"""
        mechanisms = {}
        
        # 分析风速特征
        wind_features = [f for f in self.feature_names if 'wind_speed' in f]
        if len(wind_features) >= 2:
            mechanisms['wind_profile'] = "多高度风速特征反映风切变剖面特征"
        
        # 分析温度特征
        temp_features = [f for f in self.feature_names if 'temperature' in f]
        if temp_features:
            mechanisms['thermal_stability'] = "温度特征影响大气稳定度和边界层发展"
        
        # 分析风向特征
        dir_features = [f for f in self.feature_names if 'wind_dir' in f]
        if dir_features:
            mechanisms['wind_direction'] = "风向变化反映地形影响和边界层结构"
        
        return mechanisms
    
    def generate_recommendations(self, performance_df):
        """生成实用建议"""
        recommendations = []
        
        # 基于性能排名的建议
        best_classes = performance_df.head(2)['classification'].tolist()
        worst_classes = performance_df.tail(2)['classification'].tolist()
        
        recommendations.append(f"优先使用{best_classes}模型，其物理机制清晰且预测性能优秀")
        
        if any('strong_day' in cls for cls in worst_classes):
            recommendations.append("强切变-白天组合预测困难，建议结合天气类型进行细分")
        
        if any('night' in cls for cls in best_classes):
            recommendations.append("夜间条件下的预测模型表现较好，可重点优化夜间预测策略")
        
        # 基于样本数量的建议
        small_sample_classes = performance_df[performance_df['samples'] < 500]['classification'].tolist()
        if small_sample_classes:
            recommendations.append(f"增加{small_sample_classes}类型的训练样本以提升模型稳定性")
        
        # 基于物理机制的建议
        recommendations.append("建议结合局地气象观测，优化切变阈值设定")
        recommendations.append("考虑引入稳定度参数(如Richardson数)进一步细化分类")
        
        return recommendations
    
    def run_full_three_group_analysis(self):
        """运行完整的三级风切变分析流程"""
        print("=" * 70)
        print("🌪️ 三级风切变-昼夜分类风电预测分析")
        print("=" * 70)
        
        try:
            # 1. 加载和预处理数据
            self.load_and_prepare_data()
            
            # 2. 计算风切变系数
            h1, h2 = self.calculate_wind_shear()
            
            # 3. 三级风切变分类
            shear_counts, shear_stats = self.classify_three_group_shear()
            
            # 4. 确定昼夜分类
            day_start, day_end = self.determine_day_night()
            
            # 5. 创建三级组合分类
            class_stats = self.create_three_group_classification()
            
            # 6. 可视化分类结果
            self.visualize_three_group_classification()
            
            # 7. 按分类分组
            self.prepare_classification_groups(min_samples=200)
            
            # 8. 训练分类模型
            self.train_three_group_models()
            
            # 9. 计算SHAP值
            self.calculate_shap_values()
            
            # 10. 绘制性能对比
            performance_df = self.plot_performance_comparison()
            
            # 11. 绘制SHAP对比
            shap_comparison = self.plot_shap_comparison()
            
            # 12. 生成物理洞察
            insights = self.generate_physics_insights(performance_df)
            
            # 13. 保存模型和结果
            results_summary = self.save_models_and_results()
            
            print("\n" + "=" * 70)
            print("🎉 三级风切变分析完成！")
            print("=" * 70)
            
            print("📊 主要发现:")
            print(f"  风切变计算: 使用 {h1}m 和 {h2}m 高度数据")
            print(f"  三级阈值: 弱切变(α<{self.shear_thresholds['weak_upper']}), "
                  f"中等切变({self.shear_thresholds['weak_upper']}≤α<{self.shear_thresholds['moderate_upper']}), "
                  f"强切变(α≥{self.shear_thresholds['moderate_upper']})")
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
                
                if r2_range > 0.15:
                    print("  → 三级风切变分类很有价值，不同条件下预测差异显著")
                elif r2_range > 0.08:
                    print("  → 三级分类有一定价值，建议进一步优化阈值")
                else:
                    print("  → 各分类预测性能相近，可考虑简化分类策略")
            
            # 输出物理洞察
            print(f"\n🔬 物理机制洞察:")
            if insights['physical_consistency']:
                best = insights['physical_consistency']['best_class']
                worst = insights['physical_consistency']['worst_class']
                gap = insights['physical_consistency']['performance_gap']
                
                print(f"  最佳组合: {best} - 物理机制稳定")
                print(f"  最差组合: {worst} - 可能涉及复杂非线性过程")
                print(f"  性能差距: {gap:.3f}")
            
            # 输出实用建议
            print(f"\n💡 实用建议:")
            if insights['recommendations']:
                for i, rec in enumerate(insights['recommendations'], 1):
                    print(f"  {i}. {rec}")
            
            print(f"\n📁 结果文件保存在: {self.save_path}")
            print("  - three_group_classification.png: 三级分类分析")
            print("  - three_group_performance.png: 模型性能对比")
            print("  - three_group_shap_comparison.png: SHAP重要性对比")
            print("  - three_group_shap_importance.csv: 详细重要性数据")
            
            # 分析各组合的数据分布
            print(f"\n📈 数据分布分析:")
            for shear_type in ['weak', 'moderate', 'strong']:
                day_count = shear_counts.get(f'{shear_type}_day', 0) if hasattr(self, 'data') else 0
                night_count = shear_counts.get(f'{shear_type}_night', 0) if hasattr(self, 'data') else 0
                
                # 直接从self.data统计
                if hasattr(self, 'data') and 'three_group_class' in self.data.columns:
                    day_count = len(self.data[self.data['three_group_class'] == f'{shear_type}_day'])
                    night_count = len(self.data[self.data['three_group_class'] == f'{shear_type}_night'])
                    total = len(self.data)
                    
                    print(f"  {shear_type}切变: 白天{day_count}条({day_count/total*100:.1f}%), "
                          f"夜间{night_count}条({night_count/total*100:.1f}%)")
            
            return True
            
        except Exception as e:
            print(f"❌ 分析过程出错: {str(e)}")
            import traceback
            traceback.print_exc()
            return False

if __name__ == "__main__":
    # 配置路径
    DATA_PATH = "/Users/xiaxin/work/WindForecast_Project/01_Data/processed/imputed_data/changma_imputed_complete.csv"
    SAVE_PATH = "/Users/xiaxin/work/WindForecast_Project/03_Results/SHAP-three_group_wind_shear_analysis"
    
    # 创建保存目录
    import os
    os.makedirs(SAVE_PATH, exist_ok=True)
    
    # 创建分析器并运行
    analyzer = ThreeGroupWindShearAnalyzer(DATA_PATH, SAVE_PATH)
    success = analyzer.run_full_three_group_analysis()
    
    if success:
        print("\n🎯 三级风切变分析成功完成！")
        print("\n💡 核心优势:")
        print("  1. 更精细的物理分类 - 弱/中/强三级切变更符合大气物理")
        print("  2. 物理机制清晰 - 每种组合都有明确的边界层物理解释")
        print("  3. 预测策略差异化 - 不同条件下采用最适合的预测模型")
        print("  4. 工程应用价值 - 可根据实时气象条件选择最优模型")
        print("\n🔮 后续研究方向:")
        print("  1. 结合Richardson数等稳定度参数进一步细化")
        print("  2. 分析季节性对三级分类效果的影响")
        print("  3. 开发实时分类识别和模型切换系统")
        print("  4. 验证在不同地形条件下的适用性")
        print("  5. 探索与天气类型的耦合分类策略")
    else:
        print("\n⚠️ 分析失败，请检查错误信息和数据路径")