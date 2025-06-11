#!/usr/bin/env python3
"""
基于温度日变化的昼夜分段风电预测与SHAP重要性分析
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

class DayNightWindPredictionAnalyzer:
    def __init__(self, data_path, save_path):
        self.data_path = data_path
        self.save_path = save_path
        self.data = None
        self.day_data = None
        self.night_data = None
        self.feature_names = None
        self.models = {}
        self.shap_explainers = {}
        self.results = {}
        
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
    
    def analyze_temperature_daily_pattern(self):
        """分析温度的日变化模式，确定昼夜分界点"""
        print("🌡️ 分析温度日变化模式...")
        
        # 找到温度列
        temp_cols = [col for col in self.data.columns if 'temperature' in col]
        if not temp_cols:
            raise ValueError("未找到温度列！")
        
        # 使用第一个温度列进行分析
        temp_col = temp_cols[0]
        print(f"使用温度列: {temp_col}")
        
        # 提取小时信息
        self.data['hour'] = self.data['datetime'].dt.hour
        
        # 计算每小时的平均温度
        hourly_temp = self.data.groupby('hour')[temp_col].agg(['mean', 'std', 'count']).reset_index()
        
        # 绘制温度日变化图
        plt.figure(figsize=(12, 6))
        plt.subplot(1, 2, 1)
        plt.plot(hourly_temp['hour'], hourly_temp['mean'], 'b-', linewidth=2, marker='o')
        plt.fill_between(hourly_temp['hour'], 
                        hourly_temp['mean'] - hourly_temp['std'],
                        hourly_temp['mean'] + hourly_temp['std'], 
                        alpha=0.3)
        plt.xlabel('小时')
        plt.ylabel('温度 (°C)')
        plt.title('温度日变化模式')
        plt.grid(True, alpha=0.3)
        plt.xticks(range(0, 24, 2))
        
        # 找到温度最低点和最高点
        min_temp_hour = hourly_temp.loc[hourly_temp['mean'].idxmin(), 'hour']
        max_temp_hour = hourly_temp.loc[hourly_temp['mean'].idxmax(), 'hour']
        
        print(f"温度最低点: {min_temp_hour}:00 ({hourly_temp.loc[hourly_temp['hour']==min_temp_hour, 'mean'].iloc[0]:.1f}°C)")
        print(f"温度最高点: {max_temp_hour}:00 ({hourly_temp.loc[hourly_temp['hour']==max_temp_hour, 'mean'].iloc[0]:.1f}°C)")
        
        # 定义昼夜分界点（基于温度变化）
        # 通常日出前温度最低，午后温度最高
        if min_temp_hour < 12:
            dawn_hour = min_temp_hour + 1  # 日出约在最低温后1小时
        else:
            dawn_hour = 6  # 默认6点
            
        if max_temp_hour > 12:
            dusk_hour = max_temp_hour + 2  # 日落约在最高温后2小时
        else:
            dusk_hour = 18  # 默认18点
            
        # 调整到合理范围
        dawn_hour = max(5, min(8, dawn_hour))
        dusk_hour = max(16, min(20, dusk_hour))
        
        print(f"✓ 确定昼夜分界点: 日出 {dawn_hour}:00, 日落 {dusk_hour}:00")
        
        # 标记昼夜分界点
        plt.axvline(x=dawn_hour, color='orange', linestyle='--', alpha=0.7, label=f'日出 {dawn_hour}:00')
        plt.axvline(x=dusk_hour, color='red', linestyle='--', alpha=0.7, label=f'日落 {dusk_hour}:00')
        plt.legend()
        
        # 添加昼夜标识
        self.data['period'] = 'night'  # 默认夜间
        day_mask = (self.data['hour'] >= dawn_hour) & (self.data['hour'] < dusk_hour)
        self.data.loc[day_mask, 'period'] = 'day'
        
        # 统计昼夜数据量
        period_counts = self.data['period'].value_counts()
        print(f"数据分布: 白天 {period_counts.get('day', 0)} 条, 夜间 {period_counts.get('night', 0)} 条")
        
        # 绘制昼夜功率分布对比
        plt.subplot(1, 2, 2)
        day_power = self.data[self.data['period'] == 'day']['power']
        night_power = self.data[self.data['period'] == 'night']['power']
        
        plt.hist(day_power, bins=50, alpha=0.6, label=f'白天 (均值:{day_power.mean():.1f}MW)', color='orange')
        plt.hist(night_power, bins=50, alpha=0.6, label=f'夜间 (均值:{night_power.mean():.1f}MW)', color='navy')
        plt.xlabel('功率 (MW)')
        plt.ylabel('频次')
        plt.title('昼夜功率分布对比')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(f"{self.save_path}/temperature_daily_pattern.png", dpi=300, bbox_inches='tight')
        plt.show()
        
        return dawn_hour, dusk_hour, hourly_temp
    
    def process_wind_direction(self, data):
        """处理风向变量为sin/cos分量"""
        print("🧭 处理风向变量...")
        
        data = data.copy()
        # 找到风向列
        wind_dir_cols = [col for col in data.columns if 'wind_direction' in col]
        
        if wind_dir_cols:
            print(f"发现 {len(wind_dir_cols)} 个风向列: {wind_dir_cols}")
            
            # 处理每个风向列
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
            print(f"✓ 转换完成，添加了 {len(wind_dir_cols)*2} 个sin/cos列")
        
        return data
    
    def prepare_features(self, data):
        """准备特征矩阵"""
        # 处理风向
        data = self.process_wind_direction(data)
        
        # 选择特征列（除了datetime, power, hour, period）
        feature_cols = [col for col in data.columns 
                       if col not in ['datetime', 'power', 'hour', 'period']]
        
        features = data[feature_cols].values
        target = data['power'].values
        
        return features, target, feature_cols
    
    def train_period_models(self):
        """分别训练白天和夜间的模型"""
        print("🚀 训练昼夜分段模型...")
        
        # 分离白天和夜间数据
        self.day_data = self.data[self.data['period'] == 'day'].copy()
        self.night_data = self.data[self.data['period'] == 'night'].copy()
        
        print(f"白天数据: {len(self.day_data)} 条")
        print(f"夜间数据: {len(self.night_data)} 条")
        
        # 训练参数
        lgb_params = {
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
        
        # 训练白天模型
        if len(self.day_data) > 100:  # 确保有足够数据
            print("训练白天模型...")
            X_day, y_day, feature_names_day = self.prepare_features(self.day_data)
            X_day_train, X_day_test, y_day_train, y_day_test = train_test_split(
                X_day, y_day, test_size=0.2, random_state=42
            )
            
            day_model = lgb.LGBMRegressor(**lgb_params)
            day_model.fit(X_day_train, y_day_train)
            
            y_day_pred = day_model.predict(X_day_test)
            day_r2 = r2_score(y_day_test, y_day_pred)
            day_rmse = np.sqrt(mean_squared_error(y_day_test, y_day_pred))
            
            self.models['day'] = day_model
            self.feature_names = feature_names_day
            self.results['day'] = {
                'r2': day_r2,
                'rmse': day_rmse,
                'X_test': X_day_test,
                'y_test': y_day_test,
                'y_pred': y_day_pred
            }
            
            print(f"✓ 白天模型 - R²: {day_r2:.4f}, RMSE: {day_rmse:.2f}")
        
        # 训练夜间模型
        if len(self.night_data) > 100:  # 确保有足够数据
            print("训练夜间模型...")
            X_night, y_night, feature_names_night = self.prepare_features(self.night_data)
            X_night_train, X_night_test, y_night_train, y_night_test = train_test_split(
                X_night, y_night, test_size=0.2, random_state=42
            )
            
            night_model = lgb.LGBMRegressor(**lgb_params)
            night_model.fit(X_night_train, y_night_train)
            
            y_night_pred = night_model.predict(X_night_test)
            night_r2 = r2_score(y_night_test, y_night_pred)
            night_rmse = np.sqrt(mean_squared_error(y_night_test, y_night_pred))
            
            self.models['night'] = night_model
            self.results['night'] = {
                'r2': night_r2,
                'rmse': night_rmse,
                'X_test': X_night_test,
                'y_test': y_night_test,
                'y_pred': y_night_pred
            }
            
            print(f"✓ 夜间模型 - R²: {night_r2:.4f}, RMSE: {night_rmse:.2f}")
        
        return self.models
    
    def calculate_shap_values(self, n_samples=1000):
        """计算SHAP值"""
        print("📊 计算SHAP重要性...")
        
        for period in ['day', 'night']:
            if period in self.models:
                print(f"计算{period}模型的SHAP值...")
                
                # 获取测试数据
                X_test = self.results[period]['X_test']
                
                # 限制样本数量以加快计算
                if len(X_test) > n_samples:
                    indices = np.random.choice(len(X_test), n_samples, replace=False)
                    X_sample = X_test[indices]
                else:
                    X_sample = X_test
                
                # 创建SHAP解释器
                explainer = shap.TreeExplainer(self.models[period])
                shap_values = explainer.shap_values(X_sample)
                
                # 保存结果
                self.shap_explainers[period] = explainer
                self.results[period]['shap_values'] = shap_values
                self.results[period]['X_shap'] = X_sample
                
                print(f"✓ {period}模型SHAP计算完成 (样本数: {len(X_sample)})")
    
    def plot_shap_comparison(self):
        """绘制昼夜SHAP重要性对比"""
        print("📈 绘制SHAP重要性对比...")
        
        if 'day' not in self.results or 'night' not in self.results:
            print("⚠️ 缺少昼夜模型结果，跳过SHAP对比")
            return
        
        # 计算平均SHAP重要性
        day_importance = np.abs(self.results['day']['shap_values']).mean(axis=0)
        night_importance = np.abs(self.results['night']['shap_values']).mean(axis=0)
        
        # 创建对比DataFrame
        shap_comparison = pd.DataFrame({
            'feature': self.feature_names,
            'day_importance': day_importance,
            'night_importance': night_importance
        })
        
        # 计算差异
        shap_comparison['difference'] = shap_comparison['day_importance'] - shap_comparison['night_importance']
        shap_comparison['abs_difference'] = np.abs(shap_comparison['difference'])
        
        # 按平均重要性排序
        shap_comparison['avg_importance'] = (shap_comparison['day_importance'] + shap_comparison['night_importance']) / 2
        shap_comparison = shap_comparison.sort_values('avg_importance', ascending=False)
        
        # 绘制对比图
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle('昼夜SHAP重要性分析对比', fontsize=16, fontweight='bold')
        
        # 1. 昼夜重要性条形图对比
        top_features = shap_comparison.head(15)
        
        ax1 = axes[0, 0]
        x = np.arange(len(top_features))
        width = 0.35
        
        ax1.barh(x - width/2, top_features['day_importance'], width, 
                label='白天', color='orange', alpha=0.7)
        ax1.barh(x + width/2, top_features['night_importance'], width,
                label='夜间', color='navy', alpha=0.7)
        
        ax1.set_yticks(x)
        ax1.set_yticklabels(top_features['feature'], fontsize=8)
        ax1.set_xlabel('SHAP重要性')
        ax1.set_title('Top 15 特征重要性对比')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # 2. 重要性差异图
        ax2 = axes[0, 1]
        diff_sorted = shap_comparison.sort_values('difference', ascending=True)
        colors = ['red' if x < 0 else 'blue' for x in diff_sorted['difference']]
        
        ax2.barh(range(len(diff_sorted)), diff_sorted['difference'], color=colors, alpha=0.6)
        ax2.set_yticks(range(len(diff_sorted)))
        ax2.set_yticklabels(diff_sorted['feature'], fontsize=6)
        ax2.set_xlabel('重要性差异 (白天 - 夜间)')
        ax2.set_title('昼夜重要性差异')
        ax2.axvline(x=0, color='black', linestyle='--', alpha=0.5)
        ax2.grid(True, alpha=0.3)
        
        # 3. 散点图对比
        ax3 = axes[1, 0]
        ax3.scatter(shap_comparison['day_importance'], shap_comparison['night_importance'], 
                   alpha=0.6, s=50)
        
        # 添加对角线
        max_val = max(shap_comparison['day_importance'].max(), shap_comparison['night_importance'].max())
        ax3.plot([0, max_val], [0, max_val], 'r--', alpha=0.5, label='y=x')
        
        ax3.set_xlabel('白天SHAP重要性')
        ax3.set_ylabel('夜间SHAP重要性')
        ax3.set_title('昼夜重要性散点对比')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        
        # 标注差异最大的点
        max_diff_idx = shap_comparison['abs_difference'].idxmax()
        max_diff_feature = shap_comparison.loc[max_diff_idx, 'feature']
        max_diff_day = shap_comparison.loc[max_diff_idx, 'day_importance']
        max_diff_night = shap_comparison.loc[max_diff_idx, 'night_importance']
        
        ax3.annotate(max_diff_feature, 
                    xy=(max_diff_day, max_diff_night),
                    xytext=(10, 10), textcoords='offset points',
                    bbox=dict(boxstyle='round,pad=0.3', facecolor='yellow', alpha=0.7),
                    arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0'))
        
        # 4. 分类重要性对比
        ax4 = axes[1, 1]
        
        # 按变量类型分组
        feature_categories = {
            'wind_speed': [f for f in self.feature_names if 'wind_speed' in f],
            'wind_direction': [f for f in self.feature_names if 'wind_dir' in f],
            'temperature': [f for f in self.feature_names if 'temperature' in f],
            'other': [f for f in self.feature_names if not any(keyword in f for keyword in ['wind_speed', 'wind_dir', 'temperature'])]
        }
        
        category_importance = {}
        for category, features in feature_categories.items():
            if features:
                cat_features = shap_comparison[shap_comparison['feature'].isin(features)]
                category_importance[category] = {
                    'day': cat_features['day_importance'].sum(),
                    'night': cat_features['night_importance'].sum()
                }
        
        categories = list(category_importance.keys())
        day_cat_values = [category_importance[cat]['day'] for cat in categories]
        night_cat_values = [category_importance[cat]['night'] for cat in categories]
        
        x = np.arange(len(categories))
        ax4.bar(x - 0.2, day_cat_values, 0.4, label='白天', color='orange', alpha=0.7)
        ax4.bar(x + 0.2, night_cat_values, 0.4, label='夜间', color='navy', alpha=0.7)
        
        ax4.set_xticks(x)
        ax4.set_xticklabels(categories)
        ax4.set_ylabel('累计SHAP重要性')
        ax4.set_title('按变量类型分组的重要性')
        ax4.legend()
        ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(f"{self.save_path}/shap_day_night_comparison.png", dpi=300, bbox_inches='tight')
        plt.show()
        
        # 保存重要性对比表
        shap_comparison.to_csv(f"{self.save_path}/shap_importance_comparison.csv", index=False)
        print("✓ SHAP重要性对比表已保存")
        
        return shap_comparison
    
    def plot_model_performance(self):
        """绘制模型性能对比"""
        print("📊 绘制模型性能对比...")
        
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        fig.suptitle('昼夜模型性能对比', fontsize=16, fontweight='bold')
        
        # 性能指标对比
        periods = list(self.results.keys())
        metrics = ['r2', 'rmse']
        
        # R²对比
        ax1 = axes[0, 0]
        r2_values = [self.results[period]['r2'] for period in periods]
        bars1 = ax1.bar(periods, r2_values, color=['orange', 'navy'], alpha=0.7)
        ax1.set_ylabel('R² Score')
        ax1.set_title('R² 性能对比')
        ax1.set_ylim(0, 1)
        
        # 添加数值标签
        for i, bar in enumerate(bars1):
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                    f'{height:.3f}', ha='center', va='bottom')
        
        # RMSE对比
        ax2 = axes[0, 1]
        rmse_values = [self.results[period]['rmse'] for period in periods]
        bars2 = ax2.bar(periods, rmse_values, color=['orange', 'navy'], alpha=0.7)
        ax2.set_ylabel('RMSE (MW)')
        ax2.set_title('RMSE 对比')
        
        # 添加数值标签
        for i, bar in enumerate(bars2):
            height = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width()/2., height + height*0.01,
                    f'{height:.1f}', ha='center', va='bottom')
        
        # 预测vs实际散点图
        for i, period in enumerate(periods):
            ax = axes[1, i]
            y_test = self.results[period]['y_test']
            y_pred = self.results[period]['y_pred']
            
            ax.scatter(y_test, y_pred, alpha=0.5, s=20)
            
            # 添加完美预测线
            min_val = min(y_test.min(), y_pred.min())
            max_val = max(y_test.max(), y_pred.max())
            ax.plot([min_val, max_val], [min_val, max_val], 'r--', alpha=0.8, label='Perfect Prediction')
            
            ax.set_xlabel('实际功率 (MW)')
            ax.set_ylabel('预测功率 (MW)')
            ax.set_title(f'{period}模型预测效果')
            ax.legend()
            ax.grid(True, alpha=0.3)
            
            # 添加R²信息
            r2 = self.results[period]['r2']
            ax.text(0.05, 0.95, f'R² = {r2:.3f}', transform=ax.transAxes,
                   bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        plt.tight_layout()
        plt.savefig(f"{self.save_path}/model_performance_comparison.png", dpi=300, bbox_inches='tight')
        plt.show()
    
    def save_models_and_results(self):
        """保存模型和结果"""
        print("💾 保存模型和结果...")
        
        # 保存模型
        for period, model in self.models.items():
            model_path = f"{self.save_path}/lightgbm_model_{period}.pkl"
            joblib.dump(model, model_path)
            print(f"✓ {period}模型已保存: {model_path}")
        
        # 保存特征名称
        feature_names_path = f"{self.save_path}/feature_names.pkl"
        joblib.dump(self.feature_names, feature_names_path)
        print(f"✓ 特征名称已保存: {feature_names_path}")
        
        # 保存结果摘要
        results_summary = {}
        for period in self.results:
            results_summary[period] = {
                'r2': self.results[period]['r2'],
                'rmse': self.results[period]['rmse'],
                'data_count': len(self.results[period]['y_test'])
            }
        
        summary_path = f"{self.save_path}/results_summary.pkl"
        joblib.dump(results_summary, summary_path)
        print(f"✓ 结果摘要已保存: {summary_path}")
        
        return results_summary
    
    def run_full_analysis(self):
        """运行完整分析流程"""
        print("=" * 60)
        print("🎯 昼夜分段风电预测与SHAP分析")
        print("=" * 60)
        
        try:
            # 1. 加载和预处理数据
            self.load_and_prepare_data()
            
            # 2. 分析温度日变化并划分昼夜
            dawn_hour, dusk_hour, hourly_temp = self.analyze_temperature_daily_pattern()
            
            # 3. 训练昼夜分段模型
            self.train_period_models()
            
            # 4. 计算SHAP值
            self.calculate_shap_values()
            
            # 5. 绘制SHAP重要性对比
            shap_comparison = self.plot_shap_comparison()
            
            # 6. 绘制模型性能对比
            self.plot_model_performance()
            
            # 7. 保存模型和结果
            results_summary = self.save_models_and_results()
            
            print("\n" + "=" * 60)
            print("🎉 分析完成！")
            print("=" * 60)
            
            print("📊 主要发现:")
            print(f"  昼夜分界点: 日出 {dawn_hour}:00, 日落 {dusk_hour}:00")
            
            for period in results_summary:
                r2 = results_summary[period]['r2']
                rmse = results_summary[period]['rmse']
                count = results_summary[period]['data_count']
                print(f"  {period}模型: R²={r2:.3f}, RMSE={rmse:.1f}MW (样本数:{count})")
            
            if 'day' in results_summary and 'night' in results_summary:
                r2_diff = results_summary['day']['r2'] - results_summary['night']['r2']
                rmse_diff = results_summary['night']['rmse'] - results_summary['day']['rmse']
                print(f"  性能差异: R²差值={r2_diff:.3f}, RMSE差值={rmse_diff:.1f}MW")
            
            print(f"\n📁 结果文件保存在: {self.save_path}")
            print("  - temperature_daily_pattern.png: 温度日变化分析")
            print("  - shap_day_night_comparison.png: SHAP重要性对比")
            print("  - model_performance_comparison.png: 模型性能对比")
            print("  - shap_importance_comparison.csv: 详细重要性数据")
            
            # 分析关键发现
            if hasattr(self, 'results') and 'day' in self.results and 'night' in self.results:
                print(f"\n🔍 关键洞察:")
                
                # 性能对比洞察
                if results_summary['day']['r2'] > results_summary['night']['r2']:
                    print("  - 白天模型预测精度更高，可能因为白天气象条件更稳定")
                else:
                    print("  - 夜间模型预测精度更高，可能因为夜间扰动因素更少")
                
                # 功率差异洞察
                day_power_mean = self.day_data['power'].mean()
                night_power_mean = self.night_data['power'].mean()
                if day_power_mean > night_power_mean:
                    power_diff = day_power_mean - night_power_mean
                    print(f"  - 白天平均功率比夜间高{power_diff:.1f}MW，符合风况日变化规律")
                else:
                    power_diff = night_power_mean - day_power_mean
                    print(f"  - 夜间平均功率比白天高{power_diff:.1f}MW，可能存在夜间风增强现象")
            
            return True
            
        except Exception as e:
            print(f"❌ 分析过程出错: {str(e)}")
            import traceback
            traceback.print_exc()
            return False

if __name__ == "__main__":
    # 配置路径
    DATA_PATH = "/Users/xiaxin/work/WindForecast_Project/01_Data/processed/imputed_data/changma_imputed_complete.csv"
    SAVE_PATH = "/Users/xiaxin/work/WindForecast_Project/03_Results/day_night_analysis"
    
    # 创建保存目录
    import os
    os.makedirs(SAVE_PATH, exist_ok=True)
    
    # 创建分析器并运行
    analyzer = DayNightWindPredictionAnalyzer(DATA_PATH, SAVE_PATH)
    success = analyzer.run_full_analysis()
    
    if success:
        print("\n🎯 分析成功完成！你可以查看生成的图表和数据文件。")
        print("\n💡 建议后续分析:")
        print("  1. 深入分析昼夜差异最大的特征")
        print("  2. 研究不同季节的昼夜模式变化")
        print("  3. 结合天气类型进行细分析")
        print("  4. 优化模型参数以提升夜间预测精度")
    else:
        print("\n⚠️ 分析失败，请检查错误信息和数据路径")

# 额外的实用函数

def load_and_predict_with_period_models(model_day_path, model_night_path, feature_names_path, input_data, periods):
    """
    使用保存的昼夜模型进行预测
    
    Parameters:
    -----------
    model_day_path : str
        白天模型路径
    model_night_path : str  
        夜间模型路径
    feature_names_path : str
        特征名称路径
    input_data : pd.DataFrame
        输入数据，需包含datetime列用于判断昼夜
    periods : list
        对应每条数据的时段标识 ['day', 'night', ...]
    
    Returns:
    --------
    predictions : np.array
        预测结果
    """
    import joblib
    
    # 加载模型
    day_model = joblib.load(model_day_path)
    night_model = joblib.load(model_night_path)
    feature_names = joblib.load(feature_names_path)
    
    # 准备特征数据
    features = input_data[feature_names].values
    
    # 分别预测
    predictions = np.zeros(len(input_data))
    
    day_mask = np.array(periods) == 'day'
    night_mask = np.array(periods) == 'night'
    
    if np.any(day_mask):
        predictions[day_mask] = day_model.predict(features[day_mask])
    
    if np.any(night_mask):
        predictions[night_mask] = night_model.predict(features[night_mask])
    
    return predictions

def analyze_seasonal_day_night_patterns(data_path, save_path):
    """
    分析不同季节的昼夜模式变化
    
    Parameters:
    -----------
    data_path : str
        数据文件路径
    save_path : str
        结果保存路径
    """
    import pandas as pd
    import matplotlib.pyplot as plt
    import seaborn as sns
    
    # 加载数据
    data = pd.read_csv(data_path)
    data['datetime'] = pd.to_datetime(data['datetime'])
    data['month'] = data['datetime'].dt.month
    data['hour'] = data['datetime'].dt.hour
    
    # 定义季节
    def get_season(month):
        if month in [12, 1, 2]:
            return '冬季'
        elif month in [3, 4, 5]:
            return '春季'
        elif month in [6, 7, 8]:
            return '夏季'
        else:
            return '秋季'
    
    data['season'] = data['month'].apply(get_season)
    
    # 分析各季节的温度日变化
    temp_cols = [col for col in data.columns if 'temperature' in col]
    if temp_cols:
        temp_col = temp_cols[0]
        
        plt.figure(figsize=(15, 10))
        
        for i, season in enumerate(['春季', '夏季', '秋季', '冬季']):
            plt.subplot(2, 2, i+1)
            season_data = data[data['season'] == season]
            
            if len(season_data) > 0:
                hourly_temp = season_data.groupby('hour')[temp_col].mean()
                hourly_power = season_data.groupby('hour')['power'].mean()
                
                ax1 = plt.gca()
                color1 = 'tab:red'
                ax1.set_xlabel('小时')
                ax1.set_ylabel('温度 (°C)', color=color1)
                ax1.plot(hourly_temp.index, hourly_temp.values, color=color1, marker='o')
                ax1.tick_params(axis='y', labelcolor=color1)
                
                ax2 = ax1.twinx()
                color2 = 'tab:blue'
                ax2.set_ylabel('功率 (MW)', color=color2)
                ax2.plot(hourly_power.index, hourly_power.values, color=color2, marker='s')
                ax2.tick_params(axis='y', labelcolor=color2)
                
                plt.title(f'{season}温度与功率日变化')
                plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(f"{save_path}/seasonal_daily_patterns.png", dpi=300, bbox_inches='tight')
        plt.show()
        
        print(f"✓ 季节性日变化分析已保存: {save_path}/seasonal_daily_patterns.png")