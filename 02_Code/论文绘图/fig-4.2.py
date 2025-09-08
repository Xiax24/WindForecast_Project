#!/usr/bin/env python3
"""
按风向分类的SHAP蜂群图分析 - 使用所有特征版本
严格条件：四层风向都必须在同一区间内，70m风速3-25m/s
使用所有obs_特征进行建模，只生成东西风蜂群图对比
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import lightgbm as lgb
import shap
import warnings
import os
from pathlib import Path
import logging

warnings.filterwarnings('ignore')

# 设置matplotlib参数
plt.rcParams.update({
    'font.family': ['Arial', 'DejaVu Sans'],
    'font.size': 12,
    'axes.linewidth': 1.2,
    'figure.dpi': 300,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
    'savefig.facecolor': 'white'
})

class WindDirectionSHAPAnalyzer:
    def __init__(self, data_path, results_path):
        self.data_path = Path(data_path)
        self.results_path = Path(results_path)
        self.results_path.mkdir(parents=True, exist_ok=True)
        
        # 定义用于筛选的关键列（风向和70m风速）
        self.wind_dir_columns = [
            'obs_wind_direction_10m',
            'obs_wind_direction_30m',
            'obs_wind_direction_50m', 
            'obs_wind_direction_70m'
        ]
        
        # 用于风速筛选的列
        self.wind_speed_filter_col = 'obs_wind_speed_70m'
        
    def load_and_classify_data(self):
        """加载数据并进行严格的风向分类和风速筛选"""
        print("=== 加载数据并进行严格风向分类和风速筛选 ===")
        
        # 加载数据
        self.raw_data = pd.read_csv(self.data_path)
        print(f"原始数据形状: {self.raw_data.shape}")
        
        # 检查必要的筛选列是否存在
        required_filter_cols = self.wind_dir_columns + [self.wind_speed_filter_col, 'power']
        missing_cols = [col for col in required_filter_cols if col not in self.raw_data.columns]
        if missing_cols:
            raise ValueError(f"缺少必要的筛选列: {missing_cols}")
        
        # 基础清理：去除功率为负的数据
        data_clean = self.raw_data[self.raw_data['power'] >= 0].copy()
        print(f"去除负功率后数据形状: {data_clean.shape}")
        
        # 去除筛选列中有缺失值的数据
        data_clean = data_clean.dropna(subset=required_filter_cols)
        print(f"去除关键列缺失值后数据形状: {data_clean.shape}")
        
        # 添加70m风速筛选条件：剔除3m/s以下和25m/s以上的数据
        wind_speed_condition = (data_clean[self.wind_speed_filter_col] >= 3.0) & (data_clean[self.wind_speed_filter_col] <= 25.0)
        data_filtered = data_clean[wind_speed_condition].copy()
        
        print(f"风速筛选后数据形状: {data_filtered.shape}")
        print(f"剔除的极端风速数据: {len(data_clean) - len(data_filtered)} 条")
        print(f"  (70m风速 < 3m/s 或 > 25m/s)")
        
        # 严格的东风条件：四层风向都在东北到东南区间 (45°-135°)
        east_condition = True
        for col in self.wind_dir_columns:
            # 东北到东南：45° 到 135°
            east_condition = east_condition & (data_filtered[col] >= 45) & (data_filtered[col] <= 135)
        
        east_wind_data = data_filtered[east_condition].copy()
        
        # 严格的西风条件：四层风向都在西北到西南区间 (225°-315°)
        west_condition = True
        for col in self.wind_dir_columns:
            # 西北到西南：225° 到 315°
            west_condition = west_condition & (data_filtered[col] >= 225) & (data_filtered[col] <= 315)
            
        west_wind_data = data_filtered[west_condition].copy()
        
        # 统计结果
        print(f"\n严格筛选结果:")
        print(f"  东风区间 (四层都在东北到东南 45°-135°): {len(east_wind_data)} 条数据")
        print(f"  西风区间 (四层都在西北到西南 225°-315°): {len(west_wind_data)} 条数据")
        print(f"  其他风向数据: {len(data_filtered) - len(east_wind_data) - len(west_wind_data)} 条")
        
        # 显示风速统计
        if len(east_wind_data) > 0:
            print(f"  东风区间70m风速范围: {east_wind_data[self.wind_speed_filter_col].min():.1f} - {east_wind_data[self.wind_speed_filter_col].max():.1f} m/s")
        if len(west_wind_data) > 0:
            print(f"  西风区间70m风速范围: {west_wind_data[self.wind_speed_filter_col].min():.1f} - {west_wind_data[self.wind_speed_filter_col].max():.1f} m/s")
        
        if len(east_wind_data) < 100:
            print(f"警告: 东风数据量较少 ({len(east_wind_data)} 条)")
        if len(west_wind_data) < 100:
            print(f"警告: 西风数据量较少 ({len(west_wind_data)} 条)")
            
        return east_wind_data, west_wind_data
    
    def train_model_and_shap(self, data, wind_type):
        """训练模型并计算SHAP值 - 使用所有可用特征"""
        print(f"\n=== 训练{wind_type}模型并计算SHAP ===")
        
        # 选择所有obs_开头的特征列，排除密度和湿度
        obs_columns = [col for col in data.columns if col.startswith('obs_')]
        obs_columns = [col for col in obs_columns if 'density' not in col and 'humidity' not in col]
        
        # 排除datetime和power列
        feature_columns = [col for col in obs_columns if col not in ['datetime', 'power']]
        
        print(f"使用{len(feature_columns)}个特征进行建模:")
        for i, col in enumerate(feature_columns):
            print(f"  {i+1:2d}. {col}")
        
        # 处理风向变量为sin/cos分量
        data_processed = data.copy()
        wind_dir_cols = [col for col in feature_columns if 'wind_direction' in col]
        
        if wind_dir_cols:
            print(f"\n处理{len(wind_dir_cols)}个风向变量为sin/cos分量...")
            for col in wind_dir_cols:
                # 转换为弧度
                wind_dir_rad = np.deg2rad(data_processed[col])
                
                # 创建sin/cos分量
                sin_col = col.replace('wind_direction', 'wind_dir_sin')
                cos_col = col.replace('wind_direction', 'wind_dir_cos')
                
                data_processed[sin_col] = np.sin(wind_dir_rad)
                data_processed[cos_col] = np.cos(wind_dir_rad)
                
                print(f"    {col} → {sin_col}, {cos_col}")
            
            # 更新特征列表
            feature_columns = [col for col in feature_columns if 'wind_direction' not in col]
            sin_cos_cols = [col for col in data_processed.columns if 'wind_dir_sin' in col or 'wind_dir_cos' in col]
            feature_columns.extend(sin_cos_cols)
        
        # 去除包含缺失值的行
        data_clean = data_processed[feature_columns + ['power']].dropna()
        print(f"\n去除缺失值后的数据形状: {data_clean.shape}")
        
        # 准备特征和目标
        X = data_clean[feature_columns].values
        y = data_clean['power'].values
        
        print(f"最终特征矩阵形状: {X.shape}")
        
        # 分割数据
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        
        # 训练LightGBM模型
        lgb_params = {
            'objective': 'regression',
            'metric': 'rmse',
            'boosting_type': 'gbdt',
            'num_leaves': 31,
            'learning_rate': 0.1,
            'feature_fraction': 0.8,
            'bagging_fraction': 0.8,
            'reg_alpha': 0.1,
            'reg_lambda': 0.1,
            'min_child_samples': 20,
            'n_estimators': 100,
            'random_state': 42,
            'verbose': -1
        }
        
        model = lgb.LGBMRegressor(**lgb_params)
        model.fit(X_train, y_train)
        
        # 评估性能
        y_pred = model.predict(X_test)
        r2 = r2_score(y_test, y_pred)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        mae = mean_absolute_error(y_test, y_pred)
        
        print(f"{wind_type}模型性能:")
        print(f"  R²: {r2:.3f}")
        print(f"  RMSE: {rmse:.3f} kW")
        print(f"  MAE: {mae:.3f} kW")
        
        # 计算SHAP值
        explainer = shap.TreeExplainer(model)
        
        # 可选择使用所有测试数据或采样数据计算SHAP值
        use_all_test_data = True  # 设为True使用所有测试数据，False使用采样
        
        if use_all_test_data:
            X_sample = X_test
            print(f"计算所有{len(X_test)}个测试样本的SHAP值...")
        else:
            # 使用较小的样本量以提高计算速度
            sample_size = min(300, len(X_test))
            sample_indices = np.random.choice(len(X_test), sample_size, replace=False)
            X_sample = X_test[sample_indices]
            print(f"计算{sample_size}个采样样本的SHAP值...")
        
        shap_values = explainer.shap_values(X_sample)
        
        return {
            'model': model,
            'explainer': explainer,
            'shap_values': shap_values,
            'X_sample': X_sample,
            'performance': {'r2': r2, 'rmse': rmse, 'mae': mae},
            'feature_names': feature_columns
        }
    
    def create_dual_beeswarm_plot(self, east_results, west_results):
        """创建东西风的双蜂群图对比"""
        print("\n=== 创建双蜂群图对比 ===")
        
        # 创建图形
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 10))
        
        # 获取特征名称并处理显示
        east_features = east_results['feature_names']
        west_features = west_results['feature_names']
        
        # 创建显示名称映射
        def create_display_name(feature_name):
            """将特征名转换为更好的显示名称"""
            name = feature_name.replace('obs_', '').replace('_', ' ').title()
            # 特殊处理
            name = name.replace('Wind Speed', 'WS')
            name = name.replace('Temperature', 'Temp')
            name = name.replace('Pressure', 'Press')
            name = name.replace('Wind Dir Sin', 'WD Sin')
            name = name.replace('Wind Dir Cos', 'WD Cos')
            return name
        
        east_display_names = [create_display_name(name) for name in east_features]
        west_display_names = [create_display_name(name) for name in west_features]
        
        # 绘制东风蜂群图
        self._plot_beeswarm(ax1, east_results, "East Wind (NE-SE: 45°-135°)", east_display_names)
        
        # 绘制西风蜂群图  
        self._plot_beeswarm(ax2, west_results, "West Wind (NW-SW: 225°-315°)", west_display_names)
        
        # 调整布局
        plt.tight_layout()
        
        # 添加总标题
        fig.suptitle('SHAP Feature Importance Analysis: East vs West Wind Conditions', 
                    fontsize=16, fontweight='bold', y=0.98)
        
        # 保存图形
        save_path = self.results_path / 'wind_direction_beeswarm_comparison.png'
        plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
        plt.savefig(save_path.with_suffix('.pdf'), dpi=300, bbox_inches='tight', facecolor='white')
        
        print(f"蜂群图已保存到: {save_path}")
        print(f"东风模型使用了 {len(east_features)} 个特征")
        print(f"西风模型使用了 {len(west_features)} 个特征")
        plt.show()
    
    def _plot_beeswarm(self, ax, results, title, display_names):
        """绘制单个蜂群图"""
        shap_values = results['shap_values']
        X_sample = results['X_sample'] 
        performance = results['performance']
        feature_names = results['feature_names']
        
        # 计算特征重要性排序，选择前15个最重要的特征
        importance = np.abs(shap_values).mean(0)
        top_indices = np.argsort(importance)[-15:]  # 选择前15个最重要的特征
        
        # 获取对应的显示名称
        top_display_names = [display_names[i] for i in top_indices]
        
        # 绘制每个特征的分布
        for plot_idx, feat_idx in enumerate(top_indices):
            shap_vals = shap_values[:, feat_idx]
            feature_vals = X_sample[:, feat_idx]
            
            # 标准化特征值用于颜色映射
            if feature_vals.max() != feature_vals.min():
                norm_vals = (feature_vals - feature_vals.min()) / (feature_vals.max() - feature_vals.min())
            else:
                norm_vals = np.ones_like(feature_vals) * 0.5
            
            # 添加垂直方向的小幅随机偏移
            y_pos = np.full_like(shap_vals, plot_idx) + np.random.normal(0, 0.1, len(shap_vals))
            
            # 根据特征值大小设置颜色
            scatter_colors = plt.cm.RdBu_r(norm_vals)
            
            # 绘制散点
            scatter = ax.scatter(shap_vals, y_pos, c=scatter_colors, 
                               alpha=0.7, s=25, edgecolors='white', linewidth=0.5)
        
        # 设置y轴
        ax.set_yticks(range(len(top_display_names)))
        ax.set_yticklabels(top_display_names, fontsize=10)
        
        # 设置x轴和标签
        ax.set_xlabel('SHAP Value (Impact on Model Output)', fontsize=12, fontweight='bold')
        ax.set_title(f'{title}\n(R² = {performance["r2"]:.3f}, RMSE = {performance["rmse"]:.1f} kW)\nTop 15 Features', 
                    fontsize=12, fontweight='bold', pad=20)
        
        # 添加零线
        ax.axvline(x=0, color='black', linestyle='-', alpha=0.8, linewidth=1.5)
        
        # 美化图表
        ax.grid(True, alpha=0.3, axis='x')
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        
        # 添加颜色说明
        ax.text(0.02, 0.98, 'Feature value:', transform=ax.transAxes, 
               fontsize=10, fontweight='bold', va='top')
        ax.text(0.02, 0.94, '● High', transform=ax.transAxes, 
               color='#d73027', fontsize=10, va='top')
        ax.text(0.02, 0.90, '● Low', transform=ax.transAxes, 
               color='#313695', fontsize=10, va='top')
    
    def run_analysis(self):
        """运行完整分析流程"""
        print("=== 开始按风向分类的SHAP分析（使用所有特征）===")
        
        # 1. 数据分类
        east_data, west_data = self.load_and_classify_data()
        
        if len(east_data) < 50 or len(west_data) < 50:
            raise ValueError("数据量太少，无法进行可靠的分析")
        
        # 2. 训练模型并计算SHAP
        east_results = self.train_model_and_shap(east_data, "东风")
        west_results = self.train_model_and_shap(west_data, "西风")
        
        # 3. 创建对比可视化
        self.create_dual_beeswarm_plot(east_results, west_results)
        
        # 4. 保存数据统计
        self._save_summary_stats(east_data, west_data, east_results, west_results)
        
        print("\n=== 分析完成 ===")
        print(f"结果保存在: {self.results_path}")
        
        return east_results, west_results
    
    def _save_summary_stats(self, east_data, west_data, east_results, west_results):
        """保存统计摘要"""
        summary = {
            'East Wind': {
                'sample_count': len(east_data),
                'mean_power': east_data['power'].mean(),
                'std_power': east_data['power'].std(),
                'mean_wind_speed_70m': east_data[self.wind_speed_filter_col].mean(),
                'wind_speed_70m_range': f"{east_data[self.wind_speed_filter_col].min():.1f}-{east_data[self.wind_speed_filter_col].max():.1f}",
                'feature_count': len(east_results['feature_names']),
                'model_r2': east_results['performance']['r2'],
                'model_rmse': east_results['performance']['rmse']
            },
            'West Wind': {
                'sample_count': len(west_data),
                'mean_power': west_data['power'].mean(), 
                'std_power': west_data['power'].std(),
                'mean_wind_speed_70m': west_data[self.wind_speed_filter_col].mean(),
                'wind_speed_70m_range': f"{west_data[self.wind_speed_filter_col].min():.1f}-{west_data[self.wind_speed_filter_col].max():.1f}",
                'feature_count': len(west_results['feature_names']),
                'model_r2': west_results['performance']['r2'],
                'model_rmse': west_results['performance']['rmse']
            }
        }
        
        # 保存为CSV
        summary_df = pd.DataFrame(summary).T
        summary_path = self.results_path / 'analysis_summary.csv'
        summary_df.to_csv(summary_path)
        
        # 保存特征列表
        max_features = max(len(east_results['feature_names']), len(west_results['feature_names']))
        east_features_padded = east_results['feature_names'] + [''] * (max_features - len(east_results['feature_names']))
        west_features_padded = west_results['feature_names'] + [''] * (max_features - len(west_results['feature_names']))
        
        feature_comparison = pd.DataFrame({
            'East_Wind_Features': east_features_padded,
            'West_Wind_Features': west_features_padded
        })
        feature_path = self.results_path / 'feature_comparison.csv'
        feature_comparison.to_csv(feature_path, index=False)
        
        print(f"\n统计摘要已保存到: {summary_path}")
        print(f"特征对比已保存到: {feature_path}")
        print(summary_df)


if __name__ == "__main__":
    # 配置路径
    DATA_PATH = "/Users/xiaxin/work/WindForecast_Project/01_Data/processed/imputed_data/changma_imputed_complete.csv"
    RESULTS_PATH = "/Users/xiaxin/work/WindForecast_Project/03_Results/figures/4.discussion and conclusion/wind_direction_shap_all_features"
    
    # 运行分析
    analyzer = WindDirectionSHAPAnalyzer(DATA_PATH, RESULTS_PATH)
    east_results, west_results = analyzer.run_analysis()