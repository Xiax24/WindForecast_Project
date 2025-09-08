#!/usr/bin/env python3
"""
按风向分类的自定义SHAP可视化分析
根据obs_wind_direction_70m将数据分为东风区间和西风区间，分别进行LightGBM建模和SHAP分析
筛选条件：****大于切入风速，小于切出风速****

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
import os
warnings.filterwarnings('ignore')

# 设置绘图样式
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

class WindDirectionSHAPAnalyzer:
    def __init__(self, data_path, results_path):
        self.data_path = data_path
        self.results_path = results_path
        self.raw_data = None
        self.east_wind_data = None
        self.west_wind_data = None
        
        # 为东风和西风分别创建分析器
        self.east_analyzer = None
        self.west_analyzer = None
        
    def load_and_classify_data(self):
        """加载数据并按风向分类"""
        print("=== 加载数据并按风向分类 ===")
        
        # 加载原始数据
        self.raw_data = pd.read_csv(self.data_path)
        print(f"原始数据形状: {self.raw_data.shape}")
        
        # 检查wind_direction_70m列
        wind_dir_col = 'obs_wind_direction_10m'
        wind_dir_col30 = 'obs_wind_direction_30m'
        wind_dir_col50 = 'obs_wind_direction_50m'
        wind_dir_col70 = 'obs_wind_direction_70m'
        if wind_dir_col not in self.raw_data.columns:
            raise ValueError(f"未找到列: {wind_dir_col}")
        
        print(f"风向数据范围: {self.raw_data[wind_dir_col].min():.2f}° - {self.raw_data[wind_dir_col].max():.2f}°")
        
        # 分类数据
        # 东风区间：45° 到 135°（东北风至东南风）
        east_mask = (self.raw_data[wind_dir_col70] >= 45) & (self.raw_data[wind_dir_col70] <= 135) & (self.raw_data[wind_dir_col70] >= 45) & (self.raw_data[wind_dir_col70] <= 135) & (self.raw_data[wind_dir_col30] >= 45) & (self.raw_data[wind_dir_col30] <= 135) & (self.raw_data[wind_dir_col50] >= 45) & (self.raw_data[wind_dir_col50] <= 135)
        self.east_wind_data = self.raw_data[east_mask].copy()
        
        # 西风区间：225° 到 315°（西北风至西南风）
        west_mask = (self.raw_data[wind_dir_col70] >= 225) & (self.raw_data[wind_dir_col70] <= 315) & (self.raw_data[wind_dir_col70] >= 225) & (self.raw_data[wind_dir_col70] <= 315) & (self.raw_data[wind_dir_col30] >= 225) & (self.raw_data[wind_dir_col30] <= 315) & (self.raw_data[wind_dir_col50] >= 225) & (self.raw_data[wind_dir_col50] <= 315)
        self.west_wind_data = self.raw_data[west_mask].copy()
        
        # 统计信息
        excluded_count = len(self.raw_data) - len(self.east_wind_data) - len(self.west_wind_data)
        
        print(f"\n风向分类结果:")
        print(f"  东风区间 (45°-135°): {len(self.east_wind_data)} 条数据")
        print(f"  西风区间 (225°-315°): {len(self.west_wind_data)} 条数据")
        print(f"  南北风向 (排除): {excluded_count} 条数据")
        
        # 显示功率统计
        east_power_mean = self.east_wind_data['power'].mean()
        west_power_mean = self.west_wind_data['power'].mean()
        print(f"\n功率统计:")
        print(f"  东风区间平均功率: {east_power_mean:.2f} kW")
        print(f"  西风区间平均功率: {west_power_mean:.2f} kW")
        
        return self.east_wind_data, self.west_wind_data
    
    def run_analysis(self):
        """运行完整的风向分类SHAP分析"""
        print("=== 开始按风向分类的SHAP分析 ===\n")
        
        # 1. 数据分类
        self.load_and_classify_data()
        
        # 2. 为东风区间创建分析器
        print("\n" + "="*50)
        print("开始分析东风区间数据 (45°-135°)")
        print("="*50)
        
        east_results_path = os.path.join(self.results_path, "east_wind")
        os.makedirs(east_results_path, exist_ok=True)
        
        # 保存东风数据
        east_data_path = os.path.join(east_results_path, "east_wind_data.csv")
        self.east_wind_data.to_csv(east_data_path, index=False)
        
        self.east_analyzer = CustomSHAPVisualizer(
            data=self.east_wind_data,
            results_path=east_results_path,
            wind_type="East Wind (45°-135°)"
        )
        east_model = self.east_analyzer.run_analysis()
        
        # 3. 为西风区间创建分析器
        print("\n" + "="*50)
        print("开始分析西风区间数据 (225°-315°)")
        print("="*50)
        
        west_results_path = os.path.join(self.results_path, "west_wind")
        os.makedirs(west_results_path, exist_ok=True)
        
        # 保存西风数据
        west_data_path = os.path.join(west_results_path, "west_wind_data.csv")
        self.west_wind_data.to_csv(west_data_path, index=False)
        
        self.west_analyzer = CustomSHAPVisualizer(
            data=self.west_wind_data,
            results_path=west_results_path,
            wind_type="West Wind (225°-315°)"
        )
        west_model = self.west_analyzer.run_analysis()
        
        # 4. 生成对比分析
        self.create_comparison_analysis()
        
        print("\n" + "="*60)
        print("按风向分类的SHAP分析完成!")
        print("="*60)
        print("生成的文件:")
        print(f"  东风区间结果: {east_results_path}/")
        print(f"  西风区间结果: {west_results_path}/")
        print(f"  对比分析: {self.results_path}/wind_direction_comparison.png")
        
        return east_model, west_model
    
    def create_comparison_analysis(self):
        """创建东西风区间的对比分析"""
        print("\n生成风向对比分析...")
        
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle('Wind Direction Analysis Comparison: East vs West', fontsize=16, fontweight='bold')
        
        # 1. 功率分布对比
        ax1 = axes[0, 0]
        east_power = self.east_wind_data['power'].dropna()
        west_power = self.west_wind_data['power'].dropna()
        
        ax1.hist(east_power, bins=30, alpha=0.7, label='East Wind (45°-135°)', color='#FF6B6B', density=True)
        ax1.hist(west_power, bins=30, alpha=0.7, label='West Wind (225°-315°)', color='#4ECDC4', density=True)
        ax1.set_xlabel('Power (kW)', fontweight='bold')
        ax1.set_ylabel('Density', fontweight='bold')
        ax1.set_title('(A) Power Distribution Comparison', fontweight='bold')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # 2. 风向分布
        ax2 = axes[0, 1]
        east_dirs = self.east_wind_data['obs_wind_direction_10m'].dropna()
        west_dirs = self.west_wind_data['obs_wind_direction_10m'].dropna()
        
        ax2.hist(east_dirs, bins=20, alpha=0.7, label='East Wind', color='#FF6B6B', density=True)
        ax2.hist(west_dirs, bins=20, alpha=0.7, label='West Wind', color='#4ECDC4', density=True)
        ax2.set_xlabel('Wind Direction (°)', fontweight='bold')
        ax2.set_ylabel('Density', fontweight='bold')
        ax2.set_title('(B) Wind Direction Distribution', fontweight='bold')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # 3. 数据统计对比
        ax3 = axes[1, 0]
        stats_data = {
            'East Wind': [
                len(self.east_wind_data),
                self.east_wind_data['power'].mean(),
                self.east_wind_data['power'].std(),
                self.east_wind_data['obs_wind_speed_10m'].mean()
            ],
            'West Wind': [
                len(self.west_wind_data),
                self.west_wind_data['power'].mean(),
                self.west_wind_data['power'].std(),
                self.west_wind_data['obs_wind_speed_10m'].mean()
            ]
        }
        
        stats_labels = ['Sample Count', 'Mean Power (kW)', 'Power Std (kW)', 'Mean Wind Speed (m/s)']
        x = np.arange(len(stats_labels))
        width = 0.35
        
        bars1 = ax3.bar(x - width/2, stats_data['East Wind'], width, label='East Wind', color='#FF6B6B', alpha=0.8)
        bars2 = ax3.bar(x + width/2, stats_data['West Wind'], width, label='West Wind', color='#4ECDC4', alpha=0.8)
        
        ax3.set_xlabel('Statistics', fontweight='bold')
        ax3.set_ylabel('Value', fontweight='bold')
        ax3.set_title('(C) Statistical Comparison', fontweight='bold')
        ax3.set_xticks(x)
        ax3.set_xticklabels(stats_labels, rotation=45, ha='right')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        
        # 添加数值标签
        for bars in [bars1, bars2]:
            for bar in bars:
                height = bar.get_height()
                ax3.text(bar.get_x() + bar.get_width()/2., height + height*0.01,
                        f'{height:.1f}', ha='center', va='bottom', fontsize=8)
        
        # 4. 模型性能对比（如果有的话）
        ax4 = axes[1, 1]
        if hasattr(self.east_analyzer, 'model') and hasattr(self.west_analyzer, 'model'):
            # 获取模型性能指标
            east_r2 = getattr(self.east_analyzer, 'test_r2', 0)
            west_r2 = getattr(self.west_analyzer, 'test_r2', 0)
            east_rmse = getattr(self.east_analyzer, 'test_rmse', 0)
            west_rmse = getattr(self.west_analyzer, 'test_rmse', 0)
            
            metrics = ['R² Score', 'RMSE']
            east_metrics = [east_r2, east_rmse]
            west_metrics = [west_r2, west_rmse]
            
            x = np.arange(len(metrics))
            width = 0.35
            
            ax4.bar(x - width/2, east_metrics, width, label='East Wind', color='#FF6B6B', alpha=0.8)
            ax4.bar(x + width/2, west_metrics, width, label='West Wind', color='#4ECDC4', alpha=0.8)
            
            ax4.set_xlabel('Metrics', fontweight='bold')
            ax4.set_ylabel('Value', fontweight='bold')
            ax4.set_title('(D) Model Performance Comparison', fontweight='bold')
            ax4.set_xticks(x)
            ax4.set_xticklabels(metrics)
            ax4.legend()
            ax4.grid(True, alpha=0.3)
        else:
            ax4.text(0.5, 0.5, 'Model Performance\nComparison\n(Run analysis first)', 
                    ha='center', va='center', transform=ax4.transAxes, fontsize=12)
            ax4.set_title('(D) Model Performance Comparison', fontweight='bold')
        
        plt.tight_layout()
        
        # 保存对比图
        comparison_path = os.path.join(self.results_path, 'wind_direction_comparison.png')
        plt.savefig(comparison_path, dpi=300, bbox_inches='tight', facecolor='white')
        plt.savefig(comparison_path.replace('.png', '.pdf'), dpi=300, bbox_inches='tight', facecolor='white')
        plt.show()


class CustomSHAPVisualizer:
    def __init__(self, data, results_path, wind_type=""):
        """
        修改后的SHAP分析器，支持传入DataFrame数据
        """
        self.data = data
        self.results_path = results_path
        self.wind_type = wind_type
        self.features = None
        self.target = None
        self.feature_names = None
        self.model = None
        self.X_test = None
        self.y_test = None
        self.shap_values = None
        self.X_sample = None
        self.explainer = None
        
        # 存储性能指标
        self.test_r2 = 0
        self.test_rmse = 0
        self.test_mae = 0
        
    def load_and_prepare_data(self):
        """加载和预处理数据"""
        print(f"预处理{self.wind_type}数据...")
        
        print(f"输入数据形状: {self.data.shape}")
        
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
        print(f"训练{self.wind_type} LightGBM模型...")
        
        # 检查数据量是否足够
        if len(self.features) < 100:
            print(f"警告: 数据量较少 ({len(self.features)} 样本)，可能影响模型性能")
        
        # 分割数据
        X_train, X_test, y_train, y_test = train_test_split(
            self.features, self.target, test_size=0.2, random_state=42
        )
        
        # 保存测试数据
        self.X_test = X_test
        self.y_test = y_test
        
        # LightGBM参数（针对较小数据集调整）
        lgb_params = {
            'objective': 'regression',
            'metric': 'rmse',
            'boosting_type': 'gbdt',
            'num_leaves': min(31, max(10, len(self.features) // 20)),  # 根据数据量调整
            'learning_rate': 0.1,
            'feature_fraction': 0.8,
            'bagging_fraction': 0.8,
            'reg_alpha': 0.1,
            'reg_lambda': 0.1,
            'min_child_samples': max(10, len(X_train) // 100),  # 根据训练集大小调整
            'n_estimators': 100,  # 减少树的数量以防过拟合
            'random_state': 42,
            'verbose': -1
        }
        
        # 训练模型
        self.model = lgb.LGBMRegressor(**lgb_params)
        self.model.fit(X_train, y_train)
        
        # 评估性能
        y_pred_test = self.model.predict(X_test)
        self.test_r2 = r2_score(y_test, y_pred_test)
        self.test_rmse = np.sqrt(mean_squared_error(y_test, y_pred_test))
        self.test_mae = mean_absolute_error(y_test, y_pred_test)
        
        print(f"{self.wind_type} 模型性能:")
        print(f"  测试集 R²: {self.test_r2:.3f}")
        print(f"  测试集 RMSE: {self.test_rmse:.3f}")
        print(f"  测试集 MAE: {self.test_mae:.3f}")
        
        return self.model
    
    def calculate_shap_values(self):
        """计算SHAP值"""
        print(f"计算{self.wind_type} SHAP值...")
        
        # 创建SHAP解释器
        self.explainer = shap.TreeExplainer(self.model)
        
        # 使用测试数据样本进行SHAP分析
        sample_size = min(500, len(self.X_test))  # 减少样本数量以适应较小数据集
        indices = np.random.choice(len(self.X_test), sample_size, replace=False)
        self.X_sample = self.X_test[indices]
        
        print(f"计算{sample_size}个样本的SHAP值...")
        self.shap_values = self.explainer.shap_values(self.X_sample)
        
        return self.shap_values, self.X_sample
    
    def plot_combined_visualization(self):
        """绘制组合的SHAP可视化（2上1下布局）"""
        print(f"绘制{self.wind_type} SHAP可视化...")
        
        # 创建图形，设置大小
        fig = plt.figure(figsize=(16, 12))
        
        # 创建子图网格：2行2列，底部跨越2列
        gs = fig.add_gridspec(2, 2, height_ratios=[1, 1.2], hspace=0.3, wspace=0.3)
        
        # 子图1：特征重要性 (左上)
        ax1 = fig.add_subplot(gs[0, 0])
        self._plot_feature_importance_subplot(ax1)
        
        # 子图2：特征影响分布 (右上)
        ax2 = fig.add_subplot(gs[0, 1])
        self._plot_feature_impact_distribution_subplot(ax2)
        
        # 子图3：瀑布图 (下部跨越2列)
        ax3 = fig.add_subplot(gs[1, :])
        self._plot_waterfall_subplot(ax3)
        
        # 添加总标题
        title = f'SHAP Analysis Results - {self.wind_type}' if self.wind_type else 'SHAP Analysis Results'
        fig.suptitle(title, fontsize=18, fontweight='bold', y=0.98)
        
        # 保存图形
        plt.savefig(f'{self.results_path}/combined_shap_analysis.png', 
                   dpi=300, bbox_inches='tight', facecolor='white')
        plt.savefig(f'{self.results_path}/combined_shap_analysis.pdf', 
                   dpi=300, bbox_inches='tight', facecolor='white')
        plt.show()
    
    def _plot_feature_importance_subplot(self, ax):
        """在子图中绘制特征重要性"""
        # 计算特征重要性
        importance = np.abs(self.shap_values).mean(0)
        
        # 创建特征名称（去掉obs_前缀）
        display_names = [name.replace('obs_', '').replace('_', ' ').title() 
                        for name in self.feature_names]
        
        # 选择前10个最重要的特征
        top_indices = np.argsort(importance)[-10:]
        top_importance = importance[top_indices]
        top_names = [display_names[i] for i in top_indices]
        
        # 使用渐变颜色
        colors = plt.cm.viridis(np.linspace(0.2, 0.8, len(top_indices)))
        
        # 绘制水平条形图
        bars = ax.barh(range(len(top_indices)), top_importance, color=colors, alpha=0.8)
        
        # 设置y轴标签
        ax.set_yticks(range(len(top_indices)))
        ax.set_yticklabels(top_names, fontsize=9)
        
        # 设置标题和标签
        ax.set_xlabel('Mean |SHAP Value|', fontsize=10, fontweight='bold')
        ax.set_title('(A) Feature Importance', fontsize=12, fontweight='bold', pad=15)
        
        # 添加数值标签
        for i, (bar, value) in enumerate(zip(bars, top_importance)):
            width = bar.get_width()
            ax.text(width + width*0.02, bar.get_y() + bar.get_height()/2, 
                   f'{value:.3f}', ha='left', va='center', fontsize=8, fontweight='bold')
        
        # 美化图表
        ax.grid(True, alpha=0.3, axis='x')
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['left'].set_linewidth(0.5)
        ax.spines['bottom'].set_linewidth(0.5)
    
    def _plot_feature_impact_distribution_subplot(self, ax):
        """在子图中绘制特征影响分布"""
        # 计算特征重要性并选择前10个
        importance = np.abs(self.shap_values).mean(0)
        top_indices = np.argsort(importance)[-10:]
        
        # 创建特征名称
        display_names = [self.feature_names[i].replace('obs_', '').replace('_', ' ').title() 
                        for i in top_indices]
        
        # SHAP经典配色
        shap_colors = ['#008bfb', '#ff0051']  # 蓝色到洋红色
        
        # 创建自定义颜色映射（SHAP经典风格）
        from matplotlib.colors import LinearSegmentedColormap
        shap_cmap = LinearSegmentedColormap.from_list('shap_classic', shap_colors, N=256)
        
        # 为每个特征绘制散点
        y_positions = []
        for i, feat_idx in enumerate(top_indices):
            shap_vals = self.shap_values[:, feat_idx]
            feature_vals = self.X_sample[:, feat_idx]
            
            # 标准化特征值用于颜色映射（SHAP经典方式）
            norm_feature_vals = (feature_vals - feature_vals.min()) / (feature_vals.max() - feature_vals.min() + 1e-8)
            
            # 添加一些随机偏移避免重叠
            y_pos = np.full_like(shap_vals, i) + np.random.normal(0, 0.08, len(shap_vals))
            y_positions.append(i)
            
            # 使用SHAP经典配色
            scatter = ax.scatter(shap_vals, y_pos, c=norm_feature_vals, 
                               cmap=shap_cmap, alpha=0.7, s=18, edgecolors='white', linewidth=0.4)
        
        # 设置y轴
        ax.set_yticks(y_positions)
        ax.set_yticklabels(display_names, fontsize=9)
        
        # 设置标题和标签
        ax.set_xlabel('SHAP Value (Impact on Model Output)', fontsize=10, fontweight='bold')
        ax.set_title('(B) Feature Impact Distribution', fontsize=12, fontweight='bold', pad=15)
        
        # 添加颜色条
        cbar = plt.colorbar(scatter, ax=ax, pad=0.02, aspect=20)
        cbar.set_label('Feature Value', fontsize=9, fontweight='bold')
        cbar.ax.tick_params(labelsize=8)
        
        # 自定义颜色条标签（SHAP经典风格）
        cbar.ax.set_ylabel('Feature Value', fontweight='bold', fontsize=9)
        cbar.ax.text(1.5, 1.02, 'High', transform=cbar.ax.transAxes, 
                    ha='center', va='bottom', fontsize=8, fontweight='bold', color='#ff0051')
        cbar.ax.text(1.5, -0.02, 'Low', transform=cbar.ax.transAxes, 
                    ha='center', va='top', fontsize=8, fontweight='bold', color='#008bfb')
        
        # 添加零线（SHAP经典样式）
        ax.axvline(x=0, color='#333333', linestyle='-', alpha=0.8, linewidth=1.5)
        
        # 美化图表
        ax.grid(True, alpha=0.3, axis='x', linestyle='--')
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['left'].set_linewidth(0.8)
        ax.spines['bottom'].set_linewidth(0.8)
        ax.spines['left'].set_color('#666666')
        ax.spines['bottom'].set_color('#666666')
        
        # 添加SHAP风格的图例说明
        legend_x = 0.02
        ax.text(legend_x, 0.98, 'Each dot = one sample', transform=ax.transAxes, 
               fontsize=8, color='#666666', ha='left', va='top')
        ax.text(legend_x, 0.94, 'Color = feature value', transform=ax.transAxes, 
               fontsize=8, color='#666666', ha='left', va='top')
        ax.text(legend_x, 0.90, 'X-axis = SHAP impact', transform=ax.transAxes, 
               fontsize=8, color='#666666', ha='left', va='top')
    
    def _plot_waterfall_subplot(self, ax):
        """在子图中绘制瀑布图"""
        # 选择一个代表性样本
        y_pred_sample = self.model.predict(self.X_sample)
        median_idx = np.argsort(np.abs(y_pred_sample - np.median(y_pred_sample)))[0]
        
        # 获取该样本的SHAP值和特征值
        sample_shap = self.shap_values[median_idx]
        sample_features = self.X_sample[median_idx]
        base_value = self.explainer.expected_value
        prediction = y_pred_sample[median_idx]
        
        # 选择影响最大的特征
        top_indices = np.argsort(np.abs(sample_shap))[-10:]
        
        # 准备数据
        feature_names = [self.feature_names[i].replace('obs_', '').replace('_', ' ').title() for i in top_indices]
        shap_values_subset = sample_shap[top_indices]
        feature_values_subset = sample_features[top_indices]
        
        # 按SHAP值排序（从小到大，这样负值在上面）
        sorted_indices = np.argsort(shap_values_subset)
        feature_names = [feature_names[i] for i in sorted_indices]
        shap_values_subset = shap_values_subset[sorted_indices]
        feature_values_subset = feature_values_subset[sorted_indices]
        
        # 设置背景颜色
        ax.set_facecolor('#f9f9f9')
        
        # 计算累积位置
        cumulative = base_value
        bar_height = 0.6
        
        # 计算布局参数
        x_min = min(base_value, prediction) - 5
        x_max = max(base_value, prediction) + 2
        feature_label_x = x_min + 0.5  # 特征标签位置
        
        for i, (name, feat_val, shap_val) in enumerate(zip(feature_names, feature_values_subset, shap_values_subset)):
            y = len(feature_names) - 1 - i  # 从上到下
            
            if shap_val >= 0:
                # 正向贡献：从累积位置开始向右
                start_x = cumulative
                width = shap_val
                color = '#ff6b6b'  # 红色
                label_text = f'+{shap_val:.2f}'
            else:
                # 负向贡献：向左延伸，但显示位置要调整
                width = abs(shap_val)
                start_x = cumulative + shap_val  # 从更左的位置开始
                color = '#4ecdc4'  # 蓝色
                label_text = f'{shap_val:.2f}'
            
            # 绘制条形
            bar = ax.barh(y, width, left=start_x, height=bar_height, 
                         color=color, alpha=0.85, edgecolor='white', linewidth=1.5)
            
            # 在条形中央添加SHAP值标签
            text_x = start_x + width/2
            ax.text(text_x, y, label_text, ha='center', va='center', 
                   fontweight='bold', color='white', fontsize=9)
            
            # 在左侧添加特征名称（清晰分离）
            ax.text(feature_label_x, y, name, ha='left', va='center', 
                   fontsize=10, fontweight='bold', color='#2c3e50')
            
            # 在特征名称后面添加特征值（小字体，灰色）
            ax.text(feature_label_x, y - 0.35, f'value = {feat_val:.3f}', 
                   ha='left', va='center', fontsize=8, color='#7f8c8d', style='italic')
            
            # 更新累积位置
            cumulative += shap_val
        
        # 绘制基准值和预测值的参考线
        y_min = -0.8
        y_max = len(feature_names) - 0.2
        
        # 基准值线
        ax.axvline(x=base_value, color='#34495e', linestyle='-', alpha=0.8, linewidth=2.5)
        ax.text(base_value, y_max + 0.4, f'Baseline\n{base_value:.2f}', 
               ha='center', va='bottom', fontsize=10, fontweight='bold',
               bbox=dict(boxstyle="round,pad=0.4", facecolor='#ecf0f1', 
                        edgecolor='#34495e', alpha=0.9))
        
        # 预测值线
        ax.axvline(x=prediction, color='#e74c3c', linestyle='-', alpha=0.8, linewidth=2.5)
        ax.text(prediction, y_max + 0.4, f'Prediction\n{prediction:.2f}', 
               ha='center', va='bottom', fontsize=10, fontweight='bold',
               bbox=dict(boxstyle="round,pad=0.4", facecolor='#fadbd8', 
                        edgecolor='#e74c3c', alpha=0.9))
        
        # 设置坐标轴范围
        ax.set_xlim(x_min, x_max)
        ax.set_ylim(y_min, y_max + 1)
        ax.set_yticks([])
        
        # 设置标题和标签
        ax.set_xlabel('Model Output (kW)', fontsize=11, fontweight='bold', color='#2c3e50')
        ax.set_title(f'(C) Feature Contributions for Single Prediction ({prediction:.1f} kW)', 
                    fontsize=12, fontweight='bold', pad=20, color='#2c3e50')
        
        # 美化图表
        ax.grid(True, alpha=0.4, axis='x', linestyle='--', linewidth=0.8)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['left'].set_visible(False)
        ax.spines['bottom'].set_linewidth(1.5)
        ax.spines['bottom'].set_color('#34495e')
        
        # 添加颜色说明（位置调整）
        legend_x = 0.85
        ax.text(legend_x, 0.95, '● Increases prediction', transform=ax.transAxes, 
               color='#ff6b6b', fontweight='bold', fontsize=10, ha='left')
        ax.text(legend_x, 0.90, '● Decreases prediction', transform=ax.transAxes, 
               color='#4ecdc4', fontweight='bold', fontsize=10, ha='left')
        
        # 添加说明文字
        ax.text(0.02, 0.95, 'Features (with values)', transform=ax.transAxes, 
               fontsize=9, fontweight='bold', color='#2c3e50', ha='left')
    
    def run_analysis(self):
        """运行完整的自定义SHAP分析"""
        print(f"=== {self.wind_type} SHAP可视化分析 ===")
        
        # 1. 数据准备和模型训练
        self.load_and_prepare_data()
        self.create_features()
        self.train_lightgbm()
        
        # 2. 计算SHAP值
        self.calculate_shap_values()
        
        # 3. 生成组合可视化
        self.plot_combined_visualization()
        
        print(f"\n{self.wind_type} SHAP可视化完成!")
        print("生成的文件:")
        print(f"  - {self.results_path}/combined_shap_analysis.png")
        print(f"  - {self.results_path}/combined_shap_analysis.pdf")
        
        return self.model


if __name__ == "__main__":
    # 配置路径
    DATA_PATH = "/Users/xiaxin/work/WindForecast_Project/01_Data/processed/imputed_data/changma_imputed_complete.csv"
    RESULTS_PATH = "/Users/xiaxin/work/WindForecast_Project/03_Results/figures/3-10/wind_direction_shap"
    
    # 创建结果目录
    os.makedirs(RESULTS_PATH, exist_ok=True)
    
    # 运行风向分类分析
    analyzer = WindDirectionSHAPAnalyzer(DATA_PATH, RESULTS_PATH)
    east_model, west_model = analyzer.run_analysis()