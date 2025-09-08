#!/usr/bin/env python3
"""
自定义SHAP可视化 - 用纯Python/Matplotlib实现
避免SHAP库的布局问题，完全控制图表样式
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

class CustomSHAPVisualizer:
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
        self.shap_values = None
        self.X_sample = None
        self.explainer = None
        
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
    
    def calculate_shap_values(self):
        """计算SHAP值"""
        print("计算SHAP值...")
        
        # 创建SHAP解释器
        self.explainer = shap.TreeExplainer(self.model)
        
        # 使用测试数据样本进行SHAP分析
        sample_size = min(1000, len(self.X_test))
        indices = np.random.choice(len(self.X_test), sample_size, replace=False)
        self.X_sample = self.X_test[indices]
        
        print(f"计算{sample_size}个样本的SHAP值...")
        self.shap_values = self.explainer.shap_values(self.X_sample)
        
        return self.shap_values, self.X_sample
    
    def plot_combined_visualization(self):
        """绘制组合的SHAP可视化（2上1下布局）"""
        print("绘制组合SHAP可视化...")
        
        # 创建图形，设置大小
        fig = plt.figure(figsize=(16, 14))
        
        # 创建子图网格：2行2列，底部跨越2列
        gs = fig.add_gridspec(2, 2, height_ratios=[1, 1.2], hspace=0.25, wspace=0.05)
        
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
        # fig.suptitle('SHAP Analysis Results', fontsize=18, fontweight='bold', y=0.98)
        
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
        
        # 设置标题和标签

        ax.set_xlabel('Variable importance score', fontsize=20, fontfamily='Arial')#Times New Roman
        ax.set_title('(a)', fontsize=20, fontweight='normal', pad=10)
        # 方法2：根据数据自动设置范围（推荐）
        max_importance = np.max(top_importance)
        min_importance = np.min(top_importance)
        
        # 给最大值添加一些余量，方便显示数值标签
        x_margin = max_importance * 0.15  # 15%的余量
        ax.set_xlim(0, max_importance + x_margin)
        ax.tick_params(axis='x', labelsize=18)  # x轴刻度字体大小
        ax.tick_params(axis='y', labelsize=18)  # y轴刻度字体大小
        ax.set_yticklabels(top_names, fontsize=18, fontfamily='Arial')
        # ax.set_xticklabels(top_names, fontsize=18, fontfamily='Arial')
        # 添加数值标签
        for i, (bar, value) in enumerate(zip(bars, top_importance)):
            width = bar.get_width()
            ax.text(width + width*0.02, bar.get_y() + bar.get_height()/2, 
                   f'{value:.2f}', ha='left', va='center', fontsize=15, fontweight='normal')
        
        # 美化图表
        ax.grid(True, alpha=0.3, axis='x')
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['left'].set_linewidth(0.5)
        ax.spines['bottom'].set_linewidth(0.5)
    
    def _plot_feature_impact_distribution_subplot(self, ax):
        """在子图中绘制特征影响分布 - 隐藏y轴标签的紧凑版本"""
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
                            cmap=shap_cmap, alpha=0.7, s=18, edgecolors='white', linewidth=0.2)
        
        # ============ 隐藏y轴标签，保持位置对齐 ============
        ax.set_yticks(y_positions)
        ax.set_yticklabels([])  # 设置为空列表，隐藏标签
        
        # 或者完全隐藏y轴刻度线和标签
        # ax.set_yticks([])
        
        # 设置刻度字体
        ax.tick_params(axis='x', labelsize=18)
        ax.tick_params(axis='y', labelsize=18, left=False)  # 隐藏y轴刻度线
        
        # ============================================
        
        # 设置标题和标签
        ax.set_xlabel('SHAP value', fontsize=20, fontweight='normal', fontfamily='Arial')
        ax.set_title('(b)', fontsize=20, fontweight='normal', pad=10)
        
        # 添加颜色条
        cbar = plt.colorbar(scatter, ax=ax, pad=0.02, aspect=20)
        cbar.set_label('Feature value', fontsize=12, fontweight='normal', fontfamily='Arial')
        cbar.ax.tick_params(labelsize=16)  # 从10改为16，或者其他你想要的大小
        # 自定义颜色条标签（SHAP经典风格）
        cbar.ax.set_ylabel('Feature value', fontweight='normal', fontsize=18, fontfamily='Arial')
        cbar.ax.text(1.5, 1.02, 'High', transform=cbar.ax.transAxes, 
                    ha='center', va='bottom', fontsize=18, fontweight='normal', 
                    color='#ff0051', fontfamily='Arial')
        cbar.ax.text(1.5, -0.02, 'Low', transform=cbar.ax.transAxes, 
                    ha='center', va='top', fontsize=18, fontweight='normal', 
                    color='#008bfb', fontfamily='Arial')
        
        # 添加零线（SHAP经典样式）
        ax.axvline(x=0, color='#333333', linestyle='-', alpha=0.8, linewidth=1.0)
        
        # 美化图表
        ax.grid(True, alpha=0.3, axis='x', linestyle='--')
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['left'].set_visible(False)  # 隐藏左边框
        ax.spines['bottom'].set_linewidth(0.8)
        ax.spines['bottom'].set_color('#666666')
        
        # 简化的图例说明
        legend_x = 0.5
        ax.text(legend_x, 0.05, 'Each point represents one sample', transform=ax.transAxes, 
            fontsize=12, color='#666666', ha='left', va='top', fontfamily='Arial')

    def plot_combined_visualization_compact(self):
        """绘制组合的SHAP可视化（紧凑布局版本）"""
        print("绘制组合SHAP可视化...")
        
        # 创建图形，设置大小
        fig = plt.figure(figsize=(16, 14))
        
        # 创建子图网格：2行2列，底部跨越2列
        # 减少wspace来让左右两个子图更接近
        gs = fig.add_gridspec(2, 2, height_ratios=[1, 1.2], hspace=0.25, wspace=0.05)  # 从0.3减少到0.15
        
        # 子图1：特征重要性 (左上)
        ax1 = fig.add_subplot(gs[0, 0])
        self._plot_feature_importance_subplot(ax1)
        
        # 子图2：特征影响分布 (右上) - 隐藏y轴标签
        ax2 = fig.add_subplot(gs[0, 1])
        self._plot_feature_impact_distribution_subplot(ax2)
        
        # 子图3：瀑布图 (下部跨越2列)
        ax3 = fig.add_subplot(gs[1, :])
        self._plot_waterfall_subplot(ax3)
        
        # 保存图形
        plt.savefig(f'{self.results_path}/combined_shap_analysis.png',
                    dpi=300, bbox_inches='tight', facecolor='white')
        plt.savefig(f'{self.results_path}/combined_shap_analysis.pdf',
                    dpi=300, bbox_inches='tight', facecolor='white')
        plt.show()

    # 其他布局选项
    def alternative_layout_options():
        """其他可能的布局选择"""
        
        options = {
            'option1': {
                'description': '完全隐藏y轴',
                'yticks': [],
                'yticklabels': [],
                'left_spine': False,
                'tick_params': {'left': False, 'labelleft': False}
            },
            
            'option2': {
                'description': '保留刻度线，隐藏标签',
                'yticks': 'keep_positions',
                'yticklabels': [],
                'left_spine': True,
                'tick_params': {'left': True, 'labelleft': False}
            },
            
            'option3': {
                'description': '只在图(a)显示标签',
                'note': '这是推荐的方案，既保持对齐又节省空间'
            }
        }
        
        return options



    
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
        feature_label_x = x_min - 3.5  # 特征标签位置
        
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
                   fontweight='bold', color='blue', fontsize=18)
            
            # 在左侧添加特征名称（清晰分离）
            ax.text(feature_label_x+0.2, y, name, ha='left', va='center', 
                   fontsize=18, fontweight='normal', color='#2c3e50')
            
            # 在特征名称后面添加特征值（小字体，灰色）
            ax.text(feature_label_x+4, y, f'value = {feat_val:.2f}', 
                   ha='left', va='center', fontsize=18, color='#7f8c8d', style='italic')
            
            # 更新累积位置
            cumulative += shap_val
        
        # 绘制基准值和预测值的参考线
        y_min = -0.8
        y_max = len(feature_names) - 0.2
        
        # 基准值线
        ax.axvline(x=base_value, color='#34495e', linestyle='-', alpha=0.8, linewidth=2.5)
        ax.text(base_value, y_max + 0.4, f'Baseline {base_value:.2f}', 
               ha='center', va='bottom', fontsize=18, fontweight='bold',
               bbox=dict(boxstyle="round,pad=0.4", facecolor='#ecf0f1', 
                        edgecolor='#34495e', alpha=0.9))
        
        # 预测值线
        ax.axvline(x=prediction, color='#e74c3c', linestyle='-', alpha=0.8, linewidth=2.5)
        ax.text(prediction, y_max + 0.4, f'Prediction {prediction:.2f}', 
               ha='center', va='bottom', fontsize=18, fontweight='bold',
               bbox=dict(boxstyle="round,pad=0.4", facecolor='#fadbd8', 
                        edgecolor='#e74c3c', alpha=0.9))
        
        # 设置坐标轴范围
        ax.set_xlim(x_min, x_max)
        ax.set_ylim(y_min, y_max + 1)
        ax.set_yticks([])
        ax.tick_params(axis='x', labelsize=20)
        # 设置标题和标签
        ax.set_xlabel('Model Output Power(kW)', fontsize=20, fontweight='normal', color='#2c3e50')
        ax.set_title(f'(c)', 
                    fontsize=22, fontweight='normal', pad=15, color='#2c3e50')
        
        # 美化图表
        ax.grid(True, alpha=0.4, axis='x', linestyle='--', linewidth=0.8)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['left'].set_visible(False)
        ax.spines['bottom'].set_linewidth(1.5)
        ax.spines['bottom'].set_color('#34495e')
        
        # 添加颜色说明（位置调整）
        legend_x = 0.65
        ax.text(legend_x, 0.15, '● Increases prediction', transform=ax.transAxes, 
               color='#ff6b6b', fontweight='bold', fontsize=18, ha='left')
        ax.text(legend_x, 0.10, '● Decreases prediction', transform=ax.transAxes, 
               color='#4ecdc4', fontweight='bold', fontsize=18, ha='left')
        
        # 添加说明文字
        ax.text(-0.13, 0.95, 'Features', transform=ax.transAxes, 
               fontsize=18, fontweight='bold', color='#2c3e50', ha='left')
    
    def run_analysis(self):
        """运行完整的自定义SHAP分析"""
        print("=== 自定义SHAP可视化分析 ===")
        
        # 1. 数据准备和模型训练
        self.load_and_prepare_data()
        self.create_features()
        self.train_lightgbm()
        
        # 2. 计算SHAP值
        self.calculate_shap_values()
        
        # 3. 生成组合可视化
        self.plot_combined_visualization()
        
        print("\n自定义SHAP可视化完成!")
        print("生成的文件:")
        print(f"  - {self.results_path}/combined_shap_analysis.png")
        print(f"  - {self.results_path}/combined_shap_analysis.pdf")
        
        return self.model

if __name__ == "__main__":
    # 配置路径
    DATA_PATH = "/Users/xiaxin/work/WindForecast_Project/01_Data/processed/imputed_data/changma_imputed_complete.csv"
    RESULTS_PATH = "/Users/xiaxin/work/WindForecast_Project/03_Results/figures/3.1results/custom_shap"
    
    # 创建结果目录
    import os
    os.makedirs(RESULTS_PATH, exist_ok=True)
    
    # 运行自定义分析
    visualizer = CustomSHAPVisualizer(DATA_PATH, RESULTS_PATH)
    model = visualizer.run_analysis()