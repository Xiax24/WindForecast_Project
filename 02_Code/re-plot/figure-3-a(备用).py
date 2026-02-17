#!/usr/bin/env python3
"""
Figure 3 - Panel A: 基于机器学习试验的物理机制实证
总体预测效能对比 (Performance Benchmark)

对比三种方案：
1. Hub-height (HH): 仅输入 70m 风速
2. Standard REWS (SR): 输入 30/50/70m 三层风速
3. Extended REWS (ER): 输入 10/30/50/70m 四层风速

绘图内容：三个并排散点图 (Predicted vs. Observed Power)
核心指标：R², RMSE
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import lightgbm as lgb
import warnings
from pathlib import Path

warnings.filterwarnings('ignore')

# 设置matplotlib参数 - 与参考代码保持一致
plt.rcParams.update({
    'font.family': ['Arial', 'DejaVu Sans'],
    'font.size': 25,
    'axes.linewidth': 1.2,
    'figure.dpi': 500,
    'savefig.dpi': 500,
    'savefig.bbox': 'tight',
    'savefig.facecolor': 'white'
})


class REWSComparisonAnalyzer:
    """REWS方案对比分析器"""
    
    def __init__(self, data_path, output_dir):
        self.data_path = Path(data_path)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # 定义三种方案的输入特征
        self.model_configs = {
            'Hub-height': {
                'features': ['obs_wind_speed_70m'],
                'label': 'Hub-height\n(70m only)',
                'color': "#92908F"  # 棕色
            },
            'Standard REWS': {
                'features': ['obs_wind_speed_30m', 'obs_wind_speed_50m', 'obs_wind_speed_70m'],
                'label': 'Standard REWS\n(30/50/70m)',
                'color': '#4A90E2'  # 蓝色
            },
            'Extended REWS': {
                'features': ['obs_wind_speed_10m', 'obs_wind_speed_30m', 
                           'obs_wind_speed_50m', 'obs_wind_speed_70m'],
                'label': 'Extended REWS\n(10/30/50/70m)',
                'color': '#E74C3C'  # 红色
            }
        }
        
        # LightGBM参数 - 与参考代码一致
        self.lgb_params = {
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
    
    def load_and_clean_data(self):
        """加载并清洗数据"""
        print("=== 加载并清洗数据 ===")
        
        # 加载数据
        data = pd.read_csv(self.data_path)
        print(f"原始数据形状: {data.shape}")
        
        # 检查所需列是否存在
        all_features = set()
        for config in self.model_configs.values():
            all_features.update(config['features'])
        
        required_cols = list(all_features) + ['power']
        missing_cols = [col for col in required_cols if col not in data.columns]
        if missing_cols:
            raise ValueError(f"缺少必要列: {missing_cols}")
        
        # 基础清理
        # 1. 去除功率为负的数据
        data_clean = data[data['power'] >= 0].copy()
        print(f"去除负功率后: {data_clean.shape}")
        
        # 2. 去除关键列的缺失值
        data_clean = data_clean.dropna(subset=required_cols)
        print(f"去除缺失值后: {data_clean.shape}")
        
        # 3. 风速筛选：3-25 m/s（基于70m风速）
        wind_speed_condition = (
            (data_clean['obs_wind_speed_70m'] >= 3.0) & 
            (data_clean['obs_wind_speed_70m'] <= 25.0)
        )
        data_clean = data_clean[wind_speed_condition].copy()
        print(f"风速筛选后 (3-25 m/s @ 70m): {data_clean.shape}")
        
        # 数据统计
        print(f"\n数据统计:")
        print(f"  功率范围: {data_clean['power'].min():.1f} - {data_clean['power'].max():.1f} kW")
        print(f"  平均功率: {data_clean['power'].mean():.1f} kW")
        print(f"  70m风速范围: {data_clean['obs_wind_speed_70m'].min():.1f} - "
              f"{data_clean['obs_wind_speed_70m'].max():.1f} m/s")
        
        self.data = data_clean
        return data_clean
    
    def train_single_model(self, model_name):
        """训练单个模型方案"""
        print(f"\n=== 训练 {model_name} 模型 ===")
        
        config = self.model_configs[model_name]
        features = config['features']
        
        print(f"输入特征 ({len(features)}个):")
        for i, feat in enumerate(features, 1):
            print(f"  {i}. {feat}")
        
        # 准备数据
        X = self.data[features].values
        y = self.data['power'].values
        
        print(f"特征矩阵形状: {X.shape}")
        
        # 分割训练集和测试集
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        
        print(f"训练集: {X_train.shape}, 测试集: {X_test.shape}")
        
        # 训练模型
        model = lgb.LGBMRegressor(**self.lgb_params)
        model.fit(X_train, y_train)
        
        # 预测
        y_pred_train = model.predict(X_train)
        y_pred_test = model.predict(X_test)
        
        # 计算指标
        r2_train = r2_score(y_train, y_pred_train)
        r2_test = r2_score(y_test, y_pred_test)
        rmse_train = np.sqrt(mean_squared_error(y_train, y_pred_train))
        rmse_test = np.sqrt(mean_squared_error(y_test, y_pred_test))
        
        print(f"\n{model_name} 性能:")
        print(f"  训练集 - R²: {r2_train:.3f}, RMSE: {rmse_train:.1f} kW")
        print(f"  测试集 - R²: {r2_test:.3f}, RMSE: {rmse_test:.1f} kW")
        
        return {
            'model': model,
            'y_test': y_test,
            'y_pred_test': y_pred_test,
            'y_train': y_train,
            'y_pred_train': y_pred_train,
            'r2_test': r2_test,
            'rmse_test': rmse_test,
            'r2_train': r2_train,
            'rmse_train': rmse_train,
            'features': features
        }
    
    def train_all_models(self):
        """训练所有三个模型"""
        print("\n" + "="*60)
        print("开始训练所有模型方案")
        print("="*60)
        
        self.results = {}
        for model_name in self.model_configs.keys():
            self.results[model_name] = self.train_single_model(model_name)
        
        return self.results
    
    def create_panel_a(self):
        """创建Panel A: 三个并排散点图"""
        print("\n=== 创建 Panel A 散点图 ===")
        
        # 创建图形：1行3列
        fig, axes = plt.subplots(1, 3, figsize=(30, 9))
        
        model_names = ['Hub-height', 'Standard REWS', 'Extended REWS']
        subplot_labels = ['(a)', '(b)', '(c)']
        
        for idx, (ax, model_name, subplot_label) in enumerate(zip(axes, model_names, subplot_labels)):
            self._plot_single_scatter(ax, model_name, subplot_label)
        
        # 调整子图间距
        plt.subplots_adjust(left=0.06, right=0.98, top=0.92, bottom=0.12, wspace=0.25)
        
        # 保存图形
        dataset_name = self.data_path.stem  # 获取数据集名称
        save_path = self.output_dir / f'panel_a_performance_comparison_{dataset_name}.png'
        plt.savefig(save_path, dpi=600, bbox_inches='tight', facecolor='white')
        plt.savefig(save_path.with_suffix('.pdf'), dpi=600, bbox_inches='tight', facecolor='white')
        
        print(f"\nPanel A 已保存到: {save_path}")
        plt.show()
    
    def _plot_single_scatter(self, ax, model_name, subplot_label):
        """绘制单个散点图"""
        result = self.results[model_name]
        config = self.model_configs[model_name]
        
        y_test = result['y_test']
        y_pred = result['y_pred_test']
        r2 = result['r2_test']
        rmse = result['rmse_test']
        color = config['color']
        
        # 绘制散点
        ax.scatter(y_test, y_pred, c=color, alpha=0.5, s=20, 
                  edgecolors='none', label='Predictions')
        
        # 计算坐标轴范围
        min_val = min(y_test.min(), y_pred.min())
        max_val = max(y_test.max(), y_pred.max())
        margin = (max_val - min_val) * 0.05
        axis_min = max(0, min_val - margin)
        axis_max = max_val + margin
        
        # 绘制1:1理想线
        ax.plot([axis_min, axis_max], [axis_min, axis_max], 
               'k--', linewidth=2, alpha=0.8, label='1:1 Line')
        
        # 设置坐标轴
        ax.set_xlim(axis_min, axis_max)
        ax.set_ylim(axis_min, axis_max)
        ax.set_aspect('equal', adjustable='box')
        
        # 设置标签
        ax.set_xlabel('Observed Power (kW)', fontsize=25, fontweight='normal')
        ax.set_ylabel('Predicted Power (kW)', fontsize=25, fontweight='normal')
        
        # 设置标题
        title = f"{subplot_label} {config['label']}"
        ax.set_title(title, fontsize=25, fontweight='normal', pad=15)
        
        # 添加性能指标文本框
        textstr = f'$R^2$ = {r2:.3f}\nRMSE = {rmse:.1f} kW'
        ax.text(0.05, 0.95, textstr, transform=ax.transAxes, 
               fontsize=22, fontweight='normal', va='top', ha='left',
               bbox=dict(boxstyle='round,pad=0.5', facecolor='white', 
                        alpha=0.9, edgecolor='gray', linewidth=1.5))
        
        # 美化图表
        ax.grid(True, alpha=0.3, linestyle='--')
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['left'].set_linewidth(1.5)
        ax.spines['bottom'].set_linewidth(1.5)
        
        # 设置刻度参数
        ax.tick_params(axis='both', which='major', labelsize=20, width=1.5, length=6)
    
    def save_performance_summary(self):
        """保存性能摘要到CSV"""
        print("\n=== 保存性能摘要 ===")
        
        summary_data = []
        for model_name in self.model_configs.keys():
            result = self.results[model_name]
            summary_data.append({
                'Model': model_name,
                'Features': ', '.join(result['features']),
                'N_Features': len(result['features']),
                'R2_Train': result['r2_train'],
                'RMSE_Train': result['rmse_train'],
                'R2_Test': result['r2_test'],
                'RMSE_Test': result['rmse_test']
            })
        
        summary_df = pd.DataFrame(summary_data)
        
        dataset_name = self.data_path.stem
        summary_path = self.output_dir / f'panel_a_performance_summary_{dataset_name}.csv'
        summary_df.to_csv(summary_path, index=False)
        
        print(f"\n性能摘要已保存到: {summary_path}")
        print("\n" + "="*80)
        print(summary_df.to_string(index=False))
        print("="*80)
    
    def run_analysis(self):
        """运行完整分析流程"""
        print("\n" + "="*80)
        print("Figure 3 - Panel A: REWS方案性能对比分析")
        print("="*80)
        
        # 1. 加载数据
        self.load_and_clean_data()
        
        # 2. 训练所有模型
        self.train_all_models()
        
        # 3. 创建可视化
        self.create_panel_a()
        
        # 4. 保存摘要
        self.save_performance_summary()
        
        print("\n" + "="*80)
        print("分析完成！")
        print("="*80)
        
        return self.results


def compare_datasets():
    """对比两个数据集的结果"""
    print("\n" + "="*80)
    print("开始对比两个数据集")
    print("="*80)
    
    data_paths = [
        "/Users/xiaxin/work/WindForecast_Project/01_Data/processed/imputed_data/changma_imputed_complete.csv",
        "/Users/xiaxin/work/WindForecast_Project/01_Data/processed/matched_data/changma_matched.csv"
    ]
    
    output_dir = Path('/Users/xiaxin/work/WindForecast_Project/03_Results/re-plot-figures/figure-3/')
    
    all_results = {}
    
    for data_path in data_paths:
        dataset_name = Path(data_path).stem
        print(f"\n{'='*80}")
        print(f"处理数据集: {dataset_name}")
        print(f"{'='*80}")
        
        analyzer = REWSComparisonAnalyzer(data_path, output_dir)
        results = analyzer.run_analysis()
        all_results[dataset_name] = results
    
    # 创建数据集对比摘要
    print("\n" + "="*80)
    print("数据集性能对比")
    print("="*80)
    
    comparison_data = []
    for dataset_name, results in all_results.items():
        for model_name, result in results.items():
            comparison_data.append({
                'Dataset': dataset_name,
                'Model': model_name,
                'R2_Test': result['r2_test'],
                'RMSE_Test': result['rmse_test']
            })
    
    comparison_df = pd.DataFrame(comparison_data)
    comparison_path = output_dir / 'dataset_comparison_summary.csv'
    comparison_df.to_csv(comparison_path, index=False)
    
    print(f"\n数据集对比摘要已保存到: {comparison_path}")
    print("\n" + comparison_df.to_string(index=False))
    
    return all_results


if __name__ == "__main__":
    # 运行两个数据集的对比分析
    all_results = compare_datasets()