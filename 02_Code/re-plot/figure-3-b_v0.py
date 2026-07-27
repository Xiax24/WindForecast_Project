#!/usr/bin/env python3
"""
Figure 3 - 辅助可视化: 三模型预测散点叠加图
左图: Free-flow区三个模型的预测散点叠加
右图: Wake区三个模型的预测散点叠加
用于直观查看不同模型的预测分布差异
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score, mean_squared_error
from scipy.stats import norm
import lightgbm as lgb
import warnings
from pathlib import Path
from qc_common import qc_operating_range, sector_mask, SECTOR_FREE, SECTOR_WAKE, report_sectors

warnings.filterwarnings('ignore')


def diebold_mariano_test(target, pred1, pred2, h=1):
    """Diebold-Mariano Test"""
    e1 = (target - pred1)**2
    e2 = (target - pred2)**2
    d = e1 - e2
    
    d_mean = np.mean(d)
    n = len(d)
    
    def autocovariance(xi, k):
        n = len(xi)
        if k >= n:
            return 0
        return np.sum((xi[:n-k] - np.mean(xi)) * (xi[k:] - np.mean(xi))) / n

    var_d = autocovariance(d, 0)
    for i in range(1, h):
        var_d += 2 * autocovariance(d, i)
    
    if var_d <= 0:
        return 0, 1.0
    
    dm_stat = d_mean / np.sqrt(var_d / n)
    p_value = 1 - norm.cdf(abs(dm_stat))
    
    return dm_stat, p_value

# 设置matplotlib参数
plt.rcParams.update({
    'font.family': ['Arial', 'DejaVu Sans'],
    'font.size': 28,
    'axes.linewidth': 1.5,
    'figure.dpi': 300,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
    'savefig.facecolor': 'white'
})


class ScatterOverlayVisualizer:
    """三模型散点叠加可视化器"""
    
    def __init__(self, data_path, output_dir):
        self.data_path = Path(data_path)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # 风向列
        self.wind_dir_columns = [
            'obs_wind_direction_10m',
            'obs_wind_direction_30m',
            'obs_wind_direction_50m',
            'obs_wind_direction_70m'
        ]
        
        # 模型配置
        self.model_configs = {
            'Hub-height': {
                'features': ['obs_wind_speed_70m'],
                'label': 'Hub-height',
                'color': '#B4B4B3',
                'alpha': 0.7,
                'marker': 'o',
                's': 30
            },
            'Standard REWS': {
                'features': ['obs_wind_speed_30m', 'obs_wind_speed_50m', 'obs_wind_speed_70m'],
                'label': 'Standard REWS',
                'color': "#5AEFFF",  #'#7CB9FF',
                'alpha': 0.5,
                'marker': 'o',
                's': 30
            },
            'Extended REWS': {
                'features': ['obs_wind_speed_10m', 'obs_wind_speed_30m', 
                           'obs_wind_speed_50m', 'obs_wind_speed_70m'],
                'label': 'Extended REWS',
                'color': '#893CE7',  # '#f41111',
                'alpha': 0.3,
                'marker': 'o',
                's': 30
            }
        }
        
        # LightGBM参数
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
        
        # 嵌入柱状图显著性标注配置 - 可手动调整
        self.inset_annotation_configs = {
            'Free-flow': {
                'HH_vs_SR': {
                    'y_offset': 0.18,
                    'line_color': '#C7C7CA',
                    'star_color': '#C7C7CA',
                    'star_size': 15
                },
                'SR_vs_ER': {
                    'y_offset': 0.19,
                    'line_color': '#C7C7CA',
                    'star_color': '#C7C7CA',
                    'star_size': 15
                },
                'HH_vs_ER': {
                    'y_offset': 0.30,
                    'line_color': '#C7C7CA',
                    'star_color': '#C7C7CA',
                    'star_size': 15
                }
            },
            'Wake': {
                'HH_vs_SR': {
                    'y_offset': 0.18,
                    'line_color': '#C7C7CA',
                    'star_color': '#C7C7CA',
                    'star_size': 15
                },
                'SR_vs_ER': {
                    'y_offset': 0.19,
                    'line_color': '#C7C7CA',
                    'star_color': '#C7C7CA',
                    'star_size': 15
                },
                'HH_vs_ER': {
                    'y_offset': 0.30,
                    'line_color': '#C7C7CA',
                    'star_color': '#C7C7CA',
                    'star_size': 15
                }
            }
        }
    
    def strict_direction_mask(self, data, direction_range):
        """严格筛选：所有高度风向都在指定区间"""
        min_deg, max_deg = direction_range
        mask = np.ones(len(data), dtype=bool)
        
        for col in self.wind_dir_columns:
            wd = data[col].values
            if min_deg > max_deg:
                h_mask = (wd >= min_deg) | (wd <= max_deg)
            else:
                h_mask = (wd >= min_deg) & (wd <= max_deg)
            mask = mask & h_mask & ~np.isnan(wd)
        
        return mask
    
    def load_and_classify_data(self):
        """加载数据并进行风向分类"""
        print("=== 加载数据并进行严格风向分类 ===")
        
        data = pd.read_csv(self.data_path)
        print(f"原始数据形状: {data.shape}")
        
        # 检查必要列
        all_features = set()
        for config in self.model_configs.values():
            all_features.update(config['features'])
        
        required_cols = list(all_features) + self.wind_dir_columns + ['power']
        
        # 基础清理
        data_clean = data[data['power'] >= 0].copy()
        data_clean = data_clean.dropna(subset=required_cols)
        
        # 风速筛选
        wind_speed_condition = (
            (data_clean['obs_wind_speed_70m'] >= 3.0) & 
            (data_clean['obs_wind_speed_70m'] <= 25.0)
        )
        data_clean = data_clean[wind_speed_condition].copy()
        
        # 严格风向分类
        mask_free = self.strict_direction_mask(data_clean, (225, 315))  # 西风
        mask_wake = self.strict_direction_mask(data_clean, (45, 135))   # 东风
        
        self.data_all = data_clean
        self.data_free = data_clean[mask_free].copy()
        self.data_wake = data_clean[mask_wake].copy()
        
        print(f"Free-flow: {len(self.data_free)} 条")
        print(f"Wake: {len(self.data_wake)} 条")
        
        return data_clean, self.data_free, self.data_wake
    
    def train_and_predict(self, model_name, data_sector):
        """训练模型并预测"""
        config = self.model_configs[model_name]
        features = config['features']
        
        # 在全数据上训练
        X_train = self.data_all[features].values
        y_train = self.data_all['power'].values
        
        model = lgb.LGBMRegressor(**self.lgb_params)
        model.fit(X_train, y_train)
        
        # 在特定扇区上预测
        X_sector = data_sector[features].values
        y_sector = data_sector['power'].values
        y_pred = model.predict(X_sector)
        
        # 计算指标
        r2 = r2_score(y_sector, y_pred)
        rmse = np.sqrt(mean_squared_error(y_sector, y_pred))
        
        return {
            'y_true': y_sector,
            'y_pred': y_pred,
            'r2': r2,
            'rmse': rmse,
            'model': model  # 保存模型供后续使用
        }
    
    def create_scatter_overlay(self):
        """创建叠加散点图（带嵌入式柱状图）"""
        print("\n=== 创建散点叠加图（带嵌入柱状图）===")
        
        # 创建图形 - 使用gridspec_kw直接控制间距
        fig, axes = plt.subplots(1, 2, figsize=(20, 9), 
                                gridspec_kw={'wspace': 0.01})  # 直接在这里设置间距
        
        sectors = [
            ('Free-flow', self.data_free, axes[0], 'Free-stream'),
            ('Wake', self.data_wake, axes[1], 'Wake Regime')
        ]
        
        # 存储结果用于嵌入柱状图
        all_results = {'Free-flow': {}, 'Wake': {}}
        
        for sector_name, data_sector, ax, title in sectors:
            print(f"\n处理 {sector_name} 区...")
            
            # 计算坐标轴范围
            y_true_all = data_sector['power'].values
            axis_min = 0
            axis_max = max(y_true_all.max(), 2000)  # 预留一些空间
            
            # 绘制1:1理想线
            ax.plot([axis_min, axis_max], [axis_min, axis_max], 
                   'k--', linewidth=2.5, alpha=0.6, 
                   label='1:1 Line', zorder=1)
            
            # 为每个模型绘制散点
            for model_name in ['Hub-height', 'Standard REWS', 'Extended REWS']:
                config = self.model_configs[model_name]
                result = self.train_and_predict(model_name, data_sector)
                
                y_true = result['y_true']
                y_pred = result['y_pred']
                r2 = result['r2']
                rmse = result['rmse']
                
                # 保存结果
                all_results[sector_name][model_name] = result
                
                # print(f"  {model_name}: R²={r2:.3f}, RMSE={rmse:.1f} kW")
                
                # 绘制散点
                ax.scatter(y_true, y_pred,
                          c=config['color'],
                          alpha=config['alpha'],
                          s=config['s'],
                          marker=config['marker'],
                          edgecolors='none',
                          label=f"{config['label']}",
                          zorder=2)
            
            # 设置坐标轴
            ax.set_xlim(axis_min, 200)
            ax.set_ylim(axis_min, 200)
            # 设置刻度间隔（每25一个刻度）
            ax.set_xticks(np.arange(0, 201, 25))  # X轴：0, 25, 50, 75, 100, 125, 150, 175, 200
            ax.set_yticks(np.arange(0, 201, 25))  # Y轴：0, 25, 50, 75, 100, 125, 150, 175, 200
            ax.set_aspect('equal', adjustable='box')
            
            ax.set_xlabel('Observed Power (MW)', fontsize=28, fontweight='normal')
            if sector_name == 'Free-flow':
                ax.set_ylabel('Modeled Power (MW)', fontsize=28, fontweight='normal')
            
            ax.set_title(title, fontsize=28, fontweight='bold', pad=15, loc='center')
            
            # 图例
            ax.legend(loc='lower right', fontsize=20, frameon=True, 
                     fancybox=False, edgecolor='gray', framealpha=0.9)
            
            # 网格和美化
            ax.grid(False)
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            ax.spines['left'].set_linewidth(2)
            ax.spines['bottom'].set_linewidth(2)
            ax.tick_params(axis='both', which='major', labelsize=24, 
                          width=2, length=6)
            
            # 添加嵌入式柱状图
            self._add_inset_barplot(ax, sector_name, all_results)
        
        # 保存（不使用bbox_inches='tight'）
        dataset_name = self.data_path.stem
        save_path = self.output_dir / f'final_3_b_{dataset_name}.png'
        plt.savefig(save_path, dpi=600, facecolor='white')
        plt.savefig(save_path.with_suffix('.pdf'), dpi=600, facecolor='white')
        plt.close()
    
    def _add_inset_barplot(self, ax, sector_name, all_results):
        """在散点图中添加嵌入式柱状图（带完整显著性标注，无标题）"""
        # 创建inset axes（位于左上角）
        inset_ax = ax.inset_axes([0.12, 0.65, 0.38, 0.35])
        
        # 提取该扇区的R²数据
        model_names = ['Hub-height', 'Standard REWS', 'Extended REWS']
        r2_values = [all_results[sector_name][m]['r2'] for m in model_names]
        
        # 绘制柱状图
        x_pos = np.arange(len(model_names))
        width = 0.65
        
        for i, (model_name, r2) in enumerate(zip(model_names, r2_values)):
            config = self.model_configs[model_name]
            bar = inset_ax.bar(x_pos[i], r2, width,
                              color=config['color'],
                              alpha=0.85,
                              edgecolor='white',
                              linewidth=1.2)
            
            # 标注R²值
            inset_ax.text(x_pos[i], r2 + 0.015,
                         f'{r2:.2f}',
                         ha='center', va='bottom',
                         fontsize=18, fontweight='bold',
                         color=config['color'])
        
        # 添加所有显著性标注（6个）
        self._add_all_inset_significance(inset_ax, x_pos, width, all_results, sector_name)
        
        # 设置inset坐标轴（不设置title）
        inset_ax.set_xticks(x_pos)
        inset_ax.set_xticklabels(['HH', 'SR', 'ER'], fontsize=15, fontweight='bold')
        inset_ax.set_ylabel('$R^2$', fontsize=15, fontweight='bold')
        inset_ax.set_ylim(0, 1.15)  # 增大y轴范围以容纳标注
        
        # 美化inset
        inset_ax.grid(False)
        inset_ax.spines['top'].set_visible(False)
        inset_ax.spines['right'].set_visible(False)
        inset_ax.tick_params(axis='both', which='major', labelsize=15)
        inset_ax.set_facecolor('white')
        inset_ax.patch.set_alpha(0.95)
    
    def _add_all_inset_significance(self, ax, x_pos, width, all_results, sector_name):
        """为嵌入柱状图添加所有显著性标注（使用配置字典）"""
        model_names = ['Hub-height', 'Standard REWS', 'Extended REWS']
        
        # 获取该扇区的所有R²值
        r2_values = [all_results[sector_name][m]['r2'] for m in model_names]
        
        # 定义对比关系：(测试名称, 左侧索引, 右侧索引)
        comparisons = [
            ('HH_vs_SR', 0, 1),
            ('SR_vs_ER', 1, 2),
            ('HH_vs_ER', 0, 2)
        ]
        
        for test_name, left_idx, right_idx in comparisons:
            # 检查是否有配置
            if test_name not in self.inset_annotation_configs[sector_name]:
                print(f"警告: 缺少 {sector_name} - {test_name} 的配置")
                continue
            
            config = self.inset_annotation_configs[sector_name][test_name]
            
            # 获取对应模型的预测结果
            left_model = model_names[left_idx]
            right_model = model_names[right_idx]
            
            y_true = all_results[sector_name][left_model]['y_true']
            left_pred = all_results[sector_name][left_model]['y_pred']
            right_pred = all_results[sector_name][right_model]['y_pred']
            
            # 执行DM检验
            dm_stat, p_val = diebold_mariano_test(y_true, left_pred, right_pred)
            
            # 确定显著性符号
            if p_val < 0.01:
                stars = '***'
            elif p_val < 0.05:
                stars = '**'
            elif p_val < 0.1:
                stars = '*'
            else:
                stars = 'n.s.'
            
            # 计算标注位置
            x_left = x_pos[left_idx]
            x_right = x_pos[right_idx]
            
            # 基于实际柱子高度 + 配置的偏移
            base_height = max(r2_values[left_idx], r2_values[right_idx])
            y_line = base_height + config['y_offset']
            
            # 绘制连接线
            ax.plot([x_left, x_left, x_right, x_right], 
                   [y_line-0.008, y_line, y_line, y_line-0.008], 
                   color=config['line_color'], 
                   lw=1.0, 
                   zorder=4)
            
            # 标注显著性符号
            ax.text((x_left + x_right)/2, y_line + 0.005, 
                   stars, 
                   ha='center', va='bottom', 
                   fontsize=config['star_size'], 
                   fontweight='bold',
                   color=config['star_color'])
    
    def run(self):
        self.load_and_classify_data()
        self.create_scatter_overlay()



if __name__ == "__main__":
    DATA_PATH = "/Users/xiaxin/work/WindForecast_Project/01_Data/processed/matched_data/changma_matched.csv"
    OUTPUT_DIR = "/Users/xiaxin/work/WindForecast_Project/03_Results/re-plot-figures/figure-3/"
    
    visualizer = ScatterOverlayVisualizer(DATA_PATH, OUTPUT_DIR)
    visualizer.run()