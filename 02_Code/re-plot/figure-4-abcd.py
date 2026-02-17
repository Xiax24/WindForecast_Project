#!/usr/bin/env python3
"""
Figure 4b - 四种建模策略对比
策略1: HH (Hub Height) - 只用70m
策略2: SR (Standard Retrieval) - 用30m, 50m, 70m
策略3: ER (Extended Retrieval) - 用10m, 30m, 50m, 70m
策略4: Wind-Direction-Aware Strategy - 根据NWP预测风向动态选择模型
  - Free-stream (225-315°): WS 10m, 70m, 50m + Temp 10m
  - Wake (45-135°): WS 10m, 30m, 70m + Temp 10m
  - Others: WS 10m, 30m, 50m, 70m + Temp 10m
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error
from scipy import stats
import lightgbm as lgb
import warnings
from pathlib import Path

warnings.filterwarnings('ignore')

# ============================================================================
# 样式设置
# ============================================================================
plt.rcParams.update({
    'font.family': ['Arial', 'DejaVu Sans'],
    'font.size': 20,  # 增大基础字体
    'axes.linewidth': 1.5,
    'figure.dpi': 300,
})


# ============================================================================
# DM检验
# ============================================================================
def diebold_mariano_test(errors1, errors2, h=1):
    """Diebold-Mariano检验"""
    d = errors1**2 - errors2**2
    mean_d = np.mean(d)
    
    def autocovariance(series, lag):
        n = len(series)
        mean = np.mean(series)
        return np.sum((series[:n-lag] - mean) * (series[lag:] - mean)) / n
    
    gamma_0 = autocovariance(d, 0)
    variance = gamma_0
    for lag in range(1, h):
        gamma_lag = autocovariance(d, lag)
        variance += 2 * (1 - lag/(h+1)) * gamma_lag
    
    dm_stat = mean_d / np.sqrt(variance / len(d))
    p_value = 1 - stats.norm.cdf(dm_stat)
    
    return dm_stat, p_value


# ============================================================================
# 主类：四种策略对比
# ============================================================================
class FourStrategyComparison:
    """四种建模策略的完整对比"""
    
    def __init__(self, data_path, output_dir, nwp_source='ec'):
        """
        Parameters:
        -----------
        nwp_source: 'ec' or 'gfs' - 选择用哪个NWP系统的风向进行分类
        """
        self.data_path = Path(data_path)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.nwp_source = nwp_source
        
        # 策略配置
        self.strategies = {
            'HH': {
                'name': 'HH',
                'wind_features': [f'{nwp_source}_wind_speed_70m'],
                # 为了与WDA策略保持公平比较，加入Temp 10m
                'other_features': [f'{nwp_source}_temperature_10m']
            },
            'SR': {
                'name': 'SR',
                'wind_features': [
                    f'{nwp_source}_wind_speed_30m',
                    f'{nwp_source}_wind_speed_50m',
                    f'{nwp_source}_wind_speed_70m'
                ],
                # 为了与WDA策略保持公平比较，加入Temp 10m
                'other_features': [f'{nwp_source}_temperature_10m']
            },
            'ER': {
                'name': 'ER',
                'wind_features': [
                    f'{nwp_source}_wind_speed_10m',
                    f'{nwp_source}_wind_speed_30m',
                    f'{nwp_source}_wind_speed_50m',
                    f'{nwp_source}_wind_speed_70m'
                ],
                # 为了与WDA策略保持公平比较，加入Temp 10m
                'other_features': [f'{nwp_source}_temperature_10m']
            },
            'WDA': {
                'name': 'WDA',
                'type': 'adaptive',  # 标记为自适应策略
                'models': {
                    'free': {
                        'wind_features': [
                            f'{nwp_source}_wind_speed_10m',
                            f'{nwp_source}_wind_speed_50m',
                            f'{nwp_source}_wind_speed_70m'
                        ],
                        'other_features': [f'{nwp_source}_temperature_10m']
                    },
                    'wake': {
                        'wind_features': [
                            f'{nwp_source}_wind_speed_10m',
                            f'{nwp_source}_wind_speed_30m',
                            f'{nwp_source}_wind_speed_70m'
                        ],
                        'other_features': [f'{nwp_source}_temperature_10m']
                    },
                    'others': {
                        'wind_features': [
                            f'{nwp_source}_wind_speed_10m',
                            f'{nwp_source}_wind_speed_30m',
                            f'{nwp_source}_wind_speed_50m',
                            f'{nwp_source}_wind_speed_70m'
                        ],
                        'other_features': [f'{nwp_source}_temperature_10m']
                    }
                }
            }
        }
        
        # LightGBM参数
        self.lgb_params = {
            'objective': 'regression',
            'metric': 'rmse',
            'num_leaves': 31,
            'learning_rate': 0.05,
            'feature_fraction': 0.9,
            'bagging_fraction': 0.8,
            'bagging_freq': 5,
            'verbose': -1,
            'random_state': 42
        }
    
    def classify_wind_direction(self, wind_dir):
        """根据风向分类"""
        # Free-stream: 225-315°
        free_mask = (wind_dir >= 225) & (wind_dir <= 315)
        # Wake: 45-135°
        wake_mask = (wind_dir >= 45) & (wind_dir <= 135)
        # Others: 其他
        others_mask = ~(free_mask | wake_mask)
        
        direction_type = np.empty(len(wind_dir), dtype='U10')
        direction_type[free_mask] = 'free'
        direction_type[wake_mask] = 'wake'
        direction_type[others_mask] = 'others'
        
        return direction_type
    
    def load_and_split_data(self):
        """加载数据并进行80/20划分"""
        print("="*70)
        print(f"加载数据 (使用 {self.nwp_source.upper()} 风向进行分类)")
        print("="*70)
        
        df = pd.read_csv(self.data_path)
        
        # 基础清理
        df = df[df['power'] >= 0].copy()
        
        # 收集所有策略会用到的列
        required_cols = set()
        
        # HH, SR, ER策略的列
        for strategy in ['HH', 'SR', 'ER']:
            config = self.strategies[strategy]
            required_cols.update(config['wind_features'])
            required_cols.update(config['other_features'])
        
        # WDA策略的列
        wda_config = self.strategies['WDA']
        for model_type in ['free', 'wake', 'others']:
            required_cols.update(wda_config['models'][model_type]['wind_features'])
            required_cols.update(wda_config['models'][model_type]['other_features'])
        
        # 添加对应的观测列
        obs_cols = set()
        for nwp_col in required_cols:
            obs_col = nwp_col.replace(f'{self.nwp_source}_', 'obs_')
            obs_cols.add(obs_col)
        
        # 添加必需的其他列
        required_cols.add(f'{self.nwp_source}_wind_direction_70m')
        required_cols.add('power')
        
        # 合并所有需要的列
        all_needed_cols = list(required_cols | obs_cols)
        
        # 只保留存在的列
        available_cols = [col for col in all_needed_cols if col in df.columns]
        
        print(f"需要的列数: {len(all_needed_cols)}")
        print(f"实际存在的列数: {len(available_cols)}")
        
        # 只对存在的列进行缺失值清理
        df = df.dropna(subset=available_cols)
        
        # 风速筛选（如果obs_wind_speed_70m存在）
        if 'obs_wind_speed_70m' in df.columns:
            df = df[(df['obs_wind_speed_70m'] >= 3.0) & 
                    (df['obs_wind_speed_70m'] <= 25.0)].copy()
        
        print(f"清洗后数据量: {len(df)}")
        
        # 80/20划分
        train_df, test_df = train_test_split(
            df, test_size=0.2, random_state=42, shuffle=False
        )
        
        print(f"训练集: {len(train_df)} (80%)")
        print(f"测试集: {len(test_df)} (20%)")
        
        # 为训练集和测试集添加风向分类
        train_df['wind_direction_type'] = self.classify_wind_direction(
            train_df[f'{self.nwp_source}_wind_direction_70m'].values
        )
        test_df['wind_direction_type'] = self.classify_wind_direction(
            test_df[f'{self.nwp_source}_wind_direction_70m'].values
        )
        
        # 统计训练集风向分布
        print(f"\n训练集风向分布:")
        for dtype in ['free', 'wake', 'others']:
            count = (train_df['wind_direction_type'] == dtype).sum()
            pct = count / len(train_df) * 100
            print(f"  {dtype}: {count} ({pct:.1f}%)")
        
        # 统计测试集风向分布
        print(f"\n测试集风向分布:")
        for dtype in ['free', 'wake', 'others']:
            count = (test_df['wind_direction_type'] == dtype).sum()
            pct = count / len(test_df) * 100
            print(f"  {dtype}: {count} ({pct:.1f}%)")
        
        self.train_df = train_df
        self.test_df = test_df
        
        return train_df, test_df
    
    def two_stage_train(self, train_data, wind_features, other_features):
        """
        两阶段训练
        阶段1: 每个NWP气象要素 -> 对应的观测气象要素 (逐个订正)
        阶段2: 所有订正后的气象要素 -> 功率
        """
        all_features = wind_features + other_features
        correction_models = {}
        obs_features_list = []
        
        # 阶段1: 逐个气象要素订正
        for nwp_feature in all_features:
            # 确定对应的观测特征名
            obs_feature = nwp_feature.replace(f'{self.nwp_source}_', 'obs_')
            
            if obs_feature not in train_data.columns:
                print(f"  警告: 找不到观测特征 {obs_feature}，跳过")
                continue
            
            # 检查是否有缺失值
            if train_data[nwp_feature].isna().any() or train_data[obs_feature].isna().any():
                print(f"  警告: {nwp_feature} 或 {obs_feature} 包含NaN，跳过")
                continue
            
            # 训练单变量订正模型: NWP -> Obs
            X = train_data[[nwp_feature]].values
            y = train_data[obs_feature].values
            
            model = lgb.LGBMRegressor(**self.lgb_params, n_estimators=100)
            model.fit(X, y)
            
            correction_models[nwp_feature] = {
                'model': model,
                'obs_feature': obs_feature
            }
            obs_features_list.append(obs_feature)
        
        if len(correction_models) == 0:
            raise ValueError("没有可用的订正模型，请检查数据")
        
        # 阶段2: 功率转换模型
        # 使用所有观测气象要素训练
        X_power = train_data[obs_features_list].values
        y_power = train_data['power'].values
        
        # 再次检查是否有NaN
        if np.isnan(X_power).any() or np.isnan(y_power).any():
            raise ValueError("功率模型训练数据包含NaN")
        
        power_model = lgb.LGBMRegressor(**self.lgb_params, n_estimators=100)
        power_model.fit(X_power, y_power)
        
        return correction_models, power_model
    
    def two_stage_predict(self, test_data, correction_models, power_model, wind_features, other_features):
        """
        两阶段预测
        阶段1: 用订正模型逐个订正每个气象要素
        阶段2: 用订正后的所有气象要素预测功率
        """
        all_features = wind_features + other_features
        
        # 阶段1: 逐个订正
        corrected_features = []
        
        for nwp_feature in all_features:
            if nwp_feature not in correction_models:
                continue
            
            model_info = correction_models[nwp_feature]
            model = model_info['model']
            
            # 订正
            X = test_data[[nwp_feature]].values
            corrected = model.predict(X)
            
            corrected_features.append(corrected)
        
        # 阶段2: 用所有订正后的要素预测功率
        X_power = np.column_stack(corrected_features)
        power_pred = power_model.predict(X_power)
        
        return power_pred
    
    def train_strategy_hh_sr_er(self, strategy_name):
        """训练HH/SR/ER策略（单一模型）"""
        print(f"\n{'='*70}")
        print(f"训练策略: {strategy_name}")
        print(f"{'='*70}")
        
        config = self.strategies[strategy_name]
        wind_features = config['wind_features']
        other_features = config['other_features']
        
        print(f"特征: {wind_features + other_features}")
        
        # 在全部训练集上训练
        correction_models, power_model = self.two_stage_train(
            self.train_df, wind_features, other_features
        )
        
        # 在测试集上预测
        y_pred = self.two_stage_predict(
            self.test_df, correction_models, power_model, 
            wind_features, other_features
        )
        
        y_true = self.test_df['power'].values
        
        # 计算指标
        r2 = r2_score(y_true, y_pred)
        rmse = np.sqrt(mean_squared_error(y_true, y_pred))
        mae = np.mean(np.abs(y_true - y_pred))
        corr = np.corrcoef(y_true, y_pred)[0, 1]
        
        print(f"\n测试集结果:")
        print(f"  R²: {r2:.4f}")
        print(f"  RMSE: {rmse:.2f} kW")
        print(f"  MAE: {mae:.2f} kW")
        print(f"  Correlation: {corr:.4f}")
        
        return {
            'y_true': y_true,
            'y_pred': y_pred,
            'r2': r2,
            'rmse': rmse,
            'mae': mae,
            'correlation': corr
        }
    
    def train_strategy_wda(self):
        """训练Wind-Direction-Aware策略（三个模型）"""
        print(f"\n{'='*70}")
        print(f"训练策略: Wind-Direction-Aware (WDA)")
        print(f"{'='*70}")
        
        config = self.strategies['WDA']
        models_dict = {}
        
        # 训练三个子模型
        for model_type in ['free', 'wake', 'others']:
            print(f"\n--- 训练 {model_type.upper()} 模型 ---")
            
            model_config = config['models'][model_type]
            wind_features = model_config['wind_features']
            other_features = model_config['other_features']
            
            # 筛选该类型的训练数据
            train_subset = self.train_df[
                self.train_df['wind_direction_type'] == model_type
            ].copy()
            
            print(f"训练样本数: {len(train_subset)}")
            print(f"特征: {wind_features + other_features}")
            
            # 训练
            correction_models, power_model = self.two_stage_train(
                train_subset, wind_features, other_features
            )
            
            models_dict[model_type] = {
                'correction_models': correction_models,
                'power_model': power_model,
                'wind_features': wind_features,
                'other_features': other_features
            }
        
        # 在测试集上预测（动态选择模型）
        print(f"\n--- 测试集预测（动态模型选择）---")
        
        y_pred_all = np.zeros(len(self.test_df))
        
        for model_type in ['free', 'wake', 'others']:
            # 筛选该类型的测试数据
            mask = self.test_df['wind_direction_type'] == model_type
            test_subset = self.test_df[mask].copy()
            
            if len(test_subset) == 0:
                continue
            
            print(f"{model_type}: {len(test_subset)} 条")
            
            # 使用对应模型预测
            model_info = models_dict[model_type]
            y_pred = self.two_stage_predict(
                test_subset,
                model_info['correction_models'],
                model_info['power_model'],
                model_info['wind_features'],
                model_info['other_features']
            )
            
            # 填充到完整预测数组
            y_pred_all[mask] = y_pred
        
        y_true = self.test_df['power'].values
        
        # 计算指标
        r2 = r2_score(y_true, y_pred_all)
        rmse = np.sqrt(mean_squared_error(y_true, y_pred_all))
        mae = np.mean(np.abs(y_true - y_pred_all))
        corr = np.corrcoef(y_true, y_pred_all)[0, 1]
        
        print(f"\n测试集整体结果:")
        print(f"  R²: {r2:.4f}")
        print(f"  RMSE: {rmse:.2f} kW")
        print(f"  MAE: {mae:.2f} kW")
        print(f"  Correlation: {corr:.4f}")
        
        return {
            'y_true': y_true,
            'y_pred': y_pred_all,
            'r2': r2,
            'rmse': rmse,
            'mae': mae,
            'correlation': corr,
            'models': models_dict
        }
    
    def compare_all_strategies(self):
        """对比所有四种策略"""
        print("\n" + "="*70)
        print("开始四种策略对比实验")
        print("="*70)
        
        results = {}
        
        # 策略1-3: HH, SR, ER
        for strategy_name in ['HH', 'SR', 'ER']:
            results[strategy_name] = self.train_strategy_hh_sr_er(strategy_name)
        
        # 策略4: WDA
        results['WDA'] = self.train_strategy_wda()
        
        # 打印对比表格
        self.print_comparison_table(results)
        
        # 进行显著性检验
        self.perform_significance_tests(results)
        
        return results
    
    def print_comparison_table(self, results):
        """打印对比表格"""
        print("\n" + "="*70)
        print("四种策略性能对比 (测试集)")
        print("="*70)
        print(f"{'策略':<20} {'R²':>10} {'RMSE':>12} {'MAE':>12} {'Corr':>10}")
        print("-"*70)
        
        for strategy_name in ['HH', 'SR', 'ER', 'WDA']:
            result = results[strategy_name]
            strategy_full = self.strategies[strategy_name]['name']
            print(f"{strategy_full:<20} "
                  f"{result['r2']:>10.4f} "
                  f"{result['rmse']:>12.2f} "
                  f"{result['mae']:>12.2f} "
                  f"{result['correlation']:>10.4f}")
        
        print("="*70)
    
    def perform_significance_tests(self, results):
        """进行显著性检验"""
        print("\n" + "="*70)
        print("Diebold-Mariano 显著性检验")
        print("="*70)
        
        y_true = results['HH']['y_true']
        
        comparisons = [
            ('HH', 'SR'),
            ('SR', 'ER'),
            ('ER', 'WDA'),
            ('HH', 'WDA')
        ]
        
        for strategy1, strategy2 in comparisons:
            errors1 = y_true - results[strategy1]['y_pred']
            errors2 = y_true - results[strategy2]['y_pred']
            
            dm_stat, p_val = diebold_mariano_test(errors1, errors2)
            
            sig_marker = ''
            if p_val < 0.01:
                sig_marker = '***'
            elif p_val < 0.05:
                sig_marker = '**'
            elif p_val < 0.1:
                sig_marker = '*'
            else:
                sig_marker = 'n.s.'
            
            rmse_diff = results[strategy1]['rmse'] - results[strategy2]['rmse']
            improve_pct = (rmse_diff / results[strategy1]['rmse']) * 100
            
            print(f"\n{strategy2} vs {strategy1}:")
            print(f"  DM统计量: {dm_stat:.4f}")
            print(f"  p值: {p_val:.4f} {sig_marker}")
            print(f"  RMSE改进: {improve_pct:.2f}%")
        
        print("\n显著性标记: *** p<0.01, ** p<0.05, * p<0.1, n.s. 不显著")
        print("="*70)
        
    def create_visualization(self, results):
        """创建可视化对比图"""
        print("\n创建可视化图表...")
        
        # 导入KDE
        from scipy.stats import gaussian_kde
        
        # 可调参数
        figsize = (20, 12)  # 增加宽度以容纳右边的柱状图
        hspace = 0.13  # 上下间距
        wspace = 0.3   # 左右间距（散点图和柱状图之间）
        left = 0.06    # 左边距
        right = 0.76   # 右边距
        top = 0.95     # 上边距
        bottom = 0.08  # 下边距
        title_fontsize = 25
        label_fontsize = 25
        tick_labelsize = 24
        text_fontsize = 25
        bar_label_fontsize = 20  # 柱状图标签字体
        
        # 坐标轴范围设置
        ylim_min = 0
        ylim_max = 201
        ytick_interval = 25
        xlim_min = 0
        xlim_max = 201
        xtick_interval = 25
        
        # KDE散点图参数
        scatter_size = 10
        scatter_alpha = 0.6
        
        # 使用 GridSpec 精确控制布局：2行3列
        from matplotlib.gridspec import GridSpec
        
        fig = plt.figure(figsize=figsize)
        # 创建2行3列，左边2列给散点图，右边1列给柱状图
        gs = GridSpec(2, 3, figure=fig, 
                    hspace=hspace, wspace=wspace,
                    left=left, right=right, top=top, bottom=bottom,
                    width_ratios=[1, 1, 0.6])  # 右边列稍宽一些以容纳横向柱状图
        
        strategy_names = ['HH', 'SR', 'ER', 'WDA']
        colors = ["#535353", '#A8DADC', "#A0BCD9", "#A86EEE"]
        cmaps = ['coolwarm', 'coolwarm', 'coolwarm', 'coolwarm']
        
        # 计算显著性（相对于HH基准）
        y_true_base = results['HH']['y_true']
        significance_markers = {}
        for strategy_name in strategy_names:
            if strategy_name == 'HH':
                significance_markers[strategy_name] = ''
            else:
                errors_base = y_true_base - results['HH']['y_pred']
                errors_current = y_true_base - results[strategy_name]['y_pred']
                dm_stat, p_val = diebold_mariano_test(errors_base, errors_current)
                
                if p_val < 0.01:
                    significance_markers[strategy_name] = '***'
                elif p_val < 0.05:
                    significance_markers[strategy_name] = '**'
                elif p_val < 0.1:
                    significance_markers[strategy_name] = '*'
                else:
                    significance_markers[strategy_name] = ''
        
        # ========================================================================
        # 左边：2x2散点图
        # ========================================================================
        scatter_positions = [(0, 0), (0, 1), (1, 0), (1, 1)]  # 行列位置
        
        for idx, (strategy_name, color, cmap) in enumerate(zip(strategy_names, colors, cmaps)):
            row, col = scatter_positions[idx]
            ax = fig.add_subplot(gs[row, col])
            
            result = results[strategy_name]
            y_true = result['y_true']
            y_pred = result['y_pred']
            
            # KDE密度散点图
            try:
                if len(y_true) > 10:
                    xy = np.vstack([y_true, y_pred])
                    density = gaussian_kde(xy)(xy)
                    sorted_idx = density.argsort()
                    
                    scatter = ax.scatter(y_true[sorted_idx], y_pred[sorted_idx], 
                                        c=density[sorted_idx], 
                                        s=scatter_size,
                                        cmap=cmap,
                                        alpha=scatter_alpha,
                                        edgecolors='none',
                                        rasterized=True)
                else:
                    ax.scatter(y_true, y_pred, alpha=0.5, s=scatter_size, 
                            c=color, edgecolors='none')
            except Exception as e:
                print(f"  {strategy_name} KDE失败，使用普通散点: {e}")
                ax.scatter(y_true, y_pred, alpha=0.5, s=scatter_size, 
                        c=color, edgecolors='none')
            
            # 1:1线
            ax.plot([xlim_min, xlim_max], [ylim_min, ylim_max], 
                'k--', linewidth=1.5, alpha=0.6)
            
            # 计算相关系数
            correlation = np.corrcoef(y_true, y_pred)[0, 1]
            
            # 统计信息
            sig_mark = significance_markers[strategy_name]
            textstr = (f"$R^2$ = {result['r2']:.2f}{sig_mark}\n"
                    f"RMSE = {result['rmse']:.2f} MW\n"
                    f"$r$ = {correlation:.2f}")
            
            ax.text(0.05, 0.95, textstr, transform=ax.transAxes,
                fontsize=text_fontsize, verticalalignment='top', bbox=None)
            
            # 策略名称
            strategy_text = f"{self.strategies[strategy_name]['name']}"
            ax.text(0.95, 0.05, strategy_text, transform=ax.transAxes,
                fontsize=title_fontsize, verticalalignment='bottom',
                horizontalalignment='right', fontweight='bold', bbox=None)
            
            # 设置坐标轴
            ax.set_xlim(xlim_min, xlim_max)
            ax.set_ylim(ylim_min, ylim_max)
            ax.set_xticks(np.arange(xlim_min, xlim_max + 1, xtick_interval))
            ax.set_yticks(np.arange(ylim_min, ylim_max + 1, ytick_interval))
            ax.tick_params(axis='both', which='major', labelsize=tick_labelsize)
            
            # 标签
            if idx >= 2:  # 下面两幅图
                ax.set_xlabel('Observed Power MW)', fontsize=label_fontsize)
            
            if idx % 2 == 0:  # 左边两幅图
                ax.set_ylabel('Forecasted Power MW)', fontsize=label_fontsize)
            
            ax.grid(True, alpha=0.3)
            ax.set_aspect('equal', adjustable='box')
        
        # ========================================================================
        # 右边：横向柱状图 - Free-stream和Wake条件下的相关系数对比
        # ========================================================================
        
        # 计算每种策略在不同风向条件下的相关系数
        correlations_free = []
        correlations_wake = []
        
        for strategy_name in strategy_names:
            result = results[strategy_name]
            
            # Free-stream条件
            free_mask = self.test_df['wind_direction_type'] == 'free'
            y_true_free = result['y_true'][free_mask]
            y_pred_free = result['y_pred'][free_mask]
            if len(y_true_free) > 1:
                corr_free = np.corrcoef(y_true_free, y_pred_free)[0, 1]
            else:
                corr_free = np.nan
            correlations_free.append(corr_free)
            
            # Wake条件
            wake_mask = self.test_df['wind_direction_type'] == 'wake'
            y_true_wake = result['y_true'][wake_mask]
            y_pred_wake = result['y_pred'][wake_mask]
            if len(y_true_wake) > 1:
                corr_wake = np.corrcoef(y_true_wake, y_pred_wake)[0, 1]
            else:
                corr_wake = np.nan
            correlations_wake.append(corr_wake)
        
        # 上面的横向柱状图：Free-stream
        ax_bar_free = fig.add_subplot(gs[0, 2])
        y_pos = np.arange(len(strategy_names))
        bars_free = ax_bar_free.barh(y_pos, correlations_free, color=colors, alpha=0.8, height=0.6)
        
        # 不显示xlabel（上面的子图）
        # ax_bar_free.set_xlabel('Correlation ($r$)', fontsize=bar_label_fontsize)
        
        # 删除title，改为text到右上角
        # ax_bar_free.set_title('Free-stream', fontsize=bar_label_fontsize, fontweight='bold', pad=10)
        ax_bar_free.text(0.99, 0.99, 'Free-stream', transform=ax_bar_free.transAxes,
                        fontsize=bar_label_fontsize, fontweight='bold',
                        verticalalignment='top', horizontalalignment='right',
                        bbox=None)
        
        ax_bar_free.set_yticks(y_pos)
        ax_bar_free.set_yticklabels(strategy_names, fontsize=bar_label_fontsize)
        ax_bar_free.tick_params(axis='x', labelsize=bar_label_fontsize)
        ax_bar_free.set_xlim([0.4, 1.0])  # 相关系数范围
        ax_bar_free.grid(True, axis='x', alpha=0.3)
        ax_bar_free.invert_yaxis()  # 让第一个策略在上面
        
        # 在柱子右侧标注数值
        for i, (bar, val) in enumerate(zip(bars_free, correlations_free)):
            if not np.isnan(val):
                ax_bar_free.text(val + 0.01, bar.get_y() + bar.get_height()/2, 
                            f'{val:.2f}', ha='left', va='center', 
                            fontsize=bar_label_fontsize)
        
        # 下面的横向柱状图：Wake
        ax_bar_wake = fig.add_subplot(gs[1, 2])
        bars_wake = ax_bar_wake.barh(y_pos, correlations_wake, color=colors, alpha=0.8, height=0.6)
        
        ax_bar_wake.set_xlabel('Correlation ($r$)', fontsize=bar_label_fontsize)
        
        # 删除title，改为text到右上角
        # ax_bar_wake.set_title('Wake-affected', fontsize=bar_label_fontsize, fontweight='bold', pad=10)
        ax_bar_wake.text(0.99, 0.99, 'Wake-affected', transform=ax_bar_wake.transAxes,
                        fontsize=bar_label_fontsize, fontweight='bold',
                        verticalalignment='top', horizontalalignment='right',
                        bbox=None)
        
        ax_bar_wake.set_yticks(y_pos)
        ax_bar_wake.set_yticklabels(strategy_names, fontsize=bar_label_fontsize)
        ax_bar_wake.tick_params(axis='x', labelsize=bar_label_fontsize)
        ax_bar_wake.set_xlim([0.4, 1.0])
        ax_bar_wake.grid(True, axis='x', alpha=0.3)
        ax_bar_wake.invert_yaxis()  # 让第一个策略在上面
        
        # 在柱子右侧标注数值
        for i, (bar, val) in enumerate(zip(bars_wake, correlations_wake)):
            if not np.isnan(val):
                ax_bar_wake.text(val + 0.01, bar.get_y() + bar.get_height()/2, 
                            f'{val:.2f}', ha='left', va='center', 
                            fontsize=bar_label_fontsize)
        
        # ========================================================================
        # 保存
        # ========================================================================
        save_path = self.output_dir / f'final_4_b_four_strategy_comparison_{self.nwp_source}.png'
        plt.savefig(save_path, dpi=300)
        plt.savefig(save_path.with_suffix('.pdf'), dpi=300)
        print(f"图表已保存: {save_path}")
        
        plt.close()
    # ========================================================================
    # 20% 测试集更细分对比：All / Free-stream / Wake-affected
    # ========================================================================
    @staticmethod
    def _rmse(y_true, y_pred):
        return float(np.sqrt(mean_squared_error(y_true, y_pred)))

    @staticmethod
    def _corr(y_true, y_pred):
        if len(y_true) < 2:
            return float('nan')
        return float(np.corrcoef(y_true, y_pred)[0, 1])

    def compute_subset_metrics(self, results):
        """计算 All / Free-stream / Wake 的 RMSE 与相关系数（基于 20% test set）"""
        if not hasattr(self, 'test_df'):
            raise RuntimeError("请先运行 load_and_split_data() 以生成 test_df")

        # 三个子集 mask
        masks = {
            'all': np.ones(len(self.test_df), dtype=bool),
            'free': (self.test_df['wind_direction_type'].values == 'free'),
            'wake': (self.test_df['wind_direction_type'].values == 'wake')
        }

        y_true_all = self.test_df['power'].values
        strategy_order = ['HH', 'ER', 'SR', 'WDA']

        metrics = {
            'rmse': {k: {} for k in masks.keys()},
            'corr': {k: {} for k in masks.keys()}
        }

        for subset_name, mask in masks.items():
            y_true = y_true_all[mask]

            for s in strategy_order:
                y_pred = results[s]['y_pred'][mask]
                # 若该子集为空，返回 NaN
                if len(y_true) == 0:
                    metrics['rmse'][subset_name][s] = float('nan')
                    metrics['corr'][subset_name][s] = float('nan')
                else:
                    metrics['rmse'][subset_name][s] = self._rmse(y_true, y_pred)
                    metrics['corr'][subset_name][s] = self._corr(y_true, y_pred)

        return metrics

    def create_subset_barplots(self, results):
        """绘制 20% test set 的柱状图对比：RMSE 一张、相关系数一张"""
        print("\n创建 20% 测试集子集柱状图对比...")

        # 可调参数
        bar_figsize = (12, 5)
        bar_label_fontsize = 16
        bar_tick_labelsize = 14
        bar_legend_fontsize = 14

        metrics = self.compute_subset_metrics(results)

        # 横坐标三个组
        groups = ['All test set', 'Free-stream test set', 'Wake-affected test set']
        subset_keys = ['all', 'free', 'wake']

        # 每组四根柱子
        strategy_order = ['HH', 'ER', 'SR', 'WDA']
        legend_labels = ['HH', 'ER', 'SR', 'Strategy']  # Strategy 对应 WDA

        x = np.arange(len(groups))
        width = 0.18
        offsets = np.linspace(-1.5 * width, 1.5 * width, len(strategy_order))

        def _plot(metric_key, ylabel, fname_suffix):
            fig, ax = plt.subplots(figsize=bar_figsize)

            for i, s in enumerate(strategy_order):
                vals = [metrics[metric_key][sk][s] for sk in subset_keys]
                ax.bar(x + offsets[i], vals, width, label=legend_labels[i])

            ax.set_xticks(x)
            ax.set_xticklabels(groups, fontsize=bar_tick_labelsize)
            ax.set_ylabel(ylabel, fontsize=bar_label_fontsize)
            ax.tick_params(axis='y', which='major', labelsize=bar_tick_labelsize)
            ax.grid(True, axis='y', alpha=0.3)
            ax.legend(frameon=False, ncol=4, fontsize=bar_legend_fontsize)

            plt.tight_layout()
            # save_path = self.output_dir / f'four_strategy_subset_{fname_suffix}_{self.nwp_source}.png'
            # plt.savefig(save_path, dpi=300, bbox_inches='tight')
            # plt.savefig(save_path.with_suffix('.pdf'), dpi=300, bbox_inches='tight')
            # print(f"图表已保存: {save_path}")
            plt.close()

        _plot('rmse', 'RMSE (MW)', 'rmse')
        _plot('corr', 'Correlation coefficient', 'corr')
    
    def run(self):
        """运行完整对比实验"""
        # 加载数据
        self.load_and_split_data()
        
        # 对比所有策略
        results = self.compare_all_strategies()
        
        # 创建可视化
        self.create_visualization(results)

        # 20% 测试集更细分对比（All / Free-stream / Wake-affected）
        # self.create_subset_barplots(results)
        
        print("\n" + "="*70)
        print("实验完成！")
        print("="*70)
        
        return results


# ============================================================================
# 主函数
# ============================================================================
if __name__ == "__main__":
    DATA_PATH = "/Users/xiaxin/work/WindForecast_Project/01_Data/processed/matched_data/changma_matched.csv"
    OUTPUT_DIR = "/Users/xiaxin/work/WindForecast_Project/03_Results/re-plot-figures/figure-4/"
    
    # 对比ECMWF和GFS两个系统
    for nwp_source in ['ec', 'gfs']:
        print("\n" + "#"*70)
        print(f"# 运行 {nwp_source.upper()} 系统的四种策略对比")
        print("#"*70)
        
        analyzer = FourStrategyComparison(DATA_PATH, OUTPUT_DIR, nwp_source=nwp_source)
        results = analyzer.run()