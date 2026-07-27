#!/usr/bin/env python3
"""
Figure 3 - Panel B (manuscript Fig 4B): 分扇区重构性能对比
v3: held-out evaluation + proper Diebold-Mariano test.

修正内容（相对 v2）：
1. [评估口径] 旧版在全部样本上训练、并在同一批样本的扇区子集上评估
   （in-sample，R² 偏高，DM 无意义）。现改为：
     - 与 figure-3-a 完全相同的 80/20 划分（相同 QC、相同 dropna 列、
       random_state=42），每个方案只训练一次（80% 训练集）；
     - 扇区掩码施加在 20% 测试集上，R²/RMSE/DM 全部只在
       held-out 测试集的扇区子集上计算。
   由此 Panel A 与 Panel B 使用同一批模型，与正文描述一致
   ("applying these same globally trained models to withheld test subsets")。
2. [DM 检验] 旧版 h=1 使自协方差循环为空，退化为普通配对 t 检验，
   在正自相关误差下夸大显著性。现实现：
     - 损失差分按时间排序后计算 Bartlett 核 HAC 方差
       （截断阶 Newey-West 自动规则 L = floor(4*(n/100)^(2/9))）；
     - Harvey-Leybourne-Newbold 小样本修正；
     - t(n-1) 分布双侧 p 值。
3. [温度] INCLUDE_TEMP 开关。正文 2.4 写明三方案均含 10 m 温度，
   默认 True；figure-3-a 需同步加入同一特征（见文末说明）。
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
from qc_common import (qc_operating_range, sector_mask,
                       SECTOR_FREE, SECTOR_WAKE, WD_SECTOR_COL)

warnings.filterwarnings('ignore')

# 与正文 2.4 一致：HH/SR/ER 均含 10 m 温度。
# 若暂不改 figure-3-a，可临时设为 False 保持两图同口径。
INCLUDE_TEMP = False
TEMP_COL = 'obs_temperature_10m'


# ============================================================================
# Diebold-Mariano 检验（HAC + HLN，双侧 t）
# ============================================================================
def diebold_mariano_test(y_true, pred1, pred2, times=None, max_lag=None):
    """
    DM test on squared-error loss.

    Parameters
    ----------
    y_true, pred1, pred2 : array-like
        观测值与两组预测值（配对）。
    times : array-like or None
        每个样本的时间戳。随机划分打散了时间顺序，HAC 方差必须在
        按时间重排后的损失差分序列上估计；不提供则按原顺序（仅当
        输入本身按时间排列时才合理）。
    max_lag : int or None
        Bartlett 核截断阶。None 时用 Newey-West 自动规则
        L = floor(4*(n/100)^(2/9))。

    Returns
    -------
    dm_hln : float   HLN 修正后的 DM 统计量
    p      : float   双侧 p 值（t 分布，df = n-1）
    L      : int     实际使用的截断阶
    """
    y_true = np.asarray(y_true, float)
    e1 = (y_true - np.asarray(pred1, float)) ** 2
    e2 = (y_true - np.asarray(pred2, float)) ** 2
    d = e1 - e2

    if times is not None:
        order = np.argsort(np.asarray(times))
        d = d[order]

    n = len(d)
    dbar = d.mean()
    dc = d - dbar

    if max_lag is None:
        max_lag = int(np.floor(4.0 * (n / 100.0) ** (2.0 / 9.0)))
    max_lag = max(0, min(max_lag, n - 1))

    # Bartlett-kernel HAC variance of the loss differential
    gamma0 = np.mean(dc * dc)
    var = gamma0
    for k in range(1, max_lag + 1):
        w = 1.0 - k / (max_lag + 1.0)
        var += 2.0 * w * np.mean(dc[:-k] * dc[k:])
    if var <= 0:            # HAC 可能非正定，退回 lag-0（保守）
        var = gamma0

    dm = dbar / np.sqrt(var / n)

    # Harvey-Leybourne-Newbold 小样本修正（h = L+1）
    h = max_lag + 1
    hln = np.sqrt((n + 1 - 2 * h + h * (h - 1) / n) / n)
    dm_hln = hln * dm

    p = 2.0 * (1.0 - stats.t.cdf(abs(dm_hln), df=n - 1))
    return dm_hln, p, max_lag


# ============================================================================
# 样式设置（与 v2 一致）
# ============================================================================
plt.rcParams.update({
    'font.family': ['Arial', 'DejaVu Sans'],
    'font.size': 28,
    'axes.linewidth': 1.5,
    'figure.dpi': 300,
    'savefig.bbox': 'tight',
    'savefig.facecolor': 'white'
})


class ScatterOverlayVisualizer:
    """三模型测试集扇区散点叠加可视化器"""

    def __init__(self, data_path, output_dir):
        self.data_path = Path(data_path)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # 70 m is the only clean vane and is also what the operational WDA
        # strategy uses -> single sector-defining variable.
        self.wind_dir_columns = [WD_SECTOR_COL]

        # 模型配置（特征与 figure-3-a 严格一致）
        temp = [TEMP_COL] if INCLUDE_TEMP else []
        self.model_configs = {
            'Hub-height': {
                'features': ['obs_wind_speed_70m'] + temp,
                'label': 'Hub-height',
                'color': '#B4B4B3', 'alpha': 0.7, 'marker': 'o', 's': 30
            },
            'Standard REWS': {
                'features': ['obs_wind_speed_30m', 'obs_wind_speed_50m',
                             'obs_wind_speed_70m'] + temp,
                'label': 'Standard REWS',
                'color': '#5AEFFF', 'alpha': 0.5, 'marker': 'o', 's': 30
            },
            'Extended REWS': {
                'features': ['obs_wind_speed_10m', 'obs_wind_speed_30m',
                             'obs_wind_speed_50m', 'obs_wind_speed_70m'] + temp,
                'label': 'Extended REWS',
                'color': '#893CE7', 'alpha': 0.3, 'marker': 'o', 's': 30
            }
        }

        # LightGBM参数（与 figure-3-a 一致）
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
        base = {'y_offset': 0.18, 'line_color': '#C7C7CA',
                'star_color': '#C7C7CA', 'star_size': 15}
        self.inset_annotation_configs = {
            'Free-flow': {
                'HH_vs_SR': dict(base, y_offset=0.18),
                'SR_vs_ER': dict(base, y_offset=0.19),
                'HH_vs_ER': dict(base, y_offset=0.30),
            },
            'Wake': {
                'HH_vs_SR': dict(base, y_offset=0.18),
                'SR_vs_ER': dict(base, y_offset=0.19),
                'HH_vs_ER': dict(base, y_offset=0.30),
            }
        }

    # ------------------------------------------------------------------
    # 数据：与 figure-3-a 相同的 QC / dropna / 划分
    # ------------------------------------------------------------------
    def load_split_and_classify(self):
        print("=== 加载数据，QC，80/20 划分（与 figure-3-a 同口径）===")
        data = pd.read_csv(self.data_path)
        print(f"原始数据形状: {data.shape}")

        # 所有方案的特征并集（决定 dropna 列，必须与 figure-3-a 一致）
        all_features = set()
        for config in self.model_configs.values():
            all_features.update(config['features'])
        required_cols = sorted(all_features) + ['power']

        # QC：power>=0 / dropna / 剔除僵值 / 70m 3-25 m/s
        data_clean = qc_operating_range(data)
        data_clean = data_clean.dropna(subset=required_cols)
        print(f"QC 后样本: {len(data_clean)}")

        # 80/20 划分：random_state=42，与 figure-3-a 完全一致。
        # 注意：不在划分前按风向 dropna——风向缺失的样本仍参与训练与
        # 全集评估，只是在扇区子集中不出现，保证与 figure-3-a 同划分。
        train_df, test_df = train_test_split(
            data_clean, test_size=0.2, random_state=42
        )
        print(f"训练集: {len(train_df)} (80%) | 测试集: {len(test_df)} (20%)")

        # 扇区掩码：仅施加在测试集上（held-out 评估）
        mask_free = sector_mask(test_df, SECTOR_FREE)
        mask_wake = sector_mask(test_df, SECTOR_WAKE)
        self.test_free = test_df[mask_free].copy()
        self.test_wake = test_df[mask_wake].copy()

        n = len(test_df)
        print(f"测试集扇区: free={len(self.test_free)} ({100*len(self.test_free)/n:.1f}%)"
              f" | wake={len(self.test_wake)} ({100*len(self.test_wake)/n:.1f}%)")

        self.train_df = train_df
        self.test_df = test_df
        return train_df, test_df

    # ------------------------------------------------------------------
    # 训练：每方案只训练一次（80% 训练集）
    # ------------------------------------------------------------------
    def train_models(self):
        print("\n=== 训练三个全局模型（仅训练集）===")
        self.models = {}
        for model_name, config in self.model_configs.items():
            features = config['features']
            X_train = self.train_df[features].values
            y_train = self.train_df['power'].values

            model = lgb.LGBMRegressor(**self.lgb_params)
            model.fit(X_train, y_train)
            self.models[model_name] = model

            # 全测试集指标：应与 figure-3-a 输出一致（交叉核对用）
            y_pred = model.predict(self.test_df[features].values)
            r2 = r2_score(self.test_df['power'].values, y_pred)
            rmse = np.sqrt(mean_squared_error(self.test_df['power'].values, y_pred))
            print(f"  {model_name:15s} | 全测试集 R²={r2:.3f}, RMSE={rmse:.2f} MW"
                  f"  <- 应与 figure-3-a 一致")
        return self.models

    def evaluate_on(self, model_name, df_subset):
        """在给定（测试集）子集上评估已训练模型"""
        config = self.model_configs[model_name]
        features = config['features']
        y_true = df_subset['power'].values
        y_pred = self.models[model_name].predict(df_subset[features].values)
        return {
            'y_true': y_true,
            'y_pred': y_pred,
            'times': pd.to_datetime(df_subset['datetime']).values
                     if 'datetime' in df_subset.columns else None,
            'r2': r2_score(y_true, y_pred),
            'rmse': np.sqrt(mean_squared_error(y_true, y_pred)),
            'n': len(y_true),
        }

    # ------------------------------------------------------------------
    # 绘图（样式与 v2 一致，数据换为测试集扇区子集）
    # ------------------------------------------------------------------
    def create_scatter_overlay(self):
        print("\n=== 创建散点叠加图（held-out 测试集扇区）===")

        fig, axes = plt.subplots(1, 2, figsize=(20, 9),
                                 gridspec_kw={'wspace': 0.01})

        sectors = [
            ('Free-flow', self.test_free, axes[0], 'Free-stream'),
            ('Wake', self.test_wake, axes[1], 'Wake Regime')
        ]

        all_results = {'Free-flow': {}, 'Wake': {}}

        for sector_name, data_sector, ax, title in sectors:
            print(f"\n处理 {sector_name} 区（测试集 N={len(data_sector)}）...")

            axis_min = 0

            # 1:1 理想线
            ax.plot([axis_min, 200], [axis_min, 200],
                    'k--', linewidth=2.5, alpha=0.6,
                    label='1:1 Line', zorder=1)

            for model_name in ['Hub-height', 'Standard REWS', 'Extended REWS']:
                config = self.model_configs[model_name]
                result = self.evaluate_on(model_name, data_sector)
                all_results[sector_name][model_name] = result

                print(f"  {model_name}: R²={result['r2']:.3f}, "
                      f"RMSE={result['rmse']:.1f} MW, N={result['n']}")

                ax.scatter(result['y_true'], result['y_pred'],
                           c=config['color'], alpha=config['alpha'],
                           s=config['s'], marker=config['marker'],
                           edgecolors='none', label=config['label'], zorder=2)

            ax.set_xlim(axis_min, 200)
            ax.set_ylim(axis_min, 200)
            ax.set_xticks(np.arange(0, 201, 25))
            ax.set_yticks(np.arange(0, 201, 25))
            ax.set_aspect('equal', adjustable='box')

            ax.set_xlabel('Observed Power (MW)', fontsize=28, fontweight='normal')
            if sector_name == 'Free-flow':
                ax.set_ylabel('Modeled Power (MW)', fontsize=28, fontweight='normal')

            ax.set_title(title, fontsize=28, fontweight='bold', pad=15, loc='center')
            ax.legend(loc='lower right', fontsize=20, frameon=True,
                      fancybox=False, edgecolor='gray', framealpha=0.9)

            ax.grid(False)
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            ax.spines['left'].set_linewidth(2)
            ax.spines['bottom'].set_linewidth(2)
            ax.tick_params(axis='both', which='major', labelsize=24,
                           width=2, length=6)

            self._add_inset_barplot(ax, sector_name, all_results)

        dataset_name = self.data_path.stem
        save_path = self.output_dir / f'final_3_b_{dataset_name}_heldout.png'
        plt.savefig(save_path, dpi=600, facecolor='white')
        plt.savefig(save_path.with_suffix('.pdf'), dpi=600, facecolor='white')
        plt.close()
        print(f"\n已保存: {save_path}")
        return all_results

    def _add_inset_barplot(self, ax, sector_name, all_results):
        """嵌入式柱状图 + DM 显著性标注（held-out）"""
        inset_ax = ax.inset_axes([0.12, 0.65, 0.38, 0.35])

        model_names = ['Hub-height', 'Standard REWS', 'Extended REWS']
        r2_values = [all_results[sector_name][m]['r2'] for m in model_names]

        x_pos = np.arange(len(model_names))
        width = 0.65

        for i, (model_name, r2) in enumerate(zip(model_names, r2_values)):
            config = self.model_configs[model_name]
            inset_ax.bar(x_pos[i], r2, width, color=config['color'],
                         alpha=0.85, edgecolor='white', linewidth=1.2)
            inset_ax.text(x_pos[i], r2 + 0.015, f'{r2:.2f}',
                          ha='center', va='bottom', fontsize=18,
                          fontweight='bold', color=config['color'])

        self._add_all_inset_significance(inset_ax, x_pos, width,
                                         all_results, sector_name)

        inset_ax.set_xticks(x_pos)
        inset_ax.set_xticklabels(['HH', 'SR', 'ER'], fontsize=15, fontweight='bold')
        inset_ax.set_ylabel('$R^2$', fontsize=15, fontweight='bold')
        inset_ax.set_ylim(0, 1.15)

        inset_ax.grid(False)
        inset_ax.spines['top'].set_visible(False)
        inset_ax.spines['right'].set_visible(False)
        inset_ax.tick_params(axis='both', which='major', labelsize=15)
        inset_ax.set_facecolor('white')
        inset_ax.patch.set_alpha(0.95)

    def _add_all_inset_significance(self, ax, x_pos, width, all_results, sector_name):
        model_names = ['Hub-height', 'Standard REWS', 'Extended REWS']
        r2_values = [all_results[sector_name][m]['r2'] for m in model_names]

        comparisons = [
            ('HH_vs_SR', 0, 1),
            ('SR_vs_ER', 1, 2),
            ('HH_vs_ER', 0, 2)
        ]

        print(f"  --- DM 检验（{sector_name}，HAC+HLN，双侧 t）---")
        for test_name, left_idx, right_idx in comparisons:
            if test_name not in self.inset_annotation_configs[sector_name]:
                continue
            config = self.inset_annotation_configs[sector_name][test_name]

            left_model = model_names[left_idx]
            right_model = model_names[right_idx]
            res_l = all_results[sector_name][left_model]
            res_r = all_results[sector_name][right_model]

            dm_stat, p_val, L = diebold_mariano_test(
                res_l['y_true'], res_l['y_pred'], res_r['y_pred'],
                times=res_l['times']
            )

            if p_val < 0.01:
                stars = '***'
            elif p_val < 0.05:
                stars = '**'
            elif p_val < 0.1:
                stars = '*'
            else:
                stars = 'n.s.'

            print(f"    {right_model} vs {left_model}: "
                  f"DM={dm_stat:.3f}, p={p_val:.4f} {stars} (L={L}, N={res_l['n']})")

            x_left, x_right = x_pos[left_idx], x_pos[right_idx]
            base_height = max(r2_values[left_idx], r2_values[right_idx])
            y_line = base_height + config['y_offset']

            ax.plot([x_left, x_left, x_right, x_right],
                    [y_line - 0.008, y_line, y_line, y_line - 0.008],
                    color=config['line_color'], lw=1.0, zorder=4)
            ax.text((x_left + x_right) / 2, y_line + 0.005, stars,
                    ha='center', va='bottom', fontsize=config['star_size'],
                    fontweight='bold', color=config['star_color'])

    def run(self):
        self.load_split_and_classify()
        self.train_models()
        return self.create_scatter_overlay()


if __name__ == "__main__":
    DATA_PATH = "/Users/xiaxin/work/WindForecast_Project/01_Data/processed/matched_data/changma_matched.csv"
    OUTPUT_DIR = "/Users/xiaxin/work/WindForecast_Project/03_Results/re-plot-figures/figure-3/"

    visualizer = ScatterOverlayVisualizer(DATA_PATH, OUTPUT_DIR)
    visualizer.run()
