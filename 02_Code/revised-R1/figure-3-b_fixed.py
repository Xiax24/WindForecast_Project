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

4. [新增，回应 R2 Point 18] 误差分布 PDF 曲线图，替代散点叠加。
   R2 原话：for Fig.3B, it would be much easier for the reader to see 3
   different curves of the PDF of P_modeled/P_observed plotted together
   (1 each for HH, SR, ER)。

   这个意见是对的：现在的散点叠加里三套配置完全重合，最后画的 ER（紫）
   把 HH 与 SR 盖住，读者实际分辨不出任何差别，信息全靠 inset 柱状图承载。
   改成三条 PDF 曲线叠加确实一眼可读，谱宽即误差大小。

   但审稿人建议的比值 P_mod/P_obs 直接用会坏掉，因为分母是观测功率：
     - 115 条样本 P_obs 恰好为 0（比值 = inf）；
     - 2.3% 低于 1 MW，8.3% 低于 5 MW（装机 193.5 MW）；
     - 这些时刻模型典型输出约 20 MW，比值可达 60 以上。
   结果是 PDF 被一条极长的右尾主导，比原图更难读，而且比值的离散度在
   低 P_obs 处被分母放大，与模型好坏无关 —— 在绝对功率意义上更准的模型
   反而可能显得更差。所以这里给两种度量，由 METRICS 控制：

     'ratio' : P_mod / P_obs，但限定 P_obs >= RATIO_MIN_FRAC * 装机
               （默认 5%，即 9.7 MW），并在图上注明被剔除的样本比例。
               这是审稿人建议的量，逐字实现，只加了必要的分母保护。
     'nerr'  : (P_mod - P_obs) / 装机 * 100 [% of rated capacity]。
               风电预报文献的标准归一化误差，无分母问题、全样本可用、
               谱宽可直接读作误差大小。推荐用这个进正文。

   两种都出图，正文用哪个由作者定；回复信里用 'ratio' 那版说明
   "已按建议改画，并说明为何对分母设下限"。
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

# ---- 误差分布 PDF（R2 Point 18）--------------------------------------------
METRICS = ('ratio', 'nerr')

# Panel A 与 Panel B 的 y 轴上限。A 是 B 两个扇区的混合，若各自 autoscale
# （原来 A=0.07、B=0.06），读者无法横向比较峰高。设成 None 则退回自适应。
NERR_YMAX = 0.072      # = 三个面板中 inset 所需上限的最大值(0.0709)向上取整
RATIO_YMAX = None
RATIO_MIN_FRAC = 0.05     # 比值只在 P_obs >= 5% 装机时计算，保护分母
RATIO_XLIM = (0.0, 2.5)
NERR_XLIM = (-40.0, 40.0)

# R² + DM 显著性 inset 的位置（轴坐标系，左上角）
INSET_RECT = (0.14, 0.60, 0.29, 0.34)

PDF_LINEWIDTH = 2.0

# 线条用色：沿用散点图的三个色相，但加深 HH 与 SR。
# 原来的 #B4B4B3 / #5AEFFF 是给半透明散点用的，画成 3 px 线条在白底上
# 几乎看不见。
LINE_COLORS = {
    'Hub-height':    '#7A7A78',
    'Standard REWS': '#12A5BF',
    'Extended REWS': '#893CE7',
}


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

        self.all_results = {}

        # 嵌入柱状图显著性标注配置 - 可手动调整。
        # HH_vs_ER 那根（最上面）原来 0.30，与 SR_vs_ER 的 0.19 只差 0.11，
        # 括号和星号挤在一起 -> 抬到 0.42，同时把 inset 的 y 上限放宽。
        base = {'y_offset': 0.18, 'line_color': '#C7C7CA',
                'star_color': '#C7C7CA', 'star_size': 15}
        sig = {
            'HH_vs_SR': dict(base, y_offset=0.18),
            'SR_vs_ER': dict(base, y_offset=0.19),
            'HH_vs_ER': dict(base, y_offset=0.42),
        }
        self.inset_annotation_configs = {
            'Overall': dict(sig), 'Free-flow': dict(sig), 'Wake': dict(sig),
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

        all_results = self.all_results = {'Free-flow': {}, 'Wake': {}}

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

    def _add_inset_barplot(self, ax, sector_name, all_results,
                           rect=(0.12, 0.65, 0.38, 0.35), colors=None,
                           iqr_labels=None):
        """嵌入式柱状图 + DM 显著性标注（held-out）

        colors : dict or None
            方案 -> 颜色。PDF 曲线版传 LINE_COLORS，使柱子与曲线同色；
            散点版传 None，沿用散点自身的浅色。
        iqr_labels : dict or None
            方案 -> 已格式化的 IQR 字符串。给了就接在 HH/SR/ER 标签下面成为
            第二行，省掉图上另起一块浮动文字。
        """
        inset_ax = ax.inset_axes(list(rect))

        model_names = ['Hub-height', 'Standard REWS', 'Extended REWS']
        r2_values = [all_results[sector_name][m]['r2'] for m in model_names]

        x_pos = np.arange(len(model_names))
        width = 0.65

        for i, (model_name, r2) in enumerate(zip(model_names, r2_values)):
            c = (colors or {}).get(model_name) or self.model_configs[model_name]['color']
            inset_ax.bar(x_pos[i], r2, width, color=c,
                         alpha=0.85, edgecolor='white', linewidth=1.2)
            inset_ax.text(x_pos[i], r2 + 0.015, f'{r2:.2f}',
                          ha='center', va='bottom', fontsize=18,
                          fontweight='bold', color=c)

        self._add_all_inset_significance(inset_ax, x_pos, width,
                                         all_results, sector_name)

        inset_ax.set_xticks(x_pos)
        short = ['HH', 'SR', 'ER']
        if iqr_labels:
            inset_ax.set_xticklabels(
                [f'{s}\n{iqr_labels[m]}' for s, m in zip(short, model_names)],
                fontsize=14, fontweight='bold', linespacing=1.5)
            inset_ax.set_xlabel('IQR of error', fontsize=14, labelpad=4)
        else:
            inset_ax.set_xticklabels(short, fontsize=15, fontweight='bold')
        inset_ax.set_ylabel('$R^2$', fontsize=15, fontweight='bold')
        inset_ax.set_ylim(0, 1.4)     # 给抬高后的 HH_vs_ER 括号留位置

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

    # ------------------------------------------------------------------
    # 误差分布 PDF（R2 Point 18）
    # ------------------------------------------------------------------
    def _rated_power(self):
        """装机容量：用 QC 后全记录的最大观测功率。"""
        return float(self.test_df['power'].max().item()
                     if hasattr(self.test_df['power'].max(), 'item')
                     else max(self.train_df['power'].max(),
                              self.test_df['power'].max()))

    def _metric_values(self, y_true, y_pred, kind, rated):
        """返回 (度量值数组, 参与计算的样本比例)。"""
        if kind == 'ratio':
            keep = y_true >= RATIO_MIN_FRAC * rated
            return y_pred[keep] / y_true[keep], float(keep.mean())
        return (y_pred - y_true) / rated * 100.0, 1.0

    def report_ratio_tail(self):
        """量化未设分母下限时比值的尾部行为 —— 回复信要用。"""
        rated = self._rated_power()
        y_true = self.test_df['power'].values
        cfg = self.model_configs['Extended REWS']
        y_pred = self.models['Extended REWS'].predict(
            self.test_df[cfg['features']].values)

        print("\n" + "=" * 74)
        print("为何对 P_mod/P_obs 的分母设下限（回复 R2 Point 18）")
        print("=" * 74)
        print(f"  装机(最大观测功率) = {rated:.1f} MW，测试集 N = {len(y_true)}")
        print(f"  P_obs == 0        : {(y_true == 0).sum()} 条 -> 比值为 inf")
        nz = y_true > 0
        ratio_all = y_pred[nz] / y_true[nz]
        print(f"  P_obs > 0 时比值分位数 (ER):")
        for q in (50, 90, 99, 99.9, 100):
            print(f"    p{q:<5} = {np.percentile(ratio_all, q):10.2f}")
        thr = RATIO_MIN_FRAC * rated
        keep = y_true >= thr
        ratio_cut = y_pred[keep] / y_true[keep]
        print(f"  设下限 P_obs >= {RATIO_MIN_FRAC:.0%} 装机 = {thr:.1f} MW 后："
              f"保留 {keep.mean():.1%} 样本，p100 = {ratio_cut.max():.2f}")
        return rated

    def plot_pdf_panels(self, panels, kind, fname, fill=False, inset=False):
        """三条 PDF 曲线叠加。

        panels : [(标题, 测试集子集 df, 扇区键或 None), ...]
                 扇区键用于查 self.all_results 画 DM 显著性 inset。
        fill   : 曲线下加半透明填充（Panel A 用）。
        inset  : 左上角加 R² 柱状图 + DM 显著性括号（Panel B 用）。

        图例拆成两块，都靠右放在空白处，避免压住 x=0（或 x=1）的参考虚线：
          右上   —— 三个方案的名称 + 色条
          右侧中下 —— 对应的 R² 与 IQR 数值，按方案配色，顺序与上面一致
        """
        rated = self._rated_power()
        xlim = RATIO_XLIM if kind == 'ratio' else NERR_XLIM
        ref = 1.0 if kind == 'ratio' else 0.0
        xlabel = (r'$P_{\mathrm{modeled}}\,/\,P_{\mathrm{observed}}$'
                  if kind == 'ratio' else
                  r'$(P_{\mathrm{modeled}}-P_{\mathrm{observed}})\,/\,P_{\mathrm{rated}}$ (%)')

        fig, axes = plt.subplots(1, len(panels), figsize=(9.5 * len(panels), 8.0),
                                 squeeze=False)
        axes = axes[0]
        grid = np.linspace(*xlim, 500)
        rows = []

        for ax, (title, df_sub, sector_key) in zip(axes, panels):
            print(f"\n  [{kind}] {title} (N={len(df_sub)})")
            ymax = 0.0
            iqr_labels = {}
            pdfs = []
            for model_name in ['Hub-height', 'Standard REWS', 'Extended REWS']:
                res = self.evaluate_on(model_name, df_sub)
                vals, frac = self._metric_values(
                    res['y_true'], res['y_pred'], kind, rated)

                med = float(np.median(vals))
                iqr = float(np.percentile(vals, 75) - np.percentile(vals, 25))
                inside = float(((vals >= xlim[0]) & (vals <= xlim[1])).mean())

                pdf = stats.gaussian_kde(vals)(grid)
                pdfs.append(pdf)
                ymax = max(ymax, pdf.max())
                c = LINE_COLORS[model_name]

                # 只留 IQR（R² 已在 inset 柱子上），并且放进 inset 的 x 标签，
                # 不在图上另起浮动文字块。
                iqr_labels[model_name] = (f'{iqr:.2f}' if kind == 'ratio'
                                          else f'{iqr:.1f}%')

                ax.plot(grid, pdf, color=c, linewidth=PDF_LINEWIDTH,
                        label=model_name, zorder=3)
                if fill:
                    ax.fill_between(grid, pdf, color=c, alpha=0.12, zorder=2)

                print(f"    {model_name:15s} R²={res['r2']:.3f} "
                      f"median={med:.3f} IQR={iqr:.3f} "
                      f"用于作图的样本={frac:.1%} 落在坐标范围内={inside:.1%}")
                rows.append({'metric': kind, 'panel': title,
                             'model': model_name, 'N': res['n'],
                             'r2': res['r2'], 'rmse_MW': res['rmse'],
                             'median': med, 'iqr': iqr,
                             'frac_used': frac, 'frac_in_xlim': inside})

            ax.axvline(ref, color='k', linestyle='--', linewidth=2, alpha=0.6,
                       zorder=1)
            ax.set_xlim(*xlim)

            # y 上限：留出 inset 的位置。ratio 版的峰值偏左（比值 ~0.9），
            # 正落在左上角 inset 底下，固定倍数会被压住；这里按 inset 横向
            # 覆盖范围内的曲线最大值反推所需上限。
            ylim_hi = ymax * 1.28
            if inset and sector_key is not None:
                ix0, iy0, iw, _ = INSET_RECT
                span = xlim[1] - xlim[0]
                lo = xlim[0] + (ix0 - 0.03) * span
                hi = xlim[0] + (ix0 + iw + 0.03) * span
                sel = (grid >= lo) & (grid <= hi)
                if sel.any():
                    local = max(p[sel].max() for p in pdfs)
                    ylim_hi = max(ylim_hi, local / (iy0 - 0.03))

            fixed = RATIO_YMAX if kind == 'ratio' else NERR_YMAX
            if fixed is not None:
                if fixed < ylim_hi:
                    print(f"    [警告] 固定 y 上限 {fixed} 小于所需 {ylim_hi:.4f}，"
                          f"inset 可能压住曲线")
                ylim_hi = fixed
            ax.set_ylim(0, ylim_hi)
            ax.set_xlabel(xlabel, fontsize=26)
            if ax is axes[0]:
                ax.set_ylabel('Probability density', fontsize=26)
            ax.set_title(title, fontsize=28, fontweight='bold', pad=14)

            # 图例只放方案名称，右上空白处；数字全部收进左上角的 inset。
            ax.legend(loc='upper right', fontsize=19, frameon=False,
                      handlelength=1.6, borderaxespad=0.8)

            ax.grid(False)
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            ax.spines['left'].set_linewidth(2)
            ax.spines['bottom'].set_linewidth(2)
            ax.tick_params(axis='both', which='major', labelsize=22,
                           width=2, length=6)

            if inset and sector_key is not None:
                # 左上角：R² 柱状图 + DM 显著性括号（与原散点图版一致）。
                # 比散点版窄，右移到 0.14 —— 太靠左时 inset 自己的 y 轴刻度
                # 会伸到主图左侧，与主图的 y 刻度标签撞在一起。
                self._add_inset_barplot(ax, sector_key, self.all_results,
                                        rect=INSET_RECT, colors=LINE_COLORS,
                                        iqr_labels=iqr_labels)

            if kind == 'ratio':
                note_y = 0.55 if inset else 0.98
                ax.text(0.02, note_y,
                        f'$P_{{\\mathrm{{obs}}}} \\geq$ {RATIO_MIN_FRAC:.0%} '
                        f'of rated', transform=ax.transAxes,
                        ha='left', va='top', fontsize=17, color='#555555')

        plt.tight_layout()
        out = self.output_dir / fname
        plt.savefig(out, dpi=600, facecolor='white')
        plt.savefig(out.with_suffix('.pdf'), dpi=600, facecolor='white')
        plt.close()
        print(f"  ✓ {out.name}")
        return rows

    def create_pdf_figures(self):
        """出 4 张图：3A(总体) / 3B(分扇区) × ratio / nerr。"""
        self.report_ratio_tail()
        ds = self.data_path.stem
        rows = []

        # Panel A 也要 R²+DM 的 inset -> 先在整个测试集上补一份评估结果。
        # （create_scatter_overlay 只填了 Free-flow / Wake 两个扇区。）
        self.all_results['Overall'] = {
            m: self.evaluate_on(m, self.test_df)
            for m in ['Hub-height', 'Standard REWS', 'Extended REWS']
        }

        for kind in METRICS:
            print("\n" + "=" * 74)
            print(f"误差分布 PDF — {kind}")
            print("=" * 74)
            # Panel A：单面板，曲线下填色，样式与 Panel B 一致（含 inset）
            rows += self.plot_pdf_panels(
                [('Overall', self.test_df, 'Overall')], kind,
                f'final_3_a_pdf_{kind}_{ds}.png', fill=True, inset=True)
            # Panel B：双面板，保留原左上角的 R²+DM 显著性 inset
            rows += self.plot_pdf_panels(
                [('Free-stream', self.test_free, 'Free-flow'),
                 ('Wake Regime', self.test_wake, 'Wake')],
                kind, f'final_3_b_pdf_{kind}_{ds}.png', fill=False, inset=True)

            # A+B 合并成一行三面板：Panel A 单独占一行会浪费左右两侧的空白，
            # 整图变三行高。排成一行后编号仍可保持 A / B / C 不变
            # （第一个面板标 A，后两个合起来标 B），正文引用无需改动。
            self.plot_pdf_panels(
                [('Overall', self.test_df, 'Overall'),
                 ('Free-stream', self.test_free, 'Free-flow'),
                 ('Wake Regime', self.test_wake, 'Wake')],
                kind, f'final_3_ab_row_pdf_{kind}_{ds}.png',
                fill=False, inset=True)

        df = pd.DataFrame(rows)
        path = self.output_dir / f'data_fig3_pdf_metrics_{ds}.csv'
        df.to_csv(path, index=False, float_format='%.5g')
        print(f"\n  ✓ {path.name}")
        return df

    def run(self):
        self.load_split_and_classify()
        self.train_models()
        results = self.create_scatter_overlay()
        self.create_pdf_figures()
        return results


if __name__ == "__main__":
    DATA_PATH = "/Users/xiaxin/work/WindForecast_Project/01_Data/processed/matched_data/changma_matched.csv"
    OUTPUT_DIR = "/Users/xiaxin/work/WindForecast_Project/03_Results/re-plot-figures/figure-3/"

    visualizer = ScatterOverlayVisualizer(DATA_PATH, OUTPUT_DIR)
    visualizer.run()
