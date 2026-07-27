#!/usr/bin/env python3
"""
Figure 3 - Panel C: SHAP 机制分析 (所有特征 + 扇区独立 + 仅测试集版本)
流程：
1. 使用所有obs_特征（风速、温度、气压等）+ 风向sin/cos转换
2. 按扇区划分数据（Free-flow & Wake）
3. 各扇区独立 80/20 划分
4. 各扇区训练独立模型
5. 仅对各扇区的 20% 测试集计算 SHAP
6. 分别绘制两个扇区的 SHAP 蜂群图（Top 10 特征）
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import lightgbm as lgb
import shap
import warnings
import re
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error
from qc_common import (qc_operating_range, sector_mask,
                       SECTOR_FREE, SECTOR_WAKE, WD_BAD_COLS, report_sectors)

warnings.filterwarnings('ignore')

# 绘图参数设置
plt.rcParams.update({
    'font.family': ['Arial', 'DejaVu Sans'],
    'font.size': 50,
    'axes.linewidth': 2.0,
    'figure.dpi': 500,
    'savefig.dpi': 500,
    'savefig.bbox': 'tight',
    'savefig.facecolor': 'white'
})

class Figure3cAllFeaturesSectorTestAnalyzer:
    def __init__(self, data_path, output_dir):
        self.data_path = Path(data_path)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # 风向列定义
        self.wind_dir_columns = ['obs_wind_direction_70m']
        # self.wind_dir_columns = ['obs_wind_direction_10m', 'obs_wind_direction_30m',
        #                          'obs_wind_direction_50m', 'obs_wind_direction_70m']
        # self.wind_dir_columns = ['obs_wind_direction_10m', 'obs_wind_direction_30m', 
        #                          'obs_wind_direction_50m', 'obs_wind_direction_70m']


    def prepare_features(self, df):
        """
        准备所有obs特征：
        1. 选择所有obs_特征（排除密度、湿度）
        2. 风向转换为sin/cos
        """
        # 选择所有obs_开头的特征列
        obs_columns = [col for col in df.columns if col.startswith('obs_')]
        # 排除密度和湿度
        obs_columns = [col for col in obs_columns if 'density' not in col and 'humidity' not in col]
        # 排除datetime和power
        feature_columns = [col for col in obs_columns if col not in ['datetime', 'power']]
        # 10 m vane: quantised to 16 bins + ~30% pinned at 0/359 deg.
        # 30 m vane: +12.2 deg offset, 2021-11 failure.  50 m vane: -6.5 deg offset.
        # Only the 70 m vane is retained, matching the operational WDA sectoring.
        feature_columns = [c for c in feature_columns if c not in WD_BAD_COLS]


        # 处理风向变量为sin/cos分量
        df_processed = df.copy()
        wind_dir_cols = [col for col in feature_columns if 'wind_direction' in col]
        
        if wind_dir_cols:
            for col in wind_dir_cols:
                # 转换为弧度
                wind_dir_rad = np.deg2rad(df_processed[col])
                
                # 创建sin/cos分量
                sin_col = col.replace('wind_direction', 'wind_dir_sin')
                cos_col = col.replace('wind_direction', 'wind_dir_cos')
                
                df_processed[sin_col] = np.sin(wind_dir_rad)
                df_processed[cos_col] = np.cos(wind_dir_rad)
            
            # 更新特征列表：移除原始风向，添加sin/cos
            feature_columns = [col for col in feature_columns if 'wind_direction' not in col]
            sin_cos_cols = [col for col in df_processed.columns 
                           if 'wind_dir_sin' in col or 'wind_dir_cos' in col]
            feature_columns.extend(sin_cos_cols)
        
        return df_processed, feature_columns

    # def create_display_name(self, feature_name):
    #     """将特征名转换为显示名称"""
    #     name = feature_name.replace('obs_', '').replace('_', ' ').title()
    #     # 简化显示
    #     name = name.replace('Wind Speed', 'WS')
    #     name = name.replace('Temperature', 'Temp')
    #     name = name.replace('Pressure', 'Press')
    #     name = name.replace('Wind Dir Sin', 'WD Sin')
    #     name = name.replace('Wind Dir Cos', 'WD Cos')
    #     return name

    def create_display_name(self, feature_name):
        name = feature_name.replace('obs_', '').replace('_', ' ').title()
        name = name.replace('Wind Speed', 'WS')
        name = name.replace('Temperature', 'Temp')
        name = name.replace('Pressure', 'Press')
        name = name.replace('Wind Dir Sin', 'WD Sin')
        name = name.replace('Wind Dir Cos', 'WD Cos')
        # 修复 .title() 将高度单位 m 大写的问题
        name = re.sub(r'(\d+)M\b', r'\1m', name)
        return name


    def train_sector_specific_test_only(self):
        """扇区独立训练 + 仅测试集SHAP（70 m 单高度定扇区）"""
        # 载入数据
        df = pd.read_csv(self.data_path)

        # QC：power>=0 / dropna / 剔除僵值 / 70m 3-25 m/s
        df = qc_operating_range(df).dropna(subset=self.wind_dir_columns)

        print(f"✓ 总数据量: {len(df)} 条\n")

        # 按扇区划分（仅用 70 m 风向）
        mask_free = sector_mask(df, SECTOR_FREE)
        mask_wake = sector_mask(df, SECTOR_WAKE)
        report_sectors(df, 'SHAP')

        df_free = df[mask_free].copy()
        df_wake = df[mask_wake].copy()

        print(f"✓ Free-flow 扇区: {len(df_free)} 条")
        print(f"✓ Wake 扇区: {len(df_wake)} 条\n")

        print("=" * 60)
        print("处理 Free-flow 扇区 (所有特征 + 仅测试集SHAP)...")
        print("=" * 60)
        results_free = self._process_sector_test_only(df_free, "Free-flow")

        print("\n" + "=" * 60)
        print("处理 Wake 扇区 (所有特征 + 仅测试集SHAP)...")
        print("=" * 60)
        results_wake = self._process_sector_test_only(df_wake, "Wake")

        return {'free': results_free, 'wake': results_wake}

    def _process_sector_test_only(self, df_sector, sector_name):
        """
        处理单个扇区 - 所有特征 + 仅测试集SHAP
        """
        # 准备特征
        df_processed, feature_columns = self.prepare_features(df_sector)
        
        print(f"  使用 {len(feature_columns)} 个特征")
        
        # 去除缺失值
        df_clean = df_processed[feature_columns + ['power']].dropna()
        print(f"  去除缺失值后: {len(df_clean)} 条")
        
        X = df_clean[feature_columns]
        y = df_clean['power']
        
        # 80/20 划分
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        
        print(f"  训练集: {len(X_train)} 条 (80%)")
        print(f"  测试集: {len(X_test)} 条 (20%)")
        
        # 训练独立模型
        print(f"  训练 {sector_name} 专用模型（所有特征）...")
        model = lgb.LGBMRegressor(
            n_estimators=100, 
            learning_rate=0.1,
            num_leaves=31,
            feature_fraction=0.8,
            bagging_fraction=0.8,
            reg_alpha=0.1,
            reg_lambda=0.1,
            random_state=42, 
            verbose=-1
        )
        model.fit(X_train, y_train)
        
        # 评估
        y_train_pred = model.predict(X_train)
        y_test_pred = model.predict(X_test)
        
        train_r2 = r2_score(y_train, y_train_pred)
        test_r2 = r2_score(y_test, y_test_pred)
        train_rmse = np.sqrt(mean_squared_error(y_train, y_train_pred))
        test_rmse = np.sqrt(mean_squared_error(y_test, y_test_pred))
        
        print(f"  ✓ 训练集 - R²: {train_r2:.3f}, RMSE: {train_rmse:.1f} MW")
        print(f"  ✓ 测试集 - R²: {test_r2:.3f}, RMSE: {test_rmse:.1f} MW")
        
        # 仅对测试集计算 SHAP
        print(f"  计算 {sector_name} 测试集 {len(X_test)} 条数据的 SHAP 值...")
        explainer = shap.TreeExplainer(model)
        shap_values_test = explainer.shap_values(X_test)
        print(f"  ✓ SHAP 计算完成")
        
        imp = np.abs(shap_values_test).mean(0)
        order = np.argsort(imp)[::-1]
        print(f"\n  --- {sector_name} mean|SHAP| ranking ---")
        for k in order[:12]:
            print(f"    {feature_columns[k]:35s} {imp[k]:8.3f}")


        return {
            'shap_values': shap_values_test,
            'X': X_test.values,
            'n_test': len(X_test),
            'n_train': len(X_train),
            'train_r2': train_r2,
            'test_r2': test_r2,
            'train_rmse': train_rmse,
            'test_rmse': test_rmse,
            'sector_name': sector_name,
            'feature_names': feature_columns
        }

    def _plot_beeswarm_improved(self, ax, results, title, is_left_plot=False):
        """绘制 SHAP 蜂群图 - Top 10特征"""
        shap_values = results['shap_values']
        X_data = results['X']
        feature_names = results['feature_names']
        
        # 按平均 |SHAP| 排序。剔掉三个坏风向标（WD_BAD_COLS）之后特征只剩 7 个
        # （4 层风速 + 10m 温度 + 70m 风向 sin/cos），[-10:] 实际返回全部特征，
        # 所以轴标签不能再写 "Top 10"。这里显式记下实际画了几个。
        importance = np.abs(shap_values).mean(0)
        top_indices = np.argsort(importance)[-10:]
        self._n_shown = len(top_indices)
        self._n_total = len(feature_names)

        # 创建显示名称
        display_names = [self.create_display_name(feature_names[i]) for i in top_indices]
        rng = np.random.default_rng(42)
        for plot_idx, feat_idx in enumerate(top_indices):
            shap_vals = shap_values[:, feat_idx]
            feature_vals = X_data[:, feat_idx]
            
            # 标准化特征值
            if feature_vals.max() != feature_vals.min():
                norm_vals = (feature_vals - feature_vals.min()) / (feature_vals.max() - feature_vals.min())
            else:
                norm_vals = np.ones_like(feature_vals) * 0.5
            
            # SHAP经典配色
            # y_pos = np.full_like(shap_vals, plot_idx) + np.random.normal(0, 0.08, len(shap_vals))
            y_pos = np.full_like(shap_vals, plot_idx) + rng.normal(0, 0.08, len(shap_vals))
            from matplotlib.colors import LinearSegmentedColormap
            colors_list = ["#1E88E5", "#7E57C2", "#D81B60", "#E91E63", "#F06292"]
            cmap = LinearSegmentedColormap.from_list("shap_classic", colors_list)
            colors = cmap(norm_vals)
            
            ax.scatter(shap_vals, y_pos, c=colors, alpha=0.85, s=35, 
                       edgecolors='none', linewidth=0.15)
        
        # 装饰
        ax.set_yticks(range(len(display_names)))
        # 水平标签。原来 rotation=45 时文字向左下方伸出，重心落在两行之间，
        # 读者要费劲把标签和对应的那一行对上；特征名很短，横排放得下。
        ax.set_yticklabels(display_names, fontsize=31, rotation=0, ha='right')
        ax.tick_params(axis='x', labelsize=31)
        ax.tick_params(axis='y', labelsize=31)
        
        ax.axvline(x=0, color='black', linestyle='-', alpha=0.7, linewidth=2.0)
        # 字号按"与 Panel B 缩放后等效"反算。基准不能用 figsize 之比：
        # 两图都是 bbox_inches='tight'，裁边后实际导出宽度 B=11016 px、
        # C=15296 px，故 C 的字号 = B 的字号 x 15296/11016 = x1.389。
        # B: 标题 28 / 轴标签 26 / 刻度 22 / 图例 19。
        ax.set_title(title, fontsize=39, pad=18, weight='bold')
        ax.set_xlabel('SHAP Value', fontsize=36, weight='normal')
        
        if is_left_plot:
            # 全部特征都画出来了就别写 "Top N"，否则是假话。
            label = ('Features' if self._n_shown >= self._n_total
                     else f'Top {self._n_shown} Features')
            ax.set_ylabel(label, fontsize=36, weight='normal')
        
        # 标注框
        textstr = (
                   f'Test $R^2$ = {results["test_r2"]:.2f}\n'
                   f'RMSE = {results["test_rmse"]:.2f} MW\n'
                   f'Test $N$ = {results["n_test"]}')
        ax.text(0.55, 0.08, textstr, transform=ax.transAxes, fontsize=27,
                bbox=dict(boxstyle='round,pad=0.6', facecolor='none', alpha=0.92, 
                         edgecolor='none', linewidth=1.5))
        
        ax.grid(True, alpha=0.25, axis='x', linestyle='--', linewidth=1.2)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['left'].set_linewidth(2.0)
        ax.spines['bottom'].set_linewidth(2.0)

    def create_figure(self, results):
        fig = plt.figure(figsize=(24, 10))
        
        ax1 = plt.subplot(1, 2, 1)
        ax2 = plt.subplot(1, 2, 2)
        
        self._plot_beeswarm_improved(
            ax1, results['free'], 
            "Free-stream", 
            is_left_plot=True
        )
        self._plot_beeswarm_improved(
            ax2, results['wake'],
            "Wake Regime",
            is_left_plot=False
        )

        # 两个扇区共用 x 轴范围。原来各自 autoscale，而正文的论点恰恰是
        # 10 m 在 wake 扇区的贡献"更大"——横轴不同刻度时这个对比在视觉上
        # 是无效的，读者无法比较 SHAP 幅度。
        lo = min(ax1.get_xlim()[0], ax2.get_xlim()[0])
        hi = max(ax1.get_xlim()[1], ax2.get_xlim()[1])
        # 固定 25 的刻度间隔。面板变窄后自动定位器会退化到只剩 0 和 50，
        # 横轴读不出来。
        from matplotlib.ticker import MultipleLocator
        for a in (ax1, ax2):
            a.set_xlim(lo, hi)
            a.xaxis.set_major_locator(MultipleLocator(25))

        # 颜色条
        import matplotlib.cm as cm
        import matplotlib.colors as mcolors
        from matplotlib.colors import LinearSegmentedColormap
        
        colors_list = ["#1E88E5", "#7E57C2", "#D81B60", "#E91E63", "#F06292"]
        cmap_shap = LinearSegmentedColormap.from_list("shap_classic", colors_list)
        
        cbar_ax = fig.add_axes([0.92, 0.2, 0.018, 0.6])
        norm = mcolors.Normalize(vmin=0, vmax=1)
        sm = cm.ScalarMappable(norm=norm, cmap=cmap_shap)
        cbar = plt.colorbar(sm, cax=cbar_ax)
        cbar.set_ticks([0, 1])
        cbar.set_ticklabels(['Low', 'High'], fontsize=31)
        # cbar.set_label('Feature Value', fontsize=30, labelpad=15, weight='bold')
        cbar.ax.tick_params(labelsize=31)
        
        plt.subplots_adjust(left=0.08, right=0.90, wspace=0.42)
        
        # 保存
        save_path_png = self.output_dir / "final_3_c_all_features_sector_test.png"
        save_path_pdf = self.output_dir / "final_3_c_all_features_sector_test.pdf"
        
        plt.savefig(save_path_png, dpi=600, bbox_inches='tight')
        plt.savefig(save_path_pdf, dpi=600, bbox_inches='tight')
        # plt.show()

if __name__ == "__main__":
    DATA_PATH = "/Users/xiaxin/work/WindForecast_Project/01_Data/processed/matched_data/changma_matched.csv"
    OUTPUT_DIR = "/Users/xiaxin/work/WindForecast_Project/03_Results/re-plot-figures/figure-3/"
    
    analyzer = Figure3cAllFeaturesSectorTestAnalyzer(DATA_PATH, OUTPUT_DIR)
    results = analyzer.train_sector_specific_test_only()
    analyzer.create_figure(results)