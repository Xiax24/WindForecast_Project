#!/usr/bin/env python3
"""
Figure 1 (相关廓线 + 相关矩阵热图) — 各高度风速与总功率的 Spearman 相关（重写版）

相对原版的改动
--------------
1. 掩膜换 70m 单高度判据 + flatline 剔除（原版是废弃的四高度 strict 判据）。

2. 样本口径：复用 imf_masks 的 dfv（四高度风速 + power 均有效），
   与 figure-2 各面板共用同一批样本。
   这一点对本图尤其关键：r(10m,P) 与 r(70m,P) 必须在同一批行上计算，
   否则 0.81 vs 0.79 连 like-for-like 都不是。

3. **补配对 bootstrap** —— 这是必须的，不是可选。
   R2 Point 10 原话："It's an interesting result, but is it statistically
   significant?" 问的就是 r(10m) 与 r(70m) 那 0.02 的差。
   原版只给每个相关各自的 p 值，那检验的是"相关是否 != 0"
   （两万样本下当然是），答非所问。
   要回答的是"两个相关是否不同" -> 对同一批行重采样，
   同时算两个相关、取差，看差的 CI 含不含 0。
   配对是关键：两个高度的风速高度共变（r ~ 0.87），
   非配对 bootstrap 会严重高估差值的不确定性。

4. 不施加 3-25 m/s 风速筛选。该筛选卡在 70m 上，会直接截断 70m 的量程、
   只间接截断 10m，属于 range restriction，会系统性压低 70m 的相关、
   抬高正在检验的 10m-vs-70m 对比。本图是高度间的公平比较，故不设。
   （重建/预报实验里设 3-25 是合理的：那里定义的是机组运行包线，
     且对 HH/SR/ER 一视同仁。）

5. **并入 Figure S2 的三面板相关矩阵热图**（原来在废弃的 figure-1-c-0.py 里，
   那份用的是四高度 strict 掩膜，与正文其余部分口径不一致）。
   同时按 R2 Point 9 重画坐标轴：

   R2 原话：heights should be relabelled as "P_10m...P_70m" on the vertical
   axes and "U_70m...U30m" on the horizontal axes, with the 10m speed there
   omitted because it is not used.

   审稿人这里其实读错了矩阵的结构（纵轴每一行是某个高度的**风速** U_h，
   不是功率；只有第一列才是它与总功率 P 的相关），但读错的根源在我们：
   旧版两个轴都只写 "Power / 70m / 50m / 30m / 10m"，既没区分变量是 U 还是 P，
   又留着两条被 mask 成空白的边（纵轴最上面的 Power 行、横轴最右边的 10m 列）,
   看上去就像丢了数据。所以采纳他的诉求（标清变量、去掉没用的 10m），
   但不逐字采纳他的标法（那会把 U-U 的格子标成 P）：

     纵轴 上->下 :  U_70m, U_50m, U_30m, U_10m        (4 行, 删掉空的 Power 行)
     横轴 左->右 :  P,     U_70m, U_50m, U_30m        (4 列, 删掉空的 10m 列)

   于是每个格子都非空，矩阵读法唯一：第一列 = 各高度风速与总功率的相关，
   其余列 = 高度间风速的相关。

6. Spearman vs Pearson（R2 Point 9 后半句 "why Spearman and not Pearson?"）：
   正文保留 Spearman —— U-P 关系由功率曲线决定，单调但强非线性（S 形 + 额定段
   截断），Pearson 测的线性一致性在这里会低估依赖强度、且对切出风速附近的离群
   点敏感。为支撑回复信，这里同时输出一版 Pearson 热图与逐格对照表，
   用来说明高度间的排序与扇区间的差异不随相关系数的选择改变。
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from scipy import stats

from imf_masks import build_aligned_frame, build_sector_masks

# ---------------------------------------------------------------- 配置
DATA_PATH = '/Users/xiaxin/work/WindForecast_Project/01_Data/processed/matched_data/changma_matched.csv'
OUT_DIR = '/Users/xiaxin/work/WindForecast_Project/03_Results/re-plot-figures/figure-1/'

HEIGHTS = [10, 30, 50, 70]
POWER_COL = 'power'
N_BOOT = 2000
RNG_SEED = 42

plt.rcParams.update({
    'font.family': 'Arial', 'font.size': 11,
    'figure.dpi': 300, 'savefig.dpi': 300,
    # 让下标 U_70m / P 也走 Arial，别掉到 DejaVu 去
    'mathtext.fontset': 'custom',
    'mathtext.rm': 'Arial', 'mathtext.it': 'Arial', 'mathtext.bf': 'Arial:bold',
    'mathtext.default': 'regular',
})

COLORS = {'all': '#2d2d2d', 'free': '#f41111', 'wake': '#1996de'}

# ---------------------------------------------------------------- 热图配置
# 纵轴（上->下）各高度风速；横轴（左->右）总功率 + 各高度风速。
# 两轴各去掉一条原本被 mask 成空白的边：纵轴的 P 行、横轴的 U_10m 列（R2 Point 9）。
HEATMAP_ROWS = [70, 50, 30, 10]
HEATMAP_COLS = ['P', 70, 50, 30]

PANELS = [('all',  'Overall'),
          ('free', 'Westerly (free-stream)'),
          ('wake', 'Easterly (wake)')]

# 单向暖色渐变，沿用原版的红色系但去掉蓝端。
# 原版是 0.5-1.0 的 red-white-blue 双向色标，而实际取值全落在 0.8-1.0：
# 蓝色半边完全空置（还暗示"低相关"），所有格子挤在红色渐变的一小段里，
# 0.80 与 0.89 几乎分辨不出。这里改成随数据范围自适应的单向渐变。
# 起点用浅橙而非纯白，以便与上三角的空白格区分开。
HEATMAP_CMAP_COLORS = ["#FDEDE5", "#F9C3AC", "#F45F31", "#A32903"]


def u_label(h):
    return rf'$U_{{{h}\,\mathrm{{m}}}}$'


def var_label(v):
    return r'$P$' if v == 'P' else u_label(v)


def var_column(v):
    return POWER_COL if v == 'P' else f'obs_wind_speed_{v}m'


def stars(p):
    return '***' if p < 0.001 else '**' if p < 0.01 else '*' if p < 0.05 else 'n.s.'


def spearman_pair(x, y):
    return stats.spearmanr(x, y)[0]


def corr_with_p(x, y, method='spearman'):
    ok = np.isfinite(x) & np.isfinite(y)
    if ok.sum() < 4:
        return np.nan, 1.0
    f = stats.spearmanr if method == 'spearman' else stats.pearsonr
    r, p = f(x[ok], y[ok])
    return float(r), float(p)


def paired_bootstrap_diff(ws_a, ws_b, power, n_boot=N_BOOT, seed=RNG_SEED):
    """同一批行重采样 -> 同时算 r(a,P) 与 r(b,P) -> 差的分布。"""
    rng = np.random.default_rng(seed)
    n = len(power)
    diffs = np.empty(n_boot)
    for i in range(n_boot):
        idx = rng.integers(0, n, n)
        ra = spearman_pair(ws_a[idx], power[idx])
        rb = spearman_pair(ws_b[idx], power[idx])
        diffs[i] = ra - rb
    return diffs


def main():
    print("=" * 74)
    print("Figure 1 — 相关廓线（70m 单高度判据）")
    print("=" * 74)

    dfv = build_aligned_frame(DATA_PATH)
    m_free, m_wake = build_sector_masks(dfv)

    # flatline 也要从 all 里剔，保持与扇区一致
    from qc_common import flatline_mask, WS_COLS
    bad = np.zeros(len(dfv), bool)
    for c in WS_COLS:
        if c in dfv.columns:
            bad |= flatline_mask(dfv[c].values)
    m_all = ~bad

    power = dfv[POWER_COL].values
    WS = {h: dfv[f'obs_wind_speed_{h}m'].values for h in HEIGHTS}

    subsets = {'all': m_all, 'free': m_free, 'wake': m_wake}
    print(f"\n  样本: all={m_all.sum()}, free={m_free.sum()}, wake={m_wake.sum()}")

    # ---------------- 相关系数
    corr = {}
    print("\n" + "=" * 74)
    print("Spearman 相关（风速 vs 总功率）")
    print("=" * 74)
    print(f"  {'高度':>5} {'all':>9} {'free':>9} {'wake':>9}")
    for h in HEIGHTS:
        row = []
        for tag, m in subsets.items():
            r = spearman_pair(WS[h][m], power[m])
            corr.setdefault(tag, []).append(r)
            row.append(r)
        print(f"  {h:>3}m {row[0]:>9.4f} {row[1]:>9.4f} {row[2]:>9.4f}")

    for tag in subsets:
        corr[tag] = np.array(corr[tag])

    # ---------------- 配对 bootstrap: 10m vs 70m
    print("\n" + "=" * 74)
    print("配对 bootstrap: r(10m,P) - r(70m,P)   [回应 R2 Point 10]")
    print("=" * 74)
    print(f"  {'子集':>6} {'r_10m':>8} {'r_70m':>8} {'差':>8} "
          f"{'95% CI of 差':>20}  判定")

    rows = []
    for tag, m in subsets.items():
        r10 = spearman_pair(WS[10][m], power[m])
        r70 = spearman_pair(WS[70][m], power[m])
        d = paired_bootstrap_diff(WS[10][m], WS[70][m], power[m])
        lo, hi = np.percentile(d, [2.5, 97.5])
        sig = "显著 (CI不含0)" if not (lo <= 0 <= hi) else "不显著 (CI含0)"
        print(f"  {tag:>6} {r10:>8.4f} {r70:>8.4f} {r10-r70:>8.4f} "
              f"[{lo:>7.4f},{hi:>7.4f}]  {sig}")
        rows.append({'subset': tag, 'N': int(m.sum()),
                     'r_10m': r10, 'r_70m': r70, 'diff': r10 - r70,
                     'CI_lo': lo, 'CI_hi': hi,
                     'significant': not (lo <= 0 <= hi)})

    pd.DataFrame(rows).to_csv(
        os.path.join(OUT_DIR, 'fig1_corr_bootstrap.csv'),
        index=False, float_format='%.5g')

    print("\n  解读：")
    print("    all 显著为正   -> 4.1 开场'10m 相关高于 70m'成立，且现在有检验支撑")
    print("    all 不显著     -> 那句必须降级为'两者相当'，4.1 开场要重写")
    print("    wake 显著为正  -> 这是 ER/WDA 的核心依据，最重要的一行")

    # 全高度相关矩阵（Figure S1 用）
    print("\n" + "-" * 74)
    print("各高度两两相关 + 与功率（Figure S1 数据）")
    print("-" * 74)
    for tag, m in subsets.items():
        cols = [f'obs_wind_speed_{h}m' for h in HEIGHTS] + [POWER_COL]
        cm = dfv.loc[m, cols].corr(method='spearman')
        cm.to_csv(os.path.join(OUT_DIR, f'fig1_corrmatrix_{tag}.csv'),
                  float_format='%.4f')
        print(f"  [{tag}] -> fig1_corrmatrix_{tag}.csv")

    plot_profile(corr, OUT_DIR)
    heatmaps(dfv, subsets, OUT_DIR)
    print("\n" + "=" * 74)
    print("完成")
    print("=" * 74)


def plot_profile(corr, out_dir):
    fig, ax = plt.subplots(figsize=(4, 5))

    for tag, label in [('all', 'All Data'),
                       ('wake', 'Wake (Easterly)'),
                       ('free', 'Free-stream (Westerly)')]:
        c = corr[tag]
        ax.plot(c, HEIGHTS, '-', color=COLORS[tag], linewidth=3, zorder=2)
        imax = int(np.nanargmax(c))
        for i, (v, h) in enumerate(zip(c, HEIGHTS)):
            ax.plot(v, h, 'o', color=COLORS[tag], markersize=8,
                    markerfacecolor=COLORS[tag] if i == imax else 'white',
                    markeredgecolor=COLORS[tag], markeredgewidth=1.5, zorder=3)
        ax.plot([], [], 'o-', color=COLORS[tag], linewidth=3, markersize=8,
                markerfacecolor='white', markeredgecolor=COLORS[tag],
                markeredgewidth=1.5, label=label)

    ax.set_ylabel('Height (m)', fontweight='bold', fontsize=18)
    ax.set_title('Spearman Correlation', fontweight='bold', pad=10, fontsize=19)
    ax.set_ylim(0, max(HEIGHTS) + 18)
    ax.set_yticks(np.arange(0, max(HEIGHTS) + 19, 10))
    ax.set_yticklabels([str(h) if h in HEIGHTS else ''
                        for h in np.arange(0, max(HEIGHTS) + 18, 10)])

    # 自适应 x 范围（原版写死 0.70 下限，新判据下可能裁掉数据）
    allc = np.concatenate([corr[t] for t in ('all', 'free', 'wake')])
    lo, hi = np.nanmin(allc), np.nanmax(allc)
    pad = max(0.01, (hi - lo) * 0.15)
    ax.set_xlim(lo - pad, hi + pad)

    ax.grid(True, linestyle='dotted', alpha=0.5, linewidth=0.8, color='gray')
    ax.tick_params(axis='both', which='major', labelsize=18, width=1.2,
                   length=5, direction='in')
    ax.legend(loc='upper right', frameon=False, fontsize=11, markerfirst=False)

    plt.tight_layout()
    os.makedirs(out_dir, exist_ok=True)
    base = os.path.join(out_dir, 'fig1_correlation_profile')
    for ext in ('.png', '.pdf'):
        plt.savefig(base + ext, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"\n  ✓ {base}.png")


def compute_heatmap_matrices(dfv, subsets, method='spearman'):
    """每个子集一张 4x4 (行=U_h, 列=[P, U_h'])，上三角(j>i)为 NaN。"""
    nr, nc = len(HEATMAP_ROWS), len(HEATMAP_COLS)
    out = {}
    for tag, m in subsets.items():
        R = np.full((nr, nc), np.nan)
        P = np.full((nr, nc), np.nan)
        for i, h in enumerate(HEATMAP_ROWS):
            x = dfv[f'obs_wind_speed_{h}m'].values[m]
            for j, v in enumerate(HEATMAP_COLS):
                if j > i:                     # 上三角：与下三角重复，留空
                    continue
                y = dfv[var_column(v)].values[m]
                R[i, j], P[i, j] = corr_with_p(x, y, method)
        out[tag] = {'r': R, 'p': P, 'n': int(m.sum())}
    return out


def plot_correlation_heatmap(mats, out_dir, method='spearman', fname=None):
    """三面板相关矩阵热图（Figure S2）。"""
    cmap = mcolors.LinearSegmentedColormap.from_list(
        'corr', HEATMAP_CMAP_COLORS, N=256)
    cmap = cmap.copy()
    cmap.set_bad('white')

    # 三个面板共用一条色标（面板间必须可比），范围随数据自适应：
    # 向下取到 0.05 的整数倍，避免把 0.8-1.0 的数据画在 0.5-1.0 的标尺上。
    allr = np.concatenate([np.ravel(mats[t]['r']) for t, _ in PANELS])
    vmin = np.floor(np.nanmin(allr) * 20) / 20
    vmax = 1.0
    norm = mcolors.Normalize(vmin=vmin, vmax=vmax)

    fs = 19            # 刻度/标题
    fs_annot = 17      # 格内数字

    fig, axes = plt.subplots(1, 3, figsize=(16.5, 5.0))
    fig.subplots_adjust(wspace=0.28)

    for ax, (tag, title) in zip(axes, PANELS):
        R, P = mats[tag]['r'], mats[tag]['p']
        ax.imshow(np.ma.masked_invalid(R), cmap=cmap, norm=norm,
                  aspect='equal', interpolation='nearest')

        for i in range(R.shape[0]):
            for j in range(R.shape[1]):
                if not np.isfinite(R[i, j]):
                    continue
                rgba = cmap(norm(R[i, j]))
                lum = 0.299 * rgba[0] + 0.587 * rgba[1] + 0.114 * rgba[2]
                ax.text(j, i, f'{R[i, j]:.2f}\n{stars(P[i, j])}',
                        ha='center', va='center', fontsize=fs_annot,
                        color='white' if lum < 0.55 else 'black',
                        linespacing=1.15)

        ax.set_xticks(range(len(HEATMAP_COLS)))
        ax.set_xticklabels([var_label(v) for v in HEATMAP_COLS], fontsize=fs)
        ax.set_yticks(range(len(HEATMAP_ROWS)))
        ax.set_yticklabels([u_label(h) for h in HEATMAP_ROWS], fontsize=fs)
        ax.tick_params(length=0, pad=6)

        # 细白线分隔格子，替代 seaborn 的 linewidths
        ax.set_xticks(np.arange(-0.5, len(HEATMAP_COLS), 1), minor=True)
        ax.set_yticks(np.arange(-0.5, len(HEATMAP_ROWS), 1), minor=True)
        ax.grid(which='minor', color='white', linewidth=2)
        ax.grid(which='major', visible=False)
        ax.tick_params(which='minor', length=0)
        for s in ax.spines.values():
            s.set_visible(False)

        ax.set_title(title, fontsize=fs, fontweight='bold', pad=12)

    label = (r'Spearman $\rho$' if method == 'spearman' else r'Pearson $r$')
    cbar = fig.colorbar(plt.cm.ScalarMappable(norm=norm, cmap=cmap),
                        ax=axes, fraction=0.030, pad=0.02, shrink=0.82)
    cbar.set_label(label, fontsize=fs, labelpad=14)
    cbar.ax.tick_params(labelsize=fs)
    cbar.outline.set_visible(False)

    os.makedirs(out_dir, exist_ok=True)
    if fname is None:
        fname = ('windspeed_power_correlation_heatmap_all' if method == 'spearman'
                 else 'windspeed_power_correlation_heatmap_all_pearson')
    base = os.path.join(out_dir, fname)
    for ext in ('.png', '.pdf'):
        plt.savefig(base + ext, dpi=500, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"  ✓ {base}.png / .pdf")


def report_heatmap(mats, out_dir, method='spearman'):
    """存逐格数值，供 caption 与回复信引用。"""
    rows = []
    for tag, title in PANELS:
        R, P, n = mats[tag]['r'], mats[tag]['p'], mats[tag]['n']
        for i, h in enumerate(HEATMAP_ROWS):
            for j, v in enumerate(HEATMAP_COLS):
                if not np.isfinite(R[i, j]):
                    continue
                rows.append({'subset': tag, 'N': n, 'method': method,
                             'y_var': f'U_{h}m',
                             'x_var': 'P' if v == 'P' else f'U_{v}m',
                             'r': R[i, j], 'p': P[i, j]})
    df = pd.DataFrame(rows)
    path = os.path.join(out_dir, f'figS2_corr_cells_{method}.csv')
    df.to_csv(path, index=False, float_format='%.5g')
    print(f"  ✓ {os.path.basename(path)}")
    return df


def heatmaps(dfv, subsets, out_dir):
    print("\n" + "=" * 74)
    print("Figure S2 — 三面板相关矩阵热图 [回应 R2 Point 9]")
    print("=" * 74)

    tables = {}
    for method in ('spearman', 'pearson'):
        mats = compute_heatmap_matrices(dfv, subsets, method)
        plot_correlation_heatmap(mats, out_dir, method)
        tables[method] = report_heatmap(mats, out_dir, method)

    # Spearman vs Pearson 对照 —— 回复信要用的就是这张表
    key = ['subset', 'y_var', 'x_var']
    cmp = (tables['spearman'].rename(columns={'r': 'spearman'})[key + ['N', 'spearman']]
           .merge(tables['pearson'].rename(columns={'r': 'pearson'})[key + ['pearson']],
                  on=key))
    cmp['diff'] = cmp['spearman'] - cmp['pearson']
    cmp.to_csv(os.path.join(out_dir, 'figS2_spearman_vs_pearson.csv'),
               index=False, float_format='%.5g')

    up = cmp[cmp['x_var'] == 'P']
    print("\n  风速 vs 总功率：Spearman / Pearson 对照")
    print(f"    {'子集':>6} {'高度':>7} {'Spearman':>9} {'Pearson':>9} {'差':>8}")
    for _, r in up.iterrows():
        print(f"    {r['subset']:>6} {r['y_var']:>7} {r['spearman']:>9.4f} "
              f"{r['pearson']:>9.4f} {r['diff']:>8.4f}")
    print("\n  ✓ figS2_spearman_vs_pearson.csv")
    print("    检查点：若两列给出的高度排序一致、且 wake<free 的方向一致，")
    print("    则回复信可写『结论不随相关系数的选择改变』。")


if __name__ == '__main__':
    main()
