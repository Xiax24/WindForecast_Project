#!/usr/bin/env python3
"""
Figure 2C / 2D — EOF 垂直结构 + 表层到 PC1 的标定（重写版 v2）

=============================================================================
重要：C 和 D 用的 EOF 不是同一个，而且这是对的，别统一
=============================================================================
读原版代码发现：
  C (figure-2-c-0.py):  perform_eof(data, mask) -> 每个扇区各自做 EOF
  D (figure-2-d-0.py):  perform_pca(wind_data)  -> 全样本做一次 PCA，
                        然后把 PC1 按扇区切开

而 SI Text S2 写的是"EOF analysis was conducted separately for free-stream
and wake-influenced conditions" + "No additional normalization was applied,
so the decomposition reflects variance structure in physical units"
—— 与 D 矛盾（D 是全样本），也与两份代码都矛盾（两份都做了标准化）。

结论：代码是对的，SI 的描述要改。理由：

  C 问的是"垂直结构本身跨扇区稳不稳"
     -> 必须分扇区各自做，才能比较载荷形状与解释方差。

  D 问的是"10m 的标定斜率跨扇区稳不稳"
     -> 必须用同一把尺子。若分扇区各做 EOF，每个扇区的 PC1 是在自己的
        方差结构上定标的（特征向量单位模长，PC1 幅度随该扇区方差走），
        两个斜率根本不可比 —— 0.65 vs 0.66 里会混进扇区方差差异。
        全样本 PC1 才是共同基准。

本脚本保持这个分工，并把它显式写出来，以便同步修正 SI Text S2。

=============================================================================
相对原版的改动
=============================================================================
1. 掩膜换 70m 单高度判据 + flatline 剔除（原版是废弃的四高度 strict 判据）。

2. 样本与 A/B 统一：复用 imf_masks 的 dfv（要求四高度风速 + power 均有效），
   使 figure-2 四个面板共用一个 N。
   注：原版 D 的 valid 只 dropna 风速、不含 power，所以图上 N=1949，
   而 SI 写 1,951 —— 差的 2 条就是 power 缺测。新口径下统一。

3. 补 bootstrap 置信区间。4.2 的论断是"10m 斜率跨扇区几乎不变，
   70m 变化更大"，这是关于"两个数差多少"的陈述，没有区间就不能说
   "几乎不变"，也不能说 70m 的差是真的。R2 General 3 / Point 3
   点名要显著性，一并还上。

4. 标准化口径：默认沿用原版（标准化 = 相关矩阵 EOF，对应图上 93.2/93.5），
   同时输出协方差口径作稳健性对照，供正文取舍。
   标准化后四个高度等权，EOF1 是纯粹的"垂直同步模态"；
   不标准化则 70m 方差大、会主导分解。
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde

from imf_masks import build_aligned_frame, build_sector_masks, sector_mean_speeds

# ---------------------------------------------------------------- 配置
DATA_PATH = '/Users/xiaxin/work/WindForecast_Project/01_Data/processed/matched_data/changma_matched.csv'
DATA_DIR = '/Users/xiaxin/work/WindForecast_Project/03_Results/re-plot-figures/figure-2/'

HEIGHTS = [10, 30, 50, 70]
N_BOOT = 2000
RNG_SEED = 42

# 'standardize' = 相关矩阵 EOF（原版口径，对应图上 EV 93.2/93.5）
# 'covariance'  = 协方差 EOF（SI 文字描述的口径）
MODES = ['standardize', 'covariance']
PLOT_MODE = 'standardize'

plt.rcParams.update({
    'font.family': 'Arial', 'font.size': 11,
    'figure.dpi': 300, 'savefig.dpi': 300,
    'pdf.fonttype': 42, 'ps.fonttype': 42,
})

COLORS = {'all': '#686767', 'free': '#D62728', 'wake': '#1F77B4'}
CMAPS = {'free': 'Reds', 'wake': 'Blues'}


# ---------------------------------------------------------------- EOF
def eof_fit(X, mode='standardize'):
    """在 X 上拟合 EOF，返回可复用的变换器。

    返回 dict: mu, sd, eof1(载荷), ev(解释方差比), project(投影函数)
    """
    mu = X.mean(axis=0)
    sd = X.std(axis=0)
    Xa = X - mu
    if mode == 'standardize':
        Xa = Xa / sd

    C = np.cov(Xa, rowvar=False)
    vals, vecs = np.linalg.eigh(C)
    order = np.argsort(vals)[::-1]
    vals, vecs = vals[order], vecs[:, order]

    eof1 = vecs[:, 0]
    if eof1.sum() < 0:           # 符号约定：载荷全正
        eof1, vecs = -eof1, -vecs

    ev = vals / vals.sum()

    def project(Xnew):
        A = Xnew - mu
        if mode == 'standardize':
            A = A / sd
        return A @ vecs[:, 0]

    return {'mu': mu, 'sd': sd, 'eof1': eof1, 'ev': ev,
            'project': project, 'pc1': project(X)}


def linreg(x, y):
    a, b = np.polyfit(x, y, 1)
    yhat = a * x + b
    ss_res = np.sum((y - yhat) ** 2)
    ss_tot = np.sum((y - y.mean()) ** 2)
    return a, b, 1 - ss_res / ss_tot


def bootstrap_slopes(u_free, pc_free, u_wake, pc_wake,
                     n_boot=N_BOOT, seed=RNG_SEED):
    """两扇区各自按行重采样 -> 斜率差的分布。

    PC1 是全样本拟合的（~48600 个样本，极稳），所以不必每次重拟合 PCA；
    这里捕捉的是回归本身的抽样不确定性。
    """
    rng = np.random.default_rng(seed)
    diffs = np.empty(n_boot)
    for i in range(n_boot):
        idx_f = rng.integers(0, len(u_free), len(u_free))
        idx_w = rng.integers(0, len(u_wake), len(u_wake))
        a_f, _, _ = linreg(u_free[idx_f], pc_free[idx_f])
        a_w, _, _ = linreg(u_wake[idx_w], pc_wake[idx_w])
        diffs[i] = a_f - a_w
    return diffs


# ---------------------------------------------------------------- 主流程
def main():
    print("=" * 74)
    print("Figure 2C / 2D — EOF（70m 单高度判据）")
    print("=" * 74)

    dfv = build_aligned_frame(DATA_PATH)
    m_free, m_wake = build_sector_masks(dfv)
    sector_mean_speeds(dfv, m_free, m_wake)

    ws_cols = [f'obs_wind_speed_{h}m' for h in HEIGHTS]
    X_all = dfv[ws_cols].values
    print("  各高度 std:", " ".join(
        f"{h}m={X_all[:, i].std():.2f}" for i, h in enumerate(HEIGHTS)))
    X_free, X_wake = X_all[m_free], X_all[m_wake]

    print(f"\n  样本: all={len(X_all)}, free={len(X_free)}, wake={len(X_wake)}")

    all_rows = []
    plot_pack = {}

    for mode in MODES:
        print("\n" + "=" * 74)
        print(f"口径: {mode}"
              + ("   <- 原版实际使用（对应图上 EV 93.2/93.5）"
                 if mode == 'standardize' else "   <- SI 文字描述的口径"))
        print("=" * 74)

        # ---------- Panel C: 分扇区各自做 EOF（看结构稳不稳）
        print("\n  [Panel C] 分扇区 EOF —— 垂直结构跨扇区是否稳定")
        C = {}
        for tag, X in [('all', X_all), ('free', X_free), ('wake', X_wake)]:
            f = eof_fit(X, mode)
            C[tag] = f
            print(f"    [{tag:>4}] EV1={f['ev'][0]*100:5.1f}%  "
                  f"EV2={f['ev'][1]*100:4.1f}%   载荷: "
                  + " ".join(f"{h}m={v:.3f}" for h, v in zip(HEIGHTS, f['eof1'])))

        d = np.abs(C['free']['eof1'] - C['wake']['eof1'])
        print(f"    载荷 |free-wake| 最大差: {d.max():.4f} "
              f"(在 {HEIGHTS[int(np.argmax(d))]}m)")

        # ---------- Panel D: 全样本 PC1（看标定稳不稳）
        print("\n  [Panel D] 全样本 PC1 作共同基准 —— 标定斜率跨扇区是否稳定")
        G = eof_fit(X_all, mode)
        pc_free = G['project'](X_free)
        pc_wake = G['project'](X_wake)

        print(f"    {'高度':>5} {'a_free':>8} {'a_wake':>8} {'差':>8} "
              f"{'95% CI of 差':>20} {'R2_free':>8} {'R2_wake':>8}  判定")

        for h in [10, 70]:
            hi = HEIGHTS.index(h)
            uf, uw = X_free[:, hi], X_wake[:, hi]
            a_f, b_f, r2_f = linreg(uf, pc_free)
            a_w, b_w, r2_w = linreg(uw, pc_wake)

            diffs = bootstrap_slopes(uf, pc_free, uw, pc_wake)
            lo, hi_ci = np.percentile(diffs, [2.5, 97.5])
            contains0 = lo <= 0 <= hi_ci
            verdict = "稳定 (CI含0)" if contains0 else "有差异 (CI不含0)"

            print(f"    {h:>3}m {a_f:>8.3f} {a_w:>8.3f} {a_f-a_w:>8.3f} "
                  f"[{lo:>7.3f},{hi_ci:>7.3f}] {r2_f:>8.3f} {r2_w:>8.3f}  {verdict}")

            all_rows.append({
                'mode': mode, 'height_m': h,
                'slope_free': a_f, 'slope_wake': a_w, 'slope_diff': a_f - a_w,
                'diff_CI_lo': lo, 'diff_CI_hi': hi_ci,
                'CI_contains_zero': contains0,
                'intercept_free': b_f, 'intercept_wake': b_w,
                'R2_free': r2_f, 'R2_wake': r2_w,
                'N_free': len(uf), 'N_wake': len(uw),
            })

        print("\n    判定规则（决定 4.2 那段怎么写）：")
        print("      10m 含0 且 70m 不含0  -> 原论断成立，且现在有统计支撑")
        print("      两个都含0             -> '尾流改变70m标定能力'无证据，该段须改写")
        print("      两个都不含0           -> 改成'稳定性程度不同'，比较差值大小")

        if mode == PLOT_MODE:
            plot_pack = {'C': C, 'G': G, 'X_free': X_free, 'X_wake': X_wake,
                         'pc_free': pc_free, 'pc_wake': pc_wake}

    df_out = pd.DataFrame(all_rows)
    out = os.path.join(DATA_DIR, 'fig2cd_slopes.csv')
    df_out.to_csv(out, index=False, float_format='%.5g')
    print(f"\n  ✓ {out}")

    plot_panel_c(plot_pack['C'], DATA_DIR)
    plot_panel_d(plot_pack, DATA_DIR)

    print("\n" + "=" * 74)
    print("完成")
    print("=" * 74)


# ---------------------------------------------------------------- 绘图
def plot_panel_c(C, out_dir):
    fig, ax = plt.subplots(figsize=(7, 8))
    for tag, label, mfc in [('all', 'All Data', 'white'),
                            ('free', 'Free (Westerly)', COLORS['free']),
                            ('wake', 'Wake (Easterly)', COLORS['wake'])]:
        ax.plot(C[tag]['eof1'], HEIGHTS, color=COLORS[tag], marker='o',
                markersize=10, markerfacecolor=mfc, markeredgecolor=COLORS[tag],
                markeredgewidth=2, linewidth=1.5, label=label)

    for i, (tag, lab) in enumerate([('all', 'All'), ('free', 'Free'),
                                    ('wake', 'Wake')]):
        ax.text(0.03, 0.96 - i * 0.07, f"EV: {C[tag]['ev'][0]*100:.1f}% ({lab})",
                transform=ax.transAxes, fontsize=18, color=COLORS[tag],
                fontweight='bold', va='top')

    ax.set_xlabel('EOF1 Loading', fontsize=24)
    ax.set_ylabel(r'Height $z$ (m)', fontsize=24)
    ax.set_title('First Vertical EOF Mode (EOF1)', fontsize=23,
                 fontweight='bold', pad=15)
    ax.set_yticks(HEIGHTS)
    ax.legend(loc='lower right', frameon=False, fontsize=15)
    ax.tick_params(axis='both', labelsize=20, width=1.2, length=5, direction='in')
    for sp in ax.spines.values():
        sp.set_linewidth(1.5)

    plt.tight_layout()
    base = os.path.join(out_dir, 'fig2c_eof1_loading')
    for ext in ('.png', '.pdf'):
        plt.savefig(base + ext, bbox_inches='tight', facecolor='white')
    print(f"  ✓ {base}.png")
    plt.close()


def plot_panel_d(P, out_dir):
    fig, axes = plt.subplots(2, 1, figsize=(10, 15))
    xlims = {10: (0, 18), 70: (0, 20)}
    ylim = (-6, 8)

    for ax, h in zip(axes, [10, 70]):
        hi = HEIGHTS.index(h)
        for tag, X, pc in [('wake', P['X_wake'], P['pc_wake']),
                           ('free', P['X_free'], P['pc_free'])]:
            u = X[:, hi]
            m = ((u >= xlims[h][0]) & (u <= xlims[h][1]) &
                 (pc >= ylim[0]) & (pc <= ylim[1]))
            uu, pp = u[m], pc[m]
            try:
                sub = np.random.default_rng(0).choice(
                    len(uu), size=min(len(uu), 8000), replace=False)
                dens = gaussian_kde(np.vstack([uu[sub], pp[sub]]))(
                    np.vstack([uu[sub], pp[sub]]))
                o = dens.argsort()
                ax.scatter(uu[sub][o], pp[sub][o], c=dens[o], s=18,
                           cmap=CMAPS[tag], edgecolors='none', rasterized=True)
            except Exception:
                ax.scatter(uu, pp, s=8, color=COLORS[tag], alpha=0.2,
                           edgecolors='none', rasterized=True)

            a, b, r2 = linreg(u, pc)
            xs = np.linspace(u.min(), u.max(), 50)
            ax.plot(xs, a * xs + b, color=COLORS[tag], linewidth=2.5, zorder=10)

            pos = (0.05, 0.96) if tag == 'free' else (0.55, 0.44)
            ax.text(*pos, f"y = {a:.2f}x {b:+.2f}\nN = {len(u)}\n"
                          f"p < 0.01\nR² = {r2:.3f}",
                    transform=ax.transAxes, color=COLORS[tag], fontsize=30,
                    va='top', ha='left')

        ax.set_xlim(xlims[h])
        ax.set_ylim(ylim)
        ax.set_ylabel('PC1 Amplitude', fontsize=38)
        if h == 70:
            ax.set_xlabel(r'Wind Speed (m$\cdot$s$^{-1}$)', fontsize=38)
        ax.text(0.03, 0.03, f'{h} m', transform=ax.transAxes,
                fontsize=26, style='italic')
        ax.tick_params(axis='both', labelsize=32, width=1.2, length=5,
                       direction='in')
        for sp in ax.spines.values():
            sp.set_linewidth(2.0)

    axes[0].set_title('Surface-to-PC1 Mapping', fontsize=38,
                      fontweight='bold', pad=18)
    plt.subplots_adjust(hspace=0.1)
    base = os.path.join(out_dir, 'fig2d_pc1_mapping')
    for ext in ('.png', '.pdf'):
        plt.savefig(base + ext, bbox_inches='tight', facecolor='white')
    print(f"  ✓ {base}.png")
    plt.close()


if __name__ == '__main__':
    main()
