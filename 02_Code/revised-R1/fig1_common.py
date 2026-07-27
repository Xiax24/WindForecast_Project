#!/usr/bin/env python3
"""
fig1_common.py — Figure 1 四个面板共用的数据口径与排版参数。

四个面板分别独立出图（由作者在版面软件里拼装），所以字号、配色、
标记形状必须在这里统一定义，不能各画各的。
"""

import os
import sys

import numpy as np
import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt

REPLOT_DIR = '/Users/xiaxin/work/WindForecast_Project/02_Code/re-plot'
if REPLOT_DIR not in sys.path:
    sys.path.insert(0, REPLOT_DIR)

from imf_masks import build_aligned_frame, build_sector_masks   # noqa: E402
from qc_common import flatline_mask, WS_COLS                    # noqa: E402

DATA_PATH = ('/Users/xiaxin/work/WindForecast_Project/01_Data/processed/'
             'matched_data/changma_matched.csv')
SUPP_DIR = '/Users/xiaxin/Desktop/GRL-revise/补充数据材料'
DEM_PATH = os.path.join(SUPP_DIR, 'yumen_glo30_extended.tif')
TURBINE_JSON = os.path.join(SUPP_DIR, 'yumen_dem/turbine_locations_from_xlsx.json')

# 输出落在结果目录，与其余 *_fixed.py 一致。
# 不要写成 dirname(__file__) —— 那样从 02_Code/re-plot/ 跑会把 png 吐进代码目录。
OUT_DIR = ('/Users/xiaxin/work/WindForecast_Project/03_Results/'
           're-plot-figures/figure-1/')

HEIGHTS = [10, 30, 50, 70]
POWER_COL = 'power'
N_BOOT = 2000
RNG_SEED = 42

MAST = {'lon': 96 + 48 / 60 + 41.00 / 3600,
        'lat': 40 + 12 / 60 + 19.10 / 3600}

# 配色与标记：C 和 D 共用，拼版后读者才能把两张图对上
COLORS = {'all': '#2d2d2d', 'free': '#f41111', 'wake': '#1996de'}
# 三条曲线靠颜色已能区分，不再叠加形状差异（D 面板本来就全是圆圈）
MARKERS = {'all': 'o', 'free': 'o', 'wake': 'o'}
LABELS = {'all': 'All data',
          'free': 'Free-stream (westerly)',
          'wake': 'Wake (easterly)'}


# ---------------------------------------------------------------- 字号
# 四个面板独立出图、由作者拼版，所以字号必须在这里集中定义。
# 之前每个脚本各写各的硬编码值（B 的径向标签 6.5、图例 6.8，
# A 的 colorbar 刻度 7.5、图例 7，C/D 的图例 7.2），拼到一起字号乱跳。
# 按最终印刷尺寸出图，这里的 8 pt 就是印出来的 8 pt ——
# 前提是拼版时四张图用同一个缩放比例。
FS_TICK = 8       # 所有刻度数字（含 colorbar 刻度、风玫瑰径向/方位标签）
FS_LABEL = 9      # 轴标题、colorbar 标题
FS_LEGEND = 7.5   # 图例条目
FS_ANNOT = 7.5    # 图内注记（比例尺数字、指北针 N）

# ---------------------------------------------------------------- 框线
# A 的地图边框圈着一块深色底图、B 的外圈是一整个闭合黑圆，
# 两者的视觉重量都压过 C、D 的坐标框；四张拼在一起时左列会明显发沉。
# 统一减细并从纯黑降到深灰，A/B 才不会抢在 C/D 前面。
AXES_LW = 0.6
AXES_COLOR = '#3a3a3a'


def apply_style():
    """四个面板统一的字号体系。按最终印刷尺寸出图，图上 8 pt 即印出 8 pt。"""
    plt.rcParams.update({
        'font.family': 'Arial',
        'font.size': FS_TICK,
        'axes.labelsize': FS_LABEL,
        'axes.titlesize': FS_LABEL,
        'xtick.labelsize': FS_TICK,
        'ytick.labelsize': FS_TICK,
        'legend.fontsize': FS_LEGEND,
        'axes.linewidth': AXES_LW,
        'axes.edgecolor': AXES_COLOR,
        'xtick.color': AXES_COLOR,
        'ytick.color': AXES_COLOR,
        'pdf.fonttype': 42,
        'ps.fonttype': 42,
        'figure.dpi': 200,
        'savefig.dpi': 600,
        'mathtext.fontset': 'custom',
        'mathtext.rm': 'Arial', 'mathtext.it': 'Arial:italic',
        'mathtext.bf': 'Arial:bold', 'mathtext.default': 'regular',
    })


class PanelCtx:
    """让同一份绘图代码既能单张出图、也能画进拼版大图。

    每个面板脚本用英寸描述自己的内部版面（原点在本面板方框的左下角），
    由本对象换算成 figure 归一化坐标；单张出图时 x0 = y0 = 0。

    保留偏移能力是为了以后若再需要把面板画进一张大图时不用改绘图代码。
    Figure 1 的拼版由作者在版面软件里完成（figure1.ai），不再出拼版脚本。
    """

    def __init__(self):
        self.fig = None
        self.x0 = 0.0
        self.y0 = 0.0

    def bind(self, fig, x0=0.0, y0=0.0):
        self.fig, self.x0, self.y0 = fig, x0, y0
        return self

    def rect(self, x, y, w, h):
        fw, fh = self.fig.get_size_inches()
        return [(self.x0 + x) / fw, (self.y0 + y) / fh, w / fw, h / fh]

    def ax(self, x, y, w, h, **kw):
        return self.fig.add_axes(self.rect(x, y, w, h), **kw)

    def pt(self, x, y):
        """图内某点的 figure 归一化坐标（给 legend 的 bbox_to_anchor 用）。"""
        fw, fh = self.fig.get_size_inches()
        return ((self.x0 + x) / fw, (self.y0 + y) / fh)


def load_observations(verbose=True):
    """四个面板共用同一批行：IMF 对齐样本 + 70m 单高度扇区判据 + flatline 剔除。"""
    dfv = build_aligned_frame(DATA_PATH, verbose=verbose)
    m_free, m_wake = build_sector_masks(dfv, verbose=verbose)

    bad = np.zeros(len(dfv), bool)
    for c in WS_COLS:
        if c in dfv.columns:
            bad |= flatline_mask(dfv[c].values)
    m_all = ~bad

    if verbose:
        print(f"\n  样本: all={m_all.sum()}, free={m_free.sum()}, wake={m_wake.sum()}")
    return dfv, m_all, m_free, m_wake


def align_xlabels(fig, axes, gap_in=0.035):
    """把若干个轴的 x 轴标签对齐到同一条基线，并尽量贴近刻度。

    两个问题一起解决：

    1. **基线不齐**。xlabel 默认 va='top'，定位的是文本包围盒的顶部；
       'm·s^-1' 里的 mathtext 上标把盒子撑高，顶部对齐就意味着基线更低，
       看起来一个高一个低。这里统一改成按基线对齐。

    2. **高度靠手调**。写死一个 y 值，提高一点上标就顶到刻度数字，
       压低一点又留一大片空白。这里直接量：取所有轴刻度标签的最低点，
       让最高的那个标签顶端落在它下方 gap_in 英寸处，反推公共基线。

    必须在所有刻度、标签都设完之后调用。
    """
    fig.canvas.draw()
    rend = fig.canvas.get_renderer()

    tick_bottom = min(t.get_window_extent(rend).y0
                      for ax in axes for t in ax.get_xticklabels()
                      if t.get_text())

    for ax in axes:
        ax.xaxis.label.set_verticalalignment('baseline')

    # 先落在一个参考位置，量出各标签顶端相对基线的高度
    ref = -0.12
    for ax in axes:
        ax.xaxis.set_label_coords(0.5, ref)
    fig.canvas.draw()
    label_top = max(ax.xaxis.label.get_window_extent(rend).y1 for ax in axes)

    dy_px = (tick_bottom - gap_in * fig.dpi) - label_top
    dy_frac = dy_px / (axes[0].get_window_extent(rend).height)
    for ax in axes:
        ax.xaxis.set_label_coords(0.5, ref + dy_frac)


def height_axis(ax, label=True):
    """C、D 共用的高度轴：同样的范围与刻度，拼版后两张图能对齐。"""
    ax.set_ylim(0, 80)
    ax.set_yticks([0, 10, 30, 50, 70, 80])
    ax.set_yticklabels(['', '10', '30', '50', '70', ''])
    if label:
        ax.set_ylabel('Height (m)')


def save(fig, name):
    os.makedirs(OUT_DIR, exist_ok=True)
    base = os.path.join(OUT_DIR, name)
    fig.savefig(base + '.png', facecolor='white')
    fig.savefig(base + '.pdf', facecolor='white')
    plt.close(fig)
    print(f"\n  ✓ {base}.png / .pdf")
