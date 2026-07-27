#!/usr/bin/env python3
"""
Figure 1A — 场址地形（Copernicus GLO-30）、134 个机位与气象塔，附中国定位图。

取代已作废的 `03_Results/figures/风电场布设位置示意图/位置示意图.ai`：
那张示意图写的是 33 台、4 列、列内 300 m、列间 1200 m，
与正文/Text S1 的 134 台、5 列、列内中位 247 m (3.01 D)、列间 1469 m (17.9 D)
完全对不上，不能再用。

相对 `补充数据材料/code/plot_yumen_dem_all_turbines.py` 的修正
--------------------------------------------------------------
1. 图例整体移到图框外。原来是压在地形上的一个不透明色块，
   还和比例尺叠在一起 —— 地图上最不该被遮的就是数据本身。
2. 去掉气象塔旁的文字标注（图例里已有三角形符号，重复且压地形）；
   三角形也去掉深色描边，青色本身对比度已经够。
3. 比例尺移到右下角（左下是第一列机组的末端、气象塔也在左侧）。
   保留黑白相间的分段块（样式对齐 季节预测-方案一/papers/
   paper1_seasonal_wind_assessment/fig1a_terrain.py），只去掉文字的
   白描边 —— 中途曾把分段块一起删成一条光秃秃的线，反而更难看。
4. 指北针放到左上角（第 1 列机组以西、气象塔以北的空白）。
   样式沿用玉门原图：N 在下、实心箭头朝上。
5. 用词：
     "Met mast"             -> "Meteorological mast"（与正文一致）
     "Wind turbine (n=134)" -> "Wind turbines (134)"
       （n 在本文中一律表样本量，不用于设备计数）
6. colorbar 右侧留够刻度数字 + 竖排标题的宽度。原来右边距不足，
   "1720" 这类四位数会被图边直接切掉。刻度按 40 m 一档显式给定。
7. 按最终印刷尺寸出图，图上 8 pt 即印出的 8 pt。

本轮又改的三处（都是"图和正文说的不是一回事"）
-----------------------------------------------
8. **配色换成单向递变**。原来低海拔端是深青绿色 —— 这是戈壁滩，
   把全场最低处染成看着像水体/植被的颜色，且用一条跨度那么大的
   双色调色带去表达全场仅 133 m 的落差，视觉上把一块极平的地方
   画得像有地貌起伏。改成浅砂 -> 深褐的单一色系，深 = 高。
9. **晕渲减弱**。原来 shade 直接算在 sigma=1.2 的地形上，
   1-2 m 的沙垄纹理被强化得很显眼，而 Text S1 写的是
   "相对最佳拟合平面的残余起伏，农场范围内 1.91 m RMS"，
   图的观感和 SI 的数字对不上。现在晕渲改算在 sigma=5.0 的地形上
   （约 150 m 尺度），并只以 0.35 的权重混进纯色，
   保留区域坡度的立体感、压掉微尺度纹理。
   等高线仍用 sigma=1.2 的地形（那是几何量，不该跟着削）。
10. **机位点缩小、地图加宽**。列内中位间距 247 m，按上一版
    2.10 in 的地图宽度换算，圆心间距 1.70 mm 而点直径 1.05 mm，
    只剩 0.65 mm 空隙，再缩一点第一列就糊成一条线。
    现在地图 2.25 in、点 s=5，空隙约 0.93 mm。
    **拼版时 panel A 不要窄于 88 mm。**

定位图的数据来源（重要，别再走弯路）
------------------------------------
边界几何来自 **cnmaps-data 1.1.2（provider = official）**，
装在另一个项目的虚拟环境里：
    /Users/xiaxin/Desktop/季节预测-方案一/.venv   (Python 3.12)
用 `cn_boundaries.py` 从那里导出到 `cn_boundaries.json`（本目录），
本脚本只读这个 json，不依赖那个 venv。

之前试过、走不通的两条路，不要重来：
  - basemap 自带的 countries 数据是**弧段**不是闭合多边形。用 shapely
    把海岸线 + 国界 polygonize 后，包含站点的那块多边形边界是
    -9.5..180°E / 1.3..77.7°N —— 横跨整个欧亚大陆，说明国界有缺口、
    闭合不了，中国没法单独填色。
  - basemap 的 states 数据在中国境内 **0 段**（只覆盖美国、加拿大、
    澳大利亚），省界拿不到。

cnmaps 的数据同时给出国界、省界和**海疆线（十段线，10 段 199 点，
amap/maritime/100000.geojson）**，三样齐了才画得成这张图。

⚠️ 仍需作者确认：本图属于"编辑过的地图"。若单位成果报送或国内
   项目验收要求标准地图审图号，须按自然资源部标准地图服务
   （http://bzdt.ch.mnr.gov.cn）另行处理。GRL 投稿本身不受此约束。

图注里必须交代（图上表达不了的）
--------------------------------
- 灰细线是 20 m 间隔的等高线（图例排不下第三项，只能进图注）；
- 等高线绘自 sigma = 1.2 像素高斯平滑后的地形；
- 晕渲为视觉增强（光源方位 315°、高度角 42°，算在 sigma = 5.0 像素的
  平滑地形上，以 0.35 权重混入），不携带定量信息。

本机没有 rasterio，用 PIL 直接读 GeoTIFF：GLO-30 是 EPSG:4326 上的规则网格，
比例尺与角点由 ModelPixelScale(33550) / ModelTiepoint(33922) 两个标签给出。
"""

import json
import os

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.colors import LightSource, LinearSegmentedColormap
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle
from matplotlib.ticker import FuncFormatter, MultipleLocator
from PIL import Image
from scipy.ndimage import gaussian_filter

from fig1_common import (DEM_PATH, TURBINE_JSON, MAST, apply_style, save,
                         FS_TICK, FS_LABEL, FS_LEGEND, FS_ANNOT,
                         AXES_LW, AXES_COLOR, PanelCtx)

PANEL_W, PANEL_H = 3.45, 3.37     # 本面板方框的英寸尺寸（单张出图即画布大小）
CTX = PanelCtx()

MAP_X, MAP_Y, MAP_W = 0.52, 0.46, 2.25          # 地图轴（英寸）
MAP_ASPECT = 1.2508                              # 由 set_aspect 锁死的高宽比
MAP_H = MAP_W * MAP_ASPECT

CONTOUR_INTERVAL = 20      # m
CONTOUR_SIGMA = 1.2        # 像素，等高线用
SHADE_SIGMA = 5.0          # 像素，晕渲用（≈150 m，压掉沙垄与冲沟纹理）
SHADE_BLEND = 0.35         # 晕渲混入纯色的权重（越小越平）
LIGHT_AZ, LIGHT_ALT = 315, 42

# 定位图：几何文件、范围、在地图内的位置（英寸，相对地图轴左下角）
BOUNDARY_JSON = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                             'cn_boundaries.json')
INSET_EXTENT = (72, 136, 3, 54.5)     # 含南海诸岛，纵横比约 0.92
INSET_ASPECT = 0.918

# 定位图放哪个角：
#   'll' 左下 —— 作者最初的要求，但那里是第 1、2 列机组和气象塔所在，
#                 透明叠加会让国界线和"Wind farm"压在机位点上。
#   'lr' 右下 —— 全图唯一没有机位的空白（第 4 列止于 40.190°N，
#                 第 3 列在 96.849°E 以西），透明叠加零冲突。
INSET_CORNER = 'lr'
INSET_PAD = 0.045                     # 距地图角的内边距

# 定位图的墨色与场址色。都压在地形色系里，不引入鲜艳色 ——
# 底图是低饱和的砂褐，纯红 #e02b1d 那种正色在上面会跳得很难看。
INK = '#33302a'                       # 国界/省界/文字统一用这一个墨色
SITE_COLOR = '#00a9e0'                # 与主图气象塔同色，不另起色相
# 定位图底：None = 全透明（地形的等高线与晕渲会透上来，线条容易搅在一起）；
#           给颜色则铺一层浅底，把定位图和地形分开。
INSET_BACKDROP = None
INSET_BACKDROP_ALPHA = 0.62
INSET_W = {'ll': 0.88, 'lr': 0.60}[INSET_CORNER]
INSET_H = INSET_W * INSET_ASPECT

# 单向递变：浅砂 -> 深褐，深 = 高。
# 不用双色调色带 —— 全场落差只有 133 m，双色会把平地画成有起伏。
TERRAIN_COLORS = ['#fdfaf4', '#f0e7d6', '#ddc9a9', '#c5a983',
                  '#a88a63', '#8a6d4c']


def read_dem(lon_min, lon_max, lat_min, lat_max):
    im = Image.open(DEM_PATH)
    px, py = im.tag_v2[33550][0], im.tag_v2[33550][1]
    tie = im.tag_v2[33922]
    origin_lon, origin_lat = tie[3], tie[4]

    arr = np.array(im).astype(float)
    arr[(arr < -500) | (arr > 9000)] = np.nan
    nrow, ncol = arr.shape

    c0 = max(int(np.floor((lon_min - origin_lon) / px)), 0)
    c1 = min(int(np.ceil((lon_max - origin_lon) / px)) + 1, ncol)
    r0 = max(int(np.floor((origin_lat - lat_max) / py)), 0)
    r1 = min(int(np.ceil((origin_lat - lat_min) / py)) + 1, nrow)

    sub = arr[r0:r1, c0:c1]
    ext = [origin_lon + c0 * px, origin_lon + c1 * px,
           origin_lat - r1 * py, origin_lat - r0 * py]
    return sub, ext


def locator_inset(x, y, w, h, site_lon, site_lat):
    """中国定位图：全透明，无底色无边框，只有线条。

    画的东西：国界外环、海疆线（十段线）、甘肃省界（淡填充 + 描边）、
    场址星标与文字。几何来自 cn_boundaries.json，见 cn_boundaries.py。
    """
    with open(BOUNDARY_JSON, encoding='utf-8') as fh:
        geo = json.load(fh)

    ax = CTX.ax(x, y, w, h, zorder=12)
    if INSET_BACKDROP is None:
        ax.set_facecolor('none')
        ax.patch.set_alpha(0)
    else:
        ax.set_facecolor(INSET_BACKDROP)
        ax.patch.set_alpha(INSET_BACKDROP_ALPHA)

    for ring in geo['china']:
        r = np.asarray(ring)
        ax.plot(r[:, 0], r[:, 1], '-', color=INK, linewidth=0.32, zorder=2)
    for seg in geo['maritime']:
        s = np.asarray(seg)
        ax.plot(s[:, 0], s[:, 1], '-', color=INK, linewidth=0.32, zorder=2)

    # 甘肃主要靠那层薄填充认出来，描边只比国界略重一点。
    # 之前描边给到 0.72 太壮，在 18 mm 的图里像一条粗黑带。
    for ring in geo['gansu']:
        r = np.asarray(ring)
        ax.fill(r[:, 0], r[:, 1], facecolor=INK, alpha=0.18,
                edgecolor='none', zorder=3)
        ax.plot(r[:, 0], r[:, 1], '-', color=INK, linewidth=0.42, zorder=4)

    # 场址点：不加文字、不加引线，含义写在图注里。
    # 颜色沿用主图气象塔的青色 —— panel A 里唯一的强调色就这一个，
    # 不再引入第三种色相。
    ax.scatter([site_lon], [site_lat], s=5.5, marker='o',
               facecolors=SITE_COLOR, edgecolors='none', zorder=5)

    lo, hi, la, ha = INSET_EXTENT
    ax.set_xlim(lo, hi)
    ax.set_ylim(la, ha)
    ax.set_aspect(1 / np.cos(np.deg2rad((la + ha) / 2)))
    ax.set_xticks([])
    ax.set_yticks([])
    for sp in ax.spines.values():
        sp.set_visible(False)
    return ax


def north_arrow(ax, x, y_tail, y_tip):
    """指北针：N 在下、实心箭头朝上。

    沿用玉门原图 plot_yumen_dem_all_turbines.py 的写法 ——
    annotate 把 "N" 放在 xytext（下方），箭头由文字指向 xy（上方）。
    位置改到左上角：那里在第 1 列机组（96.8151°E）以西、气象塔以北，是空的。
    经纬网已隐含真北，箭头即真北，不另标 grid / magnetic north。
    """
    ax.annotate('N', xy=(x, y_tip), xytext=(x, y_tail),
                ha='center', va='center', fontsize=FS_TICK, color='#222222',
                arrowprops=dict(arrowstyle='-|>', color='#222222', lw=0.9),
                zorder=10)


def scale_bar(ax, x_right, y0, lat_ref, km=1.0, n=2, height=0.0018):
    """黑白相间的分段比例尺，右端对齐到 x_right（含 "km" 两个字母）。

    样式对齐 季节预测-方案一/papers/paper1_seasonal_wind_assessment/
    fig1a_terrain.py 里的那条：交替色块 + 上方刻度数字 + 右侧 km。
    文字不加白描边 —— 之前被嫌弃的是描边，不是分段块本身，
    上一版把两者一起删成一条光秃秃的线，反而更难看。
    """
    seg = km / (111.32 * np.cos(np.deg2rad(lat_ref)))
    x0 = x_right - n * seg - 0.0135

    for i in range(n):
        ax.add_patch(Rectangle((x0 + i * seg, y0), seg, height,
                               facecolor='#1a1a1a' if i % 2 == 0 else '#ffffff',
                               edgecolor='#1a1a1a', linewidth=0.7, zorder=9))
    for i in range(n + 1):
        ax.text(x0 + i * seg, y0 + height + 0.0004, f'{int(i * km)}',
                ha='center', va='bottom', fontsize=FS_ANNOT, color='#1a1a1a', zorder=10)
    ax.text(x0 + n * seg + 0.0022, y0 + height + 0.0004, 'km',
            ha='left', va='bottom', fontsize=FS_ANNOT, color='#1a1a1a', zorder=10)


def draw(fig, x0=0.0, y0=0.0):
    CTX.bind(fig, x0, y0)

    with open(TURBINE_JSON, encoding='utf-8') as fh:
        turbines = json.load(fh)['turbines']
    lon = np.array([r['longitude'] for r in turbines])
    lat = np.array([r['latitude'] for r in turbines])
    print(f"  机位 {len(turbines)} 台 | "
          f"经度 {lon.min():.4f}-{lon.max():.4f} | 纬度 {lat.min():.4f}-{lat.max():.4f}")

    lon_min = min(lon.min(), MAST['lon']) - 0.006
    lon_max = max(lon.max(), MAST['lon']) + 0.006
    lat_min = min(lat.min(), MAST['lat']) - 0.005
    lat_max = max(lat.max(), MAST['lat']) + 0.005
    lat_ref = (lat_min + lat_max) / 2

    dem, ext = read_dem(lon_min, lon_max, lat_min, lat_max)
    valid = np.isfinite(dem)
    filled = np.where(valid, dem, np.nanmedian(dem))
    contour_dem = gaussian_filter(filled, sigma=CONTOUR_SIGMA)
    shade_dem = gaussian_filter(filled, sigma=SHADE_SIGMA)
    print(f"  DEM 窗口 {dem.shape} | 高程 {np.nanmin(dem):.0f}-{np.nanmax(dem):.0f} m")

    cmap = LinearSegmentedColormap.from_list('yumen_seq', TERRAIN_COLORS, N=256)
    vmin = 20 * np.floor(np.nanmin(dem) / 20)
    vmax = 20 * np.ceil(np.nanmax(dem) / 20)
    norm = mcolors.Normalize(vmin=vmin, vmax=vmax)

    # 晕渲只占 SHADE_BLEND 的权重，其余是纯色标，避免微地形被放大
    plain = cmap(norm(contour_dem))
    shaded = LightSource(azdeg=LIGHT_AZ, altdeg=LIGHT_ALT).shade(
        shade_dem, cmap=cmap, norm=norm, vert_exag=1.0, dx=24, dy=31,
        blend_mode='soft')
    rgb = SHADE_BLEND * shaded[..., :3] + (1 - SHADE_BLEND) * plain[..., :3]
    img = np.dstack([rgb, valid.astype(float)])

    ax = CTX.ax(MAP_X, MAP_Y, MAP_W, MAP_H)
    ax.set_facecolor('#f4f2ec')
    ax.imshow(img, extent=ext, origin='upper', interpolation='bilinear', zorder=1)

    levels = np.arange(20 * np.ceil(vmin / 20), 20 * np.floor(vmax / 20) + 1,
                       CONTOUR_INTERVAL)
    ax.contour(np.linspace(ext[0], ext[1], dem.shape[1]),
               np.linspace(ext[3], ext[2], dem.shape[0]),
               contour_dem, levels=levels, colors='#4a4034',
               linewidths=0.28, alpha=0.32, zorder=2)

    ax.scatter(lon, lat, s=5, marker='o', facecolors='#ffffff',
               edgecolors='#2b3a42', linewidths=0.35, zorder=6)
    ax.scatter([MAST['lon']], [MAST['lat']], s=46, marker='^',
               facecolors='#00a9e0', edgecolors='none', zorder=8)

    ax.set_xlim(lon_min, lon_max)
    ax.set_ylim(lat_min, lat_max)
    ax.set_aspect(1 / np.cos(np.deg2rad(lat_ref)))
    ax.xaxis.set_major_locator(MultipleLocator(0.02))
    ax.yaxis.set_major_locator(MultipleLocator(0.02))
    ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _: f'{v:.2f}°E'))
    ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f'{v:.2f}°N'))
    ax.tick_params(direction='out', length=2.6, width=AXES_LW, labelsize=FS_TICK, pad=2)
    ax.grid(color='#4a4034', linewidth=0.3, linestyle=':', alpha=0.22, zorder=0)

    # 定位图：全透明叠在地形上。右下角时比例尺压到它下面，左下角时贴右边框。
    if INSET_CORNER == 'lr':
        ins_x = MAP_X + MAP_W - INSET_W - INSET_PAD
        # 右下空白只有 lat 40.1627-40.1899（第 4 列最低机位）共 0.877 in，
        # 自下而上分配：比例尺 0.16 + 间隔 0.06 + 定位图 0.606 + 顶部余量 0.05
        ins_y = MAP_Y + 0.26
        bar_right = lon_max - 0.0035
        bar_y = lat_min + 0.0030
    else:
        ins_x = MAP_X + INSET_PAD
        ins_y = MAP_Y + INSET_PAD
        bar_right = lon_max - 0.0035
        bar_y = lat_min + 0.0038

    scale_bar(ax, x_right=bar_right, y0=bar_y, lat_ref=lat_ref, km=1.0, n=2)
    north_arrow(ax, x=lon_min + 0.0045,
                y_tail=lat_max - 0.0175, y_tip=lat_max - 0.0072)
    locator_inset(ins_x, ins_y, INSET_W, INSET_H, MAST['lon'], MAST['lat'])

    cax = CTX.ax(MAP_X + MAP_W + 0.08, 0.60, 0.10, 2.32)
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cb = fig.colorbar(sm, cax=cax, orientation='vertical')
    cb.set_ticks(np.arange(1580, int(vmax) + 1, 40))
    cb.set_label('Elevation (m a.s.l.)', fontsize=FS_LABEL, labelpad=3)
    cb.ax.tick_params(labelsize=FS_TICK, length=2.2, width=AXES_LW,
                      color=AXES_COLOR, pad=1.5)
    cb.outline.set_linewidth(AXES_LW)
    cb.outline.set_edgecolor(AXES_COLOR)

    # 图例在图框外、紧贴地图下方，单行两项。
    # 等高线间隔那一项挪进图注：三项排一行会比地图本身还宽，
    # 折成两行又会在地图和图例之间空出一大块。
    handles = [
        Line2D([], [], marker='o', linestyle='none', markersize=2.9,
               markerfacecolor='#ffffff', markeredgecolor='#2b3a42',
               markeredgewidth=0.45, label=f'Wind turbines ({len(turbines)})'),
        Line2D([], [], marker='^', linestyle='none', markersize=5.0,
               markerfacecolor='#00a9e0', markeredgecolor='none',
               label='Meteorological mast'),
    ]
    fig.legend(handles=handles, loc='lower center',
               bbox_to_anchor=CTX.pt(MAP_X + MAP_W / 2, 0.05), ncol=2,
               frameon=False, fontsize=FS_LEGEND, handletextpad=0.45,
               columnspacing=1.3, borderaxespad=0)

    return ax


def main():
    apply_style()
    fig = plt.figure(figsize=(PANEL_W, PANEL_H), facecolor='white')
    draw(fig)
    save(fig, 'figure-1a_site_dem')


if __name__ == '__main__':
    main()
