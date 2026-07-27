#!/usr/bin/env python3
"""
cn_boundaries.py — 从 cnmaps-data 导出中国国界、甘肃省界、海疆线（十段线）。

**这个脚本要用另一个环境跑**，不是 windforecast：

    /Users/xiaxin/Desktop/季节预测-方案一/.venv/bin/python cn_boundaries.py

因为 cnmaps / cnmaps-data 只装在那个项目的 venv 里（Python 3.12）。
导出的 cn_boundaries.json 落在本目录，之后 figure-1-a_dem.py 只读 json，
不再依赖那个 venv，也不需要 cartopy / geopandas。

导出内容
--------
china     国界外环（简化 tol=0.06°，丢掉 <0.02 平方度的碎岛，
          台湾、海南、南海主要岛礁保留）
gansu     甘肃省界外环（简化 tol=0.03°）
maritime  海疆线，即十段线（原样，10 段 199 点，
          来自 cnmaps_data/.../amap/maritime/100000.geojson）

数据来源：cnmaps-data 1.1.2，manifest 里 provider 标为 "official"。
合规提醒见 figure-1-a_dem.py 的 docstring。
"""

import json
import os

import numpy as np
from cnmaps import get_adm_maps, BASE_DATA_DIR

OUT_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                        'cn_boundaries.json')


def export_polygon(poly, tol, min_area):
    """外环 -> 简化后的坐标环，丢掉小于 min_area 平方度的碎岛。"""
    geoms = poly.geoms if hasattr(poly, 'geoms') else [poly]
    rings = []
    for g in geoms:
        if g.area < min_area:
            continue
        s = g.simplify(tol, preserve_topology=True)
        r = np.asarray(s.exterior.coords)
        if len(r) >= 4:
            rings.append(r)
    return rings


def load_maritime():
    path = os.path.join(str(BASE_DATA_DIR), 'amap', 'maritime', '100000.geojson')
    gj = json.load(open(path))
    segs = []

    def walk(c):
        if isinstance(c[0][0], (int, float)):
            segs.append(np.asarray(c, dtype=float))
        else:
            for x in c:
                walk(x)

    walk(gj['geometry']['coordinates'])
    return segs


def main():
    china = get_adm_maps(level='国', only_polygon=True, record='first')
    gansu = get_adm_maps(province='甘肃省', level='省', only_polygon=True, record='first')

    cn = export_polygon(china, tol=0.06, min_area=0.02)
    gs = export_polygon(gansu, tol=0.03, min_area=0.01)
    mar = load_maritime()

    print(f'  国界  {len(cn):>3} 环, {sum(len(r) for r in cn):>5} 点')
    print(f'  甘肃  {len(gs):>3} 环, {sum(len(r) for r in gs):>5} 点')
    print(f'  海疆  {len(mar):>3} 段, {sum(len(r) for r in mar):>5} 点')

    json.dump({'china': [r.round(3).tolist() for r in cn],
               'gansu': [r.round(3).tolist() for r in gs],
               'maritime': [r.round(3).tolist() for r in mar],
               'source': 'cnmaps-data 1.1.2 (provider=official)'},
              open(OUT_PATH, 'w'))
    print(f'  ✓ {OUT_PATH} ({os.path.getsize(OUT_PATH) // 1024} KB)')


if __name__ == '__main__':
    main()
