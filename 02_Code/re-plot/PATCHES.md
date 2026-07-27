# Patches: 4-height screening  ->  70 m-only sectoring

Put `qc_common.py` in the same folder as the plotting scripts.

---

## 1. `figure-3-a.py`  (HH / SR / ER on full dataset)

**No sector mask exists in this script — only QC changes.**

### 1.1 Add import (after `from pathlib import Path`)
```python
from qc_common import qc_operating_range
```

### 1.2 Replace the whole cleaning block inside `load_and_clean_data`
Replace from `# 基础清理` down to `print(f"风速筛选后 ...")` with:
```python
        data_clean = qc_operating_range(data)          # >=0 power, no NaN, 3-25 m/s @70m, month exclusion
        data_clean = data_clean.dropna(subset=required_cols)
        print(f"最终样本: {data_clean.shape}")
```

---

## 2. `figure-3-b.py`  (regime-resolved overlay + DM insets)

### 2.1 Add import
```python
from qc_common import qc_operating_range, sector_mask, SECTOR_FREE, SECTOR_WAKE, report_sectors
```

### 2.2 DELETE the whole method `strict_direction_mask` (lines ~175-188)

### 2.3 In `__init__`, replace
```python
        self.wind_dir_columns = [
            'obs_wind_direction_10m',
            'obs_wind_direction_30m',
            'obs_wind_direction_50m',
            'obs_wind_direction_70m'
        ]
```
with
```python
        # 70 m is the only clean vane and is also what the operational WDA
        # strategy uses -> single sector-defining variable.
        self.wind_dir_columns = ['obs_wind_direction_70m']
```

### 2.4 In `load_and_classify_data`, replace the cleaning + masking block
```python
        # 基础清理
        data_clean = data[data['power'] >= 0].copy()
        data_clean = data_clean.dropna(subset=required_cols)
        wind_speed_condition = (...)
        data_clean = data_clean[wind_speed_condition].copy()
        mask_free = self.strict_direction_mask(data_clean, (225, 315))
        mask_wake = self.strict_direction_mask(data_clean, (45, 135))
```
with
```python
        data_clean = qc_operating_range(data)
        data_clean = data_clean.dropna(subset=required_cols)
        mask_free = sector_mask(data_clean, SECTOR_FREE)
        mask_wake = sector_mask(data_clean, SECTOR_WAKE)
        report_sectors(data_clean, 'reconstruction')
```

---

## 3. `figure-3-c.py`  (SHAP, sector-specific models)

### 3.1 Add import
```python
from qc_common import (qc_operating_range, sector_mask,
                       SECTOR_FREE, SECTOR_WAKE, WD_BAD_COLS, report_sectors)
```

### 3.2 In `__init__`, replace
```python
        self.wind_dir_columns = ['obs_wind_direction_10m', 'obs_wind_direction_30m',
                                 'obs_wind_direction_50m', 'obs_wind_direction_70m']
```
with
```python
        self.wind_dir_columns = ['obs_wind_direction_70m']
```

### 3.3 DELETE the whole method `strict_direction_mask`

### 3.4 In `prepare_features`, drop the unusable vanes.
After the line
```python
        feature_columns = [col for col in obs_columns if col not in ['datetime', 'power']]
```
insert
```python
        # 10 m vane: quantised to 16 bins + ~30% pinned at 0/359 deg.
        # 30 m vane: +12.2 deg offset, 2021-11 failure.  50 m vane: -6.5 deg offset.
        # Only the 70 m vane is retained, matching the operational WDA sectoring.
        feature_columns = [c for c in feature_columns if c not in WD_BAD_COLS]
```

### 3.5 In `train_sector_specific_test_only`, replace
```python
        df = df[df['power'] >= 0].dropna(subset=self.wind_dir_columns + ['power'])
        df = df[(df['obs_wind_speed_70m'] >= 3.0) & (df['obs_wind_speed_70m'] <= 25.0)]
        ...
        mask_free = self.strict_direction_mask(df, (225, 315))
        mask_wake = self.strict_direction_mask(df, (45, 135))
```
with
```python
        df = qc_operating_range(df).dropna(subset=self.wind_dir_columns)
        mask_free = sector_mask(df, SECTOR_FREE)
        mask_wake = sector_mask(df, SECTOR_WAKE)
        report_sectors(df, 'SHAP')
```

### 3.6 Reproducibility: in `_plot_beeswarm_improved`, before the loop add
```python
        rng = np.random.default_rng(42)
```
and replace `np.random.normal(0, 0.08, len(shap_vals))`
with `rng.normal(0, 0.08, len(shap_vals))`

---

## 4. `figure-4-abcd.py`  (day-ahead forecast, WDA)

**Already sectors on `{nwp}_wind_direction_70m` — no mask change needed.**

### 4.1 Add import
```python
from qc_common import apply_month_exclusion
```

### 4.2 In `load_and_split_data`, right after the 3-25 m/s filter block, add
```python
        df = apply_month_exclusion(df)
```
(placed BEFORE the chronological 80/20 split, so the split boundary is
recomputed on the retained record.)

### 4.3 WDA sector inputs — DO NOT EDIT YET.
`wda_config` currently hard-codes, per sector:
  free  -> 10/50/70 m ;  wake -> 10/30/70 m ;  other -> 10/30/50/70 m
These came from the OLD SHAP ranking, which was computed on the
4-height-screened subset and included the corrupted WD features.
Re-derive from the new `figure-3-c` output, then update.

---

## 5. `figure-1-{a,b,c,d}_*.py`  (Figure 1 重画，2026-07-26)

Figure 1 拆成四个独立脚本，各出各的图，由作者在版面软件里拼装：

| 脚本 | 面板 | 输出 |
|---|---|---|
| `figure-1-a_dem.py` | 场址地形 + 134 机位 + 气象塔 + 中国定位图 | `figure-1a_site_dem` |
| `figure-1-b_windrose.py` | 70 m 风玫瑰 | `figure-1b_windrose_70m` |
| `figure-1-c_corr.py` | 风速–总功率相关廓线 | `figure-1c_correlation_profile` |
| `figure-1-d_deficit.py` | 分扇区廓线 + 风速亏损 | `figure-1d_sector_profiles_deficit` |

共用 `fig1_common.py`（样本口径、配色、字号、输出目录）。四个面板的样本
口径统一为 IMF 对齐样本 + 70m 单高度扇区判据 + flatline 剔除
（all = 47,448 / free = 19,543 / wake = 22,420）。

旧脚本 `figure-1-a.py`、`figure-1-b_fixed.py`、`figure-1-c_fixed.py` 由这四个取代。
`03_Results/figures/风电场布设位置示意图/位置示意图.ai` 已作废
（33 台 4 列，与正文的 134 台 5 列对不上）。

### 5.1 拼版硬约束
**panel A 不要窄于 88 mm。** 机位列内中位间距 247 m，按 2.25 in 的地图宽度
换算圆心间距 1.82 mm、点直径 0.89 mm，只剩 0.93 mm 空隙；再缩第一列就糊成一条线。

### 5.2 `cn_boundaries.py` 要用另一个环境跑
定位图的国界 / 甘肃省界 / 十段线来自 **cnmaps**，只装在
`/Users/xiaxin/Desktop/季节预测-方案一/.venv`（Python 3.12）：

```bash
/Users/xiaxin/Desktop/季节预测-方案一/.venv/bin/python cn_boundaries.py
```

产物 `cn_boundaries.json` 已随脚本入库，出图时只读它、不依赖那个 venv。
除非要改简化容差或范围，平时不用重跑。

**不要试图用 cartopy / basemap 替代**：两者在 windforecast 里 import 就崩
（pyproj -> libproj.22 -> 缺 libtiff.5.dylib）；且 basemap 的 countries 是弧段
不是闭合多边形（polygonize 出的"中国"横跨欧亚大陆，填不了色），
states 数据在中国境内 0 段。这两条路 2026-07-26 都撞死过。

### 5.3 图注必须交代（图上表达不了）
- 灰细线是 20 m 间隔等高线；
- 晕渲为视觉增强（315° / 42°，算在 sigma = 5.0 像素的平滑地形上、0.35 权重混入），
  不携带定量信息 —— 不写清楚会和 Text S1 的"残余起伏 1.91 m RMS"对不上；
- 定位图里青点 = 场址、灰块 = 甘肃。
