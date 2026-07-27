# Figure 3 — 文件说明

最后整理：2026-07-25（回应审稿意见 R2 Point 18：把散点图改为误差分布 PDF 曲线）

---

## 正文 Figure 3 实际使用的文件

拼版文件是 `figure3.ai`，导出 `figure3.pdf` / `figure3.png`。
三个面板的来源如下（**拼版请置入 .pdf 而非 .png**，理由见文末）：

| 面板 | 内容 | 源文件 | 生成脚本 |
|---|---|---|---|
| A | 总体误差分布 PDF（HH/SR/ER 三条曲线） | `final_3_a_pdf_nerr_changma_matched.pdf` | `02_Code/re-plot/figure-3-b_fixed.py` |
| B | 分扇区误差分布 PDF（free-stream / wake） | `final_3_b_pdf_nerr_changma_matched.pdf` | `02_Code/re-plot/figure-3-b_fixed.py` |
| C | 分扇区 SHAP 蜂群图 | `final_3_c_all_features_sector_test.pdf` | `02_Code/re-plot/figure-3-c_fixed.py` |

注意 Panel A 也由 `figure-3-b_fixed.py` 生成（不是 `figure-3-a_fixed.py`）。
A 与 B 必须来自同一批已训练模型才能同口径，而那批模型在 3-b 里。
`figure-3-a_fixed.py` 只产出被替代的旧版散点/KDE 图，其文件头有说明。

横轴是归一化误差 `(P_mod − P_obs)/P_rated`，单位 % 装机（P_rated = 193 MW）。

---

## SI / 回复信使用的文件

| 文件 | 用途 |
|---|---|
| `final_3_a_pdf_ratio_changma_matched.{png,pdf}` | 审稿人字面建议的比值 `P_mod/P_obs` 版本，总体 |
| `final_3_b_pdf_ratio_changma_matched.{png,pdf}` | 同上，分扇区 |

比值版对分母设了下限 `P_obs ≥ 5%` 装机。不设下限时测试集有 23 条
`P_obs = 0`，其余样本的比值 p99 = 30、p99.9 = 215、最大 7485，
PDF 会被一条纯由近零观测功率造成的尾部主导。回复信用这两张图说明
"已按建议改画"，正文用 nerr 版。

---

## 数据表

| 文件 | 内容 |
|---|---|
| `data_fig3_pdf_metrics_changma_matched.csv` | A/B 全部面板的 R²、RMSE、中位数、IQR（含 ratio 与 nerr 两种度量） |
| `data_panel_a_performance_summary_changma_matched.csv` | 旧版 Panel A 的指标 |
| `data_panel_b_sector_summary_changma_matched.csv` | 旧版 Panel B 分扇区指标 |
| `data_panel_b_significance_tests_changma_matched.csv` | 旧版 Panel B 的显著性检验 |
| `panel_a_performance_summary_changma_matched.csv` | 更旧的一份 Panel A 指标 |
| `data_comparison_summary.csv` / `dataset_comparison_summary.csv` | 数据集对比 |

---

## `_superseded/` — 已被替代，保留备查，勿用于投稿

| 文件 | 说明 |
|---|---|
| `final_3_a_kde_changma_matched.*` | 旧 Panel A：三个并排 KDE 散点图，被 A 的 PDF 曲线版替代 |
| `final_3_b_changma_matched.*` | 更旧的 Panel B，用的是 in-sample 评估（在训练数据上评估，R² 偏高、DM 检验无意义） |
| `final_3_b_changma_matched_heldout.*` | 修正为 held-out 评估后的散点叠加版。数值口径正确，但三套配置的散点完全重合、分辨不出差别，正是 R2 Point 18 要求改掉的图 |
| `备用scatter_overlay_with_bars_kde_changma_matched.*` | 早期备用版 |
| `final_3_ab_row_pdf_*` | A+B 合并成一行三面板的**备选排版**（本身没错，只是最终未采用；如果以后想压缩版面可以回来用） |

`final_3_b_changma_matched_heldout.*` 每次运行 `figure-3-b_fixed.py` 都会在顶层
重新生成（`create_scatter_overlay()` 顺带产出，其返回值用于 A/B 的 inset）。
重跑后如果顶层又出现这个文件，直接再移进 `_superseded/` 即可，它不进正文。

---

## 为什么拼版要用 .pdf 而不是 .png

PNG 的像素尺寸很大（Panel C 是 14967 px 宽）。Illustrator 置入位图按 72 ppi
解释，三行堆叠后画板约 208 × 202 英寸，超过 PDF 页面的 200 × 200 英寸上限，
存 PDF 时会报"该页的尺寸超出范围"。同名 .pdf 是矢量图，页面尺寸
8.9 / 18.4 / 24.9 英寸，堆叠后约 25 × 24 英寸，远低于上限，且文字可无损缩放。

三个面板的字号是按"缩放到同宽后彼此等效"调好的（Panel C 的字号 =
Panel B 的字号 × 1.36，该比值来自实测导出宽度 14967 / 11016）。
**前提是拼版时三行缩放到相同宽度**；若改变相对宽度，字号需重算。
