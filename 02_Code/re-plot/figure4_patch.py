# ============================================================================
# PATCH for figure-4-abcd.py  (2026-07-18)
# 三处修改。建议将 diebold_mariano_test 放入 qc_common.py，
# figure-3-b.py 与 figure-4-abcd.py 共同 import，保证两图检验实现一致。
# 若你 figure-3-b.py 中的修正版与此处有出入，以 figure-3-b.py 为准。
# ============================================================================

import numpy as np
from scipy import stats


# ----------------------------------------------------------------------------
# 修改 1/3：替换原 diebold_mariano_test（原版 h=1 使 lag 循环为空，
# 方差退化为 gamma_0，且用正态分布 p 值）
# 签名与调用方式不变：输入原始误差，内部平方。
# ----------------------------------------------------------------------------
def diebold_mariano_test(errors1, errors2):
    """Diebold-Mariano test on squared errors.

    Variance of the loss differential: Bartlett-kernel HAC with
    Newey-West (1994) automatic lag selection.
    Small-sample: Harvey-Leybourne-Newbold correction.
    p-value: two-sided, t distribution with n-1 dof.
    """
    d = np.asarray(errors1, dtype=float) ** 2 - np.asarray(errors2, dtype=float) ** 2
    d = d[~np.isnan(d)]
    n = len(d)
    if n < 10:
        return np.nan, np.nan

    mean_d = d.mean()
    dc = d - mean_d

    # Newey-West automatic bandwidth (rule of thumb)
    L = int(np.floor(4.0 * (n / 100.0) ** (2.0 / 9.0)))
    L = max(L, 0)

    gamma0 = np.dot(dc, dc) / n
    var = gamma0
    for lag in range(1, L + 1):
        gamma = np.dot(dc[:-lag], dc[lag:]) / n
        var += 2.0 * (1.0 - lag / (L + 1.0)) * gamma  # Bartlett weights
    var = max(var, 1e-12)  # guard against non-positive HAC variance

    dm = mean_d / np.sqrt(var / n)

    # Harvey-Leybourne-Newbold correction; horizon h = L + 1 aligns the
    # correction with the HAC bandwidth (same-timestamp setting, no true
    # forecast horizon; autocorrelation stems from persistence of WRF error).
    h = L + 1
    hln = np.sqrt((n + 1 - 2 * h + h * (h - 1) / n) / n)
    dm_hln = dm * hln

    p_value = 2.0 * (1.0 - stats.t.cdf(abs(dm_hln), df=n - 1))
    return dm_hln, p_value


# ----------------------------------------------------------------------------
# 修改 2/3：_build_wda_models 增加 'manuscript' 模式（与定稿正文 2.4 一致）
# 顶部开关改为三选一：WDA_MODE = 'manuscript' | 'full' | 'reduced'
# 'full' 保留：与 ER 同输入，隔离 sectoring 效应，作消融进 SI/response。
# ----------------------------------------------------------------------------
@staticmethod
def _build_wda_models(nwp, mode):
    ws = lambda h: f'{nwp}_wind_speed_{h}m'
    T = [f'{nwp}_temperature_10m']
    if mode == 'manuscript':
        # SHAP-informed sector-specific inputs (frozen main text 2.4):
        # free: 10/50/70 (SHAP: WS50 > WS30 in free-stream sector)
        # wake: 10/30/70 (SHAP: WS30 > WS50 in wake sector)
        free = [ws(10), ws(50), ws(70)]
        wake = [ws(10), ws(30), ws(70)]
    elif mode == 'reduced':
        free = wake = [ws(10), ws(70)]
    elif mode == 'full':
        free = wake = [ws(10), ws(30), ws(50), ws(70)]
    else:
        raise ValueError(mode)
    others = [ws(10), ws(30), ws(50), ws(70)]
    return {'free':   {'wind_features': free,   'other_features': T},
            'wake':   {'wind_features': wake,   'other_features': T},
            'others': {'wind_features': others, 'other_features': T}}


# ----------------------------------------------------------------------------
# 修改 3/3：load_and_split_data 中，train_test_split(shuffle=False) 之前
# 显式按时间排序（时序切分不应依赖 CSV 恰好有序）：
#
#     df = df.sort_values('datetime').reset_index(drop=True)
#     train_df, test_df = train_test_split(
#         df, test_size=0.2, random_state=42, shuffle=False)
# ----------------------------------------------------------------------------
