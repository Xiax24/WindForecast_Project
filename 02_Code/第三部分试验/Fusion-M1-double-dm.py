#!/usr/bin/env python3
"""
DM (Diebold-Mariano) Test for Both Fusion-M1 and Fusion-M2
同时检验Fusion-M1和Fusion-M2是否显著优于其他所有预测模型
"""

import os
import numpy as np
import pandas as pd
from scipy import stats
import json

# ----------------------------
# DM检验所需函数
# ----------------------------
def _loss_vector(e, loss="mse", power=None):
    """计算损失向量"""
    if callable(loss):
        return np.asarray(loss(e), dtype=float)
    if loss == "mse":
        return e ** 2.0
    if loss == "mae":
        return np.abs(e)
    if loss == "power":
        if power is None or power <= 0:
            raise ValueError("When loss='power', provide a positive `power`.")
        return np.abs(e) ** float(power)
    raise ValueError(f"Unknown loss '{loss}'")

def _newey_west_long_run_variance(x, lag=None):
    """Newey-West长期方差估计"""
    x = np.asarray(x, dtype=float)
    n = x.size
    if n < 2:
        return np.nan
    x = x - np.nanmean(x)
    # 自动滞后（若未指定）
    if lag is None:
        lag = int(np.floor(4.0 * (n / 100.0) ** (2.0 / 9.0)))
    lag = max(0, min(lag, n - 1))
    gamma0 = np.dot(x, x) / n
    s0 = gamma0
    for k in range(1, lag + 1):
        gamma_k = np.dot(x[k:], x[:-k]) / n
        weight = 1.0 - k / (lag + 1.0)  # Bartlett权重
        s0 += 2.0 * weight * gamma_k
    return s0

def _hln_correction(n, h):
    """Harvey-Leybourne-Newbold小样本修正"""
    n = float(n); h = float(h)
    return np.sqrt((n + 1.0 - 2.0 * h + (h * (h - 1.0)) / n) / n)

def dm_test(e1, e2, h=1, loss="mse", power=None,
            alternative="two_sided", small_sample="HLN", hac_lag="h-1"):
    """
    Diebold-Mariano检验
    
    H0: E[d_t]=0, d_t = L(e1_t) - L(e2_t)
    这里e1对应模型1, e2对应模型2
    """
    e1 = np.asarray(e1, dtype=float)
    e2 = np.asarray(e2, dtype=float)
    if e1.shape != e2.shape:
        raise ValueError("e1和e2长度必须一致。")

    mask = np.isfinite(e1) & np.isfinite(e2)
    e1 = e1[mask]; e2 = e2[mask]
    n = e1.size
    if n < 3:
        raise ValueError("有效样本不足。")

    L1 = _loss_vector(e1, loss=loss, power=power)
    L2 = _loss_vector(e2, loss=loss, power=power)
    d = L1 - L2
    d_bar = float(np.mean(d))

    # Newey-West 滞后
    if hac_lag == "h-1":
        lag = max(int(h) - 1, 0)
    elif hac_lag is None:
        lag = None
    else:
        lag = int(hac_lag)

    s0 = _newey_west_long_run_variance(d, lag=lag)
    if not np.isfinite(s0) or s0 <= 0:
        s0 = np.var(d, ddof=1)
    var_dbar = s0 / n
    se_dbar = float(np.sqrt(var_dbar)) if var_dbar > 0 else np.nan
    if not np.isfinite(se_dbar) or se_dbar == 0:
        raise ValueError("标准误为0或NaN，请检查输入。")

    dm_stat = d_bar / se_dbar

    if small_sample == "HLN":
        scale = _hln_correction(n, h)
        dm_stat *= scale
        df = max(n - 1, 1)
        if alternative == "two_sided":
            p = 2.0 * (1.0 - stats.t.cdf(abs(dm_stat), df=df))
        elif alternative == "greater":
            p = 1.0 - stats.t.cdf(dm_stat, df=df)
        elif alternative == "less":
            p = stats.t.cdf(dm_stat, df=df)
        else:
            raise ValueError("alternative取值错误")
        method = "DM-HLN"
    else:
        if alternative == "two_sided":
            p = 2.0 * (1.0 - stats.norm.cdf(abs(dm_stat)))
        elif alternative == "greater":
            p = 1.0 - stats.norm.cdf(dm_stat)
        elif alternative == "less":
            p = stats.norm.cdf(dm_stat)
        else:
            raise ValueError("alternative取值错误")
        method = "DM-asymptotic"

    return {
        "statistic": float(dm_stat),
        "p_value": float(p),
        "d_bar": float(d_bar),
        "se_dbar": float(se_dbar),
        "n": int(n),
        "lag_used": int(lag) if lag is not None else None,
        "method": method,
        "alternative": alternative,
        "h": int(h),
        "mean_L1": float(np.mean(L1)),
        "mean_L2": float(np.mean(L2)),
    }

def errors_from(y_true, y_pred):
    """计算误差序列"""
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    if y_true.shape != y_pred.shape:
        raise ValueError("y_true与y_pred长度必须一致。")
    return y_true - y_pred

def dm_from_predictions(y_true, yhat1, yhat2, h=1, loss="mse", power=None,
                        alternative="two_sided", small_sample="HLN", hac_lag="h-1"):
    """从预测值计算DM检验"""
    e1 = errors_from(y_true, yhat1)
    e2 = errors_from(y_true, yhat2)
    return dm_test(e1, e2, h=h, loss=loss, power=power,
                   alternative=alternative, small_sample=small_sample, hac_lag=hac_lag)

def stars(p):
    """显著性星号标记"""
    return "***" if p <= 0.001 else ("**" if p <= 0.01 else ("*" if p <= 0.05 else ""))

# ----------------------------
# 加载所有模型的预测结果
# ----------------------------
def load_all_model_predictions(results_dir):
    """
    加载所有模型的预测结果
    
    Returns:
        dict: {model_name: DataFrame}
    """
    
    print(f"📂 加载模型预测结果...")
    
    all_models = {}
    
    # 定义所有基础模型名称（不包括Fusion-M1和Fusion-M2）
    base_model_names = [
        'G-M1-10m', 'G-M1-70m', 'E-M1-10m', 'E-M1-70m',
        'G-M2-10m', 'G-M2-70m', 'E-M2-10m', 'E-M2-70m',
        'G-M4-Dual', 'E-M4-Dual',
        'X-M2-10m', 'X-M2-70m',
        'X-M3-10m', 'X-M3-70m'
    ]
    
    # 加载每个基础模型的结果
    for model_name in base_model_names:
        model_dir = os.path.join(results_dir, model_name)
        result_file = os.path.join(model_dir, f'{model_name}_detailed_results.csv')
        
        if os.path.exists(result_file):
            df = pd.read_csv(result_file)
            df['datetime'] = pd.to_datetime(df['datetime'])
            all_models[model_name] = df
            print(f"   ✅ {model_name}: {len(df)} samples")
        else:
            print(f"   ⚠️ {model_name}: 结果文件不存在")
    
    # 加载Fusion-M1
    fusion_m1_dir = os.path.join(results_dir, 'Fusion-M1')
    fusion_m1_file = os.path.join(fusion_m1_dir, 'Fusion-M1_detailed_results.csv')
    
    if os.path.exists(fusion_m1_file):
        df = pd.read_csv(fusion_m1_file)
        df['datetime'] = pd.to_datetime(df['datetime'])
        all_models['Fusion-M1'] = df
        print(f"   ✅ Fusion-M1: {len(df)} samples")
    else:
        print(f"   ⚠️ Fusion-M1: 结果文件不存在")
    
    # 加载Fusion-M2（基于风向分类的模型）
    fusion_m2_dir = os.path.join(os.path.dirname(results_dir), 'wind_direction_based_fusion')
    fusion_m2_file = os.path.join(fusion_m2_dir, 'detailed_results.csv')
    
    if os.path.exists(fusion_m2_file):
        df = pd.read_csv(fusion_m2_file)
        df['datetime'] = pd.to_datetime(df['datetime'])
        all_models['Fusion-M2'] = df
        print(f"   ✅ Fusion-M2: {len(df)} samples")
    else:
        print(f"   ⚠️ Fusion-M2: 结果文件不存在")
    
    print(f"\n   共加载 {len(all_models)} 个模型的预测结果")
    
    return all_models

def calculate_time_step(df):
    """计算时间步长（小时）"""
    if len(df) < 2:
        return 1  # 默认1小时
    
    time_diff = df['datetime'].diff().dropna().median()
    hours = time_diff.total_seconds() / 3600
    return max(int(round(hours)), 1)

def run_dm_tests_for_target(all_models, target_model_name, loss_kind="mse", 
                             horizon_hours=24, small_sample="HLN", hac_lag_mode="h-1"):
    """
    对指定目标模型运行DM检验
    
    Args:
        all_models: dict of DataFrames
        target_model_name: 目标模型名称
        loss_kind: 损失函数类型
        horizon_hours: 预测时长（小时）
        small_sample: 小样本修正方法
        hac_lag_mode: HAC滞后模式
    
    Returns:
        DataFrame: DM检验结果
    """
    
    print(f"\n{'='*100}")
    print(f"📊 执行DM检验: {target_model_name}")
    print(f"{'='*100}")
    print(f"   损失函数: {loss_kind}")
    print(f"   预测时长: {horizon_hours}小时")
    
    # 获取目标模型数据
    if target_model_name not in all_models:
        raise ValueError(f"目标模型 {target_model_name} 未找到")
    
    target_df = all_models[target_model_name]
    
    # 计算时间步长和h
    time_step_hours = calculate_time_step(target_df)
    h_steps = max(int(round(horizon_hours / time_step_hours)), 1)
    
    print(f"   时间步长: {time_step_hours}小时")
    print(f"   h参数: {h_steps}步")
    
    # 合并目标模型和其他模型的数据
    results = []
    
    for other_model in sorted(all_models.keys()):
        if other_model == target_model_name:
            continue
        
        print(f"\n   对比 {target_model_name} vs {other_model}...")
        
        other_df = all_models[other_model]
        
        # 按datetime合并
        merged = pd.merge(
            target_df[['datetime', 'actual_power', 'predicted_power']],
            other_df[['datetime', 'predicted_power']],
            on='datetime',
            how='inner',
            suffixes=('_target', '_other')
        )
        
        if len(merged) < 10:
            print(f"      ⚠️ 共同样本数不足 ({len(merged)}), 跳过")
            continue
        
        print(f"      共同样本数: {len(merged)}")
        
        # 提取数据
        y_true = merged['actual_power'].values
        y_target = merged['predicted_power_target'].values
        y_other = merged['predicted_power_other'].values
        
        # 两侧检验
        try:
            res_two = dm_from_predictions(
                y_true, y_target, y_other, h=h_steps, loss=loss_kind,
                alternative="two_sided", small_sample=small_sample, hac_lag=hac_lag_mode
            )
            
            # 单侧：目标模型更好 => L_target - L_other < 0
            res_less = dm_from_predictions(
                y_true, y_target, y_other, h=h_steps, loss=loss_kind,
                alternative="less", small_sample=small_sample, hac_lag=hac_lag_mode
            )
            
            # 单侧：目标模型更差 => L_target - L_other > 0
            res_great = dm_from_predictions(
                y_true, y_target, y_other, h=h_steps, loss=loss_kind,
                alternative="greater", small_sample=small_sample, hac_lag=hac_lag_mode
            )
            
            # 判断哪个模型更好
            direction = f"{target_model_name} < {other_model}" if res_two["d_bar"] < 0 else f"{target_model_name} > {other_model}"
            better = target_model_name if res_two["d_bar"] < 0 else other_model
            
            # 计算RMSE和MAE用于对比
            rmse_target = np.sqrt(res_two["mean_L1"]) if loss_kind == "mse" else np.nan
            rmse_other = np.sqrt(res_two["mean_L2"]) if loss_kind == "mse" else np.nan
            mae_target = res_two["mean_L1"] if loss_kind == "mae" else np.nan
            mae_other = res_two["mean_L2"] if loss_kind == "mae" else np.nan
            
            results.append({
                "target_model": target_model_name,
                "other_model": other_model,
                "comparison": f"{target_model_name} vs {other_model}",
                "h_steps": h_steps,
                "loss": loss_kind,
                "dm_statistic": round(res_two["statistic"], 6),
                "p_two_sided": res_two["p_value"],
                "p_target_better": res_less["p_value"],
                "p_target_worse": res_great["p_value"],
                "sig_two_sided": stars(res_two["p_value"]),
                "sig_target_better": stars(res_less["p_value"]),
                "d_bar": res_two["d_bar"],
                "mean_loss_target": res_two["mean_L1"],
                "mean_loss_other": res_two["mean_L2"],
                "rmse_target": rmse_target,
                "rmse_other": rmse_other,
                "mae_target": mae_target,
                "mae_other": mae_other,
                "better_model": better,
                "direction": direction,
                "n": res_two["n"],
                "nw_lag": res_two["lag_used"],
                "method": res_two["method"],
            })
            
            print(f"      DM统计量: {res_two['statistic']:.4f}, p值: {res_two['p_value']:.4f} {stars(res_two['p_value'])}")
            
        except Exception as e:
            print(f"      ❌ DM检验失败: {str(e)}")
            continue
    
    result_df = pd.DataFrame(results)
    
    if len(result_df) > 0:
        # 按better_model和p值排序
        result_df = result_df.sort_values(["better_model", "p_two_sided"]).reset_index(drop=True)
    
    return result_df

def print_dm_summary(result_df, target_model_name):
    """打印DM检验汇总"""
    
    print("\n" + "="*100)
    print(f"📊 DM检验结果汇总 - {target_model_name} vs 其他模型")
    print("="*100)
    
    if len(result_df) == 0:
        print("   无有效结果")
        return
    
    # 1. 显著性汇总
    print("\n1. 显著性汇总 (α=0.05):")
    print("-"*100)
    
    target_better = result_df[result_df['better_model'] == target_model_name]
    target_better_sig = target_better[target_better['p_two_sided'] <= 0.05]
    
    other_better = result_df[result_df['better_model'] != target_model_name]
    other_better_sig = other_better[other_better['p_two_sided'] <= 0.05]
    
    print(f"   {target_model_name}更好: {len(target_better)} 个对比")
    print(f"      其中显著: {len(target_better_sig)} 个 ({len(target_better_sig)/len(result_df)*100:.1f}%)")
    print(f"   其他模型更好: {len(other_better)} 个对比")
    print(f"      其中显著: {len(other_better_sig)} 个 ({len(other_better_sig)/len(result_df)*100:.1f}%)")
    
    # 2. 显著优于的模型列表
    if len(target_better_sig) > 0:
        print(f"\n2. {target_model_name}显著优于的模型 (α=0.05):")
        print("-"*100)
        for _, row in target_better_sig.iterrows():
            print(f"   ✓ {row['other_model']}: DM={row['dm_statistic']:.4f}, p={row['p_two_sided']:.4f} {row['sig_two_sided']}")
    
    # 3. 显著劣于的模型列表
    if len(other_better_sig) > 0:
        print(f"\n3. {target_model_name}显著劣于的模型 (α=0.05):")
        print("-"*100)
        for _, row in other_better_sig.iterrows():
            print(f"   ✗ {row['other_model']}: DM={row['dm_statistic']:.4f}, p={row['p_two_sided']:.4f} {row['sig_two_sided']}")
    
    # 4. 无显著差异的模型
    no_sig_diff = result_df[result_df['p_two_sided'] > 0.05]
    if len(no_sig_diff) > 0:
        print(f"\n4. 与{target_model_name}无显著差异的模型 (α=0.05):")
        print("-"*100)
        for _, row in no_sig_diff.iterrows():
            print(f"   ≈ {row['other_model']}: p={row['p_two_sided']:.4f}")
    
    print("\n" + "="*100)

def save_dm_results(result_df, save_dir, target_model_name):
    """保存DM检验结果"""
    
    os.makedirs(save_dir, exist_ok=True)
    
    # 保存CSV
    csv_path = os.path.join(save_dir, f'{target_model_name}_DM_test_results.csv')
    result_df.to_csv(csv_path, index=False)
    print(f"\n✅ CSV结果已保存: {csv_path}")
    
    # 保存TXT（对齐格式）
    txt_path = os.path.join(save_dir, f'{target_model_name}_DM_test_results.txt')
    with open(txt_path, 'w', encoding='utf-8') as f:
        f.write(result_df.to_string(index=False))
    print(f"✅ TXT结果已保存: {txt_path}")
    
    # 保存JSON（包含元数据）
    json_data = {
        'target_model': target_model_name,
        'n_comparisons': len(result_df),
        'n_target_better': int((result_df['better_model'] == target_model_name).sum()),
        'n_target_better_sig': int(((result_df['better_model'] == target_model_name) & 
                                     (result_df['p_two_sided'] <= 0.05)).sum()),
        'results': result_df.to_dict('records')
    }
    
    json_path = os.path.join(save_dir, f'{target_model_name}_DM_test_results.json')
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(json_data, f, indent=2, ensure_ascii=False)
    print(f"✅ JSON结果已保存: {json_path}")

def compare_fusion_models(result_m1, result_m2, save_dir):
    """对比Fusion-M1和Fusion-M2的DM检验结果"""
    
    print("\n" + "="*100)
    print("🔍 Fusion-M1 vs Fusion-M2 对比分析")
    print("="*100)
    
    if len(result_m1) == 0 or len(result_m2) == 0:
        print("   数据不足，无法对比")
        return
    
    # 统计显著优于的模型数量
    m1_better_sig = ((result_m1['better_model'] == 'Fusion-M1') & 
                     (result_m1['p_two_sided'] <= 0.05)).sum()
    m2_better_sig = ((result_m2['better_model'] == 'Fusion-M2') & 
                     (result_m2['p_two_sided'] <= 0.05)).sum()
    
    print(f"\n1. 显著优于其他模型的数量对比 (α=0.05):")
    print("-"*100)
    print(f"   Fusion-M1: {m1_better_sig} 个")
    print(f"   Fusion-M2: {m2_better_sig} 个")
    
    if m2_better_sig > m1_better_sig:
        print(f"   👑 Fusion-M2在显著性上更优！(多 {m2_better_sig - m1_better_sig} 个)")
    elif m1_better_sig > m2_better_sig:
        print(f"   👑 Fusion-M1在显著性上更优！(多 {m1_better_sig - m2_better_sig} 个)")
    else:
        print(f"   ⚖️ 两者显著优于的模型数量相同")
    
    # 找出共同对比的模型
    common_models = set(result_m1['other_model']) & set(result_m2['other_model'])
    
    if len(common_models) > 0:
        print(f"\n2. 对相同模型的对比 (共{len(common_models)}个):")
        print("-"*100)
        
        comparison_data = []
        
        for model in sorted(common_models):
            m1_row = result_m1[result_m1['other_model'] == model].iloc[0]
            m2_row = result_m2[result_m2['other_model'] == model].iloc[0]
            
            comparison_data.append({
                'model': model,
                'm1_dm_stat': m1_row['dm_statistic'],
                'm1_p_value': m1_row['p_two_sided'],
                'm1_sig': m1_row['sig_two_sided'],
                'm1_better': m1_row['better_model'] == 'Fusion-M1',
                'm2_dm_stat': m2_row['dm_statistic'],
                'm2_p_value': m2_row['p_two_sided'],
                'm2_sig': m2_row['sig_two_sided'],
                'm2_better': m2_row['better_model'] == 'Fusion-M2',
            })
        
        comp_df = pd.DataFrame(comparison_data)
        
        # 显示对比表
        print(f"\n{'模型':<15} {'M1-DM统计量':<15} {'M1-p值':<12} {'M1显著':<8} | {'M2-DM统计量':<15} {'M2-p值':<12} {'M2显著':<8}")
        print("-"*100)
        
        for _, row in comp_df.iterrows():
            m1_symbol = "✓" if row['m1_better'] else "✗"
            m2_symbol = "✓" if row['m2_better'] else "✗"
            print(f"{row['model']:<15} {row['m1_dm_stat']:<15.4f} {row['m1_p_value']:<12.4f} {row['m1_sig']:<8} | "
                  f"{row['m2_dm_stat']:<15.4f} {row['m2_p_value']:<12.4f} {row['m2_sig']:<8}")
        
        # 保存对比结果
        comp_csv = os.path.join(save_dir, 'Fusion_M1_vs_M2_comparison.csv')
        comp_df.to_csv(comp_csv, index=False)
        print(f"\n✅ 对比结果已保存: {comp_csv}")
    
    print("\n" + "="*100)

# ----------------------------
# 主函数
# ----------------------------
def main():
    """主函数"""
    
    print("="*100)
    print("🔬 Diebold-Mariano (DM) 检验")
    print("   同时检验Fusion-M1和Fusion-M2是否显著优于其他所有模型")
    print("="*100)
    
    # 配置参数
    RESULTS_BASE_DIR = "/Users/xiaxin/work/WindForecast_Project/03_Results/建模试验/simplified_enhanced_experiments"
    SAVE_DIR = "/Users/xiaxin/work/WindForecast_Project/03_Results/建模试验/DM_test_results"
    
    LOSS_KIND = "mse"           # 使用MSE作为损失
    HORIZON_HOURS = 24          # 24小时预测
    SMALL_SAMPLE = "HLN"        # HLN修正
    HAC_LAG_MODE = "h-1"        # Newey-West滞后 = h-1
    
    # 1. 加载所有模型的预测结果
    all_models = load_all_model_predictions(RESULTS_BASE_DIR)
    
    # 2. 对Fusion-M1运行DM检验
    if 'Fusion-M1' in all_models:
        result_m1 = run_dm_tests_for_target(
            all_models,
            target_model_name='Fusion-M1',
            loss_kind=LOSS_KIND,
            horizon_hours=HORIZON_HOURS,
            small_sample=SMALL_SAMPLE,
            hac_lag_mode=HAC_LAG_MODE
        )
        
        print_dm_summary(result_m1, 'Fusion-M1')
        save_dm_results(result_m1, SAVE_DIR, 'Fusion-M1')
    else:
        print("\n⚠️ Fusion-M1数据未找到")
        result_m1 = pd.DataFrame()
    
    # 3. 对Fusion-M2运行DM检验
    if 'Fusion-M2' in all_models:
        result_m2 = run_dm_tests_for_target(
            all_models,
            target_model_name='Fusion-M2',
            loss_kind=LOSS_KIND,
            horizon_hours=HORIZON_HOURS,
            small_sample=SMALL_SAMPLE,
            hac_lag_mode=HAC_LAG_MODE
        )
        
        print_dm_summary(result_m2, 'Fusion-M2')
        save_dm_results(result_m2, SAVE_DIR, 'Fusion-M2')
    else:
        print("\n⚠️ Fusion-M2数据未找到")
        result_m2 = pd.DataFrame()
    
    # 4. 对比Fusion-M1和Fusion-M2
    if len(result_m1) > 0 and len(result_m2) > 0:
        compare_fusion_models(result_m1, result_m2, SAVE_DIR)
    
    print("\n" + "="*100)
    print("🎉 DM检验完成!")
    print(f"📁 结果保存在: {SAVE_DIR}")
    print("="*100)
    
    print("\n📊 生成的文件:")
    print("   - Fusion-M1_DM_test_results.csv/txt/json")
    print("   - Fusion-M2_DM_test_results.csv/txt/json")
    print("   - Fusion_M1_vs_M2_comparison.csv (对比结果)")

if __name__ == "__main__":
    main()