import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from scipy import stats
import lightgbm as lgb
import warnings
warnings.filterwarnings('ignore')

# ============================================================================
# 样式设置
# ============================================================================
plt.rcParams['font.family'] = 'Arial'
plt.rcParams['font.sans-serif'] = ['Arial', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['figure.dpi'] = 600

# ============================================================================
# 核心算法
# ============================================================================
def diebold_mariano_test(errors1, errors2, h=1):
    d = errors1**2 - errors2**2
    mean_d = np.mean(d)
    
    def autocovariance(series, lag):
        n = len(series)
        mean = np.mean(series)
        return np.sum((series[:n-lag] - mean) * (series[lag:] - mean)) / n
    
    gamma_0 = autocovariance(d, 0)
    variance = gamma_0
    for lag in range(1, h):
        gamma_lag = autocovariance(d, lag)
        variance += 2 * (1 - lag/(h+1)) * gamma_lag
    
    dm_stat = mean_d / np.sqrt(variance / len(d))
    p_value = 1 - stats.norm.cdf(dm_stat)
    
    return dm_stat, p_value

def load_and_prepare_data(data_path):
    """加载并准备数据"""
    df = pd.read_csv(data_path)
    required_cols = [
        'obs_wind_speed_10m', 'obs_wind_speed_30m', 
        'obs_wind_speed_50m', 'obs_wind_speed_70m',
        'ec_wind_speed_10m', 'ec_wind_speed_30m',
        'ec_wind_speed_50m', 'ec_wind_speed_70m',
        'ec_temperature_10m',
        'gfs_wind_speed_10m', 'gfs_wind_speed_30m',
        'gfs_wind_speed_50m', 'gfs_wind_speed_70m',
        'gfs_temperature_10m',
        'power'
    ]
    df_clean = df[required_cols].dropna()
    train_df, test_df = train_test_split(df_clean, test_size=0.2, random_state=42, shuffle=False)
    return train_df, test_df

# ============================================================================
# 两阶段模型：逐个气象要素订正
# ============================================================================
def two_stage_train(train_df, nwp_source, wind_features, other_features):
    """
    两阶段训练
    阶段1: 每个NWP气象要素 -> 对应的观测气象要素 (逐个订正)
    阶段2: 所有订正后的气象要素 -> 功率
    """
    all_features = wind_features + other_features
    correction_models = {}
    obs_features_list = []
    
    # LightGBM参数
    lgb_params = {
        'objective': 'regression',
        'metric': 'rmse',
        'num_leaves': 31,
        'learning_rate': 0.05,
        'feature_fraction': 0.9,
        'bagging_fraction': 0.8,
        'bagging_freq': 5,
        'verbose': -1,
        'random_state': 42
    }
    
    # 阶段1: 逐个气象要素订正
    for nwp_feature in all_features:
        # 确定对应的观测特征名
        obs_feature = nwp_feature.replace(f'{nwp_source}_', 'obs_')
        
        if obs_feature not in train_df.columns:
            print(f"  警告: 找不到观测特征 {obs_feature}，跳过")
            continue
        
        # 检查是否有缺失值
        if train_df[nwp_feature].isna().any() or train_df[obs_feature].isna().any():
            print(f"  警告: {nwp_feature} 或 {obs_feature} 包含NaN，跳过")
            continue
        
        # 训练单变量订正模型: NWP -> Obs
        X = train_df[[nwp_feature]].values
        y = train_df[obs_feature].values
        
        model = lgb.LGBMRegressor(**lgb_params, n_estimators=100)
        model.fit(X, y)
        
        correction_models[nwp_feature] = {
            'model': model,
            'obs_feature': obs_feature
        }
        obs_features_list.append(obs_feature)
    
    if len(correction_models) == 0:
        raise ValueError("没有可用的订正模型，请检查数据")
    
    # 阶段2: 功率转换模型
    # 使用所有观测气象要素训练
    X_power = train_df[obs_features_list].values
    y_power = train_df['power'].values
    
    # 再次检查是否有NaN
    if np.isnan(X_power).any() or np.isnan(y_power).any():
        raise ValueError("功率模型训练数据包含NaN")
    
    power_model = lgb.LGBMRegressor(**lgb_params, n_estimators=100)
    power_model.fit(X_power, y_power)
    
    return correction_models, power_model

def two_stage_predict(test_df, correction_models, power_model, nwp_source, wind_features, other_features):
    """
    两阶段预测
    阶段1: 用订正模型逐个订正每个气象要素
    阶段2: 用订正后的所有气象要素预测功率
    """
    all_features = wind_features + other_features
    
    # 阶段1: 逐个订正
    corrected_features = []
    
    for nwp_feature in all_features:
        if nwp_feature not in correction_models:
            continue
        
        model_info = correction_models[nwp_feature]
        model = model_info['model']
        
        # 订正
        X = test_df[[nwp_feature]].values
        corrected = model.predict(X)
        
        corrected_features.append(corrected)
    
    # 阶段2: 用所有订正后的要素预测功率
    X_power = np.column_stack(corrected_features)
    power_pred = power_model.predict(X_power)
    
    return power_pred

def calculate_metrics(y_true, y_pred):
    """计算评估指标"""
    return {
        'RMSE': np.sqrt(mean_squared_error(y_true, y_pred)),
        'R2': r2_score(y_true, y_pred),
        'Correlation': np.corrcoef(y_true, y_pred)[0, 1],
    }

def two_stage_pipeline(train_df, test_df, nwp_source, method):
    """完整的两阶段流程"""
    # 根据方法选择特征
    if method == 'raw':
        wind_features = [f'{nwp_source}_wind_speed_70m']
        other_features = [f'{nwp_source}_temperature_10m']
    elif method == 'standard':
        wind_features = [f'{nwp_source}_wind_speed_{h}m' for h in [30, 50, 70]]
        other_features = [f'{nwp_source}_temperature_10m']
    elif method == 'extended':
        wind_features = [f'{nwp_source}_wind_speed_{h}m' for h in [10, 30, 50, 70]]
        other_features = [f'{nwp_source}_temperature_10m']
    else:
        raise ValueError(f"未知方法: {method}")
    
    # 训练
    correction_models, power_model = two_stage_train(
        train_df, nwp_source, wind_features, other_features
    )
    
    # 预测
    power_pred = two_stage_predict(
        test_df, correction_models, power_model, 
        nwp_source, wind_features, other_features
    )
    
    return power_pred

def evaluate_all_models(train_df, test_df):
    """评估所有模型"""
    results = {}
    predictions = {}
    y_true = test_df['power'].values
    
    for nwp in ['ec', 'gfs']:
        for method in ['raw', 'standard', 'extended']:
            name = f'{nwp.upper()}-{method.capitalize()}'
            print(f"\n训练模型: {name}")
            pred = two_stage_pipeline(train_df, test_df, nwp, method)
            results[name] = calculate_metrics(y_true, pred)
            predictions[name] = pred
    
    # DM检验
    dm_results = {}
    for nwp in ['EC', 'GFS']:
        errors_raw = y_true - predictions[f'{nwp}-Raw']
        errors_std = y_true - predictions[f'{nwp}-Standard']
        errors_ext = y_true - predictions[f'{nwp}-Extended']
        
        dm_stat, p_val = diebold_mariano_test(errors_raw, errors_std)
        dm_results[f'{nwp}-Standard vs {nwp}-Raw'] = (dm_stat, p_val)
        
        dm_stat, p_val = diebold_mariano_test(errors_std, errors_ext)
        dm_results[f'{nwp}-Extended vs {nwp}-Standard'] = (dm_stat, p_val)
    
    return results, dm_results

# ============================================================================
# 三轴图可视化
# ============================================================================
def create_triple_axis_chart(results, dm_results, output_dir):
    """
    创建三轴点线图：左y轴-相关系数，右y轴1-RMSE，右y轴2-R²
    """
    fig, ax1 = plt.subplots(figsize=(11, 8))
    
    # 准备数据
    model_groups = ['HH', 'SR', 'ER']
    x_pos = np.arange(len(model_groups))
    
    # EC组数据
    ec_corr = [results['EC-Raw']['Correlation'], 
               results['EC-Standard']['Correlation'], 
               results['EC-Extended']['Correlation']]
    ec_rmse = [results['EC-Raw']['RMSE'],
               results['EC-Standard']['RMSE'],
               results['EC-Extended']['RMSE']]
    ec_r2 = [results['EC-Raw']['R2'],
             results['EC-Standard']['R2'],
             results['EC-Extended']['R2']]
    
    # GFS组数据
    gfs_corr = [results['GFS-Raw']['Correlation'],
                results['GFS-Standard']['Correlation'],
                results['GFS-Extended']['Correlation']]
    gfs_rmse = [results['GFS-Raw']['RMSE'],
                results['GFS-Standard']['RMSE'],
                results['GFS-Extended']['RMSE']]
    gfs_r2 = [results['GFS-Raw']['R2'],
              results['GFS-Standard']['R2'],
              results['GFS-Extended']['R2']]
    
    # 颜色设置
    corr_color = 'black'
    rmse_color = '#1f77b4'
    r2_color = '#2ca02c'
    
    # ========== 第一个y轴：相关系数 ==========
    ax1.set_ylabel('Correlation Coefficient', fontsize=30, fontweight='normal', color=corr_color)
    
    line1 = ax1.plot(x_pos, ec_corr, 'o-', linewidth=2, markersize=18,
                     color=corr_color, linestyle='-', label='EC Correlation', alpha=0.8)
    line2 = ax1.plot(x_pos, gfs_corr, 's-', linewidth=2, markersize=18,
                     color=corr_color, linestyle='-', label='GFS Correlation', alpha=0.4)
    
    ax1.tick_params(axis='y', labelcolor=corr_color, labelsize=26)
    ax1.spines['left'].set_color(corr_color)
    ax1.spines['left'].set_linewidth(3)
    ax1.spines['top'].set_visible(True)
    ax1.spines['right'].set_visible(False)
    ax1.grid(False)
    
    # 自定义Y轴范围 - 左Y轴（Correlation）
    corr_ymin, corr_ymax = 0.55, 0.63
    corr_yticks = np.arange(0.55, 0.64, 0.01)
    ax1.set_ylim(corr_ymin, corr_ymax)
    ax1.set_yticks(corr_yticks)
    
    # ========== 第二个y轴：RMSE ==========
    ax2 = ax1.twinx()
    ax2.set_ylabel('RMSE (MW)', fontsize=26, fontweight='normal', color=rmse_color)
    
    line3 = ax2.plot(x_pos, ec_rmse, 'o-', linewidth=2, markersize=18,
                     color=rmse_color, linestyle='-', label='EC RMSE', alpha=0.8)
    line4 = ax2.plot(x_pos, gfs_rmse, 's-', linewidth=2, markersize=18,
                     color=rmse_color, linestyle='-', label='GFS RMSE', alpha=0.4)
    
    ax2.tick_params(axis='y', labelcolor=rmse_color, labelsize=26)
    ax2.spines['top'].set_visible(False)
    ax2.spines['left'].set_visible(False)
    ax2.spines['bottom'].set_visible(False)
    ax2.spines['right'].set_color(rmse_color)
    ax2.spines['right'].set_linewidth(3)
    
    # 自定义Y轴范围 - 右Y轴1（RMSE）
    rmse_ymin, rmse_ymax = 32.5, 35.0
    rmse_yticks = np.arange(32.5, 35.1, 0.5)
    ax2.set_ylim(rmse_ymin, rmse_ymax)
    ax2.set_yticks(rmse_yticks)
    
    # ========== 第三个y轴：R² ==========
    ax3 = ax1.twinx()
    ax3.spines['right'].set_position(('outward', 120))
    ax3.set_ylabel('R²', fontsize=26, fontweight='normal', color=r2_color)
    
    line5 = ax3.plot(x_pos, ec_r2, 'o-', linewidth=2, markersize=18,
                     color=r2_color, linestyle='-', label='EC R²', alpha=0.8)
    line6 = ax3.plot(x_pos, gfs_r2, 's-', linewidth=2, markersize=18,
                     color=r2_color, linestyle='-', label='GFS R²', alpha=0.4)
    
    ax3.tick_params(axis='y', labelcolor=r2_color, labelsize=26)
    ax3.spines['top'].set_visible(False)
    ax3.spines['left'].set_visible(False)
    ax3.spines['bottom'].set_visible(False)
    ax3.spines['right'].set_color(r2_color)
    ax3.spines['right'].set_linewidth(3)
    
    # 自定义Y轴范围 - 右Y轴2（R²）
    r2_ymin, r2_ymax = 0.24, 0.38
    r2_yticks = np.arange(0.24, 0.39, 0.02)
    ax3.set_ylim(r2_ymin, r2_ymax)
    ax3.set_yticks(r2_yticks)
    
    # ========== x轴设置 ==========
    ax1.set_xticks(x_pos)
    ax1.set_xticklabels(model_groups, fontsize=26, fontweight='normal')
    
    # ========== 图例 ==========
    lines = line1 + line2
    labels = ['ECMWF-WRF', 'GFS-WRF']
    ax1.legend(lines, labels, loc='upper center', fontsize=26, frameon=False, ncol=2,
              columnspacing=1.0)
    
    plt.tight_layout()
    
    # 保存
    for ext in ['png', 'pdf']:
        plt.savefig(f"{output_dir}/final_4_a_Fig_triple_axis.{ext}", 
                   dpi=600, bbox_inches='tight', facecolor='white')
    
    return fig, (ax1, ax2, ax3)

# ============================================================================
# 主函数
# ============================================================================
def main():
    data_path = '/Users/xiaxin/work/WindForecast_Project/01_Data/processed/matched_data/changma_matched.csv'
    output_dir = "/Users/xiaxin/work/WindForecast_Project/03_Results/re-plot-figures/figure-4/"
    
    print("="*70)
    print("Figure 4a - 三种策略对比 (逐个气象要素订正)")
    print("="*70)
    
    train_df, test_df = load_and_prepare_data(data_path)
    print(f"\n训练集样本数: {len(train_df)}")
    print(f"测试集样本数: {len(test_df)}")
    
    results, dm_results = evaluate_all_models(train_df, test_df)
    
    # 打印结果表格
    print("\n" + "="*70)
    print("模型性能对比 (测试集)")
    print("="*70)
    print(f"{'模型':<20} {'Correlation':>12} {'RMSE':>12} {'R²':>10}")
    print("-"*70)
    for name in ['EC-Raw', 'EC-Standard', 'EC-Extended', 'GFS-Raw', 'GFS-Standard', 'GFS-Extended']:
        print(f"{name:<20} {results[name]['Correlation']:>12.4f} "
              f"{results[name]['RMSE']:>12.2f} {results[name]['R2']:>10.4f}")
    
    # 打印DM检验结果
    print("\n" + "="*70)
    print("Diebold-Mariano 显著性检验")
    print("="*70)
    for comparison, (dm_stat, p_val) in dm_results.items():
        sig_marker = '***' if p_val < 0.01 else '**' if p_val < 0.05 else '*' if p_val < 0.1 else 'n.s.'
        print(f"{comparison}:")
        print(f"  DM统计量: {dm_stat:.4f}, p值: {p_val:.4f} {sig_marker}")
    print("\n显著性标记: *** p<0.01, ** p<0.05, * p<0.1, n.s. 不显著")
    
    # 创建三轴图
    fig, axes = create_triple_axis_chart(results, dm_results, output_dir)
    
    print("\n" + "="*70)
    print("完成！图表已保存。")
    print("="*70)

if __name__ == "__main__":
    main()