"""
10m Wind Speed Correction using LightGBM + SHAP
使用LightGBM和SHAP进行10m风速订正
增强版本：添加ec_gfs_diff变量 + 输出完整预测数据用于下一步EOF分析
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, TimeSeriesSplit
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from scipy.stats import pearsonr
import lightgbm as lgb
import shap
import warnings
warnings.filterwarnings('ignore')

# 设置绘图参数
plt.rcParams['figure.dpi'] = 100
plt.rcParams['savefig.dpi'] = 300
plt.rcParams['font.size'] = 10

def load_and_prepare_data(file_path):
    """Load and prepare data for modeling"""
    print("🔄 Loading and preparing data...")
    
    # Load data
    df = pd.read_csv(file_path)
    df['datetime'] = pd.to_datetime(df['datetime'])
    
    print(f"📊 Data loaded: {len(df)} records")
    print(f"📅 Time range: {df['datetime'].min()} to {df['datetime'].max()}")
    
    # Focus on 10m wind speed + 多高度观测数据
    target_col = 'obs_wind_speed_10m'
    feature_cols = ['ec_wind_speed_10m', 'gfs_wind_speed_10m']
    
    # 检查多高度观测数据（用于下一步EOF分析）
    profile_cols = ['obs_wind_speed_10m', 'obs_wind_speed_30m', 
                   'obs_wind_speed_50m', 'obs_wind_speed_70m']
    
    # Check for required columns
    required_cols = [target_col] + feature_cols
    missing_cols = [col for col in required_cols if col not in df.columns]
    
    if missing_cols:
        print(f"❌ Missing columns: {missing_cols}")
        return None
    
    # 检查风廓线数据完整性
    profile_available = all(col in df.columns for col in profile_cols)
    print(f"🔍 Wind profile data availability: {profile_available}")
    if profile_available:
        print(f"   Available heights: {profile_cols}")
    else:
        missing_profile = [col for col in profile_cols if col not in df.columns]
        print(f"   Missing profile data: {missing_profile}")
    
    # Clean data (保留所有列用于后续分析)
    all_cols = ['datetime'] + required_cols
    if profile_available:
        all_cols.extend([col for col in profile_cols if col not in all_cols])
    
    df_clean = df[all_cols].dropna(subset=required_cols)
    print(f"✅ Clean data: {len(df_clean)} records ({len(df_clean)/len(df)*100:.1f}%)")
    
    # Basic statistics
    print(f"\n📈 Basic Statistics:")
    print(f"Observed 10m: mean={df_clean[target_col].mean():.2f}, std={df_clean[target_col].std():.2f}")
    print(f"EC 10m: mean={df_clean['ec_wind_speed_10m'].mean():.2f}, std={df_clean['ec_wind_speed_10m'].std():.2f}")
    print(f"GFS 10m: mean={df_clean['gfs_wind_speed_10m'].mean():.2f}, std={df_clean['gfs_wind_speed_10m'].std():.2f}")
    
    if profile_available:
        print(f"\n🌪️ Wind Profile Statistics:")
        for col in profile_cols:
            if col in df_clean.columns:
                print(f"{col}: mean={df_clean[col].mean():.2f}, std={df_clean[col].std():.2f}")
    
    return df_clean

def create_features(df):
    """Create enhanced core features for the model including ec_gfs_diff"""
    print("🔧 Creating enhanced core features...")
    
    df_features = df.copy()
    
    # Time-based features (only the ones we need)
    df_features['hour'] = df_features['datetime'].dt.hour
    df_features['day_of_year'] = df_features['datetime'].dt.dayofyear
    
    # Cyclical encoding for time features (only the ones we need)
    df_features['hour_sin'] = np.sin(2 * np.pi * df_features['hour'] / 24)
    df_features['day_sin'] = np.sin(2 * np.pi * df_features['day_of_year'] / 365)
    
    # Model combination features
    df_features['ec_gfs_mean'] = (df_features['ec_wind_speed_10m'] + df_features['gfs_wind_speed_10m']) / 2
    df_features['ec_gfs_diff'] = abs(df_features['ec_wind_speed_10m'] - df_features['gfs_wind_speed_10m'])  # 新增变量
    
    print(f"✅ Enhanced core features created: {df_features.shape[1]-2} features, {len(df_features)} samples")
    
    # Feature list (现在有6个特征)
    feature_names = [
        'ec_gfs_mean',          # EC和GFS的平均值 (最重要)
        'ec_gfs_diff',          # EC和GFS的差值 (新增：模式分歧度)
        'hour_sin',             # 小时的正弦编码 (日变化)
        'day_sin',              # 日期的正弦编码 (季节变化)
        'ec_wind_speed_10m',    # EC原始预报
        'gfs_wind_speed_10m'    # GFS原始预报
    ]
    
    print(f"📋 Selected features: {feature_names}")
    
    # 分析新增变量的统计特性
    print(f"\n📊 新增变量 ec_gfs_diff 统计:")
    print(f"  平均值: {df_features['ec_gfs_diff'].mean():.3f} m/s")
    print(f"  标准差: {df_features['ec_gfs_diff'].std():.3f} m/s")
    print(f"  最小值: {df_features['ec_gfs_diff'].min():.3f} m/s")
    print(f"  最大值: {df_features['ec_gfs_diff'].max():.3f} m/s")
    print(f"  中位数: {df_features['ec_gfs_diff'].median():.3f} m/s")
    
    # 分析模式分歧度的分布
    print(f"\n🔍 模式分歧度分析:")
    low_diff = (df_features['ec_gfs_diff'] < 0.5).sum()
    med_diff = ((df_features['ec_gfs_diff'] >= 0.5) & (df_features['ec_gfs_diff'] < 2.0)).sum()
    high_diff = (df_features['ec_gfs_diff'] >= 2.0).sum()
    
    total = len(df_features)
    print(f"  低分歧 (<0.5 m/s): {low_diff} ({low_diff/total*100:.1f}%)")
    print(f"  中分歧 (0.5-2.0 m/s): {med_diff} ({med_diff/total*100:.1f}%)")
    print(f"  高分歧 (>2.0 m/s): {high_diff} ({high_diff/total*100:.1f}%)")
    
    return df_features, feature_names

def split_data(df, feature_names, target_col='obs_wind_speed_10m', test_size=0.2):
    """Split data into train and test sets with time series consideration"""
    print(f"✂️ Splitting data (test_size={test_size})...")
    
    # Sort by datetime to maintain temporal order
    df_sorted = df.sort_values('datetime').reset_index(drop=True)
    
    # Time-based split (more realistic for time series)
    split_index = int(len(df_sorted) * (1 - test_size))
    
    train_df = df_sorted[:split_index]
    test_df = df_sorted[split_index:]
    
    # Prepare features and target
    X_train = train_df[feature_names]
    y_train = train_df[target_col]
    X_test = test_df[feature_names]
    y_test = test_df[target_col]
    
    print(f"📊 Train set: {len(X_train)} samples ({train_df['datetime'].min()} to {train_df['datetime'].max()})")
    print(f"📊 Test set: {len(X_test)} samples ({test_df['datetime'].min()} to {test_df['datetime'].max()})")
    
    # 分析训练集和测试集中ec_gfs_diff的分布
    print(f"\n📈 各数据集中 ec_gfs_diff 分布:")
    print(f"训练集: 均值={X_train['ec_gfs_diff'].mean():.3f}, 标准差={X_train['ec_gfs_diff'].std():.3f}")
    print(f"测试集: 均值={X_test['ec_gfs_diff'].mean():.3f}, 标准差={X_test['ec_gfs_diff'].std():.3f}")
    
    return X_train, X_test, y_train, y_test, train_df, test_df, df_sorted, split_index

def train_lightgbm_model(X_train, y_train, X_test, y_test):
    """Train LightGBM model with hyperparameter tuning"""
    print("🚀 Training LightGBM model...")
    
    # LightGBM parameters
    params = {
        'objective': 'regression',
        'metric': 'rmse',
        'boosting_type': 'gbdt',
        'num_leaves': 31,
        'learning_rate': 0.05,
        'feature_fraction': 0.9,
        'bagging_fraction': 0.8,
        'bagging_freq': 5,
        'verbose': -1,
        'random_state': 42
    }
    
    # Create datasets
    train_data = lgb.Dataset(X_train, label=y_train)
    valid_data = lgb.Dataset(X_test, label=y_test, reference=train_data)
    
    # Train model
    model = lgb.train(
        params,
        train_data,
        valid_sets=[train_data, valid_data],
        valid_names=['train', 'valid'],
        num_boost_round=1000,
        callbacks=[lgb.early_stopping(stopping_rounds=50), lgb.log_evaluation(0)]
    )
    
    print(f"✅ Model trained with {model.num_trees()} trees")
    
    return model

def predict_full_dataset(model, df_sorted, feature_names, split_index, target_col='obs_wind_speed_10m'):
    """对整个数据集进行预测并输出完整结果"""
    print("🔮 Predicting on full dataset for next step...")
    
    # 准备全数据集特征
    X_all = df_sorted[feature_names]
    
    # 预测全数据集
    corrected_10m_all = model.predict(X_all, num_iteration=model.best_iteration)
    
    # 创建输出数据框
    df_output = df_sorted[['datetime']].copy()
    df_output['obs_wind_speed_10m'] = df_sorted[target_col]
    df_output['corrected_wind_speed_10m'] = corrected_10m_all
    df_output['ec_wind_speed_10m'] = df_sorted['ec_wind_speed_10m']
    df_output['gfs_wind_speed_10m'] = df_sorted['gfs_wind_speed_10m']
    df_output['ec_gfs_mean'] = df_sorted['ec_gfs_mean']
    df_output['ec_gfs_diff'] = df_sorted['ec_gfs_diff']
    
    # 添加训练/测试标记
    df_output['data_split'] = 'train'
    df_output.loc[split_index:, 'data_split'] = 'test'
    
    # 如果有风廓线数据，也添加进去
    profile_cols = ['obs_wind_speed_30m', 'obs_wind_speed_50m', 'obs_wind_speed_70m']
    for col in profile_cols:
        if col in df_sorted.columns:
            df_output[col] = df_sorted[col]
    
    print(f"✅ Full dataset prediction completed: {len(df_output)} records")
    print(f"📊 Training period: {(df_output['data_split']=='train').sum()} records")
    print(f"📊 Testing period: {(df_output['data_split']=='test').sum()} records")
    
    # 计算训练期和测试期的统计信息
    train_mask = df_output['data_split'] == 'train'
    test_mask = df_output['data_split'] == 'test'
    
    print(f"\n📈 全数据集订正效果统计:")
    print(f"训练期 - 观测均值: {df_output.loc[train_mask, 'obs_wind_speed_10m'].mean():.3f}, 订正均值: {df_output.loc[train_mask, 'corrected_wind_speed_10m'].mean():.3f}")
    print(f"测试期 - 观测均值: {df_output.loc[test_mask, 'obs_wind_speed_10m'].mean():.3f}, 订正均值: {df_output.loc[test_mask, 'corrected_wind_speed_10m'].mean():.3f}")
    
    # 检查风廓线数据可用性（用于下一步EOF分析）
    available_profile_cols = [col for col in profile_cols if col in df_output.columns]
    if available_profile_cols:
        print(f"\n🌪️ 可用风廓线数据: {available_profile_cols}")
        print(f"   训练期完整风廓线数据: {df_output.loc[train_mask, available_profile_cols].dropna().shape[0]} 条")
        print(f"   全期完整风廓线数据: {df_output[['obs_wind_speed_10m'] + available_profile_cols].dropna().shape[0]} 条")
    else:
        print(f"⚠️  无可用风廓线数据，下一步EOF分析可能受限")
    
    return df_output

def evaluate_model(model, X_train, y_train, X_test, y_test, feature_names):
    """Evaluate model performance and calculate metrics"""
    print("📊 Evaluating model performance...")
    
    # Predictions
    y_train_pred = model.predict(X_train, num_iteration=model.best_iteration)
    y_test_pred = model.predict(X_test, num_iteration=model.best_iteration)
    
    # Calculate metrics
    def calculate_metrics(y_true, y_pred, set_name):
        corr, _ = pearsonr(y_true, y_pred)
        rmse = np.sqrt(mean_squared_error(y_true, y_pred))
        mae = mean_absolute_error(y_true, y_pred)
        r2 = r2_score(y_true, y_pred)
        bias = np.mean(y_pred - y_true)
        
        print(f"\n{set_name} Metrics:")
        print(f"  Correlation: {corr:.4f}")
        print(f"  RMSE: {rmse:.4f}")
        print(f"  MAE: {mae:.4f}")
        print(f"  R²: {r2:.4f}")
        print(f"  Bias: {bias:.4f}")
        
        return {
            'correlation': corr,
            'rmse': rmse,
            'mae': mae,
            'r2': r2,
            'bias': bias
        }
    
    # Calculate metrics for both sets
    train_metrics = calculate_metrics(y_train, y_train_pred, "🔵 Training")
    test_metrics = calculate_metrics(y_test, y_test_pred, "🔴 Testing")
    
    return {
        'train_metrics': train_metrics,
        'test_metrics': test_metrics,
        'y_train_pred': y_train_pred,
        'y_test_pred': y_test_pred
    }

def save_full_dataset_output(df_output, output_dir):
    """保存完整数据集输出用于下一步分析"""
    print("💾 Saving full dataset output for next step...")
    
    import os
    os.makedirs(output_dir, exist_ok=True)
    
    # 保存完整的订正数据
    full_output_path = f"{output_dir}/corrected_10m_wind_full_dataset.csv"
    df_output.to_csv(full_output_path, index=False)
    print(f"✅ Full corrected dataset saved: {full_output_path}")
    
    # 保存仅训练期数据（用于EOF分解）
    train_output = df_output[df_output['data_split'] == 'train'].copy()
    train_output_path = f"{output_dir}/corrected_10m_wind_training_period.csv"
    train_output.to_csv(train_output_path, index=False)
    print(f"✅ Training period data saved: {train_output_path}")
    
    # 保存数据描述文件
    description = f"""
10m风速订正完整数据集说明
=========================

数据文件:
1. corrected_10m_wind_full_dataset.csv - 完整数据集(100%)
2. corrected_10m_wind_training_period.csv - 训练期数据(80%)

主要列说明:
- datetime: 时间
- obs_wind_speed_10m: 观测10m风速
- corrected_wind_speed_10m: 订正后10m风速（用于下一步EOF分析）
- ec_wind_speed_10m: EC原始预报
- gfs_wind_speed_10m: GFS原始预报
- ec_gfs_mean: EC和GFS平均值
- ec_gfs_diff: EC和GFS差值（模式分歧度）
- data_split: train/test 数据划分标记
- obs_wind_speed_30m/50m/70m: 多高度观测数据（如果可用）

下一步使用说明:
- 使用 corrected_wind_speed_10m 作为EOF1预测的输入特征
- 使用训练期的风廓线数据进行EOF分解
- 训练期数据: {len(train_output)} 条记录
- 完整数据集: {len(df_output)} 条记录
- 时间范围: {df_output['datetime'].min()} 到 {df_output['datetime'].max()}

数据质量检查:
- 10m订正相关性: 在测试集上需要验证
- 风廓线完整性: 检查30m/50m/70m数据可用性
"""
    
    desc_path = f"{output_dir}/dataset_description.txt"
    with open(desc_path, 'w', encoding='utf-8') as f:
        f.write(description)
    print(f"✅ Dataset description saved: {desc_path}")
    
    # 数据质量报告
    print(f"\n📋 输出数据质量报告:")
    print(f"   总记录数: {len(df_output)}")
    print(f"   训练期记录: {len(train_output)}")
    print(f"   测试期记录: {len(df_output) - len(train_output)}")
    print(f"   时间跨度: {df_output['datetime'].max() - df_output['datetime'].min()}")
    
    # 检查缺失值
    missing_summary = df_output.isnull().sum()
    print(f"   缺失值检查:")
    for col, missing_count in missing_summary.items():
        if missing_count > 0:
            print(f"     {col}: {missing_count} ({missing_count/len(df_output)*100:.1f}%)")
        else:
            print(f"     {col}: 完整")
    
    return full_output_path, train_output_path

def create_output_visualization(df_output, output_dir):
    """创建输出数据的可视化"""
    print("📊 Creating output data visualization...")
    
    # 创建可视化
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    # 1. 时间序列对比（采样显示）
    ax1 = axes[0, 0]
    sample_size = min(2000, len(df_output))
    sample_indices = np.linspace(0, len(df_output)-1, sample_size, dtype=int)
    sample_data = df_output.iloc[sample_indices]
    
    ax1.plot(sample_data['datetime'], sample_data['obs_wind_speed_10m'], 
             'k-', linewidth=0.8, alpha=0.7, label='Observed 10m')
    ax1.plot(sample_data['datetime'], sample_data['corrected_wind_speed_10m'], 
             'r-', linewidth=0.8, alpha=0.7, label='Corrected 10m')
    ax1.plot(sample_data['datetime'], sample_data['ec_wind_speed_10m'], 
             'b-', linewidth=0.6, alpha=0.5, label='EC Original')
    
    ax1.set_xlabel('Time')
    ax1.set_ylabel('Wind Speed (m/s)')
    ax1.set_title('10m Wind Speed Correction Time Series')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    plt.setp(ax1.xaxis.get_majorticklabels(), rotation=45)
    
    # 2. 散点图对比
    ax2 = axes[0, 1]
    ax2.scatter(df_output['obs_wind_speed_10m'], df_output['corrected_wind_speed_10m'], 
                alpha=0.3, s=1, color='red', label='Corrected vs Observed')
    ax2.scatter(df_output['obs_wind_speed_10m'], df_output['ec_wind_speed_10m'], 
                alpha=0.2, s=1, color='blue', label='EC vs Observed')
    
    min_val, max_val = df_output['obs_wind_speed_10m'].min(), df_output['obs_wind_speed_10m'].max()
    ax2.plot([min_val, max_val], [min_val, max_val], 'k--', linewidth=2, label='Perfect')
    
    ax2.set_xlabel('Observed Wind Speed (m/s)')
    ax2.set_ylabel('Predicted Wind Speed (m/s)')
    ax2.set_title('Scatter Plot: Corrected vs Original')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # 3. 误差分布
    ax3 = axes[1, 0]
    corrected_error = df_output['corrected_wind_speed_10m'] - df_output['obs_wind_speed_10m']
    ec_error = df_output['ec_wind_speed_10m'] - df_output['obs_wind_speed_10m']
    
    ax3.hist(corrected_error, bins=50, alpha=0.6, color='red', label='Corrected Error', density=True)
    ax3.hist(ec_error, bins=50, alpha=0.6, color='blue', label='EC Error', density=True)
    ax3.axvline(0, color='black', linestyle='--', linewidth=2)
    
    ax3.set_xlabel('Prediction Error (m/s)')
    ax3.set_ylabel('Density')
    ax3.set_title('Error Distribution')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # 4. EC-GFS分歧度分析
    ax4 = axes[1, 1]
    ax4.hist(df_output['ec_gfs_diff'], bins=50, alpha=0.7, color='green', edgecolor='black')
    ax4.axvline(df_output['ec_gfs_diff'].mean(), color='red', linestyle='--', linewidth=2,
                label=f"Mean: {df_output['ec_gfs_diff'].mean():.3f}")
    
    ax4.set_xlabel('EC-GFS Difference (m/s)')
    ax4.set_ylabel('Frequency')
    ax4.set_title('Model Disagreement Distribution')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # 保存图片
    viz_path = f"{output_dir}/full_dataset_output_visualization.png"
    plt.savefig(viz_path, dpi=300, bbox_inches='tight')
    print(f"✅ Output visualization saved: {viz_path}")
    plt.show()
    
    return fig

def comprehensive_comparison_analysis(y_test, y_test_pred, X_test, test_df, output_dir):
    """Comprehensive comparison analysis on the same test dataset"""
    print("\n📊 ========== COMPREHENSIVE COMPARISON ANALYSIS ==========")
    print("Comparing performance on the SAME test dataset:")
    
    # Get baseline predictions
    ec_baseline = X_test['ec_wind_speed_10m'].values
    gfs_baseline = X_test['gfs_wind_speed_10m'].values
    lgb_corrected = y_test_pred
    observed = y_test.values
    
    def calculate_comprehensive_metrics(y_true, y_pred, model_name):
        """Calculate all performance metrics"""
        corr, _ = pearsonr(y_true, y_pred)
        rmse = np.sqrt(mean_squared_error(y_true, y_pred))
        mae = mean_absolute_error(y_true, y_pred)
        r2 = r2_score(y_true, y_pred)
        bias = np.mean(y_pred - y_true)
        
        # Additional metrics
        mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100  # Mean Absolute Percentage Error
        std_residuals = np.std(y_pred - y_true)
        
        return {
            'Model': model_name,
            'Correlation': corr,
            'RMSE': rmse,
            'MAE': mae,
            'R²': r2,
            'Bias': bias,
            'MAPE(%)': mape,
            'Std_Residuals': std_residuals,
            'Sample_Size': len(y_true)
        }
    
    # Calculate metrics for all models
    ec_metrics = calculate_comprehensive_metrics(observed, ec_baseline, 'EC_Original')
    gfs_metrics = calculate_comprehensive_metrics(observed, gfs_baseline, 'GFS_Original')  
    lgb_metrics = calculate_comprehensive_metrics(observed, lgb_corrected, 'LightGBM_Enhanced')
    
    # Create comparison dataframe
    comparison_df = pd.DataFrame([ec_metrics, gfs_metrics, lgb_metrics])
    
    print("\n🎯 PERFORMANCE COMPARISON ON SAME TEST DATASET:")
    print("=" * 100)
    print(comparison_df.round(4).to_string(index=False))
    
    # Calculate improvements
    print(f"\n📈 IMPROVEMENTS vs EC Original:")
    print("-" * 50)
    
    corr_improvement = (lgb_metrics['Correlation'] - ec_metrics['Correlation']) / ec_metrics['Correlation'] * 100
    rmse_improvement = (ec_metrics['RMSE'] - lgb_metrics['RMSE']) / ec_metrics['RMSE'] * 100
    mae_improvement = (ec_metrics['MAE'] - lgb_metrics['MAE']) / ec_metrics['MAE'] * 100
    r2_improvement = (lgb_metrics['R²'] - ec_metrics['R²']) / abs(ec_metrics['R²']) * 100
    
    print(f"✅ Correlation:     {lgb_metrics['Correlation']:.4f} vs {ec_metrics['Correlation']:.4f} = +{corr_improvement:+.1f}%")
    print(f"✅ RMSE:           {lgb_metrics['RMSE']:.4f} vs {ec_metrics['RMSE']:.4f} = {rmse_improvement:+.1f}%")
    print(f"✅ MAE:            {lgb_metrics['MAE']:.4f} vs {ec_metrics['MAE']:.4f} = {mae_improvement:+.1f}%")
    print(f"✅ R²:             {lgb_metrics['R²']:.4f} vs {ec_metrics['R²']:.4f} = +{r2_improvement:+.1f}%")
    
    return comparison_df, {
        'ec_improvement': {
            'correlation': corr_improvement,
            'rmse': rmse_improvement,
            'mae': mae_improvement,
            'r2': r2_improvement
        }
    }

def create_shap_analysis(model, X_train, X_test, feature_names, output_dir):
    """Create SHAP analysis for model interpretability"""
    print("🔍 Creating SHAP analysis...")
    
    # Create SHAP explainer
    explainer = shap.TreeExplainer(model)
    
    # Calculate SHAP values for test set (sample for speed)
    sample_size = min(1000, len(X_test))
    X_sample = X_test.sample(n=sample_size, random_state=42)
    shap_values = explainer.shap_values(X_sample)
    
    # Feature importance plot
    plt.figure(figsize=(10, 8))
    shap.summary_plot(shap_values, X_sample, feature_names=feature_names, show=False)
    plt.title('SHAP Feature Importance Summary (Enhanced Model)', fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    shap_summary_path = f"{output_dir}/shap_summary_plot.png"
    plt.savefig(shap_summary_path, dpi=300, bbox_inches='tight')
    print(f"✅ SHAP summary plot saved: {shap_summary_path}")
    plt.show()
    
    # Feature importance bar plot
    plt.figure(figsize=(10, 6))
    shap.summary_plot(shap_values, X_sample, feature_names=feature_names, plot_type="bar", show=False)
    plt.title('SHAP Feature Importance (Mean |SHAP value|)', fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    shap_bar_path = f"{output_dir}/shap_importance_bar.png"
    plt.savefig(shap_bar_path, dpi=300, bbox_inches='tight')
    print(f"✅ SHAP bar plot saved: {shap_bar_path}")
    plt.show()
    
    # Calculate feature importance
    feature_importance = np.abs(shap_values).mean(0)
    importance_df = pd.DataFrame({
        'feature': feature_names,
        'importance': feature_importance
    }).sort_values('importance', ascending=False)
    
    print("\n🏆 Enhanced Model - Top Feature Importance:")
    print(importance_df.round(4))
    
    # 分析ec_gfs_diff的重要性排名
    ec_gfs_diff_rank = importance_df[importance_df['feature'] == 'ec_gfs_diff'].index[0] + 1
    print(f"\n📊 ec_gfs_diff 重要性排名: 第 {ec_gfs_diff_rank} 位 (共 {len(feature_names)} 个特征)")
    
    return importance_df, shap_values

def save_results(model, evaluation_results, comparison_df, importance_df, output_dir):
    """Save all results to files"""
    print("💾 Saving results...")
    
    import os
    os.makedirs(output_dir, exist_ok=True)
    
    # Save model
    model_path = f"{output_dir}/enhanced_lightgbm_model.txt"
    model.save_model(model_path)
    print(f"✅ Enhanced model saved: {model_path}")
    
    # Save metrics
    metrics_data = {
        'train_metrics': evaluation_results['train_metrics'],
        'test_metrics': evaluation_results['test_metrics']
    }
    
    metrics_df = pd.DataFrame(metrics_data).T
    metrics_path = f"{output_dir}/enhanced_model_metrics.csv"
    metrics_df.to_csv(metrics_path)
    print(f"✅ Enhanced metrics saved: {metrics_path}")
    
    # Save comparison
    comparison_path = f"{output_dir}/enhanced_model_comparison.csv"
    comparison_df.to_csv(comparison_path, index=False)
    print(f"✅ Enhanced comparison saved: {comparison_path}")
    
    # Save feature importance
    importance_path = f"{output_dir}/enhanced_feature_importance.csv"
    importance_df.to_csv(importance_path, index=False)
    print(f"✅ Enhanced feature importance saved: {importance_path}")

def main():
    """Main execution function"""
    # Paths
    data_path = '/Users/xiaxin/work/WindForecast_Project/01_Data/processed/imputed_data/changma_clean.csv'
    output_dir = '/Users/xiaxin/work/WindForecast_Project/03_Results/建模/10m风速建模/LightGBM_增强版/'
    
    try:
        print("🚀 Starting Enhanced 10m Wind Speed Correction with LightGBM...")
        print("✨ 新增特征: ec_gfs_diff (EC和GFS的分歧度)")
        print("🎯 目标: 输出完整预测数据用于下一步EOF分析")
        
        # 0. Create output directory first
        import os
        os.makedirs(output_dir, exist_ok=True)
        print(f"📁 Output directory created: {output_dir}")
        
        # 1. Load and prepare data
        df = load_and_prepare_data(data_path)
        if df is None:
            return
        
        # 2. Create enhanced features (including ec_gfs_diff)
        df_features, feature_names = create_features(df)
        
        # 3. Split data (返回完整排序数据和分割索引)
        X_train, X_test, y_train, y_test, train_df, test_df, df_sorted, split_index = split_data(df_features, feature_names)
        
        # 4. Train model
        model = train_lightgbm_model(X_train, y_train, X_test, y_test)
        
        # 5. Evaluate model
        evaluation_results = evaluate_model(model, X_train, y_train, X_test, y_test, feature_names)
        
        # 6. 🔥 NEW: 预测完整数据集并输出（关键步骤！）
        df_output = predict_full_dataset(model, df_features, feature_names, split_index)
        
        # 7. 🔥 NEW: 保存完整数据集输出用于下一步EOF分析
        full_output_path, train_output_path = save_full_dataset_output(df_output, output_dir)
        
        # 8. 🔥 NEW: 创建输出数据可视化
        output_fig = create_output_visualization(df_output, output_dir)
        
        # 9. COMPREHENSIVE COMPARISON ANALYSIS
        comparison_df, improvements = comprehensive_comparison_analysis(
            y_test, evaluation_results['y_test_pred'], X_test, test_df, output_dir)
        
        # 10. SHAP analysis
        importance_df, shap_values = create_shap_analysis(model, X_train, X_test, feature_names, output_dir)
        
        # 11. Save results
        save_results(model, evaluation_results, comparison_df, importance_df, output_dir)
        
        # 12. FINAL SUMMARY WITH NEXT STEP GUIDANCE
        print("\n" + "="*80)
        print("🎉 ENHANCED 10m WIND SPEED CORRECTION COMPLETED SUCCESSFULLY!")
        print("🎯 数据已准备好用于下一步EOF分析")
        print("="*80)
        
        # Extract key metrics for final summary
        lgb_corr = comparison_df.loc[comparison_df['Model'] == 'LightGBM_Enhanced', 'Correlation'].iloc[0]
        ec_corr = comparison_df.loc[comparison_df['Model'] == 'EC_Original', 'Correlation'].iloc[0]
        lgb_rmse = comparison_df.loc[comparison_df['Model'] == 'LightGBM_Enhanced', 'RMSE'].iloc[0]
        ec_rmse = comparison_df.loc[comparison_df['Model'] == 'EC_Original', 'RMSE'].iloc[0]
        
        print(f"📊 STEP 1 完成情况 - 10m风速订正:")
        print(f"   • 模型训练: ✅ LightGBM + ec_gfs_diff特征")
        print(f"   • 测试集性能: 相关性 {lgb_corr:.4f} (提升 {improvements['ec_improvement']['correlation']:.1f}%)")
        print(f"   • 全数据集预测: ✅ {len(df_output)} 条记录")
        print(f"   • 数据输出: ✅ 已保存用于EOF分析")
        
        print(f"\n📁 下一步EOF分析所需文件:")
        print(f"   🎯 主要输入文件: {full_output_path}")
        print(f"   📋 训练期数据: {train_output_path}")
        print(f"   📊 核心列: corrected_wind_speed_10m (订正后的10m风速)")
        
        # 检查风廓线数据
        profile_cols = ['obs_wind_speed_30m', 'obs_wind_speed_50m', 'obs_wind_speed_70m']
        available_profile = [col for col in profile_cols if col in df_output.columns]
        
        if available_profile:
            print(f"\n🌪️ 风廓线数据检查:")
            print(f"   ✅ 可用高度: {available_profile}")
            
            # 检查训练期完整数据
            train_data = df_output[df_output['data_split'] == 'train']
            complete_profile = train_data[['obs_wind_speed_10m'] + available_profile].dropna()
            print(f"   📊 训练期完整风廓线: {len(complete_profile)} 条 ({len(complete_profile)/len(train_data)*100:.1f}%)")
            
            if len(complete_profile) > 100:
                print(f"   🎯 数据充足，可以进行EOF分解")
            else:
                print(f"   ⚠️  数据不足，EOF分析可能受限")
        else:
            print(f"   ❌ 无风廓线数据，无法进行EOF分析")
        
        print(f"\n💡 STEP 2 - EOF分析建议:")
        print(f"   1. 使用训练期数据 ({train_output_path}) 进行EOF分解")
        print(f"   2. 提取风廓线主模态 (EOF1) 作为目标变量")
        print(f"   3. 用 corrected_wind_speed_10m 预测 EOF1 时间系数")
        print(f"   4. 重构70m风速用于功率预测")
        
        success_indicator = "🟢 READY FOR STEP 2" if available_profile and len(complete_profile) > 100 else "🟡 LIMITED DATA" if available_profile else "🔴 MISSING PROFILE DATA"
        print(f"\n🎯 STEP 2 准备状态: {success_indicator}")
        
        return model, evaluation_results, comparison_df, importance_df, df_output, full_output_path
        
    except Exception as e:
        print(f"❌ Error: {str(e)}")
        import traceback
        traceback.print_exc()
        return None, None, None, None, None, None

if __name__ == "__main__":
    model, evaluation_results, comparison_df, importance_df, df_output, full_output_path = main()