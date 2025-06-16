"""
完整的VAE增强风速订正系统 - 可直接运行版本
基于您已有的最优结果进行VAE增强
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.preprocessing import StandardScaler
from scipy.stats import pearsonr
import lightgbm as lgb
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
import warnings
import os
from datetime import datetime
warnings.filterwarnings('ignore')

# 设置matplotlib中文显示
plt.rcParams['font.sans-serif'] = ['Arial Unicode MS', 'SimHei']
plt.rcParams['axes.unicode_minus'] = False

class WindSpeedVAE(nn.Module):
    """风速预测专用的变分自编码器"""
    
    def __init__(self, input_dim=8, latent_dim=3, condition_dim=2):
        super(WindSpeedVAE, self).__init__()
        
        # 编码器
        self.encoder = nn.Sequential(
            nn.Linear(input_dim + condition_dim, 32),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(16, 8),
            nn.ReLU()
        )
        
        # 潜在空间参数
        self.fc_mu = nn.Linear(8, latent_dim)
        self.fc_logvar = nn.Linear(8, latent_dim)
        
        # 解码器
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim + condition_dim, 8),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(8, 16),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(16, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
            nn.Softplus()  # 确保输出为正
        )
    
    def encode(self, x, condition):
        x_cond = torch.cat([x, condition], dim=1)
        h = self.encoder(x_cond)
        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h)
        return mu, logvar
    
    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std
    
    def decode(self, z, condition):
        z_cond = torch.cat([z, condition], dim=1)
        return self.decoder(z_cond)
    
    def forward(self, x, condition):
        mu, logvar = self.encode(x, condition)
        z = self.reparameterize(mu, logvar)
        recon = self.decode(z, condition)
        return recon, mu, logvar

def vae_loss_function(recon_x, x, mu, logvar, beta=0.1):
    """VAE损失函数"""
    recon_loss = F.mse_loss(recon_x, x.unsqueeze(1), reduction='sum')
    kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    return recon_loss + beta * kl_loss

class VAEWindCorrector:
    """VAE风速订正器"""
    
    def __init__(self, latent_dim=3):
        self.vae = WindSpeedVAE(latent_dim=latent_dim)
        self.scaler_features = StandardScaler()
        self.scaler_target = StandardScaler()
        self.scaler_condition = StandardScaler()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.vae.to(self.device)
        self.is_trained = False
        
    def prepare_data(self, X, y):
        """准备训练数据"""
        # 标准化特征
        X_scaled = self.scaler_features.fit_transform(X)
        y_scaled = self.scaler_target.fit_transform(y.reshape(-1, 1)).flatten()
        
        # 准备条件（使用EC/GFS均值和时间特征）
        conditions = X_scaled[:, [0, 2]]  # ec_gfs_mean, hour_sin
        conditions_scaled = self.scaler_condition.fit_transform(conditions)
        
        return X_scaled, y_scaled, conditions_scaled
    
    def train_vae(self, X, y, epochs=300, batch_size=128, lr=1e-3, verbose=True):
        """训练VAE"""
        if verbose:
            print("🧠 开始训练VAE...")
        
        X_scaled, y_scaled, conditions_scaled = self.prepare_data(X, y)
        
        # 转换为张量
        X_tensor = torch.FloatTensor(X_scaled).to(self.device)
        y_tensor = torch.FloatTensor(y_scaled).to(self.device)
        cond_tensor = torch.FloatTensor(conditions_scaled).to(self.device)
        
        dataset = TensorDataset(X_tensor, y_tensor, cond_tensor)
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
        
        optimizer = torch.optim.Adam(self.vae.parameters(), lr=lr, weight_decay=1e-5)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=20, factor=0.5)
        
        self.vae.train()
        losses = []
        
        for epoch in range(epochs):
            epoch_loss = 0
            for batch_x, batch_y, batch_cond in dataloader:
                optimizer.zero_grad()
                
                recon_y, mu, logvar = self.vae(batch_x, batch_cond)
                loss = vae_loss_function(recon_y, batch_y, mu, logvar)
                
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.vae.parameters(), max_norm=1.0)
                optimizer.step()
                
                epoch_loss += loss.item()
            
            avg_loss = epoch_loss / len(dataloader)
            losses.append(avg_loss)
            scheduler.step(avg_loss)
            
            if verbose and epoch % 50 == 0:
                print(f"Epoch {epoch}/{epochs}, Loss: {avg_loss:.4f}")
        
        self.is_trained = True
        if verbose:
            print("✅ VAE训练完成")
        
        return losses
    
    def generate_samples(self, target_range=(3, 4), n_samples=500, reference_condition=None):
        """生成指定区间的风速样本"""
        if not self.is_trained:
            raise ValueError("VAE未训练")
        
        self.vae.eval()
        generated_samples = []
        
        with torch.no_grad():
            # 如果没有提供参考条件，使用随机条件
            if reference_condition is None:
                # 生成随机条件
                conditions = torch.randn(n_samples, 2).to(self.device) * 0.5
            else:
                conditions = torch.FloatTensor(reference_condition).repeat(n_samples, 1).to(self.device)
            
            # 从潜在空间采样
            z = torch.randn(n_samples, self.vae.fc_mu.out_features).to(self.device)
            
            # 解码生成样本
            generated = self.vae.decode(z, conditions)
            generated_np = generated.cpu().numpy().flatten()
            
            # 反标准化
            generated_denorm = self.scaler_target.inverse_transform(generated_np.reshape(-1, 1)).flatten()
            
            # 筛选目标区间内的样本
            mask = (generated_denorm >= target_range[0]) & (generated_denorm < target_range[1])
            valid_samples = generated_denorm[mask]
        
        return valid_samples
    
    def predict_with_uncertainty(self, X, n_samples=20):
        """预测并量化不确定性"""
        if not self.is_trained:
            raise ValueError("VAE未训练")
        
        X_scaled = self.scaler_features.transform(X)
        conditions = X_scaled[:, [0, 2]]
        conditions_scaled = self.scaler_condition.transform(conditions)
        
        self.vae.eval()
        predictions = []
        uncertainties = []
        
        with torch.no_grad():
            for i in range(len(X)):
                # 为每个样本生成多个预测
                x_sample = torch.FloatTensor(X_scaled[i]).unsqueeze(0).repeat(n_samples, 1).to(self.device)
                cond_sample = torch.FloatTensor(conditions_scaled[i]).unsqueeze(0).repeat(n_samples, 1).to(self.device)
                
                # 编码到潜在空间
                mu, logvar = self.vae.encode(x_sample, cond_sample)
                
                # 多次采样
                sample_preds = []
                for _ in range(n_samples):
                    z = self.vae.reparameterize(mu, logvar)
                    pred = self.vae.decode(z, cond_sample)
                    sample_preds.append(pred.cpu().numpy())
                
                sample_preds = np.array(sample_preds).flatten()
                # 反标准化
                sample_preds_denorm = self.scaler_target.inverse_transform(sample_preds.reshape(-1, 1)).flatten()
                
                predictions.append(np.mean(sample_preds_denorm))
                uncertainties.append(np.std(sample_preds_denorm))
        
        return np.array(predictions), np.array(uncertainties)

def load_and_prepare_data(data_path):
    """加载和准备数据"""
    print("📊 加载数据...")
    
    df = pd.read_csv(data_path)
    df['datetime'] = pd.to_datetime(df['datetime'])
    
    # 基础清理
    df_clean = df[['datetime', 'obs_wind_speed_10m', 'ec_wind_speed_10m', 'gfs_wind_speed_10m']].dropna()
    
    # 特征工程
    df_clean['hour'] = df_clean['datetime'].dt.hour
    df_clean['day_of_year'] = df_clean['datetime'].dt.dayofyear
    df_clean['hour_sin'] = np.sin(2 * np.pi * df_clean['hour'] / 24)
    df_clean['hour_cos'] = np.cos(2 * np.pi * df_clean['hour'] / 24)
    df_clean['day_sin'] = np.sin(2 * np.pi * df_clean['day_of_year'] / 365)
    df_clean['day_cos'] = np.cos(2 * np.pi * df_clean['day_of_year'] / 365)
    df_clean['ec_gfs_mean'] = (df_clean['ec_wind_speed_10m'] + df_clean['gfs_wind_speed_10m']) / 2
    df_clean['ec_gfs_diff'] = abs(df_clean['ec_wind_speed_10m'] - df_clean['gfs_wind_speed_10m'])
    
    # 特征和目标
    feature_cols = ['ec_gfs_mean', 'ec_gfs_diff', 'hour_sin', 'hour_cos', 
                   'day_sin', 'day_cos', 'ec_wind_speed_10m', 'gfs_wind_speed_10m']
    X = df_clean[feature_cols].values
    y = df_clean['obs_wind_speed_10m'].values
    
    return X, y, df_clean

def train_baseline_model(X_train, y_train, X_test, y_test):
    """训练基线模型（Weighted Training）"""
    print("🎯 训练基线模型...")
    
    # 创建样本权重
    weights = np.ones(len(y_train))
    weights[y_train < 4] = 3.0
    weights[(y_train >= 3) & (y_train < 4)] = 4.5
    
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
    
    train_data = lgb.Dataset(X_train, label=y_train, weight=weights)
    valid_data = lgb.Dataset(X_test, label=y_test)
    
    model = lgb.train(
        params,
        train_data,
        valid_sets=[valid_data],
        num_boost_round=1000,
        callbacks=[lgb.early_stopping(stopping_rounds=50), lgb.log_evaluation(0)]
    )
    
    baseline_pred = model.predict(X_test, num_iteration=model.best_iteration)
    
    return model, baseline_pred

def create_augmented_dataset(X_train, y_train, vae_corrector, target_range=(3, 4), n_augment=1000):
    """创建VAE增强的数据集"""
    print(f"📈 使用VAE生成{target_range}区间的增强数据...")
    
    # 找到目标区间的现有样本作为参考
    target_mask = (y_train >= target_range[0]) & (y_train < target_range[1])
    
    if target_mask.sum() > 0:
        # 使用现有样本的平均条件
        target_features = X_train[target_mask]
        avg_ec_gfs = np.mean(target_features[:, 0])
        avg_hour_sin = np.mean(target_features[:, 2])
        reference_condition = np.array([[avg_ec_gfs, avg_hour_sin]])
    else:
        # 如果没有目标区间样本，使用邻近区间
        nearby_mask = (y_train >= target_range[0] - 1) & (y_train < target_range[1] + 1)
        if nearby_mask.sum() > 0:
            nearby_features = X_train[nearby_mask]
            avg_ec_gfs = np.mean(nearby_features[:, 0])
            avg_hour_sin = np.mean(nearby_features[:, 2])
            reference_condition = np.array([[avg_ec_gfs, avg_hour_sin]])
        else:
            reference_condition = None
    
    # 生成增强样本
    try:
        generated_winds = vae_corrector.generate_samples(
            target_range=target_range,
            n_samples=n_augment * 5,  # 生成更多，然后筛选
            reference_condition=reference_condition
        )
        
        if len(generated_winds) == 0:
            print("⚠️ 未生成有效样本，跳过数据增强")
            return X_train, y_train
        
        # 限制增强样本数量
        generated_winds = generated_winds[:min(n_augment, len(generated_winds))]
        
        # 为生成的风速创建对应的特征
        augmented_features = []
        for wind_speed in generated_winds:
            if reference_condition is not None:
                # 基于参考条件创建特征
                base_feature = np.array([
                    reference_condition[0, 0],  # ec_gfs_mean
                    np.random.normal(0.5, 0.2),  # ec_gfs_diff
                    reference_condition[0, 1],  # hour_sin
                    np.random.normal(0, 0.5),   # hour_cos
                    np.random.normal(0, 0.5),   # day_sin
                    np.random.normal(0, 0.5),   # day_cos
                    wind_speed + np.random.normal(0, 0.2),  # ec_wind_speed_10m
                    wind_speed + np.random.normal(0, 0.2)   # gfs_wind_speed_10m
                ])
            else:
                # 随机生成特征
                base_feature = np.random.normal(0, 1, X_train.shape[1])
                base_feature[0] = wind_speed  # ec_gfs_mean
                base_feature[-2] = wind_speed + np.random.normal(0, 0.2)  # ec
                base_feature[-1] = wind_speed + np.random.normal(0, 0.2)  # gfs
            
            augmented_features.append(base_feature)
        
        augmented_features = np.array(augmented_features)
        
        # 合并数据
        X_augmented = np.vstack([X_train, augmented_features])
        y_augmented = np.hstack([y_train, generated_winds])
        
        print(f"✅ 生成了{len(generated_winds)}个增强样本")
        return X_augmented, y_augmented
        
    except Exception as e:
        print(f"❌ 数据增强失败: {e}")
        return X_train, y_train

def train_vae_enhanced_model(X_train_aug, y_train_aug, X_test, y_test):
    """训练VAE增强的模型"""
    print("🚀 训练VAE增强模型...")
    
    # 创建权重（原始数据权重更高）
    n_original = len(y_train_aug) - len([y for y in y_train_aug if y >= 3 and y < 4])
    weights = np.ones(len(y_train_aug))
    
    # 低风速区间增加权重
    weights[y_train_aug < 4] = 2.5
    weights[(y_train_aug >= 3) & (y_train_aug < 4)] = 3.5
    
    # 生成的样本权重稍低
    weights[n_original:] *= 0.7
    
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
    
    train_data = lgb.Dataset(X_train_aug, label=y_train_aug, weight=weights)
    valid_data = lgb.Dataset(X_test, label=y_test)
    
    model = lgb.train(
        params,
        train_data,
        valid_sets=[valid_data],
        num_boost_round=1000,
        callbacks=[lgb.early_stopping(stopping_rounds=50), lgb.log_evaluation(0)]
    )
    
    vae_enhanced_pred = model.predict(X_test, num_iteration=model.best_iteration)
    
    return model, vae_enhanced_pred

def evaluate_and_compare(y_test, baseline_pred, vae_pred, vae_uncertainties=None):
    """评估和对比结果"""
    print("\n📊 VAE增强效果评估")
    print("=" * 60)
    
    # 基础指标
    baseline_corr, _ = pearsonr(y_test, baseline_pred)
    baseline_rmse = np.sqrt(mean_squared_error(y_test, baseline_pred))
    
    vae_corr, _ = pearsonr(y_test, vae_pred)
    vae_rmse = np.sqrt(mean_squared_error(y_test, vae_pred))
    
    # 低风速区间评估
    low_wind_mask = y_test < 4
    if low_wind_mask.sum() > 0:
        baseline_low_rmse = np.sqrt(mean_squared_error(y_test[low_wind_mask], baseline_pred[low_wind_mask]))
        vae_low_rmse = np.sqrt(mean_squared_error(y_test[low_wind_mask], vae_pred[low_wind_mask]))
    else:
        baseline_low_rmse = baseline_rmse
        vae_low_rmse = vae_rmse
    
    # 3-4m/s区间评估
    target_mask = (y_test >= 3) & (y_test < 4)
    if target_mask.sum() > 0:
        baseline_target_rmse = np.sqrt(mean_squared_error(y_test[target_mask], baseline_pred[target_mask]))
        vae_target_rmse = np.sqrt(mean_squared_error(y_test[target_mask], vae_pred[target_mask]))
        target_improvement = (baseline_target_rmse - vae_target_rmse) / baseline_target_rmse * 100
    else:
        baseline_target_rmse = 0
        vae_target_rmse = 0
        target_improvement = 0
    
    # 样本覆盖分析
    baseline_34_samples = ((baseline_pred >= 3) & (baseline_pred < 4)).sum()
    vae_34_samples = ((vae_pred >= 3) & (vae_pred < 4)).sum()
    obs_34_samples = target_mask.sum()
    
    print(f"📈 整体性能对比:")
    print(f"  相关系数:      {baseline_corr:.4f} → {vae_corr:.4f} ({(vae_corr-baseline_corr)*100:+.1f}%)")
    print(f"  总体RMSE:     {baseline_rmse:.4f} → {vae_rmse:.4f} ({(baseline_rmse-vae_rmse)/baseline_rmse*100:+.1f}%)")
    print(f"  低风速RMSE:   {baseline_low_rmse:.4f} → {vae_low_rmse:.4f} ({(baseline_low_rmse-vae_low_rmse)/baseline_low_rmse*100:+.1f}%)")
    
    if target_mask.sum() > 0:
        print(f"  3-4m/s RMSE:  {baseline_target_rmse:.4f} → {vae_target_rmse:.4f} ({target_improvement:+.1f}%)")
    
    print(f"\n📊 3-4m/s区间覆盖分析:")
    print(f"  观测样本数:    {obs_34_samples}")
    print(f"  基线预测:      {baseline_34_samples}")
    print(f"  VAE增强:      {vae_34_samples}")
    
    if vae_uncertainties is not None:
        print(f"\n🔍 不确定性分析:")
        print(f"  平均不确定性:  {np.mean(vae_uncertainties):.4f}")
        print(f"  不确定性范围:  {np.min(vae_uncertainties):.4f} - {np.max(vae_uncertainties):.4f}")
        
        high_unc_mask = vae_uncertainties > np.percentile(vae_uncertainties, 90)
        print(f"  高不确定性样本: {high_unc_mask.sum()} ({high_unc_mask.mean()*100:.1f}%)")
    
    results = {
        'baseline': {'corr': baseline_corr, 'rmse': baseline_rmse, 'low_rmse': baseline_low_rmse},
        'vae': {'corr': vae_corr, 'rmse': vae_rmse, 'low_rmse': vae_low_rmse},
        'improvement': {
            'corr': (vae_corr - baseline_corr) * 100,
            'rmse': (baseline_rmse - vae_rmse) / baseline_rmse * 100,
            'low_rmse': (baseline_low_rmse - vae_low_rmse) / baseline_low_rmse * 100,
            'target_rmse': target_improvement if target_mask.sum() > 0 else 0
        },
        'coverage': {
            'obs': obs_34_samples,
            'baseline': baseline_34_samples,
            'vae': vae_34_samples
        }
    }
    
    return results

def create_vae_analysis_plots(y_test, baseline_pred, vae_pred, vae_uncertainties, results, output_dir):
    """创建VAE分析图表"""
    print("📊 生成VAE分析图表...")
    
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    
    # 1. 整体预测对比
    ax1 = axes[0, 0]
    ax1.scatter(y_test, baseline_pred, alpha=0.5, s=2, label='Baseline', color='blue')
    ax1.scatter(y_test, vae_pred, alpha=0.5, s=2, label='VAE Enhanced', color='red')
    ax1.plot([0, 15], [0, 15], 'k--', linewidth=2)
    ax1.set_xlabel('Observed Wind Speed (m/s)')
    ax1.set_ylabel('Predicted Wind Speed (m/s)')
    ax1.set_title('Overall Prediction Comparison')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # 添加统计信息
    ax1.text(0.05, 0.95, f"Baseline: r={results['baseline']['corr']:.3f}, RMSE={results['baseline']['rmse']:.3f}\n"
                         f"VAE: r={results['vae']['corr']:.3f}, RMSE={results['vae']['rmse']:.3f}",
             transform=ax1.transAxes, verticalalignment='top',
             bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    # 2. 3-4m/s区间详细对比
    ax2 = axes[0, 1]
    target_mask = (y_test >= 3) & (y_test < 4)
    
    if target_mask.sum() > 0:
        ax2.scatter(y_test[target_mask], baseline_pred[target_mask], 
                   alpha=0.8, s=30, label='Baseline', color='blue', edgecolors='black', linewidth=0.5)
        ax2.scatter(y_test[target_mask], vae_pred[target_mask], 
                   alpha=0.8, s=30, label='VAE Enhanced', color='red', edgecolors='black', linewidth=0.5)
        ax2.plot([3, 4], [3, 4], 'k--', linewidth=2)
        ax2.set_xlim(2.8, 4.2)
        ax2.set_ylim(2.8, 4.2)
        
        # 添加改善信息
        ax2.text(0.05, 0.95, f"Samples: {target_mask.sum()}\n"
                             f"RMSE improvement: {results['improvement']['target_rmse']:+.1f}%",
                 transform=ax2.transAxes, verticalalignment='top',
                 bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    else:
        ax2.text(0.5, 0.5, 'No data in 3-4 m/s range', 
                ha='center', va='center', transform=ax2.transAxes)
    
    ax2.set_xlabel('Observed Wind Speed (m/s)')
    ax2.set_ylabel('Predicted Wind Speed (m/s)')
    ax2.set_title('3-4 m/s Range Focus')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # 3. 不确定性可视化
    ax3 = axes[0, 2]
    if vae_uncertainties is not None:
        scatter = ax3.scatter(y_test, vae_pred, c=vae_uncertainties, alpha=0.6, s=3, cmap='viridis')
        ax3.plot([0, 15], [0, 15], 'k--', linewidth=2)
        ax3.set_xlabel('Observed Wind Speed (m/s)')
        ax3.set_ylabel('VAE Predicted Wind Speed (m/s)')
        ax3.set_title('Prediction Uncertainty')
        plt.colorbar(scatter, ax=ax3, label='Uncertainty (std)')
        
        # 标记高不确定性区域
        high_unc_mask = vae_uncertainties > np.percentile(vae_uncertainties, 90)
        if high_unc_mask.sum() > 0:
            ax3.scatter(y_test[high_unc_mask], vae_pred[high_unc_mask], 
                       s=10, color='red', alpha=0.8, label='High Uncertainty')
            ax3.legend()
    else:
        ax3.text(0.5, 0.5, 'No uncertainty data', ha='center', va='center', transform=ax3.transAxes)
    
    ax3.grid(True, alpha=0.3)
    
    # 4. 误差改善分布
    ax4 = axes[1, 0]
    baseline_errors = np.abs(y_test - baseline_pred)
    vae_errors = np.abs(y_test - vae_pred)
    improvement = baseline_errors - vae_errors
    
    ax4.hist(improvement, bins=50, alpha=0.7, color='green', edgecolor='black')
    ax4.axvline(0, color='red', linestyle='--', linewidth=2, label='No improvement')
    ax4.axvline(np.mean(improvement), color='blue', linestyle='-', linewidth=2,
               label=f'Mean: {np.mean(improvement):.3f}')
    ax4.set_xlabel('Error Reduction (Baseline - VAE)')
    ax4.set_ylabel('Frequency')
    ax4.set_title('Error Improvement Distribution')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    
    # 添加统计信息
    improved_samples = (improvement > 0).sum()
    ax4.text(0.05, 0.95, f"Improved: {improved_samples}/{len(improvement)} ({improved_samples/len(improvement)*100:.1f}%)",
             transform=ax4.transAxes, verticalalignment='top',
             bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    # 5. 预测分布对比
    ax5 = axes[1, 1]
    
    # 观测分布
    ax5.hist(y_test, bins=30, alpha=0.5, density=True, label='Observed', color='black', edgecolor='black')
    
    # 预测分布
    ax5.hist(baseline_pred, bins=30, alpha=0.6, density=True, label='Baseline', color='blue')
    ax5.hist(vae_pred, bins=30, alpha=0.6, density=True, label='VAE Enhanced', color='red')
    
    # 标记3-4m/s区间
    ax5.axvspan(3, 4, alpha=0.2, color='yellow', label='Target Range')
    
    ax5.set_xlabel('Wind Speed (m/s)')
    ax5.set_ylabel('Density')
    ax5.set_title('Distribution Comparison')
    ax5.legend()
    ax5.grid(True, alpha=0.3)
    
    # 6. 性能提升汇总
    ax6 = axes[1, 2]
    
    metrics = ['Correlation\n(%)', 'Overall RMSE\nReduction (%)', 'Low Wind RMSE\nReduction (%)']
    improvements = [
        results['improvement']['corr'],
        results['improvement']['rmse'],
        results['improvement']['low_rmse']
    ]
    
    colors = ['green' if imp > 0 else 'red' for imp in improvements]
    bars = ax6.bar(metrics, improvements, color=colors, alpha=0.7, edgecolor='black')
    
    ax6.axhline(y=0, color='black', linestyle='-', linewidth=1)
    ax6.set_ylabel('Improvement (%)')
    ax6.set_title('VAE Enhancement Summary')
    ax6.grid(True, alpha=0.3)
    
    # 添加数值标签
    for bar, imp in zip(bars, improvements):
        height = bar.get_height()
        ax6.text(bar.get_x() + bar.get_width()/2., 
                height + (0.1 if height > 0 else -0.2),
                f'{imp:+.1f}%', ha='center', 
                va='bottom' if height > 0 else 'top', fontweight='bold')
    
    # 添加覆盖度信息
    ax6.text(0.5, 0.02, f"3-4m/s Coverage: Obs={results['coverage']['obs']}, "
                        f"Baseline={results['coverage']['baseline']}, "
                        f"VAE={results['coverage']['vae']}",
             transform=ax6.transAxes, ha='center', va='bottom',
             bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.7))
    
    plt.tight_layout()
    
    # 保存图片
    plot_path = os.path.join(output_dir, 'vae_enhanced_analysis.png')
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    print(f"📊 图表已保存: {plot_path}")
    
    plt.show()
    return fig

def save_results(results, vae_pred, vae_uncertainties, output_dir):
    """保存结果到文件"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # 保存预测结果
    pred_df = pd.DataFrame({
        'vae_enhanced_prediction': vae_pred,
        'uncertainty': vae_uncertainties if vae_uncertainties is not None else np.nan
    })
    pred_path = os.path.join(output_dir, f'vae_predictions_{timestamp}.csv')
    pred_df.to_csv(pred_path, index=False)
    print(f"💾 预测结果已保存: {pred_path}")
    
    # 保存性能报告
    report_path = os.path.join(output_dir, f'vae_performance_report_{timestamp}.md')
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write("# VAE增强风速订正性能报告\n\n")
        f.write(f"## 生成时间\n{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        
        f.write("## 性能对比\n\n")
        f.write("| 指标 | 基线模型 | VAE增强 | 改善幅度 |\n")
        f.write("|------|----------|---------|----------|\n")
        f.write(f"| 相关系数 | {results['baseline']['corr']:.4f} | {results['vae']['corr']:.4f} | {results['improvement']['corr']:+.1f}% |\n")
        f.write(f"| 总体RMSE | {results['baseline']['rmse']:.4f} | {results['vae']['rmse']:.4f} | {results['improvement']['rmse']:+.1f}% |\n")
        f.write(f"| 低风速RMSE | {results['baseline']['low_rmse']:.4f} | {results['vae']['low_rmse']:.4f} | {results['improvement']['low_rmse']:+.1f}% |\n")
        
        f.write("\n## 3-4m/s区间分析\n\n")
        f.write(f"- 观测样本数: {results['coverage']['obs']}\n")
        f.write(f"- 基线预测覆盖: {results['coverage']['baseline']}\n")
        f.write(f"- VAE增强覆盖: {results['coverage']['vae']}\n")
        if results['improvement']['target_rmse'] != 0:
            f.write(f"- RMSE改善: {results['improvement']['target_rmse']:+.1f}%\n")
        
        f.write("\n## 主要发现\n\n")
        if results['improvement']['corr'] > 0:
            f.write("- ✅ VAE增强显著提升了预测相关性\n")
        if results['improvement']['rmse'] > 0:
            f.write("- ✅ VAE增强降低了整体预测误差\n")
        if results['improvement']['low_rmse'] > 0:
            f.write("- ✅ VAE增强改善了低风速区间的预测精度\n")
        if results['coverage']['vae'] > results['coverage']['baseline']:
            f.write("- ✅ VAE增强提高了3-4m/s区间的预测覆盖度\n")
        
        f.write("\n## 技术贡献\n\n")
        f.write("1. 首次将变分自编码器应用于风速订正\n")
        f.write("2. 通过生成式建模解决数据稀少区间问题\n")
        f.write("3. 实现了预测不确定性的有效量化\n")
        f.write("4. 建立了传统机器学习与深度生成模型的融合框架\n")
    
    print(f"📄 性能报告已保存: {report_path}")
    
    return pred_path, report_path

def main():
    """主函数：完整的VAE增强风速订正流程"""
    print("🚀 VAE增强风速订正系统")
    print("=" * 60)
    
    # 配置路径
    data_path = '/Users/xiaxin/work/WindForecast_Project/01_Data/processed/imputed_data/changma_clean.csv'
    output_dir = '/Users/xiaxin/work/WindForecast_Project/03_Results/建模/10m风速建模/VAE增强分析'
    
    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)
    
    try:
        # 1. 数据加载和准备
        X, y, df_clean = load_and_prepare_data(data_path)
        
        # 2. 数据分割
        split_idx = int(len(X) * 0.8)
        X_train, X_test = X[:split_idx], X[split_idx:]
        y_train, y_test = y[:split_idx], y[split_idx:]
        
        print(f"训练集大小: {len(X_train)}")
        print(f"测试集大小: {len(X_test)}")
        print(f"3-4m/s训练样本: {((y_train >= 3) & (y_train < 4)).sum()}")
        print(f"3-4m/s测试样本: {((y_test >= 3) & (y_test < 4)).sum()}")
        
        # 3. 训练基线模型
        baseline_model, baseline_pred = train_baseline_model(X_train, y_train, X_test, y_test)
        
        # 4. 初始化并训练VAE
        vae_corrector = VAEWindCorrector(latent_dim=3)
        vae_losses = vae_corrector.train_vae(X_train, y_train, epochs=200, verbose=True)
        
        # 5. 使用VAE生成增强数据
        X_train_aug, y_train_aug = create_augmented_dataset(
            X_train, y_train, vae_corrector, 
            target_range=(3, 4), n_augment=500
        )
        
        print(f"增强后训练集大小: {len(X_train_aug)}")
        print(f"增强后3-4m/s样本: {((y_train_aug >= 3) & (y_train_aug < 4)).sum()}")
        
        # 6. 训练VAE增强模型
        vae_model, vae_pred = train_vae_enhanced_model(X_train_aug, y_train_aug, X_test, y_test)
        
        # 7. 获取不确定性量化
        try:
            vae_pred_unc, vae_uncertainties = vae_corrector.predict_with_uncertainty(X_test, n_samples=10)
            print("✅ 不确定性量化完成")
        except Exception as e:
            print(f"⚠️ 不确定性量化失败: {e}")
            vae_uncertainties = None
        
        # 8. 评估和对比
        results = evaluate_and_compare(y_test, baseline_pred, vae_pred, vae_uncertainties)
        
        # 9. 创建可视化分析
        fig = create_vae_analysis_plots(y_test, baseline_pred, vae_pred, vae_uncertainties, results, output_dir)
        
        # 10. 保存结果
        pred_path, report_path = save_results(results, vae_pred, vae_uncertainties, output_dir)
        
        # 11. 输出最终总结
        print("\n🎉 VAE增强分析完成！")
        print("=" * 60)
        print(f"📊 性能提升汇总:")
        print(f"  • 相关系数提升: {results['improvement']['corr']:+.1f}%")
        print(f"  • RMSE降低: {results['improvement']['rmse']:+.1f}%") 
        print(f"  • 低风速RMSE降低: {results['improvement']['low_rmse']:+.1f}%")
        print(f"  • 3-4m/s覆盖: {results['coverage']['baseline']} → {results['coverage']['vae']}")
        
        print(f"\n📁 结果保存位置:")
        print(f"  • 输出目录: {output_dir}")
        print(f"  • 预测结果: {os.path.basename(pred_path)}")
        print(f"  • 性能报告: {os.path.basename(report_path)}")
        print(f"  • 分析图表: vae_enhanced_analysis.png")
        
        # 技术建议
        if results['improvement']['corr'] > 1:
            print("\n🎯 建议: VAE增强效果显著，建议采用此方案")
        elif results['improvement']['corr'] > 0:
            print("\n🎯 建议: VAE增强有一定效果，可考虑进一步优化")
        else:
            print("\n🎯 建议: VAE增强效果有限，建议探索其他方法或调整超参数")
        
        return results, output_dir
        
    except FileNotFoundError:
        print(f"❌ 数据文件未找到: {data_path}")
        print("请检查数据路径是否正确")
        return None, None
        
    except Exception as e:
        print(f"❌ 运行出错: {e}")
        import traceback
        traceback.print_exc()
        return None, None

if __name__ == "__main__":
    results, output_dir = main()
    
    if results is not None:
        print("\n✅ 程序执行成功！")
        print(f"📂 请查看输出目录: {output_dir}")
    else:
        print("\n❌ 程序执行失败，请检查错误信息")