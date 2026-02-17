#!/usr/bin/env python3
"""
步骤1: 自适应CEEMDAN分解所有变量并保存结果
使用与之前分析相同的自适应方法，确保结果一致性
分解：10m, 30m, 50m, 70m 风速 + 功率
"""

if __name__ == '__main__':
    import pandas as pd
    import numpy as np
    from scipy.signal import argrelextrema
    from scipy.interpolate import interp1d
    import os
    import warnings
    from datetime import datetime
    warnings.filterwarnings('ignore')
    
    # 固定随机种子
    np.random.seed(42)
    
    # ========== 自适应CEEMDAN核心函数（与之前完全相同） ==========
    
    def extract_imf_sifting_adaptive(signal, mode, max_sifts=20):
        """
        自适应筛选过程 - 根据IMF阶数调整参数
        """
        h = signal.copy()
        
        # 根据IMF阶数调整停止阈值
        if mode <= 2:
            stop_threshold = 0.001  # 前几个IMF要求更严格
        else:
            stop_threshold = 0.005  # 后面的IMF可以放松一些
        
        for sift in range(max_sifts):
            try:
                max_indices = argrelextrema(h, np.greater)[0]
                min_indices = argrelextrema(h, np.less)[0]
                
                # 需要足够的极值点
                if len(max_indices) < 2 or len(min_indices) < 2:
                    break
                
                # 极值点分布检查
                if len(h) > 100:  # 对于长序列，检查极值点分布
                    extrema_spacing = np.diff(np.sort(np.concatenate([max_indices, min_indices])))
                    if np.mean(extrema_spacing) > len(h) / 4:  # 极值点太稀疏
                        break
                
                try:
                    # 包络插值
                    if len(max_indices) >= 2:
                        f_max = interp1d(max_indices, h[max_indices], 
                                       kind='cubic', fill_value='extrapolate',
                                       bounds_error=False)
                        upper_env = f_max(np.arange(len(h)))
                        
                        if np.any(np.isnan(upper_env)) or np.any(np.isinf(upper_env)):
                            upper_env = np.full_like(h, np.max(h))
                    else:
                        upper_env = np.full_like(h, np.max(h))
                    
                    if len(min_indices) >= 2:
                        f_min = interp1d(min_indices, h[min_indices], 
                                       kind='cubic', fill_value='extrapolate',
                                       bounds_error=False)
                        lower_env = f_min(np.arange(len(h)))
                        
                        if np.any(np.isnan(lower_env)) or np.any(np.isinf(lower_env)):
                            lower_env = np.full_like(h, np.min(h))
                    else:
                        lower_env = np.full_like(h, np.min(h))
                    
                except Exception:
                    upper_env = np.full_like(h, np.max(h))
                    lower_env = np.full_like(h, np.min(h))
                
                mean_env = (upper_env + lower_env) / 2
                h_new = h - mean_env
                
                # 自适应停止准则
                if np.std(h_new - h) < stop_threshold * np.std(h):
                    break
                
                h = h_new
                
            except Exception as e:
                break
        
        return h
    
    def improved_simple_ceemdan_adaptive(data, ensemble_size=100, noise_std=0.005):
        """
        自适应CEEMDAN实现 - 根据数据特性动态确定IMF数量
        与之前的分析方法完全相同
        """
        print(f"  Using adaptive CEEMDAN (ensemble_size={ensemble_size})")
        
        imfs = []
        residue = data.copy().astype(float)
        original_data = data.copy().astype(float)
        
        # Adaptive noise level
        base_noise_std = noise_std * np.std(original_data)
        original_energy = np.sum(original_data ** 2)
        
        # 动态最大IMF数量：基于数据长度
        max_imfs = min(int(np.log2(len(data))), 15)  # 更合理的上限
        print(f"    Maximum IMFs for this data: {max_imfs}")
        
        for mode in range(max_imfs):
            print(f"    Processing IMF {mode + 1}...", end=' ')
            
            # 更严格的停止条件
            if len(residue) < 20:
                print(f"Stopping: insufficient data length")
                break
                
            if np.std(residue) < 1e-8 * np.std(original_data):
                print(f"Stopping: negligible residue variation")
                break
            
            # 检查残差的能量
            residue_energy = np.sum(residue ** 2)
            if residue_energy < 0.005 * original_energy:
                print(f"Stopping: residue energy too low")
                break
            
            # 检查极值点数量
            try:
                max_indices = argrelextrema(residue, np.greater)[0]
                min_indices = argrelextrema(residue, np.less)[0]
                total_extrema = len(max_indices) + len(min_indices)
                
                if total_extrema < 6:
                    print(f"Stopping: insufficient extrema ({total_extrema})")
                    break
                    
                if len(max_indices) <= 2 or len(min_indices) <= 2:
                    print(f"Stopping: residue becoming monotonic")
                    break
            except:
                print(f"Stopping: extrema detection failed")
                break
            
            # CEEMDAN ensemble processing
            ensemble_imfs = []
            
            for ens in range(ensemble_size):
                np.random.seed(42 + mode * ensemble_size + ens)
                
                if mode == 0:
                    noise = base_noise_std * np.random.randn(len(original_data))
                    noisy_signal = original_data + noise
                else:
                    mode_noise_std = base_noise_std / (2 ** (mode - 1))
                    noise = mode_noise_std * np.random.randn(len(residue))
                    noisy_signal = residue + noise
                
                try:
                    imf = extract_imf_sifting_adaptive(noisy_signal, mode)
                    if len(imf) == len(residue) and np.std(imf) > 1e-10:
                        ensemble_imfs.append(imf)
                except Exception as e:
                    continue
            
            if len(ensemble_imfs) < ensemble_size // 3:
                print(f"Stopping: insufficient ensemble success ({len(ensemble_imfs)}/{ensemble_size})")
                break
            
            # Average ensemble
            imf = np.mean(ensemble_imfs, axis=0)
            
            # IMF质量检查
            imf_energy = np.sum(imf ** 2)
            if imf_energy < 1e-6 * original_energy:
                print(f"Stopping: IMF energy too low")
                break
            
            # 检查IMF是否有意义
            imf_std = np.std(imf)
            if imf_std < 1e-8 * np.std(original_data):
                print(f"Stopping: IMF variation negligible")
                break
            
            imfs.append(imf)
            residue = residue - imf
            
            var_ratio = np.var(imf) / np.var(original_data)
            print(f"OK (var_ratio={var_ratio:.4f})")
            
            # 连续IMF检查
            if len(imfs) >= 2:
                correlation = np.corrcoef(imfs[-1], imfs[-2])[0, 1]
                if abs(correlation) > 0.95:
                    print(f"    Warning: High correlation with previous IMF ({correlation:.3f})")
                    if len(imfs) >= 4:
                        print(f"    Stopping: potential over-decomposition detected")
                        break
        
        # Add final residue as trend
        if len(residue) > 0:
            final_residue_energy = np.sum(residue ** 2)
            if final_residue_energy > 1e-10 * original_energy:
                imfs.append(residue)
                print(f"    Added final residue as trend")
        
        print(f"  Completed: {len(imfs)} IMFs extracted (adaptive)")
        
        # 质量检查
        if len(imfs) > 0:
            reconstructed = np.sum(imfs, axis=0)
            reconstruction_error = np.mean((original_data - reconstructed) ** 2)
            print(f"  Reconstruction RMSE: {np.sqrt(reconstruction_error):.8f}")
        
        return np.array(imfs)
    
    # ========== 主程序 ==========
    
    # 路径
    data_path = '/Users/xiaxin/work/WindForecast_Project/01_Data/processed/matched_data/changma_matched.csv'
    output_dir = '/Users/xiaxin/work/WindForecast_Project/03_Results/re-plot-figures/figure-2/'
    
    print("=" * 70)
    print("Adaptive CEEMDAN Decomposition - All Heights + Power")
    print("Using the SAME method as previous analysis")
    print("=" * 70)
    print(f"Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # 1. 读取数据
    print("\n[1/5] Loading data...")
    df = pd.read_csv(data_path)
    
    if 'datetime' in df.columns:
        time_col = 'datetime'
    elif 'timestamp' in df.columns:
        time_col = 'timestamp'
    else:
        print("ERROR: No time column found")
        exit(1)
    
    df[time_col] = pd.to_datetime(df[time_col])
    df = df.sort_values(time_col).reset_index(drop=True)
    
    # 时间分辨率
    dt = (df[time_col].iloc[1] - df[time_col].iloc[0]).total_seconds() / 3600.0
    print(f"  Resolution: {dt*60:.1f} min ({dt:.4f} hours)")
    
    # 2. 提取数据
    print("\n[2/5] Extracting wind speed and power...")
    
    heights = [10, 30, 50, 70]
    ws_data = {}
    ws_clean = {}
    
    for h in heights:
        col = f'obs_wind_speed_{h}m'
        if col in df.columns:
            ws_data[h] = df[col].values
            print(f"  ✓ Found {col}")
        else:
            print(f"  ✗ Missing {col}")
            ws_data[h] = None
    
    # 功率
    if 'obs_power' in df.columns:
        power_data = df['obs_power'].values
        print(f"  ✓ Found obs_power")
    else:
        print(f"  ✗ Missing obs_power")
        power_data = None
    
    # 找到所有高度都有效的数据点
    print("\n  Finding valid data points...")
    valid = np.ones(len(df), dtype=bool)
    for h in heights:
        if ws_data[h] is not None:
            valid &= ~np.isnan(ws_data[h])
    
    if power_data is not None:
        valid &= ~np.isnan(power_data)
    
    # 提取有效数据
    for h in heights:
        if ws_data[h] is not None:
            ws_clean[h] = ws_data[h][valid]
    
    if power_data is not None:
        power_clean = power_data[valid]
    else:
        power_clean = None
    
    print(f"  Total samples: {len(df)}")
    print(f"  Valid samples: {valid.sum()} ({valid.sum()/len(df)*100:.1f}%)")
    
    # 3. 风向掩码
    print("\n[3/5] Creating wind direction masks...")
    
    def strict_direction_mask(df, heights, direction_range):
        min_deg, max_deg = direction_range
        mask = np.ones(len(df), dtype=bool)
        for h in heights:
            wd = df[f'obs_wind_direction_{h}m'].values
            if min_deg < max_deg:
                mask &= (wd >= min_deg) & (wd <= max_deg)
            else:
                mask &= (wd >= min_deg) | (wd <= max_deg)
        return mask
    
    mask_west_full = strict_direction_mask(df, heights, (225, 315))
    mask_east_full = strict_direction_mask(df, heights, (45, 135))
    
    mask_west = mask_west_full[valid]
    mask_east = mask_east_full[valid]
    
    print(f"  West wind: {mask_west.sum()} ({mask_west.sum()/len(mask_west)*100:.1f}%)")
    print(f"  East wind: {mask_east.sum()} ({mask_east.sum()/len(mask_east)*100:.1f}%)")
    
    # 4. 自适应CEEMDAN分解
    print("\n[4/5] Adaptive CEEMDAN decomposition...")
    print(f"  Parameters: ensemble_size=100, noise_std=0.005")
    print(f"  Started at: {datetime.now().strftime('%H:%M:%S')}")
    
    total_start = datetime.now()
    
    # 存储所有分解结果
    imfs_all = {}
    
    # 分解所有高度的风速
    for h in heights:
        if h not in ws_clean:
            continue
            
        print(f"\n  [{h}m] Decomposing wind speed...")
        print(f"    Data range: {ws_clean[h].min():.3f} to {ws_clean[h].max():.3f}")
        print(f"    Data std: {ws_clean[h].std():.6f}")
        
        start_time = datetime.now()
        
        try:
            imfs = improved_simple_ceemdan_adaptive(ws_clean[h], 
                                                   ensemble_size=100, 
                                                   noise_std=0.005)
            imfs_all[f'ws_{h}m'] = imfs
            elapsed = (datetime.now() - start_time).total_seconds() / 60
            print(f"    ✓ {len(imfs)} IMFs | Time: {elapsed:.1f} min")
        except Exception as e:
            print(f"    ✗ Decomposition failed: {e}")
            import traceback
            traceback.print_exc()
    
    # 分解功率
    if power_clean is not None:
        print(f"\n  [Power] Decomposing...")
        print(f"    Data range: {power_clean.min():.3f} to {power_clean.max():.3f}")
        print(f"    Data std: {power_clean.std():.6f}")
        
        start_time = datetime.now()
        
        try:
            imfs = improved_simple_ceemdan_adaptive(power_clean, 
                                                   ensemble_size=100, 
                                                   noise_std=0.005)
            imfs_all['power'] = imfs
            elapsed = (datetime.now() - start_time).total_seconds() / 60
            print(f"    ✓ {len(imfs)} IMFs | Time: {elapsed:.1f} min")
        except Exception as e:
            print(f"    ✗ Decomposition failed: {e}")
    
    total_elapsed = (datetime.now() - total_start).total_seconds() / 60
    print(f"\n  Total decomposition time: {total_elapsed:.1f} min")
    print(f"  Variables decomposed: {len(imfs_all)}")
    
    # 5. 保存结果
    print("\n[5/5] Saving results...")
    os.makedirs(output_dir, exist_ok=True)
    
    # 保存为相同的文件名，替代之前的结果
    save_path = os.path.join(output_dir, 'ceemdan_results_full.npz')
    
    # 构建保存字典
    save_dict = {
        'mask_west': mask_west,
        'mask_east': mask_east,
        'dt_hours': dt,
        'n_samples': valid.sum(),
        'timestamp': datetime.now().isoformat(),
        'method': 'Adaptive_CEEMDAN',  # 标注使用的是自适应方法
        'ensemble_size': 100,
        'noise_std': 0.005,
        'trials': 100,  # 为了兼容性，也加上这个字段
        'random_seed': 42,
        'heights': np.array(heights),
        'variables': list(imfs_all.keys()),
    }
    
    # 添加所有IMF结果
    for key, imfs in imfs_all.items():
        save_dict[f'imfs_{key}'] = np.array(imfs, dtype=object)
    
    np.savez_compressed(save_path, **save_dict)
    
    print(f"  ✓ Saved: {save_path}")
    print(f"  File size: {os.path.getsize(save_path) / 1024 / 1024:.2f} MB")
    
    # 保存摘要
    summary_path = os.path.join(output_dir, 'ceemdan_summary_full.xlsx')
    
    summary_data = {
        'Parameter': [
            'Data file',
            'Total samples',
            'Valid samples',
            'Time resolution (hours)',
            'West wind samples',
            'East wind samples',
            'Method',
            'Ensemble size',
            'Noise std',
            'Random seed',
            'Processing date',
            'Processing time (min)',
            'Results file'
        ],
        'Value': [
            data_path.split('/')[-1],
            len(df),
            valid.sum(),
            f'{dt:.4f}',
            mask_west.sum(),
            mask_east.sum(),
            'Adaptive CEEMDAN',
            100,
            0.005,
            42,
            datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            f'{total_elapsed:.1f}',
            'ceemdan_results_full.npz'
        ]
    }
    
    # 添加各变量的IMF数量
    for key, imfs in imfs_all.items():
        summary_data['Parameter'].append(f'Number of IMFs ({key})')
        summary_data['Value'].append(len(imfs))
    
    summary_df = pd.DataFrame(summary_data)
    summary_df.to_excel(summary_path, index=False)
    print(f"  ✓ Summary: {summary_path}")
    
    # 打印IMF数量对比
    print(f"\n  Saved variables and IMF counts:")
    for key in imfs_all.keys():
        print(f"    - {key}: {len(imfs_all[key])} IMFs")
    
    print("\n" + "=" * 70)
    print("✓ Adaptive CEEMDAN decomposition completed!")
    print("=" * 70)
    print("\nMethod used: Adaptive CEEMDAN (same as previous successful analysis)")
    print("Output file: ceemdan_results_full.npz (replaces previous version)")
    print("\nKey features:")
    print("  • Each variable has its natural IMF count (not forced)")
    print("  • Adaptive stopping criteria prevent over-decomposition")
    print("  • Results compatible with all existing plotting scripts")
    
    print(f"\nTotal processing time: {total_elapsed:.1f} minutes")
    print("\n✅ All plotting scripts will now use this new adaptive data!")
    print("   No need to modify any script - they all read 'ceemdan_results_full.npz'")
    print("\nNext steps:")
    print("  1. python step2_plot_from_saved.py           → Energy spectrum")
    print("  2. python step2b_correlation_plot.py         → Correlation analysis")
    print("  3. python full_ceemdan_correlation_plot.py   → Full correlation plot")