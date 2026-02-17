#!/usr/bin/env python3
"""
步骤1: CEEMDAN分解所有变量并保存结果
分解：10m, 30m, 50m, 70m 风速 + 功率
只需运行一次，结果保存为.npz文件
"""

if __name__ == '__main__':
    import pandas as pd
    import numpy as np
    from PyEMD import CEEMDAN
    import os
    import warnings
    from datetime import datetime
    warnings.filterwarnings('ignore')
    
    # 固定随机种子
    np.random.seed(42)
    
    # 路径
    data_path = '/Users/xiaxin/work/WindForecast_Project/01_Data/processed/matched_data/changma_matched.csv'
    output_dir = '/Users/xiaxin/work/WindForecast_Project/03_Results/re-plot-figures/figure-2/'
    
    print("=" * 70)
    print("CEEMDAN Decomposition - All Heights + Power")
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
    
    # 风速：所有高度
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
    if 'power' in df.columns:
        power_data = df['power'].values
        print(f"  ✓ Found power")
    else:
        print(f"  ✗ Missing power (will skip power decomposition)")
        power_data = None
    
    # 找到所有高度都有效的数据点
    print("\n  Finding valid data points...")
    valid = np.ones(len(df), dtype=bool)
    for h in heights:
        if ws_data[h] is not None:
            valid &= ~np.isnan(ws_data[h])
    
    # 如果有功率数据，也要求功率有效
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
    
    for h in heights:
        if h in ws_clean:
            print(f"    {h}m: {len(ws_clean[h])} samples")
    if power_clean is not None:
        print(f"    Power: {len(power_clean)} samples")
    
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
    
    # 4. CEEMDAN分解
    print("\n[4/5] CEEMDAN decomposition (this may take 40-90 minutes)...")
    print(f"  Started at: {datetime.now().strftime('%H:%M:%S')}")
    
    total_start = datetime.now()
    ceemdan = CEEMDAN(trials=100, epsilon=0.005)
    
    # 存储所有分解结果
    imfs_all = {}
    
    # 分解所有高度的风速
    for h in heights:
        if h not in ws_clean:
            continue
            
        print(f"\n  [{h}m] Decomposing wind speed...")
        start_time = datetime.now()
        
        try:
            imfs = ceemdan.ceemdan(ws_clean[h], max_imf=12)
            imfs_all[f'ws_{h}m'] = imfs
            elapsed = (datetime.now() - start_time).total_seconds() / 60
            print(f"    ✓ {len(imfs)} IMFs | Time: {elapsed:.1f} min")
        except Exception as e:
            print(f"    ✗ CEEMDAN failed: {e}")
            print("    Falling back to EMD...")
            from PyEMD import EMD
            emd = EMD()
            imfs = emd.emd(ws_clean[h], max_imf=12)
            imfs_all[f'ws_{h}m'] = imfs
            elapsed = (datetime.now() - start_time).total_seconds() / 60
            print(f"    ✓ {len(imfs)} IMFs (EMD) | Time: {elapsed:.1f} min")
    
    # 分解功率
    if power_clean is not None:
        print(f"\n  [Power] Decomposing...")
        start_time = datetime.now()
        
        try:
            imfs = ceemdan.ceemdan(power_clean, max_imf=12)
            imfs_all['power'] = imfs
            elapsed = (datetime.now() - start_time).total_seconds() / 60
            print(f"    ✓ {len(imfs)} IMFs | Time: {elapsed:.1f} min")
        except Exception as e:
            print(f"    ✗ CEEMDAN failed: {e}")
            print("    Falling back to EMD...")
            from PyEMD import EMD
            emd = EMD()
            imfs = emd.emd(power_clean, max_imf=12)
            imfs_all['power'] = imfs
            elapsed = (datetime.now() - start_time).total_seconds() / 60
            print(f"    ✓ {len(imfs)} IMFs (EMD) | Time: {elapsed:.1f} min")
    
    total_elapsed = (datetime.now() - total_start).total_seconds() / 60
    print(f"\n  Total decomposition time: {total_elapsed:.1f} min")
    print(f"  Variables decomposed: {len(imfs_all)}")
    
    # 5. 保存结果
    print("\n[5/5] Saving results...")
    os.makedirs(output_dir, exist_ok=True)
    
    # 保存为.npz文件（压缩的numpy二进制格式）
    save_path = os.path.join(output_dir, 'ceemdan_results_full.npz')
    
    # 构建保存字典
    save_dict = {
        # 风向掩码
        'mask_west': mask_west,
        'mask_east': mask_east,
        # 元数据
        'dt_hours': dt,
        'n_samples': valid.sum(),
        'timestamp': datetime.now().isoformat(),
        'method': 'CEEMDAN',
        'trials': 100,
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
    
    # 额外保存一个人类可读的摘要（Excel）
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
            'Ensemble trials',
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
            'CEEMDAN',
            100,
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
    
    # 打印保存的变量列表
    print(f"\n  Saved variables:")
    for key in imfs_all.keys():
        print(f"    - {key}: {len(imfs_all[key])} IMFs")
    
    print("\n" + "=" * 70)
    print("✓ CEEMDAN decomposition completed and saved!")
    print("=" * 70)
    print("\nDecomposed variables:")
    for h in heights:
        if f'ws_{h}m' in imfs_all:
            print(f"  ✓ Wind speed {h}m: {len(imfs_all[f'ws_{h}m'])} IMFs")
    if 'power' in imfs_all:
        print(f"  ✓ Power: {len(imfs_all['power'])} IMFs")
    
    print(f"\nTotal processing time: {total_elapsed:.1f} minutes")
    print("\nNext step: Run 'step2_plot_from_saved.py' to create figures")
    print("(This will be fast - only takes seconds!)")