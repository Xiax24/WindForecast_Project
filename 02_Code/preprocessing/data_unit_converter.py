import pandas as pd
import numpy as np
from pathlib import Path

def fix_wrf_units():
    """修复WRF数据的单位问题"""
    
    project_root = Path("/Users/xiaxin/work/WindForecast_Project")
    matched_data_dir = project_root / "01_Data/processed/matched_data"
    
    stations = ['changma', 'kuangqu', 'sanlijijingzi']
    
    print("🔧 开始修复WRF数据单位问题...")
    print("=" * 60)
    
    for station in stations:
        matched_file = matched_data_dir / f"{station}_matched.csv"
        
        if not matched_file.exists():
            print(f"⚠️  {station} 匹配数据文件不存在，跳过")
            continue
            
        print(f"\n处理 {station} 站点...")
        
        try:
            # 读取匹配数据
            df = pd.read_csv(matched_file, index_col=0)
            df.index = pd.to_datetime(df.index)
            
            print(f"  原始数据形状: {df.shape}")
            
            # 1. 修复温度单位（K → °C）
            temp_columns = [col for col in df.columns if 'temperature' in col and ('ec_' in col or 'gfs_' in col)]
            
            if temp_columns:
                print(f"  修复温度单位 ({len(temp_columns)} 个变量):")
                for col in temp_columns:
                    # 检查是否是开尔文温度（通常>200）
                    if df[col].notna().sum() > 0:
                        mean_temp = df[col].mean()
                        if mean_temp > 100:  # 很可能是开尔文
                            df[col] = df[col] - 273.15  # 转换为摄氏度
                            print(f"    ✅ {col}: K → °C (均值: {mean_temp:.1f}K → {df[col].mean():.1f}°C)")
                        else:
                            print(f"    ✓ {col}: 已经是°C (均值: {mean_temp:.1f}°C)")
            
            # 2. 修复密度单位
            density_columns = [col for col in df.columns if 'density' in col and ('ec_' in col or 'gfs_' in col)]
            
            if density_columns:
                print(f"  检查密度单位 ({len(density_columns)} 个变量):")
                for col in density_columns:
                    if df[col].notna().sum() > 0:
                        mean_density = df[col].mean()
                        
                        # 检查密度是否过小（可能需要转换）
                        if mean_density < 0.1:  # 正常大气密度约1.0-1.3 kg/m³
                            # 可能的转换因子
                            if 0.001 < mean_density < 0.02:  # 可能是g/cm³或其他单位
                                # 尝试不同的转换
                                conversion_factor = 1000  # 假设从g/cm³转换
                                new_density = df[col] * conversion_factor
                                new_mean = new_density.mean()
                                
                                if 0.8 < new_mean < 2.0:  # 合理的大气密度范围
                                    df[col] = new_density
                                    print(f"    ✅ {col}: ×1000 (均值: {mean_density:.6f} → {new_mean:.3f} kg/m³)")
                                else:
                                    print(f"    ⚠️  {col}: 数值异常 (均值: {mean_density:.6f}, 转换后: {new_mean:.3f})")
                            else:
                                print(f"    ⚠️  {col}: 数值过小，无法确定转换方法 (均值: {mean_density:.6f})")
                        else:
                            print(f"    ✓ {col}: 单位正常 (均值: {mean_density:.3f} kg/m³)")
            
            # 3. 验证修复结果
            print(f"  \n修复后数据验证:")
            
            # 检查温度范围
            temp_obs_cols = [col for col in df.columns if 'temperature' in col and 'obs_' in col]
            temp_wrf_cols = [col for col in df.columns if 'temperature' in col and ('ec_' in col or 'gfs_' in col)]
            
            if temp_obs_cols and temp_wrf_cols:
                obs_temp_range = (df[temp_obs_cols[0]].min(), df[temp_obs_cols[0]].max())
                wrf_temp_range = (df[temp_wrf_cols[0]].min(), df[temp_wrf_cols[0]].max())
                print(f"    温度范围 - 观测: {obs_temp_range[0]:.1f}~{obs_temp_range[1]:.1f}°C")
                print(f"    温度范围 - WRF:  {wrf_temp_range[0]:.1f}~{wrf_temp_range[1]:.1f}°C")
            
            # 检查密度范围
            density_obs_cols = [col for col in df.columns if 'density' in col and 'obs_' in col]
            density_wrf_cols = [col for col in df.columns if 'density' in col and ('ec_' in col or 'gfs_' in col)]
            
            if density_obs_cols and density_wrf_cols:
                obs_density_range = (df[density_obs_cols[0]].min(), df[density_obs_cols[0]].max())
                wrf_density_range = (df[density_wrf_cols[0]].min(), df[density_wrf_cols[0]].max())
                print(f"    密度范围 - 观测: {obs_density_range[0]:.3f}~{obs_density_range[1]:.3f} kg/m³")
                print(f"    密度范围 - WRF:  {wrf_density_range[0]:.3f}~{wrf_density_range[1]:.3f} kg/m³")
            
            # 4. 保存修复后的数据
            backup_file = matched_data_dir / f"{station}_matched_backup.csv"
            corrected_file = matched_data_dir / f"{station}_matched_corrected.csv"
            
            # 备份原始文件
            df_original = pd.read_csv(matched_file, index_col=0)
            df_original.to_csv(backup_file)
            
            # 保存修复后的文件
            df.to_csv(corrected_file)
            df.to_csv(matched_file)  # 覆盖原文件
            
            print(f"  ✅ 修复完成!")
            print(f"    原始文件备份: {backup_file.name}")
            print(f"    修复后文件: {corrected_file.name}")
            
        except Exception as e:
            print(f"  ❌ 处理 {station} 时出错: {e}")
            import traceback
            traceback.print_exc()

def validate_units():
    """验证修复后的单位是否正确"""
    
    project_root = Path("/Users/xiaxin/work/WindForecast_Project")
    matched_data_dir = project_root / "01_Data/processed/matched_data"
    
    print("\n" + "=" * 60)
    print("🔍 验证修复后的数据单位")
    print("=" * 60)
    
    stations = ['changma', 'kuangqu', 'sanlijijingzi']
    
    for station in stations:
        matched_file = matched_data_dir / f"{station}_matched.csv"
        
        if not matched_file.exists():
            continue
            
        print(f"\n{station.upper()} 站点验证:")
        print("-" * 30)
        
        try:
            df = pd.read_csv(matched_file, index_col=0)
            
            # 检查温度
            temp_cols = [col for col in df.columns if 'temperature' in col]
            if temp_cols:
                print("温度数据检查:")
                for col in temp_cols[:3]:  # 只显示前3个
                    if df[col].notna().sum() > 0:
                        mean_val = df[col].mean()
                        min_val = df[col].min()
                        max_val = df[col].max()
                        
                        # 判断是否合理
                        if -50 < mean_val < 50 and -50 < min_val < 60 and -30 < max_val < 60:
                            status = "✅ 正常"
                        else:
                            status = "⚠️  异常"
                        
                        print(f"  {col}: {min_val:.1f}~{max_val:.1f}°C (均值:{mean_val:.1f}) {status}")
            
            # 检查密度
            density_cols = [col for col in df.columns if 'density' in col]
            if density_cols:
                print("密度数据检查:")
                for col in density_cols[:3]:  # 只显示前3个
                    if df[col].notna().sum() > 0:
                        mean_val = df[col].mean()
                        min_val = df[col].min()
                        max_val = df[col].max()
                        
                        # 判断是否合理（大气密度通常0.8-1.5 kg/m³）
                        if 0.5 < mean_val < 2.0 and 0.3 < min_val < 2.5 and 0.5 < max_val < 2.5:
                            status = "✅ 正常"
                        else:
                            status = "⚠️  异常"
                        
                        print(f"  {col}: {min_val:.3f}~{max_val:.3f} kg/m³ (均值:{mean_val:.3f}) {status}")
            
        except Exception as e:
            print(f"❌ 验证 {station} 时出错: {e}")

def recalculate_density_from_wrf():
    """重新从WRF温度和气压数据计算密度"""
    
    project_root = Path("/Users/xiaxin/work/WindForecast_Project")
    matched_data_dir = project_root / "01_Data/processed/matched_data"
    
    print("\n" + "=" * 60)
    print("🧮 重新计算WRF密度数据")
    print("=" * 60)
    
    # 理想气体常数（干空气）
    R_specific = 287.05  # J/(kg·K)
    
    stations = ['changma', 'kuangqu', 'sanlijijingzi']
    
    for station in stations:
        matched_file = matched_data_dir / f"{station}_matched.csv"
        
        if not matched_file.exists():
            continue
            
        print(f"\n重新计算 {station} 站点密度...")
        
        try:
            df = pd.read_csv(matched_file, index_col=0)
            df.index = pd.to_datetime(df.index)
            
            # 查找温度和气压列（来自原始WRF数据）
            for model in ['ec', 'gfs']:
                temp_col = f'{model}_temperature_10m'
                density_col = f'{model}_density_10m'
                
                # 从原始WRF数据获取气压（这里需要从原始WRF文件读取）
                wrf_file = project_root / f"01_Data/raw/wrf/{model}_driven/{station}.csv"
                
                if wrf_file.exists() and temp_col in df.columns:
                    print(f"  处理 {model.upper()} 数据...")
                    
                    # 读取原始WRF数据获取气压
                    wrf_df = pd.read_csv(wrf_file)
                    time_col = 'datetime' if 'datetime' in wrf_df.columns else 'date'
                    wrf_df['datetime'] = pd.to_datetime(wrf_df[time_col])
                    wrf_df.set_index('datetime', inplace=True)
                    
                    # 检查是否有30m气压数据
                    if 'p_30m' in wrf_df.columns:
                        # 对齐时间
                        common_times = df.index.intersection(wrf_df.index)
                        
                        if len(common_times) > 100:
                            # 获取温度（已转换为°C）和气压
                            temp_celsius = df.loc[common_times, temp_col]
                            temp_kelvin = temp_celsius + 273.15  # 转回开尔文用于计算
                            pressure_pa = wrf_df.loc[common_times, 'p_30m']  # 假设是Pa
                            
                            # 计算密度 ρ = P/(R*T)
                            calculated_density = pressure_pa / (R_specific * temp_kelvin)
                            
                            # 检查计算结果是否合理
                            mean_density = calculated_density.mean()
                            
                            if 0.5 < mean_density < 2.0:  # 合理范围
                                df.loc[common_times, density_col] = calculated_density
                                print(f"    ✅ 重新计算密度成功 (均值: {mean_density:.3f} kg/m³)")
                            else:
                                print(f"    ⚠️  计算结果异常 (均值: {mean_density:.3f} kg/m³)")
                        else:
                            print(f"    ⚠️  时间匹配点不足: {len(common_times)}")
                    else:
                        print(f"    ⚠️  未找到气压数据 (p_30m)")
            
            # 保存更新后的数据
            df.to_csv(matched_file)
            print(f"  ✅ {station} 数据已更新")
            
        except Exception as e:
            print(f"  ❌ 处理 {station} 时出错: {e}")

if __name__ == "__main__":
    # 1. 修复单位问题
    fix_wrf_units()
    
    # 2. 验证修复结果
    validate_units()
    
    # 3. 可选：重新计算密度（如果需要）
    print("\n是否需要重新从原始WRF数据计算密度？(y/n)")
    # recalculate_density_from_wrf()  # 取消注释以运行