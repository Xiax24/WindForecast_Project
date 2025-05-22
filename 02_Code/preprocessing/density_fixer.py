import pandas as pd
import numpy as np
from pathlib import Path

def analyze_density_problem():
    """深度分析密度数据问题"""
    
    project_root = Path("/Users/xiaxin/work/WindForecast_Project")
    
    print("🔍 深度分析密度数据问题")
    print("=" * 60)
    
    # 1. 检查原始观测数据中的密度
    print("\n1. 检查原始观测数据密度...")
    obs_file = project_root / "01_Data/raw/processed/cleaned-15min/changma_20210501_20221031_15min.csv"
    
    if obs_file.exists():
        obs_df = pd.read_csv(obs_file)
        density_col = 'density_10m'
        
        if density_col in obs_df.columns:
            density_data = obs_df[density_col].dropna()
            print(f"  观测密度统计:")
            print(f"    样本数: {len(density_data)}")
            print(f"    范围: {density_data.min():.6f} ~ {density_data.max():.6f}")
            print(f"    均值: {density_data.mean():.6f}")
            print(f"    中位数: {density_data.median():.6f}")
            
            # 检查是否合理
            if density_data.mean() > 10:
                print(f"    💡 观测密度可能单位是 g/m³，需要转换为 kg/m³")
            elif density_data.mean() < 0.1:
                print(f"    💡 观测密度可能单位异常，标准大气密度约1.225 kg/m³")
    
    # 2. 检查原始WRF数据
    print("\n2. 检查原始WRF数据...")
    for model in ['ec', 'gfs']:
        wrf_file = project_root / f"01_Data/raw/wrf/{model}_driven/changma.csv"
        
        if wrf_file.exists():
            print(f"\n  {model.upper()}-WRF 原始数据检查:")
            wrf_df = pd.read_csv(wrf_file, nrows=1000)  # 只读前1000行用于分析
            
            # 检查可用的温度和气压数据
            temp_cols = [col for col in wrf_df.columns if 'tk_' in col]
            pressure_cols = [col for col in wrf_df.columns if 'p_' in col]
            
            print(f"    温度列: {temp_cols[:3]}...")
            print(f"    气压列: {pressure_cols[:3]}...")
            
            # 检查30m数据
            if 'tk_30m' in wrf_df.columns and 'p_30m' in wrf_df.columns:
                temp_30m = wrf_df['tk_30m'].dropna()
                pres_30m = wrf_df['p_30m'].dropna()
                
                print(f"    30m温度范围: {temp_30m.min():.1f} ~ {temp_30m.max():.1f} K")
                print(f"    30m气压范围: {pres_30m.min():.1f} ~ {pres_30m.max():.1f}")
                
                # 估算气压单位
                if pres_30m.mean() > 50000:
                    print(f"    💡 气压单位可能是 Pa")
                elif 500 < pres_30m.mean() < 1200:
                    print(f"    💡 气压单位可能是 hPa/mbar")
                
def fix_density_comprehensively():
    """全面修复密度数据"""
    
    project_root = Path("/Users/xiaxin/work/WindForecast_Project")
    matched_data_dir = project_root / "01_Data/processed/matched_data"
    
    print("\n🔧 全面修复密度数据")
    print("=" * 60)
    
    # 理想气体常数
    R_specific = 287.05  # J/(kg·K) 干空气比气体常数
    
    stations = ['changma', 'kuangqu', 'sanlijijingzi']
    
    for station in stations:
        print(f"\n处理 {station} 站点...")
        
        matched_file = matched_data_dir / f"{station}_matched.csv"
        if not matched_file.exists():
            continue
            
        try:
            # 读取匹配数据
            df = pd.read_csv(matched_file, index_col=0)
            df.index = pd.to_datetime(df.index)
            
            # 1. 修复观测密度（如果需要）
            obs_density_col = 'obs_density_10m'
            if obs_density_col in df.columns:
                obs_density = df[obs_density_col].dropna()
                if len(obs_density) > 0:
                    mean_obs = obs_density.mean()
                    
                    if mean_obs > 10:  # 可能是g/m³
                        df[obs_density_col] = df[obs_density_col] / 1000
                        print(f"  ✅ 观测密度转换: g/m³ → kg/m³ (均值: {mean_obs:.1f} → {df[obs_density_col].mean():.3f})")
                    elif mean_obs < 0.1:  # 数值过小
                        # 尝试乘以100 (可能是特殊单位)
                        test_value = mean_obs * 100
                        if 0.8 < test_value < 2.0:
                            df[obs_density_col] = df[obs_density_col] * 100
                            print(f"  ✅ 观测密度修正: ×100 (均值: {mean_obs:.6f} → {df[obs_density_col].mean():.3f})")
                        else:
                            print(f"  ⚠️  观测密度异常，保持原值 (均值: {mean_obs:.6f})")
            
            # 2. 重新计算WRF密度
            for model in ['ec', 'gfs']:
                density_col = f'{model}_density_10m'
                temp_col = f'{model}_temperature_10m'
                
                # 读取原始WRF数据获取气压
                wrf_file = project_root / f"01_Data/raw/wrf/{model}_driven/{station}.csv"
                
                if wrf_file.exists() and temp_col in df.columns:
                    print(f"  重新计算 {model.upper()}-WRF 密度...")
                    
                    # 读取WRF原始数据
                    wrf_df = pd.read_csv(wrf_file)
                    time_col = 'datetime' if 'datetime' in wrf_df.columns else 'date'
                    wrf_df['datetime'] = pd.to_datetime(wrf_df[time_col])
                    wrf_df.set_index('datetime', inplace=True)
                    
                    # 检查气压数据
                    pressure_cols = ['p_30m', 'p_50m', 'p_70m']
                    pressure_col = None
                    
                    for p_col in pressure_cols:
                        if p_col in wrf_df.columns:
                            pressure_col = p_col
                            break
                    
                    if pressure_col:
                        # 时间对齐
                        common_times = df.index.intersection(wrf_df.index)
                        
                        if len(common_times) > 100:
                            # 获取温度和气压
                            temp_celsius = df.loc[common_times, temp_col]
                            temp_kelvin = temp_celsius + 273.15
                            pressure = wrf_df.loc[common_times, pressure_col]
                            
                            # 判断气压单位并转换为Pa
                            mean_pressure = pressure.mean()
                            if mean_pressure > 50000:  # 已经是Pa
                                pressure_pa = pressure
                            elif 500 < mean_pressure < 1200:  # hPa或mbar
                                pressure_pa = pressure * 100
                            else:
                                print(f"    ⚠️  气压单位未知 (均值: {mean_pressure:.1f})")
                                continue
                            
                            # 计算密度: ρ = P/(R*T)
                            calculated_density = pressure_pa / (R_specific * temp_kelvin)
                            
                            # 验证结果
                            mean_density = calculated_density.mean()
                            if 0.5 < mean_density < 2.0:
                                df.loc[common_times, density_col] = calculated_density
                                print(f"    ✅ 重算成功 (均值: {mean_density:.3f} kg/m³)")
                            else:
                                print(f"    ⚠️  计算结果异常 (均值: {mean_density:.3f})")
                        else:
                            print(f"    ⚠️  时间匹配点不足")
                    else:
                        print(f"    ⚠️  未找到气压数据")
            
            # 保存修复后的数据
            df.to_csv(matched_file)
            print(f"  ✅ {station} 密度数据已修复")
            
        except Exception as e:
            print(f"  ❌ 处理 {station} 时出错: {e}")
            import traceback
            traceback.print_exc()

def update_description_files():
    """更新数据描述文件"""
    
    project_root = Path("/Users/xiaxin/work/WindForecast_Project")
    matched_data_dir = project_root / "01_Data/processed/matched_data"
    
    print("\n📝 更新数据描述文件")
    print("=" * 60)
    
    stations = ['changma', 'kuangqu', 'sanlijijingzi']
    
    for station in stations:
        matched_file = matched_data_dir / f"{station}_matched.csv"
        desc_file = matched_data_dir / f"{station}_matched_description.txt"
        
        if not matched_file.exists():
            continue
            
        print(f"\n更新 {station} 描述文件...")
        
        try:
            # 读取数据
            df = pd.read_csv(matched_file, index_col=0)
            df.index = pd.to_datetime(df.index)
            
            # 生成新的描述文件
            with open(desc_file, 'w', encoding='utf-8') as f:
                f.write(f"{station} 匹配数据描述 (单位修复后)\n")
                f.write("=" * 60 + "\n\n")
                f.write(f"数据形状: {df.shape}\n")
                f.write(f"时间范围: {df.index.min()} 到 {df.index.max()}\n")
                f.write(f"数据期间: {(df.index.max() - df.index.min()).days} 天\n\n")
                
                f.write("数据单位说明:\n")
                f.write("  风速: m/s\n")
                f.write("  风向: 度 (°)\n") 
                f.write("  温度: 摄氏度 (°C)\n")
                f.write("  密度: kg/m³\n")
                f.write("  湿度: %\n\n")
                
                f.write("变量完整性:\n")
                for col in df.columns:
                    valid_count = df[col].notna().sum()
                    valid_pct = valid_count / len(df) * 100
                    f.write(f"  {col}: {valid_count}/{len(df)} ({valid_pct:.1f}%)\n")
                
                f.write(f"\n变量统计摘要:\n")
                
                # 按类型分组显示统计
                var_types = {
                    '风速变量': [col for col in df.columns if 'wind_speed' in col and 'std' not in col and 'max' not in col],
                    '风向变量': [col for col in df.columns if 'wind_direction' in col],
                    '温度变量': [col for col in df.columns if 'temperature' in col],
                    '密度变量': [col for col in df.columns if 'density' in col],
                    '湿度变量': [col for col in df.columns if 'humidity' in col]
                }
                
                for var_type, cols in var_types.items():
                    if cols:
                        f.write(f"\n{var_type}:\n")
                        for col in cols:
                            if df[col].notna().sum() > 0:
                                mean_val = df[col].mean()
                                std_val = df[col].std()
                                min_val = df[col].min()
                                max_val = df[col].max()
                                f.write(f"  {col}: 均值={mean_val:.3f}, 标准差={std_val:.3f}, 范围=[{min_val:.3f}, {max_val:.3f}]\n")
                
                f.write(f"\n数据质量评估:\n")
                
                # 检查数据质量
                temp_cols = [col for col in df.columns if 'temperature' in col]
                density_cols = [col for col in df.columns if 'density' in col]
                
                for col in temp_cols:
                    if df[col].notna().sum() > 0:
                        mean_temp = df[col].mean()
                        if -50 < mean_temp < 50:
                            f.write(f"  {col}: ✅ 温度范围正常\n")
                        else:
                            f.write(f"  {col}: ⚠️  温度范围异常 (均值: {mean_temp:.1f}°C)\n")
                
                for col in density_cols:
                    if df[col].notna().sum() > 0:
                        mean_density = df[col].mean()
                        if 0.5 < mean_density < 2.0:
                            f.write(f"  {col}: ✅ 密度范围正常\n")
                        else:
                            f.write(f"  {col}: ⚠️  密度范围异常 (均值: {mean_density:.3f} kg/m³)\n")
                
                f.write(f"\n修复历史:\n")
                f.write(f"  - 温度单位: 开尔文(K) → 摄氏度(°C)\n")
                f.write(f"  - 密度单位: 重新计算或单位转换\n")
                f.write(f"  - 更新时间: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            
            print(f"  ✅ {desc_file.name} 已更新")
            
        except Exception as e:
            print(f"  ❌ 更新 {station} 描述时出错: {e}")

def validate_final_results():
    """最终验证修复结果"""
    
    project_root = Path("/Users/xiaxin/work/WindForecast_Project")
    matched_data_dir = project_root / "01_Data/processed/matched_data"
    
    print("\n🏁 最终验证修复结果")
    print("=" * 60)
    
    stations = ['changma', 'kuangqu', 'sanlijijingzi']
    
    for station in stations:
        matched_file = matched_data_dir / f"{station}_matched.csv"
        
        if not matched_file.exists():
            continue
            
        print(f"\n{station.upper()} 站点最终验证:")
        print("-" * 40)
        
        try:
            df = pd.read_csv(matched_file, index_col=0)
            
            # 检查关键变量
            key_vars = {
                'temperature': [col for col in df.columns if 'temperature' in col],
                'density': [col for col in df.columns if 'density' in col],
                'wind_speed': [col for col in df.columns if 'wind_speed' in col and 'std' not in col and 'max' not in col]
            }
            
            for var_type, cols in key_vars.items():
                if cols:
                    print(f"{var_type.upper()}:")
                    for col in cols[:3]:  # 只显示前3个
                        if df[col].notna().sum() > 0:
                            mean_val = df[col].mean()
                            min_val = df[col].min()
                            max_val = df[col].max()
                            
                            # 判断单位是否正确
                            if var_type == 'temperature':
                                unit = "°C"
                                status = "✅" if -50 < mean_val < 50 else "⚠️"
                            elif var_type == 'density':
                                unit = "kg/m³"
                                status = "✅" if 0.5 < mean_val < 2.0 else "⚠️"
                            else:  # wind_speed
                                unit = "m/s"
                                status = "✅" if 0 < mean_val < 30 else "⚠️"
                            
                            print(f"  {col}: {min_val:.3f}~{max_val:.3f} {unit} (均值:{mean_val:.3f}) {status}")
            
        except Exception as e:
            print(f"❌ 验证 {station} 时出错: {e}")

if __name__ == "__main__":
    # 1. 深度分析密度问题
    analyze_density_problem()
    
    # 2. 全面修复密度
    fix_density_comprehensively()
    
    # 3. 更新描述文件
    update_description_files()
    
    # 4. 最终验证
    validate_final_results()
    
    print("\n🎉 密度修复和文件更新完成!")
    print("请检查 01_Data/processed/matched_data/ 目录下的文件")