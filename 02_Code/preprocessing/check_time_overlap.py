import pandas as pd
from pathlib import Path

def check_time_overlap():
    """检查观测数据与WRF数据的时间重叠情况"""
    
    project_root = Path("/Users/xiaxin/work/WindForecast_Project")
    
    stations = ['changma', 'kuangqu', 'sanlijijingzi']
    
    print("=" * 80)
    print("观测数据与WRF数据时间重叠检查")
    print("=" * 80)
    
    for station in stations:
        print(f"\n{'='*20} {station.upper()} 站点 {'='*20}")
        
        try:
            # 1. 读取观测数据时间范围
            if station == 'changma':
                obs_file = project_root / "01_Data/raw/processed/cleaned-15min/changma_20210501_20221031_15min.csv"
            elif station == 'kuangqu':
                obs_file = project_root / "01_Data/raw/processed/cleaned-15min/kuangqu_20210501_20220601_15min.csv"
            else:  # sanlijijingzi
                obs_file = project_root / "01_Data/raw/processed/cleaned-15min/sanlijijingzi_20210601_20220616_15min.csv"
            
            obs_df = pd.read_csv(obs_file)
            obs_df['datetime'] = pd.to_datetime(obs_df['datetime'])
            
            obs_start = obs_df['datetime'].min()
            obs_end = obs_df['datetime'].max()
            obs_count = len(obs_df)
            
            print(f"观测数据:")
            print(f"  时间范围: {obs_start} 到 {obs_end}")
            print(f"  数据点数: {obs_count}")
            print(f"  期间长度: {(obs_end - obs_start).days} 天")
            
            # 2. 读取EC-WRF数据时间范围
            ec_wrf_file = project_root / f"01_Data/raw/wrf/ec_driven/{station}.csv"
            ec_df = pd.read_csv(ec_wrf_file)
            time_col = 'datetime' if 'datetime' in ec_df.columns else 'date'
            ec_df['datetime'] = pd.to_datetime(ec_df[time_col])
            
            ec_start = ec_df['datetime'].min()
            ec_end = ec_df['datetime'].max()
            ec_count = len(ec_df)
            
            print(f"EC-WRF数据:")
            print(f"  时间范围: {ec_start} 到 {ec_end}")
            print(f"  数据点数: {ec_count}")
            print(f"  期间长度: {(ec_end - ec_start).days} 天")
            
            # 3. 读取GFS-WRF数据时间范围
            gfs_wrf_file = project_root / f"01_Data/raw/wrf/gfs_driven/{station}.csv"
            gfs_df = pd.read_csv(gfs_wrf_file)
            time_col = 'datetime' if 'datetime' in gfs_df.columns else 'date'
            gfs_df['datetime'] = pd.to_datetime(gfs_df[time_col])
            
            gfs_start = gfs_df['datetime'].min()
            gfs_end = gfs_df['datetime'].max()
            gfs_count = len(gfs_df)
            
            print(f"GFS-WRF数据:")
            print(f"  时间范围: {gfs_start} 到 {gfs_end}")
            print(f"  数据点数: {gfs_count}")
            print(f"  期间长度: {(gfs_end - gfs_start).days} 天")
            
            # 4. 计算重叠时间范围
            common_start = max(obs_start, ec_start, gfs_start)
            common_end = min(obs_end, ec_end, gfs_end)
            
            if common_start <= common_end:
                overlap_days = (common_end - common_start).days
                print(f"\n✅ 共同时间范围:")
                print(f"  开始时间: {common_start}")
                print(f"  结束时间: {common_end}")
                print(f"  重叠期间: {overlap_days} 天")
                
                # 估算重叠数据点数量（每小时4个点）
                estimated_points = overlap_days * 24 * 4
                print(f"  预计数据点: ~{estimated_points} 个")
                
                if overlap_days > 30:
                    print(f"  ✅ 重叠期间充足，适合分析")
                else:
                    print(f"  ⚠️  重叠期间较短，可能影响分析质量")
            else:
                print(f"\n❌ 没有共同时间范围!")
                print(f"  观测数据最晚开始: {obs_start}")
                print(f"  WRF数据最早结束: {min(ec_end, gfs_end)}")
        
        except Exception as e:
            print(f"\n❌ 处理 {station} 时出错: {e}")
            import traceback
            traceback.print_exc()

if __name__ == "__main__":
    check_time_overlap()