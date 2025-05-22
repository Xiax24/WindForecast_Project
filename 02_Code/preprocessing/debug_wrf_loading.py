import pandas as pd
from pathlib import Path
import traceback

def debug_wrf_loading():
    """调试WRF数据加载问题"""
    
    project_root = Path("/Users/xiaxin/work/WindForecast_Project")
    
    # WRF数据文件路径
    wrf_files = {
        'EC-WRF': [
            project_root / "01_Data/raw/wrf/ec_driven/changma.csv",
            project_root / "01_Data/raw/wrf/ec_driven/kuangqu.csv",
            project_root / "01_Data/raw/wrf/ec_driven/sanlijijingzi.csv"
        ],
        'GFS-WRF': [
            project_root / "01_Data/raw/wrf/gfs_driven/changma.csv",
            project_root / "01_Data/raw/wrf/gfs_driven/kuangqu.csv",
            project_root / "01_Data/raw/wrf/gfs_driven/sanlijijingzi.csv"
        ]
    }
    
    for model_type, file_list in wrf_files.items():
        print(f"\n{'='*60}")
        print(f"测试 {model_type} 数据")
        print(f"{'='*60}")
        
        for wrf_file in file_list:
            print(f"\n测试文件: {wrf_file.name}")
            print("-" * 40)
            
            # 1. 检查文件是否存在
            print(f"文件存在: {wrf_file.exists()}")
            
            if not wrf_file.exists():
                print("❌ 文件不存在!")
                continue
            
            try:
                # 2. 尝试读取前几行
                print("\n2. 读取前3行查看结构:")
                df_peek = pd.read_csv(wrf_file, nrows=3)
                print(f"形状: {df_peek.shape}")
                print(f"列名: {df_peek.columns.tolist()}")
                
                # 3. 查找时间列
                possible_time_cols = ['datetime', 'date', 'time', 'Date', 'DateTime']
                time_column = None
                
                for col in possible_time_cols:
                    if col in df_peek.columns:
                        time_column = col
                        break
                
                if time_column:
                    print(f"\n3. 找到时间列: {time_column}")
                    print(f"时间列样例: {df_peek[time_column].tolist()}")
                    
                    # 4. 尝试转换datetime
                    print(f"\n4. 尝试转换datetime:")
                    df_peek['datetime_converted'] = pd.to_datetime(df_peek[time_column])
                    print("✅ datetime转换成功!")
                    print(f"转换后的datetime: {df_peek['datetime_converted'].tolist()}")
                    
                else:
                    print("\n❌ 没有找到时间列!")
                    print("所有列名:")
                    for i, col in enumerate(df_peek.columns):
                        print(f"  {i}: {col}")
                
                # 5. 检查关键变量
                print(f"\n5. 检查关键变量:")
                wind_vars = [col for col in df_peek.columns if 'wind' in col.lower() or 'ws_' in col.lower()]
                print(f"风速相关变量: {wind_vars[:5]}...")
                
                temp_vars = [col for col in df_peek.columns if 'tk_' in col.lower() or 'temp' in col.lower()]
                print(f"温度相关变量: {temp_vars[:5]}...")
                
                uv_vars = [col for col in df_peek.columns if col.startswith('u_') or col.startswith('v_')]
                print(f"u/v分量变量: {uv_vars[:5]}...")
        
            except Exception as e:
                print(f"\n❌ 处理过程中出错:")
                print(f"错误类型: {type(e).__name__}")
                print(f"错误信息: {str(e)}")
                print("\n详细错误信息:")
                traceback.print_exc()

def test_wrf_file_loading():
    """测试单个WRF文件的完整加载过程"""
    
    project_root = Path("/Users/xiaxin/work/WindForecast_Project")
    
    # 测试文件
    test_files = [
        ("EC-WRF昌马", project_root / "01_Data/raw/wrf/ec_driven/changma.csv"),
        ("GFS-WRF昌马", project_root / "01_Data/raw/wrf/gfs_driven/changma.csv")
    ]
    
    for file_desc, test_file in test_files:
        print(f"\n{'='*60}")
        print(f"测试完整加载过程: {file_desc}")
        print(f"{'='*60}")
        
        if not test_file.exists():
            print("❌ 文件不存在!")
            continue
        
        try:
            print("1. 读取CSV文件...")
            wrf_df = pd.read_csv(test_file)
            print(f"✅ 文件读取成功! 形状: {wrf_df.shape}")
            
            print("2. 查找时间列...")
            possible_time_cols = ['datetime', 'date', 'time', 'Date', 'DateTime']
            time_column = None
            
            for col in possible_time_cols:
                if col in wrf_df.columns:
                    time_column = col
                    break
            
            if time_column:
                print(f"✅ 找到时间列: {time_column}")
                print(f"时间列样例: {wrf_df[time_column].head().tolist()}")
            else:
                print("❌ 没有找到时间列!")
                print(f"所有列名: {wrf_df.columns.tolist()}")
                continue
            
            print("3. 转换datetime...")
            wrf_df['datetime'] = pd.to_datetime(wrf_df[time_column])
            print(f"✅ datetime转换成功!")
            
            print("4. 设置索引...")
            wrf_df.set_index('datetime', inplace=True)
            print(f"✅ 索引设置成功!")
            
            print("5. 最终结果:")
            print(f"数据形状: {wrf_df.shape}")
            print(f"时间范围: {wrf_df.index.min()} 到 {wrf_df.index.max()}")
            print(f"主要列名: {wrf_df.columns.tolist()[:10]}")
            
            # 检查关键变量
            key_vars = ['wind_speed_10m', 'ws_30m', 'ws_50m', 'ws_70m']
            available_key_vars = [var for var in key_vars if var in wrf_df.columns]
            print(f"关键风速变量: {available_key_vars}")
            
        except Exception as e:
            print(f"\n❌ 完整加载过程失败:")
            print(f"错误类型: {type(e).__name__}")
            print(f"错误信息: {str(e)}")
            traceback.print_exc()

if __name__ == "__main__":
    # 运行WRF数据诊断
    debug_wrf_loading()
    
    print("\n" + "="*80)
    print("测试WRF文件完整加载过程")
    print("="*80)
    
    # 测试完整加载
    test_wrf_file_loading()