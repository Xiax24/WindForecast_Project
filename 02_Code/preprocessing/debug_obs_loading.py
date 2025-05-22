import pandas as pd
from pathlib import Path
import traceback

def debug_obs_loading():
    """调试观测数据加载问题"""
    
    project_root = Path("/Users/xiaxin/work/WindForecast_Project")
    
    # 测试文件路径
    test_files = [
        project_root / "01_Data/raw/processed/cleaned-15min/changma_20210501_20221031_15min.csv",
        project_root / "01_Data/raw/processed/cleaned-15min/kuangqu_20210501_20220601_15min.csv",
        project_root / "01_Data/raw/processed/cleaned-15min/sanlijijingzi_20210601_20220616_15min.csv"
    ]
    
    for obs_file in test_files:
        print(f"\n{'='*60}")
        print(f"测试文件: {obs_file.name}")
        print(f"{'='*60}")
        
        # 1. 检查文件是否存在
        print(f"文件存在: {obs_file.exists()}")
        
        if not obs_file.exists():
            print("❌ 文件不存在!")
            continue
        
        try:
            # 2. 尝试读取前几行
            print("\n2. 读取前3行查看结构:")
            df_peek = pd.read_csv(obs_file, nrows=3)
            print(f"形状: {df_peek.shape}")
            print(f"列名: {df_peek.columns.tolist()}")
            print("前3行:")
            print(df_peek)
            
            # 3. 检查datetime列
            if 'datetime' in df_peek.columns:
                print(f"\n3. datetime列样例:")
                print(df_peek['datetime'].tolist())
                
                # 4. 尝试转换datetime
                print(f"\n4. 尝试转换datetime:")
                df_peek['datetime'] = pd.to_datetime(df_peek['datetime'])
                print("✅ datetime转换成功!")
                print(f"转换后的datetime: {df_peek['datetime'].tolist()}")
                
                # 5. 尝试设置索引
                print(f"\n5. 尝试设置datetime为索引:")
                df_peek.set_index('datetime', inplace=True)
                print("✅ 索引设置成功!")
                print(f"索引类型: {type(df_peek.index)}")
                print(f"索引范围: {df_peek.index.min()} 到 {df_peek.index.max()}")
                
            else:
                print("\n❌ 没有找到 'datetime' 列!")
                print("可能的时间列:")
                time_cols = [col for col in df_peek.columns 
                            if any(keyword in col.lower() 
                                  for keyword in ['time', 'date'])]
                print(time_cols)
        
        except Exception as e:
            print(f"\n❌ 处理过程中出错:")
            print(f"错误类型: {type(e).__name__}")
            print(f"错误信息: {str(e)}")
            print("\n详细错误信息:")
            traceback.print_exc()

def test_single_file_loading():
    """测试单个文件的完整加载过程"""
    
    project_root = Path("/Users/xiaxin/work/WindForecast_Project")
    test_file = project_root / "01_Data/raw/processed/cleaned-15min/changma_20210501_20221031_15min.csv"
    
    print(f"\n{'='*60}")
    print("测试完整的文件加载过程 (昌马站)")
    print(f"{'='*60}")
    
    try:
        print("1. 读取CSV文件...")
        obs_df = pd.read_csv(test_file)
        print(f"✅ 文件读取成功! 形状: {obs_df.shape}")
        
        print("2. 检查datetime列...")
        if 'datetime' in obs_df.columns:
            print("✅ 找到datetime列!")
            print(f"datetime列样例: {obs_df['datetime'].head().tolist()}")
        else:
            print("❌ 没有datetime列!")
            print(f"所有列名: {obs_df.columns.tolist()}")
            return
        
        print("3. 转换datetime...")
        obs_df['datetime'] = pd.to_datetime(obs_df['datetime'])
        print(f"✅ datetime转换成功!")
        
        print("4. 设置索引...")
        obs_df.set_index('datetime', inplace=True)
        print(f"✅ 索引设置成功!")
        
        print("5. 最终结果:")
        print(f"数据形状: {obs_df.shape}")
        print(f"时间范围: {obs_df.index.min()} 到 {obs_df.index.max()}")
        print(f"主要列名: {obs_df.columns.tolist()[:10]}")
        
        return obs_df
        
    except Exception as e:
        print(f"\n❌ 完整加载过程失败:")
        print(f"错误类型: {type(e).__name__}")
        print(f"错误信息: {str(e)}")
        traceback.print_exc()
        return None

if __name__ == "__main__":
    # 运行调试
    debug_obs_loading()
    
    print("\n" + "="*80)
    print("测试完整加载过程")
    print("="*80)
    
    # 测试完整加载
    df = test_single_file_loading()
    
    if df is not None:
        print("\n🎉 文件加载测试成功!")
    else:
        print("\n❌ 文件加载测试失败!")