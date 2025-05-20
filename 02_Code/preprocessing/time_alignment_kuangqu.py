"""
时间间隔统一处理脚本
将不同时间间隔的观测数据统一为15分钟间隔
"""

import pandas as pd
import numpy as np
import os
from datetime import datetime

class TimeAligner:
    def __init__(self, processed_data_path):
        """
        初始化时间对齐器
        
        参数:
        processed_data_path: 已处理数据的路径
        """
        self.processed_data_path = processed_data_path
        # 直接设置到01_Data/raw/processed/aligned目录
        project_root = "/Users/xiaxin/work/WindForecast_Project"
        self.aligned_data_path = os.path.join(project_root, "01_Data", "raw", "processed", "aligned")
        
        # 确保输出目录存在
        os.makedirs(self.aligned_data_path, exist_ok=True)

    def load_station_data(self, station_name):
        """加载测站数据"""
        import glob
        
        # 查找对应的cleaned数据文件 - 修改匹配模式
        pattern = os.path.join(self.processed_data_path, f"{station_name}_*_cleaned.csv")
        files = glob.glob(pattern)
        
        # 如果上面的模式没找到，尝试更宽泛的匹配
        if not files:
            pattern = os.path.join(self.processed_data_path, f"{station_name}*.csv")
            files = glob.glob(pattern)
        
        # 再试试不区分大小写
        if not files:
            pattern = os.path.join(self.processed_data_path, "*.csv")
            all_files = glob.glob(pattern)
            files = [f for f in all_files if station_name.lower() in os.path.basename(f).lower()]
        
        if not files:
            print(f"未找到 {station_name} 的数据文件")
            print(f"搜索路径: {self.processed_data_path}")
            # 显示目录下所有csv文件
            all_csv = glob.glob(os.path.join(self.processed_data_path, "*.csv"))
            print(f"目录下所有CSV文件: {[os.path.basename(f) for f in all_csv]}")
            return None
        
        file_path = files[0]  # 取第一个匹配的文件
        print(f"加载 {station_name} 数据: {os.path.basename(file_path)}")
        
        try:
            df = pd.read_csv(file_path, index_col=0, parse_dates=True)
            print(f"  数据形状: {df.shape}")
            print(f"  时间范围: {df.index.min()} 到 {df.index.max()}")
            return df
        except Exception as e:
            print(f"读取文件出错: {e}")
            return None

    def analyze_time_resolution(self, df, station_name):
        """分析数据的实际时间分辨率"""
        if df is None or df.empty:
            return None
            
        # 计算时间间隔
        time_diffs = df.index.to_series().diff().dropna()
        
        # 统计最常见的时间间隔
        time_diff_counts = time_diffs.value_counts()
        most_common_interval = time_diff_counts.index[0]
        
        print(f"\n{station_name} 时间分辨率分析:")
        print(f"  最常见时间间隔: {most_common_interval}")
        print(f"  出现次数: {time_diff_counts.iloc[0]}")
        
        # 显示前几个最常见的间隔
        print(f"  时间间隔分布前5:")
        for interval, count in time_diff_counts.head(5).items():
            print(f"    {interval}: {count} 次")
        
        return most_common_interval

    def resample_to_15min_nearest(self, df, station_name):
        """
        使用最近邻方法重采样到15分钟
        
        这是推荐方法，保持原始测量值的物理意义
        """
        if df is None or df.empty:
            return None
            
        print(f"\n重采样 {station_name} 到15分钟间隔（最近邻方法）...")
        
        # 创建目标15分钟时间序列
        start_time = df.index.min().ceil('15min')  # 向上取整到15分钟
        end_time = df.index.max().floor('15min')   # 向下取整到15分钟
        
        # 确保起始时间是正确的15分钟边界
        if start_time.minute % 15 != 0:
            # 调整到最近的15分钟边界 (00, 15, 30, 45)
            minute = start_time.minute
            if minute < 15:
                new_minute = 15
            elif minute < 30:
                new_minute = 30
            elif minute < 45:
                new_minute = 45
            else:
                new_minute = 0
                start_time = start_time + pd.Timedelta(hours=1)
            start_time = start_time.replace(minute=new_minute, second=0, microsecond=0)
        
        target_index = pd.date_range(start_time, end_time, freq='15min')
        print(f"  目标时间序列: {len(target_index)} 个点")
        print(f"  目标时间范围: {target_index[0]} 到 {target_index[-1]}")
        
        # 使用最近邻重采样
        df_resampled = df.resample('15min', origin=start_time).nearest()
        
        # 只保留目标时间点
        df_15min = df_resampled.reindex(target_index)
        
        print(f"  重采样后数据形状: {df_15min.shape}")
        print(f"  数据完整性检查:")
        
        for col in df_15min.columns:
            valid_count = df_15min[col].notna().sum()
            valid_pct = valid_count / len(df_15min) * 100
            print(f"    {col}: {valid_pct:.1f}% 有效")
        
        return df_15min

    def save_aligned_data(self, dataframes, merged_df=None):
        """保存时间对齐后的数据"""
        print("\n保存时间对齐后的数据...")
        
        saved_files = []
        
        # 保存各站点的15分钟数据
        for station, df in dataframes.items():
            if df is not None and not df.empty:
                start_date = df.index.min().strftime('%Y%m%d')
                end_date = df.index.max().strftime('%Y%m%d')
                
                filename = f"{station}_{start_date}_{end_date}_15min.csv"
                filepath = os.path.join(self.aligned_data_path, filename)
                
                df.to_csv(filepath)
                print(f"  保存 {station}: {filepath}")
                saved_files.append(filepath)
                
                # 为单站点保存摘要
                summary_filename = f"{station}_{start_date}_{end_date}_15min_summary.txt"
                summary_filepath = os.path.join(self.aligned_data_path, summary_filename)
                
                with open(summary_filepath, 'w', encoding='utf-8') as f:
                    f.write(f"{station} 站点15分钟数据摘要\n")
                    f.write("="*50 + "\n\n")
                    f.write(f"处理时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
                    f.write(f"数据形状: {df.shape}\n")
                    f.write(f"时间范围: {df.index.min()} 到 {df.index.max()}\n")
                    f.write(f"时间分辨率: 15分钟\n")
                    f.write(f"数据期间: {(df.index.max() - df.index.min()).days} 天\n\n")
                    
                    f.write("变量列表:\n")
                    for col in df.columns:
                        valid_count = df[col].notna().sum()
                        valid_pct = valid_count / len(df) * 100
                        f.write(f"  {col}: {valid_count} 有效值 ({valid_pct:.1f}%)\n")
                    
                    f.write("\n基本统计:\n")
                    f.write(df.describe().to_string())
                
                print(f"  保存摘要: {summary_filepath}")
        
        return saved_files[0] if saved_files else None

def main():
    """主函数 - 处理时间对齐"""
    # 设置路径 - 修正为实际的文件位置
    processed_data_path = "/Users/xiaxin/work/WindForecast_Project/01_Data/raw/processed/cleaned"
    
    # 初始化时间对齐器
    aligner = TimeAligner(processed_data_path)
    
    print("="*60)
    print("开始矿区站点时间间隔统一处理")
    print("="*60)
    
    # 只处理矿区站点
    stations = ['kuangqu']  # 只处理矿区站点
    station_data = {}
    
    for station in stations:
        df = aligner.load_station_data(station)
        if df is not None:
            # 分析时间分辨率
            aligner.analyze_time_resolution(df, station)
            station_data[station] = df
    
    if not station_data:
        print("没有可用的站点数据")
        return
    
    # 重采样到15分钟
    resampled_data = {}
    for station, df in station_data.items():
        # 使用最近邻方法重采样
        df_15min = aligner.resample_to_15min_nearest(df, station)
        if df_15min is not None:
            resampled_data[station] = df_15min
    
    # 保存结果
    output_file = aligner.save_aligned_data(resampled_data)
    
    if output_file:
        print(f"\n🎉 时间对齐处理完成！")
        print(f"📁 主要输出文件: {output_file}")
        print(f"📁 输出目录: {aligner.aligned_data_path}")
    else:
        print(f"\n❌ 时间对齐处理失败")

if __name__ == "__main__":
    main()