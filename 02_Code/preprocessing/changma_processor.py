"""
修复版昌马数据处理脚本
解决时间索引问题和数据保存问题
"""

import pandas as pd
import numpy as np
import os
import glob
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

class ChangmaDataProcessorFixed:
    def __init__(self, base_path):
        """
        初始化昌马数据处理器
        
        参数:
        base_path: 昌马数据根目录路径
        """
        self.base_path = base_path
        self.processed_path = os.path.join(base_path, '..', '..', 'processed', 'cleaned')
        
        # 确保输出目录存在
        os.makedirs(self.processed_path, exist_ok=True)
        
        # 缺失值标记
        self.missing_values = [-99.0, -99, '-99.0', '-99', 'NULL', 'null', '', ' ', '\\N']
        
        # 高度映射
        self.height_mapping = {
            '10米': 10, '10m': 10, '10': 10,
            '30米': 30, '30m': 30, '30': 30,
            '50米': 50, '50m': 50, '50': 50,
            '70米': 70, '70m': 70, '70': 70
        }

    def detect_height_from_path(self, path):
        """从路径中检测高度信息"""
        path_lower = path.lower()
        
        for key, height in self.height_mapping.items():
            if key in path_lower:
                return height
        
        # 如果没有找到，尝试提取数字
        import re
        numbers = re.findall(r'\d+', os.path.basename(path))
        for num in numbers:
            if int(num) in [10, 30, 50, 70]:
                return int(num)
        
        return None

    def standardize_changma_columns(self, df):
        """标准化昌马数据的列名"""
        df_std = df.copy()
        
        column_mapping = {
            '时间': 'datetime', 'Time': 'datetime', 'time': 'datetime', '日期': 'datetime',
            '场站': 'station', '层高': 'layer_height',
            '实测风速': 'wind_speed', '风速': 'wind_speed', 'WindSpeed': 'wind_speed',
            '实测风向': 'wind_direction', '风向': 'wind_direction', 'WindDirection': 'wind_direction',
            '实测温度': 'temperature', '温度': 'temperature', 'Temperature': 'temperature',
            '实测湿度': 'humidity', '湿度': 'humidity', 'Humidity': 'humidity',
            '实测气压': 'pressure', '气压': 'pressure', 'Pressure': 'pressure',
            '大气密度（kg/m³）': 'density', '大气密度': 'density', '密度': 'density'
        }
        
        # 重命名列
        for old_name, new_name in column_mapping.items():
            if old_name in df_std.columns:
                df_std = df_std.rename(columns={old_name: new_name})
        
        return df_std

    def process_changma_datetime(self, df):
        """处理昌马数据的时间列"""
        if 'datetime' not in df.columns:
            print(f"      警告: 未找到时间列，可用列: {df.columns.tolist()}")
            return df
        
        df_time = df.copy()
        
        try:
            original_count = len(df_time)
            
            # 移除明显无效的时间值
            df_time = df_time.dropna(subset=['datetime'])
            
            # 转换为datetime - 修复：不立即设置为索引
            df_time['datetime'] = pd.to_datetime(df_time['datetime'], errors='coerce')
            
            # 移除转换失败的行
            df_time = df_time.dropna(subset=['datetime'])
            
            final_count = len(df_time)
            if final_count < original_count:
                print(f"      时间处理: {original_count} -> {final_count} 行")
            
            if not df_time.empty:
                print(f"      时间范围: {df_time['datetime'].min()} 到 {df_time['datetime'].max()}")
            
        except Exception as e:
            print(f"      时间处理错误: {e}")
            if 'datetime' in df.columns:
                print(f"      时间列样本: {df['datetime'].head().tolist()}")
        
        return df_time

    def clean_changma_data(self, df):
        """清理昌马数据"""
        if df.empty:
            return df
            
        df_clean = df.copy()
        
        # 处理缺失值标记
        for col in df_clean.columns:
            if col not in ['height', 'station', 'layer_height', 'datetime']:
                df_clean[col] = df_clean[col].replace(self.missing_values, np.nan)
        
        # 数据有效性检查
        if 'wind_speed' in df_clean.columns:
            mask_negative = df_clean['wind_speed'] < 0
            mask_extreme = df_clean['wind_speed'] > 50
            
            if mask_negative.sum() > 0:
                print(f"      发现 {mask_negative.sum()} 个负风速值，设为NaN")
                df_clean.loc[mask_negative, 'wind_speed'] = np.nan
            
            if mask_extreme.sum() > 0:
                print(f"      发现 {mask_extreme.sum()} 个超过50m/s的极端风速值，设为NaN")
                df_clean.loc[mask_extreme, 'wind_speed'] = np.nan
        
        # 温度检查
        if 'temperature' in df_clean.columns:
            mask_temp_low = df_clean['temperature'] < -50
            mask_temp_high = df_clean['temperature'] > 60
            
            if mask_temp_low.sum() > 0:
                print(f"      发现 {mask_temp_low.sum()} 个低于-50°C的温度值，设为NaN")
                df_clean.loc[mask_temp_low, 'temperature'] = np.nan
                
            if mask_temp_high.sum() > 0:
                print(f"      发现 {mask_temp_high.sum()} 个高于60°C的温度值，设为NaN")
                df_clean.loc[mask_temp_high, 'temperature'] = np.nan
        
        # 风向检查
        if 'wind_direction' in df_clean.columns:
            mask_wd_invalid = (df_clean['wind_direction'] < 0) | (df_clean['wind_direction'] > 360)
            if mask_wd_invalid.sum() > 0:
                print(f"      发现 {mask_wd_invalid.sum()} 个无效风向值，设为NaN")
                df_clean.loc[mask_wd_invalid, 'wind_direction'] = np.nan
        
        # 统计清理效果
        if len(df_clean) > 0:
            key_vars = ['wind_speed', 'wind_direction', 'temperature', 'humidity', 'pressure']
            existing_vars = [v for v in key_vars if v in df_clean.columns]
            
            print(f"      数据质量检查:")
            for var in existing_vars:
                missing_pct = df_clean[var].isnull().mean() * 100
                print(f"        {var}: {missing_pct:.1f}% 缺失")
        
        return df_clean

    def load_height_data(self, height_path, height_value):
        """加载特定高度的所有数据文件"""
        print(f"  处理 {height_value}m 高度数据...")
        
        excel_files = glob.glob(os.path.join(height_path, "*.xls*"))
        excel_files.sort()
        
        if not excel_files:
            print(f"    未找到Excel文件")
            return pd.DataFrame()
        
        print(f"    找到 {len(excel_files)} 个文件")
        
        all_data = []
        
        for i, file_path in enumerate(excel_files):
            filename = os.path.basename(file_path)
            print(f"    处理文件 {i+1}/{len(excel_files)}: {filename}")
            
            df = None
            try:
                df = pd.read_excel(file_path, engine='openpyxl')
                print(f"      使用openpyxl引擎成功读取")
            except:
                try:
                    df = pd.read_excel(file_path, engine='xlrd')
                    print(f"      使用xlrd引擎成功读取")
                except:
                    try:
                        df = pd.read_excel(file_path)
                        print(f"      使用默认引擎成功读取")
                    except Exception as e:
                        print(f"      所有引擎都失败: {e}")
                        continue
            
            if df is not None:
                try:
                    print(f"      原始数据形状: {df.shape}")
                    print(f"      列名: {df.columns.tolist()}")
                    
                    # 标准化列名
                    df_cleaned = self.standardize_changma_columns(df)
                    
                    # 处理时间列（但不设置为索引）
                    df_cleaned = self.process_changma_datetime(df_cleaned)
                    
                    # 添加高度信息
                    df_cleaned['height'] = height_value
                    
                    # 处理缺失值和异常值
                    df_cleaned = self.clean_changma_data(df_cleaned)
                    
                    if not df_cleaned.empty:
                        all_data.append(df_cleaned)
                        print(f"      清理后数据形状: {df_cleaned.shape}")
                    else:
                        print(f"      数据清理后为空")
                        
                except Exception as e:
                    print(f"      数据处理错误: {e}")
        
        if all_data:
            # 合并所有数据
            combined_df = pd.concat(all_data, ignore_index=True)
            
            # 按时间排序（但仍保持datetime作为列）
            if 'datetime' in combined_df.columns:
                combined_df = combined_df.sort_values('datetime')
                combined_df = combined_df.reset_index(drop=True)
                
            print(f"    {height_value}m 合并后数据形状: {combined_df.shape}")
            return combined_df
        else:
            print(f"    {height_value}m 没有有效数据")
            return pd.DataFrame()

    def process_all_heights(self):
        """处理所有高度的数据"""
        print("开始处理昌马测风塔的所有高度数据...")
        
        items = os.listdir(self.base_path)
        height_dirs = []
        
        for item in items:
            item_path = os.path.join(self.base_path, item)
            if os.path.isdir(item_path):
                height = self.detect_height_from_path(item)
                if height is not None:
                    height_dirs.append((height, item_path))
        
        height_dirs.sort()
        print(f"找到高度目录: {[(h, os.path.basename(p)) for h, p in height_dirs]}")
        
        height_data = {}
        
        for height, height_path in height_dirs:
            df = self.load_height_data(height_path, height)
            if not df.empty:
                height_data[height] = df
            else:
                print(f"  {height}m 数据为空，跳过")
        
        return height_data

    def create_wide_format_data_fixed(self, height_data):
        """修复版：将不同高度的数据合并为宽格式"""
        print("\n创建宽格式数据（修复版）...")
        
        if not height_data:
            print("没有高度数据可合并")
            return pd.DataFrame()
        
        # 首先检查每个高度数据的时间列情况
        print("检查时间列情况:")
        valid_height_data = {}
        
        for height, df in height_data.items():
            print(f"  {height}m: 形状 {df.shape}")
            if 'datetime' in df.columns:
                valid_times = df['datetime'].notna().sum()
                print(f"    有效时间点: {valid_times}")
                if valid_times > 0:
                    valid_height_data[height] = df
                else:
                    print(f"    警告: {height}m 数据没有有效时间点")
            else:
                print(f"    警告: {height}m 数据没有时间列")
        
        if not valid_height_data:
            print("没有包含有效时间的高度数据")
            return pd.DataFrame()
        
        # 创建统一的时间序列
        all_times = []
        for height, df in valid_height_data.items():
            times = df[df['datetime'].notna()]['datetime'].values
            all_times.extend(times)
        
        if not all_times:
            print("没有有效的时间数据")
            return pd.DataFrame()
        
        # 转换为DatetimeIndex并去重
        time_index = pd.DatetimeIndex(all_times).drop_duplicates().sort_values()
        print(f"统一时间索引: {len(time_index)} 个时间点")
        print(f"时间范围: {time_index.min()} 到 {time_index.max()}")
        
        # 初始化宽格式DataFrame
        df_wide = pd.DataFrame(index=time_index)
        
        # 为每个高度添加变量
        variables = ['wind_speed', 'wind_direction', 'temperature', 'humidity', 'pressure', 'density']
        
        for height in sorted(valid_height_data.keys()):
            df_height = valid_height_data[height]
            print(f"  添加 {height}m 数据...")
            
            # 设置时间为索引（创建副本避免修改原数据）
            df_work = df_height.copy()
            df_work = df_work.set_index('datetime')
            
            # 处理重复时间（取平均值）
            if df_work.index.has_duplicates:
                print(f"    发现重复时间，取平均值")
                df_work = df_work.groupby(df_work.index).mean()
            
            for var in variables:
                if var in df_work.columns:
                    col_name = f"{var}_{height}m"
                    
                    # 对齐到统一时间索引
                    aligned_series = df_work[var].reindex(time_index)
                    df_wide[col_name] = aligned_series
                    
                    # 统计有效数据量
                    valid_count = df_wide[col_name].notna().sum()
                    valid_pct = valid_count / len(df_wide) * 100
                    print(f"    {col_name}: {valid_count} 有效值 ({valid_pct:.1f}%)")
        
        print(f"\n宽格式数据创建完成:")
        print(f"  形状: {df_wide.shape}")
        print(f"  列: {df_wide.columns.tolist()}")
        
        return df_wide

    def save_changma_data(self, df_wide):
        """保存昌马处理后的数据"""
        if df_wide.empty:
            print("没有数据可保存")
            return None
        
        # 生成文件名
        start_date = df_wide.index.min().strftime('%Y%m%d')
        end_date = df_wide.index.max().strftime('%Y%m%d')
        
        # 保存CSV文件
        csv_filename = f"changma_{start_date}_{end_date}_cleaned.csv"
        csv_filepath = os.path.join(self.processed_path, csv_filename)
        
        try:
            df_wide.to_csv(csv_filepath)
            print(f"\n✓ 数据已保存到: {csv_filepath}")
            
            # 检查保存的文件
            file_size = os.path.getsize(csv_filepath) / 1024  # KB
            print(f"  文件大小: {file_size:.1f} KB")
            
        except Exception as e:
            print(f"\n✗ 保存CSV文件失败: {e}")
            return None
        
        # 保存摘要文件
        summary_filename = f"changma_{start_date}_{end_date}_summary.txt"
        summary_filepath = os.path.join(self.processed_path, summary_filename)
        
        try:
            with open(summary_filepath, 'w', encoding='utf-8') as f:
                f.write("昌马测风塔数据处理摘要\n")
                f.write("="*50 + "\n\n")
                f.write(f"处理时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
                f.write(f"数据形状: {df_wide.shape}\n")
                f.write(f"时间范围: {df_wide.index.min()} 到 {df_wide.index.max()}\n")
                f.write(f"数据期间: {(df_wide.index.max() - df_wide.index.min()).days} 天\n\n")
                
                f.write("变量列表:\n")
                for col in df_wide.columns:
                    valid_count = df_wide[col].notna().sum()
                    valid_pct = valid_count / len(df_wide) * 100
                    f.write(f"  {col}: {valid_count} 有效值 ({valid_pct:.1f}%)\n")
                
                f.write("\n基本统计:\n")
                f.write(df_wide.describe().to_string())
            
            print(f"✓ 摘要已保存到: {summary_filepath}")
            
        except Exception as e:
            print(f"✗ 保存摘要文件失败: {e}")
        
        return csv_filepath

    def generate_data_summary(self, df_wide):
        """生成数据摘要"""
        if df_wide.empty:
            print("没有数据可生成摘要")
            return
        
        print("\n" + "="*60)
        print("昌马测风塔数据摘要")
        print("="*60)
        
        # 基本信息
        print(f"数据形状: {df_wide.shape}")
        print(f"时间范围: {df_wide.index.min()} 到 {df_wide.index.max()}")
        print(f"数据期间: {(df_wide.index.max() - df_wide.index.min()).days} 天")
        
        # 变量统计
        print(f"\n可用变量:")
        for col in df_wide.columns:
            valid_count = df_wide[col].notna().sum()
            valid_pct = valid_count / len(df_wide) * 100
            if valid_count > 0:
                mean_val = df_wide[col].mean()
                print(f"  {col}: {valid_count} 有效值 ({valid_pct:.1f}%), 均值: {mean_val:.2f}")
            else:
                print(f"  {col}: {valid_count} 有效值 ({valid_pct:.1f}%), 无有效数据")
        
        # 按高度分析风速
        wind_speed_cols = [col for col in df_wide.columns if 'wind_speed' in col]
        if wind_speed_cols:
            print(f"\n风速统计:")
            for col in sorted(wind_speed_cols):
                valid_data = df_wide[col].dropna()
                if len(valid_data) > 0:
                    stats = valid_data.describe()
                    print(f"  {col}:")
                    print(f"    平均: {stats['mean']:.2f} m/s")
                    print(f"    最大: {stats['max']:.2f} m/s")
                    print(f"    最小: {stats['min']:.2f} m/s")
                    print(f"    标准差: {stats['std']:.2f} m/s")
                else:
                    print(f"  {col}: 无有效数据")

def main():
    """主函数 - 处理昌马测风塔数据"""
    # 设置昌马数据路径
    changma_path = "/Users/xiaxin/work/WindForecast_Project/01_Data/raw/obs/changma"
    
    # 初始化处理器
    processor = ChangmaDataProcessorFixed(changma_path)
    
    print("="*60)
    print("开始昌马数据处理（修复版）")
    print("="*60)
    
    # 处理所有高度数据
    height_data = processor.process_all_heights()
    
    if height_data:
        print(f"\n成功处理 {len(height_data)} 个高度的数据")
        
        # 创建宽格式数据（使用修复版方法）
        df_wide = processor.create_wide_format_data_fixed(height_data)
        
        if not df_wide.empty:
            # 生成摘要
            processor.generate_data_summary(df_wide)
            
            # 保存数据
            output_file = processor.save_changma_data(df_wide)
            
            if output_file:
                print(f"\n🎉 昌马测风塔数据处理成功完成！")
                print(f"📁 输出文件: {output_file}")
                
                # 快速验证
                print(f"\n快速验证:")
                try:
                    test_df = pd.read_csv(output_file, index_col=0, parse_dates=True, nrows=5)
                    print(f"  ✓ 文件可正常读取")
                    print(f"  ✓ 数据形状: {test_df.shape}")
                    print(f"  ✓ 主要列: {[col for col in test_df.columns if 'wind_speed' in col]}")
                except Exception as e:
                    print(f"  ✗ 文件验证失败: {e}")
            else:
                print(f"\n❌ 数据保存失败")
        else:
            print("❌ 宽格式数据创建失败")
    else:
        print("❌ 未能加载任何高度数据")

if __name__ == "__main__":
    main()