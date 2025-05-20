"""
三十里井子风电场数据处理脚本（修复版）
基于昌马数据处理脚本修改，处理日期时间分列的格式
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

class SanlijijingziDataProcessor:
    def __init__(self, base_path):
        """
        初始化三十里井子风电场数据处理器
        
        参数:
        base_path: 三十里井子数据根目录路径
        """
        self.base_path = base_path
        self.processed_path = '/Users/xiaxin/work/WindForecast_Project/01_Data/raw/processed/cleaned'
        
        # 确保输出目录存在
        os.makedirs(self.processed_path, exist_ok=True)
        
        # 缺失值标记
        self.missing_values = [-99.0, -99, '-99.0', '-99', 'NULL', 'null', '', ' ', '\\N']
        
        # 高度映射
        self.height_mapping = {
            '10米': 10, '10m': 10, '10': 10,
            '30米': 30, '30m': 30, '30': 30,
            '50米': 50, '50m': 50, '50': 50,
            '70米': 70, '70m': 70, '70': 70,
            '100米': 100, '100m': 100, '100': 100
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
            if int(num) in [10, 30, 50, 70, 100]:
                return int(num)
        
        return None

    def standardize_sanlijijingzi_columns(self, df):
        """标准化三十里井子数据的列名"""
        df_std = df.copy()
        
        # 三十里井子的特定列名映射
        column_mapping = {
            '日期': 'date',
            '时间': 'time', 
            '风速最大值(m/s)': 'wind_speed_max',
            '风速最小值(m/s)': 'wind_speed_min',
            '风速平均值(m/s)': 'wind_speed',
            '风速标准偏差(m/s)': 'wind_speed_std',
            '瞬时风速(m/s)': 'wind_speed_instant',
            '平均风向(°)': 'wind_direction',
            '瞬时风向(°)': 'wind_direction_instant',
            # 可能的其他变体
            '实测风速': 'wind_speed', 
            '风速': 'wind_speed', 
            '实测风向': 'wind_direction', 
            '风向': 'wind_direction',
            '实测温度': 'temperature', 
            '温度': 'temperature',
            '实测湿度': 'humidity', 
            '湿度': 'humidity',
            '实测气压': 'pressure', 
            '气压': 'pressure',
            '大气密度': 'density', 
            '密度': 'density'
        }
        
        # 重命名列
        for old_name, new_name in column_mapping.items():
            if old_name in df_std.columns:
                df_std = df_std.rename(columns={old_name: new_name})
        
        return df_std

    def process_sanlijijingzi_datetime(self, df):
        """处理三十里井子数据的时间列（日期和时间分列）"""
        if 'date' not in df.columns or 'time' not in df.columns:
            print(f"      警告: 未找到日期时间列，可用列: {list(df.columns)}")
            return df
        
        df_time = df.copy()
        
        try:
            original_count = len(df_time)
            
            # 移除明显无效的时间值
            df_time = df_time.dropna(subset=['date', 'time'])
            
            # 合并日期和时间列，创建datetime列
            # 将日期和时间转换为字符串，然后合并
            df_time['date_str'] = df_time['date'].astype(str)
            df_time['time_str'] = df_time['time'].astype(str)
            
            # 合并日期和时间
            df_time['datetime_str'] = df_time['date_str'] + ' ' + df_time['time_str']
            
            # 转换为datetime
            df_time['datetime'] = pd.to_datetime(df_time['datetime_str'], errors='coerce')
            
            # 移除转换失败的行
            df_time = df_time.dropna(subset=['datetime'])
            
            # 删除临时列
            df_time = df_time.drop(['date_str', 'time_str', 'datetime_str'], axis=1)
            
            final_count = len(df_time)
            if final_count < original_count:
                print(f"      时间处理: {original_count} -> {final_count} 行")
            
            if not df_time.empty:
                print(f"      时间范围: {df_time['datetime'].min()} 到 {df_time['datetime'].max()}")
            
        except Exception as e:
            print(f"      时间处理错误: {e}")
            if 'date' in df.columns and 'time' in df.columns:
                print(f"      日期列样本: {df['date'].head().tolist()}")
                print(f"      时间列样本: {df['time'].head().tolist()}")
        
        return df_time

    def detect_stuck_values(self, df, min_consecutive=5):
        """
        检测和清理僵值（连续相同的数值）
        
        参数:
        df: 数据框
        min_consecutive: 认为是僵值的最小连续相同值数量，默认为5
        
        返回:
        清理后的数据框
        """
        df_cleaned = df.copy()
        
        # 需要检查僵值的数值列
        numeric_columns = df_cleaned.select_dtypes(include=[np.number]).columns
        # 排除不需要检查的列
        exclude_cols = ['height']
        check_columns = [col for col in numeric_columns if col not in exclude_cols]
        
        total_stuck_removed = 0
        
        for col in check_columns:
            if col in df_cleaned.columns and df_cleaned[col].notna().sum() > 0:
                # 创建一个用于标记连续相同值的系列
                series = df_cleaned[col].copy()
                
                # 找出非NaN值的位置
                non_nan_mask = series.notna()
                
                if non_nan_mask.sum() < min_consecutive:
                    continue  # 如果有效数据不足min_consecutive个，跳过
                
                # 检测连续相同值
                # 方法：比较当前值与前一个值是否相同
                is_same_as_previous = series == series.shift(1)
                
                # 创建组ID：每当值发生变化时，组ID就增加
                group_id = (~is_same_as_previous).cumsum()
                
                # 对每个组计算大小
                group_sizes = series.groupby(group_id).transform('size')
                
                # 标记连续相同值>=min_consecutive的位置为僵值
                stuck_mask = (group_sizes >= min_consecutive) & non_nan_mask
                
                if stuck_mask.sum() > 0:
                    # 将僵值设为NaN
                    df_cleaned.loc[stuck_mask, col] = np.nan
                    stuck_count = stuck_mask.sum()
                    total_stuck_removed += stuck_count
                    print(f"      {col}: 检测到 {stuck_count} 个僵值，已设为NaN")
        
        if total_stuck_removed > 0:
            print(f"      僵值检测: 总计清理 {total_stuck_removed} 个僵值")
        else:
            print(f"      僵值检测: 未发现僵值")
            
        return df_cleaned

    def clean_sanlijijingzi_data(self, df):
        """清理三十里井子数据"""
        if df.empty:
            return df
            
        df_clean = df.copy()
        
        # 处理缺失值标记
        for col in df_clean.columns:
            if col not in ['height', 'date', 'time', 'datetime']:
                df_clean[col] = df_clean[col].replace(self.missing_values, np.nan)
        
        # 数据有效性检查 - 检查所有风速相关变量
        wind_speed_vars = ['wind_speed', 'wind_speed_max', 'wind_speed_min', 'wind_speed_instant']
        for var in wind_speed_vars:
            if var in df_clean.columns:
                mask_negative = df_clean[var] < 0
                mask_extreme = df_clean[var] > 50
                
                if mask_negative.sum() > 0:
                    print(f"      发现 {mask_negative.sum()} 个负{var}值，设为NaN")
                    df_clean.loc[mask_negative, var] = np.nan
                
                if mask_extreme.sum() > 0:
                    print(f"      发现 {mask_extreme.sum()} 个超过50m/s的极端{var}值，设为NaN")
                    df_clean.loc[mask_extreme, var] = np.nan
        
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
        wind_direction_vars = ['wind_direction', 'wind_direction_instant']
        for var in wind_direction_vars:
            if var in df_clean.columns:
                mask_wd_invalid = (df_clean[var] < 0) | (df_clean[var] > 360)
                if mask_wd_invalid.sum() > 0:
                    print(f"      发现 {mask_wd_invalid.sum()} 个无效{var}值，设为NaN")
                    df_clean.loc[mask_wd_invalid, var] = np.nan
        
        # 僵值检测和清理
        print(f"      执行僵值检测（连续5个相同值）...")
        df_clean = self.detect_stuck_values(df_clean, min_consecutive=5)
        
        # 统计清理效果
        if len(df_clean) > 0:
            key_vars = ['wind_speed', 'wind_speed_max', 'wind_speed_min', 'wind_speed_instant', 
                       'wind_direction', 'wind_direction_instant', 'temperature', 'humidity', 'pressure']
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
            
            # 跳过15分钟报表文件
            if '15分钟报表' in filename:
                print(f"      跳过15分钟报表文件，保持数据格式一致")
                continue
            
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
                    print(f"      列名: {list(df.columns)}")
                    
                    # 标准化列名
                    df_cleaned = self.standardize_sanlijijingzi_columns(df)
                    
                    # 处理时间列
                    df_cleaned = self.process_sanlijijingzi_datetime(df_cleaned)
                    
                    # 添加高度信息
                    df_cleaned['height'] = height_value
                    
                    # 处理缺失值和异常值
                    df_cleaned = self.clean_sanlijijingzi_data(df_cleaned)
                    
                    if not df_cleaned.empty:
                        all_data.append(df_cleaned)
                        print(f"      清理后数据形状: {df_cleaned.shape}")
                    else:
                        print(f"      数据清理后为空")
                        
                except Exception as e:
                    print(f"      数据处理错误: {e}")
                    import traceback
                    traceback.print_exc()
        
        if all_data:
            # 合并所有数据
            combined_df = pd.concat(all_data, ignore_index=True)
            
            # 按时间排序
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
        print("开始处理三十里井子风电场的所有高度数据...")
        
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

    def create_wide_format_data(self, height_data):
        """将不同高度的数据合并为宽格式"""
        print("\n创建宽格式数据...")
        
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
        variables = ['wind_speed', 'wind_speed_max', 'wind_speed_min', 'wind_speed_instant', 
                    'wind_direction', 'wind_direction_instant', 'wind_speed_std',
                    'temperature', 'humidity', 'pressure', 'density']
        
        for height in sorted(valid_height_data.keys()):
            df_height = valid_height_data[height]
            print(f"  添加 {height}m 数据...")
            
            # 设置时间为索引（创建副本避免修改原数据）
            df_work = df_height.copy()
            df_work = df_work.set_index('datetime')
            
            # 处理重复时间（取平均值）
            if df_work.index.has_duplicates:
                print(f"    发现重复时间，取平均值")
                numeric_columns = df_work.select_dtypes(include=[np.number]).columns
                df_work = df_work[numeric_columns].groupby(df_work.index).mean()
            
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
        print(f"  列: {list(df_wide.columns)}")
        
        return df_wide

    def save_sanlijijingzi_data(self, df_wide):
        """保存三十里井子处理后的数据"""
        if df_wide.empty:
            print("没有数据可保存")
            return None
        
        # 生成文件名
        start_date = df_wide.index.min().strftime('%Y%m%d')
        end_date = df_wide.index.max().strftime('%Y%m%d')
        
        # 保存CSV文件
        csv_filename = f"sanlijijingzi_{start_date}_{end_date}_cleaned.csv"
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
        summary_filename = f"sanlijijingzi_{start_date}_{end_date}_summary.txt"
        summary_filepath = os.path.join(self.processed_path, summary_filename)
        
        try:
            with open(summary_filepath, 'w', encoding='utf-8') as f:
                f.write("三十里井子风电场数据处理摘要\n")
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
        print("三十里井子风电场数据摘要")
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
        wind_speed_cols = [col for col in df_wide.columns if 'wind_speed' in col and not any(x in col for x in ['max', 'min', 'std', 'instant'])]
        if wind_speed_cols:
            print(f"\n平均风速统计:")
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
    """主函数 - 处理三十里井子风电场数据"""
    # 设置三十里井子数据路径
    sanlijijingzi_path = "/Users/xiaxin/work/WindForecast_Project/01_Data/raw/obs/sanlijijingzi"
    
    # 初始化处理器
    processor = SanlijijingziDataProcessor(sanlijijingzi_path)
    
    print("="*60)
    print("开始三十里井子风电场数据处理（修复版）")
    print("="*60)
    
    # 处理所有高度数据
    height_data = processor.process_all_heights()
    
    if height_data:
        print(f"\n成功处理 {len(height_data)} 个高度的数据")
        
        # 创建宽格式数据
        df_wide = processor.create_wide_format_data(height_data)
        
        if not df_wide.empty:
            # 生成摘要
            processor.generate_data_summary(df_wide)
            
            # 保存数据
            output_file = processor.save_sanlijijingzi_data(df_wide)
            
            if output_file:
                print(f"\n🎉 三十里井子风电场数据处理成功完成！")
                print(f"📁 输出文件: {output_file}")
                
                # 快速验证
                print(f"\n快速验证:")
                try:
                    test_df = pd.read_csv(output_file, index_col=0, parse_dates=True, nrows=5)
                    print(f"  ✓ 文件可正常读取")
                    print(f"  ✓ 数据形状: {test_df.shape}")
                    wind_cols = [col for col in test_df.columns if 'wind_speed' in col]
                    print(f"  ✓ 主要列: {wind_cols}")
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