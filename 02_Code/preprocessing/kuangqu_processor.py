"""
矿区测风塔数据处理脚本（修复版）
专门处理矿区数据的特殊格式：日期和时间分离的情况
"""

import pandas as pd
import numpy as np
import os
import glob
import re
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

class KuangquDataProcessorFixed:
    def __init__(self, base_path):
        """
        初始化矿区数据处理器
        
        参数:
        base_path: 矿区数据根目录路径
        """
        self.base_path = base_path
        self.processed_path = os.path.join(base_path, '..', '..', 'processed', 'cleaned')
        
        # 确保输出目录存在
        os.makedirs(self.processed_path, exist_ok=True)
        
        # 缺失值标记
        self.missing_values = [-99.0, -99, '-99.0', '-99', 'NULL', 'null', '', ' ', '\\N', 0.0]
        
        # 矿区测风塔高度（从文件名看，有30m, 50m, 70m）
        self.available_heights = [30, 50, 70]

    def extract_height_from_filename(self, filename):
        """从矿区文件名中提取高度信息"""
        pattern = r'测风塔(\d+)信息报表'
        match = re.search(pattern, filename)
        
        if match:
            height_str = match.group(1)
            try:
                height = int(height_str)
                if height in [30, 50, 70]:
                    return height
            except ValueError:
                pass
        
        return None

    def extract_date_from_filename(self, filename):
        """从矿区文件名中提取日期信息"""
        date_pattern = r'(\d{4}-\d{2}-\d{2})-(\d{4}-\d{2}-\d{2})'
        match = re.search(date_pattern, filename)
        
        if match:
            start_date_str = match.group(1)
            end_date_str = match.group(2)
            
            try:
                start_date = datetime.strptime(start_date_str, '%Y-%m-%d')
                end_date = datetime.strptime(end_date_str, '%Y-%m-%d')
                return start_date, end_date
            except ValueError:
                pass
        
        return None, None

    def process_kuangqu_data(self, df):
        """处理矿区数据的特殊格式"""
        if df.empty:
            return df
        
        df_processed = df.copy()
        
        print(f"      原始列名: {df_processed.columns.tolist()}")
        
        # 矿区数据特殊处理：合并日期和时间列
        if '日期' in df_processed.columns and '时间' in df_processed.columns:
            print(f"      合并日期和时间列")
            
            # 处理日期和时间数据
            dates = df_processed['日期'].astype(str)
            times = df_processed['时间'].astype(str)
            
            # 组合成完整的datetime字符串
            datetime_strings = dates + ' ' + times
            
            # 转换为datetime
            try:
                df_processed['datetime'] = pd.to_datetime(datetime_strings, errors='coerce')
                print(f"      成功创建datetime列")
                
                # 删除原始的日期和时间列
                df_processed = df_processed.drop(['日期', '时间'], axis=1)
                
            except Exception as e:
                print(f"      datetime转换失败: {e}")
                print(f"      日期样本: {dates.head().tolist()}")
                print(f"      时间样本: {times.head().tolist()}")
                return pd.DataFrame()
        
        # 标准化其他列名
        column_mapping = {
            '层高': 'layer_height',
            '风向': 'wind_direction',
            '风速': 'wind_speed',
            '气温': 'temperature',
            '气压': 'pressure',
            '湿度': 'humidity',
            '空气密度': 'density'
        }
        
        for old_name, new_name in column_mapping.items():
            if old_name in df_processed.columns:
                df_processed = df_processed.rename(columns={old_name: new_name})
        
        print(f"      处理后列名: {df_processed.columns.tolist()}")
        
        return df_processed

    def clean_kuangqu_data(self, df, height_value):
        """清理矿区数据"""
        if df.empty:
            return df
            
        df_clean = df.copy()
        
        # 添加高度信息
        df_clean['height'] = height_value
        
        # 处理缺失值标记（矿区数据很多变量显示为0.0，实际是缺失）
        for col in ['temperature', 'pressure', 'humidity']:
            if col in df_clean.columns:
                # 将0.0视为缺失值（因为气温、气压、湿度不可能为0）
                df_clean[col] = df_clean[col].replace(self.missing_values, np.nan)
        
        # 处理密度数据（0.0也视为缺失）
        if 'density' in df_clean.columns:
            df_clean['density'] = df_clean['density'].replace([0.0], np.nan)
        
        # 风速和风向数据检查
        if 'wind_speed' in df_clean.columns:
            mask_negative = df_clean['wind_speed'] < 0
            mask_extreme = df_clean['wind_speed'] > 50
            
            if mask_negative.sum() > 0:
                print(f"      发现 {mask_negative.sum()} 个负风速值，设为NaN")
                df_clean.loc[mask_negative, 'wind_speed'] = np.nan
            
            if mask_extreme.sum() > 0:
                print(f"      发现 {mask_extreme.sum()} 个超过50m/s的极端风速值，设为NaN")
                df_clean.loc[mask_extreme, 'wind_speed'] = np.nan
        
        if 'wind_direction' in df_clean.columns:
            mask_wd_invalid = (df_clean['wind_direction'] < 0) | (df_clean['wind_direction'] > 360)
            if mask_wd_invalid.sum() > 0:
                print(f"      发现 {mask_wd_invalid.sum()} 个无效风向值，设为NaN")
                df_clean.loc[mask_wd_invalid, 'wind_direction'] = np.nan
        
        # 统计数据质量
        key_vars = ['wind_speed', 'wind_direction', 'temperature', 'humidity', 'pressure', 'density']
        existing_vars = [v for v in key_vars if v in df_clean.columns]
        
        if existing_vars:
            print(f"      数据质量检查:")
            for var in existing_vars:
                missing_pct = df_clean[var].isnull().mean() * 100
                print(f"        {var}: {missing_pct:.1f}% 缺失")
        
        return df_clean

    def load_height_data(self, file_list, height_value):
        """加载特定高度的所有数据文件"""
        print(f"  处理 {height_value}m 高度数据...")
        print(f"    文件数量: {len(file_list)}")
        
        all_data = []
        
        for i, file_path in enumerate(file_list):
            filename = os.path.basename(file_path)
            print(f"    处理文件 {i+1}/{len(file_list)}: {filename}")
            
            df = None
            # 使用xlrd引擎读取.xls文件
            try:
                df = pd.read_excel(file_path, engine='xlrd')
                print(f"      使用xlrd引擎成功读取")
                print(f"      原始数据形状: {df.shape}")
            except Exception as e:
                print(f"      读取失败: {e}")
                continue
            
            if df is not None and not df.empty:
                try:
                    # 处理数据格式
                    df_processed = self.process_kuangqu_data(df)
                    
                    if df_processed.empty:
                        print(f"      数据处理后为空")
                        continue
                    
                    # 清理数据
                    df_cleaned = self.clean_kuangqu_data(df_processed, height_value)
                    
                    if not df_cleaned.empty and 'datetime' in df_cleaned.columns:
                        # 按时间排序
                        df_cleaned = df_cleaned.sort_values('datetime')
                        df_cleaned = df_cleaned.reset_index(drop=True)
                        
                        print(f"      处理后数据形状: {df_cleaned.shape}")
                        print(f"      时间范围: {df_cleaned['datetime'].min()} 到 {df_cleaned['datetime'].max()}")
                        
                        all_data.append(df_cleaned)
                    else:
                        print(f"      数据清理后为空或缺少时间列")
                        
                except Exception as e:
                    print(f"      数据处理错误: {e}")
        
        if all_data:
            # 合并所有数据
            combined_df = pd.concat(all_data, ignore_index=True)
            
            # 最终按时间排序
            if 'datetime' in combined_df.columns:
                combined_df = combined_df.sort_values('datetime')
                combined_df = combined_df.reset_index(drop=True)
                
            print(f"    {height_value}m 合并后数据形状: {combined_df.shape}")
            return combined_df
        else:
            print(f"    {height_value}m 没有有效数据")
            return pd.DataFrame()

    def explore_kuangqu_structure(self):
        """探索矿区数据的文件结构"""
        print("探索矿区测风塔数据结构...")
        print(f"基础路径: {self.base_path}")
        
        if not os.path.exists(self.base_path):
            print(f"错误：路径不存在 {self.base_path}")
            return {}
        
        # 查找所有Excel文件
        excel_files = glob.glob(os.path.join(self.base_path, "*.xls*"))
        excel_files.sort()
        
        print(f"\n找到 {len(excel_files)} 个Excel文件")
        
        # 按高度分类文件
        height_files = {30: [], 50: [], 70: []}
        
        for file_path in excel_files:
            filename = os.path.basename(file_path)
            height = self.extract_height_from_filename(filename)
            
            if height and height in height_files:
                height_files[height].append(file_path)
        
        # 统计各高度文件数量
        print(f"\n按高度分类统计:")
        for height in sorted(height_files.keys()):
            print(f"  {height}m: {len(height_files[height])} 个文件")
        
        return height_files

    def process_all_heights(self):
        """处理所有高度的数据"""
        print("开始处理矿区测风塔的所有高度数据...")
        
        # 探索文件结构
        height_files = self.explore_kuangqu_structure()
        
        if not height_files:
            print("未找到有效的数据文件")
            return {}
        
        height_data = {}
        
        for height in sorted(height_files.keys()):
            files = height_files[height]
            if files:
                print(f"\n{'='*50}")
                print(f"处理 {height}m 高度数据")
                print(f"{'='*50}")
                
                df = self.load_height_data(files, height)
                if not df.empty:
                    height_data[height] = df
                    print(f"✓ {height}m 数据处理成功")
                else:
                    print(f"✗ {height}m 数据处理失败")
            else:
                print(f"跳过 {height}m（无文件）")
        
        return height_data

    def create_wide_format_data(self, height_data):
        """将不同高度的数据合并为宽格式"""
        print("\n创建宽格式数据...")
        
        if not height_data:
            print("没有高度数据可合并")
            return pd.DataFrame()
        
        # 创建统一时间序列
        all_times = []
        for height, df in height_data.items():
            if 'datetime' in df.columns:
                times = df['datetime'].dropna().values
                all_times.extend(times)
                print(f"  {height}m: {len(times)} 个有效时间点")
        
        if not all_times:
            print("没有有效的时间数据")
            return pd.DataFrame()
        
        time_index = pd.DatetimeIndex(all_times).drop_duplicates().sort_values()
        print(f"统一时间索引: {len(time_index)} 个时间点")
        print(f"时间范围: {time_index.min()} 到 {time_index.max()}")
        
        # 初始化宽格式DataFrame
        df_wide = pd.DataFrame(index=time_index)
        
        # 为每个高度添加变量
        variables = ['wind_speed', 'wind_direction', 'temperature', 'humidity', 'pressure', 'density']
        
        for height in sorted(height_data.keys()):
            df_height = height_data[height]
            print(f"  添加 {height}m 数据...")
            
            # 设置时间为索引
            df_work = df_height.copy()
            df_work = df_work.set_index('datetime')
            
            # 处理重复时间
            if df_work.index.has_duplicates:
                print(f"    发现重复时间，取平均值")
                df_work = df_work.groupby(df_work.index).mean()
            
            for var in variables:
                if var in df_work.columns:
                    col_name = f"{var}_{height}m"
                    aligned_series = df_work[var].reindex(time_index)
                    df_wide[col_name] = aligned_series
                    
                    valid_count = df_wide[col_name].notna().sum()
                    valid_pct = valid_count / len(df_wide) * 100
                    print(f"    {col_name}: {valid_count} 有效值 ({valid_pct:.1f}%)")
        
        print(f"\n宽格式数据创建完成:")
        print(f"  形状: {df_wide.shape}")
        print(f"  列: {df_wide.columns.tolist()}")
        
        return df_wide

    def save_kuangqu_data(self, df_wide):
        """保存矿区处理后的数据"""
        if df_wide.empty:
            print("没有数据可保存")
            return None
        
        # 生成文件名
        start_date = df_wide.index.min().strftime('%Y%m%d')
        end_date = df_wide.index.max().strftime('%Y%m%d')
        
        # 保存CSV文件
        csv_filename = f"kuangqu_{start_date}_{end_date}_cleaned.csv"
        csv_filepath = os.path.join(self.processed_path, csv_filename)
        
        try:
            df_wide.to_csv(csv_filepath)
            print(f"\n✓ 数据已保存到: {csv_filepath}")
            
            file_size = os.path.getsize(csv_filepath) / 1024
            print(f"  文件大小: {file_size:.1f} KB")
            
        except Exception as e:
            print(f"\n✗ 保存CSV文件失败: {e}")
            return None
        
        # 保存摘要文件
        summary_filename = f"kuangqu_{start_date}_{end_date}_summary.txt"
        summary_filepath = os.path.join(self.processed_path, summary_filename)
        
        try:
            with open(summary_filepath, 'w', encoding='utf-8') as f:
                f.write("矿区测风塔数据处理摘要\n")
                f.write("="*50 + "\n\n")
                f.write(f"处理时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
                f.write(f"数据形状: {df_wide.shape}\n")
                f.write(f"时间范围: {df_wide.index.min()} 到 {df_wide.index.max()}\n")
                f.write(f"数据期间: {(df_wide.index.max() - df_wide.index.min()).days} 天\n\n")
                
                f.write("高度覆盖: 30m, 50m, 70m（矿区站无10m数据）\n\n")
                
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
        print("矿区测风塔数据摘要")
        print("="*60)
        
        print(f"数据形状: {df_wide.shape}")
        print(f"时间范围: {df_wide.index.min()} 到 {df_wide.index.max()}")
        print(f"数据期间: {(df_wide.index.max() - df_wide.index.min()).days} 天")
        print(f"高度覆盖: 30m, 50m, 70m")
        
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
        
        # 风速统计
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

def main():
    """主函数 - 处理矿区测风塔数据"""
    # 设置矿区数据路径
    kuangqu_path = "/Users/xiaxin/work/WindForecast_Project/01_Data/raw/obs/kuangqu"
    
    # 初始化处理器
    processor = KuangquDataProcessorFixed(kuangqu_path)
    
    print("="*60)
    print("开始矿区测风塔数据处理（修复版）")
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
            output_file = processor.save_kuangqu_data(df_wide)
            
            if output_file:
                print(f"\n🎉 矿区测风塔数据处理成功完成！")
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