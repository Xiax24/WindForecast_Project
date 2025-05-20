"""
三十里井子观测数据15分钟重采样脚本
专门处理sanlijijingzi的5分钟数据，重采样为15分钟间隔
"""

import pandas as pd
import numpy as np
import os
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

class SanlijijingziResampler:
    def __init__(self):
        """
        初始化三十里井子数据重采样器
        """
        self.input_file = "/Users/xiaxin/work/WindForecast_Project/01_Data/raw/processed/cleaned/sanlijijingzi_20210601_20220616_cleaned.csv"
        self.output_path = "/Users/xiaxin/work/WindForecast_Project/01_Data/raw/processed/aligned"
        
        # 确保输出目录存在
        os.makedirs(self.output_path, exist_ok=True)
    
    def circular_mean(self, angles):
        """
        计算风向的圆形平均值
        
        参数:
        angles: 风向角度序列（度）
        
        返回:
        圆形平均角度
        """
        if len(angles) == 0 or angles.isna().all():
            return np.nan
        
        # 移除NaN值
        valid_angles = angles.dropna()
        if len(valid_angles) == 0:
            return np.nan
        
        # 转换为弧度
        radians = np.radians(valid_angles)
        
        # 计算平均向量
        sin_sum = np.sum(np.sin(radians))
        cos_sum = np.sum(np.cos(radians))
        
        # 计算平均角度
        mean_angle = np.degrees(np.arctan2(sin_sum, cos_sum))
        
        # 确保角度在0-360度范围内
        if mean_angle < 0:
            mean_angle += 360
            
        return mean_angle
    
    def resample_to_15min(self, df):
        """
        将5分钟数据重采样为15分钟数据
        
        参数:
        df: 输入数据框（5分钟间隔）
        
        返回:
        重采样后的数据框（15分钟间隔）
        """
        print("开始重采样到15分钟间隔...")
        print(f"原始数据形状: {df.shape}")
        print(f"原始时间范围: {df.index.min()} 到 {df.index.max()}")
        
        # 定义聚合规则
        agg_rules = {}
        
        for col in df.columns:
            if 'wind_direction' in col:
                # 风向变量：使用圆形平均
                agg_rules[col] = lambda x, col=col: self.circular_mean(x) if len(x) > 0 else np.nan
            elif 'wind_speed_max' in col:
                # 最大风速：取最大值
                agg_rules[col] = 'max'
            elif 'wind_speed_min' in col:
                # 最小风速：取最小值
                agg_rules[col] = 'min'
            elif 'wind_speed_instant' in col:
                # 瞬时风速：取最后一个值
                agg_rules[col] = 'last'
            else:
                # 其他变量（平均风速、温度、湿度等）：取平均值
                agg_rules[col] = 'mean'
        
        # 对风向单独处理
        wind_direction_cols = [col for col in df.columns if 'wind_direction' in col]
        other_cols = [col for col in df.columns if 'wind_direction' not in col]
        
        # 处理非风向变量
        other_agg_rules = {col: agg_rules[col] for col in other_cols}
        resampled_other = df[other_cols].resample('15min').agg(other_agg_rules)
        
        # 处理风向变量
        resampled_wd = pd.DataFrame(index=resampled_other.index)
        for col in wind_direction_cols:
            print(f"  处理风向变量: {col}")
            resampled_wd[col] = df[col].resample('15min').apply(self.circular_mean)
        
        # 合并结果
        resampled_df = pd.concat([resampled_other, resampled_wd], axis=1)
        
        # 确保列顺序与原始数据一致
        resampled_df = resampled_df[df.columns]
        
        print(f"重采样后数据形状: {resampled_df.shape}")
        print(f"重采样后时间范围: {resampled_df.index.min()} 到 {resampled_df.index.max()}")
        
        # 数据压缩比
        compression_ratio = len(df) / len(resampled_df)
        print(f"数据压缩比: {compression_ratio:.1f}:1")
        
        return resampled_df
    
    def save_results(self, resampled_df):
        """
        保存重采样结果
        
        参数:
        resampled_df: 重采样后的数据框
        """
        # 生成输出文件名
        output_csv = os.path.join(self.output_path, "sanlijijingzi_20210601_20220616_15min_aligned.csv")
        output_summary = os.path.join(self.output_path, "sanlijijingzi_20210601_20220616_15min_aligned_summary.txt")
        
        # 保存CSV文件
        try:
            resampled_df.to_csv(output_csv)
            print(f"\n✓ 数据已保存到: {output_csv}")
            
            # 检查文件大小
            file_size = os.path.getsize(output_csv) / 1024  # KB
            print(f"  文件大小: {file_size:.1f} KB")
            
        except Exception as e:
            print(f"\n✗ 保存CSV文件失败: {e}")
            return None
        
        # 保存摘要文件
        try:
            with open(output_summary, 'w', encoding='utf-8') as f:
                f.write("三十里井子风电场15分钟重采样摘要\n")
                f.write("="*50 + "\n\n")
                f.write(f"处理时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
                f.write(f"输入文件: {self.input_file}\n")
                f.write(f"输出文件: {output_csv}\n")
                f.write(f"重采样间隔: 15分钟\n\n")
                
                f.write(f"数据形状: {resampled_df.shape}\n")
                f.write(f"时间范围: {resampled_df.index.min()} 到 {resampled_df.index.max()}\n")
                f.write(f"数据期间: {(resampled_df.index.max() - resampled_df.index.min()).days} 天\n\n")
                
                f.write("重采样规则:\n")
                f.write("  风向变量: 圆形平均值\n")
                f.write("  最大风速: 取最大值\n")
                f.write("  最小风速: 取最小值\n")
                f.write("  瞬时风速: 取最后值\n")
                f.write("  其他变量: 平均值\n\n")
                
                f.write("变量完整性:\n")
                for col in resampled_df.columns:
                    valid_count = resampled_df[col].notna().sum()
                    total_count = len(resampled_df)
                    valid_pct = valid_count / total_count * 100
                    f.write(f"  {col}: {valid_count}/{total_count} ({valid_pct:.1f}%)\n")
                
                f.write("\n基本统计:\n")
                f.write(resampled_df.describe().to_string())
            
            print(f"✓ 摘要已保存到: {output_summary}")
            
        except Exception as e:
            print(f"✗ 保存摘要文件失败: {e}")
        
        return output_csv
    
    def generate_data_summary(self, resampled_df):
        """
        生成控制台数据摘要
        """
        print("\n" + "="*60)
        print("三十里井子风电场15分钟重采样摘要")
        print("="*60)
        
        # 基本信息
        print(f"数据形状: {resampled_df.shape}")
        print(f"时间范围: {resampled_df.index.min()} 到 {resampled_df.index.max()}")
        print(f"数据期间: {(resampled_df.index.max() - resampled_df.index.min()).days} 天")
        
        # 变量完整性统计
        print(f"\n变量完整性:")
        for col in resampled_df.columns:
            valid_count = resampled_df[col].notna().sum()
            total_count = len(resampled_df)
            valid_pct = valid_count / total_count * 100
            print(f"  {col}: {valid_count}/{total_count} ({valid_pct:.1f}%)")
        
        # 风速统计
        wind_speed_cols = [col for col in resampled_df.columns if 'wind_speed' in col and 'max' not in col and 'min' not in col and 'std' not in col and 'instant' not in col]
        if wind_speed_cols:
            print(f"\n平均风速统计:")
            for col in sorted(wind_speed_cols):
                valid_data = resampled_df[col].dropna()
                if len(valid_data) > 0:
                    stats = valid_data.describe()
                    print(f"  {col}: 均值 {stats['mean']:.2f} m/s, 最大 {stats['max']:.2f} m/s")
    
    def process(self):
        """
        主处理函数
        """
        print("="*60)
        print("开始三十里井子风电场数据15分钟重采样")
        print("="*60)
        
        # 检查输入文件是否存在
        if not os.path.exists(self.input_file):
            print(f"✗ 输入文件不存在: {self.input_file}")
            return
        
        try:
            # 读取数据
            print(f"读取输入文件: {os.path.basename(self.input_file)}")
            df = pd.read_csv(self.input_file, index_col=0, parse_dates=True)
            print(f"✓ 成功读取文件，数据形状: {df.shape}")
            
            # 重采样
            resampled_df = self.resample_to_15min(df)
            
            if resampled_df.empty:
                print("✗ 重采样后数据为空")
                return
            
            # 生成摘要
            self.generate_data_summary(resampled_df)
            
            # 保存结果
            output_file = self.save_results(resampled_df)
            
            if output_file:
                print(f"\n🎉 三十里井子风电场15分钟重采样完成！")
                print(f"📁 输出文件: {output_file}")
                
                # 快速验证
                print(f"\n快速验证:")
                try:
                    test_df = pd.read_csv(output_file, index_col=0, parse_dates=True, nrows=5)
                    print(f"  ✓ 文件可正常读取")
                    print(f"  ✓ 样本数据形状: {test_df.shape}")
                    wind_cols = [col for col in test_df.columns if 'wind_speed' in col][:3]
                    print(f"  ✓ 主要风速列: {wind_cols}")
                except Exception as e:
                    print(f"  ✗ 文件验证失败: {e}")
            else:
                print(f"\n❌ 数据保存失败")
                
        except Exception as e:
            print(f"✗ 处理过程中发生错误: {e}")
            import traceback
            traceback.print_exc()

def main():
    """主函数"""
    resampler = SanlijijingziResampler()
    resampler.process()

if __name__ == "__main__":
    main()