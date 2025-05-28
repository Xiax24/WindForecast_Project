import pandas as pd
import os
import glob
import re

def add_power_to_matched_data():
    """将各个风电场的power.csv数据添加到对应的matched_data文件中"""
    # 定义文件路径
    power_files = {
        'changma': '/Users/xiaxin/work/WindForecast_Project/01_Data/raw/obs/changma/power.csv',
        'sanlijijingzi': '/Users/xiaxin/work/WindForecast_Project/01_Data/raw/obs/sanlijijingzi/power.csv',
        'kuangqu': '/Users/xiaxin/work/WindForecast_Project/01_Data/raw/obs/kuangqu/power.csv'
    }
    matched_data_dir = '/Users/xiaxin/work/WindForecast_Project/01_Data/processed/matched_data'
    
    # 获取所有matched_data文件
    matched_files = glob.glob(os.path.join(matched_data_dir, "*_matched.csv"))
    print(f"找到 {len(matched_files)} 个matched_data文件")
    
    # 处理每个matched_data文件
    for file_path in matched_files:
        file_name = os.path.basename(file_path)
        site_name = file_name.split('_matched')[0]
        print(f"\n处理 {site_name} 的matched_data文件...")
        
        # 检查该站点是否有对应的power文件
        if site_name not in power_files:
            print(f"  警告: 没有找到 {site_name} 的power.csv文件")
            continue
        
        power_file = power_files[site_name]
        if not os.path.exists(power_file):
            print(f"  错误: {power_file} 不存在")
            continue
        
        try:
            # 读取matched_data文件
            matched_df = pd.read_csv(file_path)
            print(f"  成功读取matched_data文件，共 {len(matched_df)} 行")
            
            # 识别timestamp列
            timestamp_col = None
            for col in ['timestamp', 'time', 'date', 'datetime']:
                if col in matched_df.columns:
                    timestamp_col = col
                    break
            
            if not timestamp_col:
                for col in matched_df.columns:
                    if matched_df[col].dtype == 'object':
                        try:
                            pd.to_datetime(matched_df[col])
                            timestamp_col = col
                            print(f"  使用 {col} 作为时间戳列")
                            break
                        except:
                            continue
            
            if not timestamp_col:
                print(f"  错误: 无法识别 {file_name} 的时间戳列")
                continue
            
            # 确保时间戳是datetime类型
            matched_df[timestamp_col] = pd.to_datetime(matched_df[timestamp_col])
            
            # 读取power文件
            print(f"  读取 {site_name} 的功率数据: {power_file}")
            power_df = pd.read_csv(power_file)
            print(f"  成功读取功率数据，共 {len(power_df)} 行")
            
            # 识别power文件中的timestamp列
            power_timestamp_col = None
            for col in ['timestamp', 'time', 'date', 'datetime']:
                if col in power_df.columns:
                    power_timestamp_col = col
                    break
            
            if not power_timestamp_col:
                for col in power_df.columns:
                    if power_df[col].dtype == 'object':
                        try:
                            pd.to_datetime(power_df[col])
                            power_timestamp_col = col
                            print(f"  使用 {col} 作为功率数据的时间戳列")
                            break
                        except:
                            continue
            
            if not power_timestamp_col:
                print(f"  错误: 无法识别 {power_file} 的时间戳列")
                continue
            
            # 确保power数据的时间戳是datetime类型
            power_df[power_timestamp_col] = pd.to_datetime(power_df[power_timestamp_col])
            
            # 识别power列
            power_col = None
            for col in power_df.columns:
                if col.lower() == 'power':
                    power_col = col
                    break
            
            if not power_col:
                for col in power_df.columns:
                    if ('power' in col.lower() or 'output' in col.lower() or 'generation' in col.lower()):
                        power_col = col
                        print(f"  使用 {col} 作为功率列")
                        break
            
            if not power_col:
                for col in power_df.columns:
                    if col != power_timestamp_col and pd.api.types.is_numeric_dtype(power_df[col]):
                        power_col = col
                        print(f"  使用 {col} 作为功率列")
                        break
            
            if not power_col:
                print(f"  错误: 无法识别 {power_file} 的功率列")
                continue
            
            # 创建一个只包含时间戳和功率的数据框
            power_data = power_df[[power_timestamp_col, power_col]].copy()
            power_data.columns = [power_timestamp_col, 'power']
            
            # 显示时间戳范围
            print(f"  Matched数据时间范围: {matched_df[timestamp_col].min()} 到 {matched_df[timestamp_col].max()}")
            print(f"  功率数据时间范围: {power_data[power_timestamp_col].min()} 到 {power_data[power_timestamp_col].max()}")
            
            # 根据时间戳合并数据
            if 'power' in matched_df.columns:
                print(f"  警告: {file_name} 中已存在power列，将被覆盖")
                matched_df = matched_df.drop(columns=['power'])
            
            # 合并数据
            merged_df = pd.merge(
                matched_df,
                power_data,
                left_on=timestamp_col,
                right_on=power_timestamp_col,
                how='left'
            )
            
            # 如果power_timestamp_col不同于timestamp_col，删除多余的列
            if power_timestamp_col != timestamp_col and power_timestamp_col in merged_df.columns:
                merged_df = merged_df.drop(columns=[power_timestamp_col])
            
            # 检查合并后的数据
            power_count = merged_df['power'].notna().sum()
            print(f"  合并后数据共 {len(merged_df)} 行，其中 {power_count} 行有功率数据 ({power_count/len(merged_df)*100:.1f}%)")
            
            # 保存更新后的数据
            merged_df.to_csv(file_path, index=False)
            print(f"  成功将功率数据添加到 {file_name}")
            
        except Exception as e:
            print(f"  处理 {file_name} 时出错: {e}")
    
    print("\n所有matched_data文件处理完成！")

def update_description_files():
    """根据matched_data文件中的power列更新风电场的描述文件"""
    matched_data_dir = '/Users/xiaxin/work/WindForecast_Project/01_Data/processed/matched_data'
    
    # 获取所有matched_data CSV文件
    matched_files = glob.glob(os.path.join(matched_data_dir, "*_matched.csv"))
    print(f"\n找到 {len(matched_files)} 个matched_data文件")
    
    # 获取所有description文件
    description_files = glob.glob(os.path.join(matched_data_dir, "*_*escription.txt"))
    print(f"找到 {len(description_files)} 个description文件")
    
    # 处理每个风电场
    for matched_file in matched_files:
        # 提取站点名称
        file_name = os.path.basename(matched_file)
        site_name = file_name.split('_matched')[0]
        
        print(f"\n====== 处理风电场 {site_name} ======")
        
        # 查找对应的description文件
        description_file = None
        for desc_file in description_files:
            if site_name in os.path.basename(desc_file):
                description_file = desc_file
                break
        
        if not description_file:
            print(f"  警告: 未找到 {site_name} 的描述文件")
            continue
        
        # 读取matched数据文件
        try:
            matched_df = pd.read_csv(matched_file)
            print(f"  读取matched数据: {len(matched_df)} 行")
            
            # 检查是否有power列
            if 'power' not in matched_df.columns:
                print(f"  警告: {site_name} 的matched数据中没有power列")
                continue
            
            # 找到时间戳列
            timestamp_col = None
            for col in ['timestamp', 'time', 'date', 'datetime']:
                if col in matched_df.columns:
                    timestamp_col = col
                    break
            
            if timestamp_col:
                matched_df[timestamp_col] = pd.to_datetime(matched_df[timestamp_col])
            
            # 计算功率统计信息
            power_stats = {
                'count': matched_df['power'].count(),
                'mean': matched_df['power'].mean(),
                'std': matched_df['power'].std(),
                'min': matched_df['power'].min(),
                '25%': matched_df['power'].quantile(0.25),
                '50%': matched_df['power'].quantile(0.5),
                '75%': matched_df['power'].quantile(0.75),
                'max': matched_df['power'].max()
            }
            
            # 输出功率统计信息预览
            print(f"  功率统计:")
            print(f"    数据点数: {power_stats['count']}")
            print(f"    平均功率: {power_stats['mean']:.2f}")
            print(f"    最大功率: {power_stats['max']:.2f}")
            print(f"    最小功率: {power_stats['min']:.2f}")
            
            # 获取时间范围
            if timestamp_col and not matched_df[timestamp_col].empty:
                time_range = {
                    'start': matched_df[timestamp_col].min(),
                    'end': matched_df[timestamp_col].max(),
                    'period': (matched_df[timestamp_col].max() - matched_df[timestamp_col].min()).days
                }
            else:
                time_range = None
            
            # 计算功率列的完整性
            total_rows = len(matched_df)
            valid_power_rows = matched_df['power'].count()
            power_completion = valid_power_rows / total_rows * 100 if total_rows > 0 else 0
            
            # 生成功率描述内容
            power_description = []
            power_description.append("功率数据信息:")
            power_description.append(f"  数据行数: {len(matched_df)}")
            
            if time_range:
                power_description.append(f"  时间范围: {time_range['start']} 到 {time_range['end']}")
                power_description.append(f"  数据期间: {time_range['period']} 天")
            
            power_description.append("  单位: kW 或 MW (根据风电场规模)")
            
            power_description.append("\n功率数据完整性:")
            power_description.append(f"  power: {power_stats['count']}/{len(matched_df)} ({power_completion:.1f}%)")
            
            power_description.append("\n功率数据统计:")
            power_description.append(f"  计数: {power_stats['count']}")
            power_description.append(f"  平均值: {power_stats['mean']:.2f}")
            power_description.append(f"  标准差: {power_stats['std']:.2f}")
            power_description.append(f"  最小值: {power_stats['min']:.2f}")
            power_description.append(f"  25%分位数: {power_stats['25%']:.2f}")
            power_description.append(f"  中位数: {power_stats['50%']:.2f}")
            power_description.append(f"  75%分位数: {power_stats['75%']:.2f}")
            power_description.append(f"  最大值: {power_stats['max']:.2f}")
            
            power_description = "\n".join(power_description)
            
            # 读取description文件
            print(f"  更新描述文件: {os.path.basename(description_file)}")
            with open(description_file, 'r', encoding='utf-8') as f:
                description_content = f.read()
            
            # 查找现有的功率描述部分
            power_section_pattern = r"功率数据信息:.*?(?=\n\n|$)"
            power_section_match = re.search(power_section_pattern, description_content, re.DOTALL)
            
            if power_section_match:
                # 如果已有功率描述，则替换
                updated_description = re.sub(power_section_pattern, power_description, description_content, flags=re.DOTALL)
            else:
                # 如果没有功率描述，则添加到统计部分之前
                stats_pattern = r"基本统计:"
                stats_match = re.search(stats_pattern, description_content)
                
                if stats_match:
                    # 在统计部分之前添加功率描述
                    pos = stats_match.start()
                    updated_description = description_content[:pos] + "\n" + power_description + "\n\n" + description_content[pos:]
                else:
                    # 直接添加到末尾
                    updated_description = description_content + "\n\n" + power_description
            
            # 保存更新后的描述
            with open(description_file, 'w', encoding='utf-8') as f:
                f.write(updated_description)
            
            print(f"  成功更新描述文件")
        
        except Exception as e:
            print(f"  处理 {site_name} 时出错: {e}")
    
    print("\n所有描述文件更新完成!")

if __name__ == "__main__":
    print("====== 风电场功率数据处理 ======")
    
    # 步骤1: 将power.csv数据添加到matched_data文件
    print("\n第一步: 添加功率数据到matched_data文件")
    add_power_to_matched_data()
    
    # 步骤2: 更新description文件
    print("\n第二步: 更新风电场描述文件")
    update_description_files()
    
    print("\n处理完成！请检查以下文件是否已更新:")
    print("- 昌马风电场: changma_matched.csv 和 changma_*escription.txt")
    print("- 矿区风电场: kuangqu_matched.csv 和 kuangqu_*escription.txt")
    print("- 三十里井子风电场: sanlijijingzi_matched.csv 和 sanlijijingzi_*escription.txt")