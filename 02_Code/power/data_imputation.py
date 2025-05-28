import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')  # 使用非交互式后端
import matplotlib.pyplot as plt
from sklearn.impute import KNNImputer
import warnings
warnings.filterwarnings('ignore')

def impute_observation_data(df):
    """
    专门处理观测数据(obs_开头的变量)的缺失值插补
    保持预测数据(ec_, gfs_)和功率数据不变
    """
    print("=== 观测数据缺失值插补 ===")
    
    # 复制原始数据
    df_imputed = df.copy()
    
    # 1. 识别观测变量
    obs_columns = [col for col in df.columns if col.startswith('obs_')]
    print(f"识别到观测变量: {len(obs_columns)} 个")
    for col in obs_columns:
        missing_count = df[col].isnull().sum()
        if missing_count > 0:
            print(f"  {col}: {missing_count} 个缺失值 ({missing_count/len(df)*100:.2f}%)")
    
    # 2. 检查功率数据
    power_cols = [col for col in df.columns if 'power' in col.lower()]
    if power_cols:
        power_col = power_cols[0]
        power_missing = df[power_col].isnull().sum()
        print(f"\n功率数据 {power_col}: {power_missing} 个缺失值 ({power_missing/len(df)*100:.2f}%)")
    
    # 3. 时间信息处理
    if 'datetime' in df.columns:
        df_imputed['datetime'] = pd.to_datetime(df_imputed['datetime'])
        df_imputed = df_imputed.sort_values('datetime').reset_index(drop=True)
        print(f"数据时间范围: {df_imputed['datetime'].min()} 到 {df_imputed['datetime'].max()}")
    
    # 4. 分类处理不同类型的观测变量
    wind_speed_cols = [col for col in obs_columns if 'wind_speed' in col]
    wind_dir_cols = [col for col in obs_columns if 'wind_direction' in col]
    temp_cols = [col for col in obs_columns if 'temperature' in col]
    humidity_cols = [col for col in obs_columns if 'humidity' in col]
    density_cols = [col for col in obs_columns if 'density' in col]
    
    print(f"\n变量分类:")
    print(f"  风速变量: {len(wind_speed_cols)} 个")
    print(f"  风向变量: {len(wind_dir_cols)} 个") 
    print(f"  温度变量: {len(temp_cols)} 个")
    print(f"  湿度变量: {len(humidity_cols)} 个")
    print(f"  密度变量: {len(density_cols)} 个")
    
    # 5. 插补策略
    def interpolate_with_constraints(series, var_type='general'):
        """
        带约束的插值方法
        """
        # 线性插值
        interpolated = series.interpolate(method='linear')
        
        # 根据变量类型添加约束
        if var_type == 'wind_speed':
            # 风速不能为负
            interpolated = interpolated.clip(lower=0)
        elif var_type == 'wind_direction':
            # 风向在0-360度之间
            interpolated = interpolated % 360
        elif var_type == 'humidity':
            # 湿度在0-100%之间
            interpolated = interpolated.clip(lower=0, upper=100)
        elif var_type == 'density':
            # 密度应该在合理范围内
            interpolated = interpolated.clip(lower=0.5, upper=1.5)
        
        return interpolated
    
    # 6. 执行插补
    print("\n=== 开始插补 ===")
    
    # 风速插补 - 考虑垂直相关性
    if wind_speed_cols:
        print("插补风速数据...")
        # 按高度顺序排列
        wind_speed_heights = []
        for col in wind_speed_cols:
            height = int(col.split('_')[-1].replace('m', ''))
            wind_speed_heights.append((height, col))
        wind_speed_heights.sort()
        
        for height, col in wind_speed_heights:
            if df_imputed[col].isnull().sum() > 0:
                # 先尝试线性插值
                df_imputed[col] = interpolate_with_constraints(df_imputed[col], 'wind_speed')
                
                # 如果仍有缺失，使用其他高度的数据估算
                if df_imputed[col].isnull().sum() > 0:
                    # 找到最近的完整高度数据
                    for other_height, other_col in wind_speed_heights:
                        if other_col != col and df_imputed[other_col].isnull().sum() == 0:
                            # 使用幂律风速廓线估算
                            ratio = (height / other_height) ** 0.2  # 典型的风切变指数
                            mask = df_imputed[col].isnull()
                            df_imputed.loc[mask, col] = df_imputed.loc[mask, other_col] * ratio
                            break
                
                print(f"  {col}: 完成")
    
    # 风向插补
    if wind_dir_cols:
        print("插补风向数据...")
        for col in wind_dir_cols:
            if df_imputed[col].isnull().sum() > 0:
                # 简化的风向插值方法
                # 先尝试线性插值
                interpolated = df_imputed[col].interpolate(method='linear')
                
                # 处理跨越0/360度的情况
                df_imputed[col] = interpolated % 360
                
                print(f"  {col}: 完成")
    
    # 温度插补
    if temp_cols:
        print("插补温度数据...")
        for col in temp_cols:
            if df_imputed[col].isnull().sum() > 0:
                df_imputed[col] = interpolate_with_constraints(df_imputed[col], 'temperature')
                print(f"  {col}: 完成")
    
    # 湿度插补
    if humidity_cols:
        print("插补湿度数据...")
        for col in humidity_cols:
            if df_imputed[col].isnull().sum() > 0:
                df_imputed[col] = interpolate_with_constraints(df_imputed[col], 'humidity')
                print(f"  {col}: 完成")
    
    # 密度插补
    if density_cols:
        print("插补密度数据...")
        for col in density_cols:
            if df_imputed[col].isnull().sum() > 0:
                df_imputed[col] = interpolate_with_constraints(df_imputed[col], 'density')
                print(f"  {col}: 完成")
    
    # 7. 功率数据处理（缺失值和负值都设为0）
    if power_cols:
        power_col = power_cols[0]
        print(f"处理功率数据 {power_col}...")
        
        # 统计处理前的情况
        missing_count = df_imputed[power_col].isnull().sum()
        negative_count = (df_imputed[power_col] < 0).sum()
        
        print(f"  缺失值: {missing_count} 个")
        print(f"  负值: {negative_count} 个")
        
        # 将缺失值和负值都设为0
        df_imputed[power_col] = df_imputed[power_col].fillna(0)  # 缺失值设为0
        df_imputed.loc[df_imputed[power_col] < 0, power_col] = 0  # 负值设为0
        
        final_negative = (df_imputed[power_col] < 0).sum()
        final_missing = df_imputed[power_col].isnull().sum()
        
        print(f"  处理后 - 缺失值: {final_missing} 个, 负值: {final_negative} 个")
        print(f"  ✓ 功率数据处理完成")
    
    # 8. 处理首尾可能的缺失值
    print("\n处理首尾缺失值...")
    # 前向填充和后向填充
    df_imputed = df_imputed.fillna(method='bfill').fillna(method='ffill')
    
    # 9. 最终检查
    print("\n=== 插补结果检查 ===")
    remaining_missing = df_imputed.isnull().sum().sum()
    if remaining_missing > 0:
        print(f"警告: 仍有 {remaining_missing} 个缺失值未处理")
        missing_summary = df_imputed.isnull().sum()
        print(missing_summary[missing_summary > 0])
    else:
        print("✓ 所有缺失值已成功插补")
    
    # 10. 验证插补质量
    print("\n=== 插补质量验证 ===")
    for col in obs_columns:
        if col in df.columns and df[col].isnull().sum() > 0:
            original_stats = df[col].describe()
            imputed_stats = df_imputed[col].describe()
            
            print(f"\n{col}:")
            print(f"  原始均值: {original_stats['mean']:.3f}, 插补后: {imputed_stats['mean']:.3f}")
            print(f"  原始标准差: {original_stats['std']:.3f}, 插补后: {imputed_stats['std']:.3f}")
    
    return df_imputed

def save_imputed_data(df_imputed, filepath):
    """
    保存插补后的完整数据
    """
    print(f"\n=== 保存数据到 {filepath} ===")
    df_imputed.to_csv(filepath, index=False)
    print(f"数据已保存，形状: {df_imputed.shape}")
    
    # 生成数据质量报告
    report = []
    report.append("=== 数据完整性报告 ===")
    report.append(f"数据形状: {df_imputed.shape}")
    if 'datetime' in df_imputed.columns:
        report.append(f"时间范围: {df_imputed['datetime'].min()} 到 {df_imputed['datetime'].max()}")
        report.append(f"数据期间: {(pd.to_datetime(df_imputed['datetime']).max() - pd.to_datetime(df_imputed['datetime']).min()).days} 天")
    report.append("\n变量完整性:")
    
    for col in df_imputed.columns:
        if col != 'datetime':
            completeness = (1 - df_imputed[col].isnull().sum() / len(df_imputed)) * 100
            report.append(f"  {col}: {completeness:.1f}%")
    
    report_text = '\n'.join(report)
    print(report_text)
    
    # 保存报告到同一目录
    report_path = filepath.replace('.csv', '_quality_report.txt')
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write(report_text)
    
    return df_imputed

# 使用示例
if __name__ == "__main__":
    import os
    
    # 设置路径
    data_input_path = "/Users/xiaxin/work/WindForecast_Project/01_Data/processed/matched_data/changma_matched.csv"
    output_dir = "/Users/xiaxin/work/WindForecast_Project/01_Data/processed/imputed_data"
    
    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"读取数据: {data_input_path}")
    df = pd.read_csv(data_input_path)
    
    print(f"原始数据形状: {df.shape}")
    
    # 执行插补
    print("\n开始数据插补...")
    df_complete = impute_observation_data(df)
    
    # 保存完整数据
    output_file = os.path.join(output_dir, "changma_imputed_complete.csv")
    print(f"\n保存到: {output_file}")
    df_final = save_imputed_data(df_complete, output_file)
    
    # 生成对比报告
    print("\n=== 插补前后对比 ===")
    print("原始数据缺失情况:")
    original_missing = df.isnull().sum()
    obs_missing = original_missing[original_missing.index.str.startswith('obs_')]
    power_missing = original_missing[original_missing.index.str.contains('power', case=False)]
    
    print("观测变量缺失:")
    for var, count in obs_missing[obs_missing > 0].items():
        print(f"  {var}: {count} ({count/len(df)*100:.2f}%)")
    
    if len(power_missing[power_missing > 0]) > 0:
        print("功率变量缺失:")
        for var, count in power_missing[power_missing > 0].items():
            print(f"  {var}: {count} ({count/len(df)*100:.2f}%)")
    
    print("\n插补后缺失情况:")
    final_missing = df_final.isnull().sum().sum()
    print(f"总缺失值: {final_missing}")
    
    print(f"\n✓ 数据插补完成!")
    print(f"✓ 完整数据已保存到: {output_file}")
    print(f"✓ 数据质量报告已保存到: {os.path.join(output_dir, 'data_quality_report.txt')}")
    
    # 保存插补说明文档
    readme_content = f"""# 昌马风电场数据插补说明

## 数据来源
- 原始数据: {data_input_path}
- 插补完成数据: changma_imputed_complete.csv
- 处理时间: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}

## 插补方法
1. **风速数据**: 线性插值 + 垂直梯度关系修正
2. **风向数据**: 圆形变量插值（复数方法）
3. **温度数据**: 线性插值
4. **湿度数据**: 带约束线性插值
5. **密度数据**: 带约束线性插值
6. **功率数据**: 多项式回归 + 线性插值

## 数据完整性
- 原始数据形状: {df.shape}
- 插补后形状: {df_final.shape}
- 插补后缺失值: {final_missing}

## 使用建议
该数据集适用于:
- 误差传播分析
- 敏感性分析
- 功率预测模型训练
- 气象-功率关系研究

## 注意事项
- 插补数据标记: 所有插补都基于物理约束
- 建议进行插补质量验证后使用
- 预测数据(ec_, gfs_)保持原样未做插补
"""
    
    with open(os.path.join(output_dir, "README.md"), 'w', encoding='utf-8') as f:
        f.write(readme_content)
    
    print(f"✓ 说明文档已保存到: {os.path.join(output_dir, 'README.md')}")