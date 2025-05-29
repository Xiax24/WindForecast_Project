import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')

def clean_density_data_only(df):
    """
    仅处理密度异常值，设为NaN但不插补
    其他数据保持原样
    """
    print("=== 密度数据清理（仅设NaN，不插补）===")
    
    # 复制原始数据
    df_cleaned = df.copy()
    
    # 1. 识别密度变量
    density_cols = [col for col in df.columns if 'density' in col and col.startswith('obs_')]
    
    if not density_cols:
        print("未找到观测密度变量")
        return df_cleaned
    
    print(f"发现密度变量: {len(density_cols)} 个")
    
    # 2. 处理每个密度变量
    for col in density_cols:
        print(f"\n处理 {col}:")
        
        # 统计原始情况
        total_count = len(df_cleaned[col])
        original_missing = df_cleaned[col].isnull().sum()
        original_valid = total_count - original_missing
        
        # 统计异常值
        exactly_05 = (df_cleaned[col] == 0.5).sum()
        lte_05 = (df_cleaned[col] <= 0.5).sum()
        gte_20 = (df_cleaned[col] >= 2.0).sum()
        
        # 统计正常值范围
        normal_values = df_cleaned[col][(df_cleaned[col] > 0.5) & (df_cleaned[col] < 2.0)]
        normal_count = len(normal_values.dropna())
        
        print(f"  总数据点: {total_count}")
        print(f"  原始缺失: {original_missing}")
        print(f"  原始有效: {original_valid}")
        print(f"  正好=0.5: {exactly_05}")
        print(f"  ≤0.5: {lte_05}")
        print(f"  ≥2.0: {gte_20}")
        print(f"  正常范围(0.5-2.0): {normal_count}")
        
        if normal_count > 0:
            print(f"  正常值范围: [{normal_values.min():.3f}, {normal_values.max():.3f}]")
            print(f"  正常值均值: {normal_values.mean():.3f}")
        
        # 将异常值设为NaN
        abnormal_count = lte_05 + gte_20
        if abnormal_count > 0:
            # 设置异常值为NaN
            df_cleaned.loc[df_cleaned[col] <= 0.5, col] = np.nan
            df_cleaned.loc[df_cleaned[col] >= 2.0, col] = np.nan
            
            # 统计处理后情况
            final_missing = df_cleaned[col].isnull().sum()
            final_valid = total_count - final_missing
            newly_set_nan = final_missing - original_missing
            
            print(f"  ✓ 已将 {newly_set_nan} 个异常值设为NaN")
            print(f"  处理后缺失: {final_missing} ({final_missing/total_count*100:.1f}%)")
            print(f"  处理后有效: {final_valid} ({final_valid/total_count*100:.1f}%)")
            
            # 验证没有异常值残留
            remaining_abnormal = ((df_cleaned[col] <= 0.5) | (df_cleaned[col] >= 2.0)).sum()
            if remaining_abnormal == 0:
                print(f"  ✓ 无异常值残留")
            else:
                print(f"  ⚠️ 仍有 {remaining_abnormal} 个异常值")
        else:
            print(f"  ✓ 无异常值需要处理")
    
    return df_cleaned

def save_cleaned_density_data(df_cleaned, filepath):
    """
    保存清理后的数据
    """
    print(f"\n=== 保存数据到 {filepath} ===")
    df_cleaned.to_csv(filepath, index=False)
    print(f"数据已保存，形状: {df_cleaned.shape}")
    
    # 生成清理报告
    report = []
    report.append("=== 密度数据清理报告 ===")
    report.append(f"处理时间: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}")
    report.append(f"数据形状: {df_cleaned.shape}")
    
    if 'datetime' in df_cleaned.columns:
        report.append(f"时间范围: {df_cleaned['datetime'].min()} 到 {df_cleaned['datetime'].max()}")
    
    # 密度数据详细报告
    density_cols = [col for col in df_cleaned.columns if 'density' in col and col.startswith('obs_')]
    if density_cols:
        report.append("\n密度数据清理结果:")
        for col in density_cols:
            total_count = len(df_cleaned[col])
            missing_count = df_cleaned[col].isnull().sum()
            valid_count = total_count - missing_count
            
            if valid_count > 0:
                valid_data = df_cleaned[col].dropna()
                min_val = valid_data.min()
                max_val = valid_data.max()
                mean_val = valid_data.mean()
                
                # 检查是否还有异常值
                abnormal_count = ((valid_data <= 0.5) | (valid_data >= 2.0)).sum()
                
                report.append(f"  {col}:")
                report.append(f"    有效数据: {valid_count}/{total_count} ({valid_count/total_count*100:.1f}%)")
                report.append(f"    缺失数据: {missing_count} ({missing_count/total_count*100:.1f}%)")
                report.append(f"    数值范围: [{min_val:.3f}, {max_val:.3f}]")
                report.append(f"    均值: {mean_val:.3f}")
                report.append(f"    异常值: {abnormal_count}")
            else:
                report.append(f"  {col}: 无有效数据")
    
    report.append("\n处理说明:")
    report.append("- 异常值定义: ≤0.5 kg/m³ 或 ≥2.0 kg/m³")
    report.append("- 处理方式: 设为NaN，不进行插补")
    report.append("- 其他变量: 保持原样不变")
    report.append("- 建议: 密度数据可能需要重新获取或使用其他数据源")
    
    report_text = '\n'.join(report)
    print(report_text)
    
    # 保存报告
    report_path = filepath.replace('.csv', '_density_cleaning_report.txt')
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write(report_text)
    
    return df_cleaned

def analyze_density_data_quality(df):
    """
    分析密度数据质量
    """
    print("\n=== 密度数据质量分析 ===")
    
    density_cols = [col for col in df.columns if 'density' in col and col.startswith('obs_')]
    
    if not density_cols:
        print("未找到观测密度变量")
        return
    
    for col in density_cols:
        print(f"\n{col} 详细分析:")
        
        # 基本统计
        total = len(df[col])
        missing = df[col].isnull().sum()
        valid = total - missing
        
        print(f"  数据点总数: {total}")
        print(f"  缺失数据: {missing} ({missing/total*100:.1f}%)")
        print(f"  有效数据: {valid} ({valid/total*100:.1f}%)")
        
        if valid > 0:
            data = df[col].dropna()
            
            # 数值分布分析
            print(f"  数值统计:")
            print(f"    最小值: {data.min():.6f}")
            print(f"    最大值: {data.max():.6f}")
            print(f"    均值: {data.mean():.6f}")
            print(f"    中位数: {data.median():.6f}")
            print(f"    标准差: {data.std():.6f}")
            
            # 异常值统计
            exactly_05 = (data == 0.5).sum()
            lte_05 = (data <= 0.5).sum()
            gte_20 = (data >= 2.0).sum()
            between_05_20 = ((data > 0.5) & (data < 2.0)).sum()
            
            print(f"  数值分布:")
            print(f"    正好=0.5: {exactly_05} ({exactly_05/valid*100:.1f}%)")
            print(f"    ≤0.5: {lte_05} ({lte_05/valid*100:.1f}%)")
            print(f"    ≥2.0: {gte_20} ({gte_20/valid*100:.1f}%)")
            print(f"    0.5-2.0之间: {between_05_20} ({between_05_20/valid*100:.1f}%)")
            
            # 合理性评估
            reasonable_count = between_05_20
            reasonable_percentage = reasonable_count / valid * 100
            
            print(f"  数据质量评估:")
            if reasonable_percentage > 50:
                print(f"    ✓ 数据质量尚可 ({reasonable_percentage:.1f}% 在合理范围)")
            elif reasonable_percentage > 20:
                print(f"    ⚠️ 数据质量一般 ({reasonable_percentage:.1f}% 在合理范围)")
            else:
                print(f"    ❌ 数据质量很差 ({reasonable_percentage:.1f}% 在合理范围)")
                print(f"    建议: 考虑不使用密度数据或寻找替代数据源")

# 使用示例
if __name__ == "__main__":
    import os
    
    # 设置路径
    data_input_path = "/Users/xiaxin/work/WindForecast_Project/01_Data/processed/imputed_data/changma_imputed_complete_v2.csv"
    output_dir = "/Users/xiaxin/work/WindForecast_Project/01_Data/processed/imputed_data"
    
    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"读取数据: {data_input_path}")
    df = pd.read_csv(data_input_path)
    print(f"原始数据形状: {df.shape}")
    
    # 分析密度数据质量
    analyze_density_data_quality(df)
    
    # 仅清理密度异常值，不插补
    print(f"\n开始密度数据清理（仅设NaN，不插补）...")
    df_cleaned = clean_density_data_only(df)
    
    # 保存清理后的数据
    output_file = os.path.join(output_dir, "changma_density_cleaned.csv")
    print(f"\n保存到: {output_file}")
    df_final = save_cleaned_density_data(df_cleaned, output_file)
    
    print(f"\n✓ 密度数据清理完成!")
    print(f"✓ 异常密度值已设为NaN，未进行插补")
    print(f"✓ 数据已保存到: {output_file}")
    print(f"✓ 建议: 后续分析中可选择排除密度变量或使用其他方法处理")
    
    # 最终验证
    print(f"\n=== 最终验证 ===")
    density_cols = [col for col in df_final.columns if 'density' in col and col.startswith('obs_')]
    for col in density_cols:
        abnormal_count = ((df_final[col] <= 0.5) | (df_final[col] >= 2.0)).sum()
        missing_count = df_final[col].isnull().sum()
        valid_count = len(df_final[col]) - missing_count
        
        print(f"{col}: 异常值={abnormal_count}, 缺失值={missing_count}, 有效值={valid_count}")
        
        if valid_count > 0:
            valid_data = df_final[col].dropna()
            print(f"  有效值范围: [{valid_data.min():.3f}, {valid_data.max():.3f}]")
            print(f"  有效值均值: {valid_data.mean():.3f}")
    
    # 生成使用建议
    suggestions = """
### 后续分析建议:

1. **相关性分析**: 可以排除密度变量，仅使用风速、风向、温度、湿度进行分析
2. **功率预测**: 密度对功率影响较小，可以不使用密度变量
3. **替代方案**: 可以考虑使用温度和压力数据计算理论密度
4. **数据源**: 建议联系数据提供方确认密度数据的测量方法和单位
5. **质量控制**: 后续数据收集时需要加强密度数据的质量控制
    """
    
    suggestions_path = os.path.join(output_dir, "density_data_usage_suggestions.txt")
    with open(suggestions_path, 'w', encoding='utf-8') as f:
        f.write(suggestions)
    
    print(f"\n✓ 使用建议已保存到: {suggestions_path}")