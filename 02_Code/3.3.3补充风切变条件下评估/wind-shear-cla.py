import pandas as pd
import numpy as np
import os

def calculate_windshear_index(file_path):
    """
    计算风切变指数
    
    参数:
    file_path: Excel文件路径
    
    返回:
    保存包含时间和风切变指数的CSV文件
    """
    
    # 读取Excel文件
    try:
        df = pd.read_excel(file_path)
        print(f"成功读取文件: {file_path}")
        print(f"数据形状: {df.shape}")
        print(f"列名: {df.columns.tolist()}")
    except Exception as e:
        print(f"读取文件时出错: {e}")
        return
    
    # 确保列名正确（根据截图调整）
    expected_columns = ['datetime', 'obs_wind_speed_10m', 'obs_wind_speed_30m', 
                       'obs_wind_speed_50m', 'obs_wind_speed_70m']
    
    if not all(col in df.columns for col in expected_columns):
        print("列名可能不匹配，当前列名:")
        print(df.columns.tolist())
        # 如果列名不匹配，可以尝试重命名
        if len(df.columns) >= 5:
            df.columns = expected_columns
            print("已重命名列名")
    
    # 提取需要的列：时间、10m风速、70m风速
    datetime_col = df.iloc[:, 0]  # 第一列为时间
    wind_10m = df['obs_wind_speed_10m']  # 10米高度风速
    wind_70m = df['obs_wind_speed_70m']  # 70米高度风速
    
    # 定义高度
    h1 = 10  # 10米
    h2 = 70  # 70米
    
    # 计算风切变指数
    # α = ln(v2/v1) / ln(h2/h1)
    # 为避免除零和对数运算错误，需要处理异常情况
    windshear = []
    
    for i in range(len(df)):
        v1 = wind_10m.iloc[i]  # 10m风速
        v2 = wind_70m.iloc[i]  # 70m风速
        
        # 检查数据有效性
        if pd.isna(v1) or pd.isna(v2) or v1 <= 0 or v2 <= 0:
            windshear.append(np.nan)
        else:
            try:
                # 计算风切变指数
                alpha = np.log(v2/v1) / np.log(h2/h1)
                windshear.append(alpha)
            except:
                windshear.append(np.nan)
    
    # 创建结果DataFrame
    result_df = pd.DataFrame({
        'datetime': datetime_col,
        'windshear': windshear
    })
    
    # 生成输出文件路径
    input_dir = os.path.dirname(file_path)
    output_path = os.path.join(input_dir, '测试集梯度风观测_windshear.csv')
    
    # 保存为CSV文件
    try:
        result_df.to_csv(output_path, index=False, encoding='utf-8-sig')
        print(f"风切变指数计算完成！")
        print(f"结果已保存至: {output_path}")
        
        # 显示统计信息
        print(f"\n统计信息:")
        print(f"总数据点: {len(result_df)}")
        print(f"有效风切变指数: {result_df['windshear'].notna().sum()}")
        print(f"平均风切变指数: {result_df['windshear'].mean():.4f}")
        print(f"风切变指数范围: {result_df['windshear'].min():.4f} ~ {result_df['windshear'].max():.4f}")
        
        # 显示前几行结果
        print(f"\n前5行结果:")
        print(result_df.head())
        
    except Exception as e:
        print(f"保存文件时出错: {e}")
        return
    
    return result_df

# 主程序
if __name__ == "__main__":
    # 文件路径
    file_path = "/Users/xiaxin/work/WindForecast_Project/01_Data/processed/imputed_data/测试集梯度风观测.xlsx"
    
    # 检查文件是否存在
    if os.path.exists(file_path):
        # 计算风切变指数
        result = calculate_windshear_index(file_path)
    else:
        print(f"文件不存在: {file_path}")
        print("请检查文件路径是否正确")