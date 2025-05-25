import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
from datetime import datetime
import matplotlib.dates as mdates
from matplotlib.ticker import MaxNLocator

# 完全避免使用中文，使用英文标签
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['Arial', 'Helvetica', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

# 设置更现代的风格
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_context("paper", font_scale=1.2)

# 设置更现代的风格
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_context("paper", font_scale=1.2)

# 数据路径
data_path = '/Users/xiaxin/work/WindForecast_Project/01_Data/processed/matched_data/changma_matched.csv'
results_dir = '/Users/xiaxin/work/WindForecast_Project/03_Results'

# 确保结果目录存在
os.makedirs(results_dir, exist_ok=True)
os.makedirs(os.path.join(results_dir, 'time_series'), exist_ok=True)
os.makedirs(os.path.join(results_dir, 'smoothed'), exist_ok=True)

# 加载数据
def load_data(file_path):
    try:
        df = pd.read_csv(file_path)
        
        # 检查是否有时间列
        if 'datetime' in df.columns:
            df['datetime'] = pd.to_datetime(df['datetime'])
        elif 'time' in df.columns:
            df['datetime'] = pd.to_datetime(df['time'])
        elif 'date' in df.columns:
            df['datetime'] = pd.to_datetime(df['date'])
        else:
            # 尝试查找可能的时间列
            time_cols = [col for col in df.columns if 'time' in col.lower() or 'date' in col.lower()]
            if time_cols:
                df['datetime'] = pd.to_datetime(df[time_cols[0]])
            else:
                print("警告: 未找到时间列，将使用索引作为时间")
                df['datetime'] = pd.date_range(start='2023-01-01', periods=len(df), freq='H')
                
        # 确保datetime是索引
        df = df.set_index('datetime')
        
        return df
    except Exception as e:
        print(f"加载数据时出错: {e}")
        return None

# 绘制时间序列图
def plot_time_series(df, variable, title, y_label, color='#1f77b4', figsize=(12, 6)):
    """绘制单个变量的时间序列图"""
    fig, ax = plt.subplots(figsize=figsize)
    
    # 绘制主时间序列
    ax.plot(df.index, df[variable], color=color, linewidth=1.5, alpha=0.8)
    
    # 设置标题和标签
    # 将中文标题转换为英文，避免字体问题
    english_title = title.replace('昌马站', 'Changma Station')
    english_ylabel = y_label.replace('温度', 'Temperature').replace('风速', 'Wind Speed')
    
    ax.set_title(english_title, fontsize=16, pad=20)
    ax.set_xlabel('Time', fontsize=12)
    ax.set_ylabel(english_ylabel, fontsize=12)
    
    # 设置x轴日期格式
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
    ax.xaxis.set_major_locator(mdates.AutoDateLocator())
    
    # 自动旋转日期标签
    plt.gcf().autofmt_xdate()
    
    # 添加网格线
    ax.grid(True, linestyle='--', alpha=0.7)
    
    # 添加均值线
    mean_val = df[variable].mean()
    ax.axhline(y=mean_val, color='r', linestyle='--', alpha=0.7, label=f'Mean: {mean_val:.2f}')
    
    # 设置图例
    ax.legend(loc='best', frameon=True, facecolor='white', framealpha=0.9)
    
    # 紧凑布局
    plt.tight_layout()
    
    # 保存图片
    save_path = os.path.join(results_dir, 'time_series', f'{variable}_time_series.png')
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Saved: {save_path}")
    
    plt.close()

# 绘制带有平滑线和标准差区间的图
def plot_smoothed_with_std(df, variable, title, y_label, window=24, color='#1f77b4', figsize=(12, 6)):
    """绘制带有平滑线和标准差区间的图"""
    fig, ax = plt.subplots(figsize=figsize)
    
    # 计算滚动平均值和标准差
    rolling_mean = df[variable].rolling(window=window, center=True).mean()
    rolling_std = df[variable].rolling(window=window, center=True).std()
    
    # 设置标题和标签 - 使用英文
    english_title = title.replace('昌马站', 'Changma Station').replace('平滑时间序列 (带标准差)', 'Smoothed Time Series (with Std Dev)')
    english_ylabel = y_label.replace('温度', 'Temperature').replace('风速', 'Wind Speed')
    
    # 绘制原始数据（淡色）
    ax.plot(df.index, df[variable], alpha=0.3, color=color, linewidth=0.8, label='Raw Data')
    
    # 绘制平滑线
    ax.plot(df.index, rolling_mean, color=color, linewidth=2.5, label=f'{window}h Moving Avg')
    
    # 绘制标准差区间
    ax.fill_between(df.index, 
                   rolling_mean - rolling_std, 
                   rolling_mean + rolling_std, 
                   color=color, alpha=0.2, label='±1 Std Dev')
    
    # 设置标题和标签
    ax.set_title(english_title, fontsize=16, pad=20)
    ax.set_xlabel('Time', fontsize=12)
    ax.set_ylabel(english_ylabel, fontsize=12)
    
    # 设置x轴日期格式
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
    ax.xaxis.set_major_locator(mdates.AutoDateLocator())
    
    # 自动旋转日期标签
    plt.gcf().autofmt_xdate()
    
    # 添加网格线
    ax.grid(True, linestyle='--', alpha=0.7)
    
    # 设置图例
    ax.legend(loc='best', frameon=True, facecolor='white', framealpha=0.9)
    
    # 紧凑布局
    plt.tight_layout()
    
    # 保存图片
    save_path = os.path.join(results_dir, 'smoothed', f'{variable}_smoothed.png')
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Saved: {save_path}")
    
    plt.close()

# 主函数
def main():
    # 加载数据
    print("Loading data...")
    df = load_data(data_path)
    
    if df is None:
        print("Failed to load data, exiting.")
        return
    
    print(f"Data loaded successfully, shape: {df.shape}")
    print("Columns:", df.columns.tolist())
    
    # 检查列名，确定命名方式
    columns = df.columns.tolist()
    print("First 10 column names:", columns[:10])
    
    # 打印有用的调试信息
    print("\nData preview:")
    print(df.head())
    
    # 检查数据类型
    print("\nData types:")
    print(df.dtypes)
    
    # 识别温度和风速列
    temp_cols = [col for col in df.columns if 'temp' in col.lower()]
    wind_cols = [col for col in df.columns if ('wind' in col.lower() and 'speed' in col.lower()) or 
                                            ('ws_' in col.lower())]
    
    # 如果没有自动识别到，使用可能的列名
    if not temp_cols:
        temp_cols = ['temperature', 'temp', 'T10m', 'T_10m', 'temp_10m', 'temperature_10m']
        temp_cols = [col for col in temp_cols if col in df.columns]
    
    if not wind_cols:
        # 可能的风速列名模式
        heights = ['10m', '30m', '50m', '70m', '100m']
        prefixes = ['wind_speed_', 'windspeed_', 'ws_', 'v_', 'wind_']
        sources = ['obs_', 'ec_', 'gfs_', '']
        
        possible_wind_cols = []
        for s in sources:
            for p in prefixes:
                for h in heights:
                    possible_wind_cols.append(f"{s}{p}{h}")
                    possible_wind_cols.append(f"{s}{p}_{h}")
        
        wind_cols = [col for col in possible_wind_cols if col in df.columns]
    
    print(f"\nDetected temperature columns: {temp_cols}")
    print(f"Detected wind speed columns: {wind_cols}")
    
    # 如果仍未找到，手动指定
    if not temp_cols and not wind_cols:
        print("\nCould not auto-detect temperature and wind columns. See column names:")
        for i, col in enumerate(df.columns):
            print(f"{i}: {col}")
        
        # 根据错误信息中的输出，手动指定列名
        temp_cols = ['obs_temperature_10m', 'ec_temperature_10m', 'gfs_temperature_10m']
        temp_cols = [col for col in temp_cols if col in df.columns]
        
        wind_cols = [
            'obs_wind_speed_10m', 'ec_wind_speed_10m', 'gfs_wind_speed_10m',
            'obs_wind_speed_30m', 'ec_wind_speed_30m', 'gfs_wind_speed_30m',
            'obs_wind_speed_50m', 'ec_wind_speed_50m', 'gfs_wind_speed_50m',
            'obs_wind_speed_70m', 'ec_wind_speed_70m', 'gfs_wind_speed_70m'
        ]
        wind_cols = [col for col in wind_cols if col in df.columns]
        
        print(f"\nManually specified temperature columns: {temp_cols}")
        print(f"Manually specified wind columns: {wind_cols}")
    
    # 按高度和来源组织风速列
    heights = []
    for col in wind_cols:
        if '10m' in col:
            heights.append('10m')
        elif '30m' in col:
            heights.append('30m')
        elif '50m' in col:
            heights.append('50m')
        elif '70m' in col:
            heights.append('70m')
        elif '100m' in col:
            heights.append('100m')
        else:
            heights.append('unknown')
    
    unique_heights = sorted(list(set(heights)))
    print(f"\nDetected height levels: {unique_heights}")
    
    # 绘制温度图
    for col in temp_cols:
        print(f"Plotting temperature: {col}")
        plot_time_series(df, col, f"Changma Station {col} Time Series", "Temperature (°C)", color='#e41a1c')
        plot_smoothed_with_std(df, col, f"Changma Station {col} Smoothed Time Series (with Std Dev)", "Temperature (°C)", window=24, color='#e41a1c')
    
    # 绘制风速图 - 按高度分组
    colors = ['#377eb8', '#4daf4a', '#984ea3', '#ff7f00', '#ffff33']  # 不同来源使用不同颜色
    
    # 先按高度分组
    for height in unique_heights:
        height_cols = [col for col, h in zip(wind_cols, heights) if h == height]
        
        # 按来源排序
        sources = []
        for col in height_cols:
            if col.startswith('obs_'):
                sources.append('obs')
            elif col.startswith('ec_'):
                sources.append('ec')
            elif col.startswith('gfs_'):
                sources.append('gfs')
            else:
                sources.append('unknown')
        
        # 按来源排序
        sorted_cols = [col for _, col in sorted(zip(sources, height_cols))]
        
        # 为每个高度单独绘图
        for i, col in enumerate(sorted_cols):
            color = colors[i % len(colors)]
            print(f"Plotting wind speed: {col}")
            plot_time_series(df, col, f"Changma Station {col} Time Series", "Wind Speed (m/s)", color=color)
            plot_smoothed_with_std(df, col, f"Changma Station {col} Smoothed Time Series (with Std Dev)", "Wind Speed (m/s)", window=24, color=color)
    
    print("\nAll plots completed!")

if __name__ == "__main__":
    main()