import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
from datetime import datetime
import matplotlib.dates as mdates

# 数据和结果路径
data_path = '/Users/xiaxin/work/WindForecast_Project/01_Data/processed/matched_data/changma_matched.csv'
results_dir = '/Users/xiaxin/work/WindForecast_Project/03_Results/daily_smoothed'

# 确保结果目录存在
os.makedirs(results_dir, exist_ok=True)
os.makedirs(os.path.join(results_dir, 'temperature'), exist_ok=True)
os.makedirs(os.path.join(results_dir, 'wind_speed'), exist_ok=True)
os.makedirs(os.path.join(results_dir, 'multi_height'), exist_ok=True)
os.makedirs(os.path.join(results_dir, 'monthly'), exist_ok=True)

# 设置绘图风格 - 使用英文字体避免中文显示问题
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['Arial', 'Helvetica', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_context("paper", font_scale=1.2)

def load_data(file_path):
    """加载数据并确保时间索引正确设置"""
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
                print("Warning: No time column found, using index as time")
                df['datetime'] = pd.date_range(start='2023-01-01', periods=len(df), freq='H')
                
        # 确保datetime是索引
        df = df.set_index('datetime')
        
        return df
    except Exception as e:
        print(f"Error loading data: {e}")
        return None

def calculate_daily_window(df):
    """计算基于数据频率的每日数据点数量"""
    # 计算每天的数据点数量
    if len(df) <= 1:
        return 24  # 默认每小时一个点
    
    time_diff = df.index[1] - df.index[0]
    if time_diff.total_seconds() == 0:
        return 24
    
    points_per_day = pd.Timedelta(days=1) / time_diff
    
    # 如果数据粒度小于1小时，为了安全起见，设置一个最小值
    if points_per_day < 1:
        points_per_day = 24  # 假设每小时至少有一个数据点
        
    return max(1, int(points_per_day))

def plot_daily_smoothed(df, variable, window_days=1, color='#1f77b4', figsize=(12, 6), show_raw=False, min_periods=None):
    """绘制基于天的平滑时间序列图
    
    参数:
        df: 包含数据的DataFrame
        variable: 要绘制的变量名
        window_days: 平滑窗口天数
        color: 线条颜色
        figsize: 图表大小
        show_raw: 是否显示原始数据
        min_periods: 计算平滑值所需的最小有效值数，默认为None（使用window_size的1/4）
    """
    # 检查变量是否存在
    if variable not in df.columns:
        print(f"Error: Column '{variable}' not found in data")
        return
    
    # 变量名称格式化，用于标题和图例
    var_type = 'Temperature' if 'temp' in variable.lower() else 'Wind Speed'
    var_unit = '°C' if 'temp' in variable.lower() else 'm/s'
    
    # 获取数据来源和高度信息
    source = 'Unknown'
    height = ''
    
    if variable.startswith('obs_'):
        source = 'Observation'
    elif variable.startswith('ec_'):
        source = 'ECMWF'
    elif variable.startswith('gfs_'):
        source = 'GFS'
        
    if '10m' in variable:
        height = '10m'
    elif '30m' in variable:
        height = '30m'
    elif '50m' in variable:
        height = '50m'
    elif '70m' in variable:
        height = '70m'
    elif '100m' in variable:
        height = '100m'
    
    # 计算窗口大小（基于数据点数量）
    window_size = window_days * calculate_daily_window(df)
    
    # 如果min_periods未指定，则使用窗口大小的1/4（对于月平滑更宽松以处理缺失数据）
    if min_periods is None:
        if window_days >= 30:  # 月平滑
            min_periods = max(int(window_size / 4), 1)  # 至少1，但更宽松，允许75%缺失
        else:  # 日平滑
            min_periods = max(int(window_size * 3 / 4), 1)  # 更严格，最多允许25%缺失
    
    print(f"Variable: {variable}, Window size: {window_size} points ({window_days} days), Min periods: {min_periods}")
    
    # 检查数据质量
    valid_data_pct = 100 * df[variable].count() / len(df)
    print(f"Data completeness: {valid_data_pct:.1f}%")
    
    # 计算移动平均和标准差 - 使用min_periods参数处理缺失值
    rolling_mean = df[variable].rolling(window=window_size, center=True, min_periods=min_periods).mean()
    rolling_std = df[variable].rolling(window=window_size, center=True, min_periods=min_periods).std()
    
    # 创建图表
    fig, ax = plt.subplots(figsize=figsize)
    
    # 绘制原始数据（如果需要）
    if show_raw:
        ax.plot(df.index, df[variable], alpha=0.3, color=color, linewidth=0.8, label='Raw Data')
    
    # 绘制平滑线
    smoothing_label = f'{window_days}-Day' if window_days < 30 else f'{window_days/30:.1f}-Month'
    ax.plot(df.index, rolling_mean, color=color, linewidth=2.5, label=f'{smoothing_label} Moving Avg')
    
    # 绘制标准差区间
    ax.fill_between(df.index, 
                   rolling_mean - rolling_std, 
                   rolling_mean + rolling_std, 
                   color=color, alpha=0.2, label='±1 Std Dev')
    
    # 设置标题和标签
    time_scale = "Daily" if window_days < 30 else "Monthly"
    title = f"Changma Station {source} {height} {var_type} ({time_scale} Smoothed)"
    ax.set_title(title, fontsize=16, pad=20)
    ax.set_xlabel('Date', fontsize=12)
    ax.set_ylabel(f'{var_type} ({var_unit})', fontsize=12)
    
    # 设置x轴日期格式
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
    ax.xaxis.set_major_locator(mdates.AutoDateLocator())
    plt.gcf().autofmt_xdate()
    
    # 添加网格线和图例
    ax.grid(True, linestyle='--', alpha=0.7)
    ax.legend(loc='best', frameon=True, facecolor='white', framealpha=0.9)
    
    # 紧凑布局
    plt.tight_layout()
    
    # 确定保存路径
    subdir = 'temperature' if 'temp' in variable.lower() else 'wind_speed'
    time_scale_str = "daily" if window_days < 30 else "monthly"
    window_str = f"{window_days}day" if window_days < 30 else f"{int(window_days/30)}month"
    
    if time_scale_str == "monthly":
        subdir = "monthly"  # 月平滑图存放在monthly子目录
        
    save_path = os.path.join(results_dir, subdir, f'{variable}_{time_scale_str}_{window_str}_smoothed.png')
    
    # 保存图片
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Saved: {save_path}")
    
    plt.close()

def plot_multi_height_comparison(df, source, window_days=1, figsize=(14, 8), show_raw=False, min_periods=None):
    """为指定数据源绘制多高度风速比较图
    
    参数:
        df: 包含数据的DataFrame
        source: 数据来源 (obs, ec, gfs)
        window_days: 平滑窗口天数
        figsize: 图表大小
        show_raw: 是否显示原始数据
        min_periods: 计算平滑值所需的最小有效值数
    """
    # 获取指定数据源的所有风速列
    wind_cols = [col for col in df.columns if col.startswith(f'{source}_wind_speed_')]
    
    if not wind_cols:
        print(f"No wind speed columns found for source: {source}")
        return
    
    # 提取高度信息
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
    
    # 按高度排序
    height_order = {'10m': 0, '30m': 1, '50m': 2, '70m': 3, '100m': 4, 'unknown': 9}
    sorted_indices = sorted(range(len(heights)), key=lambda i: height_order.get(heights[i], 9))
    sorted_cols = [wind_cols[i] for i in sorted_indices]
    sorted_heights = [heights[i] for i in sorted_indices]
    
    if len(sorted_cols) < 2:
        print(f"Need at least 2 heights for comparison, only found: {sorted_heights}")
        return
    
    # 计算窗口大小
    window_size = window_days * calculate_daily_window(df)
    time_scale = "Daily" if window_days < 30 else "Monthly"
    
    # 如果min_periods未指定，则使用窗口大小的1/4（对于月平滑更宽松以处理缺失数据）
    if min_periods is None:
        if window_days >= 30:  # 月平滑
            min_periods = max(int(window_size / 4), 1)  # 至少1，但更宽松，允许75%缺失
        else:  # 日平滑
            min_periods = max(int(window_size * 3 / 4), 1)  # 更严格，最多允许25%缺失
    
    print(f"{time_scale} multi-height plot for {source}, window size: {window_size}, min_periods: {min_periods}")
    
    # 创建图表
    fig, ax = plt.subplots(figsize=figsize)
    
    # 设置颜色映射 - 使用与高度相关的色谱
    cmap = plt.cm.viridis
    colors = [cmap(i/len(sorted_heights)) for i in range(len(sorted_heights))]
    
    # 检查数据质量
    for col in sorted_cols:
        valid_data_pct = 100 * df[col].count() / len(df)
        print(f"Column {col} completeness: {valid_data_pct:.1f}%")
    
    # 绘制每个高度的数据
    for i, (column, height) in enumerate(zip(sorted_cols, sorted_heights)):
        # 如果需要，绘制原始数据
        if show_raw:
            ax.plot(df.index, df[column], color=colors[i], alpha=0.15, linewidth=0.5)
        
        # 计算移动平均
        rolling_mean = df[column].rolling(
            window=window_size, 
            center=True, 
            min_periods=min_periods
        ).mean()
        
        # 绘制平滑线
        ax.plot(df.index, rolling_mean, color=colors[i], linewidth=2.5, label=f"{height}")
    
    # 设置标题和标签
    source_name = {'obs': 'Observation', 'ec': 'ECMWF', 'gfs': 'GFS'}.get(source, source.upper())
    smoothing_label = f'{window_days}-Day' if window_days < 30 else f'{window_days/30:.1f}-Month'
    
    ax.set_title(f"Changma Station {source_name} Wind Speed at Different Heights ({smoothing_label} Smoothed)", 
                fontsize=16, pad=20)
    ax.set_xlabel('Date', fontsize=12)
    ax.set_ylabel('Wind Speed (m/s)', fontsize=12)
    
    # 设置x轴日期格式
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
    ax.xaxis.set_major_locator(mdates.AutoDateLocator())
    plt.gcf().autofmt_xdate()
    
    # 添加网格线和图例
    ax.grid(True, linestyle='--', alpha=0.7)
    ax.legend(loc='best', frameon=True, facecolor='white', framealpha=0.9)
    
    # 紧凑布局
    plt.tight_layout()
    
    # 保存图片
    time_scale_str = "daily" if window_days < 30 else "monthly"
    window_str = f"{window_days}day" if window_days < 30 else f"{int(window_days/30)}month"
    
    # 根据时间尺度决定保存位置
    if time_scale_str == "monthly":
        save_dir = os.path.join(results_dir, "monthly")
    else:
        save_dir = os.path.join(results_dir, "multi_height")
    
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, f'{source}_multi_height_{time_scale_str}_{window_str}.png')
    
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Saved: {save_path}")
    
    plt.close()

def main():
    print("Loading data from:", data_path)
    df = load_data(data_path)
    
    if df is None:
        print("Error: Failed to load data.")
        return
    
    print(f"Data loaded successfully. Shape: {df.shape}")
    print(f"Date range: {df.index.min()} to {df.index.max()}")
    
    # 检查数据缺失情况
    missing_data = df.isnull().sum()
    print("\nMissing data count:")
    for col in missing_data[missing_data > 0].index:
        print(f"  {col}: {missing_data[col]} missing values ({100 * missing_data[col] / len(df):.1f}%)")
    
    # 识别温度和风速列
    temp_cols = [col for col in df.columns if 'temp' in col.lower()]
    wind_cols = [col for col in df.columns if ('wind' in col.lower() and 'speed' in col.lower())]
    
    print(f"\nFound {len(temp_cols)} temperature columns and {len(wind_cols)} wind speed columns")
    
    # 识别数据来源
    sources = set()
    for col in df.columns:
        if col.startswith('obs_'):
            sources.add('obs')
        elif col.startswith('ec_'):
            sources.add('ec')
        elif col.startswith('gfs_'):
            sources.add('gfs')
    
    print(f"Data sources: {sources}")
    
    # 窗口天数选项
    daily_window = 1  # 1天平滑
    monthly_window = 30  # 30天（月）平滑
    
    # 对于月平滑，min_periods设为更宽松的值，以处理缺失值
    daily_min_periods = None  # 默认值，大约为window_size的3/4
    monthly_min_periods = 2  # 非常宽松，只要有2个有效值就计算
    
    # 绘制每个温度变量的平滑图
    print("\nGenerating temperature plots...")
    for col in temp_cols:
        # 日平滑
        plot_daily_smoothed(df, col, window_days=daily_window, color='#e41a1c', 
                          show_raw=False, min_periods=daily_min_periods)
        # 月平滑
        plot_daily_smoothed(df, col, window_days=monthly_window, color='#e41a1c', 
                          show_raw=False, min_periods=monthly_min_periods)
    
    # 绘制每个风速变量的平滑图
    print("\nGenerating wind speed plots...")
    wind_colors = {
        'obs': '#377eb8',
        'ec': '#4daf4a',
        'gfs': '#984ea3'
    }
    
    for col in wind_cols:
        source = 'unknown'
        if col.startswith('obs_'):
            source = 'obs'
        elif col.startswith('ec_'):
            source = 'ec'
        elif col.startswith('gfs_'):
            source = 'gfs'
            
        color = wind_colors.get(source, '#ff7f00')
        # 日平滑
        plot_daily_smoothed(df, col, window_days=daily_window, color=color, 
                          show_raw=False, min_periods=daily_min_periods)
        # 月平滑
        plot_daily_smoothed(df, col, window_days=monthly_window, color=color, 
                          show_raw=False, min_periods=monthly_min_periods)
    
    # 绘制多高度对比图
    print("\nGenerating multi-height comparison plots...")
    for source in sources:
        # 日平滑多高度对比
        plot_multi_height_comparison(df, source, window_days=daily_window, 
                                   show_raw=False, min_periods=daily_min_periods)
        # 月平滑多高度对比
        plot_multi_height_comparison(df, source, window_days=monthly_window, 
                                   show_raw=False, min_periods=monthly_min_periods)
    
    print("\nAll plots completed successfully!")

if __name__ == "__main__":
    main()