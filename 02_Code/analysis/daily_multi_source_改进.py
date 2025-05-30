import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
from datetime import datetime
import matplotlib.dates as mdates

# ================================
# 颜色配置区域 - 可自定义修改
# ================================

# 数据源颜色配置
SOURCE_COLORS = {
    'obs': '#984ea3',  # 紫色 - Observation
    'ec': "#f05656",   # 绿色 - ECMWF  
    'gfs': '#377eb8'   # 蓝色 - GFS
}

# 变量类型颜色配置
VARIABLE_COLORS = {
    'temperature': {
        'obs': '#d62728',   # 深红色 - Observation
        'ec': '#ff7f0e',    # 橙色 - ECMWF
        'gfs': "#dfff96"    # 浅红色 - GFS
    },
    'wind_speed': None         # None表示使用数据源颜色
}

# 功率颜色配置
POWER_COLOR = '#2ca02c'  # 绿色 - Power

# 标签配置
LABEL_CONFIG = {
    'sources': {
        'obs': 'Observation',
        'ec': 'ECMWF', 
        'gfs': 'GFS'
    },
    'variables': {
        'temperature': 'Temperature',
        'wind_speed': 'Wind Speed'
    },
    'units': {
        'temperature': '°C',
        'wind_speed': 'm/s'
    }
}

# Y轴范围配置
Y_AXIS_CONFIG = {
    'wind_speed': {
        'ylim': (0, 16),     # Y轴范围 0-16 m/s
        'yticks': range(0, 17, 1)  # Y轴刻度 0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16
    },
    'temperature': {
        'ylim': None,        # None表示自动调整
        'yticks': None       # None表示自动调整
    }
}

# X轴格式配置
X_AXIS_CONFIG = {
    'date_format': '%Y-%m',      # 日期格式：年-月
    'rotation': 45,              # 标签旋转45度
    'interval': 1,               # 每月显示一次
    'ha': 'right'                # 右对齐
}

# ================================
# 路径配置
# ================================

# 数据和结果路径
data_path = '/Users/xiaxin/work/WindForecast_Project/01_Data/processed/matched_data/changma_matched.csv'
results_dir = '/Users/xiaxin/work/WindForecast_Project/03_Results/daily_smoothed_multi_source'

# 确保结果目录存在
os.makedirs(results_dir, exist_ok=True)
os.makedirs(os.path.join(results_dir, 'temperature'), exist_ok=True)
os.makedirs(os.path.join(results_dir, 'wind_speed'), exist_ok=True)
os.makedirs(os.path.join(results_dir, 'monthly'), exist_ok=True)

# 设置绘图风格
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
    if len(df) <= 1:
        return 24  # 默认每小时一个点
    
    time_diff = df.index[1] - df.index[0]
    if time_diff.total_seconds() == 0:
        return 24
    
    points_per_day = pd.Timedelta(days=1) / time_diff
    
    if points_per_day < 1:
        points_per_day = 24
        
    return max(1, int(points_per_day))

def extract_variable_info(variable_name):
    """从变量名中提取变量类型和高度信息"""
    # 移除数据源前缀
    var_base = variable_name
    for prefix in ['obs_', 'ec_', 'gfs_']:
        if variable_name.startswith(prefix):
            var_base = variable_name[len(prefix):]
            break
    
    # 确定变量类型
    if 'temp' in var_base.lower():
        var_type = 'temperature'
        var_unit = LABEL_CONFIG['units']['temperature']
        var_display = LABEL_CONFIG['variables']['temperature']
    elif 'wind_speed' in var_base.lower():
        var_type = 'wind_speed'
        var_unit = LABEL_CONFIG['units']['wind_speed'] 
        var_display = LABEL_CONFIG['variables']['wind_speed']
    else:
        var_type = 'unknown'
        var_unit = ''
        var_display = var_base
    
    # 提取高度信息
    height = 'unknown'
    if '10m' in var_base:
        height = '10m'
    elif '30m' in var_base:
        height = '30m'
    elif '50m' in var_base:
        height = '50m'
    elif '70m' in var_base:
        height = '70m'
    elif '100m' in var_base:
        height = '100m'
    
    return var_type, var_unit, var_display, height, var_base

def find_matching_variables(df):
    """找到具有相同变量类型和高度的不同数据源的变量"""
    variables_info = {}
    
    # 分析所有变量
    for col in df.columns:
        var_type, var_unit, var_display, height, var_base = extract_variable_info(col)
        
        # 确定数据源
        source = 'unknown'
        if col.startswith('obs_'):
            source = 'obs'
        elif col.startswith('ec_'):
            source = 'ec'
        elif col.startswith('gfs_'):
            source = 'gfs'
        
        # 创建分组键
        group_key = f"{var_type}_{height}"
        
        if group_key not in variables_info:
            variables_info[group_key] = {
                'var_type': var_type,
                'var_unit': var_unit,
                'var_display': var_display,
                'height': height,
                'sources': {}
            }
        
        variables_info[group_key]['sources'][source] = col
    
    # 只保留有多个数据源的变量组
    matched_variables = {k: v for k, v in variables_info.items() 
                        if len(v['sources']) > 1 and v['var_type'] != 'unknown'}
    
    return matched_variables

def plot_daily_smoothed_multi_source(df, var_group, window_days=1, figsize=(14, 8), 
                                   show_raw=False, show_std=True, min_periods=None):
    """为同一变量的多个数据源绘制日平滑对比图（含标准差阴影）
    
    参数:
        df: 包含数据的DataFrame
        var_group: 变量组信息字典
        window_days: 平滑窗口天数
        figsize: 图表大小
        show_raw: 是否显示原始数据
        show_std: 是否显示标准差阴影
        min_periods: 计算平滑值所需的最小有效值数
    """
    
    var_type = var_group['var_type']
    var_unit = var_group['var_unit']
    var_display = var_group['var_display']
    height = var_group['height']
    sources = var_group['sources']
    
    # 计算窗口大小
    window_size = window_days * calculate_daily_window(df)
    time_scale = "Daily" if window_days < 30 else "Monthly"
    
    # 如果min_periods未指定，则使用窗口大小的1/4（对于月平滑更宽松以处理缺失数据）
    if min_periods is None:
        if window_days >= 30:  # 月平滑
            min_periods = max(int(window_size / 4), 1)
        else:  # 日平滑
            min_periods = max(int(window_size * 3 / 4), 1)
    
    print(f"{time_scale} smoothed comparison for {var_display} at {height}, sources: {list(sources.keys())}")
    
    # 创建图表
    fig, ax = plt.subplots(figsize=figsize)
    
    # 定义数据源的颜色和线型
    source_styles = {
        'obs': {'color': SOURCE_COLORS['obs'], 'linestyle': '-', 'linewidth': 2.5, 'alpha': 1.0, 'name': LABEL_CONFIG['sources']['obs']},
        'ec': {'color': SOURCE_COLORS['ec'], 'linestyle': '-', 'linewidth': 2.5, 'alpha': 0.9, 'name': LABEL_CONFIG['sources']['ec']},
        'gfs': {'color': SOURCE_COLORS['gfs'], 'linestyle': '-', 'linewidth': 2.5, 'alpha': 0.9, 'name': LABEL_CONFIG['sources']['gfs']}
    }
    
    # 为每个数据源绘制数据
    for source, column in sources.items():
        if source not in source_styles:
            continue
            
        style = source_styles[source]
        
        # 确定颜色：温度变量使用暖色系，风速变量使用数据源颜色
        if var_type == 'temperature' and isinstance(VARIABLE_COLORS['temperature'], dict):
            color = VARIABLE_COLORS['temperature'].get(source, style['color'])
        elif var_type == 'temperature' and VARIABLE_COLORS['temperature'] is not None:
            color = VARIABLE_COLORS['temperature']
        else:
            color = style['color']
        
        # 检查数据质量
        valid_data_pct = 100 * df[column].count() / len(df)
        print(f"  {source} ({column}) completeness: {valid_data_pct:.1f}%")
        
        # 如果需要，绘制原始数据
        if show_raw:
            ax.plot(df.index, df[column], 
                   color=color, 
                   alpha=0.15, 
                   linewidth=0.5,
                   linestyle=style['linestyle'],
                   label=f'{style["name"]} Raw')
        
        # 计算移动平均和标准差
        rolling_mean = df[column].rolling(
            window=window_size, 
            center=True, 
            min_periods=min_periods
        ).mean()
        
        rolling_std = df[column].rolling(
            window=window_size, 
            center=True, 
            min_periods=min_periods
        ).std()
        
        # 绘制标准差阴影
        if show_std:
            ax.fill_between(df.index, 
                           rolling_mean - rolling_std, 
                           rolling_mean + rolling_std, 
                           color=color, alpha=0.2, 
                           label=f'{style["name"]} ±1σ')
        
        # 绘制平滑线
        smoothing_label = f'{window_days}-Day' if window_days < 30 else f'{window_days/30:.1f}-Month'
        ax.plot(df.index, rolling_mean, 
               color=color, 
               linestyle=style['linestyle'],
               linewidth=style['linewidth'],
               alpha=style['alpha'],
               label=f'{style["name"]} {smoothing_label}')
    
    # 设置标题和标签
    smoothing_label = f'{window_days}-Day' if window_days < 30 else f'{window_days/30:.1f}-Month'
    height_str = f' at {height}' if height != 'unknown' else ''
    
    ax.set_title(f"Changma Station {var_display}{height_str} - Multi-Source Comparison ({smoothing_label} Smoothed)", 
                fontsize=16, pad=20)
    ax.set_xlabel('Date', fontsize=12)
    ax.set_ylabel(f'{var_display} ({var_unit})', fontsize=12)
    
    # 设置x轴日期格式
    ax.xaxis.set_major_formatter(mdates.DateFormatter(X_AXIS_CONFIG['date_format']))
    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=X_AXIS_CONFIG['interval']))
    plt.setp(ax.xaxis.get_majorticklabels(), 
             rotation=X_AXIS_CONFIG['rotation'], 
             ha=X_AXIS_CONFIG['ha'])
    
    # 添加网格线和图例
    ax.grid(True, linestyle='--', alpha=0.7)
    ax.legend(loc='best', frameon=True, facecolor='white', framealpha=0.9)
    
    # 设置Y轴范围和刻度（如果是风速变量）
    if var_type == 'wind_speed' and Y_AXIS_CONFIG['wind_speed']['ylim'] is not None:
        ax.set_ylim(Y_AXIS_CONFIG['wind_speed']['ylim'])
        ax.set_yticks(Y_AXIS_CONFIG['wind_speed']['yticks'])
    elif var_type == 'temperature' and Y_AXIS_CONFIG['temperature']['ylim'] is not None:
        ax.set_ylim(Y_AXIS_CONFIG['temperature']['ylim'])
        if Y_AXIS_CONFIG['temperature']['yticks'] is not None:
            ax.set_yticks(Y_AXIS_CONFIG['temperature']['yticks'])
    
    # 紧凑布局
    plt.tight_layout()
    
    # 保存图片
    time_scale_str = "daily" if window_days < 30 else "monthly"
    window_str = f"{window_days}day" if window_days < 30 else f"{int(window_days/30)}month"
    
    # 根据时间尺度和变量类型决定保存位置
    if time_scale_str == "monthly":
        save_dir = os.path.join(results_dir, "monthly")
    else:
        save_dir = os.path.join(results_dir, var_type)
    
    os.makedirs(save_dir, exist_ok=True)
    
    # 生成文件名
    height_str = f"_{height}" if height != 'unknown' else ""
    std_str = "_with_std" if show_std else ""
    raw_str = "_with_raw" if show_raw else ""
    filename = f'{var_type}{height_str}_smoothed_multi_source_{time_scale_str}_{window_str}{std_str}{raw_str}.png'
    save_path = os.path.join(save_dir, filename)
    
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"  Saved: {save_path}")
    
    plt.close()

def plot_all_variables_smoothed_overview_with_power(df, matched_variables, window_days=1, figsize=(16, 12), 
                                                  show_std=True):
    """绘制所有变量的日平滑总览图（含标准差阴影和功率数据）"""
    
    # 找到功率列
    power_cols = [col for col in df.columns if 'power' in col.lower()]
    power_col = power_cols[0] if power_cols else None
    print(f"Power column found: {power_col}")
    
    # 按变量类型分组
    temp_vars = {k: v for k, v in matched_variables.items() if v['var_type'] == 'temperature'}
    wind_vars = {k: v for k, v in matched_variables.items() if v['var_type'] == 'wind_speed'}
    
    if not temp_vars and not wind_vars:
        print("No matched variables found for overview plot")
        return
    
    # 计算子图布局
    n_temp = len(temp_vars)
    n_wind = len(wind_vars)
    total_plots = n_temp + n_wind
    
    if total_plots == 0:
        return
    
    # 计算网格布局
    cols = min(2, total_plots)
    rows = (total_plots + cols - 1) // cols
    
    fig, axes = plt.subplots(rows, cols, figsize=figsize)
    if total_plots == 1:
        axes = [axes]
    elif rows == 1:
        axes = [axes] if cols == 1 else axes
    else:
        axes = axes.flatten()
    
    # 计算窗口大小
    window_size = window_days * calculate_daily_window(df)
    min_periods = max(int(window_size * 3 / 4), 1) if window_days < 30 else max(int(window_size / 4), 1)
    
    # 数据源样式
    source_styles = {
        'obs': {'color': SOURCE_COLORS['obs'], 'linestyle': '-', 'linewidth': 2.0, 'name': LABEL_CONFIG['sources']['obs']},
        'ec': {'color': SOURCE_COLORS['ec'], 'linestyle': '-', 'linewidth': 2.0, 'name': LABEL_CONFIG['sources']['ec']},
        'gfs': {'color': SOURCE_COLORS['gfs'], 'linestyle': '-', 'linewidth': 2.0, 'name': LABEL_CONFIG['sources']['gfs']}
    }
    
    plot_idx = 0
    
    # 绘制温度变量
    for var_key, var_group in temp_vars.items():
        if plot_idx >= len(axes):
            break
            
        ax = axes[plot_idx]
        
        for source, column in var_group['sources'].items():
            if source in source_styles:
                style = source_styles[source]
                
                # 为温度变量使用暖色系
                if isinstance(VARIABLE_COLORS['temperature'], dict):
                    color = VARIABLE_COLORS['temperature'].get(source, style['color'])
                else:
                    color = style['color']
                
                # 计算移动平均和标准差
                rolling_mean = df[column].rolling(
                    window=window_size, center=True, min_periods=min_periods
                ).mean()
                rolling_std = df[column].rolling(
                    window=window_size, center=True, min_periods=min_periods
                ).std()
                
                # 绘制标准差阴影
                if show_std:
                    ax.fill_between(df.index, 
                                   rolling_mean - rolling_std, 
                                   rolling_mean + rolling_std, 
                                   color=color, alpha=0.15)
                
                # 绘制平滑线
                ax.plot(df.index, rolling_mean, 
                       color=color, 
                       linestyle=style['linestyle'],
                       linewidth=style['linewidth'],
                       label=style['name'])
        
        # 添加功率数据到右y轴
        if power_col and power_col in df.columns:
            ax2 = ax.twinx()
            
            # 计算功率的移动平均和标准差
            power_rolling_mean = df[power_col].rolling(
                window=window_size, center=True, min_periods=min_periods
            ).mean()
            power_rolling_std = df[power_col].rolling(
                window=window_size, center=True, min_periods=min_periods
            ).std()
            
            # 绘制功率标准差阴影
            if show_std and not power_rolling_std.isna().all():
                ax2.fill_between(df.index, 
                               power_rolling_mean - power_rolling_std, 
                               power_rolling_mean + power_rolling_std, 
                               color=POWER_COLOR, alpha=0.15)
            
            # 绘制功率平滑线
            ax2.plot(df.index, power_rolling_mean, 
                    color=POWER_COLOR, 
                    linestyle='-',
                    linewidth=2.0,
                    label='Power')
            
            # 设置右y轴
            ax2.set_ylabel('Power (MW)', fontsize=10, color=POWER_COLOR)
            ax2.tick_params(axis='y', labelcolor=POWER_COLOR)
            ax2.grid(False)
        
        height_str = f' at {var_group["height"]}' if var_group["height"] != 'unknown' else ''
        ax.set_title(f'Temperature{height_str}', fontsize=12)
        ax.set_ylabel('Temperature (°C)', fontsize=10)
        ax.grid(True, alpha=0.3)
        
        # 设置X轴格式
        ax.xaxis.set_major_formatter(mdates.DateFormatter(X_AXIS_CONFIG['date_format']))
        ax.xaxis.set_major_locator(mdates.MonthLocator(interval=X_AXIS_CONFIG['interval']))
        plt.setp(ax.xaxis.get_majorticklabels(), 
                 rotation=X_AXIS_CONFIG['rotation'], 
                 ha=X_AXIS_CONFIG['ha'])
        
        # 组合图例
        if plot_idx == 0:  # 只在第一个子图显示图例
            handles1, labels1 = ax.get_legend_handles_labels()
            handles2, labels2 = [], []
            if power_col:
                try:
                    handles2, labels2 = ax2.get_legend_handles_labels()
                except:
                    pass
            
            all_handles = handles1 + handles2
            all_labels = labels1 + labels2
            if all_handles:
                ax.legend(all_handles, all_labels, loc='best', fontsize=9)
        
        plot_idx += 1
    
    # 绘制风速变量
    for var_key, var_group in wind_vars.items():
        if plot_idx >= len(axes):
            break
            
        ax = axes[plot_idx]
        
        for source, column in var_group['sources'].items():
            if source in source_styles:
                style = source_styles[source]
                
                # 风速变量使用数据源颜色
                color = style['color']
                
                # 计算移动平均和标准差
                rolling_mean = df[column].rolling(
                    window=window_size, center=True, min_periods=min_periods
                ).mean()
                rolling_std = df[column].rolling(
                    window=window_size, center=True, min_periods=min_periods
                ).std()
                
                # 绘制标准差阴影
                if show_std:
                    ax.fill_between(df.index, 
                                   rolling_mean - rolling_std, 
                                   rolling_mean + rolling_std, 
                                   color=color, alpha=0.15)
                
                # 绘制平滑线
                ax.plot(df.index, rolling_mean, 
                       color=color, 
                       linestyle=style['linestyle'],
                       linewidth=style['linewidth'],
                       label=style['name'])
        
        # 添加功率数据到右y轴
        if power_col and power_col in df.columns:
            ax2 = ax.twinx()
            
            # 计算功率的移动平均和标准差
            power_rolling_mean = df[power_col].rolling(
                window=window_size, center=True, min_periods=min_periods
            ).mean()
            power_rolling_std = df[power_col].rolling(
                window=window_size, center=True, min_periods=min_periods
            ).std()
            
            # 绘制功率标准差阴影
            if show_std and not power_rolling_std.isna().all():
                ax2.fill_between(df.index, 
                               power_rolling_mean - power_rolling_std, 
                               power_rolling_mean + power_rolling_std, 
                               color=POWER_COLOR, alpha=0.15)
            
            # 绘制功率平滑线
            ax2.plot(df.index, power_rolling_mean, 
                    color=POWER_COLOR, 
                    linestyle='-',
                    linewidth=2.0,
                    label='Power')
            
            # 设置右y轴
            ax2.set_ylabel('Power (MW)', fontsize=10, color=POWER_COLOR)
            ax2.tick_params(axis='y', labelcolor=POWER_COLOR)
            ax2.grid(False)
        
        height_str = f' at {var_group["height"]}' if var_group["height"] != 'unknown' else ''
        ax.set_title(f'Wind Speed{height_str}', fontsize=12)
        ax.set_ylabel('Wind Speed (m/s)', fontsize=10)
        ax.grid(True, alpha=0.3)
        
        # 设置X轴格式
        ax.xaxis.set_major_formatter(mdates.DateFormatter(X_AXIS_CONFIG['date_format']))
        ax.xaxis.set_major_locator(mdates.MonthLocator(interval=X_AXIS_CONFIG['interval']))
        plt.setp(ax.xaxis.get_majorticklabels(), 
                 rotation=X_AXIS_CONFIG['rotation'], 
                 ha=X_AXIS_CONFIG['ha'])
        
        # 设置Y轴范围和刻度（风速固定0-16 m/s）
        ax.set_ylim(Y_AXIS_CONFIG['wind_speed']['ylim'])
        ax.set_yticks(Y_AXIS_CONFIG['wind_speed']['yticks'])
        
        plot_idx += 1
    
    # 隐藏多余的子图
    for i in range(plot_idx, len(axes)):
        axes[i].set_visible(False)
    
    # 总标题
    time_scale = "Daily" if window_days < 30 else "Monthly"
    smoothing_label = f'{window_days}-Day' if window_days < 30 else f'{window_days/30:.1f}-Month'
    std_str = " with ±1σ" if show_std else ""
    power_str = " and Power" if power_col else ""
    fig.suptitle(f'Changma Station Multi-Source Comparison Overview ({smoothing_label} Smoothed{std_str}{power_str})', 
                fontsize=16, y=0.98)
    
    plt.tight_layout()
    plt.subplots_adjust(top=0.93)
    
    # 保存图片
    time_scale_str = "daily" if window_days < 30 else "monthly"
    window_str = f"{window_days}day" if window_days < 30 else f"{int(window_days/30)}month"
    std_str = "_with_std" if show_std else ""
    power_str = "_with_power" if power_col else ""
    
    save_dir = os.path.join(results_dir, "monthly" if time_scale_str == "monthly" else ".")
    os.makedirs(save_dir, exist_ok=True)
    
    filename = f'all_variables_smoothed_overview_{time_scale_str}_{window_str}{std_str}{power_str}.png'
    save_path = os.path.join(save_dir, filename)
    
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Smoothed overview with power saved: {save_path}")
    
    plt.close()

def main():
    print("Loading data from:", data_path)
    df = load_data(data_path)
    
    if df is None:
        print("Error: Failed to load data.")
        return
    
    print(f"Data loaded successfully. Shape: {df.shape}")
    print(f"Date range: {df.index.min()} to {df.index.max()}")
    
    # 找到匹配的变量组
    matched_variables = find_matching_variables(df)
    
    if not matched_variables:
        print("No matching variables found across different data sources.")
        return
    
    print(f"\nFound {len(matched_variables)} variable groups with multiple data sources:")
    for group_key, group_info in matched_variables.items():
        sources_str = ", ".join(group_info['sources'].keys())
        print(f"  {group_info['var_display']} at {group_info['height']}: {sources_str}")
    
    # 窗口天数选项
    daily_window = 1  # 1天平滑
    monthly_window = 30  # 30天（月）平滑
    
    daily_min_periods = None  # 默认值
    monthly_min_periods = 2  # 宽松设置
    
    print("\nGenerating daily smoothed multi-source comparison plots with std deviation...")
    
    # 为每个变量组生成带标准差的日平滑对比图
    for group_key, group_info in matched_variables.items():
        print(f"\nProcessing {group_key}...")
        
        # 日平滑对比图（带标准差阴影）
        plot_daily_smoothed_multi_source(df, group_info, window_days=daily_window, 
                                       show_raw=False, show_std=True, 
                                       min_periods=daily_min_periods)
        
        # 月平滑对比图（带标准差阴影）
        plot_daily_smoothed_multi_source(df, group_info, window_days=monthly_window, 
                                       show_raw=False, show_std=True, 
                                       min_periods=monthly_min_periods)
    
    # 生成带标准差和功率的总览图
    print("\nGenerating smoothed overview plots with std deviation and power...")
    plot_all_variables_smoothed_overview_with_power(df, matched_variables, window_days=daily_window, show_std=True)
    plot_all_variables_smoothed_overview_with_power(df, matched_variables, window_days=monthly_window, show_std=True)
    
    print("\nAll daily smoothed multi-source comparison plots completed successfully!")
    print(f"Results saved in: {results_dir}")

if __name__ == "__main__":
    main()