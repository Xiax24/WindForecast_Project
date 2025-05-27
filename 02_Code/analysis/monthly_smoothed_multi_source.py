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
    'ec': '#4daf4a',   # 绿色 - ECMWF  
    'gfs': '#377eb8'   # 蓝色 - GFS
}

# 变量类型颜色配置
VARIABLE_COLORS = {
    'temperature': {
        'obs': '#d62728',   # 深红色 - Observation
        'ec': '#ff7f0e',    # 橙色 - ECMWF
        'gfs': "#ff96be"    # 浅红色 - GFS
    },
    'wind_speed': None         # None表示使用数据源颜色
}

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
        'ylim': (3, 10),     # Y轴范围 0-10 m/s
        'yticks': range(3, 11, 1)  # Y轴刻度 0,1,2,3,4,5,6,7,8,9,10
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
results_dir = '/Users/xiaxin/work/WindForecast_Project/03_Results/multi_source_comparison'

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
        var_unit = '°C'
        var_display = 'Temperature'
    elif 'wind_speed' in var_base.lower():
        var_type = 'wind_speed'
        var_unit = 'm/s'
        var_display = 'Wind Speed'
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

def plot_multi_source_comparison(df, var_group, window_days=1, figsize=(14, 8), 
                                show_raw=False, min_periods=None):
    """为同一变量的多个数据源绘制对比图
    
    参数:
        df: 包含数据的DataFrame
        var_group: 变量组信息字典
        window_days: 平滑窗口天数
        figsize: 图表大小
        show_raw: 是否显示原始数据
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
    
    print(f"{time_scale} comparison for {var_display} at {height}, sources: {list(sources.keys())}")
    
    # 创建图表
    fig, ax = plt.subplots(figsize=figsize)
    
    # 定义数据源的颜色和线型
    source_styles = {
        'obs': {'color': '#377eb8', 'linestyle': '-', 'linewidth': 2.5, 'alpha': 1.0, 'name': 'Observation'},
        'ec': {'color': '#4daf4a', 'linestyle': '--', 'linewidth': 2.5, 'alpha': 0.9, 'name': 'ECMWF'},
        'gfs': {'color': '#984ea3', 'linestyle': '-.', 'linewidth': 2.5, 'alpha': 0.9, 'name': 'GFS'}
    }
    
    # 为每个数据源绘制数据
    for source, column in sources.items():
        if source not in source_styles:
            continue
            
        style = source_styles[source]
        
        # 检查数据质量
        valid_data_pct = 100 * df[column].count() / len(df)
        print(f"  {source} ({column}) completeness: {valid_data_pct:.1f}%")
        
        # 确定颜色：温度变量使用暖色系，风速变量使用数据源颜色
        if var_type == 'temperature' and isinstance(VARIABLE_COLORS['temperature'], dict):
            color = VARIABLE_COLORS['temperature'].get(source, style['color'])
        elif var_type == 'temperature' and VARIABLE_COLORS['temperature'] is not None:
            color = VARIABLE_COLORS['temperature']
        else:
            color = style['color']
        
        # 如果需要，绘制原始数据
        if show_raw:
            ax.plot(df.index, df[column], 
                   color=color, 
                   alpha=0.15, 
                   linewidth=0.5,
                   linestyle=style['linestyle'])
        
        # 计算移动平均
        rolling_mean = df[column].rolling(
            window=window_size, 
            center=True, 
            min_periods=min_periods
        ).mean()
        
        # 绘制平滑线
        ax.plot(df.index, rolling_mean, 
               color=color, 
               linestyle=style['linestyle'],
               linewidth=style['linewidth'],
               alpha=style['alpha'],
               label=style['name'])
    
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
    filename = f'{var_type}{height_str}_multi_source_{time_scale_str}_{window_str}.png'
    save_path = os.path.join(save_dir, filename)
    
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"  Saved: {save_path}")
    
    plt.close()

def plot_all_variables_overview(df, matched_variables, window_days=1, figsize=(16, 12)):
    """绘制所有变量的总览图"""
    
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
    
    # 数据源样式（使用全局配置）
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
                
                rolling_mean = df[column].rolling(
                    window=window_size, center=True, min_periods=min_periods
                ).mean()
                
                ax.plot(df.index, rolling_mean, 
                       color=color, 
                       linestyle=style['linestyle'],
                       linewidth=style['linewidth'],
                       label=style['name'])
        
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
        
        # 设置Y轴范围和刻度
        if var_group["var_type"] == 'wind_speed':
            ax.set_ylim(Y_AXIS_CONFIG['wind_speed']['ylim'])
            ax.set_yticks(Y_AXIS_CONFIG['wind_speed']['yticks'])
        elif var_group["var_type"] == 'temperature' and Y_AXIS_CONFIG['temperature']['ylim'] is not None:
            ax.set_ylim(Y_AXIS_CONFIG['temperature']['ylim'])
            if Y_AXIS_CONFIG['temperature']['yticks'] is not None:
                ax.set_yticks(Y_AXIS_CONFIG['temperature']['yticks'])
        
        if plot_idx == 0:  # 只在第一个子图显示图例
            ax.legend(loc='best', fontsize=9)
        
        plot_idx += 1
    
    # 绘制风速变量
    for var_key, var_group in wind_vars.items():
        if plot_idx >= len(axes):
            break
            
        ax = axes[plot_idx]
        
        for source, column in var_group['sources'].items():
            if source in source_styles:
                style = source_styles[source]
                
                # 为温度变量使用暖色系
                if var_group['var_type'] == 'temperature' and isinstance(VARIABLE_COLORS['temperature'], dict):
                    color = VARIABLE_COLORS['temperature'].get(source, style['color'])
                else:
                    color = style['color']
                
                rolling_mean = df[column].rolling(
                    window=window_size, center=True, min_periods=min_periods
                ).mean()
                
                ax.plot(df.index, rolling_mean, 
                       color=color, 
                       linestyle=style['linestyle'],
                       linewidth=style['linewidth'],
                       label=style['name'])
        
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
        
        # 设置Y轴范围和刻度（风速固定0-10 m/s）
        ax.set_ylim(Y_AXIS_CONFIG['wind_speed']['ylim'])
        ax.set_yticks(Y_AXIS_CONFIG['wind_speed']['yticks'])
        
        plot_idx += 1
    
    # 隐藏多余的子图
    for i in range(plot_idx, len(axes)):
        axes[i].set_visible(False)
    
    # 移除之前的X轴设置代码，因为现在每个子图都单独设置了
    
    # 总标题
    time_scale = "Daily" if window_days < 30 else "Monthly"
    smoothing_label = f'{window_days}-Day' if window_days < 30 else f'{window_days/30:.1f}-Month'
    fig.suptitle(f'Changma Station Multi-Source Comparison Overview ({smoothing_label} Smoothed)', 
                fontsize=16, y=0.98)
    
    plt.tight_layout()
    plt.subplots_adjust(top=0.93)
    
    # 保存图片
    time_scale_str = "daily" if window_days < 30 else "monthly"
    window_str = f"{window_days}day" if window_days < 30 else f"{int(window_days/30)}month"
    
    save_dir = os.path.join(results_dir, "monthly" if time_scale_str == "monthly" else ".")
    os.makedirs(save_dir, exist_ok=True)
    
    filename = f'all_variables_overview_{time_scale_str}_{window_str}.png'
    save_path = os.path.join(save_dir, filename)
    
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Overview saved: {save_path}")
    
    plt.close()

def plot_all_sources_multi_height(df, window_days=1, figsize=(16, 10), 
                                 show_raw=False, min_periods=None):
    """绘制所有数据源的多层风速对比图"""
    
    # 获取所有风速列
    wind_cols = [col for col in df.columns if 'wind_speed' in col.lower()]
    
    if not wind_cols:
        print("No wind speed columns found")
        return
    
    # 按数据源和高度分组
    sources_data = {}
    
    for col in wind_cols:
        # 确定数据源
        source = 'unknown'
        if col.startswith('obs_'):
            source = 'obs'
        elif col.startswith('ec_'):
            source = 'ec'
        elif col.startswith('gfs_'):
            source = 'gfs'
        
        # 提取高度
        height = 'unknown'
        if '10m' in col:
            height = '10m'
        elif '30m' in col:
            height = '30m'
        elif '50m' in col:
            height = '50m'
        elif '70m' in col:
            height = '70m'
        elif '100m' in col:
            height = '100m'
        
        if source not in sources_data:
            sources_data[source] = {}
        
        sources_data[source][height] = col
    
    if not sources_data:
        print("No valid wind speed data found")
        return
    
    # 计算窗口大小
    window_size = window_days * calculate_daily_window(df)
    time_scale = "Daily" if window_days < 30 else "Monthly"
    
    if min_periods is None:
        if window_days >= 30:
            min_periods = max(int(window_size / 4), 1)
        else:
            min_periods = max(int(window_size * 3 / 4), 1)
    
    print(f"{time_scale} all-sources multi-height plot, window size: {window_size}, min_periods: {min_periods}")
    
    # 创建图表
    fig, ax = plt.subplots(figsize=figsize)
    
    # 定义数据源的基础颜色（使用全局配置）
    source_styles = {
        'obs': {'color': SOURCE_COLORS['obs'], 'name': LABEL_CONFIG['sources']['obs'], 'linestyle': '-'},
        'ec': {'color': SOURCE_COLORS['ec'], 'name': LABEL_CONFIG['sources']['ec'], 'linestyle': '-'},
        'gfs': {'color': SOURCE_COLORS['gfs'], 'name': LABEL_CONFIG['sources']['gfs'], 'linestyle': '-'}
    }
    
    # 定义高度的透明度和线宽变化
    height_styles = {
        '10m': {'alpha': 0.7, 'linewidth': 2.0},
        '30m': {'alpha': 0.8, 'linewidth': 2.2},
        '50m': {'alpha': 0.9, 'linewidth': 2.5},
        '70m': {'alpha': 1.0, 'linewidth': 2.8},
        '100m': {'alpha': 1.0, 'linewidth': 3.0}
    }
    
    # 高度排序
    height_order = ['10m', '30m', '50m', '70m', '100m']
    
    # 绘制每个数据源的多层数据
    for source in ['obs', 'ec', 'gfs']:  # 按特定顺序绘制
        if source not in sources_data:
            continue
            
        source_style = source_styles[source]
        heights_data = sources_data[source]
        
        print(f"  Processing {source_style['name']} with heights: {list(heights_data.keys())}")
        
        # 按高度顺序绘制
        for height in height_order:
            if height not in heights_data:
                continue
                
            column = heights_data[height]
            height_style = height_styles.get(height, {'alpha': 0.8, 'linewidth': 2.0})
            
            # 检查数据质量
            valid_data_pct = 100 * df[column].count() / len(df)
            print(f"    {height} ({column}) completeness: {valid_data_pct:.1f}%")
            
            # 如果需要，绘制原始数据
            if show_raw:
                ax.plot(df.index, df[column], 
                       color=source_style['color'], 
                       alpha=0.1, 
                       linewidth=0.5,
                       linestyle=source_style['linestyle'])
            
            # 计算移动平均
            rolling_mean = df[column].rolling(
                window=window_size, 
                center=True, 
                min_periods=min_periods
            ).mean()
            
            # 绘制平滑线
            label = f"{source_style['name']} {height}"
            ax.plot(df.index, rolling_mean, 
                   color=source_style['color'], 
                   linestyle=source_style['linestyle'],
                   linewidth=height_style['linewidth'],
                   alpha=height_style['alpha'],
                   label=label)
    
    # 设置标题和标签
    smoothing_label = f'{window_days}-Day' if window_days < 30 else f'{window_days/30:.1f}-Month'
    
    ax.set_title(f"Changma Station Wind Speed - All Sources Multi-Height Comparison ({smoothing_label} Smoothed)", 
                fontsize=16, pad=20)
    ax.set_xlabel('Date', fontsize=12)
    ax.set_ylabel('Wind Speed (m/s)', fontsize=12)
    
    # 设置x轴日期格式
    ax.xaxis.set_major_formatter(mdates.DateFormatter(X_AXIS_CONFIG['date_format']))
    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=X_AXIS_CONFIG['interval']))
    plt.setp(ax.xaxis.get_majorticklabels(), 
             rotation=X_AXIS_CONFIG['rotation'], 
             ha=X_AXIS_CONFIG['ha'])
    
    # 添加网格线和图例
    ax.grid(True, linestyle='--', alpha=0.7)
    
    # 创建自定义图例 - 按数据源分组
    handles, labels = ax.get_legend_handles_labels()
    
    # 按数据源分组图例
    legend_groups = {}
    for handle, label in zip(handles, labels):
        source_name = label.split()[0] + (' ' + label.split()[1] if len(label.split()) > 1 and label.split()[1] in ['Observation'] else '')
        if source_name not in legend_groups:
            legend_groups[source_name] = []
        legend_groups[source_name].append((handle, label))
    
    # 创建分组图例
    legend_handles = []
    legend_labels = []
    
    for source_name in ['Observation', 'ECMWF', 'GFS']:
        if source_name in legend_groups:
            # 添加数据源标题
            legend_handles.append(plt.Line2D([0], [0], color='none'))
            legend_labels.append(f"--- {source_name} ---")
            
            # 添加该数据源的所有高度
            for handle, label in legend_groups[source_name]:
                height = label.split()[-1]  # 获取高度
                legend_handles.append(handle)
                legend_labels.append(f"  {height}")
    
    ax.legend(legend_handles, legend_labels, loc='center left', bbox_to_anchor=(1, 0.5), 
             frameon=True, facecolor='white', framealpha=0.9)
    
    # 紧凑布局
    plt.tight_layout()
    
    # 保存图片
    time_scale_str = "daily" if window_days < 30 else "monthly"
    window_str = f"{window_days}day" if window_days < 30 else f"{int(window_days/30)}month"
    
    save_dir = os.path.join(results_dir, "monthly" if time_scale_str == "monthly" else "multi_height")
    os.makedirs(save_dir, exist_ok=True)
    
    filename = f'all_sources_multi_height_{time_scale_str}_{window_str}.png'
    save_path = os.path.join(save_dir, filename)
    
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"All-sources multi-height plot saved: {save_path}")
    
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
    
    print("\nGenerating multi-source comparison plots...")
    
    # 为每个变量组生成对比图
    for group_key, group_info in matched_variables.items():
        print(f"\nProcessing {group_key}...")
        
        # 日平滑对比图
        plot_multi_source_comparison(df, group_info, window_days=daily_window, 
                                   show_raw=False, min_periods=daily_min_periods)
        
        # 月平滑对比图
        plot_multi_source_comparison(df, group_info, window_days=monthly_window, 
                                   show_raw=False, min_periods=monthly_min_periods)
    
    # 生成总览图
    print("\nGenerating overview plots...")
    plot_all_variables_overview(df, matched_variables, window_days=daily_window)
    plot_all_variables_overview(df, matched_variables, window_days=monthly_window)
    
    # 生成所有数据源多层风速对比图
    print("\nGenerating all-sources multi-height wind speed plots...")
    plot_all_sources_multi_height(df, window_days=daily_window, 
                                 show_raw=False, min_periods=daily_min_periods)
    plot_all_sources_multi_height(df, window_days=monthly_window, 
                                 show_raw=False, min_periods=monthly_min_periods)
    
    print("\nAll multi-source comparison plots completed successfully!")
    print(f"Results saved in: {results_dir}")

if __name__ == "__main__":
    main()