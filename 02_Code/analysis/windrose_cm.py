#!/usr/bin/env python3
"""
昌马站风向玫瑰图分析 - 修复版
修复字体问题和列名识别问题

日期: 2025-05-24
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
from datetime import datetime
import matplotlib.cm as cm
from matplotlib.colors import LinearSegmentedColormap
import seaborn as sns

# 尝试导入windrose包，如果不存在则提供安装指南
try:
    from windrose import WindroseAxes
    from matplotlib.projections import register_projection
    register_projection(WindroseAxes)
except ImportError:
    print("请先安装windrose包: pip install windrose")
    print("或者访问 https://github.com/python-windrose/windrose 获取更多信息")
    exit(1)

# 设置全局字体 - 修复字体问题
plt.rcParams['font.sans-serif'] = ['Arial Unicode MS', 'SimHei', 'DejaVu Sans']  # 尝试多种字体
plt.rcParams['axes.unicode_minus'] = False  # 用于正确显示负号
plt.rcParams['figure.figsize'] = (10, 8)  # 图形大小
plt.rcParams['savefig.dpi'] = 300  # 保存图片DPI

class WindRoseAnalyzer:
    """风向玫瑰图分析器修复版"""
    
    def __init__(self):
        # 季节映射
        self.season_map = {
            1: '冬季', 2: '冬季', 3: '春季',
            4: '春季', 5: '春季', 6: '夏季',
            7: '夏季', 8: '夏季', 9: '秋季',
            10: '秋季', 11: '秋季', 12: '冬季'
        }
        
        # 关注高度
        self.heights = [10, 70]  # 只关注10m和70m两个高度
        
        # 风向分类（16方位）
        self.direction_bins = np.arange(0, 360+22.5, 22.5)
        self.direction_labels = ['N','NNE','NE','ENE','E','ESE','SE','SSE',
                                'S','SSW','SW','WSW','W','WNW','NW','NNW']
    
    def load_and_preprocess(self, file_path):
        """加载并预处理数据"""
        print(f"加载数据: {file_path}")
        
        # 读取CSV文件
        data = pd.read_csv(file_path)
        
        # 输出所有列名以调试
        print("原始数据列名:")
        for i, col in enumerate(data.columns):
            print(f"  {i}: {col}")
        
        # 识别时间列并转换
        time_columns = [col for col in data.columns if 'time' in col.lower() or 'date' in col.lower()]
        
        if time_columns:
            time_col = time_columns[0]
            data['timestamp'] = pd.to_datetime(data[time_col])
        else:
            # 如果没有明确的时间列，尝试使用第一列
            first_col = data.columns[0]
            data['timestamp'] = pd.to_datetime(data[first_col])
        
        # 添加时间相关列
        data['hour'] = data['timestamp'].dt.hour
        data['month'] = data['timestamp'].dt.month
        data['season'] = data['month'].map(self.season_map)
        data['year'] = data['timestamp'].dt.year
        
        # 创建新的DataFrame
        processed_data = pd.DataFrame()
        processed_data['timestamp'] = data['timestamp']
        processed_data['hour'] = data['hour']
        processed_data['month'] = data['month']
        processed_data['season'] = data['season']
        processed_data['year'] = data['year']
        
        # 根据数据描述文件，识别确切的列名
        for height in self.heights:
            # 风速列
            obs_speed_col = f'obs_wind_speed_{height}m'
            ec_speed_col = f'ec_wind_speed_{height}m'
            gfs_speed_col = f'gfs_wind_speed_{height}m'
            
            # 风向列
            obs_dir_col = f'obs_wind_direction_{height}m'
            ec_dir_col = f'ec_wind_direction_{height}m'
            gfs_dir_col = f'gfs_wind_direction_{height}m'
            
            # 检查并添加这些列
            if obs_speed_col in data.columns:
                processed_data[f'wind_speed_{height}m'] = data[obs_speed_col]
            
            if obs_dir_col in data.columns:
                processed_data[f'wind_dir_{height}m'] = data[obs_dir_col]
            
            # 计算预报误差
            if obs_speed_col in data.columns and ec_speed_col in data.columns:
                processed_data[f'ec_error_{height}m'] = data[ec_speed_col] - data[obs_speed_col]
                processed_data[f'ec_abs_error_{height}m'] = (data[ec_speed_col] - data[obs_speed_col]).abs()
            
            if obs_speed_col in data.columns and gfs_speed_col in data.columns:
                processed_data[f'gfs_error_{height}m'] = data[gfs_speed_col] - data[obs_speed_col]
                processed_data[f'gfs_abs_error_{height}m'] = (data[gfs_speed_col] - data[obs_speed_col]).abs()
        
        # 查找稳定度列
        if 'stability_final' in data.columns:
            processed_data['stability_final'] = data['stability_final']
        
        # 清理异常值
        for height in self.heights:
            speed_col = f'wind_speed_{height}m'
            dir_col = f'wind_dir_{height}m'
            
            if speed_col in processed_data.columns:
                processed_data[speed_col] = processed_data[speed_col].apply(
                    lambda x: np.nan if pd.isna(x) or x < 0 or x > 50 else x
                )
            
            if dir_col in processed_data.columns:
                processed_data[dir_col] = processed_data[dir_col].apply(
                    lambda x: np.nan if pd.isna(x) or x < 0 or x >= 360 else x
                )
                
                # 创建16方位风向分类
                processed_data[f'wind_dir_{height}m_16dir'] = pd.cut(
                    processed_data[dir_col], 
                    bins=self.direction_bins, 
                    labels=self.direction_labels, 
                    include_lowest=True
                )
        
        # 打印数据信息
        print(f"处理后数据形状: {processed_data.shape}")
        print(f"时间范围: {processed_data['timestamp'].min()} 到 {processed_data['timestamp'].max()}")
        
        # 显示数据可用性
        print("\n处理后数据列名:")
        for col in processed_data.columns:
            valid_count = processed_data[col].count()
            valid_percent = valid_count / len(processed_data) * 100
            print(f"  {col}: {valid_count}/{len(processed_data)} ({valid_percent:.1f}%)")
        
        self.data = processed_data
        return processed_data
    
    def create_standard_wind_rose(self, height, output_dir, time_period=None):
        """创建标准风向玫瑰图（频率分布）"""
        dir_col = f'wind_dir_{height}m'
        speed_col = f'wind_speed_{height}m'
        
        if dir_col not in self.data.columns or speed_col not in self.data.columns:
            print(f"警告: {height}m高度的风向或风速数据不存在")
            return None
        
        # 筛选数据
        if time_period == 'year':
            data = self.data
            period_label = "全年"
        elif time_period in ['春季', '夏季', '秋季', '冬季']:
            data = self.data[self.data['season'] == time_period]
            period_label = time_period
        elif isinstance(time_period, int):
            data = self.data[self.data['month'] == time_period]
            period_label = f"{time_period}月"
        else:
            data = self.data
            period_label = "全年"
        
        # 删除缺失值
        data = data.dropna(subset=[dir_col, speed_col])
        
        # 检查数据量
        if len(data) < 50:
            print(f"警告: {period_label}数据量不足 ({len(data)}个样本)，跳过生成")
            return None
        
        # 创建图形
        fig = plt.figure(figsize=(10, 10))
        ax = WindroseAxes.from_ax(fig=fig)
        
        # 绘制风玫瑰图
        ax.bar(
            data[dir_col], 
            data[speed_col], 
            normed=True, 
            opening=0.8, 
            edgecolor='white',
            nsector=16
        )
        
        # 添加图例和标题
        ax.set_legend(title="风速 (m/s)")
        title = f"昌马站 {height}m高度 {period_label} 风向频率分布"
        ax.set_title(title, fontsize=14)
        
        # 保存图片
        filename = f"{height}m_{period_label}_风向玫瑰图.png"
        output_path = os.path.join(output_dir, filename)
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"已生成: {output_path}")
        return output_path
    
    def create_comparison_wind_rose(self, height, output_dir, time_period=None):
        """创建EC vs GFS误差对比的风向玫瑰图"""
        dir_col = f'wind_dir_{height}m'
        dir16_col = f'wind_dir_{height}m_16dir'
        ec_error_col = f'ec_abs_error_{height}m'
        gfs_error_col = f'gfs_abs_error_{height}m'
        
        if (dir_col not in self.data.columns or 
            ec_error_col not in self.data.columns or 
            gfs_error_col not in self.data.columns):
            print(f"警告: {height}m高度的风向或误差数据不存在")
            return None
        
        # 筛选数据
        if time_period == 'year':
            data = self.data
            period_label = "全年"
        elif time_period in ['春季', '夏季', '秋季', '冬季']:
            data = self.data[self.data['season'] == time_period]
            period_label = time_period
        elif isinstance(time_period, int):
            data = self.data[self.data['month'] == time_period]
            period_label = f"{time_period}月"
        else:
            data = self.data
            period_label = "全年"
        
        # 删除缺失值
        data = data.dropna(subset=[dir16_col, ec_error_col, gfs_error_col])
        
        # 检查数据量
        if len(data) < 50:
            print(f"警告: {period_label}数据量不足 ({len(data)}个样本)，跳过生成")
            return None
        
        # 计算误差差异 (正值表示EC更好，负值表示GFS更好)
        data['error_diff'] = data[gfs_error_col] - data[ec_error_col]
        
        # 按16方位风向分组计算平均误差差异
        diff_by_dir = data.groupby(dir16_col)['error_diff'].mean().reset_index()
        
        # 转换风向标签为角度
        dir_degrees = np.arange(0, 360, 22.5)
        diff_by_dir['dir_degrees'] = dir_degrees
        
        # 创建图形
        fig = plt.figure(figsize=(10, 10))
        ax = plt.subplot(111, projection='polar')
        
        # 创建颜色映射 (蓝色为EC更好，红色为GFS更好)
        cmap = LinearSegmentedColormap.from_list(
            'ec_gfs', 
            [(0, 'red'), (0.5, 'white'), (1, 'blue')]
        )
        
        # 获取最大绝对差异值用于对称颜色映射
        max_abs_diff = max(abs(diff_by_dir['error_diff'].max()), abs(diff_by_dir['error_diff'].min()))
        norm = plt.Normalize(-max_abs_diff, max_abs_diff)
        
        # 绘制极坐标柱状图
        theta = np.radians(diff_by_dir['dir_degrees'])
        width = np.radians(22.5)
        bars = ax.bar(
            theta, 
            abs(diff_by_dir['error_diff']),  # 使用绝对值作为柱高
            width=width, 
            bottom=0.0,
            alpha=0.7,
            edgecolor='k',
            linewidth=0.5
        )
        
        # 为柱状图上色
        for i, bar in enumerate(bars):
            diff_value = diff_by_dir.iloc[i]['error_diff']
            bar.set_facecolor(cmap(norm(diff_value)))
        
        # 设置方位标签
        ax.set_xticks(np.radians(dir_degrees))
        ax.set_xticklabels(self.direction_labels)
        
        # 添加彩条
        sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
        sm.set_array([])
        cbar = plt.colorbar(sm, ax=ax, orientation='vertical', pad=0.1)
        cbar.set_label('误差差异 (GFS-EC) m/s', fontsize=12)
        
        # 添加图例说明
        legend_text = "正值(蓝色): EC更优\n负值(红色): GFS更优"
        props = dict(boxstyle='round', facecolor='white', alpha=0.7)
        ax.text(1.2, 0.05, legend_text, transform=ax.transAxes, fontsize=10,
                verticalalignment='bottom', bbox=props)
        
        # 设置标题
        title = f"昌马站 {height}m高度 {period_label}\nEC vs GFS模式预报误差对比"
        ax.set_title(title, fontsize=14)
        
        # 保存图片
        filename = f"{height}m_{period_label}_EC_GFS对比风玫瑰图.png"
        output_path = os.path.join(output_dir, filename)
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"已生成: {output_path}")
        return output_path
    
    def create_stability_wind_roses(self, height, output_dir):
        """创建不同稳定度条件下的风向玫瑰图"""
        dir_col = f'wind_dir_{height}m'
        speed_col = f'wind_speed_{height}m'
        
        if 'stability_final' not in self.data.columns:
            print("警告: 数据中不存在稳定度信息")
            return None
        
        if dir_col not in self.data.columns or speed_col not in self.data.columns:
            print(f"警告: {height}m高度的风向或风速数据不存在")
            return None
        
        # 创建图形
        fig = plt.figure(figsize=(15, 5))
        
        # 稳定度条件
        stability_conditions = ['unstable', 'neutral', 'stable']
        stability_names = {'unstable': '不稳定', 'neutral': '中性', 'stable': '稳定'}
        
        for i, stab in enumerate(stability_conditions):
            ax = fig.add_subplot(1, 3, i+1, projection='windrose')
            
            # 筛选特定稳定度条件的数据
            stab_data = self.data[self.data['stability_final'] == stab]
            stab_data = stab_data.dropna(subset=[dir_col, speed_col])
            
            # 检查是否有足够的数据
            if len(stab_data) < 10:
                ax.text(0.5, 0.5, f"数据不足: {len(stab_data)}个样本",
                        horizontalalignment='center',
                        verticalalignment='center',
                        transform=ax.transAxes)
            else:
                # 绘制风玫瑰图
                ax.bar(
                    stab_data[dir_col], 
                    stab_data[speed_col], 
                    normed=True, 
                    opening=0.8, 
                    edgecolor='white',
                    nsector=16
                )
            
            stab_name = stability_names.get(stab, stab)
            ax.set_title(f"{stab_name} (n={len(stab_data)})")
            
            if i == 0:
                ax.set_legend(title="风速 (m/s)")
            else:
                ax.set_legend([])
        
        fig.suptitle(f"昌马站 {height}m高度 不同稳定度条件下的风向分布", fontsize=16)
        plt.tight_layout(rect=[0, 0, 1, 0.95])
        
        # 保存图片
        filename = f"{height}m_稳定度风玫瑰图.png"
        output_path = os.path.join(output_dir, filename)
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"已生成: {output_path}")
        return output_path
    
    def generate_seasonal_wind_roses(self, output_dir):
        """生成季节性风向玫瑰图"""
        # 创建输出目录
        os.makedirs(output_dir, exist_ok=True)
        
        # 季节列表
        seasons = ['春季', '夏季', '秋季', '冬季']
        
        # 为每个高度生成全年风向玫瑰图
        for height in self.heights:
            print(f"\n生成{height}m高度风向玫瑰图...")
            
            # 全年风玫瑰图
            self.create_standard_wind_rose(height, output_dir, time_period='year')
            
            # 季节风玫瑰图
            for season in seasons:
                self.create_standard_wind_rose(height, output_dir, time_period=season)
        
        return True
    
    def generate_comparison_roses(self, output_dir):
        """生成EC vs GFS对比风玫瑰图"""
        # 创建输出目录
        os.makedirs(output_dir, exist_ok=True)
        
        # 季节列表
        seasons = ['春季', '夏季', '秋季', '冬季']
        
        # 为每个高度生成对比风玫瑰图
        for height in self.heights:
            print(f"\n生成{height}m高度EC vs GFS对比风玫瑰图...")
            
            # 全年对比图
            self.create_comparison_wind_rose(height, output_dir, time_period='year')
            
            # 季节对比图
            for season in seasons:
                self.create_comparison_wind_rose(height, output_dir, time_period=season)
        
        return True
    
    def generate_stability_roses(self, output_dir):
        """生成稳定度相关风玫瑰图"""
        # 创建输出目录
        os.makedirs(output_dir, exist_ok=True)
        
        # 检查稳定度数据是否存在
        if 'stability_final' not in self.data.columns:
            print("警告: 数据中不存在稳定度信息，跳过稳定度风玫瑰图生成")
            return False
        
        # 为每个高度生成稳定度风玫瑰图
        for height in self.heights:
            print(f"\n生成{height}m高度稳定度风玫瑰图...")
            self.create_stability_wind_roses(height, output_dir)
        
        return True
    
    def generate_analysis_report(self, output_dir):
        """生成分析报告"""
        print("\n生成风向玫瑰图分析报告...")
        
        report_path = os.path.join(output_dir, "昌马站风向玫瑰图分析报告.md")
        
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write("# 昌马站风向玫瑰图分析报告\n\n")
            f.write(f"**分析日期:** {datetime.now().strftime('%Y-%m-%d')}\n\n")
            
            # 数据概况
            f.write("## 1. 数据概况\n\n")
            f.write(f"- **数据时间范围:** {self.data['timestamp'].min().strftime('%Y-%m-%d')} 至 {self.data['timestamp'].max().strftime('%Y-%m-%d')}\n")
            f.write(f"- **数据总量:** {len(self.data)}个时间点\n\n")
            
            # 风向频率分析
            f.write("## 2. 风向频率分析\n\n")
            
            # 分析每个高度的风向频率
            for height in self.heights:
                dir16_col = f'wind_dir_{height}m_16dir'
                
                if dir16_col in self.data.columns:
                    f.write(f"### {height}m高度风向分布\n\n")
                    
                    # 计算风向频率
                    dir_counts = self.data[dir16_col].value_counts()
                    total_valid = dir_counts.sum()
                    
                    if total_valid > 0:
                        # 获取前5个主导风向
                        f.write("**主导风向:**\n\n")
                        f.write("| 风向 | 频率 | 百分比 |\n")
                        f.write("|------|------|--------|\n")
                        
                        for direction, count in dir_counts.head(5).items():
                            percent = count / total_valid * 100
                            f.write(f"| {direction} | {count} | {percent:.1f}% |\n")
                        
                        f.write("\n")
                        
                        # 季节性风向变化
                        f.write("**季节性风向变化:**\n\n")
                        f.write("| 季节 | 主导风向 | 频率占比 |\n")
                        f.write("|------|----------|----------|\n")
                        
                        for season in ['春季', '夏季', '秋季', '冬季']:
                            season_data = self.data[self.data['season'] == season]
                            if len(season_data) > 0:
                                season_dir_counts = season_data[dir16_col].value_counts()
                                if len(season_dir_counts) > 0:
                                    main_dir = season_dir_counts.index[0]
                                    main_percent = season_dir_counts.iloc[0] / len(season_data.dropna(subset=[dir16_col])) * 100
                                    f.write(f"| {season} | {main_dir} | {main_percent:.1f}% |\n")
                                else:
                                    f.write(f"| {season} | 数据不足 | - |\n")
                        
                        f.write("\n")
            
            # 风速特征
            f.write("## 3. 风速特征\n\n")
            
            # 分析每个高度的风速特征
            for height in self.heights:
                speed_col = f'wind_speed_{height}m'
                
                if speed_col in self.data.columns:
                    f.write(f"### {height}m高度风速统计\n\n")
                    
                    # 基本统计量
                    speed_data = self.data[speed_col].dropna()
                    if len(speed_data) > 0:
                        speed_stats = speed_data.describe()
                        
                        f.write("**全年风速统计:**\n\n")
                        f.write(f"- 平均风速: {speed_stats['mean']:.2f} m/s\n")
                        f.write(f"- 中位数风速: {speed_stats['50%']:.2f} m/s\n")
                        f.write(f"- 最大风速: {speed_stats['max']:.2f} m/s\n")
                        f.write(f"- 标准差: {speed_stats['std']:.2f} m/s\n\n")
                        
                        # 季节性风速变化
                        f.write("**季节性风速变化:**\n\n")
                        f.write("| 季节 | 平均风速 (m/s) | 最大风速 (m/s) |\n")
                        f.write("|------|----------------|----------------|\n")
                        
                        for season in ['春季', '夏季', '秋季', '冬季']:
                            season_data = self.data[self.data['season'] == season]
                            if len(season_data) > 0:
                                season_speed = season_data[speed_col].dropna()
                                if len(season_speed) > 0:
                                    season_mean = season_speed.mean()
                                    season_max = season_speed.max()
                                    
                                    if not pd.isna(season_mean) and not pd.isna(season_max):
                                        f.write(f"| {season} | {season_mean:.2f} | {season_max:.2f} |\n")
                                    else:
                                        f.write(f"| {season} | 数据不足 | - |\n")
                                else:
                                    f.write(f"| {season} | 数据不足 | - |\n")
                        
                        f.write("\n")
            
            # 预报误差分析
            f.write("## 4. 预报误差分析\n\n")
            
            # 分析每个高度的预报误差
            for height in self.heights:
                ec_error_col = f'ec_abs_error_{height}m'
                gfs_error_col = f'gfs_abs_error_{height}m'
                
                if ec_error_col in self.data.columns and gfs_error_col in self.data.columns:
                    f.write(f"### {height}m高度预报误差\n\n")
                    
                    # 计算整体误差
                    ec_mae = self.data[ec_error_col].mean()
                    gfs_mae = self.data[gfs_error_col].mean()
                    
                    if not pd.isna(ec_mae) and not pd.isna(gfs_mae):
                        better_model = "EC-WRF" if ec_mae < gfs_mae else "GFS-WRF"
                        improvement = abs(ec_mae - gfs_mae) / max(ec_mae, gfs_mae) * 100
                        
                        f.write("**整体预报性能:**\n\n")
                        f.write(f"- EC-WRF平均绝对误差: {ec_mae:.3f} m/s\n")
                        f.write(f"- GFS-WRF平均绝对误差: {gfs_mae:.3f} m/s\n")
                        f.write(f"- 表现更优的模式: {better_model} (改善 {improvement:.1f}%)\n\n")
                        
                        # 按风向分析误差
                        dir16_col = f'wind_dir_{height}m_16dir'
                        if dir16_col in self.data.columns:
                            # 删除缺失值
                            err_data = self.data.dropna(subset=[dir16_col, ec_error_col, gfs_error_col])
                            
                            if len(err_data) > 0:
                                f.write("**不同风向的预报误差:**\n\n")
                                
                                # 按16方位风向分组计算误差
                                error_by_dir = err_data.groupby(dir16_col)[[ec_error_col, gfs_error_col]].mean().reset_index()
                                error_by_dir['better_model'] = error_by_dir.apply(
                                    lambda x: 'EC-WRF' if x[ec_error_col] < x[gfs_error_col] else 'GFS-WRF', axis=1
                                )
                                error_by_dir['improvement'] = error_by_dir.apply(
                                    lambda x: abs(x[ec_error_col] - x[gfs_error_col]) / max(x[ec_error_col], x[gfs_error_col]) * 100, axis=1
                                )
                                
                                # 显示EC优势风向
                                ec_better_dirs = error_by_dir[error_by_dir['better_model'] == 'EC-WRF'].sort_values('improvement', ascending=False)
                                
                                if len(ec_better_dirs) > 0:
                                    f.write("**EC-WRF表现更优的风向:**\n\n")
                                    f.write("| 风向 | EC-MAE | GFS-MAE | 改善百分比 |\n")
                                    f.write("|------|--------|---------|------------|\n")
                                    
                                    for _, row in ec_better_dirs.head(5).iterrows():
                                        f.write(f"| {row[dir16_col]} | {row[ec_error_col]:.3f} | {row[gfs_error_col]:.3f} | {row['improvement']:.1f}% |\n")
                                
                                # 显示GFS优势风向
                                gfs_better_dirs = error_by_dir[error_by_dir['better_model'] == 'GFS-WRF'].sort_values('improvement', ascending=False)
                                
                                if len(gfs_better_dirs) > 0:
                                    f.write("\n**GFS-WRF表现更优的风向:**\n\n")
                                    f.write("| 风向 | GFS-MAE | EC-MAE | 改善百分比 |\n")
                                    f.write("|------|---------|--------|------------|\n")
                                    
                                    for _, row in gfs_better_dirs.head(5).iterrows():
                                        f.write(f"| {row[dir16_col]} | {row[gfs_error_col]:.3f} | {row[ec_error_col]:.3f} | {row['improvement']:.1f}% |\n")
                                
                                f.write("\n")
                        
                        # 季节性误差变化
                        f.write("**季节性预报误差变化:**\n\n")
                        f.write("| 季节 | EC-MAE | GFS-MAE | 表现更优的模式 |\n")
                        f.write("|------|--------|---------|----------------|\n")
                        
                        for season in ['春季', '夏季', '秋季', '冬季']:
                            season_data = self.data[self.data['season'] == season]
                            if len(season_data) > 0:
                                season_ec = season_data[ec_error_col].dropna()
                                season_gfs = season_data[gfs_error_col].dropna()
                                
                                if len(season_ec) > 0 and len(season_gfs) > 0:
                                    season_ec_mae = season_ec.mean()
                                    season_gfs_mae = season_gfs.mean()
                                    
                                    if not pd.isna(season_ec_mae) and not pd.isna(season_gfs_mae):
                                        season_better = "EC-WRF" if season_ec_mae < season_gfs_mae else "GFS-WRF"
                                        f.write(f"| {season} | {season_ec_mae:.3f} | {season_gfs_mae:.3f} | {season_better} |\n")
                                    else:
                                        f.write(f"| {season} | 数据不足 | - | - |\n")
                                else:
                                    f.write(f"| {season} | 数据不足 | - | - |\n")
                        
                        f.write("\n")
            
            # 稳定度分析
            if 'stability_final' in self.data.columns:
                # 检查稳定度数据是否有效
                stability_counts = self.data['stability_final'].value_counts()
                if len(stability_counts) > 0:
                    f.write("## 5. 稳定度与风向关系分析\n\n")
                    
                    total_valid = stability_counts.sum()
                    
                    f.write("### 稳定度分布\n\n")
                    f.write("| 稳定度条件 | 样本数 | 百分比 |\n")
                    f.write("|------------|--------|--------|\n")
                    
                    stability_names = {'unstable': '不稳定', 'neutral': '中性', 'stable': '稳定', 'unknown': '未知'}
                    
                    for stab, count in stability_counts.items():
                        stab_name = stability_names.get(stab, stab)
                        percent = count / total_valid * 100
                        f.write(f"| {stab_name} | {count} | {percent:.1f}% |\n")
                    
                    f.write("\n")
                    
                    # 稳定度与预报误差关系
                    f.write("### 稳定度条件与预报误差关系\n\n")
                    
                    for height in self.heights:
                        ec_error_col = f'ec_abs_error_{height}m'
                        gfs_error_col = f'gfs_abs_error_{height}m'
                        
                        if ec_error_col in self.data.columns and gfs_error_col in self.data.columns:
                            stability_errors = []
                            
                            # 计算每种稳定度下的误差
                            for stab in ['unstable', 'neutral', 'stable']:
                                if stab in stability_counts.index:
                                    stab_data = self.data[self.data['stability_final'] == stab]
                                    if len(stab_data) > 0:
                                        stab_ec = stab_data[ec_error_col].dropna()
                                        stab_gfs = stab_data[gfs_error_col].dropna()
                                        
                                        if len(stab_ec) > 0 and len(stab_gfs) > 0:
                                            stab_ec_mae = stab_ec.mean()
                                            stab_gfs_mae = stab_gfs.mean()
                                            
                                            if not pd.isna(stab_ec_mae) and not pd.isna(stab_gfs_mae):
                                                stab_name = stability_names.get(stab, stab)
                                                stab_better = "EC-WRF" if stab_ec_mae < stab_gfs_mae else "GFS-WRF"
                                                stability_errors.append({
                                                    'stability': stab,
                                                    'stab_name': stab_name,
                                                    'ec_mae': stab_ec_mae,
                                                    'gfs_mae': stab_gfs_mae,
                                                    'better_model': stab_better,
                                                    'sample_count': len(stab_data)
                                                })
                            
                            if stability_errors:
                                f.write(f"**{height}m高度不同稳定度条件下的预报误差:**\n\n")
                                f.write("| 稳定度条件 | 样本数 | EC-MAE | GFS-MAE | 表现更优的模式 |\n")
                                f.write("|------------|--------|--------|---------|----------------|\n")
                                
                                for item in stability_errors:
                                    f.write(f"| {item['stab_name']} | {item['sample_count']} | {item['ec_mae']:.3f} | {item['gfs_mae']:.3f} | {item['better_model']} |\n")
                                
                                f.write("\n")
            
            # 结论与建议
            f.write("## 6. 结论与建议\n\n")
            
            # 主导风向总结
            f.write("### 主要发现\n\n")
            f.write("**风向特征:**\n")
            
            for height in self.heights:
                dir16_col = f'wind_dir_{height}m_16dir'
                if dir16_col in self.data.columns:
                    dir_counts = self.data[dir16_col].value_counts()
                    if len(dir_counts) > 0:
                        top3_dirs = dir_counts.head(3).index.tolist()
                        if top3_dirs:
                            top3_str = '、'.join(top3_dirs)
                            f.write(f"- {height}m高度的主要风向为{top3_str}\n")
            
            # 预报性能总结
            f.write("\n**预报性能:**\n")
            
            for height in self.heights:
                ec_error_col = f'ec_abs_error_{height}m'
                gfs_error_col = f'gfs_abs_error_{height}m'
                
                if ec_error_col in self.data.columns and gfs_error_col in self.data.columns:
                    ec_mae = self.data[ec_error_col].dropna().mean()
                    gfs_mae = self.data[gfs_error_col].dropna().mean()
                    
                    if not pd.isna(ec_mae) and not pd.isna(gfs_mae):
                        better_model = "EC-WRF" if ec_mae < gfs_mae else "GFS-WRF"
                        improvement = abs(ec_mae - gfs_mae) / max(ec_mae, gfs_mae) * 100
                        
                        f.write(f"- {height}m高度总体上{better_model}模式表现更优，平均改善{improvement:.1f}%\n")
            
            # 风向相关性总结
            for height in self.heights:
                dir16_col = f'wind_dir_{height}m_16dir'
                ec_error_col = f'ec_abs_error_{height}m'
                gfs_error_col = f'gfs_abs_error_{height}m'
                
                if (dir16_col in self.data.columns and 
                    ec_error_col in self.data.columns and 
                    gfs_error_col in self.data.columns):
                    
                    # 删除缺失值
                    err_data = self.data.dropna(subset=[dir16_col, ec_error_col, gfs_error_col])
                    
                    if len(err_data) > 0:
                        error_by_dir = err_data.groupby(dir16_col)[[ec_error_col, gfs_error_col]].mean()
                        ec_better_count = (error_by_dir[ec_error_col] < error_by_dir[gfs_error_col]).sum()
                        gfs_better_count = (error_by_dir[gfs_error_col] < error_by_dir[ec_error_col]).sum()
                        
                        f.write(f"- 在16个风向中，EC-WRF在{ec_better_count}个风向表现更优，GFS-WRF在{gfs_better_count}个风向表现更优\n")
            
            # 稳定度总结
            if 'stability_final' in self.data.columns:
                stability_counts = self.data['stability_final'].value_counts()
                if len(stability_counts) > 0:
                    f.write("\n**稳定度分析:**\n")
                    
                    most_common_stab = stability_counts.index[0]
                    most_common_stab_name = {'unstable': '不稳定', 'neutral': '中性', 'stable': '稳定'}.get(most_common_stab, most_common_stab)
                    most_common_pct = stability_counts.iloc[0] / stability_counts.sum() * 100
                    
                    f.write(f"- 昌马站最常见的稳定度条件为{most_common_stab_name}，占比{most_common_pct:.1f}%\n")
                    
                    # 稳定度与预报性能
                    for height in self.heights:
                        ec_error_col = f'ec_abs_error_{height}m'
                        gfs_error_col = f'gfs_abs_error_{height}m'
                        
                        if ec_error_col in self.data.columns and gfs_error_col in self.data.columns:
                            stability_errors = []
                            
                            for stab in ['unstable', 'neutral', 'stable']:
                                if stab in stability_counts.index:
                                    stab_data = self.data[self.data['stability_final'] == stab]
                                    if len(stab_data) > 0:
                                        stab_ec = stab_data[ec_error_col].dropna()
                                        stab_gfs = stab_data[gfs_error_col].dropna()
                                        
                                        if len(stab_ec) > 0 and len(stab_gfs) > 0:
                                            stab_ec_mae = stab_ec.mean()
                                            stab_gfs_mae = stab_gfs.mean()
                                            
                                            if not pd.isna(stab_ec_mae) and not pd.isna(stab_gfs_mae):
                                                stability_errors.append({
                                                    'stability': stab,
                                                    'stab_name': {'unstable': '不稳定', 'neutral': '中性', 'stable': '稳定'}.get(stab, stab),
                                                    'ec_mae': stab_ec_mae,
                                                    'gfs_mae': stab_gfs_mae,
                                                    'better_model': "EC-WRF" if stab_ec_mae < stab_gfs_mae else "GFS-WRF"
                                                })
                            
                            if stability_errors:
                                # 找出EC和GFS各自在哪种稳定度条件下表现最好
                                ec_best_stab = min(stability_errors, key=lambda x: x['ec_mae'])
                                gfs_best_stab = min(stability_errors, key=lambda x: x['gfs_mae'])
                                
                                f.write(f"- {height}m高度EC-WRF模式在{ec_best_stab['stab_name']}条件下表现最佳\n")
                                f.write(f"- {height}m高度GFS-WRF模式在{gfs_best_stab['stab_name']}条件下表现最佳\n")
            
            # 建议
            f.write("\n### 预报建议\n\n")
            
            # 根据分析结果生成建议
            for height in self.heights:
                ec_error_col = f'ec_abs_error_{height}m'
                gfs_error_col = f'gfs_abs_error_{height}m'
                
                if ec_error_col in self.data.columns and gfs_error_col in self.data.columns:
                    ec_mae = self.data[ec_error_col].dropna().mean()
                    gfs_mae = self.data[gfs_error_col].dropna().mean()
                    
                    if not pd.isna(ec_mae) and not pd.isna(gfs_mae):
                        f.write(f"**{height}m高度风速预报建议:**\n\n")
                        
                        # 总体模式选择
                        if ec_mae < gfs_mae:
                            f.write("1. 总体上建议优先使用EC-WRF模式进行风速预报\n")
                        else:
                            f.write("1. 总体上建议优先使用GFS-WRF模式进行风速预报\n")
                        
                        # 按风向选择
                        dir16_col = f'wind_dir_{height}m_16dir'
                        if dir16_col in self.data.columns:
                            # 删除缺失值
                            err_data = self.data.dropna(subset=[dir16_col, ec_error_col, gfs_error_col])
                            
                            if len(err_data) > 0:
                                error_by_dir = err_data.groupby(dir16_col)[[ec_error_col, gfs_error_col]].mean().reset_index()
                                
                                if ec_mae < gfs_mae:
                                    # EC总体更好，找出GFS优势的风向
                                    gfs_better_dirs = error_by_dir[error_by_dir[gfs_error_col] < error_by_dir[ec_error_col]][dir16_col].tolist()
                                    
                                    if gfs_better_dirs:
                                        gfs_better_str = '、'.join(gfs_better_dirs[:3])
                                        f.write(f"2. 当风向为{gfs_better_str}时，建议切换到GFS-WRF模式\n")
                                else:
                                    # GFS总体更好，找出EC优势的风向
                                    ec_better_dirs = error_by_dir[error_by_dir[ec_error_col] < error_by_dir[gfs_error_col]][dir16_col].tolist()
                                    
                                    if ec_better_dirs:
                                        ec_better_str = '、'.join(ec_better_dirs[:3])
                                        f.write(f"2. 当风向为{ec_better_str}时，建议切换到EC-WRF模式\n")
                        
                        # 稳定度相关建议
                        if 'stability_final' in self.data.columns:
                            stability_errors = []
                            
                            for stab in ['unstable', 'neutral', 'stable']:
                                stab_data = self.data[self.data['stability_final'] == stab]
                                if len(stab_data) > 0:
                                    stab_ec = stab_data[ec_error_col].dropna()
                                    stab_gfs = stab_data[gfs_error_col].dropna()
                                    
                                    if len(stab_ec) > 0 and len(stab_gfs) > 0:
                                        stab_ec_mae = stab_ec.mean()
                                        stab_gfs_mae = stab_gfs.mean()
                                        
                                        if not pd.isna(stab_ec_mae) and not pd.isna(stab_gfs_mae):
                                            stab_better = "EC-WRF" if stab_ec_mae < stab_gfs_mae else "GFS-WRF"
                                            stab_name = {'unstable': '不稳定', 'neutral': '中性', 'stable': '稳定'}.get(stab, stab)
                                            
                                            stability_errors.append({
                                                'stability': stab,
                                                'stab_name': stab_name,
                                                'better_model': stab_better,
                                                'improvement': abs(stab_ec_mae - stab_gfs_mae) / max(stab_ec_mae, stab_gfs_mae) * 100
                                            })
                            
                            if stability_errors:
                                # 按改善百分比排序
                                stability_errors.sort(key=lambda x: x['improvement'], reverse=True)
                                
                                f.write(f"\n3. 稳定度条件考虑：\n")
                                
                                for item in stability_errors:
                                    f.write(f"   - {item['stab_name']}条件下建议使用{item['better_model']}模式 (改善{item['improvement']:.1f}%)\n")
                        
                        f.write("\n")
            
            # 总结
            f.write("\n### 总结\n\n")
            f.write("基于昌马站风向玫瑰图分析，可以根据风向和稳定度条件灵活选择不同的预报模式，提高风速预报准确性。")
            f.write("本分析提供的风向玫瑰图可作为昌马站风电场风资源特性的参考依据。\n\n")
            
            # 报告结束
            f.write("---\n")
            f.write("\n*报告生成时间：" + datetime.now().strftime('%Y-%m-%d %H:%M:%S') + "*")
        
        print(f"分析报告已生成：{report_path}")
        return report_path


def main():
    """主函数"""
    # 设置输入输出路径
    data_path = "/Users/xiaxin/work/WindForecast_Project/01_Data/processed/matched_data/changma_matched.csv"
    output_dir = "/Users/xiaxin/work/WindForecast_Project/03_Results/wind_rose_analysis"
    
    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)
    
    try:
        print("=" * 60)
        print("昌马站风向玫瑰图分析系统 - 修复版")
        print("=" * 60)
        
        # 初始化分析器
        analyzer = WindRoseAnalyzer()
        
        # 加载并预处理数据
        print("\n步骤1: 加载并预处理数据")
        data = analyzer.load_and_preprocess(data_path)
        
        # 创建子目录
        frequency_dir = os.path.join(output_dir, "风向玫瑰图")
        comparison_dir = os.path.join(output_dir, "预报对比图")
        stability_dir = os.path.join(output_dir, "稳定度风玫瑰图")
        
        for d in [frequency_dir, comparison_dir, stability_dir]:
            os.makedirs(d, exist_ok=True)
        
        # 生成频率风向玫瑰图
        print("\n步骤2: 生成风向频率玫瑰图")
        analyzer.generate_seasonal_wind_roses(frequency_dir)
        
        # 生成EC vs GFS对比风玫瑰图
        print("\n步骤3: 生成EC vs GFS对比风玫瑰图")
        analyzer.generate_comparison_roses(comparison_dir)
        
        # 生成稳定度风玫瑰图
        print("\n步骤4: 生成稳定度风玫瑰图")
        analyzer.generate_stability_roses(stability_dir)
        
        # 生成分析报告
        print("\n步骤5: 生成分析报告")
        report_path = analyzer.generate_analysis_report(output_dir)
        
        print("\n" + "=" * 60)
        print(f"分析完成！所有结果已保存到：{output_dir}")
        print(f"分析报告：{report_path}")
        print("=" * 60)
        
    except Exception as e:
        print(f"分析过程中出现错误: {str(e)}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()