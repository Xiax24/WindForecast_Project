#!/usr/bin/env python3
"""
昌马站季节和昼夜模式分析
通过数据驱动方法确定更合理的季节划分和昼夜分割
用于优化风电场预报策略

路径: 02_Code/analysis/seasonal_pattern_analysis.py
日期: 2025-05-24
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import os
from datetime import datetime, timedelta
import ephem
import warnings
warnings.filterwarnings('ignore')

# 设置matplotlib样式
plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams['font.sans-serif'] = ['Arial Unicode MS', 'SimHei', 'DejaVu Sans']  # 尝试多种字体
plt.rcParams['axes.unicode_minus'] = False  # 用于正确显示负号

class SeasonalPatternAnalyzer:
    """昌马站季节和昼夜模式分析器"""
    
    def __init__(self, latitude=40.2053, longitude=96.8114):
        """
        初始化分析器
        
        Args:
            latitude: 昌马站纬度（默认为实际坐标40.2053°N）
            longitude: 昌马站经度（默认为实际坐标96.8114°E）
        """
        self.latitude = latitude
        self.longitude = longitude
        self.data = None
        self.traditional_seasons = {
            1: '冬季', 2: '冬季', 3: '春季',
            4: '春季', 5: '春季', 6: '夏季',
            7: '夏季', 8: '夏季', 9: '秋季',
            10: '秋季', 11: '秋季', 12: '冬季'
        }
        
        # 标准昼夜分界时间（传统方法）
        self.standard_daytime_start = 6  # 6:00 AM
        self.standard_daytime_end = 18   # 6:00 PM
    
    def load_data(self, file_path):
        """
        加载数据
        
        Args:
            file_path: 数据文件路径
        """
        print(f"加载数据: {file_path}")
        data = pd.read_csv(file_path)
        
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
        data['day'] = data['timestamp'].dt.day
        data['month'] = data['timestamp'].dt.month
        data['year'] = data['timestamp'].dt.year
        data['dayofyear'] = data['timestamp'].dt.dayofyear
        data['season'] = data['month'].map(self.traditional_seasons)
        
        # 添加传统昼夜标记
        data['traditional_daytime'] = ((data['hour'] >= self.standard_daytime_start) & 
                                      (data['hour'] < self.standard_daytime_end))
        
        self.data = data
        
        print(f"数据形状: {data.shape}")
        print(f"时间范围: {data['timestamp'].min()} 到 {data['timestamp'].max()}")
        
        return data
    
    def calculate_sunrise_sunset(self, date):
        """
        计算特定日期的日出日落时间
        
        Args:
            date: 日期对象
            
        Returns:
            tuple: (日出时间, 日落时间)
        """
        observer = ephem.Observer()
        observer.lat = str(self.latitude)
        observer.lon = str(self.longitude)
        observer.date = date.strftime('%Y/%m/%d')
        
        sun = ephem.Sun()
        sunrise = observer.next_rising(sun).datetime().replace(tzinfo=None)
        sunset = observer.next_setting(sun).datetime().replace(tzinfo=None)
        
        return sunrise, sunset
    
    def add_astronomical_daytime(self):
        """
        添加基于天文计算的昼夜分割
        """
        if self.data is None:
            print("错误: 请先加载数据")
            return
        
        print("计算天文日出日落时间...")
        
        # 获取数据中的所有日期
        unique_dates = self.data['timestamp'].dt.date.unique()
        
        # 计算每个日期的日出日落时间
        sunrise_sunset_dict = {}
        for date in unique_dates:
            sunrise, sunset = self.calculate_sunrise_sunset(date)
            sunrise_sunset_dict[date] = (sunrise, sunset)
        
        # 为每个数据点添加标记
        def is_daytime_astronomical(row):
            date = row['timestamp'].date()
            sunrise, sunset = sunrise_sunset_dict[date]
            
            # 考虑过渡期：日出后1小时到日落前1小时为完全白天
            daytime_start = sunrise + timedelta(hours=1)
            daytime_end = sunset - timedelta(hours=1)
            
            return (row['timestamp'] >= daytime_start) and (row['timestamp'] <= daytime_end)
        
        # 应用函数
        self.data['astronomical_daytime'] = self.data.apply(is_daytime_astronomical, axis=1)
        
        # 计算传统方法与天文方法的差异
        diff_count = (self.data['traditional_daytime'] != self.data['astronomical_daytime']).sum()
        diff_percent = diff_count / len(self.data) * 100
        
        print(f"传统昼夜划分与天文昼夜划分的差异: {diff_count}个数据点 ({diff_percent:.2f}%)")
        
        return self.data
    
    def analyze_temperature_patterns(self):
        """
        分析温度变化模式，用于识别昼夜和季节转换
        """
        if self.data is None:
            print("错误: 请先加载数据")
            return
        
        # 检查温度列是否存在
        temp_cols = [col for col in self.data.columns if 'temp' in col.lower() and 'obs' in col.lower()]
        
        if not temp_cols:
            print("警告: 未找到观测温度列")
            return
        
        temp_col = temp_cols[0]
        print(f"使用温度列: {temp_col}")
        
        # 计算温度变化率
        self.data['temp_change'] = self.data[temp_col].diff()
        
        # 按月份和小时分组计算平均温度变化率
        monthly_hourly_temp_change = self.data.groupby(['month', 'hour'])['temp_change'].mean().unstack()
        
        # 找出每个月温度上升最快和下降最快的小时（近似日出和日落）
        temp_pattern = {}
        for month in range(1, 13):
            if month in monthly_hourly_temp_change.index:
                temp_changes = monthly_hourly_temp_change.loc[month]
                
                # 找出温度上升最快的小时（近似日出）
                sunrise_hour = temp_changes[temp_changes > 0].idxmax() if any(temp_changes > 0) else None
                
                # 找出温度下降最快的小时（近似日落）
                sunset_hour = temp_changes[temp_changes < 0].idxmin() if any(temp_changes < 0) else None
                
                temp_pattern[month] = {'sunrise_approx': sunrise_hour, 'sunset_approx': sunset_hour}
        
        return temp_pattern
    
    def cluster_seasonal_patterns(self, n_clusters=4):
        """
        使用聚类方法识别自然季节
        
        Args:
            n_clusters: 聚类数量（默认为4个季节）
            
        Returns:
            dict: 月份到季节的映射
        """
        if self.data is None:
            print("错误: 请先加载数据")
            return
        
        print(f"使用聚类方法识别{n_clusters}个自然季节...")
        
        # 准备特征：月平均风速、温度、稳定度分布
        features = []
        feature_names = []
        
        # 风速特征（各高度）
        for height in [10, 30, 50, 70]:
            wind_col = f'obs_wind_speed_{height}m'
            if wind_col in self.data.columns:
                monthly_wind = self.data.groupby('month')[wind_col].mean()
                features.append(monthly_wind)
                feature_names.append(f'wind_{height}m')
        
        # 温度特征
        temp_cols = [col for col in self.data.columns if 'temp' in col.lower() and 'obs' in col.lower()]
        if temp_cols:
            temp_col = temp_cols[0]
            monthly_temp = self.data.groupby('month')[temp_col].mean()
            features.append(monthly_temp)
            feature_names.append('temperature')
            
            # 温度范围
            monthly_temp_range = self.data.groupby('month')[temp_col].agg(lambda x: x.max() - x.min())
            features.append(monthly_temp_range)
            feature_names.append('temp_range')
        
        # 稳定度特征
        if 'stability_final' in self.data.columns:
            # 计算每个月的稳定度分布
            stability_distribution = pd.crosstab(
                self.data['month'], 
                self.data['stability_final'], 
                normalize='index'
            )
            
            for stab in stability_distribution.columns:
                features.append(stability_distribution[stab])
                feature_names.append(f'stability_{stab}')
        
        # 合并特征
        if not features:
            print("错误: 未找到足够的特征用于季节聚类")
            return
        
        monthly_features = pd.concat(features, axis=1)
        monthly_features.columns = feature_names
        
        # 标准化特征
        scaler = StandardScaler()
        scaled_features = scaler.fit_transform(monthly_features)
        
        # 应用PCA降维（可选）
        if len(feature_names) > 2:
            pca = PCA(n_components=min(len(feature_names), 5))
            pca_features = pca.fit_transform(scaled_features)
            explained_var = pca.explained_variance_ratio_
            print(f"PCA解释的方差比例: {explained_var}")
            print(f"累计解释方差: {np.sum(explained_var):.2f}")
            
            # 使用PCA结果进行聚类
            kmeans = KMeans(n_clusters=n_clusters, random_state=42)
            clusters = kmeans.fit_predict(pca_features)
        else:
            # 直接使用标准化特征进行聚类
            kmeans = KMeans(n_clusters=n_clusters, random_state=42)
            clusters = kmeans.fit_predict(scaled_features)
        
        # 创建月份到聚类的映射
        month_to_cluster = {month: cluster for month, cluster in zip(monthly_features.index, clusters)}
        
        # 将聚类结果解释为季节
        # 这需要进一步分析和命名
        cluster_seasons = self.interpret_clusters(monthly_features, clusters)
        
        # 创建月份到季节的映射
        month_to_season = {month: cluster_seasons[cluster] for month, cluster in month_to_cluster.items()}
        
        return month_to_season, monthly_features, clusters
    
    def interpret_clusters(self, features, clusters):
        """
        解释聚类结果，将聚类与季节名称对应
        
        Args:
            features: 月份特征
            clusters: 聚类结果
            
        Returns:
            dict: 聚类ID到季节名称的映射
        """
        # 分析每个聚类的特征
        cluster_features = {}
        for cluster in np.unique(clusters):
            cluster_months = features.index[clusters == cluster]
            cluster_features[cluster] = {
                'months': cluster_months.tolist(),
                'avg_features': features.loc[cluster_months].mean(),
                'n_months': len(cluster_months)
            }
        
        # 温度特征通常是季节最明显的标志
        if 'temperature' in features.columns:
            # 按温度排序聚类
            clusters_by_temp = sorted(cluster_features.keys(), 
                                    key=lambda c: cluster_features[c]['avg_features']['temperature'])
            
            # 根据温度确定季节
            if len(clusters_by_temp) == 4:
                # 4个季节的情况
                season_mapping = {
                    clusters_by_temp[0]: '冬季',  # 最冷
                    clusters_by_temp[1]: '春季',  # 第二冷
                    clusters_by_temp[2]: '秋季',  # 第二热
                    clusters_by_temp[3]: '夏季',  # 最热
                }
            elif len(clusters_by_temp) == 3:
                # 3个季节的情况
                season_mapping = {
                    clusters_by_temp[0]: '冬季',  # 最冷
                    clusters_by_temp[1]: '春秋季',  # 中间温度
                    clusters_by_temp[2]: '夏季',  # 最热
                }
            elif len(clusters_by_temp) == 2:
                # 2个季节的情况
                season_mapping = {
                    clusters_by_temp[0]: '冷季',  # 冷
                    clusters_by_temp[1]: '热季',  # 热
                }
            else:
                # 其他情况，简单编号
                season_mapping = {cluster: f'季节_{i+1}' for i, cluster in enumerate(cluster_features.keys())}
        else:
            # 没有温度特征时，简单编号
            season_mapping = {cluster: f'季节_{i+1}' for i, cluster in enumerate(cluster_features.keys())}
        
        return season_mapping
    
    def analyze_diurnal_patterns(self):
        """
        分析昼夜变化模式
        """
        if self.data is None:
            print("错误: 请先加载数据")
            return
        
        # 检查风速列是否存在
        wind_cols = [col for col in self.data.columns if 'wind_speed' in col.lower() and 'obs' in col.lower()]
        
        if not wind_cols:
            print("警告: 未找到观测风速列")
            return
        
        # 使用70m风速（通常是轮毂高度）
        wind_col = next((col for col in wind_cols if '70m' in col), wind_cols[0])
        print(f"使用风速列: {wind_col}")
        
        # 按小时和季节分组计算平均风速
        hourly_seasonal_wind = self.data.groupby(['season', 'hour'])[wind_col].mean().unstack()
        
        # 检查是否有稳定度数据
        if 'stability_final' in self.data.columns:
            # 按小时和季节分组计算稳定度分布
            stability_counts = pd.crosstab(
                [self.data['season'], self.data['hour']], 
                self.data['stability_final'], 
                normalize='index'
            )
            
            return hourly_seasonal_wind, stability_counts
        
        return hourly_seasonal_wind, None
    
    def plot_seasonal_clusters(self, monthly_features, clusters):
        """
        绘制季节聚类结果
        
        Args:
            monthly_features: 月份特征
            clusters: 聚类结果
        """
        if 'temperature' not in monthly_features.columns:
            print("警告: 无法绘制季节聚类图，缺少温度特征")
            return
        
        # 选择两个主要特征进行可视化
        feature1 = 'temperature'
        
        # 选择第二个特征（风速或稳定度）
        wind_features = [col for col in monthly_features.columns if 'wind' in col]
        feature2 = wind_features[0] if wind_features else monthly_features.columns[1]
        
        plt.figure(figsize=(10, 8))
        
        # 绘制散点图
        scatter = plt.scatter(
            monthly_features[feature1], 
            monthly_features[feature2], 
            c=clusters, 
            cmap='viridis', 
            s=100, 
            alpha=0.8
        )
        
        # 添加月份标签
        for i, month in enumerate(monthly_features.index):
            plt.text(
                monthly_features[feature1].iloc[i] + 0.1, 
                monthly_features[feature2].iloc[i], 
                f"{month}月", 
                fontsize=12
            )
        
        plt.colorbar(scatter, label='季节聚类')
        plt.xlabel(f'{feature1}', fontsize=14)
        plt.ylabel(f'{feature2}', fontsize=14)
        plt.title('基于气象特征的月份聚类分析', fontsize=16)
        plt.grid(True, alpha=0.3)
        
        # 保存图片
        output_dir = '/Users/xiaxin/work/WindForecast_Project/03_Results/seasonal_analysis'
        os.makedirs(output_dir, exist_ok=True)
        plt.savefig(os.path.join(output_dir, 'seasonal_clusters.png'), dpi=300, bbox_inches='tight')
        plt.close()
    
    def plot_diurnal_patterns(self, hourly_seasonal_wind, stability_counts=None):
        """
        绘制昼夜变化模式
        
        Args:
            hourly_seasonal_wind: 按小时和季节分组的风速
            stability_counts: 按小时和季节分组的稳定度分布
        """
        # 创建输出目录
        output_dir = '/Users/xiaxin/work/WindForecast_Project/03_Results/seasonal_analysis'
        os.makedirs(output_dir, exist_ok=True)
        
        # 绘制风速日变化
        plt.figure(figsize=(12, 8))
        
        for season in hourly_seasonal_wind.index.unique():
            plt.plot(
                hourly_seasonal_wind.loc[season].index, 
                hourly_seasonal_wind.loc[season].values, 
                marker='o', 
                linewidth=2, 
                label=season
            )
        
        plt.xlabel('小时', fontsize=14)
        plt.ylabel('平均风速 (m/s)', fontsize=14)
        plt.title('不同季节的风速日变化模式', fontsize=16)
        plt.xticks(range(0, 24))
        plt.grid(True, alpha=0.3)
        plt.legend(fontsize=12)
        
        # 添加传统昼夜分界线
        plt.axvline(x=self.standard_daytime_start, color='gray', linestyle='--', alpha=0.7, 
                   label='传统昼夜分界')
        plt.axvline(x=self.standard_daytime_end, color='gray', linestyle='--', alpha=0.7)
        
        # 保存图片
        plt.savefig(os.path.join(output_dir, 'diurnal_wind_patterns.png'), dpi=300, bbox_inches='tight')
        plt.close()
        
        # 如果有稳定度数据，绘制稳定度分布
        if stability_counts is not None:
            # 为每个季节绘制一个子图
            seasons = stability_counts.index.get_level_values(0).unique()
            fig, axes = plt.subplots(len(seasons), 1, figsize=(12, 4*len(seasons)), sharex=True)
            
            for i, season in enumerate(seasons):
                ax = axes[i] if len(seasons) > 1 else axes
                
                # 选择当前季节的数据
                season_data = stability_counts.xs(season, level=0)
                
                # 堆叠条形图
                bottom = np.zeros(len(season_data))
                
                for stab in ['unstable', 'neutral', 'stable']:
                    if stab in season_data.columns:
                        ax.bar(
                            season_data.index, 
                            season_data[stab], 
                            bottom=bottom, 
                            label=stab,
                            alpha=0.7
                        )
                        bottom += season_data[stab]
                
                ax.set_title(f'{season}的稳定度日变化', fontsize=14)
                ax.set_ylabel('比例', fontsize=12)
                ax.grid(True, alpha=0.3)
                
                if i == 0:
                    ax.legend(fontsize=12)
                
                # 添加传统昼夜分界线
                ax.axvline(x=self.standard_daytime_start, color='black', linestyle='--', alpha=0.7)
                ax.axvline(x=self.standard_daytime_end, color='black', linestyle='--', alpha=0.7)
            
            plt.xlabel('小时', fontsize=14)
            plt.xticks(range(0, 24))
            plt.tight_layout()
            
            # 保存图片
            plt.savefig(os.path.join(output_dir, 'diurnal_stability_patterns.png'), dpi=300, bbox_inches='tight')
            plt.close()
    
    def compare_forecast_performance(self, custom_seasons=None):
        """
        比较不同季节划分下的预报性能
        
        Args:
            custom_seasons: 自定义的月份到季节的映射
        """
        if self.data is None:
            print("错误: 请先加载数据")
            return None, None
        
        # 检查预报误差列是否存在
        ec_error_cols = [col for col in self.data.columns if 'ec' in col.lower() and 'error' in col.lower()]
        gfs_error_cols = [col for col in self.data.columns if 'gfs' in col.lower() and 'error' in col.lower()]
        
        # 如果找不到带有'error'的列，尝试查找以'ec'和'gfs'开头的风速列，然后计算误差
        if not ec_error_cols or not gfs_error_cols:
            print("未找到现成的预报误差列，尝试从风速列计算误差...")
            
            # 查找观测风速列
            obs_wind_cols = [col for col in self.data.columns if 'obs_wind_speed' in col.lower()]
            ec_wind_cols = [col for col in self.data.columns if 'ec_wind_speed' in col.lower()]
            gfs_wind_cols = [col for col in self.data.columns if 'gfs_wind_speed' in col.lower()]
            
            if obs_wind_cols and ec_wind_cols and gfs_wind_cols:
                # 优先使用70m高度的数据
                obs_col = next((col for col in obs_wind_cols if '70m' in col), obs_wind_cols[0])
                ec_col = next((col for col in ec_wind_cols if '70m' in col), ec_wind_cols[0])
                gfs_col = next((col for col in gfs_wind_cols if '70m' in col), gfs_wind_cols[0])
                
                print(f"使用以下列计算误差:")
                print(f"  观测风速: {obs_col}")
                print(f"  EC风速: {ec_col}")
                print(f"  GFS风速: {gfs_col}")
                
                # 计算误差
                self.data['ec_error_calculated'] = self.data[ec_col] - self.data[obs_col]
                self.data['gfs_error_calculated'] = self.data[gfs_col] - self.data[obs_col]
                
                # 使用计算出的误差列
                ec_error_col = 'ec_error_calculated'
                gfs_error_col = 'gfs_error_calculated'
            else:
                print("警告: 无法找到或计算预报误差")
                return None, None
        else:
            # 使用70m高度的误差（如果存在）
            ec_error_col = next((col for col in ec_error_cols if '70m' in col), ec_error_cols[0])
            gfs_error_col = next((col for col in gfs_error_cols if '70m' in col), gfs_error_cols[0])
            
            print(f"使用EC误差列: {ec_error_col}")
            print(f"使用GFS误差列: {gfs_error_col}")
        
        # 如果提供了自定义季节，添加到数据中
        if custom_seasons:
            self.data['custom_season'] = self.data['month'].map(custom_seasons)
        
        # 计算传统季节划分的误差
        traditional_performance = self.data.groupby('season').apply(
            lambda x: pd.Series({
                'ec_mae': np.abs(x[ec_error_col]).mean(),
                'gfs_mae': np.abs(x[gfs_error_col]).mean(),
                'better_model': 'EC' if np.abs(x[ec_error_col]).mean() < np.abs(x[gfs_error_col]).mean() else 'GFS',
                'sample_count': len(x)
            })
        )
        
        # 如果有自定义季节，计算其误差
        if custom_seasons:
            custom_performance = self.data.groupby('custom_season').apply(
                lambda x: pd.Series({
                    'ec_mae': np.abs(x[ec_error_col]).mean(),
                    'gfs_mae': np.abs(x[gfs_error_col]).mean(),
                    'better_model': 'EC' if np.abs(x[ec_error_col]).mean() < np.abs(x[gfs_error_col]).mean() else 'GFS',
                    'sample_count': len(x)
                })
            )
            
            return traditional_performance, custom_performance
        
        return traditional_performance, None
    
    def generate_report(self, temp_pattern, month_to_season, hourly_seasonal_wind, 
                       stability_counts, traditional_performance, custom_performance=None):
        """
        生成分析报告
        
        Args:
            temp_pattern: 温度变化模式
            month_to_season: 月份到季节的映射
            hourly_seasonal_wind: 按小时和季节分组的风速
            stability_counts: 按小时和季节分组的稳定度分布
            traditional_performance: 传统季节划分的预报性能
            custom_performance: 自定义季节划分的预报性能
        """
        # 创建输出目录
        report_dir = '/Users/xiaxin/work/WindForecast_Project/03_Results/seasonal_analysis'
        os.makedirs(report_dir, exist_ok=True)
        
        report_path = os.path.join(report_dir, '昌马站季节和昼夜模式分析报告.md')
        
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write("# 昌马站季节和昼夜模式分析报告\n\n")
            f.write(f"**分析日期:** {datetime.now().strftime('%Y-%m-%d')}\n\n")
            
            # 数据概况
            f.write("## 1. 数据概况\n\n")
            f.write(f"- **数据时间范围:** {self.data['timestamp'].min().strftime('%Y-%m-%d')} 至 {self.data['timestamp'].max().strftime('%Y-%m-%d')}\n")
            f.write(f"- **数据总量:** {len(self.data)}个时间点\n")
            f.write(f"- **地理位置:** 纬度 {self.latitude}°N, 经度 {self.longitude}°E\n\n")
            
            # 昼夜分割分析
            f.write("## 2. 昼夜分割分析\n\n")
            
            # 基于天文计算的昼夜分割
            f.write("### 2.1 天文日出日落时间\n\n")
            
            # 计算每月15日的日出日落时间
            f.write("**月度平均日出日落时间:**\n\n")
            f.write("| 月份 | 日出时间 | 日落时间 | 白天时长 |\n")
            f.write("|------|----------|----------|----------|\n")
            
            for month in range(1, 13):
                # 计算本年度每月15日的日出日落时间
                date = datetime(2022, month, 15)
                sunrise, sunset = self.calculate_sunrise_sunset(date)
                
                daylight_hours = (sunset - sunrise).total_seconds() / 3600
                
                f.write(f"| {month}月 | {sunrise.strftime('%H:%M')} | {sunset.strftime('%H:%M')} | {daylight_hours:.1f}小时 |\n")
            
            f.write("\n")
            
            # 基于温度变化的昼夜分割
            if temp_pattern:
                f.write("### 2.2 基于温度变化的昼夜转换时间\n\n")
                
                f.write("**月度温度变化特征:**\n\n")
                f.write("| 月份 | 温度上升最快时间 | 温度下降最快时间 |\n")
                f.write("|------|------------------|------------------|\n")
                
                for month, pattern in temp_pattern.items():
                    sunrise_hour = pattern['sunrise_approx'] if pattern['sunrise_approx'] is not None else '-'
                    sunset_hour = pattern['sunset_approx'] if pattern['sunset_approx'] is not None else '-'
                    
                    f.write(f"| {month}月 | {sunrise_hour}:00 | {sunset_hour}:00 |\n")
                
                f.write("\n")
            
            # 传统与优化昼夜分割的比较
            f.write("### 2.3 昼夜分割方法比较\n\n")
            
            f.write("**传统方法:**\n")
            f.write(f"- 固定时间划分: 白天 {self.standard_daytime_start}:00-{self.standard_daytime_end}:00, 夜间 {self.standard_daytime_end}:00-{self.standard_daytime_start}:00\n")
            f.write("- 优点: 简单直观，易于实现\n")
            f.write("- 缺点: 未考虑季节变化和地理位置影响\n\n")
            
            f.write("**基于天文计算的方法:**\n")
            f.write("- 根据地理位置计算实际日出日落时间\n")
            f.write("- 考虑过渡期：日出后1小时至日落前1小时为完全白天\n")
            f.write("- 优点: 科学准确，考虑季节和地理位置影响\n")
            f.write("- 缺点: 实现较复杂\n\n")
            
            # 季节划分分析
            f.write("## 3. 季节划分分析\n\n")
            
            # 传统季节划分
            f.write("### 3.1 传统季节划分\n\n")
            f.write("传统的季节划分方法按日历月份划分：\n\n")
            f.write("- **春季**: 3月-5月\n")
            f.write("- **夏季**: 6月-8月\n")
            f.write("- **秋季**: 9月-11月\n")
            f.write("- **冬季**: 12月-2月\n\n")
            
            # 基于聚类的季节划分
            f.write("### 3.2 基于气象特征的季节划分\n\n")
            
            if month_to_season:
                # 重新组织数据以便于展示
                month_seasons = [(month, season) for month, season in month_to_season.items()]
                month_seasons.sort()  # 按月份排序
                
                f.write("**基于聚类分析的季节划分:**\n\n")
                f.write("| 月份 | 传统季节 | 聚类季节 |\n")
                f.write("|------|----------|----------|\n")
                
                for month, season in month_seasons:
                    traditional_season = self.traditional_seasons.get(month, '未知')
                    f.write(f"| {month}月 | {traditional_season} | {season} |\n")
                
                # 提取特征季节
                seasons_by_month = {}
                for season in set(month_to_season.values()):
                    months = [month for month, s in month_to_season.items() if s == season]
                    months.sort()  # 按月份排序
                    seasons_by_month[season] = months
                
                f.write("\n**聚类季节划分结果:**\n\n")
                for season, months in seasons_by_month.items():
                    months_str = ", ".join([f"{m}月" for m in months])
                    f.write(f"- **{season}**: {months_str}\n")
                
                f.write("\n")
            
            # 风速日变化模式
            f.write("## 4. 风速日变化模式\n\n")
            
            if hourly_seasonal_wind is not None:
                f.write("### 4.1 不同季节的风速日变化模式\n\n")
                
                # 提取每个季节的风速日变化特征
                for season in hourly_seasonal_wind.index.unique():
                    wind_pattern = hourly_seasonal_wind.loc[season]
                    max_hour = wind_pattern.idxmax()
                    min_hour = wind_pattern.idxmin()
                    max_wind = wind_pattern.max()
                    min_wind = wind_pattern.min()
                    avg_wind = wind_pattern.mean()
                    
                    f.write(f"**{season}风速特征:**\n\n")
                    f.write(f"- 平均风速: {avg_wind:.2f} m/s\n")
                    f.write(f"- 最大风速: {max_wind:.2f} m/s (出现在{max_hour}:00)\n")
                    f.write(f"- 最小风速: {min_wind:.2f} m/s (出现在{min_hour}:00)\n")
                    f.write(f"- 日内风速变化: {max_wind - min_wind:.2f} m/s\n\n")
            
            # 稳定度日变化模式
            if stability_counts is not None:
                f.write("### 4.2 不同季节的稳定度日变化模式\n\n")
                
                # 计算每个季节白天和夜间的稳定度分布
                seasons = stability_counts.index.get_level_values(0).unique()
                
                for season in seasons:
                    f.write(f"**{season}稳定度特征:**\n\n")
                    
                    # 提取当前季节的数据
                    season_data = stability_counts.xs(season, level=0)
                    
                    # 计算白天和夜间的稳定度分布
                    daytime_hours = range(self.standard_daytime_start, self.standard_daytime_end)
                    nighttime_hours = list(range(0, self.standard_daytime_start)) + list(range(self.standard_daytime_end, 24))
                    
                    daytime_stability = season_data.loc[daytime_hours].mean()
                    nighttime_stability = season_data.loc[nighttime_hours].mean()
                    
                    f.write("| 时段 | 不稳定比例 | 中性比例 | 稳定比例 |\n")
                    f.write("|------|------------|----------|----------|\n")
                    
                    # 白天稳定度
                    unstable_day = daytime_stability.get('unstable', 0) * 100
                    neutral_day = daytime_stability.get('neutral', 0) * 100
                    stable_day = daytime_stability.get('stable', 0) * 100
                    
                    f.write(f"| 白天 | {unstable_day:.1f}% | {neutral_day:.1f}% | {stable_day:.1f}% |\n")
                    
                    # 夜间稳定度
                    unstable_night = nighttime_stability.get('unstable', 0) * 100
                    neutral_night = nighttime_stability.get('neutral', 0) * 100
                    stable_night = nighttime_stability.get('stable', 0) * 100
                    
                    f.write(f"| 夜间 | {unstable_night:.1f}% | {neutral_night:.1f}% | {stable_night:.1f}% |\n")
                    
                    # 主导稳定度类型
                    daytime_dominant = daytime_stability.idxmax() if len(daytime_stability) > 0 else '未知'
                    nighttime_dominant = nighttime_stability.idxmax() if len(nighttime_stability) > 0 else '未知'
                    
                    f.write(f"\n白天主导稳定度类型: **{daytime_dominant}**\n")
                    f.write(f"夜间主导稳定度类型: **{nighttime_dominant}**\n\n")
            
            # 预报性能比较
            f.write("## 5. 季节划分对预报性能的影响\n\n")
            
            if traditional_performance is not None:
                f.write("### 5.1 传统季节划分下的预报性能\n\n")
                
                f.write("| 季节 | EC-MAE | GFS-MAE | 更优模式 | 样本数 |\n")
                f.write("|------|--------|---------|----------|--------|\n")
                
                for season, perf in traditional_performance.iterrows():
                    f.write(f"| {season} | {perf['ec_mae']:.3f} | {perf['gfs_mae']:.3f} | {perf['better_model']} | {perf['sample_count']} |\n")
                
                # 找出EC和GFS各自表现最好的季节
                ec_best_season = traditional_performance['ec_mae'].idxmin()
                gfs_best_season = traditional_performance['gfs_mae'].idxmin()
                
                f.write(f"\n**EC模式表现最佳的季节:** {ec_best_season}\n")
                f.write(f"**GFS模式表现最佳的季节:** {gfs_best_season}\n\n")
            else:
                f.write("未找到足够的预报误差数据来评估预报性能。\n\n")
            
            if custom_performance is not None:
                f.write("### 5.2 基于聚类的季节划分下的预报性能\n\n")
                
                f.write("| 季节 | EC-MAE | GFS-MAE | 更优模式 | 样本数 |\n")
                f.write("|------|--------|---------|----------|--------|\n")
                
                for season, perf in custom_performance.iterrows():
                    f.write(f"| {season} | {perf['ec_mae']:.3f} | {perf['gfs_mae']:.3f} | {perf['better_model']} | {perf['sample_count']} |\n")
                
                # 找出EC和GFS各自表现最好的季节
                ec_best_season = custom_performance['ec_mae'].idxmin()
                gfs_best_season = custom_performance['gfs_mae'].idxmin()
                
                f.write(f"\n**EC模式表现最佳的季节:** {ec_best_season}\n")
                f.write(f"**GFS模式表现最佳的季节:** {gfs_best_season}\n\n")
                
                # 比较传统和聚类季节划分的总体性能
                if traditional_performance is not None:
                    trad_overall_ec = traditional_performance['ec_mae'].mean()
                    trad_overall_gfs = traditional_performance['gfs_mae'].mean()
                    
                    custom_overall_ec = custom_performance['ec_mae'].mean()
                    custom_overall_gfs = custom_performance['gfs_mae'].mean()
                    
                    f.write("### 5.3 季节划分方法性能比较\n\n")
                    
                    f.write("| 划分方法 | EC-MAE | GFS-MAE | 更优模式 |\n")
                    f.write("|----------|--------|---------|----------|\n")
                    f.write(f"| 传统季节划分 | {trad_overall_ec:.3f} | {trad_overall_gfs:.3f} | {('EC' if trad_overall_ec < trad_overall_gfs else 'GFS')} |\n")
                    f.write(f"| 聚类季节划分 | {custom_overall_ec:.3f} | {custom_overall_gfs:.3f} | {('EC' if custom_overall_ec < custom_overall_gfs else 'GFS')} |\n")
                    
                    # 计算性能改进
                    if min(trad_overall_ec, trad_overall_gfs) > min(custom_overall_ec, custom_overall_gfs):
                        improvement = (min(trad_overall_ec, trad_overall_gfs) - min(custom_overall_ec, custom_overall_gfs)) / min(trad_overall_ec, trad_overall_gfs) * 100
                        f.write(f"\n**基于聚类的季节划分改进了预报性能: {improvement:.2f}%**\n\n")
                    else:
                        improvement = (min(custom_overall_ec, custom_overall_gfs) - min(trad_overall_ec, trad_overall_gfs)) / min(custom_overall_ec, custom_overall_gfs) * 100
                        f.write(f"\n**传统季节划分表现更好: {improvement:.2f}%**\n\n")
            
            # 总结与建议
            f.write("## 6. 结论与建议\n\n")
            
            # 昼夜分割建议
            f.write("### 6.1 昼夜分割建议\n\n")
            
            f.write("基于分析结果，我们建议采用以下昼夜分割方法：\n\n")
            f.write("1. **使用天文计算的日出日落时间**，但考虑以下调整：\n")
            f.write("   - 白天定义为日出后1小时至日落前1小时\n")
            f.write("   - 夜间定义为日落后1小时至日出前1小时\n")
            f.write("   - 其余时间为过渡期，可根据需要归类\n\n")
            f.write("2. **季节性调整**：\n")
            
            # 如果有温度模式分析结果，添加相关建议
            if temp_pattern:
                winter_months = [12, 1, 2]
                summer_months = [6, 7, 8]
                
                winter_sunrise = [temp_pattern.get(m, {}).get('sunrise_approx') for m in winter_months if m in temp_pattern]
                winter_sunrise = [h for h in winter_sunrise if h is not None]
                winter_avg_sunrise = sum(winter_sunrise) / len(winter_sunrise) if winter_sunrise else None
                
                summer_sunrise = [temp_pattern.get(m, {}).get('sunrise_approx') for m in summer_months if m in temp_pattern]
                summer_sunrise = [h for h in summer_sunrise if h is not None]
                summer_avg_sunrise = sum(summer_sunrise) / len(summer_sunrise) if summer_sunrise else None
                
                if winter_avg_sunrise is not None and summer_avg_sunrise is not None:
                    f.write(f"   - 冬季白天开始时间约为{winter_avg_sunrise:.1f}点\n")
                    f.write(f"   - 夏季白天开始时间约为{summer_avg_sunrise:.1f}点\n")
            
            # 季节划分建议
            f.write("\n### 6.2 季节划分建议\n\n")
            
            if month_to_season:
                f.write("基于聚类分析，我们建议对昌马站采用以下季节划分：\n\n")
                
                # 提取特征季节
                seasons_by_month = {}
                for season in set(month_to_season.values()):
                    months = [month for month, s in month_to_season.items() if s == season]
                    months.sort()  # 按月份排序
                    seasons_by_month[season] = months
                
                for season, months in seasons_by_month.items():
                    months_str = ", ".join([f"{m}月" for m in months])
                    f.write(f"- **{season}**: {months_str}\n")
                
                f.write("\n此划分反映了昌马站的实际气象特征，而非简单的日历季节。\n\n")
                
                # 预报策略建议
                f.write("### 6.3 预报策略建议\n\n")
                
                if custom_performance is not None:
                    # 为每个自然季节找出更好的预报模式
                    for season, perf in custom_performance.iterrows():
                        better_model = perf['better_model']
                        improvement = abs(perf['ec_mae'] - perf['gfs_mae']) / max(perf['ec_mae'], perf['gfs_mae']) * 100
                        
                        f.write(f"**{season}预报策略:**\n\n")
                        f.write(f"- 优先使用**{better_model}**模式 (性能提升约{improvement:.1f}%)\n")
                        
                        # 如果有稳定度数据，添加稳定度相关建议
                        if stability_counts is not None:
                            seasons_stab = [s for s in stability_counts.index.get_level_values(0).unique() if s in season]
                            if seasons_stab:
                                season_stab = seasons_stab[0]
                                season_data = stability_counts.xs(season_stab, level=0)
                                
                                # 找出主导稳定度
                                daytime_stability = season_data.loc[range(self.standard_daytime_start, self.standard_daytime_end)].mean()
                                nighttime_stability = season_data.loc[list(range(0, self.standard_daytime_start)) + list(range(self.standard_daytime_end, 24))].mean()
                                
                                daytime_dominant = daytime_stability.idxmax() if len(daytime_stability) > 0 else '未知'
                                nighttime_dominant = nighttime_stability.idxmax() if len(nighttime_stability) > 0 else '未知'
                                
                                f.write(f"- 白天主要表现为**{daytime_dominant}**稳定度条件\n")
                                f.write(f"- 夜间主要表现为**{nighttime_dominant}**稳定度条件\n")
                        
                        f.write("\n")
                elif traditional_performance is not None:
                    # 如果没有自定义季节性能数据，使用传统季节数据提供建议
                    for season, perf in traditional_performance.iterrows():
                        better_model = perf['better_model']
                        improvement = abs(perf['ec_mae'] - perf['gfs_mae']) / max(perf['ec_mae'], perf['gfs_mae']) * 100
                        
                        f.write(f"**{season}预报策略:**\n\n")
                        f.write(f"- 优先使用**{better_model}**模式 (性能提升约{improvement:.1f}%)\n\n")
                else:
                    f.write("无法提供具体的预报策略建议，因为缺少预报性能数据。\n\n")
            
            # 总体推荐
            f.write("### 6.4 总体推荐\n\n")
            
            f.write("1. **实施自然季节划分**：基于气象特征的季节划分比传统日历季节更符合昌马站的实际情况\n")
            f.write("2. **应用天文昼夜分割**：根据地理位置计算的日出日落时间，更准确地反映昼夜变化\n")
            f.write("3. **季节性预报策略**：为每个自然季节选择更优的预报模式\n")
            f.write("4. **稳定度关联分析**：进一步研究稳定度与预报误差的关系，优化预报策略\n\n")
            
            # 报告结束
            f.write("---\n")
            f.write("\n*报告生成时间：" + datetime.now().strftime('%Y-%m-%d %H:%M:%S') + "*")
        
        print(f"分析报告已生成：{report_path}")
        return report_path


def main():
    """主函数"""
    # 昌马站准确坐标
    latitude = 40.2053
    longitude = 96.8114
    
    # 设置正确的数据路径和输出路径
    data_path = "/Users/xiaxin/work/WindForecast_Project/01_Data/processed/matched_data/changma_matched.csv"
    output_dir = "/Users/xiaxin/work/WindForecast_Project/03_Results/seasonal_analysis"
    
    try:
        print("=" * 60)
        print("昌马站季节和昼夜模式分析")
        print("=" * 60)
        
        # 初始化分析器
        analyzer = SeasonalPatternAnalyzer(latitude=latitude, longitude=longitude)
        
        # 加载数据
        print("\n步骤1: 加载数据")
        data = analyzer.load_data(data_path)
        
        # 添加基于天文计算的昼夜分割
        print("\n步骤2: 添加天文昼夜分割")
        data = analyzer.add_astronomical_daytime()
        
        # 分析温度变化模式
        print("\n步骤3: 分析温度变化模式")
        temp_pattern = analyzer.analyze_temperature_patterns()
        
        # 使用聚类识别自然季节
        print("\n步骤4: 使用聚类识别自然季节")
        month_to_season, monthly_features, clusters = analyzer.cluster_seasonal_patterns(n_clusters=4)
        
        # 绘制季节聚类结果
        print("\n步骤5: 绘制季节聚类结果")
        analyzer.plot_seasonal_clusters(monthly_features, clusters)
        
        # 分析昼夜变化模式
        print("\n步骤6: 分析昼夜变化模式")
        hourly_seasonal_wind, stability_counts = analyzer.analyze_diurnal_patterns()
        
        # 绘制昼夜变化模式
        print("\n步骤7: 绘制昼夜变化模式")
        analyzer.plot_diurnal_patterns(hourly_seasonal_wind, stability_counts)
        
        # 比较预报性能
        print("\n步骤8: 比较预报性能")
        try:
            traditional_performance, custom_performance = analyzer.compare_forecast_performance(month_to_season)
        except:
            print("无法进行预报性能分析，将继续生成其他部分报告")
            traditional_performance, custom_performance = None, None
        
        # 生成分析报告
        print("\n步骤9: 生成分析报告")
        report_path = analyzer.generate_report(
            temp_pattern, 
            month_to_season, 
            hourly_seasonal_wind, 
            stability_counts, 
            traditional_performance, 
            custom_performance
        )
        
        print("\n" + "=" * 60)
        print(f"分析完成！报告已保存到：{report_path}")
        print("=" * 60)
        
    except Exception as e:
        print(f"分析过程中出现错误: {str(e)}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()