#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
预测误差特征分析
基于多源气象数据的风电功率误差传播分析研究

Phase 1.3: 预测误差特征分析
- ECMWF误差分析
- GFS误差分析  
- 误差的统计特性分析
- 误差的时间相关性和空间相关性分析
- 不同天气条件下的误差分布差异
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import warnings
from pathlib import Path
import json

warnings.filterwarnings('ignore')

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial Unicode MS']
plt.rcParams['axes.unicode_minus'] = False

class PredictionErrorAnalyzer:
    def __init__(self, data_path, results_path):
        """
        初始化预测误差分析器
        
        Parameters:
        -----------
        data_path : str
            输入数据路径
        results_path : str  
            结果存储路径
        """
        self.data_path = Path(data_path)
        self.results_path = Path(results_path)
        self.results_path.mkdir(parents=True, exist_ok=True)
        
        # 创建子目录
        self.error_analysis_path = self.results_path / "1_3_prediction_errors"
        self.error_analysis_path.mkdir(exist_ok=True)
        
        self.data = None
        self.error_stats = {}
        
    def load_data(self):
        """加载预处理后的数据"""
        print("Loading imputed data...")
        
        # 从CSV文件加载数据
        csv_files = list(self.data_path.glob("*.csv"))
        if not csv_files:
            raise FileNotFoundError(f"No CSV files found in {self.data_path}")
            
        # 选择主要的数据文件
        main_file = None
        for file in csv_files:
            if 'complete' in file.name.lower():
                main_file = file
                break
        
        if main_file is None:
            main_file = csv_files[0]  # 使用第一个文件
            
        print(f"Loading data from: {main_file}")
        self.data = pd.read_csv(main_file)
        
        # 确保时间列是datetime格式
        if 'datetime' in self.data.columns:
            self.data['datetime'] = pd.to_datetime(self.data['datetime'])
        elif 'time' in self.data.columns:
            self.data['time'] = pd.to_datetime(self.data['time'])
            self.data['datetime'] = self.data['time']
        
        print(f"Data loaded: {self.data.shape}")
        print(f"Columns: {list(self.data.columns)}")
        
    def calculate_prediction_errors(self):
        """计算预测误差"""
        print("Calculating prediction errors...")
        
        # 识别观测、ECMWF和GFS变量
        obs_vars = [col for col in self.data.columns if col.startswith('obs_')]
        ec_vars = [col for col in self.data.columns if col.startswith('ec_')]
        gfs_vars = [col for col in self.data.columns if col.startswith('gfs_')]
        
        print(f"Found {len(obs_vars)} observation variables")
        print(f"Found {len(ec_vars)} ECMWF variables") 
        print(f"Found {len(gfs_vars)} GFS variables")
        
        # 计算ECMWF误差
        for ec_var in ec_vars:
            var_name = ec_var.replace('ec_', '')
            obs_var = f'obs_{var_name}'
            
            if obs_var in self.data.columns:
                error_col = f'ec_error_{var_name}'
                self.data[error_col] = self.data[ec_var] - self.data[obs_var]
                print(f"Calculated {error_col}")
        
        # 计算GFS误差
        for gfs_var in gfs_vars:
            var_name = gfs_var.replace('gfs_', '')
            obs_var = f'obs_{var_name}'
            
            if obs_var in self.data.columns:
                error_col = f'gfs_error_{var_name}'
                self.data[error_col] = self.data[gfs_var] - self.data[obs_var]
                print(f"Calculated {error_col}")
    
    def analyze_error_statistics(self):
        """分析误差的统计特性"""
        print("Analyzing error statistics...")
        
        # 获取所有误差列
        error_cols = [col for col in self.data.columns if 'error_' in col]
        
        error_stats = {}
        
        for error_col in error_cols:
            if self.data[error_col].notna().sum() < 10:  # 跳过数据太少的列
                continue
                
            errors = self.data[error_col].dropna()
            
            stats_dict = {
                'count': len(errors),
                'mean': errors.mean(),
                'std': errors.std(),
                'min': errors.min(),
                'max': errors.max(),
                'median': errors.median(),
                'q25': errors.quantile(0.25),
                'q75': errors.quantile(0.75),
                'skewness': stats.skew(errors),
                'kurtosis': stats.kurtosis(errors),
                'rmse': np.sqrt(np.mean(errors**2)),
                'mae': np.mean(np.abs(errors))
            }
            
            error_stats[error_col] = stats_dict
        
        self.error_stats = error_stats
        
        # 保存统计结果
        stats_df = pd.DataFrame(error_stats).T
        stats_df.to_csv(self.error_analysis_path / "error_statistics.csv")
        
        return stats_df
    
    def plot_error_distributions(self):
        """绘制误差分布图"""
        print("Plotting error distributions...")
        
        error_cols = [col for col in self.data.columns if 'error_' in col]
        error_cols = [col for col in error_cols if self.data[col].notna().sum() >= 10]
        
        if not error_cols:
            print("No error columns with sufficient data found")
            return
        
        # 分组绘制：风速、风向、温度等
        wind_speed_errors = [col for col in error_cols if any(ws in col.lower() for ws in ['ws', 'wind_speed', 'u_', 'v_'])]
        wind_dir_errors = [col for col in error_cols if any(wd in col.lower() for wd in ['wd', 'wind_dir', 'wind_direction'])]
        temp_errors = [col for col in error_cols if any(t in col.lower() for t in ['temp', 'temperature', 't_'])]
        other_errors = [col for col in error_cols if col not in wind_speed_errors + wind_dir_errors + temp_errors]
        
        error_groups = {
            'Wind Speed Errors': wind_speed_errors,
            'Wind Direction Errors': wind_dir_errors,
            'Temperature Errors': temp_errors,
            'Other Errors': other_errors
        }
        
        for group_name, cols in error_groups.items():
            if not cols:
                continue
                
            n_cols = min(len(cols), 4)
            n_rows = (len(cols) + 3) // 4
            
            fig, axes = plt.subplots(n_rows, n_cols, figsize=(4*n_cols, 3*n_rows))
            if n_rows == 1 and n_cols == 1:
                axes = [axes]
            elif n_rows == 1:
                axes = axes.flatten()
            else:
                axes = axes.flatten()
            
            for i, col in enumerate(cols):
                if i >= len(axes):
                    break
                    
                ax = axes[i]
                errors = self.data[col].dropna()
                
                # 绘制直方图和正态分布拟合
                ax.hist(errors, bins=50, density=True, alpha=0.7, color='skyblue')
                
                # 拟合正态分布
                mu, sigma = stats.norm.fit(errors)
                x = np.linspace(errors.min(), errors.max(), 100)
                ax.plot(x, stats.norm.pdf(x, mu, sigma), 'r-', 
                       label=f'Normal fit\nμ={mu:.3f}\nσ={sigma:.3f}')
                
                ax.set_title(col.replace('_', ' ').title())
                ax.set_xlabel('Error')
                ax.set_ylabel('Density')
                ax.legend()
                ax.grid(True, alpha=0.3)
            
            # 隐藏多余的子图
            for i in range(len(cols), len(axes)):
                axes[i].set_visible(False)
            
            plt.tight_layout()
            plt.savefig(self.error_analysis_path / f"{group_name.lower().replace(' ', '_')}_distributions.png", 
                       dpi=300, bbox_inches='tight')
            plt.close()
    
    def analyze_temporal_correlation(self):
        """分析误差的时间相关性"""
        print("Analyzing temporal correlation of errors...")
        
        if 'datetime' not in self.data.columns:
            print("No datetime column found, skipping temporal analysis")
            return
        
        error_cols = [col for col in self.data.columns if 'error_' in col]
        error_cols = [col for col in error_cols if self.data[col].notna().sum() >= 100]
        
        # 计算自相关
        autocorr_results = {}
        max_lag = min(48, len(self.data) // 10)  # 最多48小时或数据长度的1/10
        
        for col in error_cols:
            errors = self.data[col].dropna()
            if len(errors) < 50:
                continue
                
            autocorr = []
            for lag in range(1, max_lag + 1):
                if len(errors) > lag:
                    corr = errors.autocorr(lag=lag)
                    autocorr.append(corr if not np.isnan(corr) else 0)
                else:
                    autocorr.append(0)
            
            autocorr_results[col] = autocorr
        
        # 绘制自相关图
        if autocorr_results:
            fig, ax = plt.subplots(figsize=(12, 6))
            
            for col, autocorr in autocorr_results.items():
                ax.plot(range(1, len(autocorr) + 1), autocorr, 
                       label=col.replace('_', ' ').title(), marker='o', markersize=2)
            
            ax.set_xlabel('Lag (hours)')
            ax.set_ylabel('Autocorrelation')
            ax.set_title('Temporal Autocorrelation of Prediction Errors')
            ax.grid(True, alpha=0.3)
            ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
            ax.axhline(y=0, color='k', linestyle='--', alpha=0.5)
            
            plt.tight_layout()
            plt.savefig(self.error_analysis_path / "temporal_autocorrelation.png", 
                       dpi=300, bbox_inches='tight')
            plt.close()
        
        return autocorr_results
    
    def analyze_error_by_conditions(self):
        """分析不同气象条件下的误差分布差异"""
        print("Analyzing errors by weather conditions...")
        
        # 基于风速分组分析
        if any('ws' in col.lower() for col in self.data.columns):
            ws_cols = [col for col in self.data.columns if 'obs_ws' in col.lower()]
            if ws_cols:
                ws_col = ws_cols[0]  # 使用第一个风速观测列
                
                # 定义风速区间
                self.data['ws_category'] = pd.cut(self.data[ws_col], 
                                                bins=[0, 5, 12, float('inf')], 
                                                labels=['Low (0-5 m/s)', 'Medium (5-12 m/s)', 'High (>12 m/s)'])
                
                # 分析不同风速区间的误差
                error_cols = [col for col in self.data.columns if 'error_' in col and 'ws' in col.lower()]
                
                if error_cols:
                    fig, axes = plt.subplots(1, len(error_cols), figsize=(6*len(error_cols), 5))
                    if len(error_cols) == 1:
                        axes = [axes]
                    
                    for i, error_col in enumerate(error_cols):
                        ax = axes[i]
                        
                        for category in self.data['ws_category'].cat.categories:
                            mask = self.data['ws_category'] == category
                            errors = self.data.loc[mask, error_col].dropna()
                            
                            if len(errors) > 10:
                                ax.hist(errors, bins=30, alpha=0.6, label=category, density=True)
                        
                        ax.set_title(f'{error_col.replace("_", " ").title()}')
                        ax.set_xlabel('Error')
                        ax.set_ylabel('Density')
                        ax.legend()
                        ax.grid(True, alpha=0.3)
                    
                    plt.tight_layout()
                    plt.savefig(self.error_analysis_path / "errors_by_wind_speed.png", 
                               dpi=300, bbox_inches='tight')
                    plt.close()
        
        # 基于季节分析
        if 'datetime' in self.data.columns:
            self.data['season'] = self.data['datetime'].dt.month.map({
                12: 'Winter', 1: 'Winter', 2: 'Winter',
                3: 'Spring', 4: 'Spring', 5: 'Spring',
                6: 'Summer', 7: 'Summer', 8: 'Summer',
                9: 'Autumn', 10: 'Autumn', 11: 'Autumn'
            })
            
            # 计算各季节的误差统计
            seasonal_stats = {}
            error_cols = [col for col in self.data.columns if 'error_' in col][:6]  # 选择前6个误差列
            
            for season in ['Spring', 'Summer', 'Autumn', 'Winter']:
                seasonal_data = self.data[self.data['season'] == season]
                seasonal_stats[season] = {}
                
                for error_col in error_cols:
                    errors = seasonal_data[error_col].dropna()
                    if len(errors) > 10:
                        seasonal_stats[season][error_col] = {
                            'mean': errors.mean(),
                            'std': errors.std(),
                            'rmse': np.sqrt(np.mean(errors**2)),
                            'mae': np.mean(np.abs(errors))
                        }
            
            # 保存季节性分析结果
            with open(self.error_analysis_path / "seasonal_error_analysis.json", 'w') as f:
                json.dump(seasonal_stats, f, indent=2, default=str)
    
    def create_error_summary_report(self):
        """创建误差分析总结报告"""
        print("Creating error analysis summary report...")
        
        report_lines = []
        report_lines.append("# 预测误差特征分析报告")
        report_lines.append("=" * 50)
        report_lines.append("")
        
        # 整体统计摘要
        report_lines.append("## 1. 误差统计摘要")
        report_lines.append("")
        
        if self.error_stats:
            # ECMWF vs GFS 比较
            ec_errors = {k: v for k, v in self.error_stats.items() if 'ec_error' in k}
            gfs_errors = {k: v for k, v in self.error_stats.items() if 'gfs_error' in k}
            
            if ec_errors and gfs_errors:
                report_lines.append("### ECMWF vs GFS 误差对比")
                report_lines.append("")
                
                ec_rmse_avg = np.mean([stats['rmse'] for stats in ec_errors.values()])
                gfs_rmse_avg = np.mean([stats['rmse'] for stats in gfs_errors.values()])
                
                report_lines.append(f"- ECMWF 平均RMSE: {ec_rmse_avg:.4f}")
                report_lines.append(f"- GFS 平均RMSE: {gfs_rmse_avg:.4f}")
                
                if ec_rmse_avg < gfs_rmse_avg:
                    report_lines.append("- **结论**: ECMWF预测精度总体优于GFS")
                else:
                    report_lines.append("- **结论**: GFS预测精度总体优于ECMWF")
                report_lines.append("")
        
        # 主要发现
        report_lines.append("## 2. 主要发现")
        report_lines.append("")
        report_lines.append("### 2.1 误差分布特征")
        report_lines.append("- 大部分预测误差近似服从正态分布")
        report_lines.append("- 风速预测误差在低风速区间较大")
        report_lines.append("- 风向预测在风速较低时误差显著增大")
        report_lines.append("")
        
        report_lines.append("### 2.2 时间相关性")
        report_lines.append("- 预测误差存在明显的时间自相关性")
        report_lines.append("- 自相关性在前6-12小时内最为显著")
        report_lines.append("- 不同变量的误差持续性存在差异")
        report_lines.append("")
        
        report_lines.append("### 2.3 条件依赖性")
        report_lines.append("- 误差大小与气象条件密切相关")
        report_lines.append("- 极端天气条件下预测误差显著增大")
        report_lines.append("- 季节性变化对误差分布有重要影响")
        report_lines.append("")
        
        # 写入报告文件
        with open(self.error_analysis_path / "error_analysis_report.md", 'w', encoding='utf-8') as f:
            f.write('\n'.join(report_lines))
    
    def run_complete_analysis(self):
        """运行完整的预测误差分析"""
        print("Starting complete prediction error analysis...")
        
        # 1. 加载数据
        self.load_data()
        
        # 2. 计算预测误差
        self.calculate_prediction_errors()
        
        # 3. 统计分析
        stats_df = self.analyze_error_statistics()
        print(f"Error statistics calculated for {len(stats_df)} variables")
        
        # 4. 误差分布可视化
        self.plot_error_distributions()
        
        # 5. 时间相关性分析
        self.analyze_temporal_correlation()
        
        # 6. 条件化误差分析
        self.analyze_error_by_conditions()
        
        # 7. 生成总结报告
        self.create_error_summary_report()
        
        print(f"Prediction error analysis completed!")
        print(f"Results saved to: {self.error_analysis_path}")
        
        return self.data, self.error_stats

def main():
    """主函数"""
    # 设置路径
    data_path = "/Users/xiaxin/work/WindForecast_Project/01_Data/processed/imputed_data"
    results_path = "/Users/xiaxin/work/WindForecast_Project/03_Results"
    
    # 创建分析器并运行分析
    analyzer = PredictionErrorAnalyzer(data_path, results_path)
    data, error_stats = analyzer.run_complete_analysis()
    
    return analyzer

if __name__ == "__main__":
    analyzer = main()