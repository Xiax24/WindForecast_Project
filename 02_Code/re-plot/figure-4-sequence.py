#!/usr/bin/env python3
"""
批量生成所有测试集日期的案例研究图
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from pathlib import Path
import warnings

warnings.filterwarnings('ignore')

plt.rcParams.update({
    'font.family': ['Arial', 'DejaVu Sans'],
    'font.size': 14,
    'axes.linewidth': 1.5,
    'figure.dpi': 300,
})


class TimeSeriesCaseStudy:
    """时间序列案例研究 - 功率 + 风向"""
    
    def __init__(self, test_results_path, output_dir, nwp_source='ec'):
        self.test_results_path = Path(test_results_path)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.nwp_source = nwp_source
        
        # 线条样式
        self.line_styles = {
            'obs': {'color': 'black', 'linewidth': 2.5, 'label': 'Observed', 
                   'linestyle': '-', 'alpha': 1.0, 'zorder': 10},
            'HH': {'color': '#808080', 'linewidth': 1.5, 'label': 'HH', 
                  'linestyle': '-.', 'alpha': 0.7, 'zorder': 5},
            'SR': {'color': '#A8DADC', 'linewidth': 1.5, 'label': 'SR', 
                  'linestyle': '--', 'alpha': 0.7, 'zorder': 6},
            'ER': {'color': '#457B9D', 'linewidth': 1.5, 'label': 'ER', 
                  'linestyle': '-.', 'alpha': 0.7, 'zorder': 7},
            'WDA': {'color': '#E63946', 'linewidth': 2.5, 'label': 'WDA', 
                   'linestyle': '-', 'alpha': 0.9, 'zorder': 9}
        }
        
        # 风向区间颜色
        self.wind_region_colors = {
            'free': '#FFCDD2',
            'wake': "#CDE5FF",
            'others': '#E0E0E0'
        }
        
        self.wind_region_names = {
            'free': 'Free-stream',
            'wake': 'Wake-affected',
            'others': 'Transitional'
        }
    
    def get_all_test_dates(self):
        """获取测试集所有日期"""
        df = pd.read_csv(self.test_results_path)
        df['datetime'] = pd.to_datetime(df['datetime'])
        df['date'] = df['datetime'].dt.date
        
        all_dates = sorted(df['date'].unique())
        
        print(f"\n{'='*70}")
        print(f"测试集日期范围: {all_dates[0]} 到 {all_dates[-1]}")
        print(f"总共 {len(all_dates)} 天")
        print(f"{'='*70}")
        
        return all_dates
    
    def load_and_plot(self, target_date):
        """加载数据并绘制"""
        df = pd.read_csv(self.test_results_path)
        df['datetime'] = pd.to_datetime(df['datetime'])
        df['date'] = df['datetime'].dt.date
        
        if isinstance(target_date, str):
            target_date = pd.to_datetime(target_date).date()
        
        day_data = df[df['date'] == target_date].copy()
        
        if len(day_data) == 0:
            return None
        
        day_data = day_data.sort_values('datetime').reset_index(drop=True)
        
        self.plot(day_data, target_date)
        
        return day_data
    
    def plot(self, day_data, date):
        """画图 - 2行"""
        
        # 创建2行子图
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(20, 8),
                                       gridspec_kw={'height_ratios': [2, 1]})
        
        # 获取时间
        times = day_data['datetime']
        
        # 设置x轴范围：从当天00:00到次日00:00
        time_min = pd.Timestamp(date).replace(hour=0, minute=0, second=0)
        time_max = time_min + pd.Timedelta(days=1)
        
        # =================================================================
        # 第1行: 功率曲线 (15分钟间隔)
        # =================================================================
        
        # 背景阴影
        if 'wind_direction_type' in day_data.columns:
            wind_types = day_data['wind_direction_type'].values
            changes = [0] + list(np.where(wind_types[:-1] != wind_types[1:])[0] + 1) + [len(wind_types)]
            plotted_labels = set()
            for i in range(len(changes) - 1):
                start_idx = changes[i]
                end_idx = changes[i + 1]
                wind_type = wind_types[start_idx]
                label = self.wind_region_names[wind_type] if wind_type not in plotted_labels else None
                if label:
                    plotted_labels.add(wind_type)
                ax1.axvspan(times.iloc[start_idx], times.iloc[end_idx - 1],
                           alpha=0.25, color=self.wind_region_colors[wind_type],
                           label=label, zorder=0)
        
        # 功率曲线
        ax1.plot(times, day_data['power'], **self.line_styles['obs'])
        for strategy in ['HH', 'SR', 'ER', 'WDA']:
            pred_col = f'pred_{strategy}'
            if pred_col in day_data.columns:
                ax1.plot(times, day_data[pred_col], **self.line_styles[strategy])
        
        ax1.set_ylabel('Power (kW)', fontsize=18, fontweight='bold')
        ax1.tick_params(axis='both', labelsize=14)
        ax1.grid(True, alpha=0.3, linestyle='--')
        ax1.legend(loc='upper left', fontsize=14, framealpha=0.9, ncol=6)
        ax1.set_xlim(time_min, time_max)
        ax1.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d %H:%M'))
        ax1.xaxis.set_major_locator(mdates.HourLocator(interval=1))
        ax1.tick_params(labelbottom=False)
        
        date_str = date.strftime('%Y-%m-%d')
        ax1.set_title(f'Power Forecasting Comparison - {date_str}', 
                     fontsize=20, fontweight='bold', pad=15)
        
        # =================================================================
        # 第2行: 风向散点图 (15分钟间隔)
        # =================================================================
        
        wind_dir = day_data[f'{self.nwp_source}_wind_direction_70m']
        ax2.scatter(times, wind_dir, c='blue', s=30, alpha=0.6, zorder=5)
        
        ax2.set_ylabel('Wind Direction', fontsize=16, fontweight='bold', rotation=90, 
                      ha='center', va='center')
        ax2.set_xlabel('Time', fontsize=16, fontweight='bold')
        ax2.set_xlim(time_min, time_max)
        ax2.set_ylim(0, 360)
        ax2.set_yticks([0, 90, 180, 270, 360])
        ax2.set_yticklabels(['0° (N)', '90° (E)', '180° (S)', '270° (W)', '360° (N)'])
        ax2.tick_params(axis='both', labelsize=13)
        ax2.grid(True, alpha=0.3, linestyle='--')
        ax2.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d %H:%M'))
        ax2.xaxis.set_major_locator(mdates.HourLocator(interval=1))
        plt.setp(ax2.xaxis.get_majorticklabels(), rotation=30, ha='right')
        
        # 参考线
        ax2.axhline(y=45, color='blue', linestyle='--', alpha=0.3, linewidth=1)
        ax2.axhline(y=135, color='blue', linestyle='--', alpha=0.3, linewidth=1)
        ax2.axhline(y=225, color='red', linestyle='--', alpha=0.3, linewidth=1)
        ax2.axhline(y=315, color='red', linestyle='--', alpha=0.3, linewidth=1)
        
        # =================================================================
        # 保存
        # =================================================================
        plt.tight_layout()
        
        date_str = str(date).replace('-', '')
        save_path = self.output_dir / f'case_{date_str}_{self.nwp_source}.png'
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
    
    def generate_all_cases(self):
        """批量生成所有日期的图"""
        
        # 获取所有日期
        all_dates = self.get_all_test_dates()
        
        print(f"\n开始批量生成...")
        
        success_count = 0
        fail_count = 0
        
        for idx, date in enumerate(all_dates, 1):
            try:
                print(f"\n[{idx}/{len(all_dates)}] 处理 {date}...", end=' ')
                
                result = self.load_and_plot(date)
                
                if result is not None:
                    print(f"✓ 成功")
                    success_count += 1
                else:
                    print(f"✗ 无数据")
                    fail_count += 1
                    
            except Exception as e:
                print(f"✗ 错误: {e}")
                fail_count += 1
                continue
        
        print(f"\n{'='*70}")
        print(f"批量生成完成！")
        print(f"  成功: {success_count} 天")
        print(f"  失败: {fail_count} 天")
        print(f"  图片保存在: {self.output_dir}")
        print(f"{'='*70}")


# ============================================================================
# 主函数
# ============================================================================
if __name__ == "__main__":
    
    TEST_RESULTS_CSV = "/Users/xiaxin/work/WindForecast_Project/03_Results/re-plot-figures/figure-4/test_results_ec.csv"
    OUTPUT_DIR = "/Users/xiaxin/work/WindForecast_Project/03_Results/re-plot-figures/figure-4/case-studies-all/"
    
    # 创建绘图器
    plotter = TimeSeriesCaseStudy(
        test_results_path=TEST_RESULTS_CSV,
        output_dir=OUTPUT_DIR,
        nwp_source='ec'
    )
    
    # 批量生成所有日期的图
    plotter.generate_all_cases()