#!/usr/bin/env python3
"""
EOF权重风切变指数分析脚本
直接运行你的EOF分析并计算风切变指数

使用方法：
1. 确保你的数据文件路径正确
2. 运行此脚本
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import os
import warnings
warnings.filterwarnings('ignore')

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial Unicode MS', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

def calculate_eof_wind_shear_alpha(eof_patterns, heights):
    """计算EOF分解后各模态权重对应的风切变指数α"""
    
    # 确保heights是numpy数组
    heights = np.array(heights)
    
    # 存储结果
    alpha_results = {
        'eof_patterns': eof_patterns,
        'heights': heights,
        'alpha_10_70': [],
        'alpha_10_30': [],
        'alpha_30_50': [],
        'alpha_50_70': [],
        'alpha_all_pairs': {},
        'wind_speed_ratios': {},
        'pattern_analysis': []
    }
    
    # 定义高度索引映射
    height_indices = {10: 0, 30: 1, 50: 2, 70: 3}
    
    print("EOF模态权重风切变指数分析")
    print("=" * 60)
    
    for mode_idx in range(eof_patterns.shape[0]):
        pattern = eof_patterns[mode_idx, :]
        mode_name = f"EOF{mode_idx + 1}"
        
        print(f"\n{mode_name} 模态分析:")
        print(f"权重值: {pattern}")
        
        # 计算主要的风切变指数 (10m-70m)
        w_10m = pattern[height_indices[10]]
        w_70m = pattern[height_indices[70]]
        
        # 风切变指数公式: α = ln(w2/w1) / ln(h2/h1)
        if w_10m > 0 and w_70m > 0:
            alpha_10_70 = np.log(w_70m / w_10m) / np.log(70 / 10)
        elif w_10m < 0 and w_70m < 0:
            # 如果两个权重都是负数，使用绝对值
            alpha_10_70 = np.log(abs(w_70m) / abs(w_10m)) / np.log(70 / 10)
        else:
            # 如果权重符号不同，设为NaN
            alpha_10_70 = np.nan
            
        alpha_results['alpha_10_70'].append(alpha_10_70)
        
        # 计算其他高度对的风切变指数
        height_pairs = [(10, 30), (30, 50), (50, 70), (10, 50), (30, 70)]
        mode_alphas = {}
        
        for h1, h2 in height_pairs:
            w1 = pattern[height_indices[h1]]
            w2 = pattern[height_indices[h2]]
            
            if w1 > 0 and w2 > 0:
                alpha = np.log(w2 / w1) / np.log(h2 / h1)
            elif w1 < 0 and w2 < 0:
                alpha = np.log(abs(w2) / abs(w1)) / np.log(h2 / h1)
            else:
                alpha = np.nan
                
            mode_alphas[f"{h1}_{h2}"] = alpha
            
            if f"{h1}_{h2}" == "10_30":
                alpha_results['alpha_10_30'].append(alpha)
            elif f"{h1}_{h2}" == "30_50":
                alpha_results['alpha_30_50'].append(alpha)
            elif f"{h1}_{h2}" == "50_70":
                alpha_results['alpha_50_70'].append(alpha)
        
        alpha_results['alpha_all_pairs'][mode_name] = mode_alphas
        
        # 计算风速比值
        ratios = {}
        for h1, h2 in [(10, 30), (10, 50), (10, 70), (30, 50), (30, 70), (50, 70)]:
            w1 = pattern[height_indices[h1]]
            w2 = pattern[height_indices[h2]]
            if w1 != 0:
                ratios[f"{h2}m/{h1}m"] = w2 / w1
            else:
                ratios[f"{h2}m/{h1}m"] = np.inf if w2 > 0 else -np.inf if w2 < 0 else np.nan
                
        alpha_results['wind_speed_ratios'][mode_name] = ratios
        
        # 模态特征分析
        pattern_info = {
            'mode': mode_name,
            'weights': pattern,
            'alpha_10_70': alpha_10_70,
            'weight_range': np.max(pattern) - np.min(pattern),
            'weight_std': np.std(pattern),
            'monotonic': is_monotonic(pattern),
            'sign_changes': count_sign_changes(pattern),
            'dominant_height': heights[np.argmax(np.abs(pattern))],
            'weight_correlation_with_height': np.corrcoef(pattern, heights)[0, 1]
        }
        
        alpha_results['pattern_analysis'].append(pattern_info)
        
        # 打印详细结果
        print(f"  主要风切变指数 α(10m-70m): {alpha_10_70:.4f}")
        print(f"  权重与高度相关性: {pattern_info['weight_correlation_with_height']:.4f}")
        print(f"  权重符号变化次数: {pattern_info['sign_changes']}")
        print(f"  主导高度: {pattern_info['dominant_height']}m")
        
        if not np.isnan(alpha_10_70):
            if alpha_10_70 > 0.3:
                print(f"  → 强正风切变模态 (上层风速增长快)")
            elif alpha_10_70 > 0.1:
                print(f"  → 中等正风切变模态")
            elif alpha_10_70 > -0.1:
                print(f"  → 近似均匀模态 (各高度变化相似)")
            elif alpha_10_70 > -0.3:
                print(f"  → 中等负风切变模态")
            else:
                print(f"  → 强负风切变模态 (下层风速增长快)")
        else:
            print(f"  → 复杂模态 (权重符号不一致)")
    
    return alpha_results

def is_monotonic(arr):
    """检查数组是否单调"""
    return np.all(arr[1:] >= arr[:-1]) or np.all(arr[1:] <= arr[:-1])

def count_sign_changes(arr):
    """计算符号变化次数"""
    signs = np.sign(arr)
    sign_changes = np.sum(signs[1:] != signs[:-1])
    return sign_changes

def run_eof_analysis(data_path):
    """运行完整的EOF分析"""
    
    print("加载数据...")
    try:
        df = pd.read_csv(data_path)
        print(f"数据加载成功，形状: {df.shape}")
    except Exception as e:
        print(f"数据加载失败: {e}")
        return None
    
    # 转换时间列
    if 'datetime' in df.columns:
        df['datetime'] = pd.to_datetime(df['datetime'])
        df.set_index('datetime', inplace=True)
    
    # 定义风速变量和高度
    wind_variables = ['obs_wind_speed_10m', 'obs_wind_speed_30m', 'obs_wind_speed_50m', 'obs_wind_speed_70m']
    heights = [10, 30, 50, 70]
    
    # 检查数据中是否包含这些变量
    missing_vars = [var for var in wind_variables if var not in df.columns]
    if missing_vars:
        print(f"缺失变量: {missing_vars}")
        print(f"数据中包含的列: {df.columns.tolist()}")
        return None
    
    # 提取风速数据
    wind_data = df[wind_variables].copy()
    wind_data_clean = wind_data.dropna()
    
    print(f"清洗后数据点数: {len(wind_data_clean)}")
    print(f"数据范围: {wind_data_clean.index.min()} 到 {wind_data_clean.index.max()}")
    
    # 标准化数据
    data_std = (wind_data_clean - wind_data_clean.mean()) / wind_data_clean.std()
    
    # 执行PCA
    print("\n执行EOF分析...")
    pca = PCA(n_components=4)
    time_coefficients = pca.fit_transform(data_std)
    
    # EOF模态和解释方差
    eof_patterns = pca.components_
    explained_variance = pca.explained_variance_ratio_
    
    print("EOF分析完成!")
    print(f"各模态解释方差: {explained_variance * 100}")
    
    # 计算风切变指数
    print("\n" + "="*60)
    alpha_results = calculate_eof_wind_shear_alpha(eof_patterns, heights)
    
    return {
        'eof_patterns': eof_patterns,
        'explained_variance': explained_variance,
        'time_coefficients': time_coefficients,
        'alpha_results': alpha_results,
        'data_clean': wind_data_clean
    }

def create_simple_visualization(results):
    """创建简化的可视化图表"""
    
    eof_patterns = results['eof_patterns']
    explained_variance = results['explained_variance']
    alpha_results = results['alpha_results']
    heights = [10, 30, 50, 70]
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle('EOF权重风切变指数分析结果', fontsize=16, fontweight='bold')
    
    colors = ['red', 'blue', 'green', 'orange']
    
    # 图1: EOF权重廓线
    ax1 = axes[0, 0]
    for i in range(4):
        ax1.plot(heights, eof_patterns[i, :], 'o-', color=colors[i], 
                linewidth=2, markersize=8, label=f'EOF{i+1} ({explained_variance[i]*100:.1f}%)')
    ax1.axhline(y=0, color='black', linestyle='--', alpha=0.5)
    ax1.set_xlabel('高度 (m)')
    ax1.set_ylabel('EOF权重')
    ax1.set_title('EOF模态权重廓线')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # 图2: 风切变指数
    ax2 = axes[0, 1]
    alpha_10_70 = alpha_results['alpha_10_70']
    modes = [f'EOF{i+1}' for i in range(4)]
    
    bars = ax2.bar(modes, alpha_10_70, color=colors, alpha=0.7)
    ax2.axhline(y=0, color='black', linestyle='-', alpha=0.5)
    ax2.axhline(y=0.14, color='red', linestyle='--', alpha=0.7, label='α=1/7≈0.14')
    ax2.set_ylabel('风切变指数 α')
    ax2.set_title('EOF模态风切变指数 (10m-70m)')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # 添加数值标签
    for bar, alpha_val in zip(bars, alpha_10_70):
        if not np.isnan(alpha_val):
            ax2.text(bar.get_x() + bar.get_width()/2, alpha_val + 0.01*np.sign(alpha_val), 
                    f'{alpha_val:.3f}', ha='center', va='bottom' if alpha_val > 0 else 'top')
    
    # 图3: 权重与高度相关性
    ax3 = axes[1, 0]
    correlations = [info['weight_correlation_with_height'] for info in alpha_results['pattern_analysis']]
    bars = ax3.bar(modes, correlations, color=colors, alpha=0.7)
    ax3.axhline(y=0, color='black', linestyle='-', alpha=0.5)
    ax3.set_ylabel('相关系数')
    ax3.set_title('权重与高度相关性')
    ax3.set_ylim(-1, 1)
    ax3.grid(True, alpha=0.3)
    
    # 添加数值标签
    for bar, corr in zip(bars, correlations):
        ax3.text(bar.get_x() + bar.get_width()/2, corr + 0.02*np.sign(corr), 
                f'{corr:.3f}', ha='center', va='bottom' if corr > 0 else 'top')
    
    # 图4: 物理解释文本
    ax4 = axes[1, 1]
    ax4.axis('off')
    
    interpretation_text = "EOF物理解释:\n\n"
    
    for i, info in enumerate(alpha_results['pattern_analysis']):
        alpha_val = info['alpha_10_70']
        interpretation_text += f"EOF{i+1} (解释方差 {explained_variance[i]*100:.1f}%):\n"
        interpretation_text += f"  α = {alpha_val:.3f}\n"
        
        if not np.isnan(alpha_val):
            if alpha_val > 0.3:
                meaning = "强正风切变"
            elif alpha_val > 0.1:
                meaning = "中等正风切变"
            elif alpha_val > -0.1:
                meaning = "近似均匀"
            elif alpha_val > -0.3:
                meaning = "中等负风切变"
            else:
                meaning = "强负风切变"
        else:
            meaning = "复杂模态"
        
        interpretation_text += f"  类型: {meaning}\n\n"
    
    ax4.text(0.02, 0.98, interpretation_text, transform=ax4.transAxes, fontsize=11,
             verticalalignment='top', fontfamily='monospace',
             bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.8))
    
    plt.tight_layout()
    return fig

def save_results(results, output_dir):
    """保存分析结果"""
    
    os.makedirs(output_dir, exist_ok=True)
    
    alpha_results = results['alpha_results']
    
    # 创建汇总表
    summary_data = []
    for i, info in enumerate(alpha_results['pattern_analysis']):
        mode_name = info['mode']
        mode_alphas = alpha_results['alpha_all_pairs'][mode_name]
        mode_ratios = alpha_results['wind_speed_ratios'][mode_name]
        
        row = {
            'EOF_Mode': mode_name,
            'Explained_Variance_%': results['explained_variance'][i] * 100,
            'Alpha_10_70m': info['alpha_10_70'],
            'Alpha_10_30m': mode_alphas['10_30'],
            'Alpha_30_50m': mode_alphas['30_50'],
            'Alpha_50_70m': mode_alphas['50_70'],
            'Ratio_70_10m': mode_ratios['70m/10m'],
            'Weight_Height_Correlation': info['weight_correlation_with_height'],
            'Weight_Std': info['weight_std'],
            'Sign_Changes': info['sign_changes'],
            'Dominant_Height_m': info['dominant_height'],
            'Is_Monotonic': info['monotonic']
        }
        
        summary_data.append(row)
    
    summary_df = pd.DataFrame(summary_data)
    
    # 保存文件
    summary_path = os.path.join(output_dir, 'EOF_Alpha_Summary.csv')
    summary_df.to_csv(summary_path, index=False)
    
    # 保存图表
    fig = create_simple_visualization(results)
    plot_path = os.path.join(output_dir, 'EOF_Alpha_Analysis.png')
    fig.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.show()
    
    print(f"\n结果已保存:")
    print(f"  汇总表: {summary_path}")
    print(f"  图表: {plot_path}")
    
    return summary_df

# 主函数
def main():
    """主分析函数"""
    
    # 数据路径 - 请根据你的实际路径修改
    data_path = '/Users/xiaxin/work/WindForecast_Project/01_Data/processed/imputed_data/changma_clean.csv'
    output_dir = '/Users/xiaxin/work/WindForecast_Project/03_Results/08时间序列分解/EOF_Alpha_Analysis'
    
    # 检查文件是否存在
    if not os.path.exists(data_path):
        print(f"数据文件不存在: {data_path}")
        print("请修改data_path变量为正确的文件路径")
        return
    
    # 运行分析
    print("开始EOF权重风切变指数分析...")
    print("=" * 60)
    
    results = run_eof_analysis(data_path)
    
    if results is None:
        print("分析失败，请检查数据文件和路径")
        return
    
    # 保存结果
    summary_df = save_results(results, output_dir)
    
    # 打印关键结果
    print("\n" + "="*60)
    print("关键结果摘要:")
    print("="*60)
    
    for i, info in enumerate(results['alpha_results']['pattern_analysis']):
        alpha_val = info['alpha_10_70']
        explained_var = results['explained_variance'][i] * 100
        
        print(f"\nEOF{i+1} ({explained_var:.1f}% 方差):")
        print(f"  权重: {info['weights']}")
        print(f"  风切变指数 α: {alpha_val:.4f}")
        
        if not np.isnan(alpha_val):
            if alpha_val > 0.1:
                print(f"  → 正风切变模态: 上层比下层变化大")
            elif alpha_val > -0.1:
                print(f"  → 均匀模态: 各高度变化相似")
            else:
                print(f"  → 负风切变模态: 下层比上层变化大")
        else:
            print(f"  → 复杂模态: 权重符号混合")

if __name__ == "__main__":
    main()