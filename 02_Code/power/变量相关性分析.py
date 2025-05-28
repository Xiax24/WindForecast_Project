"""
1.2.2 相关性矩阵分析 - 完整修复版（含显著性检验）
分析观测变量和功率间的相关关系，风向进行特殊处理
"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from scipy.stats import pearsonr, spearmanr
import matplotlib.colors as mcolors
import warnings
warnings.filterwarnings('ignore')

# 设置图表样式
sns.set_style("whitegrid")
plt.rcParams['figure.dpi'] = 300

# 自定义颜色映射（仿照R代码中的配色）
def create_custom_colormap():
    """
    创建自定义的相关性颜色映射
    仿照R代码: colours = c("#FF8040","white","#5BC2CD")
    """#colors = ["#FF6600", "#FFA500", "white", "#20B2AA", "#008080"]
    colors = ["#A32903", "#F45F31", "white", "#3FC2E7", "#016DB5"]  # 橙色 -> 白色 -> 青色
    n_bins = 256
    custom_cmap = mcolors.LinearSegmentedColormap.from_list('custom', colors, N=n_bins)
    return custom_cmap

# 创建全局颜色映射
custom_cmap = create_custom_colormap()

def prepare_observation_data(df):
    """
    准备观测数据，处理风向的圆形特性
    """
    print("=== Prepare Observation Variables ===")
    
    # 选择观测变量和功率变量
    obs_cols = [col for col in df.columns if col.startswith('obs_')]
    power_cols = [col for col in df.columns if 'power' in col.lower()]
    
    analysis_cols = obs_cols + power_cols
    
    print(f"Original observation variables: {len(obs_cols)}")
    print(f"Power variables: {len(power_cols)}")
    
    # 提取数据
    df_analysis = df[analysis_cols].copy()
    
    # 分类变量
    wind_speed_cols = [col for col in obs_cols if 'wind_speed' in col]
    wind_dir_cols = [col for col in obs_cols if 'wind_direction' in col]
    temp_cols = [col for col in obs_cols if 'temperature' in col]
    humidity_cols = [col for col in obs_cols if 'humidity' in col]
    density_cols = [col for col in obs_cols if 'density' in col]
    
    print(f"Wind speed variables: {len(wind_speed_cols)}")
    print(f"Wind direction variables: {len(wind_dir_cols)}")
    print(f"Temperature variables: {len(temp_cols)}")
    print(f"Humidity variables: {len(humidity_cols)}")
    print(f"Density variables: {len(density_cols)}")
    

    # 处理风向数据 - 转换为连续变量（修正版）
    wind_dir_processed = {}

    for col in wind_dir_cols:
        # 气象角度转换为数学角度
        # 气象学：0°=北，顺时针；数学：0°=东，逆时针
        math_angle = (90 - df_analysis[col] + 360) % 360
        wind_dir_rad = np.deg2rad(math_angle)
        
        # 创建正弦和余弦分量
        sin_col = col.replace('wind_direction', 'wind_dir_sin')  # 南北分量
        cos_col = col.replace('wind_direction', 'wind_dir_cos')  # 东西分量
        
        df_analysis[sin_col] = np.sin(wind_dir_rad)  # 南北分量
        df_analysis[cos_col] = np.cos(wind_dir_rad)  # 东西分量
        
        wind_dir_processed[col] = {'sin': sin_col, 'cos': cos_col}
        
        print(f"Converted {col} to sin/cos components: {sin_col} (N-S), {cos_col} (E-W)")



    # 移除原始风向数据
    df_analysis = df_analysis.drop(columns=wind_dir_cols)
    
    # 获取最终的分析变量列表
    final_cols = df_analysis.columns.tolist()
    
    print(f"Final analysis variables: {len(final_cols)}")
    
    return {
        'data': df_analysis,
        'wind_speed_cols': wind_speed_cols,
        'wind_dir_processed': wind_dir_processed,
        'temp_cols': temp_cols,
        'humidity_cols': humidity_cols,
        'density_cols': density_cols,
        'power_cols': power_cols,
        'final_cols': final_cols
    }

def calculate_correlation_matrices(data_info):
    """
    计算相关性矩阵和显著性检验
    """
    print("\n=== Calculate Correlation Matrices with Significance Tests ===")
    
    df_clean = data_info['data'].dropna()
    print(f"Valid samples after removing missing values: {len(df_clean)}")
    
    if len(df_clean) < 10:
        print("Insufficient data for correlation analysis")
        return None
    
    # 计算Pearson和Spearman相关系数
    pearson_corr = df_clean.corr(method='pearson')
    spearman_corr = df_clean.corr(method='spearman')
    
    # 初始化p值矩阵
    pearson_pvalues = pd.DataFrame(index=pearson_corr.index, columns=pearson_corr.columns)
    spearman_pvalues = pd.DataFrame(index=spearman_corr.index, columns=spearman_corr.columns)
    
    print("Calculating significance tests...")
    
    # 计算p值
    for i, var1 in enumerate(df_clean.columns):
        for j, var2 in enumerate(df_clean.columns):
            if var1 == var2:
                pearson_pvalues.loc[var1, var2] = 0.0
                spearman_pvalues.loc[var1, var2] = 0.0
            else:
                # 获取配对的非空数据
                paired_data = df_clean[[var1, var2]].dropna()
                if len(paired_data) > 3:
                    try:
                        # Pearson检验
                        _, p_pearson = pearsonr(paired_data[var1], paired_data[var2])
                        pearson_pvalues.loc[var1, var2] = p_pearson
                        
                        # Spearman检验
                        _, p_spearman = spearmanr(paired_data[var1], paired_data[var2])
                        spearman_pvalues.loc[var1, var2] = p_spearman
                    except:
                        pearson_pvalues.loc[var1, var2] = 1.0
                        spearman_pvalues.loc[var1, var2] = 1.0
                else:
                    pearson_pvalues.loc[var1, var2] = 1.0
                    spearman_pvalues.loc[var1, var2] = 1.0
    
    # 转换p值为数值类型
    pearson_pvalues = pearson_pvalues.astype(float)
    spearman_pvalues = spearman_pvalues.astype(float)
    
    print("Correlation matrices and significance tests calculated successfully")
    
    return {
        'pearson': pearson_corr,
        'spearman': spearman_corr,
        'pearson_pvalues': pearson_pvalues,
        'spearman_pvalues': spearman_pvalues,
        'data_clean': df_clean,
        'variable_names': df_clean.columns.tolist()
    }

def plot_correlation_heatmaps(corr_results, data_info, output_dir):
    """
    绘制相关性热图（含显著性标记）
    """
    print("\n=== Plot Correlation Heatmaps with Significance ===")
    
    if corr_results is None:
        print("No correlation results available")
        return
    
    # 按指定顺序重新排列变量：风速 -> 风向 -> 温度 -> 湿度 -> 密度 -> 功率
    wind_speed_cols = [col for col in corr_results['pearson'].columns if 'wind_speed' in col]
    wind_dir_cols = [col for col in corr_results['pearson'].columns if 'wind_dir' in col]
    temp_cols = [col for col in corr_results['pearson'].columns if 'temperature' in col]
    humidity_cols = [col for col in corr_results['pearson'].columns if 'humidity' in col]
    density_cols = [col for col in corr_results['pearson'].columns if 'density' in col]
    power_cols = [col for col in corr_results['pearson'].columns if 'power' in col.lower()]
    
    # 按指定顺序重新排序变量
    ordered_vars = wind_speed_cols + wind_dir_cols + temp_cols + humidity_cols + density_cols + power_cols
    
    # 确保所有变量都包含在内
    remaining_vars = [col for col in corr_results['pearson'].columns if col not in ordered_vars]
    ordered_vars.extend(remaining_vars)
    
    # 重新排列相关性矩阵和p值矩阵
    pearson_ordered = corr_results['pearson'].loc[ordered_vars, ordered_vars]
    spearman_ordered = corr_results['spearman'].loc[ordered_vars, ordered_vars]
    pearson_pval_ordered = corr_results['pearson_pvalues'].loc[ordered_vars, ordered_vars]
    spearman_pval_ordered = corr_results['spearman_pvalues'].loc[ordered_vars, ordered_vars]
    
    # 创建显著性标记矩阵
    def create_significance_annotations(corr_matrix, pval_matrix):
        """创建带显著性标记的注释"""
        annotations = corr_matrix.copy().astype(str)
        for i in range(len(corr_matrix)):
            for j in range(len(corr_matrix.columns)):
                corr_val = corr_matrix.iloc[i, j]
                p_val = pval_matrix.iloc[i, j]
                
                # 添加显著性标记
                if p_val < 0.001:
                    sig_mark = '***'
                elif p_val < 0.01:
                    sig_mark = '**'
                elif p_val < 0.05:
                    sig_mark = '*'
                else:
                    sig_mark = ''
                
                annotations.iloc[i, j] = f'{corr_val:.2f}\n{sig_mark}'
        
        return annotations
    
    # 创建图形
    fig, axes = plt.subplots(1, 2, figsize=(24, 12))
    fig.suptitle('Observation Variables Correlation Analysis\n(* p<0.05, ** p<0.01, *** p<0.001)', 
                 fontsize=20, fontweight='bold')
    
    # 简化变量名显示
    simplified_labels = []
    for col in ordered_vars:
        if 'obs_' in col:
            label = col.replace('obs_', '').replace('_', ' ')
        else:
            label = col
        simplified_labels.append(label)
    
    # Pearson相关性热图  
    ax1 = axes[0]
    mask = np.triu(np.ones_like(pearson_ordered, dtype=bool))
    
    pearson_display = pearson_ordered.copy()
    pearson_display.columns = simplified_labels
    pearson_display.index = simplified_labels
    
    # 创建显著性注释
    pearson_annot = create_significance_annotations(pearson_ordered, pearson_pval_ordered)
    pearson_annot.columns = simplified_labels
    pearson_annot.index = simplified_labels
    

    sns.heatmap(pearson_display, mask=mask, annot=pearson_annot, cmap=custom_cmap, 
            center=0, square=True, fmt='', 
            vmin=-0.5, vmax=1,  # 设置colorbar显示范围
            cbar_kws={"shrink": .8}, ax=ax1,
            annot_kws={'size': 12})
    ax1.set_title('Pearson Correlation (Linear)', fontsize=14, fontweight='bold')
    
    # Spearman相关性热图
    ax2 = axes[1]
    mask = np.triu(np.ones_like(spearman_ordered, dtype=bool))
    
    spearman_display = spearman_ordered.copy()
    spearman_display.columns = simplified_labels
    spearman_display.index = simplified_labels
    
    # 创建显著性注释
    spearman_annot = create_significance_annotations(spearman_ordered, spearman_pval_ordered)
    spearman_annot.columns = simplified_labels
    spearman_annot.index = simplified_labels
    
    sns.heatmap(spearman_display, mask=mask, annot=spearman_annot, cmap=custom_cmap, 
            center=0, square=True, fmt='', 
            vmin=-0.5, vmax=1,  # 设置colorbar显示范围
            cbar_kws={"shrink": .8}, ax=ax2,
            annot_kws={'size': 12})
    
    ax2.set_title('Spearman Correlation (Monotonic)', fontsize=14, fontweight='bold')
    # 调整坐标轴标签字体大小
    ax1.tick_params(axis='both', which='major', labelsize=15)
    ax2.tick_params(axis='both', which='major', labelsize=15)
    # 设置x轴标签倾斜45度
    # ax1.tick_params(axis='x', rotation=45)
    # ax2.tick_params(axis='x', rotation=45)
    # 调整子图标题字体（如果你想要更大）
    ax1.set_title('Pearson Correlation (Linear)', fontsize=16, fontweight='bold')  # 从14改为16
    ax2.set_title('Spearman Correlation (Monotonic)', fontsize=16, fontweight='bold')

    # 调整主标题字体
    fig.suptitle('Observation Variables Correlation Analysis\n(* p<0.05, ** p<0.01, *** p<0.001)', 
                fontsize=24, fontweight='bold')  # 从20改为24

    # 调整colorbar字体
    for ax in [ax1, ax2]:
        cbar = ax.collections[0].colorbar
        cbar.ax.tick_params(labelsize=16)  # colorbar刻度数字
    ax1.grid(False)
    ax2.grid(False) 
    plt.tight_layout()



    
    
    plt.tight_layout()
    plt.savefig(f'{output_dir}/correlation_heatmaps_with_significance.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print("Correlation heatmaps with significance completed")

def analyze_power_correlations(corr_results, data_info, output_dir):
    """
    分析功率与观测变量的相关性 - 基于Spearman相关系数（含显著性检验）
    使用蓝色和橙色配色方案，只显示左侧两个图
    """
    print("\n=== Power Correlation Analysis (Spearman-based with Significance) ===")
    
    if corr_results is None:
        print("No correlation results available")
        return None
    
    power_cols = data_info['power_cols']
    if not power_cols:
        print("No power data found")
        return None
    
    power_col = power_cols[0]
    spearman_corr = corr_results['spearman']
    spearman_pvals = corr_results['spearman_pvalues']
    df_clean = corr_results['data_clean']
    
    print(f"Analyzing power correlations for: {power_col} (using Spearman correlation with significance tests)")
    
    # 获取功率相关性和显著性
    power_correlations = spearman_corr[power_col].abs().sort_values(ascending=False)
    power_correlations = power_correlations[power_correlations.index != power_col]
    power_pvalues = spearman_pvals[power_col]
    
    # 创建功率相关性图 - 显示排序图和变量类型图
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))  # 改为1行2列
    fig.suptitle('Power vs Observation Variables Correlation (Spearman-based)\n* p<0.05, ** p<0.01, *** p<0.001', 
                 fontsize=18, fontweight='bold')
    
    # 定义蓝色和橙色系颜色
    blue_colors = ['#016DB5', '#3FC2E7', '#87CEEB', '#B0E0E6', '#E0F6FF']
    orange_colors = ['#A32903', '#F45F31', '#FF8C69', '#FFA07A', '#FFE4E1']
    
    # 1. 功率相关性排序（带显著性标记）
    ax1 = axes[0]
    top_correlations = power_correlations.head(15)
    
    # 创建条形图的标签（带显著性标记）
    bar_labels = []
    colors = []
    for var in top_correlations.index:
        corr_val = spearman_corr[power_col][var]
        p_val = power_pvalues[var]
        
        # 添加显著性标记
        if p_val < 0.001:
            sig_mark = '***'
        elif p_val < 0.01:
            sig_mark = '**'
        elif p_val < 0.05:
            sig_mark = '*'
        else:
            sig_mark = ''
        
        bar_labels.append(f"{var.replace('obs_', '').replace('_', ' ')}{sig_mark}")
        # 使用蓝色和橙色系
        colors.append('#016DB5' if corr_val > 0 else '#F45F31')
    
    bars = ax1.barh(range(len(top_correlations)), top_correlations.values, color=colors, alpha=0.8)
    ax1.set_yticks(range(len(top_correlations)))
    ax1.set_yticklabels(bar_labels, fontsize=10)
    ax1.set_xlabel('Absolute Spearman Correlation Coefficient', fontsize=12)
    ax1.set_title('Power Correlation Ranking (Spearman)', fontsize=14, fontweight='bold')
    ax1.grid(True, alpha=0.3)
    
    # 在条形上添加数值标签
    for i, (bar, corr_val) in enumerate(zip(bars, top_correlations.values)):
        ax1.text(bar.get_width() + 0.01, bar.get_y() + bar.get_height()/2, 
                f'{corr_val:.3f}', va='center', fontsize=9, fontweight='bold')
    
    # 2. 按变量类型分组的相关性
    ax2 = axes[1]
    
    # 按变量类型分组
    var_types = {
        'Wind Speed': [col for col in power_correlations.index if 'wind_speed' in col],
        'Wind Direction': [col for col in power_correlations.index if 'wind_dir' in col],
        'Temperature': [col for col in power_correlations.index if 'temperature' in col],
        'Humidity': [col for col in power_correlations.index if 'humidity' in col],
        'Density': [col for col in power_correlations.index if 'density' in col]
    }
    
    type_max_corr = {}
    type_significance = {}
    for var_type, vars_list in var_types.items():
        if vars_list:
            max_corr = max([power_correlations[var] for var in vars_list])
            max_var = [var for var in vars_list if power_correlations[var] == max_corr][0]
            type_max_corr[var_type] = max_corr
            type_significance[var_type] = power_pvalues[max_var]
    
    if type_max_corr:
        types = list(type_max_corr.keys())
        values = list(type_max_corr.values())
        
        # 使用蓝色和橙色系颜色
        type_colors = ["#0C3177", '#046DB4', '#3FC2E7', '#FF8C69', "#B2E2F5"]
        bars = ax2.bar(types, values, color=type_colors[:len(types)], alpha=0.8)
        ax2.set_ylabel('Max Absolute Spearman Correlation', fontsize=12)
        ax2.set_title('Strongest Correlation by Variable Type (Spearman)', fontsize=14, fontweight='bold')
        ax2.tick_params(axis='x', rotation=45, labelsize=10)
        ax2.grid(True, alpha=0.3)
        
        # 添加数值标签和显著性标记
        for bar, var_type, value in zip(bars, types, values):
            p_val = type_significance[var_type]
            if p_val < 0.001:
                sig_mark = '***'
            elif p_val < 0.01:
                sig_mark = '**'
            elif p_val < 0.05:
                sig_mark = '*'
            else:
                sig_mark = ''
            
            ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                    f'{value:.3f}{sig_mark}', ha='center', va='bottom', 
                    fontsize=10, fontweight='bold')
    
    # 美化图表
    for ax in [ax1, ax2]:
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['left'].set_color('#666666')
        ax.spines['bottom'].set_color('#666666')
        ax.tick_params(colors='#333333')
    
    plt.tight_layout()
    plt.savefig(f'{output_dir}/power_correlation_analysis_with_significance.png', 
                dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    
    # 保存结果（含显著性信息）
    power_corr_df = pd.DataFrame({
        'variable': power_correlations.index,
        'abs_spearman_correlation': power_correlations.values,
        'spearman_correlation': [spearman_corr.loc[power_col, var] for var in power_correlations.index],
        'p_value': [power_pvalues[var] for var in power_correlations.index],
        'significance': ['***' if power_pvalues[var] < 0.001 else 
                        '**' if power_pvalues[var] < 0.01 else 
                        '*' if power_pvalues[var] < 0.05 else 'ns' 
                        for var in power_correlations.index]
    })
    power_corr_df.to_csv(f'{output_dir}/power_correlations_spearman_with_significance.csv', index=False)
    
    print(f"Top 5 variables most correlated with power (Spearman with significance):")
    for i, (var, abs_corr) in enumerate(power_correlations.head(5).items()):
        actual_corr = spearman_corr.loc[power_col, var]
        p_val = power_pvalues[var]
        sig = '***' if p_val < 0.001 else '**' if p_val < 0.01 else '*' if p_val < 0.05 else 'ns'
        print(f"  {i+1}. {var}: {actual_corr:.3f}{sig} (p={p_val:.3e})")
    
    return power_corr_df

def analyze_all_variables_correlations(corr_results, data_info, output_dir):
    """
    分析所有主要观测变量之间的相关性 - 基于Spearman相关系数
    为每个主要变量生成相关性排序图
    """
    print("\n=== All Variables Correlation Analysis (Spearman-based) ===")
    
    if corr_results is None:
        print("No correlation results available")
        return None
    
    spearman_corr = corr_results['spearman']
    spearman_pvals = corr_results['spearman_pvalues']
    df_clean = corr_results['data_clean']
    
    # 定义要分析的所有观测变量（排除power）
    main_variables = {}
    
    # 收集所有观测变量
    obs_vars = [col for col in df_clean.columns if col.startswith('obs_')]
    
    # 按类型和高度组织变量
    wind_speed_vars = sorted([col for col in obs_vars if 'wind_speed' in col])
    temp_vars = sorted([col for col in obs_vars if 'temperature' in col])
    humidity_vars = sorted([col for col in obs_vars if 'humidity' in col])
    density_vars = sorted([col for col in obs_vars if 'density' in col])
    wind_dir_sin_vars = sorted([col for col in df_clean.columns if 'wind_dir_sin' in col])
    wind_dir_cos_vars = sorted([col for col in df_clean.columns if 'wind_dir_cos' in col])
    
    # 添加所有风速变量
    for var in wind_speed_vars:
        height = var.replace('obs_wind_speed_', '').replace('m', '')
        main_variables[f'Wind Speed ({height}m)'] = var
    
    # 添加所有温度变量
    for var in temp_vars:
        height = var.replace('obs_temperature_', '').replace('m', '')
        main_variables[f'Temperature ({height}m)'] = var
    
    # 添加所有湿度变量
    for var in humidity_vars:
        height = var.replace('obs_humidity_', '').replace('m', '')
        main_variables[f'Humidity ({height}m)'] = var
    
    # 添加所有密度变量
    for var in density_vars:
        height = var.replace('obs_density_', '').replace('m', '')
        main_variables[f'Air Density ({height}m)'] = var
    
    # 添加所有风向sin分量
    for var in wind_dir_sin_vars:
        height = var.replace('obs_wind_dir_sin_', '').replace('m', '')
        main_variables[f'Wind Dir Sin ({height}m)'] = var
    
    # 添加所有风向cos分量
    for var in wind_dir_cos_vars:
        height = var.replace('obs_wind_dir_cos_', '').replace('m', '')
        main_variables[f'Wind Dir Cos ({height}m)'] = var
    
    print(f"Analyzing correlations for {len(main_variables)} observation variables across all heights")
    
    # 定义颜色系统
    def get_color_for_correlation(corr_val):
        """根据相关性值返回颜色"""
        if corr_val > 0:
            return '#016DB5'  # 蓝色 - 正相关
        else:
            return '#F45F31'  # 橙色 - 负相关
    
    # 为每个主要变量创建相关性图
    n_vars = len(main_variables)
    n_cols = 3  # 每行3个图
    n_rows = (n_vars + n_cols - 1) // n_cols
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(18, 6*n_rows))
    if n_rows == 1:
        axes = axes.reshape(1, -1)
    fig.suptitle('Correlation Analysis for All Observation Variables (Spearman-based)', 
                 fontsize=10, fontweight='bold')
    
    plot_idx = 0
    for var_name, var_col in main_variables.items():
        row = plot_idx // n_cols
        col = plot_idx % n_cols
        ax = axes[row, col]
        
        # 获取该变量与所有其他观测变量的相关性（排除power变量）
        other_obs_vars = [col for col in df_clean.columns 
                         if col.startswith('obs_') or 'wind_dir_' in col]
        other_obs_vars = [col for col in other_obs_vars if col != var_col]  # 移除自身
        
        # 计算与其他观测变量的相关性
        var_correlations = spearman_corr[var_col][other_obs_vars].abs().sort_values(ascending=False)
        var_pvalues = spearman_pvals[var_col]
        
        # 选择前15个最相关的变量
        top_correlations = var_correlations.head(15)
        
        # 创建条形图的标签和颜色
        bar_labels = []
        colors = []
        for var in top_correlations.index:
            actual_corr = spearman_corr[var_col][var]
            p_val = var_pvalues[var]
            
            # 添加显著性标记
            if p_val < 0.001:
                sig_mark = '***'
            elif p_val < 0.01:
                sig_mark = '**'
            elif p_val < 0.05:
                sig_mark = '*'
            else:
                sig_mark = ''
            
            # 简化变量名
            clean_var_name = var.replace('obs_', '').replace('_', ' ')
            bar_labels.append(f"{clean_var_name}{sig_mark}")
            colors.append(get_color_for_correlation(actual_corr))
        
        # 绘制水平条形图
        bars = ax.barh(range(len(top_correlations)), top_correlations.values, 
                      color=colors, alpha=0.8)
        
        ax.set_yticks(range(len(top_correlations)))
        ax.set_yticklabels(bar_labels, fontsize=9)
        ax.set_xlabel('Absolute Spearman Correlation Coefficient', fontsize=10)
        ax.set_title(f'{var_name} Correlation Ranking', fontsize=12, fontweight='bold')
        ax.grid(True, alpha=0.3)
        
        # 在条形上添加数值标签
        for i, (bar, corr_val) in enumerate(zip(bars, top_correlations.values)):
            ax.text(bar.get_width() + 0.01, bar.get_y() + bar.get_height()/2, 
                   f'{corr_val:.3f}', va='center', fontsize=8, fontweight='bold')
        
        # 美化图表
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['left'].set_color('#666666')
        ax.spines['bottom'].set_color('#666666')
        ax.tick_params(colors='#333333')
        
        plot_idx += 1
    
    # 隐藏多余的子图
    for i in range(plot_idx, n_rows * n_cols):
        row = i // n_cols
        col = i % n_cols
        axes[row, col].set_visible(False)
    
    plt.tight_layout()
    plt.savefig(f'{output_dir}/all_observation_variables_correlation_analysis.png', 
                dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    
    # 保存详细的相关性结果
    all_correlations_results = {}
    
    for var_name, var_col in main_variables.items():
        var_correlations = spearman_corr[var_col].abs().sort_values(ascending=False)
        var_correlations = var_correlations[var_correlations.index != var_col]
        var_pvalues = spearman_pvals[var_col]
        
        # 创建该变量的相关性DataFrame（只包含观测变量）
        other_obs_vars_for_df = [col for col in df_clean.columns 
                                if col.startswith('obs_') or 'wind_dir_' in col]
        other_obs_vars_for_df = [col for col in other_obs_vars_for_df if col != var_col]
        
        var_correlations_for_df = spearman_corr[var_col][other_obs_vars_for_df].abs().sort_values(ascending=False)
        
        var_corr_df = pd.DataFrame({
            'target_variable': var_col,
            'target_variable_name': var_name,
            'correlated_variable': var_correlations_for_df.index,
            'abs_spearman_correlation': var_correlations_for_df.values,
            'spearman_correlation': [spearman_corr.loc[var_col, var] for var in var_correlations_for_df.index],
            'p_value': [var_pvalues[var] for var in var_correlations_for_df.index],
            'significance': ['***' if var_pvalues[var] < 0.001 else 
                            '**' if var_pvalues[var] < 0.01 else 
                            '*' if var_pvalues[var] < 0.05 else 'ns' 
                            for var in var_correlations_for_df.index]
        })
        
        all_correlations_results[var_name] = var_corr_df
        
        # 保存单个变量的结果
        safe_name = var_name.replace(' ', '_').replace('(', '').replace(')', '').lower()
        var_corr_df.to_csv(f'{output_dir}/{safe_name}_correlations.csv', index=False)
        
        # 打印前5个相关性（排除power）
        print(f"\nTop 5 observation variables most correlated with {var_name}:")
        for i, (var, abs_corr) in enumerate(var_correlations.head(5).items()):
            actual_corr = spearman_corr.loc[var_col, var]
            p_val = var_pvalues[var]
            sig = '***' if p_val < 0.001 else '**' if p_val < 0.01 else '*' if p_val < 0.05 else 'ns'
            print(f"  {i+1}. {var}: {actual_corr:.3f}{sig} (p={p_val:.3e})")
    
    # 保存综合结果
    combined_df = pd.concat(all_correlations_results.values(), ignore_index=True)
    combined_df.to_csv(f'{output_dir}/all_observation_variables_correlations_combined.csv', index=False)
    
    print(f"\n✓ All observation variables correlation analysis completed!")
    print(f"✓ Analyzed {len(main_variables)} variables across all heights")
    print(f"✓ Individual results saved for each variable")
    print(f"✓ Combined results saved to: all_observation_variables_correlations_combined.csv")
    
    return all_correlations_results

def run_correlation_analysis(data_path, output_dir):
    """
    运行完整的相关性分析
    """
    import os
    
    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)
    
    print("=== 1.2.2 Observation Variables Correlation Analysis ===")
    print(f"Input data: {data_path}")
    print(f"Output directory: {output_dir}")
    
    # 加载数据
    df = pd.read_csv(data_path)
    print(f"Data shape: {df.shape}")
    
    # 准备观测数据
    data_info = prepare_observation_data(df)
    
    # 计算相关性矩阵
    corr_results = calculate_correlation_matrices(data_info)
    
    if corr_results is None:
        print("Correlation analysis failed due to insufficient data")
        return
    
    # 绘制相关性热图
    plot_correlation_heatmaps(corr_results, data_info, output_dir)
    
    # 功率相关性分析  
    power_correlations = analyze_power_correlations(corr_results, data_info, output_dir)
    
    # 保存相关性矩阵和显著性检验结果
    corr_results['pearson'].to_csv(f'{output_dir}/pearson_correlation_matrix.csv')
    corr_results['spearman'].to_csv(f'{output_dir}/spearman_correlation_matrix.csv')
    corr_results['pearson_pvalues'].to_csv(f'{output_dir}/pearson_pvalues_matrix.csv')
    corr_results['spearman_pvalues'].to_csv(f'{output_dir}/spearman_pvalues_matrix.csv')
    
    # 生成综合报告
    generate_correlation_report(corr_results, data_info, power_correlations, output_dir)
    
    print(f"\n✓ Correlation analysis with significance tests completed!")
    print(f"✓ All results saved to: {output_dir}")

def generate_correlation_report(corr_results, data_info, power_correlations, output_dir):
    """
    生成相关性分析报告
    """
    report = []
    report.append("=== 1.2.2 Observation Variables Correlation Analysis Report ===")
    report.append(f"Analysis time: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}")
    report.append(f"Variables analyzed: {len(corr_results['variable_names'])}")
    report.append(f"Valid samples: {len(corr_results['data_clean'])}")
    
    report.append("\n【Variable Categories】")
    report.append(f"Wind speed variables: {len(data_info['wind_speed_cols'])}")
    report.append(f"Wind direction variables: {len(data_info['wind_dir_processed'])} (converted to {len(data_info['wind_dir_processed'])*2} sin/cos components)")
    report.append(f"Temperature variables: {len(data_info['temp_cols'])}")
    report.append(f"Humidity variables: {len(data_info['humidity_cols'])}")
    report.append(f"Density variables: {len(data_info['density_cols'])}")
    report.append(f"Power variables: {len(data_info['power_cols'])}")
    
    report.append("\n【Wind Direction Processing】")
    report.append("Wind direction variables converted to sin/cos components:")
    for original_col, components in data_info['wind_dir_processed'].items():
        height = original_col.replace('obs_wind_direction_', '').replace('m', '')
        report.append(f"  {height}m: {components['sin']}, {components['cos']}")
    
    report.append("\n【Power Correlation Analysis (Spearman)】")
    if power_correlations is not None and len(power_correlations) > 0:
        report.append("Top 5 variables most correlated with power:")
        spearman_corr = corr_results['spearman']
        spearman_pvals = corr_results['spearman_pvalues']
        power_col = data_info['power_cols'][0]
        for i in range(min(5, len(power_correlations))):
            row = power_correlations.iloc[i]
            actual_corr = spearman_corr.loc[power_col, row['variable']]
            p_val = spearman_pvals.loc[power_col, row['variable']]
            sig = '***' if p_val < 0.001 else '**' if p_val < 0.01 else '*' if p_val < 0.05 else 'ns'
            report.append(f"  {i+1}. {row['variable']}: {actual_corr:.3f}{sig} (p={p_val:.3e})")
    else:
        report.append("No power correlation analysis results available")
    
    report.append("\n【Significance Testing】")
    report.append("Significance levels:")
    report.append("  *** p < 0.001 (highly significant)")
    report.append("  **  p < 0.01  (very significant)")
    report.append("  *   p < 0.05  (significant)")
    report.append("  ns  p ≥ 0.05  (not significant)")
    
    report.append("\n【Analysis Notes】")
    report.append("1. Only observation variables (obs_*) and power data were included")
    report.append("2. Forecast data (ec_*, gfs_*) were excluded from analysis")
    report.append("4. Missing values were removed before correlation calculation")
    report.append("5. Spearman correlation used for power analysis due to non-linear relationships")
    report.append("6. All correlations tested for statistical significance")
    
    # 保存报告
    report_text = '\n'.join(report)
    with open(f'{output_dir}/correlation_analysis_report.txt', 'w', encoding='utf-8') as f:
        f.write(report_text)
    
    print("Correlation analysis report generated")

def prepare_observation_data_with_debug(df, output_dir):
    """
    准备观测数据，处理风向的圆形特性 - 带调试输出版本
    """
    import os
    print("=== Prepare Observation Variables (Debug Version) ===")
    
    # 选择观测变量和功率变量
    obs_cols = [col for col in df.columns if col.startswith('obs_')]
    power_cols = [col for col in df.columns if 'power' in col.lower()]
    
    analysis_cols = obs_cols + power_cols
    
    print(f"Original observation variables: {len(obs_cols)}")
    print(f"Power variables: {len(power_cols)}")
    
    # 提取数据
    df_analysis = df[analysis_cols].copy()
    
    # 分类变量
    wind_speed_cols = [col for col in obs_cols if 'wind_speed' in col]
    wind_dir_cols = [col for col in obs_cols if 'wind_direction' in col]
    temp_cols = [col for col in obs_cols if 'temperature' in col]
    humidity_cols = [col for col in obs_cols if 'humidity' in col]
    density_cols = [col for col in obs_cols if 'density' in col]
    
    print(f"Wind speed variables: {len(wind_speed_cols)}")
    print(f"Wind direction variables: {len(wind_dir_cols)}")
    print(f"Temperature variables: {len(temp_cols)}")
    print(f"Humidity variables: {len(humidity_cols)}")
    print(f"Density variables: {len(density_cols)}")
    
    # 处理风向数据 - 转换为连续变量
    wind_dir_processed = {}
    
    # 创建调试数据框，包含原始风向和转换后的sin/cos分量
    debug_data = pd.DataFrame()
    
    for col in wind_dir_cols:
        print(f"\n=== Processing {col} ===")
        
        # 获取原始风向数据（去除缺失值以便观察）
        original_wind_dir = df_analysis[col].dropna()
        
        # 显示原始数据的统计信息
        print(f"Original data range: {original_wind_dir.min():.1f}° - {original_wind_dir.max():.1f}°")
        print(f"Original data mean: {original_wind_dir.mean():.1f}°")
        print(f"Sample original values: {original_wind_dir.head(10).tolist()}")
        
        # 方法1：当前代码的方法（直接转换）
        wind_dir_rad_direct = np.deg2rad(df_analysis[col])
        sin_direct = np.sin(wind_dir_rad_direct)
        cos_direct = np.cos(wind_dir_rad_direct)
        
        # 方法2：正确的气象角度转换方法
        math_angle = (90 - df_analysis[col] + 360) % 360
        wind_dir_rad_correct = np.deg2rad(math_angle)
        sin_correct = np.sin(wind_dir_rad_correct)
        cos_correct = np.cos(wind_dir_rad_correct)
        
        # 创建列名
        height = col.replace('obs_wind_direction_', '').replace('m', '')
        base_name = f"wind_dir_{height}m"
        
        # 添加到调试数据框
        debug_data[f"{base_name}_original"] = df_analysis[col]
        debug_data[f"{base_name}_sin_direct"] = sin_direct
        debug_data[f"{base_name}_cos_direct"] = cos_direct
        debug_data[f"{base_name}_sin_correct"] = sin_correct
        debug_data[f"{base_name}_cos_correct"] = cos_correct
        debug_data[f"{base_name}_math_angle"] = math_angle
        
        # 显示转换结果的统计信息
        print(f"Direct method - sin range: [{sin_direct.min():.3f}, {sin_direct.max():.3f}]")
        print(f"Direct method - cos range: [{cos_direct.min():.3f}, {cos_direct.max():.3f}]")
        print(f"Correct method - sin range: [{sin_correct.min():.3f}, {sin_correct.max():.3f}]")
        print(f"Correct method - cos range: [{cos_correct.min():.3f}, {cos_correct.max():.3f}]")
        
        # 使用当前代码的方法（保持一致性）
        sin_col = col.replace('wind_direction', 'wind_dir_sin')
        cos_col = col.replace('wind_direction', 'wind_dir_cos')
        
        df_analysis[sin_col] = sin_direct  # 使用当前方法
        df_analysis[cos_col] = cos_direct  # 使用当前方法
        
        wind_dir_processed[col] = {'sin': sin_col, 'cos': cos_col}
        
        print(f"Added to analysis: {sin_col}, {cos_col}")
    
    # 保存调试数据
    debug_file = os.path.join(output_dir, "wind_direction_conversion_debug.csv")
    debug_data.to_csv(debug_file, index=False)
    print(f"\n=== Debug data saved to: {debug_file} ===")
    
    # 创建典型示例对比
    create_wind_direction_examples(output_dir)
    
    # 移除原始风向数据
    df_analysis = df_analysis.drop(columns=wind_dir_cols)
    
    # 获取最终的分析变量列表
    final_cols = df_analysis.columns.tolist()
    
    print(f"Final analysis variables: {len(final_cols)}")
    
    return {
        'data': df_analysis,
        'wind_speed_cols': wind_speed_cols,
        'wind_dir_processed': wind_dir_processed,
        'temp_cols': temp_cols,
        'humidity_cols': humidity_cols,
        'density_cols': density_cols,
        'power_cols': power_cols,
        'final_cols': final_cols,
        'debug_data': debug_data
    }

def create_wind_direction_examples(output_dir):
    """
    创建风向转换的典型示例对比
    """
    import os
    
    # 创建典型风向示例
    examples = []
    test_angles = [0, 30, 45, 90, 135, 180, 225, 270, 315, 360]  # 典型角度
    
    for angle in test_angles:
        # 当前方法（直接转换）
        rad_direct = np.deg2rad(angle)
        sin_direct = np.sin(rad_direct)
        cos_direct = np.cos(rad_direct)
        
        # 正确方法（气象转数学）
        math_angle = (90 - angle + 360) % 360
        rad_correct = np.deg2rad(math_angle)
        sin_correct = np.sin(rad_correct)
        cos_correct = np.cos(rad_correct)
        
        # 方向描述
        if angle >= 337.5 or angle < 22.5:
            direction = "North (N)"
        elif angle >= 22.5 and angle < 67.5:
            direction = "Northeast (NE)"
        elif angle >= 67.5 and angle < 112.5:
            direction = "East (E)"
        elif angle >= 112.5 and angle < 157.5:
            direction = "Southeast (SE)"
        elif angle >= 157.5 and angle < 202.5:
            direction = "South (S)"
        elif angle >= 202.5 and angle < 247.5:
            direction = "Southwest (SW)"
        elif angle >= 247.5 and angle < 292.5:
            direction = "West (W)"
        elif angle >= 292.5 and angle < 337.5:
            direction = "Northwest (NW)"
        else:
            direction = "North (N)"
        
        examples.append({
            'meteorological_angle': angle,
            'direction': direction,
            'mathematical_angle': math_angle,
            'sin_direct_method': sin_direct,
            'cos_direct_method': cos_direct,
            'sin_correct_method': sin_correct,
            'cos_correct_method': cos_correct,
            'sin_physical_meaning_direct': get_physical_meaning(sin_direct, 'sin', 'direct'),
            'cos_physical_meaning_direct': get_physical_meaning(cos_direct, 'cos', 'direct'),
            'sin_physical_meaning_correct': get_physical_meaning(sin_correct, 'sin', 'correct'),
            'cos_physical_meaning_correct': get_physical_meaning(cos_correct, 'cos', 'correct')
        })
    
    # 保存示例
    examples_df = pd.DataFrame(examples)
    examples_file = os.path.join(output_dir, "wind_direction_conversion_examples.csv")
    examples_df.to_csv(examples_file, index=False)
    
    print(f"Conversion examples saved to: {examples_file}")
    
    # 打印部分示例到控制台
    print("\n=== Typical Wind Direction Conversion Examples ===")
    print("Angle | Direction | Direct Method (sin, cos) | Correct Method (sin, cos)")
    print("-" * 80)
    for example in examples[:8]:  # 显示前8个示例
        angle = example['meteorological_angle']
        direction = example['direction']
        sin_d = example['sin_direct_method']
        cos_d = example['cos_direct_method']
        sin_c = example['sin_correct_method']
        cos_c = example['cos_correct_method']
        print(f"{angle:3.0f}°  | {direction:12s} | ({sin_d:+.3f}, {cos_d:+.3f})     | ({sin_c:+.3f}, {cos_c:+.3f})")

def get_physical_meaning(value, func_type, method):
    """
    获取数值的物理含义描述
    """
    if abs(value) < 0.1:
        return "Neutral"
    elif func_type == 'sin':
        if method == 'correct':
            return "North" if value > 0 else "South"
        else:  # direct method
            return "Positive" if value > 0 else "Negative"
    else:  # cos
        if method == 'correct':
            return "East" if value > 0 else "West"
        else:  # direct method
            return "Positive" if value > 0 else "Negative"

# 修改主函数，使用调试版本
def run_correlation_analysis_with_debug(data_path, output_dir):
    """
    运行完整的相关性分析 - 带风向转换调试版本
    """
    import os
    
    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)
    
    print("=== 1.2.2 Observation Variables Correlation Analysis (Debug Version) ===")
    print(f"Input data: {data_path}")
    print(f"Output directory: {output_dir}")
    
    # 加载数据
    df = pd.read_csv(data_path)
    print(f"Data shape: {df.shape}")
    
    # 准备观测数据（调试版本）
    data_info = prepare_observation_data_with_debug(df, output_dir)
    
    # 计算相关性矩阵
    corr_results = calculate_correlation_matrices(data_info)
    
    if corr_results is None:
        print("Correlation analysis failed due to insufficient data")
        return
    
    # 绘制相关性热图
    plot_correlation_heatmaps(corr_results, data_info, output_dir)
    
    # 功率相关性分析  
    power_correlations = analyze_power_correlations(corr_results, data_info, output_dir)
    
    # 保存相关性矩阵和显著性检验结果
    corr_results['pearson'].to_csv(f'{output_dir}/pearson_correlation_matrix.csv')
    corr_results['spearman'].to_csv(f'{output_dir}/spearman_correlation_matrix.csv')
    corr_results['pearson_pvalues'].to_csv(f'{output_dir}/pearson_pvalues_matrix.csv')
    corr_results['spearman_pvalues'].to_csv(f'{output_dir}/spearman_pvalues_matrix.csv')
    
    # 生成综合报告
    generate_correlation_report(corr_results, data_info, power_correlations, output_dir)
    
    print(f"\n✓ Correlation analysis with wind direction debug completed!")
    print(f"✓ Check the following debug files:")
    print(f"  - wind_direction_conversion_debug.csv (actual data conversion)")
    print(f"  - wind_direction_conversion_examples.csv (typical examples)")
    print(f"✓ All results saved to: {output_dir}")

def run_correlation_analysis_extended(data_path, output_dir):
    """
    运行扩展的相关性分析 - 包含所有变量的相关性分析
    """
    import os
    
    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)
    
    print("=== Extended Correlation Analysis ===")
    print(f"Input data: {data_path}")
    print(f"Output directory: {output_dir}")
    
    # 加载数据
    df = pd.read_csv(data_path)
    print(f"Data shape: {df.shape}")
    
    # 准备观测数据
    data_info = prepare_observation_data(df)
    
    # 计算相关性矩阵
    corr_results = calculate_correlation_matrices(data_info)
    
    if corr_results is None:
        print("Correlation analysis failed due to insufficient data")
        return
    
    # 绘制相关性热图
    plot_correlation_heatmaps(corr_results, data_info, output_dir)
    
    # 功率相关性分析  
    power_correlations = analyze_power_correlations(corr_results, data_info, output_dir)
    
    # 所有变量相关性分析（新增）
    all_correlations = analyze_all_variables_correlations(corr_results, data_info, output_dir)
    
    # 保存相关性矩阵和显著性检验结果
    corr_results['pearson'].to_csv(f'{output_dir}/pearson_correlation_matrix.csv')
    corr_results['spearman'].to_csv(f'{output_dir}/spearman_correlation_matrix.csv')
    corr_results['pearson_pvalues'].to_csv(f'{output_dir}/pearson_pvalues_matrix.csv')
    corr_results['spearman_pvalues'].to_csv(f'{output_dir}/spearman_pvalues_matrix.csv')
    
    # 生成综合报告
    generate_correlation_report(corr_results, data_info, power_correlations, output_dir)
    
    print(f"\n✓ Extended correlation analysis completed!")
    print(f"✓ All results saved to: {output_dir}")

#使用示例
if __name__ == "__main__":
    # 设置路径
    data_path = "/Users/xiaxin/work/WindForecast_Project/01_Data/processed/imputed_data/changma_imputed_complete.csv"
    output_dir = "/Users/xiaxin/work/WindForecast_Project/03_Results/1_2_2_correlations"
    
    # # 运行分析
    # run_correlation_analysis(data_path, output_dir)
        

    # 运行调试版本分析
    #run_correlation_analysis_with_debug(data_path, output_dir)

    # 运行扩展分析
    run_correlation_analysis_extended(data_path, output_dir)