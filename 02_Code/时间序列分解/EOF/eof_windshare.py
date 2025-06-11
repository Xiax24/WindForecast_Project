import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from scipy import stats
import seaborn as sns
import warnings
import os
from datetime import datetime
warnings.filterwarnings('ignore')

# Set paths
data_path = '/Users/xiaxin/work/WindForecast_Project/01_Data/processed/imputed_data/changma_clean.csv'
results_path = '/Users/xiaxin/work/WindForecast_Project/03_Results/08时间序列分解/EOF_Physical_Analysis'

# Create results directory if it doesn't exist
os.makedirs(results_path, exist_ok=True)

# Load data
print("Loading data for EOF physical meaning analysis...")
df = pd.read_csv(data_path)

# Convert datetime column
df['datetime'] = pd.to_datetime(df['datetime'])
df.set_index('datetime', inplace=True)

# Define wind speed variables and heights
wind_variables = ['obs_wind_speed_10m', 'obs_wind_speed_30m', 'obs_wind_speed_50m', 'obs_wind_speed_70m']
heights = [10, 30, 50, 70]

print(f"Data shape: {df.shape}")
print(f"Date range: {df.index.min()} to {df.index.max()}")

# Extract wind speed data
wind_data = df[wind_variables].copy()
wind_data_clean = wind_data.dropna()

print(f"Data points after cleaning: {len(wind_data_clean)}")

def calculate_wind_shear_coefficient(v1, v2, h1, h2):
    """Calculate wind shear coefficient alpha"""
    valid_mask = (v1 > 0.1) & (v2 > 0.1)
    alpha = np.full(len(v1), np.nan)
    valid_indices = valid_mask
    alpha[valid_indices] = np.log(v2[valid_indices] / v1[valid_indices]) / np.log(h2 / h1)
    return alpha

def calculate_wind_shear_magnitude(wind_data):
    """Calculate actual wind shear magnitude (not just coefficient)"""
    # Wind speed difference between 70m and 10m
    wind_shear_70_10 = wind_data['obs_wind_speed_70m'] - wind_data['obs_wind_speed_10m']
    # Relative wind shear
    relative_shear = wind_shear_70_10 / wind_data['obs_wind_speed_10m']
    return wind_shear_70_10, relative_shear

def perform_detailed_eof_analysis(data):
    """Perform detailed EOF analysis with physical interpretation"""
    print("Performing detailed EOF analysis...")
    
    # Standardize the data
    data_std = (data - data.mean()) / data.std()
    
    # Perform PCA
    pca = PCA(n_components=4)
    time_coefficients = pca.fit_transform(data_std)
    
    # EOF patterns and other results
    eof_patterns = pca.components_
    explained_variance = pca.explained_variance_ratio_
    
    # Calculate loadings (correlations between original variables and PCs)
    loadings = pca.components_.T * np.sqrt(pca.explained_variance_)
    
    results = {
        'time_coefficients': time_coefficients,
        'eof_patterns': eof_patterns,
        'explained_variance': explained_variance,
        'loadings': loadings,
        'data_std': data_std,
        'pca_object': pca,
        'time_index': data.index
    }
    
    return results

def interpret_eof_physical_meaning(eof_patterns, heights):
    """Provide detailed physical interpretation of EOF patterns"""
    interpretations = []
    
    for i, pattern in enumerate(eof_patterns):
        interp = {
            'mode': i + 1,
            'pattern_values': pattern,
            'height_correlation': np.corrcoef(pattern, heights)[0, 1]
        }
        
        # Analyze pattern characteristics
        if np.all(pattern > 0) or np.all(pattern < 0):
            interp['type'] = 'uniform'
            interp['meaning'] = 'Overall wind speed intensity change (all heights vary together)'
            interp['physics'] = 'Synoptic-scale weather systems or diurnal heating effects'
        
        elif interp['height_correlation'] > 0.5:
            interp['type'] = 'positive_shear'
            interp['meaning'] = 'Upper levels increase more than lower levels'
            interp['physics'] = 'Stable stratification or decoupling of upper/lower atmosphere'
        
        elif interp['height_correlation'] < -0.5:
            interp['type'] = 'negative_shear'
            interp['meaning'] = 'Lower levels increase more than upper levels'
            interp['physics'] = 'Unstable mixing or surface heating effects'
        
        else:
            # Check for vertical structure patterns
            if pattern[0] * pattern[-1] < 0:  # Opposite signs at 10m and 70m
                interp['type'] = 'vertical_oscillation'
                interp['meaning'] = 'Vertical wind shear variation (opposite changes at different levels)'
                interp['physics'] = 'Atmospheric boundary layer dynamics or wind shear events'
            else:
                interp['type'] = 'complex'
                interp['meaning'] = 'Complex vertical structure'
                interp['physics'] = 'Multiple competing atmospheric processes'
        
        interpretations.append(interp)
    
    return interpretations

def analyze_temporal_patterns(eof_results):
    """Analyze temporal patterns in EOF coefficients"""
    time_coefficients = eof_results['time_coefficients']
    time_index = eof_results['time_index']
    
    # Create DataFrame with time information
    temporal_df = pd.DataFrame({
        'PC1': time_coefficients[:, 0],
        'PC2': time_coefficients[:, 1],
        'PC3': time_coefficients[:, 2],
        'PC4': time_coefficients[:, 3],
        'hour': time_index.hour,
        'month': time_index.month,
        'season': time_index.month.map({
            12: 'Winter', 1: 'Winter', 2: 'Winter',
            3: 'Spring', 4: 'Spring', 5: 'Spring',
            6: 'Summer', 7: 'Summer', 8: 'Summer',
            9: 'Autumn', 10: 'Autumn', 11: 'Autumn'
        }),
        'is_daytime': ((time_index.hour >= 6) & (time_index.hour <= 18)).astype(int)
    }, index=time_index)
    
    # Calculate correlations with time variables
    correlations = {}
    for pc in ['PC1', 'PC2', 'PC3', 'PC4']:
        correlations[pc] = {
            'hour_correlation': np.corrcoef(temporal_df[pc], temporal_df['hour'])[0, 1],
            'month_correlation': np.corrcoef(temporal_df[pc], temporal_df['month'])[0, 1],
            'daytime_correlation': np.corrcoef(temporal_df[pc], temporal_df['is_daytime'])[0, 1]
        }
    
    # Day vs night comparison
    day_night_stats = {}
    for pc in ['PC1', 'PC2', 'PC3', 'PC4']:
        day_values = temporal_df[temporal_df['is_daytime'] == 1][pc]
        night_values = temporal_df[temporal_df['is_daytime'] == 0][pc]
        
        if len(day_values) > 30 and len(night_values) > 30:
            t_stat, p_val = stats.ttest_ind(day_values, night_values)
            day_night_stats[pc] = {
                'day_mean': day_values.mean(),
                'night_mean': night_values.mean(),
                'day_std': day_values.std(),
                'night_std': night_values.std(),
                't_statistic': t_stat,
                'p_value': p_val,
                'significant': p_val < 0.05,
                'day_higher': day_values.mean() > night_values.mean()
            }
    
    return temporal_df, correlations, day_night_stats

def analyze_eof_vs_atmospheric_conditions(eof_results, wind_data, alpha_10_70):
    """Analyze relationship between EOF and atmospheric conditions"""
    time_coefficients = eof_results['time_coefficients']
    time_index = eof_results['time_index']
    
    # Calculate wind shear magnitude
    wind_shear_70_10, relative_shear = calculate_wind_shear_magnitude(wind_data)
    
    # Create analysis DataFrame
    analysis_df = pd.DataFrame({
        'PC1': time_coefficients[:, 0],
        'PC2': time_coefficients[:, 1],
        'PC3': time_coefficients[:, 2],
        'PC4': time_coefficients[:, 3],
        'alpha_10_70': alpha_10_70,
        'wind_shear_magnitude': wind_shear_70_10,
        'relative_shear': relative_shear,
        'wind_10m': wind_data['obs_wind_speed_10m'],
        'wind_70m': wind_data['obs_wind_speed_70m'],
        'wind_ratio_70_10': wind_data['obs_wind_speed_70m'] / wind_data['obs_wind_speed_10m']
    }, index=time_index)
    
    # Remove invalid data
    analysis_df = analysis_df.dropna()
    
    # Calculate correlations
    correlations = {}
    for pc in ['PC1', 'PC2', 'PC3', 'PC4']:
        correlations[pc] = {
            'alpha_correlation': np.corrcoef(analysis_df[pc], analysis_df['alpha_10_70'])[0, 1],
            'shear_magnitude_correlation': np.corrcoef(analysis_df[pc], analysis_df['wind_shear_magnitude'])[0, 1],
            'relative_shear_correlation': np.corrcoef(analysis_df[pc], analysis_df['relative_shear'])[0, 1],
            'wind_ratio_correlation': np.corrcoef(analysis_df[pc], analysis_df['wind_ratio_70_10'])[0, 1]
        }
    
    return analysis_df, correlations

def analyze_power_prediction_efficiency(eof_results, wind_data, power_data=None):
    """Analyze how different EOF states affect wind-power relationships"""
    if power_data is None or power_data.isna().all():
        print("Power data not available for analysis")
        return None
    
    time_coefficients = eof_results['time_coefficients']
    time_index = eof_results['time_index']
    
    # Align data
    common_index = time_index.intersection(power_data.index)
    pc1_aligned = pd.Series(time_coefficients[:, 0], index=time_index).reindex(common_index)
    power_aligned = power_data.reindex(common_index)
    wind_10m_aligned = wind_data['obs_wind_speed_10m'].reindex(common_index)
    wind_70m_aligned = wind_data['obs_wind_speed_70m'].reindex(common_index)
    
    # Remove NaN values
    valid_mask = ~(pc1_aligned.isna() | power_aligned.isna() | wind_10m_aligned.isna() | wind_70m_aligned.isna())
    
    pc1_valid = pc1_aligned[valid_mask]
    power_valid = power_aligned[valid_mask]
    wind_10m_valid = wind_10m_aligned[valid_mask]
    wind_70m_valid = wind_70m_aligned[valid_mask]
    
    if len(pc1_valid) < 100:
        print("Not enough valid data for power analysis")
        return None
    
    # Divide into high and low PC1 periods
    pc1_median = pc1_valid.median()
    high_pc1_mask = pc1_valid > pc1_median
    low_pc1_mask = pc1_valid <= pc1_median
    
    # Calculate correlations for each period
    results = {}
    
    # High PC1 periods
    if high_pc1_mask.sum() > 50:
        high_power = power_valid[high_pc1_mask]
        high_wind_10m = wind_10m_valid[high_pc1_mask]
        high_wind_70m = wind_70m_valid[high_pc1_mask]
        
        results['high_pc1'] = {
            'power_wind10m_corr': np.corrcoef(high_power, high_wind_10m)[0, 1],
            'power_wind70m_corr': np.corrcoef(high_power, high_wind_70m)[0, 1],
            'count': high_pc1_mask.sum()
        }
    
    # Low PC1 periods
    if low_pc1_mask.sum() > 50:
        low_power = power_valid[low_pc1_mask]
        low_wind_10m = wind_10m_valid[low_pc1_mask]
        low_wind_70m = wind_70m_valid[low_pc1_mask]
        
        results['low_pc1'] = {
            'power_wind10m_corr': np.corrcoef(low_power, low_wind_10m)[0, 1],
            'power_wind70m_corr': np.corrcoef(low_power, low_wind_70m)[0, 1],
            'count': low_pc1_mask.sum()
        }
    
    return results

def create_comprehensive_visualization(eof_results, temporal_analysis, atmospheric_analysis, 
                                     eof_interpretations, day_night_stats, correlations):
    """Create comprehensive visualization of EOF physical meaning"""
    
    fig = plt.figure(figsize=(24, 16))
    
    # Plot 1: EOF Patterns with Physical Interpretation
    ax1 = plt.subplot(3, 4, 1)
    
    eof_patterns = eof_results['eof_patterns']
    explained_variance = eof_results['explained_variance']
    
    colors = ['red', 'blue', 'green', 'orange']
    for i in range(4):
        ax1.plot(eof_patterns[i], heights, 'o-', color=colors[i], linewidth=2, 
                markersize=8, label=f'EOF{i+1} ({explained_variance[i]*100:.1f}%)')
    
    ax1.axvline(x=0, color='black', linestyle='--', alpha=0.5)
    ax1.set_title('EOF Patterns (Physical Structure)', fontsize=14, fontweight='bold')
    ax1.set_xlabel('EOF Loading')
    ax1.set_ylabel('Height (m)')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: PC1 vs Wind Shear Magnitude
    ax2 = plt.subplot(3, 4, 2)
    
    analysis_df = atmospheric_analysis[0]
    pc1_shear_corr = atmospheric_analysis[1]['PC1']['shear_magnitude_correlation']
    
    ax2.scatter(analysis_df['wind_shear_magnitude'], analysis_df['PC1'], 
               alpha=0.3, s=10, color='blue')
    ax2.set_xlabel('Wind Shear Magnitude (70m - 10m) [m/s]')
    ax2.set_ylabel('PC1 Values')
    ax2.set_title(f'PC1 vs Wind Shear Magnitude\nCorrelation: {pc1_shear_corr:.4f}', 
                 fontsize=14, fontweight='bold')
    ax2.grid(True, alpha=0.3)
    
    # Add regression line
    if abs(pc1_shear_corr) > 0.1:
        z = np.polyfit(analysis_df['wind_shear_magnitude'], analysis_df['PC1'], 1)
        p = np.poly1d(z)
        x_line = np.linspace(analysis_df['wind_shear_magnitude'].min(), 
                           analysis_df['wind_shear_magnitude'].max(), 100)
        ax2.plot(x_line, p(x_line), "r--", alpha=0.8)
    
    # Plot 3: Diurnal Patterns of PC1
    ax3 = plt.subplot(3, 4, 3)
    
    temporal_df = temporal_analysis[0]
    hourly_pc1 = temporal_df.groupby('hour')['PC1'].mean()
    hourly_pc1_std = temporal_df.groupby('hour')['PC1'].std()
    
    ax3.plot(hourly_pc1.index, hourly_pc1.values, 'bo-', linewidth=2, markersize=6)
    ax3.fill_between(hourly_pc1.index, 
                    hourly_pc1.values - hourly_pc1_std.values,
                    hourly_pc1.values + hourly_pc1_std.values, 
                    alpha=0.3)
    ax3.axvspan(6, 18, alpha=0.2, color='yellow', label='Daytime')
    ax3.set_xlabel('Hour of Day')
    ax3.set_ylabel('PC1 Mean Value')
    ax3.set_title('PC1 Diurnal Pattern', fontsize=14, fontweight='bold')
    ax3.set_xticks(range(0, 24, 4))
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # Plot 4: Day vs Night PC1 Comparison
    ax4 = plt.subplot(3, 4, 4)
    
    if 'PC1' in day_night_stats:
        stats = day_night_stats['PC1']
        categories = ['Day', 'Night']
        means = [stats['day_mean'], stats['night_mean']]
        stds = [stats['day_std'], stats['night_std']]
        
        bars = ax4.bar(categories, means, yerr=stds, capsize=5, 
                      color=['gold', 'navy'], alpha=0.7)
        ax4.set_ylabel('PC1 Mean Value')
        ax4.set_title(f'Day vs Night PC1\np = {stats["p_value"]:.4f}', 
                     fontsize=14, fontweight='bold')
        ax4.grid(True, alpha=0.3)
        
        # Add significance annotation
        if stats['significant']:
            ax4.text(0.5, max(means) + max(stds) * 1.1, 'Significant*', 
                    ha='center', fontweight='bold', color='red')
    
    # Plot 5: PC correlations with atmospheric variables
    ax5 = plt.subplot(3, 4, 5)
    
    atm_correlations = atmospheric_analysis[1]
    pc_names = ['PC1', 'PC2', 'PC3', 'PC4']
    variables = ['alpha_correlation', 'shear_magnitude_correlation', 'relative_shear_correlation']
    var_labels = ['Alpha Coef', 'Shear Magnitude', 'Relative Shear']
    
    corr_matrix = np.array([[atm_correlations[pc][var] for var in variables] for pc in pc_names])
    
    im = ax5.imshow(corr_matrix, cmap='RdBu_r', vmin=-0.5, vmax=0.5, aspect='auto')
    ax5.set_xticks(range(len(var_labels)))
    ax5.set_xticklabels(var_labels, rotation=45)
    ax5.set_yticks(range(len(pc_names)))
    ax5.set_yticklabels(pc_names)
    ax5.set_title('PC Correlations with\nAtmospheric Variables', fontsize=14, fontweight='bold')
    
    # Add correlation values as text
    for i in range(len(pc_names)):
        for j in range(len(variables)):
            text = ax5.text(j, i, f'{corr_matrix[i, j]:.3f}', 
                           ha="center", va="center", color="black", fontweight='bold')
    
    plt.colorbar(im, ax=ax5)
    
    # Plot 6: Seasonal PC1 patterns
    ax6 = plt.subplot(3, 4, 6)
    
    seasonal_pc1 = temporal_df.groupby('season')['PC1'].mean()
    seasonal_pc1_std = temporal_df.groupby('season')['PC1'].std()
    
    bars = ax6.bar(seasonal_pc1.index, seasonal_pc1.values, 
                  yerr=seasonal_pc1_std.values, capsize=5, 
                  color=['lightblue', 'lightgreen', 'orange', 'brown'], alpha=0.7)
    ax6.set_ylabel('PC1 Mean Value')
    ax6.set_title('PC1 Seasonal Patterns', fontsize=14, fontweight='bold')
    ax6.tick_params(axis='x', rotation=45)
    ax6.grid(True, alpha=0.3)
    
    # Plot 7-8: PC2 and PC3 patterns for comparison
    for idx, pc_num in enumerate([2, 3]):
        ax = plt.subplot(3, 4, 7 + idx)
        
        pc_values = eof_results['time_coefficients'][:, pc_num - 1]
        hourly_pc = temporal_df.groupby('hour')[f'PC{pc_num}'].mean()
        
        ax.plot(hourly_pc.index, hourly_pc.values, 'o-', linewidth=2, markersize=6,
               color=colors[pc_num - 1])
        ax.axvspan(6, 18, alpha=0.2, color='yellow')
        ax.set_xlabel('Hour of Day')
        ax.set_ylabel(f'PC{pc_num} Mean Value')
        ax.set_title(f'PC{pc_num} Diurnal Pattern\n({explained_variance[pc_num-1]*100:.1f}% variance)', 
                    fontsize=12, fontweight='bold')
        ax.set_xticks(range(0, 24, 4))
        ax.grid(True, alpha=0.3)
    
    # Plot 9: Physical Interpretation Summary
    ax9 = plt.subplot(3, 4, 9)
    ax9.axis('off')
    
    interpretation_text = "EOF Physical Interpretation:\n\n"
    
    for i, interp in enumerate(eof_interpretations):
        interpretation_text += f"EOF {interp['mode']} ({explained_variance[i]*100:.1f}% var):\n"
        interpretation_text += f"Type: {interp['type']}\n"
        interpretation_text += f"Meaning: {interp['meaning']}\n"
        interpretation_text += f"Physics: {interp['physics']}\n\n"
    
    ax9.text(0.02, 0.98, interpretation_text, transform=ax9.transAxes, fontsize=10,
             verticalalignment='top', fontfamily='monospace',
             bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.8))
    
    # Plot 10: Key Correlations Summary
    ax10 = plt.subplot(3, 4, 10)
    ax10.axis('off')
    
    correlation_text = "Key Correlations:\n\n"
    
    # Temporal correlations
    temp_corrs = temporal_analysis[1]
    correlation_text += "Temporal Patterns:\n"
    for pc in ['PC1', 'PC2']:
        correlation_text += f"{pc} vs Hour: {temp_corrs[pc]['hour_correlation']:.3f}\n"
        correlation_text += f"{pc} vs Daytime: {temp_corrs[pc]['daytime_correlation']:.3f}\n"
    
    correlation_text += "\nAtmospheric Correlations:\n"
    for pc in ['PC1', 'PC2']:
        correlation_text += f"{pc} vs Alpha: {atm_correlations[pc]['alpha_correlation']:.3f}\n"
        correlation_text += f"{pc} vs Shear: {atm_correlations[pc]['shear_magnitude_correlation']:.3f}\n"
    
    ax10.text(0.02, 0.98, correlation_text, transform=ax10.transAxes, fontsize=11,
             verticalalignment='top', fontfamily='monospace',
             bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8))
    
    # Plot 11-12: Additional analysis spaces
    ax11 = plt.subplot(3, 4, 11)
    
    # PC1 vs Wind Speed Ratio
    ratio_corr = atm_correlations['PC1']['wind_ratio_correlation']
    ax11.scatter(analysis_df['wind_ratio_70_10'], analysis_df['PC1'], 
                alpha=0.3, s=10, color='green')
    ax11.set_xlabel('Wind Speed Ratio (70m/10m)')
    ax11.set_ylabel('PC1 Values')
    ax11.set_title(f'PC1 vs Wind Speed Ratio\nCorrelation: {ratio_corr:.4f}', 
                  fontsize=12, fontweight='bold')
    ax11.grid(True, alpha=0.3)
    
    ax12 = plt.subplot(3, 4, 12)
    
    # Summary of key findings
    ax12.axis('off')
    
    # Calculate key statistics
    pc1_hour_corr = temp_corrs['PC1']['hour_correlation']
    pc1_alpha_corr = atm_correlations['PC1']['alpha_correlation']
    
    findings_text = "KEY FINDINGS:\n\n"
    
    # Determine if PC1 represents wind intensity or shear
    if abs(atm_correlations['PC1']['shear_magnitude_correlation']) < 0.3:
        findings_text += "✓ PC1 = WIND INTENSITY mode\n"
        findings_text += "  (not wind shear mode)\n\n"
    else:
        findings_text += "✓ PC1 = WIND SHEAR mode\n\n"
    
    # Diurnal pattern
    if abs(pc1_hour_corr) > 0.1:
        findings_text += f"✓ Strong diurnal pattern\n"
        findings_text += f"  (r = {pc1_hour_corr:.3f})\n\n"
    
    # Stability relationship
    if abs(pc1_alpha_corr) > 0.1:
        findings_text += f"✓ Related to stability\n"
        findings_text += f"  (r = {pc1_alpha_corr:.3f})\n\n"
    else:
        findings_text += f"✗ Weak stability relationship\n"
        findings_text += f"  (r = {pc1_alpha_corr:.3f})\n\n"
    
    if 'PC1' in day_night_stats and day_night_stats['PC1']['significant']:
        findings_text += "✓ Significant day/night difference\n"
    
    ax12.text(0.02, 0.98, findings_text, transform=ax12.transAxes, fontsize=12,
             verticalalignment='top', fontweight='bold',
             bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.8))
    
    plt.tight_layout()
    return fig

# Main analysis workflow
print("\n" + "="*80)
print("EOF PHYSICAL MEANING ANALYSIS")
print("="*80)

# Step 1: Calculate wind shear coefficient
print("\nStep 1: Calculating wind parameters...")
alpha_10_70 = calculate_wind_shear_coefficient(
    wind_data_clean['obs_wind_speed_10m'].values,
    wind_data_clean['obs_wind_speed_70m'].values,
    10, 70
)

# Step 2: Perform detailed EOF analysis
print("\nStep 2: Performing detailed EOF analysis...")
eof_results = perform_detailed_eof_analysis(wind_data_clean)

# Step 3: Interpret EOF patterns physically
print("\nStep 3: Interpreting EOF patterns...")
eof_interpretations = interpret_eof_physical_meaning(eof_results['eof_patterns'], heights)

print("EOF Pattern Interpretations:")
for interp in eof_interpretations:
    print(f"  EOF {interp['mode']}: {interp['meaning']}")
    print(f"    Physics: {interp['physics']}")

# Step 4: Analyze temporal patterns
print("\nStep 4: Analyzing temporal patterns...")
temporal_analysis = analyze_temporal_patterns(eof_results)
temporal_df, temp_correlations, day_night_stats = temporal_analysis

print("\nTemporal correlations:")
for pc, corrs in temp_correlations.items():
    print(f"  {pc}: hour={corrs['hour_correlation']:.3f}, daytime={corrs['daytime_correlation']:.3f}")

print("\nDay vs Night statistics:")
for pc, stats in day_night_stats.items():
    print(f"  {pc}: day={stats['day_mean']:.3f}, night={stats['night_mean']:.3f}, p={stats['p_value']:.4f}")

# Step 5: Analyze atmospheric conditions relationship
print("\nStep 5: Analyzing atmospheric conditions...")
atmospheric_analysis = analyze_eof_vs_atmospheric_conditions(eof_results, wind_data_clean, alpha_10_70)
analysis_df, atm_correlations = atmospheric_analysis

print("\nAtmospheric correlations:")
for pc, corrs in atm_correlations.items():
    print(f"  {pc}: alpha={corrs['alpha_correlation']:.3f}, shear_mag={corrs['shear_magnitude_correlation']:.3f}")

# Step 6: Analyze power prediction efficiency (if power data available)
print("\nStep 6: Analyzing power relationships...")
power_data = df['power'] if 'power' in df.columns else None
power_analysis = analyze_power_prediction_efficiency(eof_results, wind_data_clean, power_data)

if power_analysis:
    print("Power prediction efficiency:")
    for period, results in power_analysis.items():
        print(f"  {period}: 10m_corr={results['power_wind10m_corr']:.3f}, 70m_corr={results['power_wind70m_corr']:.3f}")

# Step 7: Create comprehensive visualization
print("\nStep 7: Creating comprehensive visualization...")
fig = create_comprehensive_visualization(
    eof_results, temporal_analysis, atmospheric_analysis, 
    eof_interpretations, day_night_stats, temp_correlations
)

plot_path = os.path.join(results_path, 'EOF_Physical_Analysis_Complete.png')
fig.savefig(plot_path, dpi=300, bbox_inches='tight')
plt.show()

# Step 8: Save detailed results
print("\nStep 8: Saving detailed results...")

# Save EOF interpretations
interp_df = pd.DataFrame(eof_interpretations)
interp_path = os.path.join(results_path, 'EOF_Physical_Interpretations.csv')
interp_df.to_csv(interp_path, index=False)

# Save temporal analysis results
temporal_path = os.path.join(results_path, 'EOF_Temporal_Analysis.csv')
temporal_df.to_csv(temporal_path)

# Save day/night statistics
day_night_df = pd.DataFrame(day_night_stats).T
day_night_path = os.path.join(results_path, 'EOF_Day_Night_Stats.csv')
day_night_df.to_csv(day_night_path)

# Save atmospheric correlations
atm_corr_df = pd.DataFrame(atm_correlations).T
atm_corr_path = os.path.join(results_path, 'EOF_Atmospheric_Correlations.csv')
atm_corr_df.to_csv(atm_corr_path)

# Save power analysis if available
if power_analysis:
    power_df = pd.DataFrame(power_analysis).T
    power_path = os.path.join(results_path, 'EOF_Power_Analysis.csv')
    power_df.to_csv(power_path)

# Create comprehensive summary
print("\nStep 9: Creating comprehensive summary...")

summary_results = {
    'Analysis_Type': 'EOF_Physical_Meaning_Analysis',
    'Total_Data_Points': len(wind_data_clean),
    'EOF1_Variance_Explained_%': eof_results['explained_variance'][0] * 100,
    'EOF1_Type': eof_interpretations[0]['type'],
    'EOF1_Physical_Meaning': eof_interpretations[0]['meaning'],
    'PC1_Hour_Correlation': temp_correlations['PC1']['hour_correlation'],
    'PC1_Daytime_Correlation': temp_correlations['PC1']['daytime_correlation'],
    'PC1_Alpha_Correlation': atm_correlations['PC1']['alpha_correlation'],
    'PC1_Shear_Magnitude_Correlation': atm_correlations['PC1']['shear_magnitude_correlation'],
    'PC1_Wind_Ratio_Correlation': atm_correlations['PC1']['wind_ratio_correlation'],
}

# Add day/night statistics
if 'PC1' in day_night_stats:
    summary_results.update({
        'PC1_Day_Mean': day_night_stats['PC1']['day_mean'],
        'PC1_Night_Mean': day_night_stats['PC1']['night_mean'],
        'PC1_Day_Night_PValue': day_night_stats['PC1']['p_value'],
        'PC1_Day_Night_Significant': day_night_stats['PC1']['significant'],
        'PC1_Day_Higher': day_night_stats['PC1']['day_higher']
    })

# Add power analysis results
if power_analysis:
    if 'high_pc1' in power_analysis:
        summary_results.update({
            'High_PC1_Power_Wind10m_Corr': power_analysis['high_pc1']['power_wind10m_corr'],
            'High_PC1_Power_Wind70m_Corr': power_analysis['high_pc1']['power_wind70m_corr']
        })
    if 'low_pc1' in power_analysis:
        summary_results.update({
            'Low_PC1_Power_Wind10m_Corr': power_analysis['low_pc1']['power_wind10m_corr'],
            'Low_PC1_Power_Wind70m_Corr': power_analysis['low_pc1']['power_wind70m_corr']
        })

summary_df = pd.DataFrame([summary_results])
summary_path = os.path.join(results_path, 'EOF_Physical_Analysis_Summary.csv')
summary_df.to_csv(summary_path, index=False)

# Generate conclusions
print("\n" + "="*80)
print("PHYSICAL ANALYSIS CONCLUSIONS")
print("="*80)

print(f"\n1. EOF MODE 1 PHYSICAL MEANING:")
print(f"   Type: {eof_interpretations[0]['type']}")
print(f"   Meaning: {eof_interpretations[0]['meaning']}")
print(f"   Variance explained: {eof_results['explained_variance'][0]*100:.1f}%")

print(f"\n2. TEMPORAL PATTERNS:")
pc1_hour_corr = temp_correlations['PC1']['hour_correlation']
pc1_daytime_corr = temp_correlations['PC1']['daytime_correlation']

if abs(pc1_hour_corr) > 0.15:
    print(f"   ✓ STRONG diurnal pattern (r = {pc1_hour_corr:.3f})")
    if pc1_hour_corr > 0:
        print(f"     → PC1 increases during afternoon/evening")
    else:
        print(f"     → PC1 increases during night/early morning")
else:
    print(f"   ✗ WEAK diurnal pattern (r = {pc1_hour_corr:.3f})")

if 'PC1' in day_night_stats:
    stats = day_night_stats['PC1']
    if stats['significant']:
        higher_period = "DAY" if stats['day_higher'] else "NIGHT"
        print(f"   ✓ Significant day/night difference (p = {stats['p_value']:.4f})")
        print(f"     → PC1 is higher during {higher_period}")
    else:
        print(f"   ✗ No significant day/night difference (p = {stats['p_value']:.4f})")

print(f"\n3. ATMOSPHERIC STABILITY RELATIONSHIP:")
pc1_alpha_corr = atm_correlations['PC1']['alpha_correlation']
pc1_shear_corr = atm_correlations['PC1']['shear_magnitude_correlation']

if abs(pc1_alpha_corr) > 0.15:
    print(f"   ✓ SIGNIFICANT correlation with wind shear coefficient (r = {pc1_alpha_corr:.3f})")
    if pc1_alpha_corr > 0:
        print(f"     → PC1 increases with STABLE conditions (higher α)")
    else:
        print(f"     → PC1 increases with UNSTABLE conditions (lower α)")
else:
    print(f"   ✗ WEAK correlation with wind shear coefficient (r = {pc1_alpha_corr:.3f})")

if abs(pc1_shear_corr) > 0.15:
    print(f"   ✓ SIGNIFICANT correlation with wind shear magnitude (r = {pc1_shear_corr:.3f})")
    if pc1_shear_corr > 0:
        print(f"     → PC1 increases with larger wind shear (70m-10m difference)")
    else:
        print(f"     → PC1 decreases with larger wind shear")
else:
    print(f"   ✗ WEAK correlation with wind shear magnitude (r = {pc1_shear_corr:.3f})")

print(f"\n4. WIND SPEED PROFILE RELATIONSHIP:")
pc1_ratio_corr = atm_correlations['PC1']['wind_ratio_correlation']

if abs(pc1_ratio_corr) > 0.15:
    print(f"   ✓ SIGNIFICANT correlation with wind speed ratio 70m/10m (r = {pc1_ratio_corr:.3f})")
    if pc1_ratio_corr > 0:
        print(f"     → PC1 increases when 70m wind is much higher than 10m wind")
    else:
        print(f"     → PC1 increases when wind speeds are more uniform vertically")
else:
    print(f"   ✗ WEAK correlation with wind speed ratio (r = {pc1_ratio_corr:.3f})")

print(f"\n5. POWER PREDICTION IMPLICATIONS:")
if power_analysis:
    if 'high_pc1' in power_analysis and 'low_pc1' in power_analysis:
        high_10m = power_analysis['high_pc1']['power_wind10m_corr']
        high_70m = power_analysis['high_pc1']['power_wind70m_corr']
        low_10m = power_analysis['low_pc1']['power_wind10m_corr']
        low_70m = power_analysis['low_pc1']['power_wind70m_corr']
        
        print(f"   High PC1 periods: 10m correlation = {high_10m:.3f}, 70m correlation = {high_70m:.3f}")
        print(f"   Low PC1 periods:  10m correlation = {low_10m:.3f}, 70m correlation = {low_70m:.3f}")
        
        if high_70m > high_10m and low_10m > low_70m:
            print(f"   ✓ CONFIRMS hypothesis: High PC1 → 70m more important, Low PC1 → 10m more important")
        elif high_10m > high_70m and low_70m > low_10m:
            print(f"   ✗ OPPOSITE to hypothesis: High PC1 → 10m more important, Low PC1 → 70m more important")
        else:
            print(f"   ? MIXED results: No clear pattern")
else:
    print(f"   Power data not available for analysis")

print(f"\n6. OVERALL INTERPRETATION:")

# Determine the primary physical meaning of EOF Mode 1
if abs(pc1_shear_corr) > 0.3:
    primary_meaning = "WIND SHEAR VARIABILITY MODE"
    explanation = "EOF1 primarily captures changes in vertical wind shear"
elif abs(pc1_hour_corr) > 0.3 or (abs(pc1_daytime_corr) > 0.3):
    primary_meaning = "DIURNAL WIND INTENSITY MODE"
    explanation = "EOF1 primarily captures diurnal variations in wind speed intensity"
elif eof_interpretations[0]['type'] == 'uniform':
    primary_meaning = "OVERALL WIND INTENSITY MODE"
    explanation = "EOF1 captures overall wind speed changes affecting all heights similarly"
else:
    primary_meaning = "COMPLEX ATMOSPHERIC MODE"
    explanation = "EOF1 captures complex atmospheric dynamics with multiple influences"

print(f"   → EOF Mode 1 = {primary_meaning}")
print(f"   → {explanation}")

if abs(pc1_alpha_corr) > 0.15 or abs(pc1_shear_corr) > 0.15:
    print(f"   → This mode IS related to atmospheric stability and wind shear")
    print(f"   → Your original hypothesis about EOF-stability relationship has merit")
else:
    print(f"   → This mode is NOT strongly related to atmospheric stability")
    print(f"   → EOF1 may represent other atmospheric processes (synoptic patterns, etc.)")

print(f"\n" + "="*80)
print("FILES SAVED:")
print("="*80)
print(f"  - {plot_path}")
print(f"  - {interp_path}")
print(f"  - {temporal_path}")
print(f"  - {day_night_path}")
print(f"  - {atm_corr_path}")
if power_analysis:
    print(f"  - {power_path}")
print(f"  - {summary_path}")

print(f"\n" + "="*80)
print("NEXT STEPS RECOMMENDATION:")
print("="*80)

if abs(pc1_shear_corr) < 0.3 and abs(pc1_alpha_corr) < 0.3:
    print("1. EOF Mode 1 is NOT primarily a wind shear mode")
    print("2. Consider analyzing EOF Mode 2 or 3 for wind shear patterns")
    print("3. The original hypothesis may need revision")
    print("4. Focus on identifying which mode captures vertical structure changes")
elif abs(pc1_hour_corr) > 0.3:
    print("1. EOF Mode 1 is primarily driven by diurnal patterns")
    print("2. This supports the day/night stability hypothesis")
    print("3. High PC1 periods may correspond to specific times of day")
    print("4. Analyze whether these times correspond to stable/unstable conditions")
else:
    print("1. EOF Mode 1 shows some relationship with atmospheric conditions")
    print("2. The relationship is moderate but potentially meaningful")
    print("3. Consider seasonal analysis or longer time periods")
    print("4. Investigate other meteorological variables (temperature, pressure)")

print(f"\n" + "="*80)