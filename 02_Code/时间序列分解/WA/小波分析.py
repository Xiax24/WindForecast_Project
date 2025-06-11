import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
import pywt
from sklearn.decomposition import PCA
import warnings
import os
warnings.filterwarnings('ignore')

# Load data and calculate PC1
data_path = '/Users/xiaxin/work/WindForecast_Project/01_Data/processed/imputed_data/changma_clean.csv'
results_path = '/Users/xiaxin/work/WindForecast_Project/03_Results/08时间序列分解/PC1_Wavelet_Analysis'

os.makedirs(results_path, exist_ok=True)

print("="*70)
print("PC1 WAVELET ANALYSIS EXPERIMENT")
print("="*70)

# Load data
df = pd.read_csv(data_path)
df['datetime'] = pd.to_datetime(df['datetime'])
df.set_index('datetime', inplace=True)

# Calculate PC1 from wind speed data
wind_variables = ['obs_wind_speed_10m', 'obs_wind_speed_30m', 'obs_wind_speed_50m', 'obs_wind_speed_70m']
wind_data = df[wind_variables].dropna()

# Standardize and perform PCA to get PC1
wind_std = (wind_data - wind_data.mean()) / wind_data.std()
pca = PCA(n_components=4)
pcs = pca.fit_transform(wind_std)
PC1 = pcs[:, 0]  # First principal component

# Create PC1 time series
pc1_series = pd.Series(PC1, index=wind_data.index)

print(f"Data loaded: {len(pc1_series)} points")
print(f"PC1 range: {PC1.min():.2f} to {PC1.max():.2f}")
print(f"PC1 explains {pca.explained_variance_ratio_[0]*100:.1f}% of variance")

# =============================================================================
# 1. Continuous Wavelet Transform (CWT) - Time-Frequency Analysis
# =============================================================================
print("\n" + "="*50)
print("EXPERIMENT 1: TIME-FREQUENCY ANALYSIS")
print("="*50)

# Define scales for CWT (corresponding to different time periods)
# For 15-minute data: scale=96 corresponds to 24 hours, scale=672 to 7 days
dt = 0.25  # 15 minutes = 0.25 hours
scales = np.arange(4, 1000, 4)  # From 1 hour to ~10 days
frequencies = pywt.scale2frequency('cmor', scales) / dt
periods_hours = 1 / frequencies

# Perform continuous wavelet transform
coefficients, frequencies_cwt = pywt.cwt(PC1, scales, 'cmor', dt)
power = np.abs(coefficients) ** 2

print(f"Wavelet transform completed")
print(f"Analyzed periods: {periods_hours.min():.1f} to {periods_hours.max():.1f} hours")

# =============================================================================
# 2. Identify dominant periods and their time variations
# =============================================================================
print("\n" + "="*50)
print("EXPERIMENT 2: DOMINANT PERIODS IDENTIFICATION")
print("="*50)

# Find periods of interest (1 day, 3 days, 7 days)
target_periods = [24, 72, 168]  # hours
target_indices = []

for period in target_periods:
    idx = np.argmin(np.abs(periods_hours - period))
    target_indices.append(idx)
    actual_period = periods_hours[idx]
    print(f"Target {period}h period -> Actual {actual_period:.1f}h period (scale index {idx})")

# Extract time series of wavelet power for these periods
daily_power = power[target_indices[0], :]     # ~24 hour cycle
synoptic_power = power[target_indices[1], :]  # ~3 day cycle  
weekly_power = power[target_indices[2], :]    # ~7 day cycle

# Create time series
time_index = pc1_series.index
daily_strength = pd.Series(daily_power, index=time_index, name='Daily_Cycle_Strength')
synoptic_strength = pd.Series(synoptic_power, index=time_index, name='Synoptic_Cycle_Strength')
weekly_strength = pd.Series(weekly_power, index=time_index, name='Weekly_Cycle_Strength')

print(f"\nCycle strength statistics:")
print(f"Daily cycle - Mean: {daily_strength.mean():.3f}, Std: {daily_strength.std():.3f}")
print(f"Synoptic cycle - Mean: {synoptic_strength.mean():.3f}, Std: {synoptic_strength.std():.3f}")
print(f"Weekly cycle - Mean: {weekly_strength.mean():.3f}, Std: {weekly_strength.std():.3f}")

# =============================================================================
# 3. Seasonal analysis of different cycles
# =============================================================================
print("\n" + "="*50)
print("EXPERIMENT 3: SEASONAL CYCLE ANALYSIS")
print("="*50)

# Add seasonal information
seasonal_data = pd.DataFrame({
    'PC1': pc1_series,
    'Daily_Strength': daily_strength,
    'Synoptic_Strength': synoptic_strength,
    'Weekly_Strength': weekly_strength
})

seasonal_data['month'] = seasonal_data.index.month
seasonal_data['season'] = seasonal_data['month'].map({
    12: 'Winter', 1: 'Winter', 2: 'Winter',
    3: 'Spring', 4: 'Spring', 5: 'Spring',
    6: 'Summer', 7: 'Summer', 8: 'Summer',
    9: 'Autumn', 10: 'Autumn', 11: 'Autumn'
})

# Calculate seasonal statistics
seasonal_stats = seasonal_data.groupby('season').agg({
    'PC1': ['mean', 'std'],
    'Daily_Strength': ['mean', 'std'],
    'Synoptic_Strength': ['mean', 'std'],
    'Weekly_Strength': ['mean', 'std']
}).round(3)

print("Seasonal Cycle Strength Analysis:")
print(seasonal_stats)

# =============================================================================
# 4. Weather pattern identification
# =============================================================================
print("\n" + "="*50)
print("EXPERIMENT 4: WEATHER PATTERN IDENTIFICATION")
print("="*50)

# Define weather patterns based on wavelet energy combinations
def identify_weather_pattern(daily_str, synoptic_str, weekly_str):
    """Identify weather pattern based on cycle strengths"""
    
    # Normalize strengths
    d_norm = (daily_str - daily_strength.mean()) / daily_strength.std()
    s_norm = (synoptic_str - synoptic_strength.mean()) / synoptic_strength.std()
    w_norm = (weekly_str - weekly_strength.mean()) / weekly_strength.std()
    
    if s_norm > 1.5 and w_norm > 1.0:
        return "Strong_Synoptic_System"  # Strong 3-7 day weather system
    elif d_norm > 1.5 and s_norm < 0.5:
        return "Local_Convective"        # Strong daily cycle, weak synoptic
    elif s_norm > 1.0 and d_norm < 0.5:
        return "Stable_Large_Scale"      # Strong synoptic, weak daily
    elif all(x < 0.5 for x in [d_norm, s_norm, w_norm]):
        return "Quiet_Period"            # All cycles weak
    else:
        return "Mixed_Pattern"           # Mixed conditions

# Apply pattern identification
seasonal_data['Weather_Pattern'] = seasonal_data.apply(
    lambda row: identify_weather_pattern(
        row['Daily_Strength'], 
        row['Synoptic_Strength'], 
        row['Weekly_Strength']
    ), axis=1
)

# Count weather patterns
pattern_counts = seasonal_data['Weather_Pattern'].value_counts()
pattern_percentages = (pattern_counts / len(seasonal_data) * 100).round(1)

print("Weather Pattern Identification:")
for pattern, percentage in pattern_percentages.items():
    print(f"  {pattern}: {percentage}% of time")

# Analyze PC1 values for different patterns
pattern_pc1_stats = seasonal_data.groupby('Weather_Pattern')['PC1'].agg(['mean', 'std', 'min', 'max']).round(2)
print(f"\nPC1 Statistics by Weather Pattern:")
print(pattern_pc1_stats)

# =============================================================================
# 5. Extreme event detection
# =============================================================================
print("\n" + "="*50)
print("EXPERIMENT 5: EXTREME EVENT DETECTION")
print("="*50)

# Detect PC1 extreme events
pc1_threshold_high = pc1_series.quantile(0.95)
pc1_threshold_low = pc1_series.quantile(0.05)

extreme_high_events = seasonal_data[seasonal_data['PC1'] > pc1_threshold_high].copy()
extreme_low_events = seasonal_data[seasonal_data['PC1'] < pc1_threshold_low].copy()

print(f"PC1 extreme thresholds: Low < {pc1_threshold_low:.2f}, High > {pc1_threshold_high:.2f}")
print(f"Extreme high events: {len(extreme_high_events)} ({len(extreme_high_events)/len(seasonal_data)*100:.1f}%)")
print(f"Extreme low events: {len(extreme_low_events)} ({len(extreme_low_events)/len(seasonal_data)*100:.1f}%)")

# Analyze wavelet characteristics of extreme events
if len(extreme_high_events) > 0:
    high_patterns = extreme_high_events['Weather_Pattern'].value_counts()
    print(f"\nWeather patterns during PC1 extreme highs:")
    for pattern, count in high_patterns.items():
        percentage = count / len(extreme_high_events) * 100
        print(f"  {pattern}: {percentage:.1f}%")

if len(extreme_low_events) > 0:
    low_patterns = extreme_low_events['Weather_Pattern'].value_counts()
    print(f"\nWeather patterns during PC1 extreme lows:")
    for pattern, count in low_patterns.items():
        percentage = count / len(extreme_low_events) * 100
        print(f"  {pattern}: {percentage:.1f}%")

# =============================================================================
# 6. Predictive features extraction
# =============================================================================
print("\n" + "="*50)
print("EXPERIMENT 6: PREDICTIVE FEATURES")
print("="*50)

# Calculate moving averages and trends for predictive features
window_sizes = [24, 96, 288]  # 6 hours, 24 hours, 72 hours (in 15-min intervals)

for window in window_sizes:
    hours = window * 0.25
    seasonal_data[f'PC1_MA_{int(hours)}h'] = pc1_series.rolling(window).mean()
    seasonal_data[f'Synoptic_MA_{int(hours)}h'] = synoptic_strength.rolling(window).mean()

# Calculate energy trends (change in wavelet power)
seasonal_data['Synoptic_Trend_24h'] = synoptic_strength.diff(96)  # 24-hour change
seasonal_data['Daily_Trend_24h'] = daily_strength.diff(96)

# Phase analysis - where are we in the dominant cycle?
seasonal_data['Synoptic_Phase'] = np.angle(coefficients[target_indices[1], :])
seasonal_data['Daily_Phase'] = np.angle(coefficients[target_indices[0], :])

# Feature correlation with future PC1
future_pc1 = pc1_series.shift(-96)  # PC1 24 hours later
feature_cols = [col for col in seasonal_data.columns if any(x in col for x in ['MA_', 'Trend_', 'Phase', 'Strength'])]

correlations = {}
for col in feature_cols:
    if seasonal_data[col].notna().sum() > 1000:  # Enough data
        corr = seasonal_data[col].corr(future_pc1)
        if not np.isnan(corr):
            correlations[col] = corr

# Sort by absolute correlation
sorted_correlations = sorted(correlations.items(), key=lambda x: abs(x[1]), reverse=True)

print("Predictive Features (correlation with PC1 24h later):")
for feature, corr in sorted_correlations[:10]:
    print(f"  {feature}: {corr:.3f}")

# =============================================================================
# 7. Visualization
# =============================================================================
print("\n" + "="*50)
print("GENERATING VISUALIZATIONS")
print("="*50)

# Create comprehensive visualization
fig = plt.figure(figsize=(20, 16))

# Plot 1: PC1 time series
ax1 = plt.subplot(4, 2, (1, 2))
plt.plot(time_index, pc1_series, linewidth=0.8, alpha=0.8)
plt.title('PC1 Time Series (18 months)', fontsize=14)
plt.ylabel('PC1 Value')
plt.grid(True, alpha=0.3)

# Plot 2: Wavelet power spectrum (scalogram)
ax2 = plt.subplot(4, 2, (3, 4))
# Subsample for visualization
subsample = slice(None, None, 10)  # Every 10th point
time_sub = time_index[subsample]
power_sub = power[:, subsample]

im = plt.contourf(time_sub, periods_hours, power_sub, levels=50, cmap='viridis')
plt.colorbar(im, label='Wavelet Power')
plt.ylabel('Period (hours)')
plt.title('PC1 Wavelet Power Spectrum (Time-Frequency)', fontsize=14)
plt.yscale('log')
plt.ylim(1, 500)

# Mark important periods
for period in [24, 72, 168]:
    plt.axhline(y=period, color='red', linestyle='--', alpha=0.7, linewidth=1)
    plt.text(time_sub[len(time_sub)//10], period*1.1, f'{period}h', color='red', fontweight='bold')

# Plot 3: Cycle strengths
ax3 = plt.subplot(4, 2, 5)
plt.plot(time_index, daily_strength, label='Daily (24h)', alpha=0.8)
plt.plot(time_index, synoptic_strength, label='Synoptic (3d)', alpha=0.8)
plt.plot(time_index, weekly_strength, label='Weekly (7d)', alpha=0.8)
plt.title('Wavelet Power by Time Scale')
plt.ylabel('Wavelet Power')
plt.legend()
plt.grid(True, alpha=0.3)

# Plot 4: Seasonal cycle strength
ax4 = plt.subplot(4, 2, 6)
seasonal_means = seasonal_data.groupby('season')[['Daily_Strength', 'Synoptic_Strength', 'Weekly_Strength']].mean()
seasonal_means.plot(kind='bar', ax=ax4)
plt.title('Seasonal Cycle Strength')
plt.ylabel('Average Wavelet Power')
plt.xticks(rotation=45)
plt.legend()
plt.grid(True, alpha=0.3)

# Plot 5: Weather pattern distribution
ax5 = plt.subplot(4, 2, 7)
pattern_percentages.plot(kind='pie', ax=ax5, autopct='%1.1f%%')
plt.title('Weather Pattern Distribution')
plt.ylabel('')

# Plot 6: PC1 vs dominant cycle strength
ax6 = plt.subplot(4, 2, 8)
plt.scatter(synoptic_strength, pc1_series, alpha=0.3, s=1)
plt.xlabel('Synoptic Cycle Strength')
plt.ylabel('PC1 Value')
plt.title('PC1 vs Synoptic Cycle Strength')
plt.grid(True, alpha=0.3)

# Add correlation
corr = synoptic_strength.corr(pc1_series)
plt.text(0.05, 0.95, f'Correlation: {corr:.3f}', transform=ax6.transAxes, 
         bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

plt.tight_layout()
plot_path = os.path.join(results_path, 'PC1_wavelet_comprehensive_analysis.png')
plt.savefig(plot_path, dpi=150, bbox_inches='tight')
plt.show()

# =============================================================================
# 8. Save results
# =============================================================================
print("\n" + "="*50)
print("SAVING RESULTS")
print("="*50)

# Save detailed results
results_summary = {
    'seasonal_stats': seasonal_stats,
    'pattern_counts': pattern_counts,
    'pattern_pc1_stats': pattern_pc1_stats,
    'predictive_correlations': dict(sorted_correlations),
    'extreme_event_counts': {
        'high_events': len(extreme_high_events),
        'low_events': len(extreme_low_events)
    }
}

# Save cycle strength data
cycle_data = pd.DataFrame({
    'datetime': time_index,
    'PC1': pc1_series.values,
    'Daily_Strength': daily_strength.values,
    'Synoptic_Strength': synoptic_strength.values,
    'Weekly_Strength': weekly_strength.values,
    'Weather_Pattern': seasonal_data['Weather_Pattern'].values
})

cycle_data_path = os.path.join(results_path, 'PC1_wavelet_features.csv')
cycle_data.to_csv(cycle_data_path, index=False)

# Save summary statistics
summary_path = os.path.join(results_path, 'PC1_wavelet_summary.txt')
with open(summary_path, 'w') as f:
    f.write("PC1 WAVELET ANALYSIS SUMMARY\n")
    f.write("="*50 + "\n\n")
    
    f.write("SEASONAL ANALYSIS:\n")
    f.write(str(seasonal_stats) + "\n\n")
    
    f.write("WEATHER PATTERNS:\n")
    for pattern, percentage in pattern_percentages.items():
        f.write(f"  {pattern}: {percentage}%\n")
    f.write("\n")
    
    f.write("PREDICTIVE FEATURES (Top 5):\n")
    for feature, corr in sorted_correlations[:5]:
        f.write(f"  {feature}: {corr:.3f}\n")

print(f"Results saved to: {results_path}")
print("Files generated:")
print(f"  - PC1_wavelet_comprehensive_analysis.png")
print(f"  - PC1_wavelet_features.csv")
print(f"  - PC1_wavelet_summary.txt")

# =============================================================================
# Summary
# =============================================================================
print("\n" + "="*70)
print("PC1 WAVELET ANALYSIS SUMMARY")
print("="*70)

print(f"\nKey Findings:")
print(f"1. Dominant Weather Pattern: {pattern_counts.index[0]} ({pattern_percentages.iloc[0]}% of time)")
print(f"2. Strongest Seasonal Effect: {seasonal_stats.loc[seasonal_stats[('PC1', 'mean')].idxmax()][('PC1', 'mean')]:.2f} PC1 in {seasonal_stats[('PC1', 'mean')].idxmax()}")
print(f"3. Best Predictive Feature: {sorted_correlations[0][0]} (r={sorted_correlations[0][1]:.3f})")
print(f"4. Synoptic-PC1 Correlation: {synoptic_strength.corr(pc1_series):.3f}")

print(f"\nPractical Applications:")
print(f"- Weather system tracking through wavelet patterns")
print(f"- Extreme event prediction using cycle combinations")  
print(f"- Seasonal planning based on cycle strength variations")
print(f"- 24-hour ahead forecasting using predictive features")

print(f"\nExperiment completed successfully!")
print(f"Check the generated files for detailed results and visualizations.")