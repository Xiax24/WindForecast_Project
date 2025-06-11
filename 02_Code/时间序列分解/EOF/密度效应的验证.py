import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
from sklearn.decomposition import PCA
import warnings
warnings.filterwarnings('ignore')

# Load data and perform joint EOF to get actual PC2
data_path = '/Users/xiaxin/work/WindForecast_Project/01_Data/processed/imputed_data/changma_clean.csv'
results_path = '/Users/xiaxin/work/WindForecast_Project/03_Results/08时间序列分解/PC2_Profile_Verification'

import os
os.makedirs(results_path, exist_ok=True)

print("="*70)
print("PC2 PROFILE-BASED DENSITY EFFECT VERIFICATION")
print("="*70)

# Load data
df = pd.read_csv(data_path)
df['datetime'] = pd.to_datetime(df['datetime'])
df.set_index('datetime', inplace=True)

# Select 4 wind speed variables + 1 temperature variable for joint EOF
joint_variables = ['obs_wind_speed_10m', 'obs_wind_speed_30m', 'obs_wind_speed_50m', 'obs_wind_speed_70m', 'obs_temperature_10m']
joint_data = df[joint_variables].copy().dropna()

print(f"Data loaded: {len(df)} total points, {len(joint_data)} valid points")

# =============================================================================
# Step 1: Recalculate Joint EOF to get actual PC2
# =============================================================================
print("\n" + "="*50)
print("STEP 1: JOINT EOF ANALYSIS")
print("="*50)

# Standardize the data
data_std = (joint_data - joint_data.mean()) / joint_data.std()

# Perform PCA (Joint EOF)
pca = PCA(n_components=5)
principal_components = pca.fit_transform(data_std)
eof_patterns = pca.components_
explained_variance = pca.explained_variance_ratio_

# Get PC2 time series
PC2_timeseries = principal_components[:, 1]  # Second component (index 1)
PC2_pattern = eof_patterns[1]  # PC2 spatial pattern

print(f"PC2 pattern: {PC2_pattern}")
print(f"PC2 explained variance: {explained_variance[1]*100:.1f}%")
print(f"Wind loadings: {PC2_pattern[:4]}")
print(f"Temperature loading: {PC2_pattern[4]:.3f}")

# Create a dataframe with PC time series aligned with original data
pc_df = pd.DataFrame(principal_components, 
                    columns=[f'PC{i+1}' for i in range(5)],
                    index=joint_data.index)

# Merge back with original data
df_with_pc = df.join(pc_df, how='inner')

print(f"Data with PC series: {len(df_with_pc)} points")

# =============================================================================
# Step 2: Identify PC2-characteristic periods
# =============================================================================
print("\n" + "="*50)
print("STEP 2: IDENTIFY PC2-CHARACTERISTIC PERIODS")
print("="*50)

# Condition 1: PC2 is dominant (strong PC2 signal)
pc2_strength_threshold = 1.0
pc2_dominant = np.abs(df_with_pc['PC2']) > pc2_strength_threshold

# Condition 2: Wind profile is uniform (matches PC2 characteristics)
# PC2 pattern shows wind speeds should be nearly uniform across heights
wind_10m = df_with_pc['obs_wind_speed_10m']
wind_30m = df_with_pc['obs_wind_speed_30m']
wind_50m = df_with_pc['obs_wind_speed_50m']
wind_70m = df_with_pc['obs_wind_speed_70m']

# Check uniformity: small differences between heights
wind_uniformity = (
    (np.abs(wind_70m - wind_10m) < 2.5) &  # Top-bottom difference < 2.5 m/s
    (np.abs(wind_50m - wind_30m) < 1.5) &  # Middle layer difference < 1.5 m/s
    (np.abs(wind_70m - wind_50m) < 1.5) &  # Upper layers difference < 1.5 m/s
    (np.abs(wind_30m - wind_10m) < 1.5)    # Lower layers difference < 1.5 m/s
)

# Condition 3: Reasonable wind speed range
wind_reasonable = (
    (wind_10m >= 4) & (wind_10m <= 15) &
    (wind_70m >= 4) & (wind_70m <= 15)
)

# Condition 4: Valid temperature and power data
valid_data = (
    df_with_pc['obs_temperature_10m'].notna() &
    df_with_pc['power'].notna()
)

# Combine all conditions for PC2-characteristic periods
pc2_characteristic = pc2_dominant & wind_uniformity & wind_reasonable & valid_data

print(f"PC2 dominant periods: {pc2_dominant.sum()} ({pc2_dominant.mean()*100:.1f}%)")
print(f"Wind uniform periods: {wind_uniformity.sum()} ({wind_uniformity.mean()*100:.1f}%)")
print(f"PC2 characteristic periods: {pc2_characteristic.sum()} ({pc2_characteristic.mean()*100:.1f}%)")

# =============================================================================
# Step 3: Analyze PC2-characteristic periods
# =============================================================================
print("\n" + "="*50)
print("STEP 3: PC2-CHARACTERISTIC PERIODS ANALYSIS")
print("="*50)

if pc2_characteristic.sum() < 100:
    print("WARNING: Too few PC2-characteristic periods for analysis")
    print("Relaxing criteria...")
    # Relax criteria if too few data points
    pc2_strength_threshold = 0.5
    pc2_dominant = np.abs(df_with_pc['PC2']) > pc2_strength_threshold
    pc2_characteristic = pc2_dominant & wind_uniformity & wind_reasonable & valid_data
    print(f"Relaxed PC2 characteristic periods: {pc2_characteristic.sum()}")

# Get PC2-characteristic data
pc2_data = df_with_pc[pc2_characteristic].copy()

# Analyze the characteristics of these periods
print(f"\nPC2-Characteristic Periods Analysis:")
print(f"Total periods: {len(pc2_data)}")
print(f"PC2 range: {pc2_data['PC2'].min():.2f} to {pc2_data['PC2'].max():.2f}")
print(f"Temperature range: {pc2_data['obs_temperature_10m'].min():.1f}°C to {pc2_data['obs_temperature_10m'].max():.1f}°C")

# Check wind profile uniformity in these periods
wind_profile_stats = pd.DataFrame({
    'Mean': [pc2_data[f'obs_wind_speed_{h}m'].mean() for h in [10,30,50,70]],
    'Std': [pc2_data[f'obs_wind_speed_{h}m'].std() for h in [10,30,50,70]]
}, index=['10m', '30m', '50m', '70m'])

print(f"\nWind Profile in PC2-Characteristic Periods:")
print(wind_profile_stats.round(2))

# =============================================================================
# Step 4: PC2-based density effect verification
# =============================================================================
print("\n" + "="*50)
print("STEP 4: PC2-BASED DENSITY EFFECT VERIFICATION")
print("="*50)

# Within PC2-characteristic periods, compare cold vs hot conditions
# Use PC2 itself to define cold/hot (since PC2 ≈ -temperature)
pc2_cold_threshold = np.percentile(pc2_data['PC2'], 75)  # Top 25% PC2 (coldest)
pc2_hot_threshold = np.percentile(pc2_data['PC2'], 25)   # Bottom 25% PC2 (hottest)

pc2_cold_periods = pc2_data['PC2'] >= pc2_cold_threshold
pc2_hot_periods = pc2_data['PC2'] <= pc2_hot_threshold

cold_data = pc2_data[pc2_cold_periods]
hot_data = pc2_data[pc2_hot_periods]

print(f"PC2 cold threshold: {pc2_cold_threshold:.2f}")
print(f"PC2 hot threshold: {pc2_hot_threshold:.2f}")
print(f"Cold periods (high PC2): {len(cold_data)} points")
print(f"Hot periods (low PC2): {len(hot_data)} points")

if len(cold_data) > 20 and len(hot_data) > 20:
    # Calculate averages for cold and hot periods
    cold_stats = {
        'PC2': cold_data['PC2'].mean(),
        'Temperature': cold_data['obs_temperature_10m'].mean(),
        'Power': cold_data['power'].mean(),
        'Wind_10m': cold_data['obs_wind_speed_10m'].mean(),
        'Wind_30m': cold_data['obs_wind_speed_30m'].mean(),
        'Wind_50m': cold_data['obs_wind_speed_50m'].mean(),
        'Wind_70m': cold_data['obs_wind_speed_70m'].mean(),
    }
    
    hot_stats = {
        'PC2': hot_data['PC2'].mean(),
        'Temperature': hot_data['obs_temperature_10m'].mean(),
        'Power': hot_data['power'].mean(),
        'Wind_10m': hot_data['obs_wind_speed_10m'].mean(),
        'Wind_30m': hot_data['obs_wind_speed_30m'].mean(),
        'Wind_50m': hot_data['obs_wind_speed_50m'].mean(),
        'Wind_70m': hot_data['obs_wind_speed_70m'].mean(),
    }
    
    print(f"\nCold PC2 Periods (High PC2, Low Temperature):")
    for key, value in cold_stats.items():
        print(f"  {key}: {value:.2f}")
    
    print(f"\nHot PC2 Periods (Low PC2, High Temperature):")
    for key, value in hot_stats.items():
        print(f"  {key}: {value:.2f}")
    
    # Calculate differences
    temp_diff = cold_stats['Temperature'] - hot_stats['Temperature']
    power_diff = cold_stats['Power'] - hot_stats['Power']
    wind_diffs = {
        f'Wind_{h}m': cold_stats[f'Wind_{h}m'] - hot_stats[f'Wind_{h}m'] 
        for h in [10, 30, 50, 70]
    }
    
    print(f"\nDifferences (Cold - Hot):")
    print(f"  Temperature: {temp_diff:.2f}°C")
    print(f"  Power: {power_diff:.2f} kW")
    for key, value in wind_diffs.items():
        print(f"  {key}: {value:.2f} m/s")
    
    # Check wind profile consistency
    max_wind_diff = max(abs(diff) for diff in wind_diffs.values())
    print(f"  Max wind difference across heights: {max_wind_diff:.2f} m/s")
    
    # Density effect calculation
    T_cold_K = cold_stats['Temperature'] + 273.15
    T_hot_K = hot_stats['Temperature'] + 273.15
    theoretical_density_ratio = T_hot_K / T_cold_K
    theoretical_power_increase = (theoretical_density_ratio - 1) * 100
    
    actual_power_increase = (power_diff / hot_stats['Power']) * 100
    
    print(f"\nDensity Effect Analysis:")
    print(f"  Theoretical density ratio: {theoretical_density_ratio:.3f}")
    print(f"  Theoretical power increase: {theoretical_power_increase:.1f}%")
    print(f"  Actual power increase: {actual_power_increase:.1f}%")
    print(f"  Ratio (Actual/Theoretical): {actual_power_increase/theoretical_power_increase:.1f}")
    
    # Verification results
    print(f"\nVerification Results:")
    if power_diff > 0:
        print("✓ Cold PC2 periods show higher power")
        if max_wind_diff < 1.0:
            print("✓ Wind profiles remain consistent (difference < 1.0 m/s)")
        else:
            print(f"⚠ Wind profiles show some variation (max diff: {max_wind_diff:.1f} m/s)")
        
        if 0.5 < actual_power_increase/theoretical_power_increase < 2.0:
            print("✓ Actual power increase is reasonable compared to theory")
        else:
            print("⚠ Actual power increase deviates significantly from theory")
    else:
        print("✗ Cold PC2 periods do NOT show higher power")

# =============================================================================
# Step 5: Statistical analysis and correlation
# =============================================================================
print("\n" + "="*50)
print("STEP 5: STATISTICAL ANALYSIS")
print("="*50)

# Correlation analysis within PC2-characteristic periods
pc2_corr_temp = np.corrcoef(pc2_data['PC2'], pc2_data['obs_temperature_10m'])[0,1]
pc2_corr_power = np.corrcoef(pc2_data['PC2'], pc2_data['power'])[0,1]
temp_corr_power = np.corrcoef(pc2_data['obs_temperature_10m'], pc2_data['power'])[0,1]

print(f"Correlations within PC2-characteristic periods:")
print(f"  PC2 vs Temperature: {pc2_corr_temp:.3f}")
print(f"  PC2 vs Power: {pc2_corr_power:.3f}")
print(f"  Temperature vs Power: {temp_corr_power:.3f}")

# Expected: PC2 vs Temperature should be negative (PC2 loading is -0.999)
# Expected: PC2 vs Power should be positive if density effect works as expected

# =============================================================================
# Step 6: Visualization
# =============================================================================
print("\n" + "="*50)
print("GENERATING VISUALIZATION")
print("="*50)

fig, axes = plt.subplots(2, 3, figsize=(18, 12))
fig.suptitle('PC2 Profile-based Density Effect Verification', fontsize=16)

# Plot 1: PC2 characteristic periods identification
axes[0,0].scatter(df_with_pc.index, df_with_pc['PC2'], alpha=0.1, s=1, label='All data', color='lightblue')
axes[0,0].scatter(pc2_data.index, pc2_data['PC2'], alpha=0.5, s=2, label='PC2 characteristic', color='red')
axes[0,0].axhline(y=pc2_strength_threshold, color='red', linestyle='--', alpha=0.7)
axes[0,0].axhline(y=-pc2_strength_threshold, color='red', linestyle='--', alpha=0.7)
axes[0,0].set_ylabel('PC2 Value')
axes[0,0].set_title('PC2 Characteristic Periods')
axes[0,0].legend()
axes[0,0].grid(True, alpha=0.3)

# Plot 2: Wind profile in PC2 periods
heights = [10, 30, 50, 70]
mean_winds = [pc2_data[f'obs_wind_speed_{h}m'].mean() for h in heights]
axes[0,1].plot(mean_winds, heights, 'bo-', linewidth=2, markersize=8)
axes[0,1].set_xlabel('Average Wind Speed (m/s)')
axes[0,1].set_ylabel('Height (m)')
axes[0,1].set_title('Average Wind Profile in PC2 Periods')
axes[0,1].grid(True, alpha=0.3)

# Plot 3: PC2 vs Temperature in characteristic periods
axes[0,2].scatter(pc2_data['PC2'], pc2_data['obs_temperature_10m'], alpha=0.3, s=10)
axes[0,2].set_xlabel('PC2')
axes[0,2].set_ylabel('Temperature (°C)')
axes[0,2].set_title(f'PC2 vs Temperature (r={pc2_corr_temp:.3f})')
axes[0,2].grid(True, alpha=0.3)

# Plot 4: PC2 vs Power in characteristic periods
axes[1,0].scatter(pc2_data['PC2'], pc2_data['power'], alpha=0.3, s=10)
axes[1,0].set_xlabel('PC2')
axes[1,0].set_ylabel('Power (kW)')
axes[1,0].set_title(f'PC2 vs Power (r={pc2_corr_power:.3f})')
axes[1,0].grid(True, alpha=0.3)

# Plot 5: Temperature vs Power in characteristic periods
axes[1,1].scatter(pc2_data['obs_temperature_10m'], pc2_data['power'], alpha=0.3, s=10)
axes[1,1].set_xlabel('Temperature (°C)')
axes[1,1].set_ylabel('Power (kW)')
axes[1,1].set_title(f'Temperature vs Power (r={temp_corr_power:.3f})')
axes[1,1].grid(True, alpha=0.3)

# Plot 6: Cold vs Hot comparison
if len(cold_data) > 20 and len(hot_data) > 20:
    categories = ['Cold PC2\n(Low Temp)', 'Hot PC2\n(High Temp)']
    powers = [cold_stats['Power'], hot_stats['Power']]
    temps = [cold_stats['Temperature'], hot_stats['Temperature']]
    
    ax6 = axes[1,2]
    bars = ax6.bar(categories, powers, color=['blue', 'red'], alpha=0.7)
    ax6.set_ylabel('Average Power (kW)', color='black')
    ax6.set_title('Power Comparison: Cold vs Hot PC2 Periods')
    
    # Add temperature as text labels
    for i, (bar, temp) in enumerate(zip(bars, temps)):
        height = bar.get_height()
        ax6.text(bar.get_x() + bar.get_width()/2., height + 1,
                f'{temp:.1f}°C', ha='center', va='bottom')
    
    ax6.grid(True, alpha=0.3)

plt.tight_layout()
plot_path = os.path.join(results_path, 'PC2_profile_verification.png')
plt.savefig(plot_path, dpi=150, bbox_inches='tight')
plt.show()

print(f"Visualization saved to: {plot_path}")

# =============================================================================
# Summary
# =============================================================================
print("\n" + "="*70)
print("SUMMARY: PC2 PROFILE-BASED DENSITY EFFECT VERIFICATION")
print("="*70)

print(f"\nMethodology:")
print(f"  - Used actual PC2 from joint EOF analysis")
print(f"  - Selected periods where PC2 is dominant (|PC2| > {pc2_strength_threshold})")
print(f"  - Ensured wind profiles are uniform (matching PC2 characteristics)")
print(f"  - Compared cold vs hot conditions within these periods")

print(f"\nKey Findings:")
if 'pc2_characteristic' in locals():
    print(f"  - PC2 characteristic periods: {pc2_characteristic.mean()*100:.1f}% of data")
if 'max_wind_diff' in locals():
    print(f"  - Wind profile consistency: {max_wind_diff:.1f} m/s max difference")
if 'theoretical_power_increase' in locals():
    print(f"  - Theoretical density effect: {theoretical_power_increase:.1f}%")
if 'actual_power_increase' in locals():
    print(f"  - Actual power increase: {actual_power_increase:.1f}%")
if 'pc2_corr_power' in locals():
    print(f"  - PC2-Power correlation: {pc2_corr_power:.3f}")

print(f"\nConclusion:")
if ('actual_power_increase' in locals() and 'theoretical_power_increase' in locals() and
    actual_power_increase > 0 and 0.5 < actual_power_increase/theoretical_power_increase < 2.0):
    print("  STRONG EVIDENCE: PC2 represents density effect with minimal contamination")
elif 'actual_power_increase' in locals() and actual_power_increase > 0:
    print("  MODERATE EVIDENCE: PC2 includes density effect but may have other components")
else:
    print("  LIMITED EVIDENCE: PC2 may not primarily represent density effect")

print(f"\nFiles saved to: {results_path}")