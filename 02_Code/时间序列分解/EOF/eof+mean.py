import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from scipy import stats
import warnings
import os
warnings.filterwarnings('ignore')

# Set paths
data_path = '/Users/xiaxin/work/WindForecast_Project/01_Data/processed/imputed_data/changma_clean.csv'
results_path = '/Users/xiaxin/work/WindForecast_Project/03_Results/08时间序列分解/EOF_Mean_Profile_Analysis'

# Create results directory
os.makedirs(results_path, exist_ok=True)

print("="*60)
print("EOF MEAN PROFILE CORRELATION ANALYSIS")
print("="*60)

# Load and prepare data
print("\n1. Loading data...")
df = pd.read_csv(data_path)
df['datetime'] = pd.to_datetime(df['datetime'])
df.set_index('datetime', inplace=True)

wind_variables = ['obs_wind_speed_10m', 'obs_wind_speed_30m', 'obs_wind_speed_50m', 'obs_wind_speed_70m']
heights = [10, 30, 50, 70]

wind_data = df[wind_variables].dropna()
print(f"Valid data points: {len(wind_data)}")

# Calculate mean wind profile
mean_profile = wind_data.mean().values
print(f"\nMean wind profile:")
for i, (h, v) in enumerate(zip(heights, mean_profile)):
    print(f"  {h}m: {v:.2f} m/s")

# Perform EOF analysis
print("\n2. Performing EOF analysis...")
data_standardized = (wind_data - wind_data.mean()) / wind_data.std()
pca = PCA(n_components=4)
time_coefficients = pca.fit_transform(data_standardized)
eof_patterns = pca.components_
explained_variance = pca.explained_variance_ratio_

print(f"EOF variance explained:")
for i, var in enumerate(explained_variance):
    print(f"  EOF{i+1}: {var*100:.1f}%")

# Calculate correlations with mean profile
print("\n3. Calculating correlations with mean profile...")
correlations_with_mean = []
for i, pattern in enumerate(eof_patterns):
    corr = np.corrcoef(pattern, mean_profile)[0, 1]
    correlations_with_mean.append(corr)
    print(f"  EOF{i+1} vs Mean Profile: r = {corr:.4f}")

# Analyze PC1 temporal patterns
print("\n4. Analyzing PC1 temporal patterns...")
pc1 = time_coefficients[:, 0]
hours = wind_data.index.hour
pc1_hour_corr = np.corrcoef(pc1, hours)[0, 1]
print(f"PC1 vs Hour correlation: {pc1_hour_corr:.4f}")

# Day vs night analysis
daytime_mask = (hours >= 6) & (hours <= 18)
pc1_day = pc1[daytime_mask]
pc1_night = pc1[~daytime_mask]

if len(pc1_day) > 30 and len(pc1_night) > 30:
    t_stat, p_val = stats.ttest_ind(pc1_day, pc1_night)
    print(f"Day vs Night: day_mean={pc1_day.mean():.4f}, night_mean={pc1_night.mean():.4f}, p={p_val:.4f}")

# Calculate wind shear analysis
print("\n5. Wind shear analysis...")
alpha_10_70 = np.log(wind_data['obs_wind_speed_70m'] / wind_data['obs_wind_speed_10m']) / np.log(7)
wind_shear_magnitude = wind_data['obs_wind_speed_70m'] - wind_data['obs_wind_speed_10m']

pc1_alpha_corr = np.corrcoef(pc1, alpha_10_70)[0, 1]
pc1_shear_corr = np.corrcoef(pc1, wind_shear_magnitude)[0, 1]

print(f"PC1 vs Alpha coefficient: {pc1_alpha_corr:.4f}")
print(f"PC1 vs Wind shear magnitude: {pc1_shear_corr:.4f}")

# Create visualization
print("\n6. Creating visualization...")
fig, axes = plt.subplots(2, 3, figsize=(18, 12))
fig.suptitle('EOF Mean Profile Correlation Analysis', fontsize=16, fontweight='bold')

# Plot 1: EOF1 vs Mean Profile
ax = axes[0, 0]
ax.plot(mean_profile, heights, 'ko-', linewidth=3, markersize=8, label='Mean Profile')
eof1_normalized = eof_patterns[0] * np.mean(mean_profile) / np.mean(np.abs(eof_patterns[0]))
ax.plot(eof1_normalized, heights, 'ro-', linewidth=2, markersize=6, label='EOF1 (normalized)')
ax.set_title(f'EOF1 vs Mean Profile\nCorrelation: {correlations_with_mean[0]:.4f}', fontweight='bold')
ax.set_xlabel('Wind Speed / EOF Loading')
ax.set_ylabel('Height (m)')
ax.legend()
ax.grid(True, alpha=0.3)

# Plot 2: All EOF patterns
ax = axes[0, 1]
colors = ['red', 'blue', 'green', 'orange']
for i in range(4):
    ax.plot(eof_patterns[i], heights, 'o-', color=colors[i], linewidth=2, 
           label=f'EOF{i+1} ({explained_variance[i]*100:.1f}%)')
ax.axvline(x=0, color='black', linestyle='--', alpha=0.5)
ax.set_title('EOF Patterns', fontweight='bold')
ax.set_xlabel('EOF Loading')
ax.set_ylabel('Height (m)')
ax.legend()
ax.grid(True, alpha=0.3)

# Plot 3: Correlations with mean profile
ax = axes[0, 2]
bars = ax.bar(['EOF1', 'EOF2', 'EOF3', 'EOF4'], correlations_with_mean, color=colors, alpha=0.7)
ax.axhline(y=0, color='black', linestyle='-', alpha=0.5)
ax.set_title('Correlations with Mean Profile', fontweight='bold')
ax.set_ylabel('Correlation Coefficient')
ax.grid(True, alpha=0.3)

# Add values on bars
for bar, corr in zip(bars, correlations_with_mean):
    height = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2., height + 0.02 if height > 0 else height - 0.05,
           f'{corr:.3f}', ha='center', va='bottom' if height > 0 else 'top', fontweight='bold')

# Plot 4: PC1 diurnal pattern
ax = axes[1, 0]
hourly_pc1 = pd.Series(pc1, index=wind_data.index).groupby(wind_data.index.hour).mean()
ax.plot(hourly_pc1.index, hourly_pc1.values, 'bo-', linewidth=2, markersize=6)
ax.axvspan(6, 18, alpha=0.2, color='yellow', label='Daytime')
ax.set_title(f'PC1 Diurnal Pattern\nHour correlation: {pc1_hour_corr:.3f}', fontweight='bold')
ax.set_xlabel('Hour of Day')
ax.set_ylabel('PC1 Mean Value')
ax.legend()
ax.grid(True, alpha=0.3)

# Plot 5: PC1 vs Wind Shear
ax = axes[1, 1]
ax.scatter(wind_shear_magnitude, pc1, alpha=0.3, s=8, color='blue')
ax.set_title(f'PC1 vs Wind Shear\nCorrelation: {pc1_shear_corr:.3f}', fontweight='bold')
ax.set_xlabel('Wind Shear Magnitude (70m-10m) [m/s]')
ax.set_ylabel('PC1 Values')
ax.grid(True, alpha=0.3)

# Plot 6: Summary text
ax = axes[1, 2]
ax.axis('off')

eof1_corr = correlations_with_mean[0]
summary_text = "KEY FINDINGS:\n\n"

if abs(eof1_corr) > 0.9:
    summary_text += f"✓ EOF1 = MEAN PROFILE SCALING\n"
    summary_text += f"  Correlation: {eof1_corr:.3f}\n\n"
    summary_text += f"EOF1 represents wind intensity\n"
    summary_text += f"variations (stronger/weaker\n"
    summary_text += f"than typical profile)\n\n"
    conclusion = "WIND INTENSITY MODE"
elif abs(eof1_corr) > 0.7:
    summary_text += f"✓ EOF1 ≈ MEAN PROFILE\n"
    summary_text += f"  Correlation: {eof1_corr:.3f}\n\n"
    summary_text += f"EOF1 mainly represents wind\n"
    summary_text += f"intensity with some shape\n"
    summary_text += f"modifications\n\n"
    conclusion = "MODIFIED INTENSITY MODE"
else:
    summary_text += f"✗ EOF1 ≠ MEAN PROFILE\n"
    summary_text += f"  Correlation: {eof1_corr:.3f}\n\n"
    summary_text += f"EOF1 has distinct structure\n"
    summary_text += f"representing specific\n"
    summary_text += f"atmospheric patterns\n\n"
    conclusion = "DISTINCT PATTERN MODE"

summary_text += f"CONCLUSION:\n"
summary_text += f"EOF1 = {conclusion}"

if abs(pc1_hour_corr) > 0.15:
    summary_text += f"\n\n✓ Strong diurnal signal"

if abs(pc1_shear_corr) > 0.15:
    summary_text += f"\n✓ Related to wind shear"

ax.text(0.05, 0.95, summary_text, transform=ax.transAxes, fontsize=11,
       verticalalignment='top', fontweight='bold',
       bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8))

plt.tight_layout()

# Save plot
plot_path = os.path.join(results_path, 'EOF_Mean_Profile_Analysis.png')
plt.savefig(plot_path, dpi=300, bbox_inches='tight')
plt.show()

# Save results
print("\n7. Saving results...")

# Create summary dataframe
results_summary = {
    'EOF1_Mean_Profile_Correlation': correlations_with_mean[0],
    'EOF2_Mean_Profile_Correlation': correlations_with_mean[1],
    'EOF3_Mean_Profile_Correlation': correlations_with_mean[2],
    'EOF4_Mean_Profile_Correlation': correlations_with_mean[3],
    'EOF1_Variance_Explained_%': explained_variance[0] * 100,
    'PC1_Hour_Correlation': pc1_hour_corr,
    'PC1_Alpha_Correlation': pc1_alpha_corr,
    'PC1_Shear_Correlation': pc1_shear_corr,
    'Mean_10m_Wind': mean_profile[0],
    'Mean_30m_Wind': mean_profile[1],
    'Mean_50m_Wind': mean_profile[2],
    'Mean_70m_Wind': mean_profile[3],
    'Conclusion': conclusion
}

summary_df = pd.DataFrame([results_summary])
summary_path = os.path.join(results_path, 'EOF_Analysis_Summary.csv')
summary_df.to_csv(summary_path, index=False)

# Save detailed results
detailed_results = pd.DataFrame({
    'EOF_Mode': ['EOF1', 'EOF2', 'EOF3', 'EOF4'],
    'Variance_Explained_%': explained_variance * 100,
    'Mean_Profile_Correlation': correlations_with_mean,
    'Pattern_10m': eof_patterns[:, 0],
    'Pattern_30m': eof_patterns[:, 1],
    'Pattern_50m': eof_patterns[:, 2],
    'Pattern_70m': eof_patterns[:, 3]
})

detailed_path = os.path.join(results_path, 'EOF_Detailed_Results.csv')
detailed_results.to_csv(detailed_path, index=False)

# Print final conclusions
print("\n" + "="*60)
print("FINAL CONCLUSIONS")
print("="*60)

print(f"\n🎯 PRIMARY FINDING:")
print(f"EOF1 correlation with mean profile: {correlations_with_mean[0]:.4f}")

if abs(correlations_with_mean[0]) > 0.9:
    print(f"\n✅ CONCLUSION: EOF1 = MEAN WIND PROFILE SCALING")
    print(f"   → EOF1 represents how much stronger/weaker winds are than normal")
    print(f"   → PC1 high = stronger typical winds, PC1 low = weaker typical winds")
    print(f"   → This explains why EOF1 'looks like unstable profile' - it's the mean shape!")
    print(f"   → Your original hypothesis needs revision: EOF1 ≠ stability mode")
    
    print(f"\n🔍 IMPLICATIONS:")
    print(f"   • EOF1 captures WIND INTENSITY, not wind profile SHAPE changes")
    print(f"   • Shape changes (stability effects) are in EOF2, EOF3, etc.")
    print(f"   • High PC1 periods = good prediction at all heights")
    print(f"   • Height-dependent prediction effects come from other EOF modes")
    
elif abs(correlations_with_mean[0]) > 0.7:
    print(f"\n✅ CONCLUSION: EOF1 ≈ MEAN PROFILE with modifications")
    print(f"   → EOF1 mainly represents wind intensity with some shape changes")
    print(f"   → The shape changes might be the stability signal you're looking for")
    
else:
    print(f"\n✅ CONCLUSION: EOF1 has DISTINCT atmospheric pattern")
    print(f"   → EOF1 represents a specific physical process")
    print(f"   → Could be stability-related, diurnal, or synoptic patterns")
    print(f"   → Your original hypothesis may be correct!")

print(f"\n📊 SUPPORTING EVIDENCE:")
if abs(pc1_hour_corr) > 0.15:
    print(f"   • Strong diurnal pattern (r = {pc1_hour_corr:.3f}) → stability connection")
    
if abs(pc1_shear_corr) > 0.15:
    print(f"   • Significant wind shear correlation (r = {pc1_shear_corr:.3f})")

print(f"\n🎯 NEXT STEPS:")
if abs(correlations_with_mean[0]) > 0.8:
    print(f"   1. Analyze EOF2/EOF3 for pure stability patterns")
    print(f"   2. Test power prediction: high PC1 should favor all heights equally")
    print(f"   3. Look for height-dependent effects in other EOF modes")
else:
    print(f"   1. Investigate what physical process EOF1 represents")
    print(f"   2. Test connection to stability using EOF1 directly")
    print(f"   3. Analyze power prediction efficiency by PC1 levels")

print(f"\n📁 FILES SAVED:")
print(f"   • {plot_path}")
print(f"   • {summary_path}")
print(f"   • {detailed_path}")

print(f"\n" + "="*60)
print("ANALYSIS COMPLETE")
print("="*60)