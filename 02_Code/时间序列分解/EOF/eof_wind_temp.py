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
results_path = '/Users/xiaxin/work/WindForecast_Project/03_Results/08时间序列分解/Joint_Wind_Temp_EOF'

# Create results directory if it doesn't exist
os.makedirs(results_path, exist_ok=True)

# Load data
print("Loading data for Joint Wind-Temperature EOF analysis...")
df = pd.read_csv(data_path)

# Convert datetime column
df['datetime'] = pd.to_datetime(df['datetime'])
df.set_index('datetime', inplace=True)

# Select 4 wind speed variables + 1 temperature variable
joint_variables = ['obs_wind_speed_10m', 'obs_wind_speed_30m', 'obs_wind_speed_50m', 'obs_wind_speed_70m', 'obs_temperature_10m']
variable_types = ['Wind', 'Wind', 'Wind', 'Wind', 'Temperature']
heights = [10, 30, 50, 70, 10]  # Heights in meters

print(f"Data shape: {df.shape}")
print(f"Date range: {df.index.min()} to {df.index.max()}")
print(f"Joint variables: {joint_variables}")

# Check data availability
for var in joint_variables:
    if var not in df.columns:
        print(f"Error: {var} not found in data")
        exit()

# Extract data for joint EOF analysis
joint_data = df[joint_variables].copy()

# Check for missing values
print(f"\nMissing values check:")
for var in joint_variables:
    missing = joint_data[var].isna().sum()
    print(f"  {var}: {missing} missing values")

# Remove rows with any missing values
joint_data_clean = joint_data.dropna()
print(f"\nAfter removing missing values: {len(joint_data_clean)} data points")

if len(joint_data_clean) < 100:
    print("Error: Not enough data for joint EOF analysis")
    exit()

# Function to perform joint EOF analysis
def perform_joint_eof_analysis(data, n_components=5):
    """
    Perform EOF analysis on joint wind speed and temperature data
    """
    print(f"\nPerforming Joint Wind-Temperature EOF analysis...")
    print(f"Data matrix shape: {data.shape} (time × variables)")
    
    # Standardize the data (important for joint analysis of different units)
    data_std = (data - data.mean()) / data.std()
    
    # Store means and stds for interpretation
    data_means = data.mean()
    data_stds = data.std()
    
    # Perform PCA (equivalent to EOF for standardized data)
    pca = PCA(n_components=n_components)
    principal_components = pca.fit_transform(data_std)
    
    # EOF patterns (spatial patterns)
    eof_patterns = pca.components_
    
    # Time coefficients (temporal patterns)
    time_coefficients = principal_components
    
    # Explained variance
    explained_variance = pca.explained_variance_ratio_
    
    # Create results dictionary
    results = {
        'data_original': data,
        'data_standardized': data_std,
        'data_means': data_means,
        'data_stds': data_stds,
        'eof_patterns': eof_patterns,
        'time_coefficients': time_coefficients,
        'explained_variance': explained_variance,
        'cumulative_variance': np.cumsum(explained_variance),
        'eigenvalues': pca.explained_variance_,
        'pca_object': pca,
        'time_index': data.index,
        'variable_names': joint_variables
    }
    
    return results

# Function to plot joint EOF patterns
def plot_joint_eof_patterns(eof_results, save_path):
    """
    Plot joint EOF patterns for wind speed and temperature
    """
    eof_patterns = eof_results['eof_patterns']
    explained_variance = eof_results['explained_variance']
    variable_names = eof_results['variable_names']
    
    fig, axes = plt.subplots(1, 5, figsize=(20, 6))
    fig.suptitle('Joint Wind-Temperature EOF Patterns', fontsize=16)
    
    for i in range(5):
        # Plot wind speed components
        wind_heights = [10, 30, 50, 70]
        wind_loadings = eof_patterns[i][:4]
        temp_loading = eof_patterns[i][4]
        
        # Plot wind profile on main axis
        axes[i].plot(wind_loadings, wind_heights, 'bo-', linewidth=2, markersize=8, label='Wind Speed')
        axes[i].axvline(x=0, color='k', linestyle='--', alpha=0.5)
        axes[i].set_xlabel('Wind Speed Loading')
        axes[i].set_ylabel('Height (m)')
        axes[i].set_ylim(0, 80)
        axes[i].grid(True, alpha=0.3)
        
        # Plot temperature point at 10m height on the same axis
        # Scale temperature loading to be visible with wind loadings
        temp_scaled = temp_loading * 0.5  # Scale factor to make it visible
        temp_point = axes[i].scatter([temp_scaled], [10], color='red', s=300, marker='*', 
                                   label=f'Temperature (scaled)', zorder=10, 
                                   edgecolor='black', linewidth=1)
        
        # Add temperature value as text annotation
        axes[i].text(0.02, 0.85, f'Temp Loading:\n{temp_loading:.3f}', 
                    transform=axes[i].transAxes, fontsize=10, color='red', fontweight='bold',
                    bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        # Add title and variance info
        axes[i].set_title(f'EOF {i+1}\n({explained_variance[i]*100:.1f}% variance)')
        
        # Add interpretation text
        interpretation = interpret_joint_eof_pattern(eof_patterns[i])
        axes[i].text(0.02, 0.02, interpretation, transform=axes[i].transAxes, 
                    verticalalignment='bottom', fontsize=9, 
                    bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))
        
        # Add legend for the first subplot
        if i == 0:
            axes[i].legend(loc='upper right')
    
    plt.tight_layout()
    
    # Save with lower DPI to avoid size issues
    try:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Pattern plot saved to: {save_path}")
    except Exception as e:
        print(f"Error saving pattern plot: {e}")
        # Try with even lower DPI
        plt.savefig(save_path, dpi=100, bbox_inches='tight')
    
    plt.show()

# Function to interpret joint EOF patterns
def interpret_joint_eof_pattern(pattern):
    """
    Provide physical interpretation of joint EOF patterns
    """
    wind_pattern = pattern[:4]
    temp_loading = pattern[4]
    
    # Analyze wind component
    if np.all(wind_pattern > 0) or np.all(wind_pattern < 0):
        wind_type = "Uniform wind"
    elif np.corrcoef(wind_pattern, heights[:4])[0, 1] > 0.7:
        wind_type = "Wind shear+"
    elif np.corrcoef(wind_pattern, heights[:4])[0, 1] < -0.7:
        wind_type = "Wind shear-"
    else:
        wind_type = "Complex wind"
    
    # Analyze temperature-wind relationship
    wind_mean = np.mean(wind_pattern)
    
    if (wind_mean > 0 and temp_loading > 0) or (wind_mean < 0 and temp_loading < 0):
        temp_wind_relation = "Positive coupling"
    elif (wind_mean > 0 and temp_loading < 0) or (wind_mean < 0 and temp_loading > 0):
        temp_wind_relation = "Negative coupling"
    else:
        temp_wind_relation = "Weak coupling"
    
    return f"{wind_type}\n{temp_wind_relation}"

# Function to plot joint time coefficients
def plot_joint_time_coefficients(eof_results, save_path, plot_period_days=None):
    """
    Plot time coefficients for joint EOF modes
    """
    time_coefficients = eof_results['time_coefficients']
    time_index = eof_results['time_index']
    explained_variance = eof_results['explained_variance']
    
    # Plot subset or all data
    if plot_period_days:
        end_idx = min(plot_period_days * 24 * 4, len(time_index))
        plot_time = time_index[:end_idx]
        plot_coeffs = time_coefficients[:end_idx]
        title_period = f"First {plot_period_days} days"
    else:
        plot_time = time_index
        plot_coeffs = time_coefficients
        title_period = f"Complete time series"
    
    fig, axes = plt.subplots(5, 1, figsize=(20, 15))
    fig.suptitle(f'Joint Wind-Temperature EOF Time Coefficients ({title_period})', fontsize=16)
    
    for i in range(5):
        axes[i].plot(plot_time, plot_coeffs[:, i], linewidth=0.8)
        axes[i].set_title(f'Joint EOF {i+1} Time Coefficient ({explained_variance[i]*100:.1f}% variance)')
        axes[i].set_ylabel(f'PC{i+1}')
        axes[i].grid(True, alpha=0.3)
        axes[i].axhline(y=0, color='k', linestyle='--', alpha=0.5)
        
        if not plot_period_days:
            axes[i].tick_params(axis='x', rotation=45)
    
    axes[-1].set_xlabel('Date')
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()

# Function to analyze joint EOF statistics
def analyze_joint_eof_statistics(eof_results):
    """
    Calculate statistics for joint EOF analysis
    """
    explained_variance = eof_results['explained_variance']
    cumulative_variance = eof_results['cumulative_variance']
    time_coefficients = eof_results['time_coefficients']
    eof_patterns = eof_results['eof_patterns']
    
    stats_list = []
    
    for i in range(5):
        pc = time_coefficients[:, i]
        pattern = eof_patterns[i]
        
        # Analyze wind-temperature coupling
        wind_loadings = pattern[:4]
        temp_loading = pattern[4]
        
        # Calculate wind-temperature coupling more safely
        wind_mean = np.mean(wind_loadings)
        
        # Simple coupling measure: same sign = positive coupling
        if (wind_mean > 0 and temp_loading > 0) or (wind_mean < 0 and temp_loading < 0):
            wind_temp_coupling = abs(wind_mean * temp_loading)  # Positive coupling strength
        elif (wind_mean > 0 and temp_loading < 0) or (wind_mean < 0 and temp_loading > 0):
            wind_temp_coupling = -abs(wind_mean * temp_loading)  # Negative coupling strength
        else:
            wind_temp_coupling = 0.0  # No coupling
        
        stats_dict = {
            'EOF_Mode': i + 1,
            'Explained_Variance_%': explained_variance[i] * 100,
            'Cumulative_Variance_%': cumulative_variance[i] * 100,
            'Wind_Loading_Mean': np.mean(wind_loadings),
            'Wind_Loading_Std': np.std(wind_loadings),
            'Temp_Loading': temp_loading,
            'Wind_Temp_Coupling': wind_temp_coupling,
            'PC_Mean': np.mean(pc),
            'PC_Std': np.std(pc),
            'PC_Skewness': stats.skew(pc),
            'PC_Kurtosis': stats.kurtosis(pc)
        }
        
        stats_list.append(stats_dict)
    
    return pd.DataFrame(stats_list)

# Function to analyze correlations with power
def analyze_joint_correlations_with_power(eof_results, power_data):
    """
    Analyze correlations between joint EOF time coefficients and power
    """
    time_coefficients = eof_results['time_coefficients']
    time_index = eof_results['time_index']
    
    # Align power data with EOF time index
    power_aligned = power_data.reindex(time_index).dropna()
    
    correlations = []
    
    for i in range(5):
        # Get overlapping time periods
        common_idx = power_aligned.index.intersection(time_index)
        if len(common_idx) > 100:
            pc_data = pd.Series(time_coefficients[:, i], index=time_index)
            pc_aligned = pc_data.reindex(common_idx)
            power_common = power_aligned.reindex(common_idx)
            
            correlation = np.corrcoef(pc_aligned, power_common)[0, 1]
            p_value = stats.pearsonr(pc_aligned, power_common)[1]
            
            correlations.append({
                'Joint_EOF_Mode': i + 1,
                'Correlation_with_Power': correlation,
                'P_Value': p_value,
                'Significance': 'Significant' if p_value < 0.05 else 'Not Significant'
            })
    
    return pd.DataFrame(correlations)

# Function to compare with wind-only EOF
def compare_with_wind_only_eof(joint_results, wind_data):
    """
    Compare joint EOF with wind-only EOF to see temperature's contribution
    """
    print("\nComparing with wind-only EOF...")
    
    # Perform wind-only EOF
    wind_std = (wind_data - wind_data.mean()) / wind_data.std()
    pca_wind = PCA(n_components=4)
    wind_pcs = pca_wind.fit_transform(wind_std)
    wind_explained = pca_wind.explained_variance_ratio_
    
    # Compare explained variance
    comparison = []
    for i in range(4):
        comparison.append({
            'Mode': i + 1,
            'Wind_Only_Variance_%': wind_explained[i] * 100,
            'Joint_Variance_%': joint_results['explained_variance'][i] * 100,
            'Variance_Change_%': (joint_results['explained_variance'][i] - wind_explained[i]) * 100
        })
    
    return pd.DataFrame(comparison)

# Main joint EOF analysis
print("\n" + "="*70)
print("JOINT WIND SPEED AND TEMPERATURE EOF ANALYSIS")
print("="*70)

# Perform joint EOF analysis
joint_eof_results = perform_joint_eof_analysis(joint_data_clean)

# Display basic results
print(f"\nJoint EOF Analysis Results:")
print(f"Variables analyzed: {len(joint_variables)} ({4} wind speeds + {1} temperature)")
print(f"Data points analyzed: {len(joint_data_clean)}")
print(f"Total variance explained by 5 modes: {joint_eof_results['cumulative_variance'][-1]*100:.2f}%")

# Plot joint EOF patterns
print("\nGenerating joint EOF pattern plots...")
pattern_path = os.path.join(results_path, 'Joint_EOF_patterns.png')
plot_joint_eof_patterns(joint_eof_results, pattern_path)

# Plot time coefficients - both complete and 30-day views
print("Generating joint time coefficient plots...")
time_path_complete = os.path.join(results_path, 'Joint_EOF_time_coefficients_complete.png')
plot_joint_time_coefficients(joint_eof_results, time_path_complete, plot_period_days=None)

time_path_30d = os.path.join(results_path, 'Joint_EOF_time_coefficients_30days.png')
plot_joint_time_coefficients(joint_eof_results, time_path_30d, plot_period_days=30)

# Calculate and save statistics
print("Calculating joint EOF statistics...")
joint_stats = analyze_joint_eof_statistics(joint_eof_results)
stats_path = os.path.join(results_path, 'Joint_EOF_statistics.csv')
joint_stats.to_csv(stats_path, index=False)

print("\nJoint EOF Statistics:")
print(joint_stats.round(3).to_string(index=False))

# Analyze correlations with power
if 'power' in df.columns:
    print("\nAnalyzing correlations with power...")
    power_data = df['power']
    joint_power_corr = analyze_joint_correlations_with_power(joint_eof_results, power_data)
    power_corr_path = os.path.join(results_path, 'Joint_EOF_power_correlations.csv')
    joint_power_corr.to_csv(power_corr_path, index=False)
    
    print("\nJoint EOF Correlations with Power:")
    print(joint_power_corr.round(3).to_string(index=False))

# Compare with wind-only EOF
wind_only_data = joint_data_clean[joint_variables[:4]]  # First 4 are wind speeds
comparison_df = compare_with_wind_only_eof(joint_eof_results, wind_only_data)
comparison_path = os.path.join(results_path, 'Joint_vs_Wind_Only_comparison.csv')
comparison_df.to_csv(comparison_path, index=False)

print("\nComparison: Joint EOF vs Wind-Only EOF:")
print(comparison_df.round(3).to_string(index=False))

# Summary and interpretation
print("\n" + "="*70)
print("JOINT EOF ANALYSIS INTERPRETATION")
print("="*70)

explained_var = joint_eof_results['explained_variance']
eof_patterns = joint_eof_results['eof_patterns']

for i in range(5):
    print(f"\nJoint EOF Mode {i+1} ({explained_var[i]*100:.1f}% variance):")
    
    wind_pattern = eof_patterns[i][:4]
    temp_loading = eof_patterns[i][4]
    
    print(f"  Wind loadings: {wind_pattern}")
    print(f"  Temperature loading: {temp_loading:.3f}")
    
    interpretation = interpret_joint_eof_pattern(eof_patterns[i])
    print(f"  Physical meaning: {interpretation.replace(chr(10), ' ')}")

print(f"\n" + "="*70)
print("JOINT WIND-TEMPERATURE EOF ANALYSIS COMPLETE")
print("="*70)
print(f"Results saved to: {results_path}")
print("Files generated:")
print("  - Joint_EOF_patterns.png")
print("  - Joint_EOF_time_coefficients_complete.png")
print("  - Joint_EOF_time_coefficients_30days.png")
print("  - Joint_EOF_statistics.csv")
print("  - Joint_EOF_power_correlations.csv")
print("  - Joint_vs_Wind_Only_comparison.csv")

print(f"\nKey findings:")
print(f"  - First joint EOF explains {explained_var[0]*100:.1f}% of total variance")
print(f"  - Temperature contribution to variance patterns")
print(f"  - Wind-temperature coupling mechanisms revealed")