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
results_path = '/Users/xiaxin/work/WindForecast_Project/03_Results/08时间序列分解/EOF'

# Create results directory if it doesn't exist
os.makedirs(results_path, exist_ok=True)

# Load data
print("Loading data for EOF analysis...")
df = pd.read_csv(data_path)

# Convert datetime column
df['datetime'] = pd.to_datetime(df['datetime'])
df.set_index('datetime', inplace=True)

# Select wind speed variables at 4 heights
wind_variables = ['obs_wind_speed_10m', 'obs_wind_speed_30m', 'obs_wind_speed_50m', 'obs_wind_speed_70m']
heights = [10, 30, 50, 70]  # Heights in meters

print(f"Data shape: {df.shape}")
print(f"Date range: {df.index.min()} to {df.index.max()}")
print(f"Wind speed variables: {wind_variables}")

# Check data availability
for var in wind_variables:
    if var not in df.columns:
        print(f"Error: {var} not found in data")
        exit()

# Extract wind speed data for EOF analysis
wind_data = df[wind_variables].copy()

# Check for missing values
print(f"\nMissing values check:")
for var in wind_variables:
    missing = wind_data[var].isna().sum()
    print(f"  {var}: {missing} missing values")

# Remove rows with any missing values
wind_data_clean = wind_data.dropna()
print(f"\nAfter removing missing values: {len(wind_data_clean)} data points")

if len(wind_data_clean) < 100:
    print("Error: Not enough data for EOF analysis")
    exit()

# Function to perform EOF analysis
def perform_eof_analysis(data, n_components=4):
    """
    Perform EOF (Empirical Orthogonal Function) analysis
    
    Parameters:
    data: DataFrame with time × space dimensions
    n_components: number of EOF modes to extract
    
    Returns:
    Dictionary containing EOF results
    """
    print(f"\nPerforming EOF analysis...")
    print(f"Data matrix shape: {data.shape} (time × height)")
    
    # Standardize the data (remove mean, divide by std)
    data_std = (data - data.mean()) / data.std()
    
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
        'eof_patterns': eof_patterns,
        'time_coefficients': time_coefficients,
        'explained_variance': explained_variance,
        'cumulative_variance': np.cumsum(explained_variance),
        'eigenvalues': pca.explained_variance_,
        'pca_object': pca,
        'time_index': data.index
    }
    
    return results

# Function to plot EOF patterns
def plot_eof_patterns(eof_results, save_path):
    """
    Plot EOF spatial patterns (vertical profiles)
    """
    eof_patterns = eof_results['eof_patterns']
    explained_variance = eof_results['explained_variance']
    
    fig, axes = plt.subplots(1, 4, figsize=(16, 6))
    fig.suptitle('EOF Patterns (Vertical Wind Speed Profiles)', fontsize=16)
    
    for i in range(4):
        # Plot EOF pattern as vertical profile
        axes[i].plot(eof_patterns[i], heights, 'bo-', linewidth=2, markersize=8)
        axes[i].axvline(x=0, color='k', linestyle='--', alpha=0.5)
        axes[i].set_title(f'EOF {i+1}\n({explained_variance[i]*100:.1f}% variance)')
        axes[i].set_xlabel('EOF Loading')
        axes[i].set_ylabel('Height (m)')
        axes[i].grid(True, alpha=0.3)
        axes[i].set_ylim(0, 80)
        
        # Add interpretation text
        pattern_type = interpret_eof_pattern(eof_patterns[i])
        axes[i].text(0.02, 0.98, pattern_type, transform=axes[i].transAxes, 
                    verticalalignment='top', fontsize=10, 
                    bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()

# Function to interpret EOF patterns
def interpret_eof_pattern(pattern):
    """
    Provide physical interpretation of EOF patterns
    """
    # Check if all values have same sign (uniform pattern)
    if np.all(pattern > 0) or np.all(pattern < 0):
        return "Uniform\n(All heights\nvary together)"
    
    # Check if pattern increases/decreases with height
    correlation_with_height = np.corrcoef(pattern, heights)[0, 1]
    
    if correlation_with_height > 0.7:
        return "Shear increase\n(Upper levels\nstronger)"
    elif correlation_with_height < -0.7:
        return "Shear decrease\n(Lower levels\nstronger)"
    else:
        return "Mixed pattern\n(Complex\nvertical structure)"

# Function to plot time coefficients
def plot_time_coefficients(eof_results, save_path, plot_period_days=None):
    """
    Plot time coefficients for EOF modes
    """
    time_coefficients = eof_results['time_coefficients']
    time_index = eof_results['time_index']
    explained_variance = eof_results['explained_variance']
    
    # Plot subset of data for clarity, or all data if plot_period_days is None
    if plot_period_days:
        end_idx = min(plot_period_days * 24 * 4, len(time_index))  # 4 points per hour
        plot_time = time_index[:end_idx]
        plot_coeffs = time_coefficients[:end_idx]
        title_period = f"First {plot_period_days} days"
    else:
        plot_time = time_index
        plot_coeffs = time_coefficients
        title_period = f"Complete time series ({len(time_index)} points)"
    
    fig, axes = plt.subplots(4, 1, figsize=(20, 12))
    fig.suptitle(f'EOF Time Coefficients ({title_period})', fontsize=16)
    
    for i in range(4):
        axes[i].plot(plot_time, plot_coeffs[:, i], linewidth=0.8)
        axes[i].set_title(f'EOF {i+1} Time Coefficient ({explained_variance[i]*100:.1f}% variance)')
        axes[i].set_ylabel(f'PC{i+1}')
        axes[i].grid(True, alpha=0.3)
        axes[i].axhline(y=0, color='k', linestyle='--', alpha=0.5)
        
        # Add monthly ticks for better readability when showing full series
        if not plot_period_days:
            axes[i].tick_params(axis='x', rotation=45)
    
    axes[-1].set_xlabel('Date')
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()

# Function to analyze EOF statistics
def analyze_eof_statistics(eof_results):
    """
    Calculate and display EOF statistics
    """
    explained_variance = eof_results['explained_variance']
    cumulative_variance = eof_results['cumulative_variance']
    time_coefficients = eof_results['time_coefficients']
    
    stats_list = []
    
    for i in range(4):
        pc = time_coefficients[:, i]
        
        stats_dict = {
            'EOF_Mode': i + 1,
            'Explained_Variance_%': explained_variance[i] * 100,
            'Cumulative_Variance_%': cumulative_variance[i] * 100,
            'PC_Mean': np.mean(pc),
            'PC_Std': np.std(pc),
            'PC_Skewness': stats.skew(pc),
            'PC_Kurtosis': stats.kurtosis(pc),
            'PC_Min': np.min(pc),
            'PC_Max': np.max(pc)
        }
        
        stats_list.append(stats_dict)
    
    return pd.DataFrame(stats_list)

# Function to plot variance explained
def plot_variance_explained(eof_results, save_path):
    """
    Plot explained variance by each EOF mode
    """
    explained_variance = eof_results['explained_variance']
    cumulative_variance = eof_results['cumulative_variance']
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    # Individual variance
    bars = ax1.bar(range(1, 5), explained_variance * 100, alpha=0.7, color='skyblue')
    ax1.set_xlabel('EOF Mode')
    ax1.set_ylabel('Explained Variance (%)')
    ax1.set_title('Variance Explained by Each EOF Mode')
    ax1.set_xticks(range(1, 5))
    ax1.grid(True, alpha=0.3)
    
    # Add value labels on bars
    for i, bar in enumerate(bars):
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height + 0.5,
                f'{height:.1f}%', ha='center', va='bottom')
    
    # Cumulative variance
    ax2.plot(range(1, 5), cumulative_variance * 100, 'ro-', linewidth=2, markersize=8)
    ax2.set_xlabel('Number of EOF Modes')
    ax2.set_ylabel('Cumulative Explained Variance (%)')
    ax2.set_title('Cumulative Variance Explained')
    ax2.set_xticks(range(1, 5))
    ax2.grid(True, alpha=0.3)
    ax2.set_ylim(0, 105)
    
    # Add value labels
    for i, val in enumerate(cumulative_variance * 100):
        ax2.text(i + 1, val + 1, f'{val:.1f}%', ha='center', va='bottom')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()

# Function to calculate correlations with meteorological variables
def analyze_correlations_with_power(eof_results, power_data):
    """
    Analyze correlations between EOF time coefficients and power
    """
    time_coefficients = eof_results['time_coefficients']
    time_index = eof_results['time_index']
    
    # Align power data with EOF time index
    power_aligned = power_data.reindex(time_index).dropna()
    
    # Calculate correlations
    correlations = []
    
    for i in range(4):
        # Get overlapping time periods
        common_idx = power_aligned.index.intersection(time_index)
        if len(common_idx) > 100:
            pc_data = pd.Series(time_coefficients[:, i], index=time_index)
            pc_aligned = pc_data.reindex(common_idx)
            power_common = power_aligned.reindex(common_idx)
            
            correlation = np.corrcoef(pc_aligned, power_common)[0, 1]
            p_value = stats.pearsonr(pc_aligned, power_common)[1]
            
            correlations.append({
                'EOF_Mode': i + 1,
                'Correlation_with_Power': correlation,
                'P_Value': p_value,
                'Significance': 'Significant' if p_value < 0.05 else 'Not Significant'
            })
    
    return pd.DataFrame(correlations)

# Function to perform seasonal analysis
def analyze_seasonal_patterns(eof_results):
    """
    Analyze seasonal patterns in EOF time coefficients
    """
    time_coefficients = eof_results['time_coefficients']
    time_index = eof_results['time_index']
    
    # Create DataFrame with time coefficients
    pc_df = pd.DataFrame(time_coefficients, 
                        columns=[f'PC{i+1}' for i in range(4)],
                        index=time_index)
    
    # Add temporal variables
    pc_df['month'] = pc_df.index.month
    pc_df['season'] = pc_df['month'].map({12: 'Winter', 1: 'Winter', 2: 'Winter',
                                        3: 'Spring', 4: 'Spring', 5: 'Spring',
                                        6: 'Summer', 7: 'Summer', 8: 'Summer',
                                        9: 'Autumn', 10: 'Autumn', 11: 'Autumn'})
    pc_df['hour'] = pc_df.index.hour
    
    # Calculate seasonal means
    seasonal_means = pc_df.groupby('season')[['PC1', 'PC2', 'PC3', 'PC4']].mean()
    
    # Calculate diurnal means
    diurnal_means = pc_df.groupby('hour')[['PC1', 'PC2', 'PC3', 'PC4']].mean()
    
    return seasonal_means, diurnal_means

# Main EOF analysis
print("\n" + "="*60)
print("MULTI-LEVEL WIND SPEED EOF ANALYSIS")
print("="*60)

# Perform EOF analysis
eof_results = perform_eof_analysis(wind_data_clean)

# Display basic results
print(f"\nEOF Analysis Results:")
print(f"Data points analyzed: {len(wind_data_clean)}")
print(f"Number of EOF modes: 4")
print(f"Total variance explained by 4 modes: {eof_results['cumulative_variance'][-1]*100:.2f}%")

# Plot EOF patterns
print("\nGenerating EOF pattern plots...")
pattern_path = os.path.join(results_path, 'EOF_patterns.png')
plot_eof_patterns(eof_results, pattern_path)

# Plot time coefficients
print("Generating time coefficient plots...")
time_path = os.path.join(results_path, 'EOF_time_coefficients_complete.png')
plot_time_coefficients(eof_results, time_path, plot_period_days=None)  # Plot complete series

# Also plot first 30 days for detailed view
time_path_30d = os.path.join(results_path, 'EOF_time_coefficients_30days.png')
plot_time_coefficients(eof_results, time_path_30d, plot_period_days=30)  # Plot first 30 days

# Plot variance explained
print("Generating variance explanation plots...")
variance_path = os.path.join(results_path, 'EOF_variance_explained.png')
plot_variance_explained(eof_results, variance_path)

# Calculate and save statistics
print("Calculating EOF statistics...")
eof_stats = analyze_eof_statistics(eof_results)
stats_path = os.path.join(results_path, 'EOF_statistics.csv')
eof_stats.to_csv(stats_path, index=False)

print("\nEOF Statistics:")
print(eof_stats.round(3).to_string(index=False))

# Analyze correlations with power
if 'power' in df.columns:
    print("\nAnalyzing correlations with power...")
    power_data = df['power']
    power_corr = analyze_correlations_with_power(eof_results, power_data)
    power_corr_path = os.path.join(results_path, 'EOF_power_correlations.csv')
    power_corr.to_csv(power_corr_path, index=False)
    
    print("\nCorrelations with Power:")
    print(power_corr.round(3).to_string(index=False))

# Analyze seasonal patterns
print("\nAnalyzing seasonal and diurnal patterns...")
seasonal_means, diurnal_means = analyze_seasonal_patterns(eof_results)

# Save seasonal analysis
seasonal_path = os.path.join(results_path, 'EOF_seasonal_patterns.csv')
seasonal_means.to_csv(seasonal_path)

diurnal_path = os.path.join(results_path, 'EOF_diurnal_patterns.csv')
diurnal_means.to_csv(diurnal_path)

print("\nSeasonal Means:")
print(seasonal_means.round(3))

print("\nDiurnal Pattern (sample hours):")
print(diurnal_means.iloc[::6].round(3))  # Show every 6th hour

# Summary and interpretation
print("\n" + "="*60)
print("EOF ANALYSIS INTERPRETATION")
print("="*60)

explained_var = eof_results['explained_variance']
for i in range(4):
    print(f"\nEOF Mode {i+1} ({explained_var[i]*100:.1f}% variance):")
    pattern = eof_results['eof_patterns'][i]
    interpretation = interpret_eof_pattern(pattern)
    print(f"  Physical meaning: {interpretation.replace(chr(10), ' ')}")
    print(f"  Pattern values: {pattern}")

print(f"\n" + "="*60)
print("EOF ANALYSIS COMPLETE")
print("="*60)
print(f"Results saved to: {results_path}")
print("Files generated:")
print("  - EOF_patterns.png")
print("  - EOF_time_coefficients_complete.png")
print("  - EOF_time_coefficients_30days.png") 
print("  - EOF_variance_explained.png")
print("  - EOF_statistics.csv")
print("  - EOF_power_correlations.csv")
print("  - EOF_seasonal_patterns.csv")
print("  - EOF_diurnal_patterns.csv")

print(f"\nKey findings:")
print(f"  - First EOF explains {explained_var[0]*100:.1f}% of total variance")
print(f"  - First two EOFs explain {eof_results['cumulative_variance'][1]*100:.1f}% of total variance")
print(f"  - All four EOFs explain {eof_results['cumulative_variance'][3]*100:.1f}% of total variance")