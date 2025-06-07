import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.seasonal import STL
import warnings
import os
warnings.filterwarnings('ignore')

# Set paths
data_path = '/Users/xiaxin/work/WindForecast_Project/01_Data/processed/imputed_data/changma_clean.csv'
results_path = '/Users/xiaxin/work/WindForecast_Project/03_Results/08时间序列分解/STL'

# Create results directory if it doesn't exist
os.makedirs(results_path, exist_ok=True)

# Load data
print("Loading data...")
df = pd.read_csv(data_path)

# Convert datetime column
df['datetime'] = pd.to_datetime(df['datetime'])
df.set_index('datetime', inplace=True)

# Select variables for analysis
variables = ['obs_wind_speed_10m', 'obs_wind_speed_70m', 'power']

print(f"Data shape: {df.shape}")
print(f"Date range: {df.index.min()} to {df.index.max()}")
print(f"Variables to analyze: {variables}")

# Function to perform STL decomposition
def perform_stl_decomposition(data, variable_name, seasonal_period=96, trend_length=None):
    """
    Perform STL decomposition on time series data
    
    Parameters:
    data: pandas Series with datetime index
    variable_name: string name of the variable
    seasonal_period: int, seasonal period (default 96 for daily cycle in 15-min data)
    trend_length: int, trend smoothing length (default None for auto)
    
    Returns:
    STL decomposition result
    """
    print(f"\nPerforming STL decomposition for {variable_name}...")
    
    # Remove missing values
    clean_data = data.dropna()
    
    if len(clean_data) < seasonal_period * 2:
        print(f"Warning: Not enough data for {variable_name}")
        return None
    
    # Set parameters for different time scales
    # For 15-minute data:
    # - Daily cycle: 96 points (24 hours * 4)
    # - Weekly cycle: 672 points (7 days * 96)
    # - Monthly cycle: ~2880 points (30 days * 96)
    # - Seasonal cycle: ~8640 points (90 days * 96)
    
    # Set seasonal parameter (must be odd and >= 3)
    seasonal_param = 7  # Controls seasonal smoothing
    
    # Set trend length to capture longer-term patterns
    # Should be longer than seasonal cycle to capture annual trends
    if trend_length is None:
        # Set to capture seasonal patterns (about 3 months worth of data)
        trend_length = seasonal_period * 90  # ~3 months for annual trend
        # Ensure it's odd
        if trend_length % 2 == 0:
            trend_length += 1
    
    print(f"  Seasonal period: {seasonal_period} (daily cycle)")
    print(f"  Trend length: {trend_length} (for seasonal/annual trends)")
    print(f"  Seasonal smoothing: {seasonal_param}")
    
    # Perform STL decomposition
    stl = STL(clean_data, 
              seasonal=seasonal_param, 
              trend=trend_length,
              period=seasonal_period,
              robust=True)  # Robust to outliers
    result = stl.fit()
    
# Function to perform multi-scale STL decomposition
def perform_multiscale_stl(data, variable_name):
    """
    Perform STL decomposition at multiple time scales
    """
    print(f"\nPerforming multi-scale STL decomposition for {variable_name}...")
    
    results = {}
    
    # 1. Daily pattern (primary analysis)
    print("  Analyzing daily patterns...")
    daily_result = perform_stl_decomposition(data, f"{variable_name}_daily", 
                                           seasonal_period=daily_period,
                                           trend_length=weekly_period*4)  # Monthly trend
    if daily_result:
        results['daily'] = daily_result
    
    # 2. Weekly pattern (analyze trend component from daily)
    if daily_result and len(daily_result.trend.dropna()) > weekly_period * 2:
        print("  Analyzing weekly patterns from daily trend...")
        weekly_result = perform_stl_decomposition(daily_result.trend, f"{variable_name}_weekly",
                                                seasonal_period=weekly_period,
                                                trend_length=monthly_period)  # Seasonal trend
        if weekly_result:
            results['weekly'] = weekly_result
    
    return results

# Function to plot STL results
def plot_stl_results(stl_result, original_data, variable_name, save_path):
    """
    Plot STL decomposition results
    """
    fig, axes = plt.subplots(4, 1, figsize=(15, 12))
    fig.suptitle(f'STL Decomposition - {variable_name}', fontsize=16, y=0.98)
    
    # Original data
    axes[0].plot(original_data.index, original_data.values, color='blue', linewidth=0.8)
    axes[0].set_title('Original Data')
    axes[0].set_ylabel('Value')
    axes[0].grid(True, alpha=0.3)
    
    # Trend
    axes[1].plot(stl_result.trend.index, stl_result.trend.values, color='red', linewidth=1.0)
    axes[1].set_title('Trend Component')
    axes[1].set_ylabel('Trend')
    axes[1].grid(True, alpha=0.3)
    
    # Seasonal
    axes[2].plot(stl_result.seasonal.index, stl_result.seasonal.values, color='green', linewidth=0.8)
    axes[2].set_title('Seasonal Component')
    axes[2].set_ylabel('Seasonal')
    axes[2].grid(True, alpha=0.3)
    
    # Residual
    axes[3].plot(stl_result.resid.index, stl_result.resid.values, color='orange', linewidth=0.8)
    axes[3].set_title('Residual Component')
    axes[3].set_ylabel('Residual')
    axes[3].set_xlabel('Date')
    axes[3].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()

# Function to calculate decomposition statistics
def calculate_decomposition_stats(stl_result, original_data, variable_name):
    """
    Calculate statistics for STL decomposition
    """
    # Reconstruction
    reconstructed = stl_result.trend + stl_result.seasonal + stl_result.resid
    
    # Remove NaN values for fair comparison
    valid_idx = ~(original_data.isna() | reconstructed.isna())
    orig_clean = original_data[valid_idx]
    recon_clean = reconstructed[valid_idx]
    
    # Calculate metrics
    mse = np.mean((orig_clean - recon_clean) ** 2)
    rmse = np.sqrt(mse)
    mae = np.mean(np.abs(orig_clean - recon_clean))
    
    # Variance explained by each component
    total_var = np.var(orig_clean)
    trend_var = np.var(stl_result.trend.dropna())
    seasonal_var = np.var(stl_result.seasonal.dropna())
    resid_var = np.var(stl_result.resid.dropna())
    
    stats = {
        'Variable': variable_name,
        'RMSE': rmse,
        'MAE': mae,
        'Total_Variance': total_var,
        'Trend_Variance': trend_var,
        'Seasonal_Variance': seasonal_var,
        'Residual_Variance': resid_var,
        'Trend_Variance_Ratio': trend_var / total_var * 100,
        'Seasonal_Variance_Ratio': seasonal_var / total_var * 100,
        'Residual_Variance_Ratio': resid_var / total_var * 100
    }
    
    return stats

# Main analysis
decomposition_results = {}
stats_list = []

# Determine seasonal period based on data frequency
freq = pd.infer_freq(df.index[:100])  # Infer from first 100 points
print(f"Inferred frequency: {freq}")

# Calculate actual time interval
time_diff = df.index[1] - df.index[0]
print(f"Time interval: {time_diff}")

# Calculate actual time interval
time_diff = df.index[1] - df.index[0]
print(f"Time interval: {time_diff}")

# Set seasonal periods for 15-minute data
# Daily cycle: 24 hours * 4 (15-min intervals per hour) = 96 intervals
daily_period = 96  
# Weekly cycle: 7 days * 96 = 672 intervals  
weekly_period = 672
# Monthly cycle: 30 days * 96 = 2880 intervals
monthly_period = 2880

print(f"Daily period: {daily_period} intervals (24 hours)")
print(f"Weekly period: {weekly_period} intervals (7 days)")  
print(f"Monthly period: {monthly_period} intervals (30 days)")

# Use daily period as primary seasonal pattern
seasonal_period = daily_period

for var in variables:
    print(f"\n{'='*50}")
    print(f"Analyzing variable: {var}")
    print(f"{'='*50}")
    
    # Check data availability
    if var not in df.columns:
        print(f"Warning: {var} not found in data")
        continue
    
    # Get data for this variable
    var_data = df[var]
    print(f"Data points: {len(var_data)}")
    print(f"Missing values: {var_data.isna().sum()}")
    print(f"Value range: {var_data.min():.2f} to {var_data.max():.2f}")
    
    # Perform STL decomposition
    stl_result = perform_stl_decomposition(var_data, var, seasonal_period)
    
    if stl_result is not None:
        # Store result
        decomposition_results[var] = stl_result
        
        # Plot results
        plot_path = os.path.join(results_path, f'STL_decomposition_{var}.png')
        plot_stl_results(stl_result, var_data.dropna(), var, plot_path)
        
        # Calculate statistics
        stats = calculate_decomposition_stats(stl_result, var_data, var)
        stats_list.append(stats)
        
        print(f"\nDecomposition completed for {var}")
        print(f"Plot saved to: {plot_path}")

# Create summary statistics table
if stats_list:
    stats_df = pd.DataFrame(stats_list)
    
    # Round numerical columns
    numerical_cols = stats_df.select_dtypes(include=[np.number]).columns
    stats_df[numerical_cols] = stats_df[numerical_cols].round(4)
    
    # Save statistics
    stats_path = os.path.join(results_path, 'STL_decomposition_statistics.csv')
    stats_df.to_csv(stats_path, index=False)
    
    print(f"\n{'='*60}")
    print("STL DECOMPOSITION SUMMARY STATISTICS")
    print(f"{'='*60}")
    print(stats_df.to_string(index=False))
    print(f"\nStatistics saved to: {stats_path}")

# Create a comparison plot showing all three variables
if len(decomposition_results) > 0:
    fig, axes = plt.subplots(len(variables), 4, figsize=(20, 5*len(variables)))
    if len(variables) == 1:
        axes = axes.reshape(1, -1)
    
    for i, var in enumerate(variables):
        if var in decomposition_results:
            result = decomposition_results[var]
            original = df[var].dropna()
            
            # Original
            axes[i, 0].plot(original.index, original.values, 'b-', linewidth=0.8)
            axes[i, 0].set_title(f'{var} - Original')
            axes[i, 0].grid(True, alpha=0.3)
            
            # Trend
            axes[i, 1].plot(result.trend.index, result.trend.values, 'r-', linewidth=1.0)
            axes[i, 1].set_title(f'{var} - Trend')
            axes[i, 1].grid(True, alpha=0.3)
            
            # Seasonal
            axes[i, 2].plot(result.seasonal.index, result.seasonal.values, 'g-', linewidth=0.8)
            axes[i, 2].set_title(f'{var} - Seasonal')
            axes[i, 2].grid(True, alpha=0.3)
            
            # Residual
            axes[i, 3].plot(result.resid.index, result.resid.values, 'orange', linewidth=0.8)
            axes[i, 3].set_title(f'{var} - Residual')
            axes[i, 3].grid(True, alpha=0.3)
    
    plt.tight_layout()
    comparison_path = os.path.join(results_path, 'STL_comparison_all_variables.png')
    plt.savefig(comparison_path, dpi=300, bbox_inches='tight')
    plt.show()
    
    print(f"\nComparison plot saved to: {comparison_path}")

# Analysis summary
print(f"\n{'='*60}")
print("STL DECOMPOSITION ANALYSIS COMPLETE")
print(f"{'='*60}")
print(f"Total variables analyzed: {len(decomposition_results)}")
print(f"Results saved to: {results_path}")
print("\nFiles generated:")
for var in decomposition_results.keys():
    print(f"  - STL_decomposition_{var}.png")
print("  - STL_comparison_all_variables.png")
print("  - STL_decomposition_statistics.csv")