#!/usr/bin/env python3
"""
Error Time Series Analysis
Calculate and plot model errors (EC-OBS and GFS-OBS) for different heights and temperature
Author: Research Team
Date: 2025-05-30
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# Set paths
input_path = "/Users/xiaxin/work/WindForecast_Project/01_Data/processed/imputed_data/changma_imputed_complete.csv"
output_path = "/Users/xiaxin/work/WindForecast_Project/03_Results/误差序列/"

# Create output directory if it doesn't exist
Path(output_path).mkdir(parents=True, exist_ok=True)

def load_and_prepare_data():
    """Load and prepare the data"""
    print("Loading data...")
    df = pd.read_csv(input_path)
    
    # Convert datetime column
    if 'datetime' in df.columns:
        df['datetime'] = pd.to_datetime(df['datetime'])
        df.set_index('datetime', inplace=True)
    elif 'time' in df.columns:
        df['time'] = pd.to_datetime(df['time'])
        df.set_index('time', inplace=True)
    
    print(f"Data shape: {df.shape}")
    print(f"Date range: {df.index.min()} to {df.index.max()}")
    
    return df

def calculate_model_errors(df):
    """Calculate model errors for wind speed and temperature"""
    
    heights = ['10m', '30m', '50m', '70m']
    errors = pd.DataFrame(index=df.index)
    
    print("Calculating wind speed errors...")
    # Wind speed errors for all heights
    for height in heights:
        obs_col = f'obs_wind_speed_{height}'
        ec_col = f'ec_wind_speed_{height}'
        gfs_col = f'gfs_wind_speed_{height}'
        
        if all(col in df.columns for col in [obs_col, ec_col, gfs_col]):
            errors[f'EC_WS_Error_{height}'] = df[ec_col] - df[obs_col]
            errors[f'GFS_WS_Error_{height}'] = df[gfs_col] - df[obs_col]
            print(f"  Wind speed errors calculated for {height}")
        else:
            print(f"  Missing columns for wind speed {height}")
    
    print("Calculating temperature errors...")
    # Temperature errors (10m only)
    obs_temp = 'obs_temperature_10m'
    ec_temp = 'ec_temperature_10m'
    gfs_temp = 'gfs_temperature_10m'
    
    if all(col in df.columns for col in [obs_temp, ec_temp, gfs_temp]):
        errors['EC_Temp_Error_10m'] = df[ec_temp] - df[obs_temp]
        errors['GFS_Temp_Error_10m'] = df[gfs_temp] - df[obs_temp]
        print("  Temperature errors calculated for 10m")
    else:
        print("  Missing columns for temperature")
    
    return errors

def smooth_data_with_std(data, window='1D'):
    """Apply smoothing and calculate standard deviation"""
    smoothed = data.resample(window).mean()
    std = data.resample(window).std()
    return smoothed, std

def plot_error_time_series(errors, output_path):
    """Create the main plot with 5 subplots for error time series"""
    
    # Set up the plot - 5 subplots (4 wind speed heights + 1 temperature)
    fig, axes = plt.subplots(5, 1, figsize=(16, 20))
    fig.suptitle('Model Error Time Series Analysis\n(EC-OBS and GFS-OBS with 1-day smoothing)', 
                 fontsize=18, fontweight='bold', y=0.98)
    
    # Define colors for models
    colors = {
        'EC': '#1f77b4',   # blue
        'GFS': '#ff7f0e'   # orange
    }
    
    # Heights for wind speed
    heights = ['10m', '30m', '50m', '70m']
    
    # Function to plot error data for each variable
    def plot_error_data(ax, error_data, variable_name, title, ylabel):
        if error_data.empty:
            ax.text(0.5, 0.5, f'No data available for {variable_name}', 
                   ha='center', va='center', transform=ax.transAxes, fontsize=12)
            ax.set_title(title, fontsize=14, fontweight='bold')
            return None
        
        ylims = []
        
        # Plot EC and GFS errors
        for model in ['EC', 'GFS']:
            # Find the corresponding error column
            error_cols = [col for col in error_data.columns if col.startswith(f'{model}_') and variable_name in col]
            
            if error_cols:
                col = error_cols[0]  # Take the first matching column
                if not error_data[col].isna().all():
                    # Apply 1-day smoothing
                    smoothed, std = smooth_data_with_std(error_data[[col]], '1D')
                    
                    if not smoothed.empty and not smoothed[col].isna().all():
                        # Plot main line
                        line = ax.plot(smoothed.index, smoothed[col], 
                                     color=colors[model], 
                                     linewidth=2.5, label=f'{model}-OBS', alpha=0.8)
                        
                        # Add standard deviation shadow
                        if not std.empty and not std[col].isna().all():
                            upper = smoothed[col] + std[col]
                            lower = smoothed[col] - std[col]
                            ax.fill_between(smoothed.index, lower, upper, 
                                           color=colors[model], 
                                           alpha=0.25)
                        
                        # Track y-limits for this subplot
                        current_ylim = ax.get_ylim()
                        ylims.append(current_ylim)
        
        # Add horizontal line at y=0
        ax.axhline(y=0, color='black', linestyle='--', alpha=0.5, linewidth=1)
        
        ax.set_title(title, fontsize=14, fontweight='bold', pad=10)
        ax.set_ylabel(ylabel, fontsize=12)
        ax.grid(True, alpha=0.3)
        ax.legend(loc='upper right', fontsize=11)
        
        # Return y-limits for potential global scaling
        return ylims
    
    # Plot wind speed errors for each height
    all_ylims = []
    for i, height in enumerate(heights):
        title = f'Wind Speed Error at {height}'
        ylabel = f'Wind Speed Error (m/s)\n{height}'
        ylims = plot_error_data(axes[i], errors, f'WS_Error_{height}', title, ylabel)
        if ylims:
            all_ylims.extend(ylims)
    
    # Plot temperature error
    temp_ylims = plot_error_data(axes[4], errors, 'Temp_Error_10m', 
                                'Temperature Error at 10m', 'Temperature Error (°C)\n10m')
    
    # Set x-axis label for bottom subplot
    axes[4].set_xlabel('Date', fontsize=12)
    
    # Optionally set consistent y-axis limits for wind speed subplots
    if all_ylims:
        # Calculate global y-limits for wind speed plots
        global_ymin = min([y[0] for y in all_ylims])
        global_ymax = max([y[1] for y in all_ylims])
        
        # Add some padding
        y_range = global_ymax - global_ymin
        global_ymin -= y_range * 0.1
        global_ymax += y_range * 0.1
        
        # Apply to wind speed subplots (first 4)
        for i in range(4):
            axes[i].set_ylim(global_ymin, global_ymax)
    
    # Adjust layout
    plt.tight_layout()
    plt.subplots_adjust(top=0.94)  # Make room for main title
    
    # Save the plot
    output_file = Path(output_path) / 'model_error_time_series.png'
    plt.savefig(output_file, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"Plot saved to: {output_file}")
    
    # Show the plot
    plt.show()
    
    return fig

def generate_error_statistics(errors, output_path):
    """Generate and save error statistics"""
    
    print("\n=== Error Statistics Summary ===")
    
    stats_summary = []
    
    # Analyze each error variable
    for col in errors.columns:
        if not errors[col].isna().all():
            stats = {
                'Variable': col,
                'Count': errors[col].count(),
                'Mean': errors[col].mean(),
                'Std': errors[col].std(),
                'RMSE': np.sqrt((errors[col] ** 2).mean()),
                'MAE': errors[col].abs().mean(),
                'Min': errors[col].min(),
                'Max': errors[col].max(),
                'Q25': errors[col].quantile(0.25),
                'Q50': errors[col].quantile(0.50),
                'Q75': errors[col].quantile(0.75)
            }
            stats_summary.append(stats)
            
            print(f"\n{col}:")
            print(f"  Count: {stats['Count']:,}")
            print(f"  Mean: {stats['Mean']:.3f}")
            print(f"  RMSE: {stats['RMSE']:.3f}")
            print(f"  MAE: {stats['MAE']:.3f}")
            print(f"  Std: {stats['Std']:.3f}")
    
    # Save statistics to CSV
    if stats_summary:
        stats_df = pd.DataFrame(stats_summary)
        stats_file = Path(output_path) / 'error_statistics.csv'
        stats_df.to_csv(stats_file, index=False)
        print(f"\nStatistics saved to: {stats_file}")
    
    return stats_summary

def main():
    """Main analysis function"""
    print("=== Model Error Time Series Analysis ===")
    
    # Load data
    df = load_and_prepare_data()
    
    # Calculate model errors
    print("\nCalculating model errors...")
    errors = calculate_model_errors(df)
    
    print(f"\nError variables calculated: {list(errors.columns)}")
    
    # Generate statistics
    print("\nGenerating error statistics...")
    stats = generate_error_statistics(errors, output_path)
    
    # Create the plot
    print("\nCreating error time series plot...")
    fig = plot_error_time_series(errors, output_path)
    
    # Save error data to CSV
    print("\nSaving error time series data...")
    errors_file = Path(output_path) / 'model_errors_time_series.csv'
    errors.to_csv(errors_file)
    print(f"Error data saved to: {errors_file}")
    
    print("\n=== Analysis Complete ===")
    print("Generated files:")
    print("- model_error_time_series.png")
    print("- model_errors_time_series.csv")
    print("- error_statistics.csv")

if __name__ == "__main__":
    main()