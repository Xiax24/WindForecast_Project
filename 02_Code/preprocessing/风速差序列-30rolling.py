#!/usr/bin/env python3
"""
Error Time Series Analysis with Power Data (Rolling Window)
Calculate and plot model errors (EC-OBS and GFS-OBS) for different heights and temperature
Use rolling window smoothing like daily_multi_source.py
Author: Research Team
Date: 2025-05-30
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import matplotlib.dates as mdates
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

def calculate_daily_window(df):
    """Calculate daily data points based on data frequency"""
    if len(df) <= 1:
        return 24  # Default hourly data
    
    time_diff = df.index[1] - df.index[0]
    if time_diff.total_seconds() == 0:
        return 24
    
    points_per_day = pd.Timedelta(days=1) / time_diff
    
    if points_per_day < 1:
        points_per_day = 24
        
    return max(1, int(points_per_day))

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

def rolling_smooth_with_std(data, window_days=30, center=True, min_periods=None):
    """Apply rolling window smoothing and calculate standard deviation"""
    # Calculate window size based on data frequency
    if hasattr(data, 'index') and len(data.index) > 1:
        time_diff = data.index[1] - data.index[0]
        points_per_day = pd.Timedelta(days=1) / time_diff
        window_size = max(1, int(window_days * points_per_day))
    else:
        window_size = window_days * 24  # Default hourly data
    
    # Set default min_periods for monthly smoothing
    if min_periods is None:
        min_periods = max(int(window_size / 4), 1)
    
    smoothed = data.rolling(
        window=window_size, 
        center=center, 
        min_periods=min_periods
    ).mean()
    
    std = data.rolling(
        window=window_size, 
        center=center, 
        min_periods=min_periods
    ).std()
    
    return smoothed, std

def plot_error_time_series_with_power_rolling(errors, df, output_path):
    """Create the main plot with 5 subplots for error time series and power data using rolling smoothing"""
    
    # Set up the plot - 5 subplots (4 wind speed heights + 1 temperature)
    fig, axes = plt.subplots(5, 1, figsize=(18, 20))
    fig.suptitle('Model Error Time Series Analysis with Power Data\n(30-Day Rolling Window Smoothing)', 
                 fontsize=18, fontweight='bold', y=0.98)
    
    # Define colors for models
    colors = {
        'EC': "#441fb4",   # blue
        'GFS': "#ff8b0e",  # orange
        'Power': "#2ca02c" # green for power
    }
    
    # Heights for wind speed
    heights = ['10m', '30m', '50m', '70m']
    
    # Find power column
    power_cols = [col for col in df.columns if 'power' in col.lower()]
    power_col = power_cols[0] if power_cols else None
    print(f"Power column found: {power_col}")
    
    # Function to plot error data for each variable
    def plot_error_data_with_power_rolling(ax, error_data, variable_name, title, ylabel, show_legend=False):
        if error_data.empty:
            ax.text(0.5, 0.5, f'No data available for {variable_name}', 
                   ha='center', va='center', transform=ax.transAxes, fontsize=12)
            ax.set_title(title, fontsize=14, fontweight='bold')
            return None
        
        # Plot error lines for each model
        for model in ['EC', 'GFS']:
            # Find the corresponding error column
            error_cols = [col for col in error_data.columns if col.startswith(f'{model}_') and variable_name in col]
            
            if error_cols:
                col = error_cols[0]
                if not error_data[col].isna().all():
                    # Apply 30-day rolling smoothing
                    smoothed, std = rolling_smooth_with_std(error_data[col], window_days=30)
                    
                    if not smoothed.isna().all():
                        # Plot standard deviation shadow
                        if not std.isna().all():
                            ax.fill_between(error_data.index, 
                                           smoothed - std, 
                                           smoothed + std, 
                                           color=colors[model], 
                                           alpha=0.2,
                                           label=f'{model}-OBS ±1σ' if show_legend else "")
                        
                        # Plot main error line
                        ax.plot(error_data.index, smoothed, 
                               color=colors[model], 
                               linewidth=2.5, 
                               alpha=0.9,
                               label=f'{model}-OBS')
        
        # Add horizontal line at y=0
        ax.axhline(y=0, color='red', linestyle='--', alpha=0.8, linewidth=1.5)
        
        # Add power data on secondary y-axis
        if power_col and power_col in df.columns:
            ax2 = ax.twinx()
            
            if not df[power_col].isna().all():
                # Apply 30-day rolling smoothing to power data
                power_smoothed, power_std = rolling_smooth_with_std(df[power_col], window_days=30)
                
                if not power_smoothed.isna().all():
                    # Plot power standard deviation shadow
                    if not power_std.isna().all():
                        ax2.fill_between(df.index, 
                                       power_smoothed - power_std,
                                       power_smoothed + power_std,
                                       color=colors['Power'], alpha=0.2,
                                       label='Power ±1σ' if show_legend else "")
                    
                    # Plot main power line
                    ax2.plot(df.index, power_smoothed, 
                            color=colors['Power'], 
                            linewidth=3, 
                            alpha=0.8,
                            label='Power')
                    
                    # Set power axis properties
                    ax2.set_ylabel('Power (MW)', fontsize=12, color=colors['Power'])
                    ax2.tick_params(axis='y', labelcolor=colors['Power'])
                    ax2.grid(False)
        
        # Set labels and formatting for main axis
        ax.set_title(title, fontsize=14, fontweight='bold', pad=10)
        ax.set_ylabel(ylabel, fontsize=12)
        ax.grid(True, alpha=0.3)
        
        # Set x-axis date format
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
        ax.xaxis.set_major_locator(mdates.MonthLocator(interval=1))
        plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha='right')
        
        # Handle legends
        if show_legend:
            # Get handles and labels from both axes
            handles1, labels1 = ax.get_legend_handles_labels()
            handles2, labels2 = [], []
            
            if power_col and power_col in df.columns:
                try:
                    handles2, labels2 = ax2.get_legend_handles_labels()
                except:
                    pass
            
            # Combine legends
            all_handles = handles1 + handles2
            all_labels = labels1 + labels2
            
            if all_handles:
                ax.legend(all_handles, all_labels, loc='upper right', fontsize=11)
        
        return None
    
    # Plot wind speed errors for each height
    for i, height in enumerate(heights):
        title = f'Wind Speed Error at {height}'
        ylabel = f'Wind Speed Error (m/s)'
        show_legend = (i == 0)  # Only show legend in first subplot
        plot_error_data_with_power_rolling(axes[i], errors, f'WS_Error_{height}', title, ylabel, show_legend)
    
    # Plot temperature error
    plot_error_data_with_power_rolling(axes[4], errors, 'Temp_Error_10m', 
                                'Temperature Error at 10m', 'Temperature Error (°C)', False)
    
    # Set x-axis label for bottom subplot
    axes[4].set_xlabel('Date', fontsize=12)
    
    # Adjust layout
    plt.tight_layout()
    plt.subplots_adjust(top=0.94)  # Make room for main title
    
    # Save the plot
    output_file = Path(output_path) / 'model_error_power_rolling_30day.png'
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
    print("=== Model Error Time Series Analysis with Rolling Window ===")
    
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
    print("\nCreating error time series plot with rolling window smoothing...")
    fig = plot_error_time_series_with_power_rolling(errors, df, output_path)
    
    # Save error data to CSV
    print("\nSaving error time series data...")
    errors_file = Path(output_path) / 'model_errors_time_series.csv'
    errors.to_csv(errors_file)
    print(f"Error data saved to: {errors_file}")
    
    print("\n=== Analysis Complete ===")
    print("Generated files:")
    print("- model_error_power_rolling_30day.png")
    print("- model_errors_time_series.csv")
    print("- error_statistics.csv")

if __name__ == "__main__":
    main()