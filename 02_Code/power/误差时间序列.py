#!/usr/bin/env python3
"""
Wind Speed and Temperature Error Time Series Analysis
Create time series plots showing EC-OBS and GFS-OBS errors with 1-day smoothing + std bands
Layout: 2 rows (EC-OBS top, GFS-OBS bottom), showing wind speed errors at different heights and temperature error
Author: Research Team
Date: 2025-05-30
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# Set paths (adjust these paths according to your setup)
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

def calculate_errors(df):
    """Calculate errors between models and observations"""
    print("Calculating errors...")
    
    error_data = df.copy()
    heights = ['10m', '30m', '50m', '70m']
    
    # Calculate wind speed errors
    for height in heights:
        obs_col = f'obs_wind_speed_{height}'
        ec_col = f'ec_wind_speed_{height}'
        gfs_col = f'gfs_wind_speed_{height}'
        
        if all(col in df.columns for col in [obs_col, ec_col, gfs_col]):
            error_data[f'ec_obs_wind_speed_error_{height}'] = df[ec_col] - df[obs_col]
            error_data[f'gfs_obs_wind_speed_error_{height}'] = df[gfs_col] - df[obs_col]
            print(f"Wind speed error calculated for {height}")
        else:
            print(f"Warning: Missing wind speed columns for {height}")
    
    # Calculate temperature errors - use 10m temperature data
    obs_temp_col = 'obs_temperature_10m'
    ec_temp_col = 'ec_temperature_10m'
    gfs_temp_col = 'gfs_temperature_10m'
    
    if all(col in df.columns for col in [obs_temp_col, ec_temp_col, gfs_temp_col]):
        error_data['ec_obs_temperature_error'] = df[ec_temp_col] - df[obs_temp_col]
        error_data['gfs_obs_temperature_error'] = df[gfs_temp_col] - df[obs_temp_col]
        print("Temperature error calculated (10m height)")
    else:
        print("Warning: Missing temperature columns")
        print("Available columns containing 'temp':", [col for col in df.columns if 'temp' in col.lower()])
    
    return error_data

def smooth_with_std(series, window='1D'):
    """Apply rolling mean and calculate standard deviation for error bands"""
    # Remove NaN values for calculation
    clean_series = series.dropna()
    
    if len(clean_series) < 10:
        return None, None, None
    
    # Apply rolling statistics
    rolling_mean = clean_series.rolling(window, center=True).mean()
    rolling_std = clean_series.rolling(window, center=True).std()
    
    return rolling_mean, rolling_std, clean_series.index

def plot_error_time_series(df, output_path):
    """Create error time series plots"""
    
    # Calculate errors
    error_data = calculate_errors(df)
    
    # Set up the plot - 2 rows, 1 column
    fig, axes = plt.subplots(2, 1, figsize=(16, 12))
    fig.suptitle('Wind Speed and Temperature Error Time Series (1-day smoothing ± std)', 
                 fontsize=16, fontweight='bold', y=0.95)
    
    heights = ['10m', '30m', '50m', '70m']
    
    # Define colors for different variables
    colors = {
        '10m': '#1f77b4',  # blue
        '30m': '#ff7f0e',  # orange  
        '50m': '#2ca02c',  # green
        '70m': '#d62728',  # red
        'temperature': '#9467bd'  # purple for 10m temperature
    }
    
    # Plot EC-OBS errors (top subplot)
    ax1 = axes[0]
    ax1.set_title('EC - OBS Errors', fontsize=14, fontweight='bold', pad=15)
    
    # Plot wind speed errors for EC-OBS
    for height in heights:
        error_col = f'ec_obs_wind_speed_error_{height}'
        if error_col in error_data.columns:
            series = error_data[error_col]
            mean_smooth, std_smooth, time_index = smooth_with_std(series)
            
            if mean_smooth is not None:
                # Plot mean line
                ax1.plot(time_index, mean_smooth, 
                        color=colors[height], linewidth=2.5, 
                        label=f'Wind Speed {height}', alpha=0.8)
                
                # Plot std bands
                upper_band = mean_smooth + std_smooth
                lower_band = mean_smooth - std_smooth
                ax1.fill_between(time_index, lower_band, upper_band,
                               color=colors[height], alpha=0.2)
    
    # Plot temperature error for EC-OBS
    temp_error_col = 'ec_obs_temperature_error'
    if temp_error_col in error_data.columns:
        series = error_data[temp_error_col]
        mean_smooth, std_smooth, time_index = smooth_with_std(series)
        
        if mean_smooth is not None:
            # Plot mean line
            ax1.plot(time_index, mean_smooth, 
                    color=colors['temperature'], linewidth=2.5, 
                    label='Temperature 10m', alpha=0.8)
            
            # Plot std bands
            upper_band = mean_smooth + std_smooth
            lower_band = mean_smooth - std_smooth
            ax1.fill_between(time_index, lower_band, upper_band,
                           color=colors['temperature'], alpha=0.2)
    
    # Plot GFS-OBS errors (bottom subplot)
    ax2 = axes[1]
    ax2.set_title('GFS - OBS Errors', fontsize=14, fontweight='bold', pad=15)
    
    # Plot wind speed errors for GFS-OBS
    for height in heights:
        error_col = f'gfs_obs_wind_speed_error_{height}'
        if error_col in error_data.columns:
            series = error_data[error_col]
            mean_smooth, std_smooth, time_index = smooth_with_std(series)
            
            if mean_smooth is not None:
                # Plot mean line
                ax2.plot(time_index, mean_smooth, 
                        color=colors[height], linewidth=2.5, 
                        label=f'Wind Speed {height}', alpha=0.8)
                
                # Plot std bands
                upper_band = mean_smooth + std_smooth
                lower_band = mean_smooth - std_smooth
                ax2.fill_between(time_index, lower_band, upper_band,
                               color=colors[height], alpha=0.2)
    
    # Plot temperature error for GFS-OBS
    temp_error_col = 'gfs_obs_temperature_error'
    if temp_error_col in error_data.columns:
        series = error_data[temp_error_col]
        mean_smooth, std_smooth, time_index = smooth_with_std(series)
        
        if mean_smooth is not None:
            # Plot mean line
            ax2.plot(time_index, mean_smooth, 
                    color=colors['temperature'], linewidth=2.5, 
                    label='Temperature 10m', alpha=0.8)
            
            # Plot std bands
            upper_band = mean_smooth + std_smooth
            lower_band = mean_smooth - std_smooth
            ax2.fill_between(time_index, lower_band, upper_band,
                           color=colors['temperature'], alpha=0.2)
    
    # Format both subplots
    for ax in axes:
        # Add horizontal reference line at y=0
        ax.axhline(y=0, color='black', linestyle='--', alpha=0.5, linewidth=1)
        
        # Format x-axis
        ax.xaxis.set_major_locator(mdates.MonthLocator())
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
        ax.xaxis.set_minor_locator(mdates.DayLocator(interval=7))
        
        # Rotate x-axis labels
        plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha='right')
        
        # Add grid
        ax.grid(True, alpha=0.3, linestyle='-', linewidth=0.5)
        ax.grid(True, alpha=0.2, linestyle='-', linewidth=0.3, which='minor')
        
        # Add legend
        ax.legend(loc='upper right', fontsize=10, framealpha=0.9)
        
        # Labels
        ax.set_ylabel('Error (Model - Obs)', fontsize=12, fontweight='bold')
    
    # Only add x-label to bottom subplot
    axes[1].set_xlabel('Date', fontsize=12, fontweight='bold')
    
    # Adjust layout
    plt.tight_layout(rect=[0, 0.03, 1, 0.92])
    plt.subplots_adjust(hspace=0.25)
    
    # Save the plot
    output_file = Path(output_path) / 'wind_speed_temperature_error_time_series.png'
    plt.savefig(output_file, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"Error time series plot saved to: {output_file}")
    
    # Show the plot
    plt.show()
    
    return fig, error_data

def calculate_error_statistics(error_data, output_path):
    """Calculate and save error statistics"""
    
    heights = ['10m', '30m', '50m', '70m']
    stats_data = []
    
    # Wind speed error statistics
    for height in heights:
        for model in ['ec', 'gfs']:
            error_col = f'{model}_obs_wind_speed_error_{height}'
            if error_col in error_data.columns:
                errors = error_data[error_col].dropna()
                if len(errors) > 0:
                    stats = {
                        'Variable': f'Wind Speed {height}',
                        'Model': model.upper(),
                        'Mean_Error': errors.mean(),
                        'RMSE': np.sqrt((errors**2).mean()),
                        'MAE': errors.abs().mean(),
                        'Std': errors.std(),
                        'Bias': errors.mean(),
                        'Count': len(errors)
                    }
                    stats_data.append(stats)
    
    # Temperature error statistics
    for model in ['ec', 'gfs']:
        error_col = f'{model}_obs_temperature_error'
        if error_col in error_data.columns:
            errors = error_data[error_col].dropna()
            if len(errors) > 0:
                stats = {
                    'Variable': 'Temperature 2m',
                    'Model': model.upper(),
                    'Mean_Error': errors.mean(),
                    'RMSE': np.sqrt((errors**2).mean()),
                    'MAE': errors.abs().mean(),
                    'Std': errors.std(),
                    'Bias': errors.mean(),
                    'Count': len(errors)
                }
                stats_data.append(stats)
    
    # Create DataFrame
    stats_df = pd.DataFrame(stats_data)
    
    # Save statistics
    output_file = Path(output_path) / 'error_statistics_summary.csv'
    stats_df.to_csv(output_file, index=False)
    print(f"Error statistics saved to: {output_file}")
    
    # Print summary
    print("\n=== Error Statistics Summary ===")
    print(stats_df.round(4).to_string(index=False))
    
    return stats_df

def main():
    """Main analysis function"""
    print("=== Wind Speed and Temperature Error Time Series Analysis ===")
    
    # Load data
    df = load_and_prepare_data()
    
    # Check available columns
    wind_speed_cols = [col for col in df.columns if 'wind_speed' in col]
    temp_cols = [col for col in df.columns if 'temperature' in col]
    print(f"\nAvailable wind speed columns: {wind_speed_cols}")
    print(f"Available temperature columns: {temp_cols}")
    
    # Create error time series plots
    print("\nCreating error time series plots...")
    fig, error_data = plot_error_time_series(df, output_path)
    
    # Calculate error statistics
    print("\nCalculating error statistics...")
    stats_df = calculate_error_statistics(error_data, output_path)
    
    print("\n=== Analysis Complete ===")
    print(f"Results saved to: {output_path}")
    print("\nThe plots show:")
    print("- 1-day rolling mean of errors (solid lines)")
    print("- ±1 standard deviation bands (shaded areas)")
    print("- EC-OBS errors in top panel")
    print("- GFS-OBS errors in bottom panel")
    print("- Wind speeds at 10m, 30m, 50m, 70m heights")
    print("- Temperature at 10m height")

if __name__ == "__main__":
    main()