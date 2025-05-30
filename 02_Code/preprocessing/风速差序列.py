#!/usr/bin/env python3
"""
Wind Speed Difference Analysis
Calculate and plot wind speed differences between different heights
Author: Research Team
Date: 2025-05-29
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
    """Load and prepare the wind speed data"""
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

def calculate_wind_speed_differences(df, model_prefix):
    """Calculate wind speed differences for a specific model"""
    
    # Define height columns based on model prefix
    height_cols = {
        '10m': f'{model_prefix}_wind_speed_10m',
        '30m': f'{model_prefix}_wind_speed_30m', 
        '50m': f'{model_prefix}_wind_speed_50m',
        '70m': f'{model_prefix}_wind_speed_70m'
    }
    
    # Check which columns exist
    existing_cols = {k: v for k, v in height_cols.items() if v in df.columns}
    
    if len(existing_cols) < 2:
        print(f"Warning: Not enough wind speed columns found for {model_prefix}")
        return pd.DataFrame()
    
    print(f"Found columns for {model_prefix}: {list(existing_cols.values())}")
    
    # Calculate differences
    differences = pd.DataFrame(index=df.index)
    
    # 70m - 10m
    if '70m' in existing_cols and '10m' in existing_cols:
        differences['70m-10m'] = df[existing_cols['70m']] - df[existing_cols['10m']]
    
    # 70m - 30m  
    if '70m' in existing_cols and '30m' in existing_cols:
        differences['70m-30m'] = df[existing_cols['70m']] - df[existing_cols['30m']]
    
    # 70m - 50m
    if '70m' in existing_cols and '50m' in existing_cols:
        differences['70m-50m'] = df[existing_cols['70m']] - df[existing_cols['50m']]
    
    # 30m - 10m
    if '30m' in existing_cols and '10m' in existing_cols:
        differences['30m-10m'] = df[existing_cols['30m']] - df[existing_cols['10m']]
    
    # 50m - 30m
    if '50m' in existing_cols and '30m' in existing_cols:
        differences['50m-30m'] = df[existing_cols['50m']] - df[existing_cols['30m']]
    
    return differences

def smooth_data_with_std(data, window='1D'):
    """Apply smoothing and calculate standard deviation"""
    smoothed = data.resample(window).mean()
    std = data.resample(window).std()
    return smoothed, std

def plot_wind_speed_differences(obs_diff, ec_diff, gfs_diff, output_path):
    """Create the main plot with three subplots"""
    
    # Set up the plot
    fig, axes = plt.subplots(3, 1, figsize=(15, 12))
    fig.suptitle('Wind Speed Differences Between Heights', fontsize=16, fontweight='bold')
    
    # Define colors for different height differences
    colors = {
        '70m-10m': '#1f77b4',  # blue
        '70m-30m': '#ff7f0e',  # orange
        '70m-50m': '#2ca02c',  # green
        '30m-10m': '#d62728',  # red
        '50m-30m': '#9467bd'   # purple
    }
    
    # Function to plot data for each model
    def plot_model_data(ax, diff_data, model_name, title):
        if diff_data.empty:
            ax.text(0.5, 0.5, f'No data available for {model_name}', 
                   ha='center', va='center', transform=ax.transAxes, fontsize=12)
            ax.set_title(title)
            return
        
        # Apply 1-day smoothing
        for col in diff_data.columns:
            if col in diff_data.columns and not diff_data[col].isna().all():
                # Smooth data
                smoothed, std = smooth_data_with_std(diff_data[[col]], '1D')
                
                if not smoothed.empty and not smoothed[col].isna().all():
                    # Plot main line
                    ax.plot(smoothed.index, smoothed[col], 
                           color=colors.get(col, 'black'), 
                           linewidth=2, label=col, alpha=0.8)
                    
                    # Add standard deviation shadow
                    if not std.empty and not std[col].isna().all():
                        upper = smoothed[col] + std[col]
                        lower = smoothed[col] - std[col]
                        ax.fill_between(smoothed.index, lower, upper, 
                                       color=colors.get(col, 'black'), 
                                       alpha=0.2)
        
        ax.set_title(title, fontsize=14, fontweight='bold')
        ax.set_ylabel('Wind Speed Difference (m/s)', fontsize=12)
        ax.grid(True, alpha=0.3)
        ax.legend(loc='upper right', fontsize=10)
        
        # Set y-axis limits to be consistent across subplots
        return ax.get_ylim()
    
    # Plot each model
    ylims = []
    
    # Subplot 1: Observations
    ylim1 = plot_model_data(axes[0], obs_diff, 'OBS', 'Observations (OBS)')
    if ylim1: ylims.append(ylim1)
    
    # Subplot 2: EC
    ylim2 = plot_model_data(axes[1], ec_diff, 'EC', 'EC Model')
    if ylim2: ylims.append(ylim2)
    
    # Subplot 3: GFS
    ylim3 = plot_model_data(axes[2], gfs_diff, 'GFS', 'GFS Model')
    if ylim3: ylims.append(ylim3)
    
    # Set consistent y-axis limits
    if ylims:
        global_ymin = min([y[0] for y in ylims])
        global_ymax = max([y[1] for y in ylims])
        for ax in axes:
            ax.set_ylim(global_ymin, global_ymax)
    
    # Set x-axis label for bottom subplot
    axes[2].set_xlabel('Date', fontsize=12)
    
    # Adjust layout
    plt.tight_layout()
    
    # Save the plot
    output_file = Path(output_path) / 'wind_speed_differences_comparison.png'
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"Plot saved to: {output_file}")
    
    # Show the plot
    plt.show()
    
    return fig

def main():
    """Main analysis function"""
    print("=== Wind Speed Difference Analysis ===")
    
    # Load data
    df = load_and_prepare_data()
    
    # Calculate differences for each model
    print("\nCalculating wind speed differences...")
    
    # Try different possible column prefixes for observations
    obs_prefixes = ['obs', 'OBS', 'observed', 'observation']
    obs_diff = pd.DataFrame()
    for prefix in obs_prefixes:
        obs_diff = calculate_wind_speed_differences(df, prefix)
        if not obs_diff.empty:
            break
    
    # EC model
    ec_diff = calculate_wind_speed_differences(df, 'ec')
    if ec_diff.empty:
        ec_diff = calculate_wind_speed_differences(df, 'EC')
    
    # GFS model
    gfs_diff = calculate_wind_speed_differences(df, 'gfs')
    if gfs_diff.empty:
        gfs_diff = calculate_wind_speed_differences(df, 'GFS')
    
    # Print summary statistics
    print("\n=== Summary Statistics ===")
    for name, data in [('OBS', obs_diff), ('EC', ec_diff), ('GFS', gfs_diff)]:
        if not data.empty:
            print(f"\n{name} Wind Speed Differences:")
            print(data.describe())
        else:
            print(f"\n{name}: No data available")
    
    # Create the plot
    print("\nCreating plots...")
    fig = plot_wind_speed_differences(obs_diff, ec_diff, gfs_diff, output_path)
    
    # Save difference data to CSV
    print("\nSaving difference data...")
    if not obs_diff.empty:
        obs_diff.to_csv(Path(output_path) / 'obs_wind_speed_differences.csv')
    if not ec_diff.empty:
        ec_diff.to_csv(Path(output_path) / 'ec_wind_speed_differences.csv')
    if not gfs_diff.empty:
        gfs_diff.to_csv(Path(output_path) / 'gfs_wind_speed_differences.csv')
    
    print("\n=== Analysis Complete ===")

if __name__ == "__main__":
    main()