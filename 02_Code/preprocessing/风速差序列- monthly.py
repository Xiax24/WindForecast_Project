#!/usr/bin/env python3
"""
Error Time Series Analysis with Power Data
Calculate and plot model errors (EC-OBS and GFS-OBS) for different heights and temperature
Add power monthly smoothed data on secondary y-axis
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

def smooth_data_with_std(data, window='1M'):
    """Apply smoothing and calculate standard deviation"""
    smoothed = data.resample(window).mean()
    std = data.resample(window).std()
    return smoothed, std

def plot_error_time_series_with_power(errors, df, output_path):
    """Create the main plot with 5 subplots for error time series and power data"""
    
    # Set up the plot - 5 subplots (4 wind speed heights + 1 temperature)
    fig, axes = plt.subplots(5, 1, figsize=(18, 20))
    fig.suptitle('Model Error Time Series Analysis with Power Data\n(EC-OBS and GFS-OBS with monthly aggregation)', 
                 fontsize=18, fontweight='bold', y=0.98)
    
    # Define colors for models
    colors = {
        'EC': "#441fb4",   # blue
        'GFS': "#0ecbff",  # orange
        'Power': "#2ca02c" # green for power
    }
    
    # Heights for wind speed
    heights = ['10m', '30m', '50m', '70m']
    
    # Find power column
    power_cols = [col for col in df.columns if 'power' in col.lower()]
    power_col = power_cols[0] if power_cols else None
    print(f"Power column found: {power_col}")
    
    # Function to plot error data for each variable
    def plot_error_data_with_power(ax, error_data, variable_name, title, ylabel, show_legend=False):
        if error_data.empty:
            ax.text(0.5, 0.5, f'No data available for {variable_name}', 
                   ha='center', va='center', transform=ax.transAxes, fontsize=12)
            ax.set_title(title, fontsize=14, fontweight='bold')
            return None
        
        ylims = []
        
        # Collect data for both models first
        model_data = {}
        for model in ['EC', 'GFS']:
            error_cols = [col for col in error_data.columns if col.startswith(f'{model}_') and variable_name in col]
            
            if error_cols:
                col = error_cols[0]
                if not error_data[col].isna().all():
                    # Apply 1-month smoothing
                    smoothed, std = smooth_data_with_std(error_data[[col]], '1M')
                    
                    if not smoothed.empty and not smoothed[col].isna().all():
                        model_data[model] = {
                            'dates': smoothed.index,
                            'values': smoothed[col],
                            'errors': std[col] if not std.empty else None
                        }
        
        # Plot bars for each model
        if model_data:
            # Get common date range
            all_dates = set()
            for model_info in model_data.values():
                all_dates.update(model_info['dates'])
            all_dates = sorted(list(all_dates))
            
            # Calculate bar positions
            x_positions = np.arange(len(all_dates))
            
            for i, (model, data_dict) in enumerate(model_data.items()):
                # Align data to common dates
                aligned_values = []
                aligned_errors = []
                
                for date in all_dates:
                    if date in data_dict['dates']:
                        idx = list(data_dict['dates']).index(date)
                        aligned_values.append(data_dict['values'].iloc[idx])
                        if data_dict['errors'] is not None:
                            aligned_errors.append(data_dict['errors'].iloc[idx])
                        else:
                            aligned_errors.append(0)
                    else:
                        aligned_values.append(np.nan)
                        aligned_errors.append(np.nan)
                
                # Calculate bar positions with offset for side-by-side bars
                bar_offset = (i - 0.5) * 0.35  # Offset for side-by-side bars
                x_pos = x_positions + bar_offset
                
                # Create bars
                bars = ax.bar(x_pos, aligned_values, 
                             width=0.35, 
                             color=colors[model], 
                             alpha=0.7,
                             label=f'{model}-OBS',
                             edgecolor=None,
                             linewidth=0.5)
                
                # Add error bars
                valid_mask = ~np.isnan(aligned_values)
                if any(valid_mask):
                    error_values = [e if not np.isnan(e) else 0 for e in aligned_errors]
                    ax.errorbar(x_pos[valid_mask], 
                               np.array(aligned_values)[valid_mask],
                               yerr=np.array(error_values)[valid_mask],
                               fmt='none',
                               color="#d1d1d1",
                               capsize=3,
                               capthick=1,
                               alpha=0.8)
                
                # Track y-limits
                valid_values = np.array(aligned_values)[valid_mask]
                valid_errors = np.array(error_values)[valid_mask]
                if len(valid_values) > 0:
                    y_max = np.max(valid_values + valid_errors)
                    y_min = np.min(valid_values - valid_errors)
                    ylims.append((y_min, y_max))
            
            # Set x-axis
            ax.set_xticks(x_positions)
            ax.set_xticklabels([date.strftime('%Y-%m') for date in all_dates], 
                              rotation=45, ha='right')
            
            # Add horizontal line at y=0
            ax.axhline(y=0, color='red', linestyle='--', alpha=0.8, linewidth=1.5)
        
        # Add power data on secondary y-axis
        if power_col and power_col in df.columns:
            ax2 = ax.twinx()
            
            if not df[power_col].isna().all():
                # Apply 1-month smoothing to power data
                power_smoothed, power_std = smooth_data_with_std(df[[power_col]], '1M')
                
                if not power_smoothed.empty and not power_smoothed[power_col].isna().all():
                    # Align power data to the same dates as error data
                    if model_data:  # Only if we have error data
                        aligned_power_values = []
                        aligned_power_errors = []
                        
                        for date in all_dates:
                            # Find closest date in power data
                            closest_date = None
                            min_diff = float('inf')
                            for pdate in power_smoothed.index:
                                diff = abs((date - pdate).days)
                                if diff < min_diff:
                                    min_diff = diff
                                    closest_date = pdate
                            
                            if closest_date is not None and min_diff <= 15:  # Within 15 days
                                power_idx = power_smoothed.index.get_loc(closest_date)
                                aligned_power_values.append(power_smoothed[power_col].iloc[power_idx])
                                if not power_std.empty:
                                    aligned_power_errors.append(power_std[power_col].iloc[power_idx])
                                else:
                                    aligned_power_errors.append(0)
                            else:
                                aligned_power_values.append(np.nan)
                                aligned_power_errors.append(np.nan)
                        
                        # Plot power line with error band
                        valid_power_mask = ~np.isnan(aligned_power_values)
                        if any(valid_power_mask):
                            x_pos_power = x_positions[valid_power_mask]
                            power_values = np.array(aligned_power_values)[valid_power_mask]
                            power_errors = np.array(aligned_power_errors)[valid_power_mask]
                            
                            # Main power line
                            ax2.plot(x_pos_power, power_values, 
                                    color=colors['Power'], linewidth=3, 
                                    marker='s', markersize=6, alpha=0.8,
                                    label='Power')
                            
                            # Power error band (std)
                            if not all(np.isnan(power_errors)) and len(power_errors) > 0:
                                ax2.fill_between(x_pos_power, 
                                               power_values - power_errors,
                                               power_values + power_errors,
                                               color=colors['Power'], alpha=0.1)
            
            # Set power axis properties
            ax2.set_ylabel('Power (MW)', fontsize=12, color=colors['Power'])
            ax2.tick_params(axis='y', labelcolor=colors['Power'])
            ax2.grid(False)
        
        # Set labels and formatting for main axis
        ax.set_title(title, fontsize=14, fontweight='bold', pad=10)
        ax.set_ylabel(ylabel, fontsize=12)
        ax.grid(False)
        
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
        
        # Return y-limits for potential global scaling
        return ylims
    
    # Plot wind speed errors for each height
    all_ylims = []
    for i, height in enumerate(heights):
        title = f'Wind Speed Error at {height}'
        ylabel = f'Wind Speed Error (m/s)\n{height}'
        show_legend = (i == 0)  # Only show legend in first subplot
        ylims = plot_error_data_with_power(axes[i], errors, f'WS_Error_{height}', title, ylabel, show_legend)
        if ylims:
            all_ylims.extend(ylims)
    
    # Plot temperature error
    temp_ylims = plot_error_data_with_power(axes[4], errors, 'Temp_Error_10m', 
                                'Temperature Error at 10m', 'Temperature Error (°C)\n10m', False)
    
    # Set x-axis label for bottom subplot
    axes[4].set_xlabel('Month', fontsize=12)
    
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
    output_file = Path(output_path) / 'model_error_power_monthly_bars.png'
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
    print("=== Model Error Time Series Analysis with Power ===")
    
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
    print("\nCreating error time series plot with power data...")
    fig = plot_error_time_series_with_power(errors, df, output_path)
    
    # Save error data to CSV
    print("\nSaving error time series data...")
    errors_file = Path(output_path) / 'model_errors_time_series.csv'
    errors.to_csv(errors_file)
    print(f"Error data saved to: {errors_file}")
    
    print("\n=== Analysis Complete ===")
    print("Generated files:")
    print("- model_error_power_monthly_bars.png")
    print("- model_errors_time_series.csv")
    print("- error_statistics.csv")

if __name__ == "__main__":
    main()