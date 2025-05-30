#!/usr/bin/env python3
"""
Wind Direction Distribution Curves Comparison
Create plots showing only fitted curves comparing obs, ec, gfs at different heights
Layout: 1 row x 4 columns (10m, 30m, 50m, 70m)
Author: Research Team
Date: 2025-05-30
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import vonmises
from scipy.optimize import minimize
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# Set paths (adjust these paths according to your setup)
input_path = "/Users/xiaxin/work/WindForecast_Project/01_Data/processed/imputed_data/changma_imputed_complete.csv"
output_path = "/Users/xiaxin/work/WindForecast_Project/03_Results/风向序列/"

# Create output directory if it doesn't exist
Path(output_path).mkdir(parents=True, exist_ok=True)

def load_and_prepare_data():
    """Load and prepare the wind direction data"""
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

def circular_mean(angles_deg):
    """Calculate circular mean for wind directions"""
    angles_rad = np.radians(angles_deg)
    sin_sum = np.sum(np.sin(angles_rad))
    cos_sum = np.sum(np.cos(angles_rad))
    mean_rad = np.arctan2(sin_sum, cos_sum)
    mean_deg = np.degrees(mean_rad)
    return mean_deg % 360

def fit_von_mises_mixture(data, n_components=2):
    """Fit mixture of von Mises distributions"""
    try:
        clean_data = data.dropna()
        if len(clean_data) < 20:
            return None, None, None
        
        def von_mises_pdf_deg(x_deg, mu_deg, kappa):
            """von Mises PDF in degrees"""
            x_rad = np.radians(x_deg)
            mu_rad = np.radians(mu_deg)
            from scipy.special import i0  # Modified Bessel function
            return np.exp(kappa * np.cos(x_rad - mu_rad)) / (2 * np.pi * i0(kappa))
        
        def mixture_pdf(x_deg, params):
            """Mixture of von Mises PDFs"""
            if n_components == 2:
                w1, mu1, kappa1, mu2, kappa2 = params
                w2 = 1 - w1
                return w1 * von_mises_pdf_deg(x_deg, mu1, kappa1) + w2 * von_mises_pdf_deg(x_deg, mu2, kappa2)
            else:
                return von_mises_pdf_deg(x_deg, params[0], params[1])
        
        # Initialize parameters based on data
        hist, bin_edges = np.histogram(clean_data, bins=36, range=(0, 360))
        bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
        
        if n_components == 2:
            # Find two peaks
            peaks = []
            for i in range(1, len(hist)-1):
                if hist[i] > hist[i-1] and hist[i] > hist[i+1] and hist[i] > np.max(hist) * 0.2:
                    peaks.append((bin_centers[i], hist[i]))
            
            if len(peaks) >= 2:
                # Sort by height and take top 2
                peaks = sorted(peaks, key=lambda x: x[1], reverse=True)[:2]
                mu1_init, mu2_init = peaks[0][0], peaks[1][0]
            else:
                # Use circular mean and opposite direction
                circ_mean = circular_mean(clean_data)
                mu1_init = circ_mean
                mu2_init = (circ_mean + 180) % 360
            
            # Initial parameters [w1, mu1, kappa1, mu2, kappa2]
            initial_params = [0.5, mu1_init, 2.0, mu2_init, 2.0]
            
            def negative_log_likelihood(params):
                try:
                    w1, mu1, kappa1, mu2, kappa2 = params
                    if w1 <= 0 or w1 >= 1 or kappa1 <= 0 or kappa2 <= 0:
                        return 1e10
                    
                    pdf_vals = mixture_pdf(clean_data, params)
                    pdf_vals = np.maximum(pdf_vals, 1e-10)  # Avoid log(0)
                    return -np.sum(np.log(pdf_vals))
                except:
                    return 1e10
            
            # Constraints
            bounds = [(0.1, 0.9), (0, 360), (0.1, 50), (0, 360), (0.1, 50)]
            
            try:
                result = minimize(negative_log_likelihood, initial_params, bounds=bounds, method='L-BFGS-B')
                if result.success:
                    fitted_params = result.x
                else:
                    fitted_params = initial_params
            except:
                fitted_params = initial_params
        
        else:
            # Single component
            circ_mean = circular_mean(clean_data)
            fitted_params = [circ_mean, 2.0]
        
        # Generate smooth curve
        x_smooth = np.linspace(0, 360, 720)
        mixture_values = mixture_pdf(x_smooth, fitted_params)
        
        # Calculate R-squared
        hist_normalized = hist / np.sum(hist)
        theoretical = mixture_pdf(bin_centers, fitted_params)
        theoretical_normalized = theoretical / np.sum(theoretical) if np.sum(theoretical) > 0 else theoretical
        
        ss_res = np.sum((hist_normalized - theoretical_normalized) ** 2)
        ss_tot = np.sum((hist_normalized - np.mean(hist_normalized)) ** 2)
        r_squared = 1 - (ss_res / ss_tot) if ss_tot != 0 else 0
        
        return x_smooth, mixture_values, max(0, min(1, r_squared))
        
    except Exception as e:
        print(f"Fitting error: {e}")
        return None, None, None

def create_curves_comparison(df, output_path):
    """Create comparison plots with only fitted curves - 1x4 layout"""
    
    # Set up the plot - 1 row, 4 columns
    fig, axes = plt.subplots(1, 4, figsize=(20, 5))
    fig.suptitle('Wind Direction Distribution Curves Comparison (obs vs ec vs gfs)', 
                 fontsize=16, fontweight='bold', y=0.98)
    
    # Define models and heights
    models = ['obs', 'ec', 'gfs']
    heights = ['10m', '30m', '50m', '70m']
    
    # Define colors and line styles for different models
    model_styles = {
        'obs': {'color': 'red', 'linestyle': '-', 'linewidth': 2.5, 'label': 'OBS'},
        'ec': {'color': 'blue', 'linestyle': '-', 'linewidth': 2.5, 'label': 'EC'},
        'gfs': {'color': 'green', 'linestyle': '-', 'linewidth': 2.5, 'label': 'GFS'}
    }
    
    # Find global y-axis range for consistency
    all_max_values = []
    
    # First pass: calculate all curves and find max values
    curve_data = {}
    for height in heights:
        curve_data[height] = {}
        for model in models:
            col_name = f'{model}_wind_direction_{height}'
            if col_name in df.columns:
                data = df[col_name]
                x_smooth, mixture_values, r_squared = fit_von_mises_mixture(data, n_components=2)
                if x_smooth is not None and mixture_values is not None:
                    curve_data[height][model] = (x_smooth, mixture_values, r_squared)
                    all_max_values.append(np.max(mixture_values))
    
    # Set global y-axis limit
    global_y_max = max(all_max_values) * 1.1 if all_max_values else 0.01
    
    # Second pass: create plots
    for col, height in enumerate(heights):
        ax = axes[col]
        
        # Plot curves for each model
        for model in models:
            if height in curve_data and model in curve_data[height]:
                x_smooth, mixture_values, r_squared = curve_data[height][model]
                
                # Plot the curve
                style = model_styles[model]
                ax.plot(x_smooth, mixture_values, 
                       color=style['color'], 
                       linestyle=style['linestyle'], 
                       linewidth=style['linewidth'],
                       label=f"{style['label']} (R²={r_squared:.3f})",
                       alpha=0.8)
            else:
                print(f"Warning: No data found for {model} at {height}")
                # Add a placeholder to legend if needed
                style = model_styles[model]
                ax.plot([], [], 
                       color=style['color'], 
                       linestyle=style['linestyle'], 
                       linewidth=style['linewidth'],
                       label=f"{style['label']} (No Data)",
                       alpha=0.3)
        
        # Set up the plot
        ax.set_xlim(0, 360)
        ax.set_ylim(0, global_y_max)
        
        # X-axis: compass direction labels
        ax.set_xticks([0, 45, 90, 135, 180, 225, 270, 315, 360])
        ax.set_xticklabels(['N', 'NE', 'E', 'SE', 'S', 'SW', 'W', 'NW', 'N'], 
                          fontsize=11, fontweight='bold')
        
        # Labels and title
        if col == 0:  # Only show ylabel for the first subplot
            ax.set_ylabel('Probability Density', fontsize=12, fontweight='bold')
        
        ax.set_xlabel('Wind Direction', fontsize=12, fontweight='bold')
        ax.set_title(f'{height}', fontsize=14, fontweight='bold', pad=15)
        
        # Add grid
        ax.grid(True, alpha=0.3, linestyle='-', linewidth=0.5)
        
        # Add legend to each subplot
        ax.legend(loc='upper right', fontsize=10, framealpha=0.9)
        
        # Set minor ticks for more detailed x-axis
        minor_ticks = np.arange(0, 361, 30)  # Every 30 degrees
        ax.set_xticks(minor_ticks, minor=True)
        ax.tick_params(axis='x', which='minor', length=3)
    
    # Adjust layout
    plt.tight_layout(rect=[0, 0.05, 1, 0.93])
    plt.subplots_adjust(wspace=0.25)
    
    # Save the plot
    output_file = Path(output_path) / 'wind_direction_curves_comparison.png'
    plt.savefig(output_file, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"Curves comparison plot saved to: {output_file}")
    
    # Show the plot
    plt.show()
    
    return fig

def create_summary_statistics(df, output_path):
    """Create a summary table of R-squared values for all models and heights"""
    
    models = ['obs', 'ec', 'gfs']
    heights = ['10m', '30m', '50m', '70m']
    
    # Create summary table
    summary_data = []
    
    for model in models:
        row_data = {'Model': model.upper()}
        for height in heights:
            col_name = f'{model}_wind_direction_{height}'
            if col_name in df.columns:
                data = df[col_name]
                x_smooth, mixture_values, r_squared = fit_von_mises_mixture(data, n_components=2)
                row_data[height] = f"{r_squared:.4f}" if r_squared is not None else "N/A"
            else:
                row_data[height] = "N/A"
        summary_data.append(row_data)
    
    # Create DataFrame
    summary_df = pd.DataFrame(summary_data)
    
    # Save to CSV
    output_file = Path(output_path) / 'wind_direction_curves_rsquared_summary.csv'
    summary_df.to_csv(output_file, index=False)
    print(f"R-squared summary saved to: {output_file}")
    
    # Print summary
    print("\n=== R-squared Values Summary ===")
    print(summary_df.to_string(index=False))
    
    return summary_df

def main():
    """Main analysis function"""
    print("=== Wind Direction Distribution Curves Comparison ===")
    
    # Load data
    df = load_and_prepare_data()
    
    # Check available columns
    wind_direction_cols = [col for col in df.columns if 'wind_direction' in col]
    print(f"\nAvailable wind direction columns: {wind_direction_cols}")
    
    # Create curves comparison plots
    print("\nCreating wind direction curves comparison plots...")
    fig = create_curves_comparison(df, output_path)
    
    # Create summary statistics
    print("\nCreating summary statistics...")
    summary_df = create_summary_statistics(df, output_path)
    
    print("\n=== Analysis Complete ===")
    print(f"Results saved to: {output_path}")
    print("\nThe plots show:")
    print("- OBS: Red solid line")
    print("- EC: Blue solid line") 
    print("- GFS: Green solid line")
    print("- R² values shown in legend for model fit quality")

if __name__ == "__main__":
    main()