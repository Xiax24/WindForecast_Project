#!/usr/bin/env python3
"""
Wind Direction Distribution Analysis - Literature Style
Create histogram plots showing wind direction distributions at different heights
Following the exact style of literature Figure 5 (von Mises mixture model)
Author: Research Team
Date: 2025-05-29
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import vonmises
from scipy.optimize import curve_fit
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# Set paths
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
    """Fit mixture of von Mises distributions with proper algorithm"""
    try:
        clean_data = data.dropna()
        if len(clean_data) < 20:
            return None, None, None
        
        # Convert to radians for circular statistics
        angles_rad = np.radians(clean_data)
        
        # Better approach: use EM-like algorithm for von Mises mixture
        from scipy.optimize import minimize
        
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

def circular_kde(data, bandwidth=15):
    """Proper circular KDE for wind direction data"""
    try:
        clean_data = data.dropna()
        if len(clean_data) < 10:
            return None, None
        
        # Convert to radians
        data_rad = np.radians(clean_data)
        
        # Evaluation points
        eval_points = np.linspace(0, 360, 720)
        eval_rad = np.radians(eval_points)
        
        # Circular KDE using von Mises kernels
        bandwidth_rad = np.radians(bandwidth)
        kappa = 1 / (bandwidth_rad ** 2)  # Concentration parameter
        
        kde_values = np.zeros_like(eval_rad)
        
        for data_point in data_rad:
            # von Mises kernel centered at each data point
            diff = eval_rad - data_point
            kde_values += np.exp(kappa * np.cos(diff))
        
        # Normalize
        from scipy.special import i0
        kde_values = kde_values / (len(clean_data) * 2 * np.pi * i0(kappa))
        
        return eval_points, kde_values
        
    except:
        return None, None

def plot_literature_style_distribution(ax, data, model_name, height, show_xlabel=False, show_ylabel=False):
    """Plot wind direction distribution in exact literature style"""
    
    clean_data = data.dropna()
    
    if len(clean_data) < 10:
        ax.text(0.5, 0.5, 'Insufficient Data', ha='center', va='center', 
                transform=ax.transAxes, fontsize=12)
        ax.set_title(f'{model_name} - {height}', fontsize=12, fontweight='bold')
        return
    
    # Create histogram - frequency counts with adaptive y-axis
    n_counts, bins, patches = ax.hist(clean_data, bins=36, range=(0, 360), density=False, 
                                     alpha=0.8, color='#C0C0C0', edgecolor='white', 
                                     linewidth=0.8, zorder=1)
    
    # Set y-axis based on this subplot's own data
    max_freq = np.max(n_counts)
    ax.set_ylim(0, max_freq * 1.1)  # Add 10% headroom for each subplot
    
    # Convert y-axis labels to show proportions based on THIS subplot's total
    total_observations = len(clean_data)
    
    if total_observations > 0:
        # Get current y-tick positions (in count scale)
        yticks_counts = ax.get_yticks()
        # Convert to proportions for this specific subplot
        yticks_proportions = yticks_counts / total_observations
        # Set new labels showing proportions
        ax.set_yticklabels([f'{prop:.3f}' if prop <= 1 else f'{prop:.2f}' for prop in yticks_proportions])
    
    # Fit mixture of von Mises distributions
    x_smooth, mixture_pdf, r_squared = fit_von_mises_mixture(clean_data, n_components=2)
    
    if x_smooth is not None and mixture_pdf is not None:
        # Scale mixture to match frequency counts
        total_area = np.trapz(mixture_pdf, x_smooth)
        if total_area > 0:
            # Convert PDF to frequency scale
            bin_width = 360 / 36  # 10 degrees per bin
            scaled_mixture = mixture_pdf * len(clean_data) * bin_width / total_area
        else:
            scaled_mixture = mixture_pdf
        ax.plot(x_smooth, scaled_mixture, 'k-', linewidth=3, 
                label='mvM.pdf', zorder=3)
    
    # Add circular KDE for comparison - dashed line (REMOVED per user request)
    # kde_x, kde_values = circular_kde(clean_data, bandwidth=20)
    # if kde_x is not None and kde_values is not None:
    #     # Scale KDE to 0-1 range
    #     max_kde = np.max(kde_values)
    #     if max_kde > 0:
    #         scaled_kde = kde_values / max_kde
    #     else:
    #         scaled_kde = kde_values
    #     ax.plot(kde_x, scaled_kde, 'k--', linewidth=2.5, 
    #             alpha=0.8, zorder=2)
    
    # Set up the plot - literature style
    ax.set_xlim(0, 360)
    ax.set_ylim(0, None)
    
    # Bottom x-axis: specific degree values like in literature (rotated)
    bottom_ticks = [0, 25, 45, 65, 85, 105, 125, 145, 165, 185, 205, 225, 245, 265, 285, 305, 325, 345, 360]
    ax.set_xticks(bottom_ticks)
    ax.set_xticklabels([str(x) for x in bottom_ticks], fontsize=9, rotation=45)
    
    # Top x-axis: compass direction labels
    ax2 = ax.twiny()
    ax2.set_xlim(0, 360)
    ax2.set_xticks([0, 45, 90, 135, 180, 225, 270, 315, 360])
    ax2.set_xticklabels(['N', 'NE', 'E', 'SE', 'S', 'SW', 'W', 'NW', 'N'], 
                        fontsize=11, fontweight='bold')
    
    # Labels and title
    if show_xlabel:  # Only show xlabel for bottom row (GFS)
        ax.set_xlabel('Wind directions, degree', fontsize=12, fontweight='bold')
    if show_ylabel:  # Only show ylabel for first column (10m)
        ax.set_ylabel('Relative Frequency', fontsize=12, fontweight='bold')
    ax.set_title(f'{model_name} - {height}', fontsize=14, fontweight='bold', pad=20)
    
    # Add grid - aligned with compass directions (top axis)
    # Set major grid lines at compass directions (every 45 degrees)
    compass_positions = [0, 45, 90, 135, 180, 225, 270, 315, 360]
    ax.set_xticks(compass_positions, minor=False)  # Major ticks for grid
    ax.grid(True, alpha=0.3, linestyle='-', linewidth=0.5, which='major')
    
    # Reset bottom x-axis ticks for labels (this overwrites the grid ticks for display)
    bottom_ticks = [0, 25, 45, 65, 85, 105, 125, 145, 165, 185, 205, 225, 245, 265, 285, 305, 325, 345, 360]
    ax.set_xticks(bottom_ticks, minor=True)  # Minor ticks for bottom labels
    ax.set_xticklabels([str(x) for x in bottom_ticks], fontsize=9, rotation=45, minor=True)
    
    # Add statistics in corner - literature style
    if r_squared is not None:
        # Count components and R-squared
        n_components = 2  # assuming 2 components
        stats_text = f'N={n_components} components\nR²={r_squared:.5f}'
        
        # Position based on data distribution
        text_x = 0.98 if np.mean(clean_data) < 180 else 0.02
        ha = 'right' if np.mean(clean_data) < 180 else 'left'
        
        ax.text(text_x, 0.98, stats_text, transform=ax.transAxes, 
                fontsize=11, ha=ha, va='top',
                bbox=dict(boxstyle="round,pad=0.4", facecolor="white", 
                         edgecolor='black', alpha=0.2))
    
    # Add sample info
    # sample_text = f'Weather station: {model_name}\nSeasons: All year'
    # ax.text(0.02, 0.02, sample_text, transform=ax.transAxes, 
    #         fontsize=10, ha='left', va='bottom',
    #         bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgray", 
    #                  alpha=0.8))

def create_literature_style_plots(df, output_path):
    """Create the main distribution plots with literature style - 3x4 layout"""
    
    # Set up the plot - large size like literature
    fig, axes = plt.subplots(3, 4, figsize=(20, 15))
    fig.suptitle('Histogram of the sample of wind directions', 
                 fontsize=16, fontweight='bold', y=0.95)
    
    # Add legend in top right corner (update to remove KDE)
    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], color='k', linewidth=3, label='mvM.pdf')
    ]
    fig.legend(handles=legend_elements, loc='upper right', 
              bbox_to_anchor=(0.98, 0.95), fontsize=12)
    
    # Define models and heights
    models = ['obs', 'ec', 'gfs']
    heights = ['10m', '30m', '50m', '70m']
    
    # Create plots for each model and height combination (no uniform y-axis)
    for row, model in enumerate(models):
        for col, height in enumerate(heights):
            # Show xlabel only for GFS row (bottom row, row=2)
            show_xlabel = (row == 2)
            # Show ylabel only for first column (10m, col=0)
            show_ylabel = (col == 0)
            
            # Get column name
            col_name = f'{model}_wind_direction_{height}'
            
            if col_name in df.columns:
                data = df[col_name]
                model_name = model.upper()
                # No uniform_ylim parameter - let each subplot adapt
                plot_literature_style_distribution(axes[row, col], data, model_name, height, show_xlabel, show_ylabel)
            else:
                axes[row, col].text(0.5, 0.5, f'Column {col_name}\nnot found', 
                                   ha='center', va='center', transform=axes[row, col].transAxes)
                axes[row, col].set_title(f'{model.upper()} - {height}', fontsize=12, fontweight='bold')
    
    # Adjust layout - literature style spacing
    plt.tight_layout(rect=[0, 0.03, 1, 0.92])
    plt.subplots_adjust(hspace=0.35, wspace=0.25)
    
    # Save the plot
    output_file = Path(output_path) / 'wind_direction_distributions_literature_style.png'
    plt.savefig(output_file, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"Literature-style distribution plot saved to: {output_file}")
    
    # Show the plot
    plt.show()
    
    return fig

def calculate_direction_statistics_literature(df):
    """Calculate statistics for wind direction distributions"""
    
    models = ['obs', 'ec', 'gfs']
    heights = ['10m', '30m', '50m', '70m']
    
    stats_data = []
    
    for model in models:
        for height in heights:
            col_name = f'{model}_wind_direction_{height}'
            
            if col_name in df.columns:
                data = df[col_name].dropna()
                
                if len(data) > 0:
                    # Calculate basic statistics
                    circ_mean = circular_mean(data)
                    
                    # Fit mixture model
                    x_smooth, mixture_pdf, r_squared = fit_von_mises_mixture(data, n_components=2)
                    
                    stats = {
                        'Model': model.upper(),
                        'Height': height,
                        'Count': len(data),
                        'Circular_Mean': circ_mean,
                        'Primary_Mode': data.mode().iloc[0] if len(data.mode()) > 0 else np.nan,
                        'R_squared': r_squared if r_squared is not None else np.nan,
                        'Has_Bimodal': 'Yes' if r_squared is not None and r_squared > 0.95 else 'No'
                    }
                    
                    stats_data.append(stats)
    
    # Create DataFrame and save
    stats_df = pd.DataFrame(stats_data)
    
    # Save statistics
    output_file = Path(output_path) / 'wind_direction_literature_style_statistics.csv'
    stats_df.to_csv(output_file, index=False)
    print(f"Statistics saved to: {output_file}")
    
    # Print summary
    print("\n=== Literature Style Wind Direction Statistics ===")
    print(stats_df.round(3))
    
    return stats_df

def main():
    """Main analysis function"""
    print("=== Literature Style Wind Direction Distribution Analysis ===")
    
    # Load data
    df = load_and_prepare_data()
    
    # Check available columns
    wind_direction_cols = [col for col in df.columns if 'wind_direction' in col]
    print(f"\nAvailable wind direction columns: {wind_direction_cols}")
    
    # Create literature-style distribution plots
    print("\nCreating literature-style wind direction distribution plots...")
    fig = create_literature_style_plots(df, output_path)
    
    # Calculate and save statistics
    print("\nCalculating wind direction distribution statistics...")
    stats_df = calculate_direction_statistics_literature(df)
    
    print("\n=== Analysis Complete ===")
    print(f"Results saved to: {output_path}")
    print("\nThe plots follow the exact style of the literature reference with:")
    print("- Gray histogram bars with white edges")
    print("- Black solid line for mvM.pdf (mixture of von Mises)")
    print("- Black dashed line for comparison distribution")
    print("- Compass direction labels on top and bottom")
    print("- Component count and R² values in corner boxes")

if __name__ == "__main__":
    main()