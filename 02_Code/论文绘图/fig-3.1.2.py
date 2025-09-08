import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import spearmanr
import matplotlib.colors as mcolors
import warnings
import os
warnings.filterwarnings('ignore')

# Set plot style with enhanced font sizes
sns.set_style("whitegrid")
plt.rcParams['figure.dpi'] = 500
# Set global font sizes
plt.rcParams['font.size'] = 20
plt.rcParams['axes.labelsize'] = 20
plt.rcParams['axes.titlesize'] = 20
plt.rcParams['xtick.labelsize'] = 20
plt.rcParams['ytick.labelsize'] = 20
plt.rcParams['legend.fontsize'] = 20
plt.rcParams['figure.titlesize'] = 20

def create_custom_colormap():
    """Create custom colormap matching the original design"""
    colors = ["#A32903", "#F45F31", "white", "#3FC2E7", "#016DB5"]
    n_bins = 256
    custom_cmap = mcolors.LinearSegmentedColormap.from_list('custom', colors, N=n_bins)
    return custom_cmap

def prepare_data(df):
    """Prepare observation data with wind direction conversion"""
    print("Preparing observation data...")
    
    # Select observation and power variables
    obs_cols = [col for col in df.columns if col.startswith('obs_')]
    power_cols = [col for col in df.columns if 'power' in col.lower()]
    analysis_cols = obs_cols + power_cols
    
    df_analysis = df[analysis_cols].copy()
    
    # Handle wind direction variables
    wind_dir_cols = [col for col in obs_cols if 'wind_direction' in col]
    
    for col in wind_dir_cols:
        # Convert meteorological angles to mathematical angles and then to sin/cos components
        math_angle = (90 - df_analysis[col] + 360) % 360
        wind_dir_rad = np.deg2rad(math_angle)
        
        # Create sin and cos components
        sin_col = col.replace('wind_direction', 'wind_dir_sin')
        cos_col = col.replace('wind_direction', 'wind_dir_cos')
        
        df_analysis[sin_col] = np.sin(wind_dir_rad)
        df_analysis[cos_col] = np.cos(wind_dir_rad)
    
    # Remove original wind direction columns
    df_analysis = df_analysis.drop(columns=wind_dir_cols)
    
    print(f"Final variables for analysis: {len(df_analysis.columns)}")
    return df_analysis

def calculate_spearman_correlation(df_clean):
    """Calculate Spearman correlation matrix with significance tests"""
    print("Calculating Spearman correlations...")
    
    # Calculate Spearman correlation matrix
    spearman_corr = df_clean.corr(method='spearman')
    
    # Calculate p-values
    n_vars = len(df_clean.columns)
    spearman_pvalues = pd.DataFrame(index=spearman_corr.index, columns=spearman_corr.columns)
    
    for i, var1 in enumerate(df_clean.columns):
        for j, var2 in enumerate(df_clean.columns):
            if var1 == var2:
                spearman_pvalues.loc[var1, var2] = 0.0
            else:
                paired_data = df_clean[[var1, var2]].dropna()
                if len(paired_data) > 3:
                    try:
                        _, p_val = spearmanr(paired_data[var1], paired_data[var2])
                        spearman_pvalues.loc[var1, var2] = p_val
                    except:
                        spearman_pvalues.loc[var1, var2] = 1.0
                else:
                    spearman_pvalues.loc[var1, var2] = 1.0
    
    spearman_pvalues = spearman_pvalues.astype(float)
    return spearman_corr, spearman_pvalues

def create_significance_annotations(corr_matrix, pval_matrix):
    """Create annotations with significance marks"""
    annotations = corr_matrix.copy().astype(str)
    for i in range(len(corr_matrix)):
        for j in range(len(corr_matrix.columns)):
            corr_val = corr_matrix.iloc[i, j]
            p_val = pval_matrix.iloc[i, j]
            
            # Add significance marks
            if p_val < 0.001:
                sig_mark = '***'
            elif p_val < 0.01:
                sig_mark = '**'
            elif p_val < 0.05:
                sig_mark = '*'
            else:
                sig_mark = ''
            
            annotations.iloc[i, j] = f'{corr_val:.2f}\n{sig_mark}'
    
    return annotations

def plot_spearman_heatmap(spearman_corr, spearman_pvalues, output_dir):
    """Plot Spearman correlation heatmap with significance"""
    print("Creating Spearman correlation heatmap...")
    
    # Order variables: wind speed -> wind direction -> temperature -> humidity -> density -> power
    columns = spearman_corr.columns.tolist()
    
    wind_speed_cols = [col for col in columns if 'wind_speed' in col]
    wind_dir_cols = [col for col in columns if 'wind_dir' in col]
    temp_cols = [col for col in columns if 'temperature' in col]
    humidity_cols = [col for col in columns if 'humidity' in col]
    density_cols = [col for col in columns if 'density' in col]
    power_cols = [col for col in columns if 'power' in col.lower()]
    
    ordered_vars = wind_speed_cols + wind_dir_cols + temp_cols + humidity_cols + density_cols + power_cols
    remaining_vars = [col for col in columns if col not in ordered_vars]
    ordered_vars.extend(remaining_vars)
    
    # Reorder matrices
    spearman_ordered = spearman_corr.loc[ordered_vars, ordered_vars]
    spearman_pval_ordered = spearman_pvalues.loc[ordered_vars, ordered_vars]
    
    # Simplify variable names for display
    simplified_labels = []
    for col in ordered_vars:
        if 'obs_' in col:
            label = col.replace('obs_', '').replace('_', ' ')
        else:
            label = col
        simplified_labels.append(label)
    
    # Create figure with larger size to accommodate larger fonts
    plt.figure(figsize=(16, 14))
    
    # Create upper triangular mask
    mask = np.triu(np.ones_like(spearman_ordered, dtype=bool))
    
    # Prepare data for display
    spearman_display = spearman_ordered.copy()
    spearman_display.columns = simplified_labels
    spearman_display.index = simplified_labels
    
    # Create annotations with significance
    spearman_annot = create_significance_annotations(spearman_ordered, spearman_pval_ordered)
    spearman_annot.columns = simplified_labels
    spearman_annot.index = simplified_labels
    
    # Create custom colormap
    custom_cmap = create_custom_colormap()
    
    # Plot heatmap with enhanced font sizes
    ax = sns.heatmap(spearman_display, 
                     mask=mask, 
                     annot=spearman_annot, 
                     cmap=custom_cmap,
                     center=0, 
                     square=True, 
                     fmt='',
                     vmin=-0.5, 
                     vmax=1,
                     cbar_kws={
                         "shrink": .8,
                         "label": "Spearman Correlation Coefficient"
                     },
                     annot_kws={
                         'size': 18,  # Annotation font size
                         'weight': 'normal'
                     })
    
    # Set title with large font
    # plt.title('Spearman Rank Correlation Analysis\n(* p<0.05, ** p<0.01, *** p<0.001)', 
    #           fontsize=22, fontweight='bold', pad=25)
    
    # Set axis labels with large fonts
    # ax.set_xlabel('Variables', fontsize=20, fontweight='bold', labelpad=15)
    # ax.set_ylabel('Variables', fontsize=20, fontweight='bold', labelpad=15)
    
    # Set tick labels with large fonts
    ax.tick_params(axis='x', which='major', labelsize=20, rotation=45, labelcolor='black')
    ax.tick_params(axis='y', which='major', labelsize=20, rotation=0, labelcolor='black')
    
    # Customize colorbar
    cbar = ax.collections[0].colorbar
    cbar.ax.tick_params(labelsize=20)  # Colorbar tick labels
    cbar.set_label('Spearman Correlation Coefficient', 
                   fontsize=23, fontweight='normal', labelpad=20)
    
    # Remove grid
    plt.grid(False)
    
    # Adjust layout to prevent label cutoff
    plt.tight_layout()
    
    # Save plot in both PNG and PDF formats
    plt.savefig(f'{output_dir}/spearman_correlation_heatmap.png', 
                dpi=500, bbox_inches='tight', facecolor='white')
    plt.savefig(f'{output_dir}/spearman_correlation_heatmap.pdf', 
                bbox_inches='tight', facecolor='white')
    plt.close()
    
    print("Spearman correlation heatmap saved successfully in PNG and PDF formats!")

def run_spearman_analysis(data_path, output_dir):
    """Run simplified Spearman correlation analysis"""
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    print("=== Spearman Rank Correlation Analysis ===")
    print(f"Input data: {data_path}")
    print(f"Output directory: {output_dir}")
    
    # Load data
    df = pd.read_csv(data_path)
    print(f"Data shape: {df.shape}")
    
    # Prepare data
    df_analysis = prepare_data(df)
    
    # Remove missing values
    df_clean = df_analysis.dropna()
    print(f"Valid samples after removing missing values: {len(df_clean)}")
    
    if len(df_clean) < 10:
        print("Insufficient data for correlation analysis")
        return
    
    # Calculate Spearman correlations
    spearman_corr, spearman_pvalues = calculate_spearman_correlation(df_clean)
    
    # Plot heatmap
    plot_spearman_heatmap(spearman_corr, spearman_pvalues, output_dir)
    
    # Save correlation matrix and p-values
    spearman_corr.to_csv(f'{output_dir}/spearman_correlation_matrix.csv')
    spearman_pvalues.to_csv(f'{output_dir}/spearman_pvalues_matrix.csv')
    
    print(f"\n✓ Spearman correlation analysis completed!")
    print(f"✓ Results saved to: {output_dir}")
    print(f"✓ Main outputs: spearman_correlation_heatmap.png and spearman_correlation_heatmap.pdf")

# Usage example
if __name__ == "__main__":
    # Set paths
    data_path = "/Users/xiaxin/work/WindForecast_Project/01_Data/processed/imputed_data/changma_imputed_complete.csv"
    output_dir = "/Users/xiaxin/work/WindForecast_Project/03_Results/figures/3.1results/spearman_correlation"
    
    # Run analysis
    run_spearman_analysis(data_path, output_dir)