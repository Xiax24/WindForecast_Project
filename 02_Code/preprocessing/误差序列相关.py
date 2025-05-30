#!/usr/bin/env python3
"""
Error Time Series Correlation Analysis - Complete Clean Version
Calculates correlations between model errors and all variables
Only includes wind speed and temperature errors (no wind direction or density errors)
Includes wind direction sin/cos components from observations
All text in English
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import spearmanr
import matplotlib.colors as mcolors
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# Settings
plt.rcParams['figure.dpi'] = 300
plt.rcParams['font.size'] = 10
sns.set_style("whitegrid")

# File paths
INPUT_FILE = "/Users/xiaxin/work/WindForecast_Project/01_Data/processed/imputed_data/changma_imputed_complete.csv"
OUTPUT_DIR = "/Users/xiaxin/work/WindForecast_Project/03_Results/error_correlation_analysis/"

def main():
    """Main analysis function"""
    print("=" * 60)
    print("Error Time Series Correlation Analysis")
    print("Wind Speed + Temperature Errors vs All Variables")
    print("Including Wind Direction Sin/Cos Components")
    print("=" * 60)
    
    # Create output directory
    output_path = Path(OUTPUT_DIR)
    output_path.mkdir(parents=True, exist_ok=True)
    
    try:
        # Step 1: Load data
        print("\n1. Loading data...")
        df = load_data()
        
        # Step 2: Calculate errors (wind speed + temperature only)
        print("\n2. Calculating error time series...")
        df_with_errors = calculate_wind_temp_errors(df)
        
        # Step 3: Prepare variables (including wind direction sin/cos)
        print("\n3. Preparing analysis variables...")
        analysis_data = prepare_all_variables(df_with_errors)
        
        # Step 4: Calculate correlations
        print("\n4. Computing correlations...")
        correlation_results = compute_correlations(analysis_data)
        
        # Step 5: Create visualization
        print("\n5. Creating correlation heatmap...")
        create_heatmap(correlation_results, output_path)
        
        # Step 6: Analyze error-power relationships
        print("\n6. Analyzing error-power relationships...")
        power_analysis = analyze_error_power(correlation_results, output_path)
        
        # Step 7: Save all results
        print("\n7. Saving results...")
        save_all_results(correlation_results, power_analysis, output_path)
        
        # Step 8: Generate report
        print("\n8. Generating analysis report...")
        create_report(correlation_results, power_analysis, output_path)
        
        print("\n" + "=" * 60)
        print("SUCCESS: Analysis completed!")
        print(f"Results saved to: {output_path}")
        print("Key files:")
        print("- error_wind_temp_correlation_heatmap.png")
        print("- error_power_correlations.csv")
        print("- correlation_matrix.csv")
        print("- analysis_report.txt")
        print("=" * 60)
        
    except Exception as e:
        print(f"\nERROR: {e}")
        import traceback
        traceback.print_exc()

def load_data():
    """Load the dataset"""
    df = pd.read_csv(INPUT_FILE)
    
    # Handle datetime index
    if 'datetime' in df.columns:
        df['datetime'] = pd.to_datetime(df['datetime'])
        df.set_index('datetime', inplace=True)
    elif 'time' in df.columns:
        df['time'] = pd.to_datetime(df['time'])
        df.set_index('time', inplace=True)
    
    print(f"  Data shape: {df.shape}")
    print(f"  Time period: {df.index.min()} to {df.index.max()}")
    
    return df

def calculate_wind_temp_errors(df):
    """Calculate wind speed and temperature errors only"""
    print("  Calculating wind speed and temperature errors...")
    
    df_result = df.copy()
    heights = ['10m', '30m', '50m', '70m']
    error_count = 0
    
    # Wind speed errors for all heights
    for height in heights:
        obs_col = f'obs_wind_speed_{height}'
        ec_col = f'ec_wind_speed_{height}'
        gfs_col = f'gfs_wind_speed_{height}'
        
        if all(col in df.columns for col in [obs_col, ec_col, gfs_col]):
            df_result[f'ec_wind_speed_error_{height}'] = df[ec_col] - df[obs_col]
            df_result[f'gfs_wind_speed_error_{height}'] = df[gfs_col] - df[obs_col]
            error_count += 2
            print(f"    Wind speed errors: {height}")
    
    # Temperature errors (10m only)
    obs_temp = 'obs_temperature_10m'
    ec_temp = 'ec_temperature_10m'
    gfs_temp = 'gfs_temperature_10m'
    
    if all(col in df.columns for col in [obs_temp, ec_temp, gfs_temp]):
        df_result['ec_temperature_error_10m'] = df[ec_temp] - df[obs_temp]
        df_result['gfs_temperature_error_10m'] = df[gfs_temp] - df[obs_temp]
        error_count += 2
        print("    Temperature errors: 10m")
    
    print(f"  Total error variables created: {error_count}")
    print("  NOTE: Skipping wind direction and density errors as requested")
    
    return df_result

def prepare_all_variables(df):
    """Prepare all variables for correlation analysis"""
    print("  Organizing variables...")
    
    # Get error variables (wind speed + temperature only)
    error_vars = [col for col in df.columns if '_error_' in col]
    
    # Get observation variables (exclude humidity and density)
    obs_vars = [col for col in df.columns 
                if col.startswith('obs_') and 'humidity' not in col and 'density' not in col]
    
    # Get forecast variables (exclude humidity, density, and errors)
    ec_vars = [col for col in df.columns 
               if col.startswith('ec_') and '_error_' not in col and 'humidity' not in col and 'density' not in col]
    
    gfs_vars = [col for col in df.columns 
                if col.startswith('gfs_') and '_error_' not in col and 'humidity' not in col and 'density' not in col]
    
    # Get power variables
    power_vars = [col for col in df.columns if 'power' in col.lower()]
    
    print(f"    Error variables: {len(error_vars)}")
    print(f"    Observation variables: {len(obs_vars)}")
    print(f"    EC forecast variables: {len(ec_vars)}")
    print(f"    GFS forecast variables: {len(gfs_vars)}")
    print(f"    Power variables: {len(power_vars)}")
    
    # Convert wind direction observations to sin/cos components
    print("  Converting wind direction to sin/cos components...")
    wind_dir_obs = [col for col in obs_vars if 'wind_direction' in col]
    wind_sincos_vars = []
    
    for col in wind_dir_obs:
        # Convert from meteorological to mathematical coordinates
        math_angle = (90 - df[col] + 360) % 360
        rad_angle = np.deg2rad(math_angle)
        
        # Create sin and cos components
        sin_name = col.replace('wind_direction', 'wind_dir_sin')
        cos_name = col.replace('wind_direction', 'wind_dir_cos')
        
        df[sin_name] = np.sin(rad_angle)
        df[cos_name] = np.cos(rad_angle)
        
        wind_sincos_vars.extend([sin_name, cos_name])
        print(f"    {col} -> {sin_name}, {cos_name}")
    
    # Update observation variables: remove original wind_direction, add sin/cos
    obs_vars_final = [col for col in obs_vars if 'wind_direction' not in col]
    obs_vars_final.extend(wind_sincos_vars)
    
    print(f"    Wind direction sin/cos components: {len(wind_sincos_vars)}")
    
    # Combine all variables for analysis
    all_analysis_vars = error_vars + obs_vars_final + ec_vars + gfs_vars + power_vars
    
    # Ensure all variables are numeric
    final_vars = []
    for var in all_analysis_vars:
        if var in df.columns:
            if df[var].dtype == 'object':
                try:
                    df[var] = pd.to_numeric(df[var], errors='coerce')
                    final_vars.append(var)
                    print(f"    Converted to numeric: {var}")
                except:
                    print(f"    WARNING: Skipped non-numeric variable: {var}")
            else:
                final_vars.append(var)
    
    # Create final analysis dataset
    analysis_df = df[final_vars].copy()
    
    print(f"  Final analysis variables: {len(final_vars)}")
    
    return {
        'data': analysis_df,
        'error_vars': error_vars,
        'obs_vars': obs_vars_final,
        'ec_vars': ec_vars,
        'gfs_vars': gfs_vars,
        'power_vars': power_vars,
        'wind_sincos_vars': wind_sincos_vars
    }

def compute_correlations(analysis_data):
    """Compute correlation matrix and significance tests"""
    print("  Computing correlation matrix...")
    
    df = analysis_data['data']
    
    # Check for high missing rate variables
    missing_rates = df.isnull().sum() / len(df) * 100
    high_missing = missing_rates[missing_rates > 50]
    
    if len(high_missing) > 0:
        print(f"    Removing {len(high_missing)} variables with >50% missing data:")
        for var, rate in high_missing.items():
            print(f"      {var}: {rate:.1f}% missing")
        df = df.drop(columns=high_missing.index)
    
    print(f"    Computing correlations for {len(df.columns)} variables...")
    
    # Calculate Spearman correlations
    corr_matrix = df.corr(method='spearman', min_periods=50)
    
    # Calculate p-values
    print("    Computing significance tests...")
    num_vars = len(df.columns)
    pvalue_matrix = pd.DataFrame(index=df.columns, columns=df.columns, dtype=float)
    
    for i, var1 in enumerate(df.columns):
        if i % 15 == 0:
            print(f"      Progress: {i+1}/{num_vars}")
        
        for var2 in df.columns:
            if var1 == var2:
                pvalue_matrix.loc[var1, var2] = 0.0
            else:
                valid_pairs = df[[var1, var2]].dropna()
                if len(valid_pairs) >= 30:
                    try:
                        _, p_value = spearmanr(valid_pairs[var1], valid_pairs[var2])
                        pvalue_matrix.loc[var1, var2] = p_value
                    except:
                        pvalue_matrix.loc[var1, var2] = 1.0
                else:
                    pvalue_matrix.loc[var1, var2] = 1.0
    
    print("    Correlation computation completed")
    
    return {
        'correlations': corr_matrix,
        'pvalues': pvalue_matrix,
        'data': df,
        'error_vars': analysis_data['error_vars'],
        'power_vars': analysis_data['power_vars'],
        'wind_sincos_vars': analysis_data['wind_sincos_vars'],
        'removed_vars': high_missing.index.tolist() if len(high_missing) > 0 else []
    }

def create_custom_colormap():
    """Create custom color map for correlations"""
    colors = ["#A32903", "#F45F31", "white", "#3FC2E7", "#016DB5"]
    return mcolors.LinearSegmentedColormap.from_list('custom', colors, N=256)

def create_heatmap(corr_results, output_path):
    """Create correlation heatmap visualization"""
    print("  Creating correlation heatmap...")
    
    corr_matrix = corr_results['correlations']
    pval_matrix = corr_results['pvalues']
    error_vars = corr_results['error_vars']
    wind_sincos_vars = corr_results['wind_sincos_vars']
    
    # Organize variables by type
    all_variables = corr_matrix.columns.tolist()
    
    # Group variables
    wind_speed_errors = sorted([v for v in all_variables if 'wind_speed_error' in v])
    temp_errors = sorted([v for v in all_variables if 'temperature_error' in v])
    
    wind_speed_obs = sorted([v for v in all_variables if v.startswith('obs_wind_speed')])
    temp_obs = sorted([v for v in all_variables if v.startswith('obs_temperature')])
    wind_sin_obs = sorted([v for v in all_variables if 'wind_dir_sin' in v])
    wind_cos_obs = sorted([v for v in all_variables if 'wind_dir_cos' in v])
    
    forecast_vars = sorted([v for v in all_variables if (v.startswith('ec_') or v.startswith('gfs_')) and '_error_' not in v])
    power_vars = [v for v in all_variables if 'power' in v.lower()]
    
    # Order variables: Errors first, then observations, then forecasts, then power
    ordered_variables = (wind_speed_errors + temp_errors + 
                        wind_speed_obs + temp_obs + wind_sin_obs + wind_cos_obs + 
                        forecast_vars + power_vars)
    
    # Add any remaining variables
    remaining = [v for v in all_variables if v not in ordered_variables]
    ordered_variables.extend(remaining)
    
    # Reorder correlation matrices
    corr_ordered = corr_matrix.loc[ordered_variables, ordered_variables]
    pval_ordered = pval_matrix.loc[ordered_variables, ordered_variables]
    
    # Create short, readable variable names
    short_names = []
    for var in ordered_variables:
        if '_error_' in var:
            parts = var.split('_')
            model = parts[0].upper()
            if 'wind_speed' in var:
                var_type = 'WS'
            elif 'temperature' in var:
                var_type = 'T'
            else:
                var_type = 'E'
            height = parts[-1]
            name = f"Err_{model}_{var_type}_{height}"
        elif var.startswith('obs_'):
            if 'wind_speed' in var:
                height = var.split('_')[-1]
                name = f"Obs_WS_{height}"
            elif 'temperature' in var:
                height = var.split('_')[-1]
                name = f"Obs_T_{height}"
            elif 'wind_dir_sin' in var:
                height = var.split('_')[-1]
                name = f"Obs_WD_Sin_{height}"
            elif 'wind_dir_cos' in var:
                height = var.split('_')[-1]
                name = f"Obs_WD_Cos_{height}"
            else:
                name = var.replace('obs_', 'Obs_')[:15]
        elif var.startswith('ec_'):
            name = var.replace('ec_', 'EC_')[:15]
        elif var.startswith('gfs_'):
            name = var.replace('gfs_', 'GFS_')[:15]
        else:
            name = var[:15]
        
        short_names.append(name)
    
    # Create the visualization
    fig, axes = plt.subplots(1, 2, figsize=(24, 12))
    fig.suptitle('Error Correlation Analysis: Wind Speed + Temperature Errors\nIncluding Wind Direction Sin/Cos Components\n* p<0.05, ** p<0.01, *** p<0.001', 
                 fontsize=16, fontweight='bold')
    
    colormap = create_custom_colormap()
    
    # Left plot: Full correlation matrix (upper triangle)
    ax1 = axes[0]
    mask_upper = np.triu(np.ones_like(corr_ordered, dtype=bool))
    
    corr_display = corr_ordered.copy()
    corr_display.columns = short_names
    corr_display.index = short_names
    
    sns.heatmap(corr_display, mask=mask_upper, cmap=colormap, center=0,
                square=True, vmin=-1, vmax=1, cbar_kws={"shrink": .6},
                ax=ax1, cbar=True)
    ax1.set_title('Complete Correlation Matrix', fontsize=14, fontweight='bold')
    ax1.tick_params(axis='both', labelsize=8)
    
    # Right plot: Focus on errors and key variables
    ax2 = axes[1]
    
    # Select error variables and important observation variables
    error_indices = [i for i, var in enumerate(ordered_variables) if '_error_' in var]
    key_obs_indices = []
    
    for i, var in enumerate(ordered_variables):
        if (('wind_speed' in var and 'obs_' in var) or 
            ('temperature' in var and 'obs_' in var) or 
            ('wind_dir_sin' in var) or 
            ('wind_dir_cos' in var) or
            'power' in var):
            key_obs_indices.append(i)
    
    # Combine indices for subset
    subset_indices = error_indices + key_obs_indices[:16]  # Limit for readability
    subset_variables = [ordered_variables[i] for i in subset_indices]
    subset_names = [short_names[i] for i in subset_indices]
    
    # Create subset matrices
    corr_subset = corr_ordered.loc[subset_variables, subset_variables]
    pval_subset = pval_ordered.loc[subset_variables, subset_variables]
    
    # Create annotations with correlation values and significance
    annotations = []
    for i in range(len(subset_variables)):
        row_annotations = []
        for j in range(len(subset_variables)):
            corr_value = corr_subset.iloc[i, j]
            p_value = pval_subset.iloc[i, j]
            
            # Add significance markers
            if p_value < 0.001:
                sig_marker = '***'
            elif p_value < 0.01:
                sig_marker = '**'
            elif p_value < 0.05:
                sig_marker = '*'
            else:
                sig_marker = ''
            
            row_annotations.append(f'{corr_value:.2f}{sig_marker}')
        annotations.append(row_annotations)
    
    corr_subset.columns = subset_names
    corr_subset.index = subset_names
    
    sns.heatmap(corr_subset, annot=annotations, cmap=colormap, center=0,
                square=True, vmin=-1, vmax=1, fmt='', cbar_kws={"shrink": .6},
                ax=ax2, annot_kws={'size': 7})
    ax2.set_title('Errors + Wind Direction Components Focus', fontsize=14, fontweight='bold')
    ax2.tick_params(axis='both', labelsize=9)
    
    # Save the plot
    plt.tight_layout()
    output_file = output_path / 'error_wind_temp_correlation_heatmap.png'
    plt.savefig(output_file, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    
    print(f"    Heatmap saved: {output_file}")
    print(f"    Wind speed errors: {len(wind_speed_errors)}")
    print(f"    Temperature errors: {len(temp_errors)}")
    print(f"    Wind direction sin/cos components: {len(wind_sincos_vars)}")

def analyze_error_power(corr_results, output_path):
    """Analyze correlations between errors and power"""
    print("  Analyzing error-power correlations...")
    
    corr_matrix = corr_results['correlations']
    pval_matrix = corr_results['pvalues']
    error_vars = corr_results['error_vars']
    power_vars = corr_results['power_vars']
    
    if not power_vars:
        print("    No power variables found")
        return None
    
    power_var = power_vars[0]
    
    # Calculate error-power correlations
    error_power_results = []
    for error_var in error_vars:
        if error_var in corr_matrix.columns:
            correlation = corr_matrix.loc[power_var, error_var]
            p_value = pval_matrix.loc[power_var, error_var]
            
            # Determine significance level
            if p_value < 0.001:
                significance = '***'
            elif p_value < 0.01:
                significance = '**'
            elif p_value < 0.05:
                significance = '*'
            else:
                significance = 'ns'
            
            error_power_results.append({
                'error_variable': error_var,
                'correlation': correlation,
                'abs_correlation': abs(correlation),
                'p_value': p_value,
                'significance': significance
            })
    
    # Sort by absolute correlation strength
    error_power_results.sort(key=lambda x: x['abs_correlation'], reverse=True)
    
    # Display top results
    print("    Top 10 error-power correlations:")
    for i, result in enumerate(error_power_results[:10]):
        print(f"      {i+1:2d}. {result['error_variable']:40s}: {result['correlation']:+.3f} {result['significance']:>3s}")
    
    return error_power_results

def save_all_results(corr_results, power_results, output_path):
    """Save all analysis results to files"""
    print("  Saving analysis results...")
    
    # Save correlation matrix
    corr_file = output_path / 'correlation_matrix.csv'
    corr_results['correlations'].to_csv(corr_file)
    print(f"    Correlation matrix: {corr_file}")
    
    # Save p-value matrix
    pval_file = output_path / 'pvalue_matrix.csv'
    corr_results['pvalues'].to_csv(pval_file)
    print(f"    P-value matrix: {pval_file}")
    
    # Save error-power correlations
    if power_results:
        power_file = output_path / 'error_power_correlations.csv'
        pd.DataFrame(power_results).to_csv(power_file, index=False)
        print(f"    Error-power correlations: {power_file}")

def create_report(corr_results, power_results, output_path):
    """Generate comprehensive analysis report"""
    print("  Generating analysis report...")
    
    report_content = [
        "=" * 70,
        "Error Time Series Correlation Analysis Report",
        "=" * 70,
        f"Generated: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}",
        f"Total variables analyzed: {len(corr_results['data'].columns)}",
        f"Error variables: {len(corr_results['error_vars'])}",
        f"Power variables: {len(corr_results['power_vars'])}",
        "",
        "ERROR VARIABLES INCLUDED:",
        "Only Wind Speed + Temperature errors (as requested)",
    ]
    
    for error_var in sorted(corr_results['error_vars']):
        report_content.append(f"  - {error_var}")
    
    # Wind direction components info
    wind_sincos_count = len(corr_results['wind_sincos_vars'])
    report_content.extend([
        "",
        f"WIND DIRECTION SIN/COS COMPONENTS: {wind_sincos_count}",
        "Included from observation data for all heights (10m, 30m, 50m, 70m):",
    ])
    
    for var in sorted(corr_results['wind_sincos_vars']):
        report_content.append(f"  - {var}")
    
    # Excluded variables
    report_content.extend([
        "",
        "EXCLUDED ERROR TYPES (as requested):",
        "  - Wind direction errors",
        "  - Air density errors",
        "",
        "EXCLUDED VARIABLE TYPES:",
        "  - Humidity variables (EC/GFS data missing)",
        "  - Density variables (excluded per request)",
    ])
    
    # High missing variables
    if corr_results['removed_vars']:
        report_content.extend([
            "",
            "REMOVED HIGH MISSING RATE VARIABLES (>50%):",
        ])
        for var in corr_results['removed_vars']:
            report_content.append(f"  - {var}")
    
    # Top error-power correlations
    if power_results:
        report_content.extend([
            "",
            "TOP 5 ERROR-POWER CORRELATIONS:",
        ])
        for i, result in enumerate(power_results[:5]):
            report_content.append(f"  {i+1}. {result['error_variable']}: {result['correlation']:+.3f} {result['significance']}")
    
    # Analysis methodology
    report_content.extend([
        "",
        "METHODOLOGY:",
        "1. Used Spearman correlation coefficient (robust for non-linear relationships)",
        "2. Significance levels: * p<0.05, ** p<0.01, *** p<0.001",
        "3. Minimum 30 valid observations required for significance testing",
        "4. Wind direction converted to sin/cos components using mathematical transformation",
        "5. Variables with >50% missing data were excluded",
        "",
        "WIND DIRECTION TRANSFORMATION:",
        "- Meteorological angle (0°=North, clockwise) -> Mathematical angle",
        "- Mathematical angle = (90° - meteorological_angle + 360°) % 360°",
        "- Sin component represents North-South wind component",
        "- Cos component represents East-West wind component",
        "",
        "=" * 70
    ])
    
    # Save report
    report_file = output_path / 'analysis_report.txt'
    with open(report_file, 'w', encoding='utf-8') as f:
        f.write('\n'.join(report_content))
    
    print(f"    Analysis report: {report_file}")

if __name__ == "__main__":
    main()