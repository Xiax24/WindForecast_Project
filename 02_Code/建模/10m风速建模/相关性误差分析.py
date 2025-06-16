"""
Simple Layer-wise Wind Speed Analysis
简化的分层风速相关性分析 - 专注于图表生成和保存
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import pearsonr
import os
import warnings
warnings.filterwarnings('ignore')

# 设置matplotlib参数
plt.style.use('default')
plt.rcParams['figure.dpi'] = 100
plt.rcParams['savefig.dpi'] = 300
plt.rcParams['font.size'] = 10

def load_and_analyze_data(file_path):
    """Load data and calculate layer-wise statistics"""
    print("Loading and analyzing data...")
    
    # Load data
    df = pd.read_csv(file_path)
    df['datetime'] = pd.to_datetime(df['datetime'])
    
    print(f"Data loaded: {len(df)} records")
    
    # Define height layers
    heights = ['10m', '30m', '50m', '70m']
    
    # Calculate statistics for each layer
    results = {}
    
    for height in heights:
        obs_col = f'obs_wind_speed_{height}'
        ec_col = f'ec_wind_speed_{height}'
        gfs_col = f'gfs_wind_speed_{height}'
        
        if all(col in df.columns for col in [obs_col, ec_col, gfs_col]):
            # Get clean data
            layer_data = df[[obs_col, ec_col, gfs_col]].dropna()
            
            if len(layer_data) > 100:  # Enough data points
                obs = layer_data[obs_col].values
                ec = layer_data[ec_col].values
                gfs = layer_data[gfs_col].values
                
                # Calculate metrics
                ec_corr, _ = pearsonr(obs, ec)
                gfs_corr, _ = pearsonr(obs, gfs)
                ec_rmse = np.sqrt(np.mean((obs - ec)**2))
                gfs_rmse = np.sqrt(np.mean((obs - gfs)**2))
                
                results[height] = {
                    'obs_data': obs,
                    'ec_data': ec,
                    'gfs_data': gfs,
                    'ec_corr': ec_corr,
                    'gfs_corr': gfs_corr,
                    'ec_rmse': ec_rmse,
                    'gfs_rmse': gfs_rmse,
                    'n_samples': len(layer_data)
                }
                
                print(f"✅ {height}: EC_r={ec_corr:.3f}, GFS_r={gfs_corr:.3f}, EC_RMSE={ec_rmse:.2f}, GFS_RMSE={gfs_rmse:.2f}")
            else:
                print(f"❌ {height}: Insufficient data ({len(layer_data)} samples)")
        else:
            print(f"❌ {height}: Missing columns")
    
    return results

def create_scatter_plots(results, output_dir):
    """Create and save scatter plots"""
    print("\nCreating scatter plots...")
    
    if not results:
        print("No data for scatter plots!")
        return None
    
    heights = list(results.keys())
    n_plots = len(heights)
    
    # Create figure
    fig, axes = plt.subplots(1, n_plots, figsize=(5*n_plots, 5))
    if n_plots == 1:
        axes = [axes]
    
    fig.suptitle('Layer-wise Wind Speed: Observed vs Forecast', fontsize=16, fontweight='bold')
    
    for i, height in enumerate(heights):
        ax = axes[i]
        data = results[height]
        
        obs = data['obs_data']
        ec = data['ec_data']
        gfs = data['gfs_data']
        
        # Sample data for plotting (max 2000 points for clarity)
        if len(obs) > 2000:
            indices = np.random.choice(len(obs), 2000, replace=False)
            obs_plot = obs[indices]
            ec_plot = ec[indices]
            gfs_plot = gfs[indices]
        else:
            obs_plot = obs
            ec_plot = ec
            gfs_plot = gfs
        
        # Scatter plots
        ax.scatter(obs_plot, ec_plot, alpha=0.4, s=2, color='steelblue', 
                  label=f'EC (r={data["ec_corr"]:.3f})')
        ax.scatter(obs_plot, gfs_plot, alpha=0.4, s=2, color='orange', 
                  label=f'GFS (r={data["gfs_corr"]:.3f})')
        
        # Perfect line
        min_val = min(obs_plot.min(), ec_plot.min(), gfs_plot.min())
        max_val = max(obs_plot.max(), ec_plot.max(), gfs_plot.max())
        ax.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2, alpha=0.8, label='Perfect')
        
        # Labels and formatting
        ax.set_xlabel(f'Observed {height} Wind Speed (m/s)', fontsize=11)
        ax.set_ylabel(f'Forecast {height} Wind Speed (m/s)', fontsize=11)
        ax.set_title(f'{height} Layer\n(n={data["n_samples"]})', fontsize=12, fontweight='bold')
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)
        
        # Add RMSE text
        ax.text(0.05, 0.95, f'RMSE:\nEC: {data["ec_rmse"]:.2f}\nGFS: {data["gfs_rmse"]:.2f}', 
               transform=ax.transAxes, va='top', ha='left', fontsize=9,
               bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8))
    
    plt.tight_layout()
    
    # Save figure
    scatter_path = os.path.join(output_dir, 'layer_wise_scatter_plots.png')
    plt.savefig(scatter_path, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"✅ Scatter plots saved: {scatter_path}")
    
    return fig

def create_performance_bars(results, output_dir):
    """Create and save performance bar charts"""
    print("Creating performance bar charts...")
    
    if not results:
        print("No data for performance charts!")
        return None
    
    heights = list(results.keys())
    
    # Extract data
    ec_corr = [results[h]['ec_corr'] for h in heights]
    gfs_corr = [results[h]['gfs_corr'] for h in heights]
    ec_rmse = [results[h]['ec_rmse'] for h in heights]
    gfs_rmse = [results[h]['gfs_rmse'] for h in heights]
    
    # Create figure with two subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))
    
    # Subplot 1: Correlation comparison
    x = np.arange(len(heights))
    width = 0.35
    
    bars1 = ax1.bar(x - width/2, ec_corr, width, label='EC', color='steelblue', alpha=0.8, edgecolor='black')
    bars2 = ax1.bar(x + width/2, gfs_corr, width, label='GFS', color='orange', alpha=0.8, edgecolor='black')
    
    # Add value labels on bars
    for i, (ec_val, gfs_val) in enumerate(zip(ec_corr, gfs_corr)):
        ax1.text(i - width/2, ec_val + 0.01, f'{ec_val:.3f}', 
                ha='center', va='bottom', fontweight='bold', fontsize=10)
        ax1.text(i + width/2, gfs_val + 0.01, f'{gfs_val:.3f}', 
                ha='center', va='bottom', fontweight='bold', fontsize=10)
    
    ax1.set_xlabel('Height Layers', fontsize=12)
    ax1.set_ylabel('Correlation Coefficient', fontsize=12)
    ax1.set_title('Correlation: Observed vs Forecast\n(Same Layer)', fontsize=14, fontweight='bold')
    ax1.set_xticks(x)
    ax1.set_xticklabels(heights)
    ax1.legend(fontsize=11)
    ax1.grid(True, alpha=0.3, axis='y')
    ax1.set_ylim(0, max(max(ec_corr), max(gfs_corr)) * 1.15)
    
    # Subplot 2: RMSE comparison
    bars3 = ax2.bar(x - width/2, ec_rmse, width, label='EC', color='steelblue', alpha=0.8, edgecolor='black')
    bars4 = ax2.bar(x + width/2, gfs_rmse, width, label='GFS', color='orange', alpha=0.8, edgecolor='black')
    
    # Add value labels on bars
    max_rmse = max(max(ec_rmse), max(gfs_rmse))
    for i, (ec_val, gfs_val) in enumerate(zip(ec_rmse, gfs_rmse)):
        ax2.text(i - width/2, ec_val + max_rmse * 0.02, f'{ec_val:.2f}', 
                ha='center', va='bottom', fontweight='bold', fontsize=10)
        ax2.text(i + width/2, gfs_val + max_rmse * 0.02, f'{gfs_val:.2f}', 
                ha='center', va='bottom', fontweight='bold', fontsize=10)
    
    ax2.set_xlabel('Height Layers', fontsize=12)
    ax2.set_ylabel('RMSE (m/s)', fontsize=12)
    ax2.set_title('Root Mean Square Error\n(Lower is Better)', fontsize=14, fontweight='bold')
    ax2.set_xticks(x)
    ax2.set_xticklabels(heights)
    ax2.legend(fontsize=11)
    ax2.grid(True, alpha=0.3, axis='y')
    ax2.set_ylim(0, max_rmse * 1.15)
    
    plt.tight_layout()
    
    # Save figure
    performance_path = os.path.join(output_dir, 'layer_wise_performance_summary.png')
    plt.savefig(performance_path, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"✅ Performance summary saved: {performance_path}")
    
    return fig

def create_summary_table(results, output_dir):
    """Create and save summary table"""
    print("Creating summary table...")
    
    if not results:
        print("No data for summary table!")
        return pd.DataFrame()
    
    # Create summary data
    summary_data = []
    for height, data in results.items():
        summary_data.append({
            'Layer': height,
            'Samples': data['n_samples'],
            'EC_Correlation': data['ec_corr'],
            'GFS_Correlation': data['gfs_corr'],
            'EC_RMSE': data['ec_rmse'],
            'GFS_RMSE': data['gfs_rmse'],
            'EC_Better_Corr': data['ec_corr'] > data['gfs_corr'],
            'EC_Better_RMSE': data['ec_rmse'] < data['gfs_rmse']
        })
    
    df_summary = pd.DataFrame(summary_data)
    
    # Print summary
    print("\n" + "="*80)
    print("LAYER-WISE PERFORMANCE SUMMARY")
    print("="*80)
    
    print(f"{'Layer':<8} {'Samples':<8} {'EC_Corr':<8} {'GFS_Corr':<9} {'EC_RMSE':<8} {'GFS_RMSE':<9} {'Winner':<10}")
    print("-" * 70)
    
    for _, row in df_summary.iterrows():
        ec_better_both = row['EC_Better_Corr'] and row['EC_Better_RMSE']
        ec_better_corr = row['EC_Better_Corr'] and not row['EC_Better_RMSE']
        ec_better_rmse = not row['EC_Better_Corr'] and row['EC_Better_RMSE']
        
        if ec_better_both:
            winner = "EC (both)"
        elif ec_better_corr:
            winner = "EC (corr)"
        elif ec_better_rmse:
            winner = "EC (rmse)"
        else:
            winner = "GFS"
        
        print(f"{row['Layer']:<8} {row['Samples']:<8} {row['EC_Correlation']:<8.3f} "
              f"{row['GFS_Correlation']:<9.3f} {row['EC_RMSE']:<8.3f} {row['GFS_RMSE']:<9.3f} {winner:<10}")
    
    # Overall comparison
    ec_avg_corr = df_summary['EC_Correlation'].mean()
    gfs_avg_corr = df_summary['GFS_Correlation'].mean()
    ec_avg_rmse = df_summary['EC_RMSE'].mean()
    gfs_avg_rmse = df_summary['GFS_RMSE'].mean()
    
    print(f"\nOVERALL AVERAGES:")
    print(f"EC  - Avg Correlation: {ec_avg_corr:.3f}, Avg RMSE: {ec_avg_rmse:.3f}")
    print(f"GFS - Avg Correlation: {gfs_avg_corr:.3f}, Avg RMSE: {gfs_avg_rmse:.3f}")
    
    if ec_avg_corr > gfs_avg_corr and ec_avg_rmse < gfs_avg_rmse:
        print("🏆 EC outperforms GFS in both correlation and RMSE")
    elif ec_avg_corr > gfs_avg_corr:
        print("📊 EC has better correlation, GFS has better RMSE")
    elif ec_avg_rmse < gfs_avg_rmse:
        print("📊 EC has better RMSE, GFS has better correlation")
    else:
        print("📊 GFS outperforms EC overall")
    
    # Correction potential
    print(f"\nCORRECTION POTENTIAL:")
    print("-" * 50)
    print(f"{'Layer':<8} {'Best_Model':<12} {'Max_R²':<8} {'Current_RMSE':<12} {'Target_RMSE':<12}")
    print("-" * 50)
    
    for _, row in df_summary.iterrows():
        if row['EC_Correlation'] >= row['GFS_Correlation']:
            best_model = "EC"
            best_corr = row['EC_Correlation']
            best_rmse = row['EC_RMSE']
        else:
            best_model = "GFS"
            best_corr = row['GFS_Correlation']
            best_rmse = row['GFS_RMSE']
        
        max_r2 = best_corr**2
        target_rmse = best_rmse * 0.8  # Conservative 20% improvement
        
        print(f"{row['Layer']:<8} {best_model:<12} {max_r2:<8.3f} {best_rmse:<12.3f} {target_rmse:<12.3f}")
    
    # Save to CSV
    csv_path = os.path.join(output_dir, 'layer_wise_summary.csv')
    df_summary.to_csv(csv_path, index=False)
    print(f"\n✅ Summary table saved: {csv_path}")
    
    return df_summary

def main():
    """Main execution function"""
    # Paths
    data_path = '/Users/xiaxin/work/WindForecast_Project/01_Data/processed/imputed_data/changma_clean.csv'
    output_dir = '/Users/xiaxin/work/WindForecast_Project/03_Results/分层相关性分析/'
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    try:
        print("🚀 Starting layer-wise wind speed analysis...")
        
        # 1. Load and analyze data
        results = load_and_analyze_data(data_path)
        
        if not results:
            print("❌ No valid layer data found!")
            return
        
        print(f"\n✅ Found {len(results)} valid layers: {list(results.keys())}")
        
        # 2. Create scatter plots
        fig1 = create_scatter_plots(results, output_dir)
        
        # 3. Create performance charts
        fig2 = create_performance_bars(results, output_dir)
        
        # 4. Create summary table
        df_summary = create_summary_table(results, output_dir)
        
        # 5. Show plots
        if fig1:
            plt.figure(fig1.number)
            plt.show()
        
        if fig2:
            plt.figure(fig2.number)
            plt.show()
        
        print("\n🎉 Analysis completed successfully!")
        print(f"📁 Results saved to: {output_dir}")
        
        return results, df_summary
        
    except Exception as e:
        print(f"❌ Error: {str(e)}")
        import traceback
        traceback.print_exc()
        return None, None

if __name__ == "__main__":
    results, df_summary = main()