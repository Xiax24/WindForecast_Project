"""
Complete EOF Decomposition Module for Wind Profile Analysis
==========================================================
This module performs Empirical Orthogonal Function (EOF) decomposition on wind profile data
and prepares complete dataset for EOF1 prediction modeling.

Key Features:
- Load wind profile data from corrected dataset
- Perform EOF decomposition using PCA
- Extract EOF1 spatial modes and time series
- Create comprehensive visualizations
- Merge with corrected data for complete modeling dataset
- Prepare all necessary files for next step (EOF1 prediction)

Author: Wind Power Prediction Pipeline
Date: 2025
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from scipy.stats import pearsonr
import os
import json
import warnings
warnings.filterwarnings('ignore')

# Set plotting parameters (no Chinese characters)
plt.rcParams['figure.dpi'] = 100
plt.rcParams['savefig.dpi'] = 300
plt.rcParams['font.size'] = 10
plt.rcParams['font.family'] = 'Arial'

class WindProfileEOF:
    """
    Complete EOF analysis class for wind profile data
    """
    
    def __init__(self, standardize=True):
        """
        Initialize EOF analyzer
        
        Parameters:
        -----------
        standardize : bool
            Whether to standardize wind profile data before EOF
        """
        self.standardize = standardize
        self.pca_model = None
        self.scaler = None
        self.eof_modes = None
        self.pc_timeseries = None
        self.explained_variance_ratio = None
        self.mean_profile = None
        self.std_profile = None
        self.height_columns = None
        self.wind_profile_data = None
        
    def load_data(self, corrected_data_path):
        """
        Load wind profile data from corrected dataset
        
        Parameters:
        -----------
        corrected_data_path : str
            Path to corrected wind speed dataset from Step 1
            
        Returns:
        --------
        pandas.DataFrame
            Clean wind profile data with datetime
        """
        print("=" * 60)
        print("LOADING WIND PROFILE DATA FOR EOF ANALYSIS")
        print("=" * 60)
        
        # Load corrected dataset
        print(f"Loading data from: {corrected_data_path}")
        df = pd.read_csv(corrected_data_path)
        df['datetime'] = pd.to_datetime(df['datetime'])
        
        # Define height columns for wind profile
        height_columns = [
            'obs_wind_speed_10m',
            'obs_wind_speed_30m', 
            'obs_wind_speed_50m',
            'obs_wind_speed_70m'
        ]
        
        # Check availability of height columns
        available_columns = [col for col in height_columns if col in df.columns]
        missing_columns = [col for col in height_columns if col not in df.columns]
        
        print(f"Total records loaded: {len(df)}")
        print(f"Time range: {df['datetime'].min()} to {df['datetime'].max()}")
        print(f"Available height columns: {available_columns}")
        
        if missing_columns:
            print(f"Missing height columns: {missing_columns}")
            print("WARNING: Missing height data will affect EOF analysis")
        
        if len(available_columns) < 3:
            raise ValueError("Need at least 3 height levels for meaningful EOF analysis")
        
        # Extract wind profile data
        required_columns = ['datetime'] + available_columns
        wind_data = df[required_columns].copy()
        
        # Remove rows with missing wind profile data
        initial_length = len(wind_data)
        wind_data = wind_data.dropna()
        final_length = len(wind_data)
        data_retention = final_length / initial_length * 100
        
        print(f"Records after removing missing values: {final_length} ({data_retention:.1f}%)")
        
        if data_retention < 70:
            print("WARNING: Low data retention rate - check data quality")
        
        # Store height information
        self.height_columns = available_columns
        self.wind_profile_data = wind_data
        
        # Basic statistics
        print(f"\nWind Profile Statistics:")
        for col in available_columns:
            wind_series = wind_data[col]
            print(f"  {col}: mean={wind_series.mean():.2f}, std={wind_series.std():.2f}, "
                  f"range=[{wind_series.min():.1f}, {wind_series.max():.1f}]")
        
        # Data quality checks
        wind_matrix = wind_data[available_columns].values
        negative_count = np.sum(wind_matrix < 0)
        extreme_count = np.sum(wind_matrix > 50)
        
        if negative_count > 0:
            print(f"WARNING: {negative_count} negative wind speed values detected")
        if extreme_count > 0:
            print(f"WARNING: {extreme_count} extreme wind speeds (>50 m/s) detected")
        
        print(f"✅ Wind profile data loaded successfully")
        return wind_data
    
    def perform_eof_decomposition(self, n_components=4):
        """
        Perform EOF decomposition on wind profile data
        
        Parameters:
        -----------
        n_components : int
            Number of EOF components to compute
            
        Returns:
        --------
        dict
            EOF decomposition results
        """
        print(f"\n" + "=" * 50)
        print(f"PERFORMING EOF DECOMPOSITION")
        print(f"=" * 50)
        
        if self.wind_profile_data is None:
            raise ValueError("Must load data first using load_data()")
        
        # Extract wind matrix
        wind_matrix = self.wind_profile_data[self.height_columns].values
        n_samples, n_heights = wind_matrix.shape
        
        print(f"Wind matrix shape: {wind_matrix.shape}")
        print(f"Heights: {self.height_columns}")
        print(f"Standardization: {'Enabled' if self.standardize else 'Disabled'}")
        
        # Calculate mean profile
        self.mean_profile = np.mean(wind_matrix, axis=0)
        
        # Prepare data for EOF
        if self.standardize:
            self.scaler = StandardScaler()
            wind_matrix_processed = self.scaler.fit_transform(wind_matrix)
            self.std_profile = self.scaler.scale_
            print(f"Data standardized - mean removed and scaled by std")
        else:
            wind_matrix_processed = wind_matrix - self.mean_profile[np.newaxis, :]
            self.std_profile = np.std(wind_matrix, axis=0)
            print(f"Data centered - only mean removed")
        
        # Perform PCA (EOF analysis)
        self.pca_model = PCA(n_components=n_components, random_state=42)
        self.pc_timeseries = self.pca_model.fit_transform(wind_matrix_processed)
        
        # Extract EOF modes (spatial patterns)
        self.eof_modes = self.pca_model.components_.T  # Shape: (n_heights, n_components)
        self.explained_variance_ratio = self.pca_model.explained_variance_ratio_
        
        # Print results
        print(f"\nEOF Decomposition Results:")
        total_variance = np.sum(self.explained_variance_ratio) * 100
        for i in range(n_components):
            print(f"  EOF{i+1}: {self.explained_variance_ratio[i]*100:.1f}% variance explained")
        print(f"  Total: {total_variance:.1f}% variance explained")
        
        # EOF1 analysis
        eof1_mode = self.eof_modes[:, 0]
        eof1_timeseries = self.pc_timeseries[:, 0]
        
        print(f"\nEOF1 Detailed Analysis:")
        print(f"  Spatial pattern coefficients:")
        for height, coeff in zip(self.height_columns, eof1_mode):
            print(f"    {height}: {coeff:.4f}")
        
        print(f"  Time series statistics:")
        print(f"    Mean: {np.mean(eof1_timeseries):.4f}")
        print(f"    Std: {np.std(eof1_timeseries):.4f}")
        print(f"    Range: [{np.min(eof1_timeseries):.3f}, {np.max(eof1_timeseries):.3f}]")
        
        # Physical interpretation
        if np.all(eof1_mode > 0) or np.all(eof1_mode < 0):
            print(f"  Physical meaning: Coherent vertical wind speed variation")
        else:
            print(f"  Physical meaning: Wind shear or vertical structure change")
        
        # Prepare results dictionary
        results = {
            'eof_modes': self.eof_modes,
            'pc_timeseries': self.pc_timeseries,
            'explained_variance_ratio': self.explained_variance_ratio,
            'total_variance_explained': total_variance,
            'mean_profile': self.mean_profile,
            'std_profile': self.std_profile,
            'height_columns': self.height_columns,
            'wind_profile_data': self.wind_profile_data,
            'n_samples': n_samples,
            'n_heights': n_heights
        }
        
        print(f"✅ EOF decomposition completed successfully")
        return results
    
    def create_visualizations(self, results, output_dir):
        """
        Create comprehensive EOF analysis visualizations
        
        Parameters:
        -----------
        results : dict
            EOF decomposition results
        output_dir : str
            Directory to save plots
            
        Returns:
        --------
        list
            Paths to saved plot files
        """
        print(f"\n" + "=" * 50)
        print(f"CREATING EOF VISUALIZATIONS")
        print(f"=" * 50)
        
        os.makedirs(output_dir, exist_ok=True)
        
        # Extract data
        eof_modes = results['eof_modes']
        pc_timeseries = results['pc_timeseries']
        explained_variance_ratio = results['explained_variance_ratio']
        height_columns = results['height_columns']
        wind_profile_data = results['wind_profile_data']
        
        # Extract heights for plotting
        heights = []
        for col in height_columns:
            # Extract height number from column name like 'obs_wind_speed_10m'
            height_str = col.split('_')[-1].replace('m', '')
            heights.append(int(height_str))
        
        plot_paths = []
        
        # 1. Comprehensive EOF Analysis Plot
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        
        # Plot 1: EOF Spatial Modes
        ax1 = axes[0, 0]
        n_modes_to_plot = min(4, eof_modes.shape[1])
        colors = ['red', 'blue', 'green', 'orange']
        
        for i in range(n_modes_to_plot):
            ax1.plot(heights, eof_modes[:, i], 'o-', linewidth=2, color=colors[i],
                    label=f'EOF{i+1} ({explained_variance_ratio[i]*100:.1f}%)', 
                    markersize=6)
        
        ax1.axhline(y=0, color='black', linestyle='--', alpha=0.5)
        ax1.set_xlabel('Height (m)')
        ax1.set_ylabel('EOF Coefficient')
        ax1.set_title('EOF Spatial Modes')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Plot 2: Explained Variance
        ax2 = axes[0, 1]
        modes = np.arange(1, len(explained_variance_ratio) + 1)
        bars = ax2.bar(modes, explained_variance_ratio * 100, 
                      alpha=0.7, color='skyblue', edgecolor='navy')
        ax2.set_xlabel('EOF Mode')
        ax2.set_ylabel('Explained Variance (%)')
        ax2.set_title('Variance Explained by Each EOF')
        ax2.grid(True, alpha=0.3)
        
        # Add percentage labels
        for bar, pct in zip(bars, explained_variance_ratio * 100):
            height = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width()/2., height + 0.5,
                    f'{pct:.1f}%', ha='center', va='bottom', fontsize=10)
        
        # Plot 3: EOF1 Time Series Sample
        ax3 = axes[0, 2]
        sample_size = min(2000, len(pc_timeseries))
        sample_indices = np.linspace(0, len(pc_timeseries)-1, sample_size, dtype=int)
        sample_times = wind_profile_data['datetime'].iloc[sample_indices]
        sample_eof1 = pc_timeseries[sample_indices, 0]
        
        ax3.plot(sample_times, sample_eof1, 'b-', linewidth=0.8, alpha=0.7)
        ax3.set_xlabel('Time')
        ax3.set_ylabel('EOF1 Coefficient')
        ax3.set_title(f'EOF1 Time Series (Sample: {sample_size} points)')
        ax3.grid(True, alpha=0.3)
        plt.setp(ax3.xaxis.get_majorticklabels(), rotation=45)
        
        # Plot 4: Wind Profile Correlations
        ax4 = axes[1, 0]
        wind_matrix = wind_profile_data[height_columns].values
        corr_matrix = np.corrcoef(wind_matrix.T)
        
        im = ax4.imshow(corr_matrix, cmap='RdYlBu_r', vmin=-1, vmax=1)
        ax4.set_xticks(range(len(heights)))
        ax4.set_yticks(range(len(heights)))
        ax4.set_xticklabels([f'{h}m' for h in heights], rotation=45)
        ax4.set_yticklabels([f'{h}m' for h in heights])
        ax4.set_title('Inter-Height Wind Speed Correlations')
        
        # Add correlation values
        for i in range(len(heights)):
            for j in range(len(heights)):
                text = ax4.text(j, i, f'{corr_matrix[i, j]:.2f}',
                               ha="center", va="center", color="black", fontsize=9)
        
        plt.colorbar(im, ax=ax4, label='Correlation')
        
        # Plot 5: EOF1 Distribution
        ax5 = axes[1, 1]
        eof1_data = pc_timeseries[:, 0]
        ax5.hist(eof1_data, bins=50, alpha=0.7, color='green', 
                edgecolor='black', density=True)
        ax5.axvline(np.mean(eof1_data), color='red', linestyle='--', linewidth=2,
                   label=f'Mean: {np.mean(eof1_data):.3f}')
        ax5.set_xlabel('EOF1 Coefficient')
        ax5.set_ylabel('Density')
        ax5.set_title('EOF1 Coefficient Distribution')
        ax5.legend()
        ax5.grid(True, alpha=0.3)
        
        # Plot 6: EOF1 vs 10m Wind Speed
        ax6 = axes[1, 2]
        wind_10m = wind_matrix[:, 0]  # First column is 10m
        
        # Create scatter plot with density
        ax6.scatter(wind_10m, eof1_data, alpha=0.3, s=1, color='purple')
        
        # Add trend line
        z = np.polyfit(wind_10m, eof1_data, 1)
        p = np.poly1d(z)
        ax6.plot(wind_10m, p(wind_10m), "r--", alpha=0.8, linewidth=2)
        
        # Calculate correlation
        corr_10m_eof1 = np.corrcoef(wind_10m, eof1_data)[0, 1]
        
        ax6.set_xlabel('10m Wind Speed (m/s)')
        ax6.set_ylabel('EOF1 Coefficient')
        ax6.set_title(f'EOF1 vs 10m Wind Speed\nCorrelation: {corr_10m_eof1:.3f}')
        ax6.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        # Save comprehensive plot
        comprehensive_plot_path = os.path.join(output_dir, 'eof_comprehensive_analysis.png')
        plt.savefig(comprehensive_plot_path, dpi=300, bbox_inches='tight')
        plot_paths.append(comprehensive_plot_path)
        print(f"✅ Comprehensive EOF plot saved: {comprehensive_plot_path}")
        plt.show()
        
        # 2. Detailed EOF1 Analysis Plot
        detailed_plot_path = self._create_eof1_detailed_plot(results, output_dir)
        plot_paths.append(detailed_plot_path)
        
        return plot_paths
    
    def _create_eof1_detailed_plot(self, results, output_dir):
        """Create detailed EOF1 analysis plot"""
        
        print("Creating detailed EOF1 analysis plot...")
        
        # Extract data
        eof_modes = results['eof_modes']
        pc_timeseries = results['pc_timeseries']
        height_columns = results['height_columns']
        wind_profile_data = results['wind_profile_data']
        
        heights = [int(col.split('_')[-1].replace('m', '')) for col in height_columns]
        eof1_coeff = pc_timeseries[:, 0]
        wind_matrix = wind_profile_data[height_columns].values
        
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        
        # Plot 1: EOF1 Spatial Pattern with Details
        ax1 = axes[0, 0]
        eof1_pattern = eof_modes[:, 0]
        bars = ax1.bar(range(len(heights)), eof1_pattern, alpha=0.7, 
                      color='skyblue', edgecolor='navy')
        ax1.set_xticks(range(len(heights)))
        ax1.set_xticklabels([f'{h}m' for h in heights])
        ax1.axhline(y=0, color='black', linestyle='--', alpha=0.5)
        ax1.set_xlabel('Height')
        ax1.set_ylabel('EOF1 Coefficient')
        ax1.set_title(f'EOF1 Spatial Pattern\n({results["explained_variance_ratio"][0]*100:.1f}% variance)')
        ax1.grid(True, alpha=0.3)
        
        # Add value labels
        for bar, coeff in zip(bars, eof1_pattern):
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., 
                    height + (0.02 if height >= 0 else -0.02),
                    f'{coeff:.3f}', ha='center', 
                    va='bottom' if height >= 0 else 'top', fontsize=10)
        
        # Plot 2: EOF1 Diurnal Variation
        ax2 = axes[0, 1]
        wind_profile_data_copy = wind_profile_data.copy()
        wind_profile_data_copy['hour'] = wind_profile_data_copy['datetime'].dt.hour
        wind_profile_data_copy['eof1'] = eof1_coeff
        
        hourly_stats = wind_profile_data_copy.groupby('hour')['eof1'].agg(['mean', 'std'])
        
        ax2.plot(hourly_stats.index, hourly_stats['mean'], 'b-', 
                linewidth=2, marker='o', markersize=4, label='Mean')
        ax2.fill_between(hourly_stats.index, 
                        hourly_stats['mean'] - hourly_stats['std'],
                        hourly_stats['mean'] + hourly_stats['std'], 
                        alpha=0.3, label='±1 Std')
        ax2.set_xlabel('Hour of Day')
        ax2.set_ylabel('EOF1 Coefficient')
        ax2.set_title('EOF1 Diurnal Variation')
        ax2.grid(True, alpha=0.3)
        ax2.legend()
        ax2.set_xticks(range(0, 24, 4))
        
        # Plot 3: Reconstruction Quality Example
        ax3 = axes[1, 0]
        
        # Sample period for demonstration
        sample_start = len(wind_profile_data) // 3
        sample_end = sample_start + 200
        sample_slice = slice(sample_start, sample_end)
        
        # Original and reconstructed 70m wind (last height)
        original_70m = wind_matrix[sample_slice, -1]
        sample_eof1 = eof1_coeff[sample_slice]
        mean_70m = results['mean_profile'][-1]
        eof1_70m_coeff = eof_modes[-1, 0]
        
        if self.standardize:
            # Need to reverse standardization
            reconstructed_70m = (sample_eof1 * eof1_70m_coeff * self.std_profile[-1] + 
                               mean_70m)
        else:
            reconstructed_70m = sample_eof1 * eof1_70m_coeff + mean_70m
        
        time_indices = range(len(sample_eof1))
        ax3.plot(time_indices, original_70m, 'b-', linewidth=1.5, 
                label=f'Observed {heights[-1]}m', alpha=0.8)
        ax3.plot(time_indices, reconstructed_70m, 'r--', linewidth=1.5, 
                label=f'EOF1 Reconstructed {heights[-1]}m', alpha=0.8)
        
        # Calculate correlation
        corr_recon = np.corrcoef(original_70m, reconstructed_70m)[0, 1]
        
        ax3.set_xlabel('Time Index')
        ax3.set_ylabel('Wind Speed (m/s)')
        ax3.set_title(f'EOF1 Reconstruction Quality\nCorrelation: {corr_recon:.3f}')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        
        # Plot 4: EOF1 vs Wind Speed Bins
        ax4 = axes[1, 1]
        
        # Bin analysis by 10m wind speed
        wind_10m = wind_matrix[:, 0]
        n_bins = 8
        wind_bins = np.linspace(wind_10m.min(), wind_10m.max(), n_bins + 1)
        bin_centers = (wind_bins[:-1] + wind_bins[1:]) / 2
        
        bin_means = []
        bin_stds = []
        bin_counts = []
        
        for i in range(n_bins):
            mask = (wind_10m >= wind_bins[i]) & (wind_10m < wind_bins[i+1])
            bin_count = np.sum(mask)
            bin_counts.append(bin_count)
            
            if bin_count > 20:  # Minimum samples for reliable statistics
                bin_means.append(np.mean(eof1_coeff[mask]))
                bin_stds.append(np.std(eof1_coeff[mask]))
            else:
                bin_means.append(np.nan)
                bin_stds.append(np.nan)
        
        bin_means = np.array(bin_means)
        bin_stds = np.array(bin_stds)
        
        # Plot only valid bins
        valid_mask = ~np.isnan(bin_means)
        
        ax4.errorbar(bin_centers[valid_mask], bin_means[valid_mask], 
                    yerr=bin_stds[valid_mask], marker='o', capsize=5, 
                    capthick=2, linewidth=2, markersize=6)
        
        ax4.set_xlabel('10m Wind Speed (m/s)')
        ax4.set_ylabel('EOF1 Coefficient')
        ax4.set_title('EOF1 Response to Wind Speed')
        ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        # Save detailed plot
        detailed_plot_path = os.path.join(output_dir, 'eof1_detailed_analysis.png')
        plt.savefig(detailed_plot_path, dpi=300, bbox_inches='tight')
        print(f"✅ Detailed EOF1 plot saved: {detailed_plot_path}")
        plt.show()
        
        return detailed_plot_path
    
    def save_results(self, results, output_dir):
        """
        Save all EOF analysis results
        
        Parameters:
        -----------
        results : dict
            EOF analysis results
        output_dir : str
            Output directory
            
        Returns:
        --------
        dict
            Paths to saved files
        """
        print(f"\n" + "=" * 50)
        print(f"SAVING EOF ANALYSIS RESULTS")
        print(f"=" * 50)
        
        os.makedirs(output_dir, exist_ok=True)
        
        # 1. Save EOF modes (spatial patterns)
        eof_modes_df = pd.DataFrame(
            results['eof_modes'],
            index=results['height_columns'],
            columns=[f'EOF{i+1}' for i in range(results['eof_modes'].shape[1])]
        )
        modes_path = os.path.join(output_dir, 'eof_spatial_modes.csv')
        eof_modes_df.to_csv(modes_path)
        print(f"✅ EOF spatial modes saved: {modes_path}")
        
        # 2. Save PC time series
        pc_df = pd.DataFrame(
            results['pc_timeseries'],
            columns=[f'PC{i+1}' for i in range(results['pc_timeseries'].shape[1])]
        )
        pc_df.insert(0, 'datetime', results['wind_profile_data']['datetime'].values)
        
        timeseries_path = os.path.join(output_dir, 'eof_timeseries.csv')
        pc_df.to_csv(timeseries_path, index=False)
        print(f"✅ EOF time series saved: {timeseries_path}")
        
        # 3. Save summary statistics
        summary = {
            'analysis_info': {
                'standardization': self.standardize,
                'n_samples': results['n_samples'],
                'n_heights': results['n_heights'],
                'height_columns': results['height_columns'],
                'time_range': {
                    'start': str(results['wind_profile_data']['datetime'].min()),
                    'end': str(results['wind_profile_data']['datetime'].max())
                }
            },
            'eof_statistics': {
                'explained_variance_ratio': results['explained_variance_ratio'].tolist(),
                'total_variance_explained': results['total_variance_explained'],
                'eof1_variance_percent': results['explained_variance_ratio'][0] * 100
            },
            'profile_statistics': {
                'mean_profile': results['mean_profile'].tolist(),
                'std_profile': results['std_profile'].tolist()
            },
            'eof1_analysis': {
                'spatial_coefficients': dict(zip(results['height_columns'], 
                                                results['eof_modes'][:, 0].tolist())),
                'timeseries_stats': {
                    'mean': float(np.mean(results['pc_timeseries'][:, 0])),
                    'std': float(np.std(results['pc_timeseries'][:, 0])),
                    'min': float(np.min(results['pc_timeseries'][:, 0])),
                    'max': float(np.max(results['pc_timeseries'][:, 0]))
                }
            }
        }
        
        summary_path = os.path.join(output_dir, 'eof_analysis_summary.json')
        with open(summary_path, 'w') as f:
            json.dump(summary, f, indent=2)
        print(f"✅ Summary statistics saved: {summary_path}")
        
        # 4. Save detailed text report
        report_path = self._save_detailed_report(results, output_dir)
        
        return {
            'modes_file': modes_path,
            'timeseries_file': timeseries_path,
            'summary_file': summary_path,
            'report_file': report_path
        }
    
    def _save_detailed_report(self, results, output_dir):
        """Save detailed text report"""
        
        report_path = os.path.join(output_dir, 'eof_analysis_detailed_report.txt')
        
        with open(report_path, 'w') as f:
            f.write("EOF Analysis Detailed Report\n")
            f.write("=" * 60 + "\n\n")
            
            # Basic information
            f.write("DATASET INFORMATION:\n")
            f.write(f"Samples: {results['n_samples']}\n")
            f.write(f"Heights: {results['height_columns']}\n")
            f.write(f"Time range: {results['wind_profile_data']['datetime'].min()} to "
                   f"{results['wind_profile_data']['datetime'].max()}\n")
            f.write(f"Standardization: {'Yes' if self.standardize else 'No'}\n\n")
            
            # EOF analysis results
            f.write("EOF DECOMPOSITION RESULTS:\n")
            for i, var_ratio in enumerate(results['explained_variance_ratio']):
                f.write(f"EOF{i+1}: {var_ratio*100:.2f}% variance explained\n")
            f.write(f"Total: {results['total_variance_explained']:.2f}% variance explained\n\n")
            
            # EOF1 detailed analysis
            f.write("EOF1 DETAILED ANALYSIS:\n")
            eof1_mode = results['eof_modes'][:, 0]
            f.write("Spatial pattern coefficients:\n")
            for height, coeff in zip(results['height_columns'], eof1_mode):
                f.write(f"  {height}: {coeff:.4f}\n")
            
            # EOF1 time series statistics
            eof1_ts = results['pc_timeseries'][:, 0]
            f.write(f"\nEOF1 time series statistics:\n")
            f.write(f"  Mean: {np.mean(eof1_ts):.4f}\n")
            f.write(f"  Standard deviation: {np.std(eof1_ts):.4f}\n")
            f.write(f"  Range: [{np.min(eof1_ts):.4f}, {np.max(eof1_ts):.4f}]\n")
            f.write(f"  Skewness: {pd.Series(eof1_ts).skew():.4f}\n")
            f.write(f"  Kurtosis: {pd.Series(eof1_ts).kurtosis():.4f}\n\n")
            
            # Physical interpretation
            f.write("PHYSICAL INTERPRETATION:\n")
            if np.all(eof1_mode > 0) or np.all(eof1_mode < 0):
                f.write("EOF1 shows uniform sign across all heights.\n")
                f.write("This represents coherent strengthening/weakening of entire wind profile.\n")
            else:
                f.write("EOF1 shows mixed signs across heights.\n")
                f.write("This represents wind shear or vertical structure changes.\n")
            
            # Correlation analysis
            f.write(f"\nCORRELATION ANALYSIS:\n")
            wind_matrix = results['wind_profile_data'][results['height_columns']].values
            wind_10m = wind_matrix[:, 0]
            
            f.write("Correlation between heights:\n")
            for i in range(len(results['height_columns'])):
                for j in range(i+1, len(results['height_columns'])):
                    corr = np.corrcoef(wind_matrix[:, i], wind_matrix[:, j])[0, 1]
                    f.write(f"  {results['height_columns'][i]} vs {results['height_columns'][j]}: {corr:.4f}\n")
            
            f.write("\nCorrelation between EOF1 and wind speeds:\n")
            for i, height in enumerate(results['height_columns']):
                corr = np.corrcoef(eof1_ts, wind_matrix[:, i])[0, 1]
                f.write(f"  EOF1 vs {height}: {corr:.4f}\n")
        
        print(f"✅ Detailed report saved: {report_path}")
        return report_path


def create_complete_modeling_dataset(corrected_data_path, eof_files, output_dir):
    """
    Create complete modeling dataset by merging corrected data with EOF results
    AND add reconstructed wind speeds for validation
    
    Parameters:
    -----------
    corrected_data_path : str
        Path to original corrected dataset from Step 1
    eof_files : dict
        Dictionary containing paths to EOF result files
    output_dir : str
        Output directory for merged dataset
        
    Returns:
    --------
    str or None
        Path to complete modeling dataset if successful
    """
    print(f"\n" + "=" * 60)
    print(f"CREATING COMPLETE MODELING DATASET WITH RECONSTRUCTED WINDS")
    print(f"=" * 60)
    
    try:
        # Load corrected data
        print(f"Loading corrected data: {corrected_data_path}")
        corrected_df = pd.read_csv(corrected_data_path)
        corrected_df['datetime'] = pd.to_datetime(corrected_df['datetime'])
        
        # Load EOF time series
        eof_timeseries_path = eof_files['timeseries_file']
        print(f"Loading EOF time series: {eof_timeseries_path}")
        eof_df = pd.read_csv(eof_timeseries_path)
        eof_df['datetime'] = pd.to_datetime(eof_df['datetime'])
        
        # Load EOF spatial modes
        eof_modes_path = eof_files['modes_file']
        print(f"Loading EOF spatial modes: {eof_modes_path}")
        eof_modes_df = pd.read_csv(eof_modes_path, index_col=0)
        
        # Load EOF summary for mean profile
        eof_summary_path = eof_files['summary_file']
        print(f"Loading EOF summary: {eof_summary_path}")
        with open(eof_summary_path, 'r') as f:
            eof_summary = json.load(f)
        
        # Merge datasets on datetime
        print(f"Merging datasets...")
        merged_df = pd.merge(corrected_df, eof_df, on='datetime', how='inner')
        
        # Add target variable for EOF1 prediction
        merged_df['eof1_target'] = merged_df['PC1']
        
        # 🔥 NEW: Add reconstructed wind speeds using EOF1
        print(f"🔄 Reconstructing wind speeds using EOF1...")
        
        # Get EOF1 spatial coefficients and mean profile
        eof1_spatial_coeffs = eof_modes_df['EOF1'].values
        mean_profile = np.array(eof_summary['profile_statistics']['mean_profile'])
        height_columns = eof_summary['analysis_info']['height_columns']
        
        # Get EOF1 time series
        eof1_timeseries = merged_df['PC1'].values
        
        print(f"EOF1 spatial coefficients: {dict(zip(height_columns, eof1_spatial_coeffs))}")
        print(f"Mean wind profile: {dict(zip(height_columns, mean_profile))}")
        
        # Reconstruct wind speeds for all heights
        for i, height_col in enumerate(height_columns):
            # Reconstruction formula: wind = mean + EOF1_timeseries * EOF1_spatial_coeff
            reconstructed_wind = mean_profile[i] + eof1_timeseries * eof1_spatial_coeffs[i]
            
            # Add to dataset
            recon_col_name = f'reconstructed_{height_col.split("_")[-1]}'  # e.g., 'reconstructed_70m'
            merged_df[recon_col_name] = reconstructed_wind
            
            print(f"✅ {recon_col_name}: mean={reconstructed_wind.mean():.2f}, "
                  f"std={reconstructed_wind.std():.2f}, "
                  f"range=[{reconstructed_wind.min():.1f}, {reconstructed_wind.max():.1f}]")
        
        # 🔥 Calculate reconstruction quality metrics
        print(f"\n📊 Reconstruction Quality Assessment:")
        reconstruction_metrics = {}
        
        for i, height_col in enumerate(height_columns):
            if height_col in merged_df.columns:  # Check if observed data exists
                observed_wind = merged_df[height_col].values
                recon_col_name = f'reconstructed_{height_col.split("_")[-1]}'
                reconstructed_wind = merged_df[recon_col_name].values
                
                # Calculate metrics
                correlation = np.corrcoef(observed_wind, reconstructed_wind)[0, 1]
                rmse = np.sqrt(np.mean((observed_wind - reconstructed_wind)**2))
                mae = np.mean(np.abs(observed_wind - reconstructed_wind))
                bias = np.mean(reconstructed_wind - observed_wind)
                
                reconstruction_metrics[height_col] = {
                    'correlation': correlation,
                    'rmse': rmse,
                    'mae': mae,
                    'bias': bias
                }
                
                print(f"  {height_col}:")
                print(f"    Correlation: {correlation:.4f}")
                print(f"    RMSE: {rmse:.4f} m/s")
                print(f"    MAE: {mae:.4f} m/s")
                print(f"    Bias: {bias:.4f} m/s")
        
        # Check merge quality
        original_corrected = len(corrected_df)
        original_eof = len(eof_df)
        merged_samples = len(merged_df)
        
        merge_rate_corrected = merged_samples / original_corrected * 100
        merge_rate_eof = merged_samples / original_eof * 100
        
        print(f"\n📈 Merge statistics:")
        print(f"  Original corrected data: {original_corrected} samples")
        print(f"  Original EOF data: {original_eof} samples")
        print(f"  Merged dataset: {merged_samples} samples")
        print(f"  Merge rate (vs corrected): {merge_rate_corrected:.1f}%")
        print(f"  Merge rate (vs EOF): {merge_rate_eof:.1f}%")
        
        if merge_rate_eof < 95:
            print(f"⚠️  WARNING: Low merge rate - some data may be missing")
        
        # Verify required columns
        required_columns = ['datetime', 'eof1_target', 'corrected_wind_speed_10m']
        missing_columns = [col for col in required_columns if col not in merged_df.columns]
        
        if missing_columns:
            print(f"❌ ERROR: Missing required columns: {missing_columns}")
            return None
        
        # Check for train/test split information
        has_split = 'data_split' in merged_df.columns
        if has_split:
            train_count = (merged_df['data_split'] == 'train').sum()
            test_count = (merged_df['data_split'] == 'test').sum()
            print(f"✂️  Train/test split found: {train_count} train, {test_count} test")
        else:
            print(f"⚠️  No train/test split found - will need to create one")
        
        # Save complete modeling dataset
        complete_path = os.path.join(output_dir, 'complete_eof1_modeling_dataset.csv')
        merged_df.to_csv(complete_path, index=False)
        
        print(f"✅ Complete modeling dataset saved: {complete_path}")
        print(f"📊 Dataset shape: {merged_df.shape}")
        
        # 🔥 Create reconstruction quality visualization
        recon_plot_path = create_reconstruction_quality_plot(merged_df, height_columns, output_dir)
        
        # 🔥 Save reconstruction metrics
        metrics_path = os.path.join(output_dir, 'reconstruction_quality_metrics.json')
        with open(metrics_path, 'w') as f:
            # Convert numpy types to native Python types for JSON serialization
            metrics_serializable = {}
            for height, metrics in reconstruction_metrics.items():
                metrics_serializable[height] = {
                    key: float(value) for key, value in metrics.items()
                }
            json.dump(metrics_serializable, f, indent=2)
        print(f"✅ Reconstruction metrics saved: {metrics_path}")
        
        # Create enhanced dataset description
        description = f"""
Complete EOF1 Modeling Dataset with Reconstructed Wind Speeds
============================================================

File: complete_eof1_modeling_dataset.csv
Created: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}

DATASET OVERVIEW:
- Total samples: {len(merged_df)}
- Time range: {merged_df['datetime'].min()} to {merged_df['datetime'].max()}
- Columns: {merged_df.shape[1]}

KEY VARIABLES FOR EOF1 PREDICTION:
- eof1_target: Target variable (EOF1 time coefficient)
- corrected_wind_speed_10m: Primary feature (corrected 10m wind speed)

AVAILABLE FEATURES:
Strategy C (Simple):
  - corrected_wind_speed_10m

Strategy A (Complete):
  - corrected_wind_speed_10m
  - ec_wind_speed_10m (if available)
  - gfs_wind_speed_10m (if available)
  - ec_gfs_mean (if available)
  - ec_gfs_diff (if available)
  - hour_sin, day_sin (if available)

OBSERVATIONAL DATA:
  - obs_wind_speed_10m, obs_wind_speed_30m
  - obs_wind_speed_50m, obs_wind_speed_70m

🔥 RECONSTRUCTED WIND SPEEDS (NEW):
  - reconstructed_10m: EOF1-based 10m wind reconstruction
  - reconstructed_30m: EOF1-based 30m wind reconstruction  
  - reconstructed_50m: EOF1-based 50m wind reconstruction
  - reconstructed_70m: EOF1-based 70m wind reconstruction

RECONSTRUCTION QUALITY:"""

        # Add reconstruction metrics to description
        for height, metrics in reconstruction_metrics.items():
            height_name = height.split('_')[-1]
            description += f"""
  {height_name}:
    - Correlation with observed: {metrics['correlation']:.4f}
    - RMSE: {metrics['rmse']:.4f} m/s
    - MAE: {metrics['mae']:.4f} m/s
    - Bias: {metrics['bias']:.4f} m/s"""

        description += f"""

EOF COMPONENTS:
  - PC1, PC2, PC3, PC4: Principal component time series
  - eof1_target: Primary target (copy of PC1)

DATA SPLIT:
  - data_split: {'train/test available' if has_split else 'NOT AVAILABLE - create 80/20 split'}

RECONSTRUCTION METHODOLOGY:
  - Method: wind_reconstructed = mean_wind + EOF1_timeseries × EOF1_spatial_coefficient
  - EOF1 explains: {eof_summary['eof_statistics']['eof1_variance_percent']:.1f}% of wind profile variance
  - All heights reconstructed using same EOF1 time series

NEXT STEPS:
1. Load this dataset
2. Train model: corrected_10m → eof1_target
3. Predict EOF1 for new data
4. Reconstruct 70m: predicted_EOF1 × EOF1_70m_coeff + mean_70m
5. Compare with reconstructed_70m column for validation

VALIDATION OPPORTUNITY:
- Use reconstructed_70m vs obs_wind_speed_70m to validate EOF approach
- High correlation indicates EOF method is viable for this dataset
"""
        
        desc_path = os.path.join(output_dir, 'modeling_dataset_description.txt')
        with open(desc_path, 'w') as f:
            f.write(description)
        
        print(f"✅ Enhanced dataset description saved: {desc_path}")
        
        return complete_path
        
    except Exception as e:
        print(f"❌ ERROR creating complete dataset: {str(e)}")
        import traceback
        traceback.print_exc()
        print(f"💡 You may need to manually merge the datasets")
        return None


def create_reconstruction_quality_plot(merged_df, height_columns, output_dir):
    """
    Create visualization showing reconstruction quality for all heights
    
    Parameters:
    -----------
    merged_df : pandas.DataFrame
        Complete dataset with observed and reconstructed winds
    height_columns : list
        List of height column names
    output_dir : str
        Output directory for plots
        
    Returns:
    --------
    str
        Path to saved plot
    """
    print(f"📊 Creating reconstruction quality visualization...")
    
    # Count available heights with both observed and reconstructed data
    available_heights = []
    for height_col in height_columns:
        recon_col = f'reconstructed_{height_col.split("_")[-1]}'
        if height_col in merged_df.columns and recon_col in merged_df.columns:
            available_heights.append((height_col, recon_col))
    
    n_heights = len(available_heights)
    if n_heights == 0:
        print(f"⚠️  No heights available for reconstruction quality plot")
        return None
    
    # Create subplot grid
    n_cols = min(2, n_heights)
    n_rows = (n_heights + n_cols - 1) // n_cols
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(6*n_cols, 5*n_rows))
    if n_heights == 1:
        axes = [axes]
    elif n_rows == 1:
        axes = axes.reshape(1, -1)
    
    plot_idx = 0
    
    for obs_col, recon_col in available_heights:
        row = plot_idx // n_cols
        col = plot_idx % n_cols
        ax = axes[row, col] if n_rows > 1 else axes[col]
        
        # Get data
        obs_data = merged_df[obs_col].values
        recon_data = merged_df[recon_col].values
        
        # Create scatter plot
        ax.scatter(obs_data, recon_data, alpha=0.3, s=1, color='blue')
        
        # Add perfect prediction line
        min_val = min(obs_data.min(), recon_data.min())
        max_val = max(obs_data.max(), recon_data.max())
        ax.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2, label='Perfect')
        
        # Calculate and display metrics
        correlation = np.corrcoef(obs_data, recon_data)[0, 1]
        rmse = np.sqrt(np.mean((obs_data - recon_data)**2))
        
        # Extract height for title
        height_name = obs_col.split('_')[-1]
        ax.set_title(f'{height_name} Reconstruction\nCorr: {correlation:.3f}, RMSE: {rmse:.2f}')
        ax.set_xlabel(f'Observed {height_name} (m/s)')
        ax.set_ylabel(f'Reconstructed {height_name} (m/s)')
        ax.grid(True, alpha=0.3)
        ax.legend()
        
        plot_idx += 1
    
    # Hide unused subplots
    for idx in range(plot_idx, n_rows * n_cols):
        row = idx // n_cols
        col = idx % n_cols
        ax = axes[row, col] if n_rows > 1 else axes[col]
        ax.set_visible(False)
    
    plt.tight_layout()
    
    # Save plot
    plot_path = os.path.join(output_dir, 'reconstruction_quality_assessment.png')
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    print(f"✅ Reconstruction quality plot saved: {plot_path}")
    plt.show()
    
    return plot_path


def main_eof_analysis(corrected_data_path, output_dir, standardize=True):
    """
    Main function to run complete EOF analysis pipeline
    
    Parameters:
    -----------
    corrected_data_path : str
        Path to corrected wind speed dataset from Step 1
    output_dir : str
        Directory to save all results
    standardize : bool
        Whether to standardize wind profile data
        
    Returns:
    --------
    dict
        Complete analysis results and file paths
    """
    print("*" * 80)
    print("WIND PROFILE EOF ANALYSIS PIPELINE")
    print("*" * 80)
    
    # Initialize analyzer
    eof_analyzer = WindProfileEOF(standardize=standardize)
    
    try:
        # Step 1: Load wind profile data
        wind_data = eof_analyzer.load_data(corrected_data_path)
        
        # Step 2: Perform EOF decomposition
        eof_results = eof_analyzer.perform_eof_decomposition(n_components=4)
        
        # Step 3: Create visualizations
        plot_paths = eof_analyzer.create_visualizations(eof_results, output_dir)
        
        # Step 4: Save EOF results
        file_paths = eof_analyzer.save_results(eof_results, output_dir)
        
        # Step 5: Create complete modeling dataset
        complete_dataset_path = create_complete_modeling_dataset(
            corrected_data_path, file_paths, output_dir
        )
        
        # Final summary
        print("\n" + "=" * 80)
        print("EOF ANALYSIS PIPELINE COMPLETED SUCCESSFULLY!")
        print("=" * 80)
        
        # Key metrics
        eof1_variance = eof_results['explained_variance_ratio'][0] * 100
        total_variance = eof_results['total_variance_explained']
        eof1_coeff_70m = eof_results['eof_modes'][-1, 0]  # 70m coefficient in EOF1
        
        print(f"\nKEY RESULTS:")
        print(f"  📊 Dataset: {eof_results['n_samples']} samples, {eof_results['n_heights']} heights")
        print(f"  🎯 EOF1 variance explained: {eof1_variance:.1f}%")
        print(f"  📈 Total variance explained: {total_variance:.1f}%")
        print(f"  ⚖️  EOF1 coefficient for 70m: {eof1_coeff_70m:.4f}")
        
        # Data quality assessment
        print(f"\n  📋 QUALITY ASSESSMENT:")
        if eof1_variance > 60:
            quality_eof1 = "EXCELLENT"
        elif eof1_variance > 40:
            quality_eof1 = "GOOD"
        else:
            quality_eof1 = "FAIR"
        
        if total_variance > 80:
            quality_total = "EXCELLENT"
        elif total_variance > 70:
            quality_total = "GOOD"
        else:
            quality_total = "FAIR"
        
        print(f"    EOF1 dominance: {quality_eof1} ({eof1_variance:.0f}%)")
        print(f"    Total capture: {quality_total} ({total_variance:.0f}%)")
        
        # Files summary
        print(f"\n  📁 FILES CREATED:")
        for key, path in file_paths.items():
            print(f"    {key}: {path}")
        
        if complete_dataset_path:
            print(f"    complete_dataset: {complete_dataset_path}")
            print(f"    🔥 reconstruction_quality_plot: {output_dir}/reconstruction_quality_assessment.png")
            print(f"    🔥 reconstruction_metrics: {output_dir}/reconstruction_quality_metrics.json")
        
        print(f"    visualizations: {len(plot_paths)} plots in {output_dir}")
        
        # Next step readiness
        print(f"\n  🎯 NEXT STEP READINESS:")
        if complete_dataset_path:
            print(f"    ✅ Complete modeling dataset with reconstructed winds ready")
            print(f"    📂 Dataset path: {complete_dataset_path}")
            print(f"    🎯 Target variable: eof1_target")
            print(f"    🔧 Main feature: corrected_wind_speed_10m")
            print(f"    🔄 Reconstructed winds: reconstructed_10m, reconstructed_30m, reconstructed_50m, reconstructed_70m")
        else:
            print(f"    ⚠️  Manual dataset merge required")
        
        if eof1_variance > 50:
            print(f"    ✅ EOF1 variance sufficient for prediction")
        else:
            print(f"    ⚠️  EOF1 variance low - consider multiple EOFs")
        
        # 🔥 Reconstruction quality summary
        if complete_dataset_path:
            print(f"\n  📊 RECONSTRUCTION QUALITY PREVIEW:")
            print(f"    🔄 All wind heights reconstructed using EOF1 method")
            print(f"    📈 Check reconstruction_quality_assessment.png for detailed validation")
            print(f"    📋 Metrics saved in reconstruction_quality_metrics.json")
            print(f"    ✨ This validates the EOF approach before building prediction models")
        
        # Return comprehensive results
        final_results = {
            'eof_analyzer': eof_analyzer,
            'eof_results': eof_results,
            'file_paths': file_paths,
            'plot_paths': plot_paths,
            'complete_dataset_path': complete_dataset_path,
            'summary_metrics': {
                'eof1_variance_percent': eof1_variance,
                'total_variance_percent': total_variance,
                'n_samples': eof_results['n_samples'],
                'n_heights': eof_results['n_heights'],
                'eof1_coeff_70m': eof1_coeff_70m,
                'quality_assessment': {
                    'eof1_quality': quality_eof1,
                    'total_quality': quality_total
                }
            }
        }
        
        print(f"\n🎉 EOF Analysis with Wind Reconstruction completed!")
        print(f"📊 Dataset now includes reconstructed wind speeds for validation")
        return final_results
        
    except Exception as e:
        print(f"\n❌ ERROR in EOF Analysis Pipeline: {str(e)}")
        import traceback
        traceback.print_exc()
        return None


# Example usage
if __name__ == "__main__":
    # Configuration
    CORRECTED_DATA_PATH = '/Users/xiaxin/work/WindForecast_Project/03_Results/建模/10m风速建模/LightGBM_增强版/corrected_10m_wind_full_dataset.csv'
    OUTPUT_DIR = '/Users/xiaxin/work/WindForecast_Project/03_Results/建模/EOF分析/'
    
    # Run complete EOF analysis
    results = main_eof_analysis(
        corrected_data_path=CORRECTED_DATA_PATH,
        output_dir=OUTPUT_DIR,
        standardize=True  # Try both True and False to compare
    )
    
    if results is not None:
        print(f"\n✅ SUCCESS: EOF analysis completed")
        print(f"📁 All results saved to: {OUTPUT_DIR}")
        
        if results['complete_dataset_path']:
            print(f"\n🎯 READY FOR NEXT STEP:")
            print(f"   Use dataset: {results['complete_dataset_path']}")
            print(f"   Target: eof1_target")
            print(f"   Main feature: corrected_wind_speed_10m")
            print(f"   EOF1 70m coefficient: {results['summary_metrics']['eof1_coeff_70m']:.4f}")
        
    else:
        print(f"\n❌ FAILED: Check error messages above")