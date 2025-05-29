#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Vertical Gradient Analysis: Analyze the relationship between wind speed and direction at different heights
Research Project: Wind Power Error Propagation Analysis Based on Multi-Source Meteorological Data
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from scipy.spatial.distance import pdist, squareform
import warnings
warnings.filterwarnings('ignore')

# Remove Chinese font settings
plt.rcParams['axes.unicode_minus'] = False

class VerticalGradientAnalyzer:
    def __init__(self, data_path, output_path):
        """
        Initialize the Vertical Gradient Analyzer
        
        Parameters:
        -----------
        data_path : str
            Path to the data file
        output_path : str
            Path to save results
        """
        self.data_path = data_path
        self.output_path = output_path
        self.data = None
        self.heights = [10, 30, 50, 70]  # Adjust heights based on actual data
        self.data_sources = ['obs', 'ec', 'gfs']
        
    def load_data(self):
        """Load data"""
        print("Loading data...")
        # Try different file names
        possible_files = [
            f"{self.data_path}/complete_imputed_data.csv",
            f"{self.data_path}/changma_imputed_complete.csv"
        ]
        
        data_loaded = False
        for file_path in possible_files:
            try:
                self.data = pd.read_csv(file_path)
                print(f"Successfully loaded data file: {file_path}")
                data_loaded = True
                break
            except FileNotFoundError:
                continue
        
        if not data_loaded:
            raise FileNotFoundError("Data file not found, please check the path and file name")
        
        self.data['datetime'] = pd.to_datetime(self.data['datetime'])
        self.data.set_index('datetime', inplace=True)
        
        # Check and map actual column names
        self._map_column_names()
        
        print(f"Data loading complete, total {len(self.data)} records")
        print(f"Sample available columns: {list(self.data.columns)[:10]}")
        
    def _map_column_names(self):
        """Map column names to fit the actual data format"""
        # Check the actual column name format in the data
        if 'obs_wind_speed_10m' in self.data.columns:
            # Data uses full format, rename for subsequent processing
            column_mapping = {}
            
            for source in self.data_sources:
                for height in self.heights:
                    # Wind speed column mapping
                    old_ws_col = f'{source}_wind_speed_{height}m'
                    new_ws_col = f'{source}_ws_{height}m'
                    if old_ws_col in self.data.columns:
                        column_mapping[old_ws_col] = new_ws_col
                    
                    # Wind direction column mapping
                    old_wd_col = f'{source}_wind_direction_{height}m'
                    new_wd_col = f'{source}_wd_{height}m'
                    if old_wd_col in self.data.columns:
                        column_mapping[old_wd_col] = new_wd_col
                    
                    # Temperature column mapping
                    old_temp_col = f'{source}_temperature_{height}m'
                    new_temp_col = f'{source}_temp_{height}m'
                    if old_temp_col in self.data.columns:
                        column_mapping[old_temp_col] = new_temp_col
                    
                    # Density column mapping (only at 10m height)
                    if height == 10:
                        old_density_col = f'{source}_density_{height}m'
                        new_density_col = f'{source}_density_{height}m'
                        if old_density_col in self.data.columns:
                            column_mapping[old_density_col] = new_density_col
            
            # Perform renaming
            self.data.rename(columns=column_mapping, inplace=True)
            print(f"Renamed {len(column_mapping)} columns to meet analysis requirements")
            
            # Add missing variables (if needed)
            if 'density' not in self.data.columns and 'obs_density_10m' in self.data.columns:
                self.data['density'] = self.data['obs_density_10m']
        
    def calculate_wind_speed_gradient(self):
        """Calculate vertical wind speed gradient"""
        print("\n=== Wind Speed Vertical Gradient Analysis ===")
        
        gradient_results = {}
        
        for source in self.data_sources:
            print(f"\nAnalyzing {source.upper()} data source:")
            
            # Extract wind speed data for this source
            ws_cols = [f'{source}_ws_{h}m' for h in self.heights]
            
            # Check which columns exist
            available_cols = [col for col in ws_cols if col in self.data.columns]
            if len(available_cols) < 2:
                print(f"  Warning: {source.upper()} data source has insufficient wind speed columns, skipping gradient calculation")
                print(f"  Expected columns: {ws_cols}")
                print(f"  Available columns: {available_cols}")
                continue
                
            ws_data = self.data[available_cols].copy()
            
            # Get the corresponding actual heights
            available_heights = []
            for col in available_cols:
                for h in self.heights:
                    if f'_{h}m' in col:
                        available_heights.append(h)
                        break
            
            # Calculate gradients between adjacent heights and additional pairs
            gradients = {}
            height_pairs = [(available_heights[i], available_heights[j]) for i in range(len(available_heights)) for j in range(i+1, len(available_heights)) if j-i >= 1]
            for h1, h2 in height_pairs:
                col1, col2 = f'{source}_ws_{h1}m', f'{source}_ws_{h2}m'
                
                if col1 in ws_data.columns and col2 in ws_data.columns:
                    # Wind speed gradient (m/s)/m = (WS_upper - WS_lower) / (H_upper - H_lower)
                    gradient = (ws_data[col2] - ws_data[col1]) / (h2 - h1)
                    gradients[f'{h1}-{h2}m'] = gradient
                    
                    print(f"  {h1}-{h2}m gradient statistics:")
                    print(f"    Mean: {gradient.mean():.4f} (m/s)/m")
                    print(f"    Standard deviation: {gradient.std():.4f} (m/s)/m")
                    print(f"    Median: {gradient.median():.4f} (m/s)/m")
                
            gradient_results[source] = gradients
            
        return gradient_results
    
    def calculate_wind_direction_gradient(self):
        """Calculate vertical wind direction gradient (handling 360-degree wraparound)"""
        print("\n=== Wind Direction Vertical Gradient Analysis ===")
        
        def wind_direction_difference(wd1, wd2):
            """Calculate wind direction difference, considering 360-degree wraparound"""
            diff = wd2 - wd1
            # Handle crossing 0/360 degrees
            diff = np.where(diff > 180, diff - 360, diff)
            diff = np.where(diff < -180, diff + 360, diff)
            return diff
        
        gradient_results = {}
        
        for source in self.data_sources:
            print(f"\nAnalyzing {source.upper()} data source:")
            
            # Extract wind direction data for this source
            wd_cols = [f'{source}_wd_{h}m' for h in self.heights]
            
            # Check which columns exist
            available_cols = [col for col in wd_cols if col in self.data.columns]
            if len(available_cols) < 2:
                print(f"  Warning: {source.upper()} data source has insufficient wind direction columns, skipping gradient calculation")
                continue
                
            wd_data = self.data[available_cols].copy()
            
            # Get the corresponding actual heights
            available_heights = []
            for col in available_cols:
                for h in self.heights:
                    if f'_{h}m' in col:
                        available_heights.append(h)
                        break
            
            # Calculate wind direction differences between adjacent heights and additional pairs
            gradients = {}
            height_pairs = [(available_heights[i], available_heights[j]) for i in range(len(available_heights)) for j in range(i+1, len(available_heights)) if j-i >= 1]
            for h1, h2 in height_pairs:
                col1, col2 = f'{source}_wd_{h1}m', f'{source}_wd_{h2}m'
                
                if col1 in wd_data.columns and col2 in wd_data.columns:
                    # Wind direction gradient degree/m
                    gradient = wind_direction_difference(wd_data[col1], wd_data[col2]) / (h2 - h1)
                    gradients[f'{h1}-{h2}m'] = gradient
                    
                    print(f"  {h1}-{h2}m wind direction difference statistics:")
                    print(f"    Mean: {gradient.mean():.4f} °/m")
                    print(f"    Standard deviation: {gradient.std():.4f} °/m")
                    print(f"    Median: {np.median(gradient):.4f} °/m")
                    print(f"    Proportion of |difference|>10°: {(np.abs(gradient * (h2-h1)) > 10).mean():.2%}")
                
            gradient_results[source] = gradients
            
        return gradient_results
    
    def analyze_wind_shear(self):
        """Analyze wind shear exponent"""
        print("\n=== Wind Shear Exponent Analysis ===")
        
        shear_results = {}
        
        for source in self.data_sources:
            print(f"\nAnalyzing {source.upper()} data source:")
            
            # Select available heights for wind shear analysis
            available_heights = []
            for h in self.heights:
                col = f'{source}_ws_{h}m'
                if col in self.data.columns:
                    available_heights.append(h)
            
            if len(available_heights) < 2:
                print(f"  Warning: {source.upper()} data source has insufficient heights, skipping wind shear analysis")
                continue
            
            # Select appropriate height pairs
            ref_pairs = []
            if len(available_heights) >= 2:
                # Use lowest and highest height
                ref_pairs.append((available_heights[0], available_heights[-1]))
                # If there are intermediate heights, add some combinations
                if len(available_heights) >= 3:
                    ref_pairs.append((available_heights[0], available_heights[1]))
                    ref_pairs.append((available_heights[1], available_heights[-1]))
            
            shear_data = {}
            for h1, h2 in ref_pairs:
                ws1_col = f'{source}_ws_{h1}m'
                ws2_col = f'{source}_ws_{h2}m'
                
                if ws1_col in self.data.columns and ws2_col in self.data.columns:
                    # Wind shear exponent formula: α = ln(WS2/WS1) / ln(H2/H1)
                    # Avoid division by zero and logarithm issues
                    mask = (self.data[ws1_col] > 0.5) & (self.data[ws2_col] > 0.5)
                    valid_data = self.data[mask]
                    
                    if len(valid_data) > 100:
                        alpha = (np.log(valid_data[ws2_col]) - np.log(valid_data[ws1_col])) / \
                               (np.log(h2) - np.log(h1))
                        
                        shear_data[f'{h1}-{h2}m'] = alpha
                        
                        print(f"  {h1}-{h2}m wind shear exponent:")
                        print(f"    Mean: {alpha.mean():.3f}")
                        print(f"    Standard deviation: {alpha.std():.3f}")
                        print(f"    Median: {alpha.median():.3f}")
                        print(f"    Valid sample count: {len(alpha)}")
                        
            shear_results[source] = shear_data
            
        return shear_results
    
    def analyze_height_correlation(self):
        """Analyze correlation between different heights"""
        print("\n=== Height Correlation Analysis ===")
        
        correlation_results = {}
        
        for source in self.data_sources:
            print(f"\nAnalyzing {source.upper()} data source:")
            
            # Wind speed correlation
            ws_cols = [f'{source}_ws_{h}m' for h in self.heights if f'{source}_ws_{h}m' in self.data.columns]
            if len(ws_cols) >= 2:
                ws_corr = self.data[ws_cols].corr()
                print("  Wind speed correlation matrix:")
                print(ws_corr.round(3))
            else:
                ws_corr = pd.DataFrame()
                print("  Warning: Insufficient wind speed columns, cannot compute correlation")
            
            # Wind direction correlation (requires special handling)
            wd_cols = [f'{source}_wd_{h}m' for h in self.heights if f'{source}_wd_{h}m' in self.data.columns]
            
            if len(wd_cols) >= 2:
                wd_data = self.data[wd_cols].copy()
                
                # Convert wind direction to unit vectors for correlation analysis
                wd_corr = pd.DataFrame(index=wd_cols, columns=wd_cols)
                for col1 in wd_cols:
                    for col2 in wd_cols:
                        if col1 == col2:
                            wd_corr.loc[col1, col2] = 1.0
                        else:
                            # Use circular correlation coefficient
                            angle1 = np.radians(wd_data[col1])
                            angle2 = np.radians(wd_data[col2])
                            
                            # Calculate circular correlation coefficient
                            cos_diff = np.cos(angle1 - angle2)
                            circular_corr = cos_diff.mean()
                            wd_corr.loc[col1, col2] = circular_corr
                
                print("\n  Wind direction correlation matrix:")
                print(wd_corr.astype(float).round(3))
            else:
                wd_corr = pd.DataFrame()
                print("\n  Warning: Insufficient wind direction columns, cannot compute correlation")
            
            correlation_results[source] = {
                'wind_speed': ws_corr,
                'wind_direction': wd_corr.astype(float) if len(wd_corr) > 0 else pd.DataFrame()
            }
            
        return correlation_results
    
    def plot_gradient_distributions(self, ws_gradients, wd_gradients):
        """Plot gradient distributions"""
        print("\nGenerating gradient distribution plots...")
        
        # Wind speed gradient distribution
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle('Vertical Gradient Distribution Analysis', fontsize=16, fontweight='bold')
        
        # Wind speed gradient
        for i, source in enumerate(self.data_sources):
            ax = axes[0, i]
            gradients = ws_gradients[source]
            
            for layer, gradient in gradients.items():
                h1, h2 = map(int, [h.rstrip('m') for h in layer.split('-')])
                ax.hist(gradient, bins=50, alpha=0.7, label=f'{h1}-{h2}m', density=True)
            
            ax.set_title(f'{source.upper()} Wind Speed Gradient Distribution')
            ax.set_xlabel('Wind Speed Gradient [(m/s)/m]')
            ax.set_ylabel('Probability Density')
            ax.legend()
            ax.grid(True, alpha=0.3)
        
        # Wind direction gradient
        for i, source in enumerate(self.data_sources):
            ax = axes[1, i]
            gradients = wd_gradients[source]
            
            for layer, gradient in gradients.items():
                h1, h2 = map(int, [h.rstrip('m') for h in layer.split('-')])
                ax.hist(gradient, bins=50, alpha=0.7, label=f'{h1}-{h2}m', density=True)
            
            ax.set_title(f'{source.upper()} Wind Direction Gradient Distribution')
            ax.set_xlabel('Wind Direction Gradient [°/m]')
            ax.set_ylabel('Probability Density')
            ax.legend()
            ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(f'{self.output_path}/vertical_gradient_distributions.png', 
                   dpi=300, bbox_inches='tight')
        plt.show()
    
    def plot_correlation_heatmaps(self, correlations):
        """Plot correlation heatmaps"""
        print("\nGenerating correlation heatmaps...")
        
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle('Correlation Analysis Between Heights', fontsize=16, fontweight='bold')
        
        # Wind speed correlation
        for i, source in enumerate(self.data_sources):
            ax = axes[0, i]
            corr_matrix = correlations[source]['wind_speed']
            
            # Rename columns and index for cleaner display
            height_labels = [f'{h}m' for h in self.heights]
            corr_matrix.index = height_labels
            corr_matrix.columns = height_labels
            
            sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', center=0,
                       ax=ax, cbar_kws={'label': 'Correlation Coefficient'})
            ax.set_title(f'{source.upper()} Wind Speed Correlation Between Heights')
        
        # Wind direction correlation
        for i, source in enumerate(self.data_sources):
            ax = axes[1, i]
            corr_matrix = correlations[source]['wind_direction']
            
            # Rename columns and index
            height_labels = [f'{h}m' for h in self.heights]
            corr_matrix.index = height_labels
            corr_matrix.columns = height_labels
            
            sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', center=0,
                       ax=ax, cbar_kws={'label': 'Circular Correlation Coefficient'})
            ax.set_title(f'{source.upper()} Wind Direction Correlation Between Heights')
        
        plt.tight_layout()
        plt.savefig(f'{self.output_path}/height_correlation_heatmaps.png', 
                   dpi=300, bbox_inches='tight')
        plt.show()
    
    def plot_wind_shear_analysis(self, shear_results):
        """Plot wind shear analysis"""
        print("\nGenerating wind shear analysis plots...")
        
        fig, axes = plt.subplots(1, 3, figsize=(18, 6))
        fig.suptitle('Wind Shear Exponent Analysis', fontsize=16, fontweight='bold')
        
        for i, source in enumerate(self.data_sources):
            ax = axes[i]
            shear_data = shear_results[source]
            
            for layer, alpha in shear_data.items():
                ax.hist(alpha, bins=50, alpha=0.7, label=f'{layer}', density=True)
            
            ax.axvline(x=0.143, color='red', linestyle='--', 
                      label='Theoretical Value (1/7 Rule)', linewidth=2)
            ax.set_title(f'{source.upper()} Wind Shear Exponent Distribution')
            ax.set_xlabel('Wind Shear Exponent α')
            ax.set_ylabel('Probability Density')
            ax.legend()
            ax.grid(True, alpha=0.3)
            ax.set_xlim(-0.5, 1.0)
        
        plt.tight_layout()
        plt.savefig(f'{self.output_path}/wind_shear_analysis.png', 
                   dpi=300, bbox_inches='tight')
        plt.show()
    
    def save_results(self, ws_gradients, wd_gradients, shear_results, correlations):
        """Save analysis results"""
        print("\nSaving analysis results...")
        
        # Create result summary
        summary = {
            'analysis_type': 'vertical_gradient_analysis',
            'data_period': f"{self.data.index.min()} to {self.data.index.max()}",
            'total_records': len(self.data),
            'heights_analyzed': self.heights,
            'data_sources': self.data_sources
        }
        
        # Wind speed gradient statistics
        ws_gradient_stats = {}
        for source in self.data_sources:
            ws_gradient_stats[source] = {}
            for layer, gradient in ws_gradients[source].items():
                ws_gradient_stats[source][layer] = {
                    'mean': float(gradient.mean()),
                    'std': float(gradient.std()),
                    'median': float(gradient.median()),
                    'q25': float(gradient.quantile(0.25)),
                    'q75': float(gradient.quantile(0.75))
                }
        
        # Wind direction gradient statistics
        wd_gradient_stats = {}
        for source in self.data_sources:
            wd_gradient_stats[source] = {}
            for layer, gradient in wd_gradients[source].items():
                wd_gradient_stats[source][layer] = {
                    'mean': float(gradient.mean()),
                    'std': float(gradient.std()),
                    'median': float(np.median(gradient)),
                    'large_diff_ratio': float((np.abs(gradient * 60) > 10).mean())  # 60m layer difference
                }
        
        # Wind shear statistics
        shear_stats = {}
        for source in self.data_sources:
            shear_stats[source] = {}
            for layer, alpha in shear_results[source].items():
                shear_stats[source][layer] = {
                    'mean': float(alpha.mean()),
                    'std': float(alpha.std()),
                    'median': float(alpha.median()),
                    'valid_samples': int(len(alpha))
                }
        
        # Correlation statistics
        correlation_stats = {}
        for source in self.data_sources:
            correlation_stats[source] = {
                'wind_speed_corr': correlations[source]['wind_speed'].to_dict(),
                'wind_direction_corr': correlations[source]['wind_direction'].to_dict()
            }
        
        # Save as JSON
        import json
        results = {
            'summary': summary,
            'wind_speed_gradients': ws_gradient_stats,
            'wind_direction_gradients': wd_gradient_stats,
            'wind_shear_indices': shear_stats,
            'height_correlations': correlation_stats
        }
        
        with open(f'{self.output_path}/vertical_gradient_analysis_results.json', 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        
        print(f"Results saved to: {self.output_path}/vertical_gradient_analysis_results.json")
    
    def run_analysis(self):
        """Run the complete vertical gradient analysis"""
        print("Starting vertical gradient analysis...")
        print("="*60)
        
        # Load data
        self.load_data()
        
        # Perform analyses
        ws_gradients = self.calculate_wind_speed_gradient()
        wd_gradients = self.calculate_wind_direction_gradient()
        shear_results = self.analyze_wind_shear()
        correlations = self.analyze_height_correlation()
        
        # Generate plots
        self.plot_gradient_distributions(ws_gradients, wd_gradients)
        self.plot_correlation_heatmaps(correlations)
        self.plot_wind_shear_analysis(shear_results)
        
        # Save results
        self.save_results(ws_gradients, wd_gradients, shear_results, correlations)
        
        print("\n" + "="*60)
        print("Vertical gradient analysis completed!")
        print(f"Result plots and data saved to: {self.output_path}")

if __name__ == "__main__":
    # Set paths
    DATA_PATH = "/Users/xiaxin/work/WindForecast_Project/01_Data/processed/imputed_data"
    OUTPUT_PATH = "/Users/xiaxin/work/WindForecast_Project/03_Results/1_2_3_vertical_analysis"
    
    # Create analyzer and run analysis
    analyzer = VerticalGradientAnalyzer(DATA_PATH, OUTPUT_PATH)
    analyzer.run_analysis()