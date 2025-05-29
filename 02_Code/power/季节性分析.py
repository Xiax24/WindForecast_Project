#!/usr/bin/env python3
"""
Debug and Fixed Seasonal Analysis for Wind Power Forecasting
Author: Research Team
Date: 2025-05-29
Purpose: Debug data loading and create seasonal/diurnal pattern analysis
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import warnings
from pathlib import Path
import os

# Suppress warnings
warnings.filterwarnings('ignore')

# Set plot style
plt.style.use('default')
plt.rcParams['figure.figsize'] = (12, 8)
plt.rcParams['font.size'] = 10

class DebugSeasonalAnalyzer:
    """
    Debug version of seasonal pattern analyzer
    """
    
    def __init__(self, data_path, results_path):
        self.data_path = Path(data_path)
        self.results_path = Path(results_path)
        self.results_path.mkdir(parents=True, exist_ok=True)
        
        # Create subdirectories
        self.seasonal_path = self.results_path / "seasonal_analysis"
        self.seasonal_path.mkdir(exist_ok=True)
        
        self.data = None
        
    def debug_data_loading(self):
        """Debug data loading step by step"""
        print("=== DEBUGGING DATA LOADING ===")
        
        # Check if data path exists
        print(f"Data path: {self.data_path}")
        print(f"Path exists: {self.data_path.exists()}")
        
        if self.data_path.exists():
            print("\nFiles in data directory:")
            for file in self.data_path.iterdir():
                print(f"  - {file.name}")
        
        # Try to find the data file
        possible_files = [
            "changma_imputed_complete.csv",
            "processed_data.csv", 
            "wind_data.csv",
            "imputed_data.csv"
        ]
        
        data_file = None
        for filename in possible_files:
            filepath = self.data_path / filename
            if filepath.exists():
                data_file = filepath
                print(f"\nFound data file: {filename}")
                break
        
        if data_file is None:
            print("\nNo recognized data file found. Available files:")
            if self.data_path.exists():
                for file in self.data_path.iterdir():
                    if file.suffix == '.csv':
                        print(f"  - {file.name}")
            return False
        
        # Try to load the data
        try:
            print(f"\nAttempting to load: {data_file}")
            self.data = pd.read_csv(data_file)
            print(f"Data loaded successfully!")
            print(f"Shape: {self.data.shape}")
            
            # Show column names
            print(f"\nColumn names ({len(self.data.columns)} total):")
            for i, col in enumerate(self.data.columns):
                print(f"  {i+1:2d}. {col}")
            
            # Show first few rows
            print(f"\nFirst 3 rows:")
            print(self.data.head(3))
            
            # Check data types
            print(f"\nData types:")
            print(self.data.dtypes)
            
            return True
            
        except Exception as e:
            print(f"Error loading data: {e}")
            return False
    
    def identify_variables(self):
        """Identify available variables in the dataset"""
        print("\n=== IDENTIFYING VARIABLES ===")
        
        if self.data is None:
            print("No data loaded!")
            return
        
        # Look for different variable patterns
        variable_patterns = {
            'datetime': ['datetime', 'time', 'timestamp', 'date'],
            'wind_speed': ['ws_', 'wind_speed', 'windspeed', 'u_', 'v_'],
            'wind_direction': ['wd_', 'wind_dir', 'winddir', 'direction'],
            'temperature': ['temp_', 'temperature', 't_', 'temp'],
            'pressure': ['pres_', 'pressure', 'p_', 'press'],
            'power': ['power', 'Power', 'POWER', 'output']
        }
        
        self.identified_vars = {}
        
        for var_type, patterns in variable_patterns.items():
            found_vars = []
            for col in self.data.columns:
                for pattern in patterns:
                    if pattern.lower() in col.lower():
                        found_vars.append(col)
                        break
            
            self.identified_vars[var_type] = found_vars
            print(f"\n{var_type.upper()} variables found:")
            if found_vars:
                for var in found_vars:
                    print(f"  - {var}")
            else:
                print("  - None found")
        
        return self.identified_vars
    
    def prepare_datetime(self):
        """Prepare datetime column"""
        print("\n=== PREPARING DATETIME ===")
        
        # Find datetime column
        datetime_col = None
        datetime_candidates = self.identified_vars.get('datetime', [])
        
        if datetime_candidates:
            datetime_col = datetime_candidates[0]
        else:
            # Look for any column that might be datetime
            for col in self.data.columns:
                if 'date' in col.lower() or 'time' in col.lower():
                    datetime_col = col
                    break
        
        if datetime_col:
            print(f"Using datetime column: {datetime_col}")
            try:
                self.data['datetime'] = pd.to_datetime(self.data[datetime_col])
                print("Datetime conversion successful!")
                
                # Add time components
                self.data['year'] = self.data['datetime'].dt.year
                self.data['month'] = self.data['datetime'].dt.month
                self.data['day'] = self.data['datetime'].dt.day
                self.data['hour'] = self.data['datetime'].dt.hour
                self.data['dayofyear'] = self.data['datetime'].dt.dayofyear
                
                # Add season
                self.data['season'] = self.data['month'].map({
                    12: 'Winter', 1: 'Winter', 2: 'Winter',
                    3: 'Spring', 4: 'Spring', 5: 'Spring', 
                    6: 'Summer', 7: 'Summer', 8: 'Summer',
                    9: 'Autumn', 10: 'Autumn', 11: 'Autumn'
                })
                
                print(f"Date range: {self.data['datetime'].min()} to {self.data['datetime'].max()}")
                print("Time components added successfully!")
                
                return True
                
            except Exception as e:
                print(f"Error converting datetime: {e}")
                return False
        else:
            print("No datetime column found!")
            return False
    
    def create_sample_analysis(self):
        """Create analysis with available variables"""
        print("\n=== CREATING SAMPLE ANALYSIS ===")
        
        # Select variables for analysis
        analysis_vars = {}
        
        # Wind speed
        if self.identified_vars['wind_speed']:
            analysis_vars['Wind Speed'] = self.identified_vars['wind_speed'][0]
        
        # Temperature  
        if self.identified_vars['temperature']:
            analysis_vars['Temperature'] = self.identified_vars['temperature'][0]
        
        # Power
        if self.identified_vars['power']:
            analysis_vars['Power'] = self.identified_vars['power'][0]
        
        # Pressure
        if self.identified_vars['pressure']:
            analysis_vars['Pressure'] = self.identified_vars['pressure'][0]
        
        print(f"Variables selected for analysis:")
        for name, col in analysis_vars.items():
            print(f"  - {name}: {col}")
        
        if not analysis_vars:
            print("No suitable variables found for analysis!")
            return
        
        # Create monthly patterns
        self._plot_monthly_patterns_debug(analysis_vars)
        
        # Create seasonal box plots
        self._plot_seasonal_boxplots_debug(analysis_vars)
        
        # Create diurnal patterns
        self._plot_diurnal_patterns_debug(analysis_vars)
    
    def _plot_monthly_patterns_debug(self, analysis_vars):
        """Plot monthly patterns with debug info"""
        print("\nCreating monthly patterns plot...")
        
        n_vars = len(analysis_vars)
        if n_vars == 0:
            return
        
        # Calculate subplot layout
        n_cols = min(3, n_vars)
        n_rows = (n_vars + n_cols - 1) // n_cols
        
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(6*n_cols, 5*n_rows))
        
        # Handle single subplot case
        if n_vars == 1:
            axes = [axes]
        elif n_rows == 1:
            axes = axes if n_vars > 1 else [axes]
        else:
            axes = axes.flatten()
        
        for i, (var_name, var_col) in enumerate(analysis_vars.items()):
            if i >= len(axes):
                break
                
            ax = axes[i]
            
            try:
                # Calculate monthly statistics
                monthly_stats = self.data.groupby('month')[var_col].agg([
                    'mean', 'std', 'median', 'count'
                ]).reset_index()
                
                print(f"  - {var_name}: {len(monthly_stats)} months of data")
                
                # Plot mean with error bars
                ax.errorbar(monthly_stats['month'], monthly_stats['mean'], 
                           yerr=monthly_stats['std'], 
                           marker='o', capsize=5, capthick=2, 
                           linewidth=2, markersize=8, label='Mean ± Std')
                
                # Plot median
                ax.plot(monthly_stats['month'], monthly_stats['median'], 
                       'r--', linewidth=2, marker='s', markersize=6, 
                       label='Median')
                
                ax.set_xlabel('Month')
                ax.set_ylabel(f'{var_name}')
                ax.set_title(f'Monthly Pattern - {var_name}')
                ax.legend()
                ax.grid(True, alpha=0.3)
                ax.set_xticks(range(1, 13))
                ax.set_xticklabels(['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun',
                                   'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'])
                
            except Exception as e:
                print(f"Error plotting {var_name}: {e}")
                ax.text(0.5, 0.5, f'Error plotting\n{var_name}', 
                       transform=ax.transAxes, ha='center', va='center')
        
        # Remove extra subplots
        for i in range(len(analysis_vars), len(axes)):
            fig.delaxes(axes[i])
        
        plt.tight_layout()
        plt.savefig(self.seasonal_path / 'monthly_patterns_debug.png', 
                   dpi=300, bbox_inches='tight')
        plt.show()
        print("Monthly patterns plot saved!")
    
    def _plot_seasonal_boxplots_debug(self, analysis_vars):
        """Plot seasonal box plots with debug info"""
        print("\nCreating seasonal box plots...")
        
        n_vars = len(analysis_vars)
        if n_vars == 0:
            return
        
        # Calculate subplot layout
        n_cols = min(2, n_vars)
        n_rows = (n_vars + n_cols - 1) // n_cols
        
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(8*n_cols, 6*n_rows))
        
        # Handle single subplot case
        if n_vars == 1:
            axes = [axes]
        elif n_rows == 1 and n_cols > 1:
            axes = list(axes)
        elif n_rows > 1:
            axes = axes.flatten()
        else:
            axes = [axes]
        
        season_order = ['Spring', 'Summer', 'Autumn', 'Winter']
        colors = ['lightgreen', 'gold', 'orange', 'lightblue']
        
        for i, (var_name, var_col) in enumerate(analysis_vars.items()):
            if i >= len(axes):
                break
                
            ax = axes[i]
            
            try:
                # Prepare data for boxplot
                box_data = []
                labels = []
                
                for season in season_order:
                    season_data = self.data[self.data['season'] == season][var_col].dropna()
                    if len(season_data) > 0:
                        box_data.append(season_data)
                        labels.append(f"{season}\n(n={len(season_data)})")
                
                if box_data:
                    bp = ax.boxplot(box_data, labels=labels, patch_artist=True)
                    
                    # Color the boxes
                    for patch, color in zip(bp['boxes'], colors[:len(box_data)]):
                        patch.set_facecolor(color)
                    
                    ax.set_ylabel(f'{var_name}')
                    ax.set_title(f'Seasonal Variability - {var_name}')
                    ax.grid(True, alpha=0.3)
                
                print(f"  - {var_name}: Seasonal boxplot created")
                
            except Exception as e:
                print(f"Error creating boxplot for {var_name}: {e}")
                ax.text(0.5, 0.5, f'Error plotting\n{var_name}', 
                       transform=ax.transAxes, ha='center', va='center')
        
        # Remove extra subplots
        for i in range(len(analysis_vars), len(axes)):
            fig.delaxes(axes[i])
        
        plt.tight_layout()
        plt.savefig(self.seasonal_path / 'seasonal_boxplots_debug.png', 
                   dpi=300, bbox_inches='tight')
        plt.show()
        print("Seasonal boxplots saved!")
    
    def _plot_diurnal_patterns_debug(self, analysis_vars):
        """Plot diurnal patterns with debug info"""
        print("\nCreating diurnal patterns plot...")
        
        n_vars = len(analysis_vars)
        if n_vars == 0:
            return
        
        fig, axes = plt.subplots(1, min(3, n_vars), figsize=(6*min(3, n_vars), 6))
        
        if n_vars == 1:
            axes = [axes]
        elif n_vars == 2:
            axes = list(axes)
        
        for i, (var_name, var_col) in enumerate(list(analysis_vars.items())[:3]):
            ax = axes[i] if n_vars > 1 else axes[0]
            
            try:
                # Calculate hourly statistics
                hourly_stats = self.data.groupby('hour')[var_col].agg([
                    'mean', 'std', 'median', 'count'
                ]).reset_index()
                
                print(f"  - {var_name}: {len(hourly_stats)} hours of data")
                
                # Plot mean with confidence interval
                ax.fill_between(hourly_stats['hour'], 
                               hourly_stats['mean'] - hourly_stats['std'],
                               hourly_stats['mean'] + hourly_stats['std'],
                               alpha=0.3, label='Mean ± Std')
                
                ax.plot(hourly_stats['hour'], hourly_stats['mean'], 
                       'b-', linewidth=3, marker='o', markersize=4, 
                       label='Mean')
                
                ax.plot(hourly_stats['hour'], hourly_stats['median'], 
                       'r--', linewidth=2, marker='s', markersize=4, 
                       label='Median')
                
                ax.set_xlabel('Hour of Day')
                ax.set_ylabel(f'{var_name}')
                ax.set_title(f'Diurnal Pattern - {var_name}')
                ax.legend()
                ax.grid(True, alpha=0.3)
                ax.set_xticks(range(0, 24, 3))
                
            except Exception as e:
                print(f"Error creating diurnal plot for {var_name}: {e}")
                ax.text(0.5, 0.5, f'Error plotting\n{var_name}', 
                       transform=ax.transAxes, ha='center', va='center')
        
        plt.tight_layout()
        plt.savefig(self.seasonal_path / 'diurnal_patterns_debug.png', 
                   dpi=300, bbox_inches='tight')
        plt.show()
        print("Diurnal patterns plot saved!")
    
    def generate_data_summary(self):
        """Generate comprehensive data summary"""
        print("\n=== GENERATING DATA SUMMARY ===")
        
        if self.data is None:
            print("No data available for summary!")
            return
        
        summary = []
        summary.append("DATA SUMMARY REPORT")
        summary.append("=" * 50)
        summary.append("")
        
        # Basic info
        summary.append(f"Dataset Shape: {self.data.shape}")
        summary.append(f"Date Range: {self.data['datetime'].min()} to {self.data['datetime'].max()}")
        summary.append(f"Total Records: {len(self.data)}")
        summary.append("")
        
        # Variable summary
        summary.append("AVAILABLE VARIABLES:")
        summary.append("-" * 30)
        for var_type, variables in self.identified_vars.items():
            summary.append(f"{var_type.upper()}:")
            if variables:
                for var in variables:
                    summary.append(f"  - {var}")
            else:
                summary.append("  - None found")
            summary.append("")
        
        # Basic statistics for key variables
        key_vars = []
        for var_list in self.identified_vars.values():
            if var_list:
                key_vars.extend(var_list[:2])  # Take first 2 from each category
        
        if key_vars:
            summary.append("BASIC STATISTICS:")
            summary.append("-" * 30)
            for var in key_vars[:5]:  # Limit to 5 variables
                if var in self.data.columns and self.data[var].notna().any():
                    try:
                        stats = self.data[var].describe()
                        summary.append(f"{var}:")
                        summary.append(f"  Mean: {stats.loc['mean']:.2f}")
                        summary.append(f"  Std:  {stats.loc['std']:.2f}")
                        summary.append(f"  Min:  {stats.loc['min']:.2f}")
                        summary.append(f"  Max:  {stats.loc['max']:.2f}")
                    except Exception as e:
                        print(f"Error computing stats for {var}: {e}")
                        summary.append(f"{var}:")
                        summary.append("  Error in computation")
                else:
                    summary.append(f"{var}:")
                    summary.append("  No valid data")
                summary.append("")
        
        # Save summary
        with open(self.seasonal_path / 'data_summary.txt', 'w') as f:
            f.write('\n'.join(summary))
        
        print("Data summary saved!")
        for line in summary:
            print(line)

def main():
    """Main debug function"""
    print("STARTING DEBUG SEASONAL ANALYSIS")
    print("=" * 60)
    
    # Define paths
    data_path = "/Users/xiaxin/work/WindForecast_Project/01_Data/processed/imputed_data"
    results_path = "/Users/xiaxin/work/WindForecast_Project/03_Results"
    
    # Initialize debug analyzer
    analyzer = DebugSeasonalAnalyzer(data_path, results_path)
    
    try:
        # Step 1: Debug data loading
        if not analyzer.debug_data_loading():
            print("Failed to load data. Exiting.")
            return
        
        # Step 2: Identify variables
        analyzer.identify_variables()
        
        # Step 3: Prepare datetime
        if not analyzer.prepare_datetime():
            print("Failed to prepare datetime. Exiting.")
            return
        
        # Step 4: Create sample analysis
        analyzer.create_sample_analysis()
        
        # Step 5: Generate summary
        analyzer.generate_data_summary()
        
        print("\n" + "=" * 60)
        print("DEBUG ANALYSIS COMPLETED SUCCESSFULLY!")
        print(f"Results saved to: {analyzer.seasonal_path}")
        
    except Exception as e:
        print(f"Error during debug analysis: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()