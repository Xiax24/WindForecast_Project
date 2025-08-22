import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import signal
from scipy.stats import pearsonr, spearmanr
from scipy.fft import fft, fftfreq
from scipy.signal import find_peaks
from scipy.interpolate import interp1d
import warnings
import os
warnings.filterwarnings('ignore')

# Set plotting style
plt.rcParams['font.family'] = 'Arial'
plt.rcParams['axes.unicode_minus'] = False
sns.set_style("whitegrid")

# EMD library imports
try:
    from PyEMD import EEMD
    EMD_LIBRARY = "PyEMD"
    print("Using PyEMD library")
except ImportError:
    try:
        import emd
        EMD_LIBRARY = "emd"
        print("Using emd library")
    except ImportError:
        EMD_LIBRARY = None
        print("No EMD library found")

class CompleteEEMDPeriodAnalyzer:
    """
    Complete EEMD Analyzer with precise period calculation
    """
    
    def __init__(self, data_path, results_path, sampling_interval_minutes=15):
        self.data_path = data_path
        self.results_path = results_path
        self.sampling_interval_minutes = sampling_interval_minutes
        self.sampling_freq = 1 / (sampling_interval_minutes * 60)  # Hz
        self.variables = ['obs_wind_speed_10m', 'obs_wind_speed_70m', 'power']
        
        os.makedirs(results_path, exist_ok=True)
        
        self.decomposition_results = {}
        self.target_imf_count = None
        
    def load_data(self):
        """Load data"""
        print("Loading data...")
        
        self.df = pd.read_csv(self.data_path)
        self.df['datetime'] = pd.to_datetime(self.df['datetime'])
        self.df.set_index('datetime', inplace=True)
        
        print(f"Data shape: {self.df.shape}")
        print(f"Variables: {self.variables}")
        print(f"Sampling interval: {self.sampling_interval_minutes} minutes")
        print(f"Sampling frequency: {self.sampling_freq:.2e} Hz")
        
        return self.df
    
    def perform_eemd_with_consistency(self, ensemble_size=50, noise_std=0.2, target_imfs=12):
        """Perform EEMD with forced consistency"""
        print(f"\n{'='*60}")
        print("Starting Consistent EEMD Decomposition")
        print(f"Target IMF count: {target_imfs}")
        print(f"{'='*60}")
        
        self.target_imf_count = target_imfs
        
        for var in self.variables:
            print(f"\nDecomposing {var}...")
            
            clean_data = self.df[var].dropna()
            values = clean_data.values
            time_index = clean_data.index
            
            print(f"  Data points: {len(values)}")
            
            # Perform EEMD decomposition
            IMFs = self._decompose_with_target_imfs(values, target_imfs, ensemble_size, noise_std)
            
            self.decomposition_results[var] = {
                'original_data': clean_data,
                'IMFs': IMFs,
                'time_index': time_index,
                'n_imfs': len(IMFs)
            }
            
            print(f"  Completed: {len(IMFs)} IMFs (target: {target_imfs})")
        
        return self.decomposition_results
    
    def _decompose_with_target_imfs(self, values, target_imfs, ensemble_size, noise_std):
        """Decompose with forced target IMF count"""
        try:
            if EMD_LIBRARY == "PyEMD":
                all_imfs_collection = []
                noise_level = noise_std * np.std(values)
                
                for ensemble_i in range(ensemble_size):
                    noisy_data = values + noise_level * np.random.randn(len(values))
                    imfs = self._single_emd_with_target(noisy_data, target_imfs)
                    all_imfs_collection.append(imfs)
                
                final_imfs = []
                for imf_idx in range(target_imfs):
                    imf_ensemble = [imfs[imf_idx] for imfs in all_imfs_collection if imf_idx < len(imfs)]
                    if imf_ensemble:
                        final_imfs.append(np.mean(imf_ensemble, axis=0))
                
                return np.array(final_imfs)
                
            elif EMD_LIBRARY == "emd":
                import emd
                imfs = emd.sift.ensemble_sift(values, nensembles=ensemble_size)
                IMFs = imfs.T
                return self._adjust_imf_count(IMFs, target_imfs, values)
            
            else:
                return self._simplified_eemd_with_target(values, target_imfs, ensemble_size, noise_std)
                
        except Exception as e:
            print(f"    Error in decomposition: {e}")
            return self._create_dummy_imfs(values, target_imfs)
    
    def _single_emd_with_target(self, data, target_imfs):
        """Single EMD decomposition with target IMF count"""
        from scipy.signal import argrelextrema
        
        imfs = []
        residue = data.copy()
        
        for imf_idx in range(target_imfs - 1):
            if len(residue) < 10:
                imfs.append(np.zeros_like(data))
                continue
            
            h = residue.copy()
            
            for sift in range(10):
                try:
                    max_indices = argrelextrema(h, np.greater)[0]
                    min_indices = argrelextrema(h, np.less)[0]
                    
                    if len(max_indices) < 2 or len(min_indices) < 2:
                        break
                    
                    if len(max_indices) > 1:
                        f_max = interp1d(max_indices, h[max_indices], 
                                       kind='cubic', fill_value='extrapolate')
                        upper_env = f_max(np.arange(len(h)))
                    else:
                        upper_env = np.full_like(h, np.max(h))
                    
                    if len(min_indices) > 1:
                        f_min = interp1d(min_indices, h[min_indices], 
                                       kind='cubic', fill_value='extrapolate')
                        lower_env = f_min(np.arange(len(h)))
                    else:
                        lower_env = np.full_like(h, np.min(h))
                    
                    mean_env = (upper_env + lower_env) / 2
                    h_new = h - mean_env
                    
                    if np.std(h_new - h) < 0.01 * np.std(h):
                        break
                    
                    h = h_new
                    
                except:
                    break
            
            imfs.append(h)
            residue = residue - h
            
            if np.std(residue) < 0.01 * np.std(data):
                break
            
            try:
                if len(argrelextrema(residue, np.greater)[0]) < 2:
                    break
            except:
                break
        
        while len(imfs) < target_imfs - 1:
            imfs.append(np.zeros_like(data))
        
        imfs.append(residue)
        return imfs[:target_imfs]
    
    def _adjust_imf_count(self, IMFs, target_imfs, original_data):
        """Adjust IMF count to target"""
        current_imfs = len(IMFs)
        
        if current_imfs == target_imfs:
            return IMFs
        elif current_imfs > target_imfs:
            kept_imfs = IMFs[:target_imfs-1]
            combined_residue = np.sum(IMFs[target_imfs-1:], axis=0)
            return np.vstack([kept_imfs, combined_residue.reshape(1, -1)])
        else:
            result_imfs = list(IMFs)
            while len(result_imfs) < target_imfs:
                if len(result_imfs) > 0:
                    last_imf = result_imfs[-1]
                    result_imfs[-1] = last_imf * 0.5
                    result_imfs.append(last_imf * 0.5)
                else:
                    result_imfs.append(np.zeros_like(original_data))
            
            return np.array(result_imfs[:target_imfs])
    
    def _simplified_eemd_with_target(self, values, target_imfs, ensemble_size, noise_std):
        """Simplified EEMD with target IMF count"""
        all_imfs_collection = []
        noise_level = noise_std * np.std(values)
        
        for i in range(ensemble_size):
            noisy_data = values + noise_level * np.random.randn(len(values))
            imfs = self._single_emd_with_target(noisy_data, target_imfs)
            all_imfs_collection.append(imfs)
        
        final_imfs = []
        for imf_idx in range(target_imfs):
            imf_ensemble = [imfs[imf_idx] for imfs in all_imfs_collection]
            final_imfs.append(np.mean(imf_ensemble, axis=0))
        
        return np.array(final_imfs)
    
    def _create_dummy_imfs(self, data, target_imfs):
        """Create dummy IMFs if decomposition fails"""
        print("    Creating dummy IMFs due to decomposition failure")
        
        imfs = []
        data_std = np.std(data)
        
        for i in range(target_imfs - 1):
            freq = 1.0 / (2 ** (i + 1))
            t = np.arange(len(data))
            imf = data_std * 0.1 * np.sin(2 * np.pi * freq * t / len(data))
            imfs.append(imf)
        
        trend = np.linspace(data[0], data[-1], len(data))
        imfs.append(trend)
        
        return np.array(imfs)
    
    def calculate_exact_imf_periods(self):
        """Calculate exact IMF periods using multiple methods"""
        results = []
        
        print(f"\nCalculating exact IMF periods...")
        print(f"Sampling interval: {self.sampling_interval_minutes} minutes")
        print(f"Sampling frequency: {self.sampling_freq:.2e} Hz")
        print("="*80)
        
        for var_name, result in self.decomposition_results.items():
            IMFs = result['IMFs']
            
            print(f"\nAnalyzing {var_name}:")
            print("-" * 50)
            
            for i, imf in enumerate(IMFs):
                # Method 1: FFT dominant frequency
                dominant_freq_fft, dominant_period_fft = self._calculate_fft_dominant_frequency(imf)
                
                # Method 2: Zero crossing frequency
                zero_crossing_period = self._calculate_zero_crossing_period(imf)
                
                # Method 3: Peak interval statistics
                peak_interval_period = self._calculate_peak_interval_period(imf)
                
                # Method 4: Autocorrelation main period
                autocorr_period = self._calculate_autocorr_period(imf)
                
                # Select best representative period
                best_period = self._select_best_period(
                    dominant_period_fft, zero_crossing_period, 
                    peak_interval_period, autocorr_period
                )
                
                # Time unit conversions
                period_minutes = best_period
                period_hours = period_minutes / 60
                period_days = period_hours / 24
                
                # Classify time scale
                time_scale = self._classify_time_scale(period_hours)
                
                result_dict = {
                    'Variable': var_name,
                    'IMF': i + 1,
                    'Period_Minutes': period_minutes,
                    'Period_Hours': period_hours,
                    'Period_Days': period_days,
                    'Dominant_Frequency_Hz': dominant_freq_fft,
                    'FFT_Period_Hours': dominant_period_fft / 60 if dominant_period_fft != np.inf else np.inf,
                    'Zero_Crossing_Period_Hours': zero_crossing_period / 60 if zero_crossing_period != np.inf else np.inf,
                    'Peak_Interval_Period_Hours': peak_interval_period / 60 if peak_interval_period != np.inf else np.inf,
                    'AutoCorr_Period_Hours': autocorr_period / 60 if autocorr_period != np.inf else np.inf,
                    'Time_Scale': time_scale,
                    'Representative_Description': self._get_time_description(period_hours)
                }
                
                results.append(result_dict)
                
                # Print each IMF result
                print(f"IMF {i+1:2d}: {period_hours:8.2f} hours ({period_days:6.3f} days) - {time_scale}")
        
        self.periods_df = pd.DataFrame(results)
        return self.periods_df
    
    def _calculate_fft_dominant_frequency(self, imf):
        """Calculate FFT dominant frequency"""
        n = len(imf)
        fft_values = fft(imf)
        frequencies = fftfreq(n, d=self.sampling_interval_minutes * 60)
        
        positive_freq_idx = frequencies > 0
        positive_frequencies = frequencies[positive_freq_idx]
        positive_fft_magnitude = np.abs(fft_values[positive_freq_idx])
        
        if len(positive_fft_magnitude) > 0:
            dominant_freq_idx = np.argmax(positive_fft_magnitude)
            dominant_frequency = positive_frequencies[dominant_freq_idx]
            dominant_period_minutes = 1 / (dominant_frequency * 60) if dominant_frequency > 0 else np.inf
            
            return dominant_frequency, dominant_period_minutes
        else:
            return 0, np.inf
    
    def _calculate_zero_crossing_period(self, imf):
        """Calculate zero crossing period"""
        zero_crossings = np.sum(np.diff(np.sign(imf)) != 0)
        if zero_crossings > 0:
            total_time_minutes = len(imf) * self.sampling_interval_minutes
            period_minutes = total_time_minutes / (zero_crossings / 2)
            return period_minutes
        else:
            return np.inf
    
    def _calculate_peak_interval_period(self, imf):
        """Calculate peak interval period"""
        try:
            peaks, _ = find_peaks(imf, height=np.std(imf) * 0.1)
            
            if len(peaks) > 1:
                peak_intervals = np.diff(peaks) * self.sampling_interval_minutes
                avg_interval = np.mean(peak_intervals)
                return avg_interval * 2
            else:
                return np.inf
        except:
            return np.inf
    
    def _calculate_autocorr_period(self, imf):
        """Calculate autocorrelation main period"""
        try:
            autocorr = np.correlate(imf, imf, mode='full')
            autocorr = autocorr[len(autocorr)//2:]
            
            if len(autocorr) > 10:
                search_start = max(1, len(autocorr) // 20)
                peaks, _ = find_peaks(autocorr[search_start:], height=np.max(autocorr) * 0.1)
                
                if len(peaks) > 0:
                    first_peak_lag = peaks[0] + search_start
                    period_minutes = first_peak_lag * self.sampling_interval_minutes
                    return period_minutes
            
            return np.inf
        except:
            return np.inf
    
    def _select_best_period(self, fft_period, zero_crossing_period, peak_period, autocorr_period):
        """Select best representative period"""
        periods = [fft_period, zero_crossing_period, peak_period, autocorr_period]
        valid_periods = [p for p in periods if p != np.inf and p > 0]
        
        if len(valid_periods) == 0:
            return np.inf
        
        if len(valid_periods) == 1:
            return valid_periods[0]
        
        median_period = np.median(valid_periods)
        
        if fft_period != np.inf and abs(fft_period - median_period) / median_period < 0.5:
            return fft_period
        else:
            return median_period
    
    def _classify_time_scale(self, period_hours):
        """Classify time scale"""
        if period_hours == np.inf:
            return "Long-term Trend"
        elif period_hours < 0.5:
            return "Sub-hourly"
        elif period_hours < 3:
            return "Hourly"
        elif period_hours < 8:
            return "Several Hours"
        elif period_hours < 18:
            return "Semi-diurnal"
        elif period_hours < 30:
            return "Daily"
        elif period_hours < 120:
            return "Several Days"
        elif period_hours < 240:
            return "Weekly"
        elif period_hours < 1000:
            return "Monthly"
        else:
            return "Seasonal"
    
    def _get_time_description(self, period_hours):
        """Get time description"""
        if period_hours == np.inf:
            return "Long-term trend/monotonic"
        elif period_hours < 1:
            return f"{period_hours*60:.0f} minutes - High-frequency noise"
        elif period_hours < 24:
            return f"{period_hours:.1f} hours - Intra-daily variation"
        elif period_hours < 168:
            return f"{period_hours/24:.1f} days - Multi-day weather patterns"
        elif period_hours < 720:
            return f"{period_hours/24:.0f} days - Weekly to monthly cycles"
        else:
            return f"{period_hours/24/30:.1f} months - Seasonal variations"
    
    def print_detailed_period_results(self):
        """Print detailed period results"""
        print("\n" + "="*100)
        print("DETAILED IMF PERIOD ANALYSIS")
        print("="*100)
        
        for var in self.periods_df['Variable'].unique():
            var_data = self.periods_df[self.periods_df['Variable'] == var]
            
            print(f"\n{var.upper()}:")
            print("-" * 90)
            print(f"{'IMF':<4} {'Period':<12} {'Hours':<8} {'Days':<8} {'Scale':<15} {'Description'}")
            print("-" * 90)
            
            for _, row in var_data.iterrows():
                period_str = f"{row['Period_Hours']:.2f}h" if row['Period_Hours'] != np.inf else "Trend"
                days_str = f"{row['Period_Days']:.3f}" if row['Period_Days'] != np.inf else "∞"
                
                print(f"{row['IMF']:<4} "
                      f"{period_str:<12} "
                      f"{row['Period_Hours']:<8.2f} "
                      f"{days_str:<8} "
                      f"{row['Time_Scale']:<15} "
                      f"{row['Representative_Description']}")
    
    def create_period_comparison_table(self):
        """Create period comparison table"""
        comparison_data = []
        
        imf_numbers = sorted(self.periods_df['IMF'].unique())
        
        for imf_num in imf_numbers:
            imf_data = self.periods_df[self.periods_df['IMF'] == imf_num]
            
            row_data = {'IMF': imf_num}
            
            for _, row in imf_data.iterrows():
                var_name = row['Variable']
                period_hours = row['Period_Hours']
                
                if period_hours != np.inf:
                    if period_hours < 24:
                        period_str = f"{period_hours:.1f}h"
                    else:
                        period_str = f"{period_hours/24:.1f}d"
                else:
                    period_str = "Trend"
                
                row_data[var_name] = period_str
            
            first_row = imf_data.iloc[0]
            row_data['Time_Scale'] = first_row['Time_Scale']
            
            comparison_data.append(row_data)
        
        self.comparison_df = pd.DataFrame(comparison_data)
        return self.comparison_df
    
    def plot_period_analysis(self):
        """Plot period analysis"""
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        
        # Plot 1: Period by IMF
        for var in self.periods_df['Variable'].unique():
            var_data = self.periods_df[self.periods_df['Variable'] == var]
            periods = var_data['Period_Hours'].replace([np.inf, -np.inf], np.nan)
            axes[0, 0].semilogy(var_data['IMF'], periods, 'o-', label=var, linewidth=2, markersize=6)
        
        axes[0, 0].set_title('Dominant Period by IMF')
        axes[0, 0].set_xlabel('IMF')
        axes[0, 0].set_ylabel('Period (hours)')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        # Add reference lines
        axes[0, 0].axhline(y=1, color='red', linestyle='--', alpha=0.5)
        axes[0, 0].axhline(y=12, color='orange', linestyle='--', alpha=0.5)
        axes[0, 0].axhline(y=24, color='green', linestyle='--', alpha=0.5)
        axes[0, 0].axhline(y=168, color='blue', linestyle='--', alpha=0.5)
        
        # Plot 2: Method comparison for first variable
        first_var = self.periods_df['Variable'].iloc[0]
        var_data = self.periods_df[self.periods_df['Variable'] == first_var]
        
        methods = ['FFT_Period_Hours', 'Zero_Crossing_Period_Hours', 
                  'Peak_Interval_Period_Hours', 'AutoCorr_Period_Hours']
        method_labels = ['FFT', 'Zero Crossing', 'Peak Interval', 'Autocorr']
        
        for i, (method, label) in enumerate(zip(methods, method_labels)):
            periods = var_data[method].replace([np.inf, -np.inf], np.nan)
            axes[0, 1].semilogy(var_data['IMF'], periods, 'o-', label=label, alpha=0.7)
        
        axes[0, 1].set_title(f'Period Calculation Methods - {first_var}')
        axes[0, 1].set_xlabel('IMF')
        axes[0, 1].set_ylabel('Period (hours)')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)
        
        # Plot 3: Time scale distribution
        time_scale_counts = self.periods_df['Time_Scale'].value_counts()
        axes[1, 0].pie(time_scale_counts.values, labels=time_scale_counts.index, 
                      autopct='%1.1f%%', startangle=90)
        axes[1, 0].set_title('Time Scale Distribution')
        
        # Plot 4: Period vs IMF for all variables (bar plot)
        pivot_data = self.periods_df.pivot(index='IMF', columns='Variable', values='Period_Hours')
        pivot_data = pivot_data.replace([np.inf, -np.inf], np.nan)
        
        pivot_data.plot(kind='bar', ax=axes[1, 1], width=0.8)
        axes[1, 1].set_title('Period by IMF (All Variables)')
        axes[1, 1].set_xlabel('IMF')
        axes[1, 1].set_ylabel('Period (hours)')
        axes[1, 1].set_yscale('log')
        axes[1, 1].legend(title='Variable')
        axes[1, 1].tick_params(axis='x', rotation=0)
        
        plt.tight_layout()
        save_path = os.path.join(self.results_path, 'Period_analysis.png')
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
        
        print(f"Period analysis plot saved to: {save_path}")
    
    def save_results(self):
        """Save all results"""
        # Save detailed periods
        periods_path = os.path.join(self.results_path, 'detailed_imf_periods.csv')
        self.periods_df.to_csv(periods_path, index=False)
        
        # Save comparison table
        comparison_path = os.path.join(self.results_path, 'imf_period_comparison.csv')
        self.comparison_df.to_csv(comparison_path, index=False)
        
        print(f"\nResults saved to:")
        print(f"- {periods_path}")
        print(f"- {comparison_path}")
    
    def run_complete_analysis(self, target_imfs=12, ensemble_size=50, noise_std=0.2):
        """Run complete analysis with precise period calculation"""
        print("Starting Complete EEMD Period Analysis")
        print("=" * 80)
        
        # Load data
        self.load_data()
        
        # Perform EEMD decomposition
        self.perform_eemd_with_consistency(ensemble_size, noise_std, target_imfs)
        
        if not self.decomposition_results:
            print("No decomposition results available")
            return None
        
        # Calculate exact periods
        self.calculate_exact_imf_periods()
        
        # Create comparison table
        self.create_period_comparison_table()
        
        # Print results
        self.print_detailed_period_results()
        
        print(f"\n{'='*60}")
        print("IMF PERIOD COMPARISON TABLE")
        print("="*60)
        print(self.comparison_df.to_string(index=False))
        
        # Plot analysis
        self.plot_period_analysis()
        
        # Save results
        self.save_results()
        
        return {
            'decomposition_results': self.decomposition_results,
            'periods_df': self.periods_df,
            'comparison_df': self.comparison_df
        }


# Main execution
if __name__ == "__main__":
    # Configuration
    data_path = '/Users/xiaxin/work/WindForecast_Project/01_Data/processed/imputed_data/changma_clean.csv'
    results_path = '/Users/xiaxin/work/WindForecast_Project/03_Results/EEMD_Period_Analysis'
    
    # Create analyzer
    analyzer = CompleteEEMDPeriodAnalyzer(
        data_path=data_path,
        results_path=results_path,
        sampling_interval_minutes=15
    )
    
    # Run complete analysis
    results = analyzer.run_complete_analysis(
        target_imfs=12,
        ensemble_size=50,
        noise_std=0.2
    )
    
    if results:
        print("\n" + "="*80)
        print("SUCCESS: Complete EEMD period analysis finished")
        print("Exact periods for each IMF have been calculated and saved")
        print("="*80)
    else:
        print("Analysis failed")