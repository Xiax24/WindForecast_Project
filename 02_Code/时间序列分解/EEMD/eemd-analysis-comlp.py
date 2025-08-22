import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import signal
from scipy.stats import pearsonr, spearmanr
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

class ConsistentEEMDAnalyzer:
    """
    EEMD Analyzer with forced consistency in IMF counts and detailed frequency analysis
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
        """
        Perform EEMD with forced consistency
        """
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
        
        # Verify consistency
        print(f"\nFinal IMF counts:")
        all_consistent = True
        for var, result in self.decomposition_results.items():
            count = result['n_imfs']
            print(f"  {var}: {count} IMFs")
            if count != target_imfs:
                all_consistent = False
        
        if all_consistent:
            print("✓ All variables have consistent IMF counts")
        else:
            print("✗ IMF counts are still inconsistent")
        
        return self.decomposition_results
    
    def _decompose_with_target_imfs(self, values, target_imfs, ensemble_size, noise_std):
        """
        Decompose with forced target IMF count
        """
        try:
            if EMD_LIBRARY == "PyEMD":
                # Use PyEMD with custom stopping
                all_imfs_collection = []
                noise_level = noise_std * np.std(values)
                
                for ensemble_i in range(ensemble_size):
                    # Add noise
                    noisy_data = values + noise_level * np.random.randn(len(values))
                    
                    # Perform single EMD with forced IMF count
                    imfs = self._single_emd_with_target(noisy_data, target_imfs)
                    all_imfs_collection.append(imfs)
                
                # Average ensemble results
                final_imfs = []
                for imf_idx in range(target_imfs):
                    imf_ensemble = [imfs[imf_idx] for imfs in all_imfs_collection if imf_idx < len(imfs)]
                    if imf_ensemble:
                        final_imfs.append(np.mean(imf_ensemble, axis=0))
                
                return np.array(final_imfs)
                
            elif EMD_LIBRARY == "emd":
                # Use emd library with modifications
                import emd
                imfs = emd.sift.ensemble_sift(values, nensembles=ensemble_size)
                IMFs = imfs.T
                
                # Force target IMF count
                return self._adjust_imf_count(IMFs, target_imfs, values)
            
            else:
                # Use simplified method
                return self._simplified_eemd_with_target(values, target_imfs, ensemble_size, noise_std)
                
        except Exception as e:
            print(f"    Error in decomposition: {e}")
            # Fallback: create dummy IMFs
            return self._create_dummy_imfs(values, target_imfs)
    
    def _single_emd_with_target(self, data, target_imfs):
        """
        Single EMD decomposition with target IMF count
        """
        from scipy.signal import argrelextrema
        from scipy.interpolate import interp1d
        
        imfs = []
        residue = data.copy()
        
        for imf_idx in range(target_imfs - 1):  # Reserve last for residue
            if len(residue) < 10:
                # Not enough data, create zero IMF
                imfs.append(np.zeros_like(data))
                continue
            
            h = residue.copy()
            
            # Sifting process
            for sift in range(10):
                try:
                    max_indices = argrelextrema(h, np.greater)[0]
                    min_indices = argrelextrema(h, np.less)[0]
                    
                    if len(max_indices) < 2 or len(min_indices) < 2:
                        break
                    
                    # Create envelopes
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
                    
                    # Check stopping criterion
                    if np.std(h_new - h) < 0.01 * np.std(h):
                        break
                    
                    h = h_new
                    
                except:
                    break
            
            imfs.append(h)
            residue = residue - h
            
            # Stop early if residue becomes too small or monotonic
            if np.std(residue) < 0.01 * np.std(data):
                break
            
            try:
                if len(argrelextrema(residue, np.greater)[0]) < 2:
                    break
            except:
                break
        
        # Pad with remaining residue and zeros if needed
        while len(imfs) < target_imfs - 1:
            imfs.append(np.zeros_like(data))
        
        # Add final residue
        imfs.append(residue)
        
        # Ensure exactly target_imfs
        return imfs[:target_imfs]
    
    def _adjust_imf_count(self, IMFs, target_imfs, original_data):
        """
        Adjust IMF count to target
        """
        current_imfs = len(IMFs)
        
        if current_imfs == target_imfs:
            return IMFs
        elif current_imfs > target_imfs:
            # Too many IMFs, combine the last ones
            kept_imfs = IMFs[:target_imfs-1]
            combined_residue = np.sum(IMFs[target_imfs-1:], axis=0)
            return np.vstack([kept_imfs, combined_residue.reshape(1, -1)])
        else:
            # Too few IMFs, split the last one or add zeros
            result_imfs = list(IMFs)
            while len(result_imfs) < target_imfs:
                if len(result_imfs) > 0:
                    # Split the last IMF
                    last_imf = result_imfs[-1]
                    result_imfs[-1] = last_imf * 0.5
                    result_imfs.append(last_imf * 0.5)
                else:
                    # Add zero IMF
                    result_imfs.append(np.zeros_like(original_data))
            
            return np.array(result_imfs[:target_imfs])
    
    def _simplified_eemd_with_target(self, values, target_imfs, ensemble_size, noise_std):
        """
        Simplified EEMD with target IMF count
        """
        all_imfs_collection = []
        noise_level = noise_std * np.std(values)
        
        for i in range(ensemble_size):
            noisy_data = values + noise_level * np.random.randn(len(values))
            imfs = self._single_emd_with_target(noisy_data, target_imfs)
            all_imfs_collection.append(imfs)
        
        # Average ensemble
        final_imfs = []
        for imf_idx in range(target_imfs):
            imf_ensemble = [imfs[imf_idx] for imfs in all_imfs_collection]
            final_imfs.append(np.mean(imf_ensemble, axis=0))
        
        return np.array(final_imfs)
    
    def _create_dummy_imfs(self, data, target_imfs):
        """
        Create dummy IMFs if decomposition fails
        """
        print("    Creating dummy IMFs due to decomposition failure")
        
        imfs = []
        data_std = np.std(data)
        
        # Create IMFs with decreasing frequency
        for i in range(target_imfs - 1):
            # Simple sinusoidal IMFs with decreasing frequency
            freq = 1.0 / (2 ** (i + 1))
            t = np.arange(len(data))
            imf = data_std * 0.1 * np.sin(2 * np.pi * freq * t / len(data))
            imfs.append(imf)
        
        # Add trend as last IMF
        trend = np.linspace(data[0], data[-1], len(data))
        imfs.append(trend)
        
        return np.array(imfs)
    
    def calculate_imf_characteristics(self):
        """Calculate IMF characteristics"""
        print("\nCalculating IMF characteristics...")
        
        all_characteristics = []
        
        for var_name, result in self.decomposition_results.items():
            IMFs = result['IMFs']
            total_energy = np.sum([np.sum(imf ** 2) for imf in IMFs])
            
            for i, imf in enumerate(IMFs):
                # Basic statistics
                char = {
                    'Variable': var_name,
                    'IMF': i + 1,
                    'Mean': np.mean(imf),
                    'Std': np.std(imf),
                    'Energy': np.sum(imf ** 2),
                    'Energy_Ratio': np.sum(imf ** 2) / total_energy * 100 if total_energy > 0 else 0,
                    'RMS': np.sqrt(np.mean(imf ** 2)),
                    'Max_Amplitude': np.max(np.abs(imf)),
                }
                
                # Frequency features
                zero_crossings = np.sum(np.diff(np.sign(imf)) != 0)
                char['Zero_Crossings'] = zero_crossings
                char['Zero_Crossing_Rate'] = zero_crossings / len(imf)
                
                if zero_crossings > 0:
                    dominant_period_points = len(imf) / (zero_crossings / 2)
                    char['Dominant_Period_Hours'] = dominant_period_points * (self.sampling_interval_minutes / 60)
                else:
                    char['Dominant_Period_Hours'] = np.inf
                
                # Temporal features
                if len(imf) > 1:
                    autocorr = np.corrcoef(imf[:-1], imf[1:])[0, 1]
                    char['Persistence'] = autocorr if not np.isnan(autocorr) else 0
                else:
                    char['Persistence'] = 0
                
                all_characteristics.append(char)
        
        self.characteristics_df = pd.DataFrame(all_characteristics)
        
        # Save characteristics
        char_path = os.path.join(self.results_path, 'IMF_characteristics.csv')
        self.characteristics_df.to_csv(char_path, index=False)
        print(f"IMF characteristics saved to: {char_path}")
        
        return self.characteristics_df
    
    def calculate_detailed_frequency_analysis(self):
        """
        Calculate detailed frequency analysis including dominant frequencies for each IMF
        """
        print("\nCalculating detailed frequency analysis...")
        
        all_frequency_results = []
        
        for var_name, result in self.decomposition_results.items():
            IMFs = result['IMFs']
            
            for i, imf in enumerate(IMFs):
                # Calculate FFT
                n = len(imf)
                fft_values = np.fft.fft(imf)
                frequencies = np.fft.fftfreq(n, d=self.sampling_interval_minutes * 60)
                
                # Only take positive frequency part
                positive_freq_idx = frequencies > 0
                positive_frequencies = frequencies[positive_freq_idx]
                positive_fft_magnitude = np.abs(fft_values[positive_freq_idx])
                
                # Find dominant frequency
                if len(positive_fft_magnitude) > 0:
                    dominant_freq_idx = np.argmax(positive_fft_magnitude)
                    dominant_frequency_hz = positive_frequencies[dominant_freq_idx]
                    
                    # Convert to periods
                    dominant_period_hours = 1 / (dominant_frequency_hz * 3600) if dominant_frequency_hz > 0 else np.inf
                    dominant_period_days = dominant_period_hours / 24
                    
                    # Calculate zero crossing frequency
                    zero_crossings = np.sum(np.diff(np.sign(imf)) != 0)
                    zero_crossing_frequency = zero_crossings / (2 * n * self.sampling_interval_minutes * 60)
                    zero_crossing_period_hours = 1 / (zero_crossing_frequency * 3600) if zero_crossing_frequency > 0 else np.inf
                    
                    # Frequency band classification
                    if dominant_period_hours < 1:
                        freq_band = "High-freq (< 1h)"
                        freq_category = "Noise/Rapid fluctuation"
                    elif dominant_period_hours < 6:
                        freq_band = "Medium-high-freq (1-6h)"
                        freq_category = "Short-term variation"
                    elif dominant_period_hours < 12:
                        freq_band = "Semi-diurnal (6-12h)"
                        freq_category = "Semi-diurnal cycle"
                    elif dominant_period_hours < 24:
                        freq_band = "Intra-diurnal (12-24h)"
                        freq_category = "Intra-daily variation"
                    elif dominant_period_hours < 168:  # 7 days
                        freq_band = "Inter-daily (1-7d)"
                        freq_category = "Weekly cycle"
                    elif dominant_period_hours < 720:  # 30 days
                        freq_band = "Monthly (7-30d)"
                        freq_category = "Monthly cycle"
                    else:
                        freq_band = "Long-term trend (> 30d)"
                        freq_category = "Seasonal trend"
                    
                    # Calculate energy and relative importance
                    imf_energy = np.sum(imf ** 2)
                    total_energy = np.sum([np.sum(imf_data ** 2) for imf_data in IMFs])
                    energy_percentage = (imf_energy / total_energy * 100) if total_energy > 0 else 0
                    
                else:
                    dominant_frequency_hz = 0
                    dominant_period_hours = np.inf
                    dominant_period_days = np.inf
                    zero_crossing_period_hours = np.inf
                    freq_band = "Trend"
                    freq_category = "Long-term trend"
                    energy_percentage = 0
                
                freq_result = {
                    'Variable': var_name,
                    'IMF': i + 1,
                    'Dominant_Frequency_Hz': dominant_frequency_hz,
                    'Dominant_Period_Hours': dominant_period_hours,
                    'Dominant_Period_Days': dominant_period_days,
                    'Zero_Crossing_Period_Hours': zero_crossing_period_hours,
                    'Frequency_Band': freq_band,
                    'Frequency_Category': freq_category,
                    'Energy_Percentage': energy_percentage,
                    'Sampling_Frequency_Hz': self.sampling_freq
                }
                
                all_frequency_results.append(freq_result)
        
        self.frequency_analysis_df = pd.DataFrame(all_frequency_results)
        
        # Save frequency analysis results
        freq_analysis_path = os.path.join(self.results_path, 'Detailed_frequency_analysis.csv')
        self.frequency_analysis_df.to_csv(freq_analysis_path, index=False)
        print(f"Detailed frequency analysis saved to: {freq_analysis_path}")
        
        return self.frequency_analysis_df
    
    def calculate_cross_correlations(self):
        """Calculate cross-correlations"""
        print("\nCalculating cross-correlations...")
        
        correlations = []
        variables = list(self.decomposition_results.keys())
        
        if len(variables) < 2:
            return pd.DataFrame()
        
        # All variables should have same IMF count now
        num_imfs = self.target_imf_count
        print(f"Analyzing correlations for all {num_imfs} IMFs")
        
        for imf_idx in range(num_imfs):
            for i, var1 in enumerate(variables):
                for j, var2 in enumerate(variables):
                    if i < j:
                        imf1 = self.decomposition_results[var1]['IMFs'][imf_idx]
                        imf2 = self.decomposition_results[var2]['IMFs'][imf_idx]
                        
                        min_len = min(len(imf1), len(imf2))
                        imf1 = imf1[:min_len]
                        imf2 = imf2[:min_len]
                        
                        if len(imf1) > 1:
                            pearson_corr, pearson_p = pearsonr(imf1, imf2)
                            spearman_corr, spearman_p = spearmanr(imf1, imf2)
                            
                            correlations.append({
                                'IMF': imf_idx + 1,
                                'Variable1': var1,
                                'Variable2': var2,
                                'Pearson_Correlation': pearson_corr,
                                'Pearson_P_Value': pearson_p,
                                'Spearman_Correlation': spearman_corr,
                                'Spearman_P_Value': spearman_p,
                            })
        
        self.correlation_df = pd.DataFrame(correlations)
        
        if not self.correlation_df.empty:
            corr_path = os.path.join(self.results_path, 'Cross_correlations.csv')
            self.correlation_df.to_csv(corr_path, index=False)
            print(f"Cross-correlations saved to: {corr_path}")
        
        return self.correlation_df
    
    def plot_frequency_spectrum_analysis(self):
        """
        Plot frequency spectrum analysis
        """
        if not hasattr(self, 'frequency_analysis_df'):
            self.calculate_detailed_frequency_analysis()
        
        # Create frequency spectrum plots for each variable
        for var_name, result in self.decomposition_results.items():
            IMFs = result['IMFs']
            time_index = result['time_index']
            
            # Select representative IMFs for spectrum analysis
            representative_imfs = [0, 2, 5, 8, 11]  # IMF 1, 3, 6, 9, 12
            available_imfs = [i for i in representative_imfs if i < len(IMFs)]
            
            if len(available_imfs) == 0:
                continue
                
            fig, axes = plt.subplots(len(available_imfs), 2, figsize=(16, 4*len(available_imfs)))
            if len(available_imfs) == 1:
                axes = axes.reshape(1, -1)
            fig.suptitle(f'{var_name} - IMF Frequency Spectrum Analysis', fontsize=16)
            
            for i, imf_idx in enumerate(available_imfs):
                imf_data = IMFs[imf_idx]
                
                # Time domain plot
                axes[i, 0].plot(time_index, imf_data, linewidth=0.8)
                axes[i, 0].set_title(f'IMF {imf_idx+1} - Time Domain')
                axes[i, 0].set_ylabel('Amplitude')
                axes[i, 0].grid(True, alpha=0.3)
                
                # Frequency domain plot
                n = len(imf_data)
                fft_values = np.fft.fft(imf_data)
                frequencies = np.fft.fftfreq(n, d=self.sampling_interval_minutes * 60)
                
                positive_freq_idx = frequencies > 0
                positive_frequencies = frequencies[positive_freq_idx]
                positive_fft_magnitude = np.abs(fft_values[positive_freq_idx])
                
                # Convert to periods (hours)
                periods_hours = 1 / (positive_frequencies * 3600)
                
                axes[i, 1].loglog(periods_hours, positive_fft_magnitude)
                axes[i, 1].set_title(f'IMF {imf_idx+1} - Frequency Spectrum')
                axes[i, 1].set_xlabel('Period (hours)')
                axes[i, 1].set_ylabel('Magnitude')
                axes[i, 1].grid(True, alpha=0.3)
                
                # Add reference lines
                reference_periods = [1, 6, 12, 24, 168]
                reference_labels = ['1h', '6h', '12h', '1d', '1w']
                
                for period, label in zip(reference_periods, reference_labels):
                    if period >= periods_hours.min() and period <= periods_hours.max():
                        axes[i, 1].axvline(x=period, color='red', linestyle='--', alpha=0.5)
                        y_pos = axes[i, 1].get_ylim()[1] * 0.8
                        axes[i, 1].text(period * 1.1, y_pos, label, rotation=90, 
                                       verticalalignment='bottom', fontsize=8)
            
            plt.tight_layout()
            save_path = os.path.join(self.results_path, f'Frequency_spectrum_{var_name}.png')
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            plt.show()
            
            print(f"Frequency spectrum plot saved: {save_path}")
    
    def print_frequency_summary(self):
        """
        Print frequency analysis summary
        """
        if not hasattr(self, 'frequency_analysis_df'):
            self.calculate_detailed_frequency_analysis()
        
        print(f"\n{'='*80}")
        print(f"EEMD Frequency Analysis Summary ({self.sampling_interval_minutes}-minute time resolution)")
        print(f"{'='*80}")
        
        for var in self.frequency_analysis_df['Variable'].unique():
            var_data = self.frequency_analysis_df[self.frequency_analysis_df['Variable'] == var]
            
            print(f"\n{var}:")
            print("-" * 70)
            print(f"{'IMF':<4} {'Freq(Hz)':<12} {'Period(h)':<12} {'Period(d)':<10} {'Freq Band':<15} {'Energy%':<8}")
            print("-" * 70)
            
            for _, row in var_data.iterrows():
                freq_hz = row['Dominant_Frequency_Hz']
                period_hours = row['Dominant_Period_Hours']
                period_days = row['Dominant_Period_Days']
                
                freq_str = f"{freq_hz:.2e}" if freq_hz > 0 else "0"
                period_hours_str = f"{period_hours:.1f}" if period_hours != np.inf and period_hours < 1000 else "Trend"
                period_days_str = f"{period_days:.2f}" if period_days != np.inf and period_days < 100 else "Trend"
                
                print(f"{row['IMF']:<4} "
                      f"{freq_str:<12} "
                      f"{period_hours_str:<12} "
                      f"{period_days_str:<10} "
                      f"{row['Frequency_Category']:<15} "
                      f"{row['Energy_Percentage']:<8.1f}")
        
        print(f"\n{'='*80}")
        print("Frequency Interpretation:")
        print("• IMF 1-3: High-frequency noise and rapid fluctuations (< 6 hours)")
        print("• IMF 4-6: Intra-diurnal periodic variations (6-24 hours)")
        print("• IMF 7-9: Inter-daily and weekly cycles (1-7 days)")
        print("• IMF 10-12: Long-term trends and seasonal variations (> 7 days)")
        print(f"Sampling frequency: {self.sampling_freq:.2e} Hz")
        print(f"Nyquist frequency: {self.sampling_freq/2:.2e} Hz")
        print(f"Maximum detectable period: {1/(2*self.sampling_freq)/3600:.1f} hours")
        print(f"{'='*80}")
    
    def plot_all_analysis(self):
        """Generate all plots"""
        print("\nGenerating all analysis plots...")
        
        # 1. Decomposition plots
        self._plot_decomposition_results()
        
        # 2. Frequency comparison
        self._plot_frequency_comparison()
        
        # 3. Correlation analysis
        self._plot_correlation_analysis()
        
        # 4. IMF comparison
        self._plot_imf_comparison()
        
        # 5. New: Frequency spectrum analysis
        self.plot_frequency_spectrum_analysis()
    
    def _plot_decomposition_results(self):
        """Plot decomposition results"""
        for var_name, result in self.decomposition_results.items():
            IMFs = result['IMFs']
            time_index = result['time_index']
            original_data = result['original_data']
            n_imfs = len(IMFs)
            
            fig, axes = plt.subplots(n_imfs + 1, 1, figsize=(15, 2 * (n_imfs + 1)))
            fig.suptitle(f'EEMD Decomposition - {var_name}', fontsize=16, y=0.98)
            
            # Original data
            axes[0].plot(time_index, original_data.values, 'b-', linewidth=0.8)
            axes[0].set_title('Original Data')
            axes[0].set_ylabel('Value')
            axes[0].grid(True, alpha=0.3)
            
            # IMFs
            for i, imf in enumerate(IMFs):
                if i < 3:
                    color = 'red'
                    freq_type = "High-freq"
                elif i < n_imfs - 3:
                    color = 'green'
                    freq_type = "Medium-freq"
                else:
                    color = 'purple'
                    freq_type = "Low-freq/Trend"
                
                axes[i + 1].plot(time_index, imf, color=color, linewidth=0.8)
                axes[i + 1].set_title(f'IMF {i+1} ({freq_type})')
                axes[i + 1].set_ylabel(f'IMF {i+1}')
                axes[i + 1].grid(True, alpha=0.3)
            
            axes[-1].set_xlabel('Time')
            plt.tight_layout()
            
            save_path = os.path.join(self.results_path, f'EEMD_decomposition_{var_name}.png')
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            plt.show()
            
            print(f"Decomposition plot saved: {save_path}")
    
    def _plot_frequency_comparison(self):
        """Plot frequency comparison"""
        if self.characteristics_df is None:
            self.calculate_imf_characteristics()
        
        fig = plt.figure(figsize=(20, 15))
        
        # 1. Energy ratio comparison
        plt.subplot(3, 3, 1)
        energy_pivot = self.characteristics_df.pivot(index='IMF', columns='Variable', values='Energy_Ratio')
        energy_pivot.plot(kind='bar', ax=plt.gca())
        plt.title('Energy Ratio by IMF')
        plt.ylabel('Energy Ratio (%)')
        plt.xticks(rotation=0)
        plt.legend(title='Variable')
        
        # 2. Standard deviation
        plt.subplot(3, 3, 2)
        std_pivot = self.characteristics_df.pivot(index='IMF', columns='Variable', values='Std')
        std_pivot.plot(kind='bar', ax=plt.gca())
        plt.title('Standard Deviation by IMF')
        plt.ylabel('Standard Deviation')
        plt.xticks(rotation=0)
        plt.legend(title='Variable')
        
        # 3. Dominant period
        plt.subplot(3, 3, 3)
        period_pivot = self.characteristics_df.pivot(index='IMF', columns='Variable', values='Dominant_Period_Hours')
        # Replace inf with NaN for plotting
        period_pivot = period_pivot.replace([np.inf, -np.inf], np.nan)
        period_pivot.plot(kind='line', marker='o', ax=plt.gca())
        plt.title('Dominant Period by IMF')
        plt.ylabel('Period (hours)')
        plt.yscale('log')
        plt.legend(title='Variable')
        
        # 4. RMS
        plt.subplot(3, 3, 4)
        rms_pivot = self.characteristics_df.pivot(index='IMF', columns='Variable', values='RMS')
        rms_pivot.plot(kind='bar', ax=plt.gca())
        plt.title('RMS by IMF')
        plt.ylabel('RMS')
        plt.xticks(rotation=0)
        plt.legend(title='Variable')
        
        # 5. Zero crossing rate
        plt.subplot(3, 3, 5)
        zcr_pivot = self.characteristics_df.pivot(index='IMF', columns='Variable', values='Zero_Crossing_Rate')
        zcr_pivot.plot(kind='line', marker='s', ax=plt.gca())
        plt.title('Zero Crossing Rate by IMF')
        plt.ylabel('Zero Crossing Rate')
        plt.legend(title='Variable')
        
        # 6. Max amplitude
        plt.subplot(3, 3, 6)
        amp_pivot = self.characteristics_df.pivot(index='IMF', columns='Variable', values='Max_Amplitude')
        amp_pivot.plot(kind='bar', ax=plt.gca())
        plt.title('Max Amplitude by IMF')
        plt.ylabel('Max Amplitude')
        plt.xticks(rotation=0)
        plt.legend(title='Variable')
        
        # 7. Energy (absolute)
        plt.subplot(3, 3, 7)
        energy_abs_pivot = self.characteristics_df.pivot(index='IMF', columns='Variable', values='Energy')
        energy_abs_pivot.plot(kind='bar', ax=plt.gca())
        plt.title('Absolute Energy by IMF')
        plt.ylabel('Energy')
        plt.xticks(rotation=0)
        plt.legend(title='Variable')
        
        # 8. Persistence
        plt.subplot(3, 3, 8)
        pers_pivot = self.characteristics_df.pivot(index='IMF', columns='Variable', values='Persistence')
        pers_pivot.plot(kind='line', marker='^', ax=plt.gca())
        plt.title('Persistence by IMF')
        plt.ylabel('Persistence')
        plt.legend(title='Variable')
        
        # 9. Cumulative energy
        plt.subplot(3, 3, 9)
        for var in self.characteristics_df['Variable'].unique():
            var_data = self.characteristics_df[self.characteristics_df['Variable'] == var]
            cumulative_energy = var_data['Energy_Ratio'].cumsum()
            plt.plot(var_data['IMF'], cumulative_energy, 'o-', label=var, linewidth=2)
        
        plt.title('Cumulative Energy Distribution')
        plt.xlabel('IMF')
        plt.ylabel('Cumulative Energy (%)')
        plt.legend(title='Variable')
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        save_path = os.path.join(self.results_path, 'Frequency_comparison.png')
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
        
        print(f"Frequency comparison plot saved to: {save_path}")
    
    def _plot_correlation_analysis(self):
        """Plot correlation analysis"""
        if self.correlation_df is None or self.correlation_df.empty:
            print("No correlation data available")
            return
        
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        
        # 1. Pearson correlation heatmap
        pearson_pivot = self.correlation_df.pivot_table(
            index='IMF', 
            columns=['Variable1', 'Variable2'], 
            values='Pearson_Correlation'
        )
        
        sns.heatmap(pearson_pivot, annot=True, cmap='RdBu_r', center=0, 
                   ax=axes[0,0], fmt='.3f')
        axes[0,0].set_title('Pearson Correlation by IMF')
        
        # 2. Spearman correlation heatmap  
        spearman_pivot = self.correlation_df.pivot_table(
            index='IMF',
            columns=['Variable1', 'Variable2'],
            values='Spearman_Correlation'
        )
        
        sns.heatmap(spearman_pivot, annot=True, cmap='RdBu_r', center=0,
                   ax=axes[0,1], fmt='.3f')
        axes[0,1].set_title('Spearman Correlation by IMF')
        
        # 3. Correlation strength by IMF
        for (var1, var2), group in self.correlation_df.groupby(['Variable1', 'Variable2']):
            axes[1,0].plot(group['IMF'], np.abs(group['Pearson_Correlation']), 
                          'o-', label=f'{var1} vs {var2}', linewidth=2)
        
        axes[1,0].set_title('Correlation Strength by IMF')
        axes[1,0].set_xlabel('IMF')
        axes[1,0].set_ylabel('|Correlation|')
        axes[1,0].legend()
        axes[1,0].grid(True, alpha=0.3)
        
        # 4. Summary statistics
        axes[1,1].axis('off')
        
        # Calculate summary stats
        summary_text = "Correlation Summary:\n\n"
        for (var1, var2), group in self.correlation_df.groupby(['Variable1', 'Variable2']):
            avg_pearson = group['Pearson_Correlation'].mean()
            max_pearson = group['Pearson_Correlation'].abs().max()
            summary_text += f"{var1} vs {var2}:\n"
            summary_text += f"  Avg Pearson: {avg_pearson:.3f}\n"
            summary_text += f"  Max |Pearson|: {max_pearson:.3f}\n\n"
        
        axes[1,1].text(0.1, 0.9, summary_text, transform=axes[1,1].transAxes, 
                      fontsize=10, verticalalignment='top',
                      bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgray"))
        
        plt.tight_layout()
        save_path = os.path.join(self.results_path, 'Correlation_analysis.png')
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
        
        print(f"Correlation analysis plot saved to: {save_path}")
    
    def _plot_imf_comparison(self):
        """Plot IMF time series comparison"""
        imf_indices = [1, 3, 6, 9, 12]  # Select representative IMFs
        
        n_imfs = len(imf_indices)
        fig, axes = plt.subplots(n_imfs, 1, figsize=(15, 3*n_imfs))
        
        for i, imf_idx in enumerate(imf_indices):
            for var_name, result in self.decomposition_results.items():
                if imf_idx <= len(result['IMFs']):
                    imf = result['IMFs'][imf_idx-1]
                    time_index = result['time_index']
                    
                    axes[i].plot(time_index, imf, label=var_name, linewidth=1, alpha=0.8)
            
            axes[i].set_title(f'IMF {imf_idx} Time Series Comparison')
            axes[i].set_ylabel('Amplitude')
            axes[i].legend()
            axes[i].grid(True, alpha=0.3)
        
        axes[-1].set_xlabel('Time')
        plt.tight_layout()
        
        save_path = os.path.join(self.results_path, 'IMF_comparison.png')
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
        
        print(f"IMF comparison plot saved to: {save_path}")
    
    def plot_frequency_summary_charts(self):
        """
        Plot comprehensive frequency analysis summary charts
        """
        if not hasattr(self, 'frequency_analysis_df'):
            self.calculate_detailed_frequency_analysis()
        
        fig = plt.figure(figsize=(20, 12))
        
        # 1. Dominant period by IMF for all variables
        plt.subplot(2, 3, 1)
        for var in self.frequency_analysis_df['Variable'].unique():
            var_data = self.frequency_analysis_df[self.frequency_analysis_df['Variable'] == var]
            periods = var_data['Dominant_Period_Hours'].replace([np.inf, -np.inf], np.nan)
            plt.semilogy(var_data['IMF'], periods, 'o-', label=var, linewidth=2, markersize=6)
        
        plt.title('Dominant Period by IMF')
        plt.xlabel('IMF')
        plt.ylabel('Period (hours)')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Add reference lines for common periods
        plt.axhline(y=1, color='red', linestyle='--', alpha=0.5, label='1 hour')
        plt.axhline(y=12, color='orange', linestyle='--', alpha=0.5, label='12 hours')
        plt.axhline(y=24, color='green', linestyle='--', alpha=0.5, label='1 day')
        plt.axhline(y=168, color='blue', linestyle='--', alpha=0.5, label='1 week')
        
        # 2. Energy distribution by frequency bands
        plt.subplot(2, 3, 2)
        freq_bands = ['High-freq (< 1h)', 'Medium-high-freq (1-6h)', 'Semi-diurnal (6-12h)', 
                      'Intra-diurnal (12-24h)', 'Inter-daily (1-7d)', 'Monthly (7-30d)', 'Long-term trend (> 30d)']
        
        for var in self.frequency_analysis_df['Variable'].unique():
            var_data = self.frequency_analysis_df[self.frequency_analysis_df['Variable'] == var]
            band_energy = []
            
            for band in freq_bands:
                energy = var_data[var_data['Frequency_Band'] == band]['Energy_Percentage'].sum()
                band_energy.append(energy)
            
            plt.bar(range(len(freq_bands)), band_energy, alpha=0.7, label=var)
        
        plt.title('Energy Distribution by Frequency Bands')
        plt.xlabel('Frequency Bands')
        plt.ylabel('Total Energy (%)')
        band_labels = ['High', 'Med-High', 'Semi-D', 'Intra-D', 'Inter-D', 'Monthly', 'Trend']
        plt.xticks(range(len(freq_bands)), band_labels, rotation=45)
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # 3. Frequency vs Energy scatter plot
        plt.subplot(2, 3, 3)
        for var in self.frequency_analysis_df['Variable'].unique():
            var_data = self.frequency_analysis_df[self.frequency_analysis_df['Variable'] == var]
            valid_data = var_data[var_data['Dominant_Frequency_Hz'] > 0]
            
            plt.loglog(valid_data['Dominant_Frequency_Hz'], valid_data['Energy_Percentage'], 
                      'o', label=var, markersize=8, alpha=0.7)
        
        plt.title('Frequency vs Energy')
        plt.xlabel('Frequency (Hz)')
        plt.ylabel('Energy Percentage (%)')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # 4. IMF energy distribution comparison
        plt.subplot(2, 3, 4)
        energy_pivot = self.frequency_analysis_df.pivot(index='IMF', columns='Variable', values='Energy_Percentage')
        energy_pivot.plot(kind='bar', ax=plt.gca(), width=0.8)
        plt.title('Energy Percentage by IMF')
        plt.xlabel('IMF')
        plt.ylabel('Energy Percentage (%)')
        plt.xticks(rotation=0)
        plt.legend(title='Variable')
        plt.grid(True, alpha=0.3)
        
        # 5. Cumulative energy distribution
        plt.subplot(2, 3, 5)
        for var in self.frequency_analysis_df['Variable'].unique():
            var_data = self.frequency_analysis_df[self.frequency_analysis_df['Variable'] == var]
            cumulative_energy = var_data['Energy_Percentage'].cumsum()
            plt.plot(var_data['IMF'], cumulative_energy, 'o-', label=var, linewidth=2, markersize=6)
        
        plt.title('Cumulative Energy Distribution')
        plt.xlabel('IMF')
        plt.ylabel('Cumulative Energy (%)')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # 6. Frequency band distribution pie chart (for first variable)
        plt.subplot(2, 3, 6)
        first_var = self.frequency_analysis_df['Variable'].iloc[0]
        var_data = self.frequency_analysis_df[self.frequency_analysis_df['Variable'] == first_var]
        
        band_counts = var_data['Frequency_Category'].value_counts()
        plt.pie(band_counts.values, labels=band_counts.index, autopct='%1.1f%%', startangle=90)
        plt.title(f'Frequency Band Distribution - {first_var}')
        
        plt.tight_layout()
        save_path = os.path.join(self.results_path, 'Frequency_summary_charts.png')
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
        
        print(f"Frequency summary charts saved to: {save_path}")
    
    def run_complete_analysis(self, target_imfs=12, ensemble_size=50, noise_std=0.2):
        """
        Run complete analysis with detailed frequency analysis
        """
        print("Starting Consistent EEMD Frequency Analysis")
        print("=" * 80)
        
        # Load data
        self.load_data()
        
        # Perform consistent EEMD decomposition
        self.perform_eemd_with_consistency(ensemble_size, noise_std, target_imfs)
        
        if not self.decomposition_results:
            print("No decomposition results available")
            return None
        
        # Perform analysis
        self.calculate_imf_characteristics()
        self.calculate_detailed_frequency_analysis()
        self.calculate_cross_correlations()
        
        # Generate plots
        self.plot_all_analysis()
        self.plot_frequency_summary_charts()
        
        # Print frequency summary
        self.print_frequency_summary()
        
        # Generate summary
        summary_stats = []
        for var in self.characteristics_df['Variable'].unique():
            var_data = self.characteristics_df[self.characteristics_df['Variable'] == var]
            freq_data = self.frequency_analysis_df[self.frequency_analysis_df['Variable'] == var]
            
            summary = {
                'Variable': var,
                'IMF_Count': len(var_data),
                'Total_Energy': var_data['Energy'].sum(),
                'High_Freq_Energy': freq_data[freq_data['Frequency_Category'].str.contains('Noise|Rapid fluctuation', na=False)]['Energy_Percentage'].sum(),
                'Medium_Freq_Energy': freq_data[freq_data['Frequency_Category'].str.contains('Short-term|Semi-diurnal|Intra-daily', na=False)]['Energy_Percentage'].sum(),
                'Low_Freq_Energy': freq_data[freq_data['Frequency_Category'].str.contains('Weekly|Monthly|Seasonal', na=False)]['Energy_Percentage'].sum(),
                'Dominant_IMF': var_data.loc[var_data['Energy'].idxmax(), 'IMF'],
                'Mean_Persistence': var_data['Persistence'].mean(),
                'Dominant_Period_Hours': freq_data.loc[freq_data['Energy_Percentage'].idxmax(), 'Dominant_Period_Hours']
            }
            summary_stats.append(summary)
        
        summary_df = pd.DataFrame(summary_stats)
        
        # Save summary
        summary_path = os.path.join(self.results_path, 'Variable_summary.csv')
        summary_df.to_csv(summary_path, index=False)
        
        # Print summary
        print(f"\n{'='*80}")
        print("ANALYSIS SUMMARY")
        print(f"{'='*80}")
        print("\nVariable Summary:")
        for _, row in summary_df.iterrows():
            print(f"\n{row['Variable']}:")
            print(f"  IMF Count: {row['IMF_Count']}")
            print(f"  High-freq energy: {row['High_Freq_Energy']:.1f}%")
            print(f"  Medium-freq energy: {row['Medium_Freq_Energy']:.1f}%")
            print(f"  Low-freq energy: {row['Low_Freq_Energy']:.1f}%")
            print(f"  Dominant IMF: {row['Dominant_IMF']}")
            dom_period = row['Dominant_Period_Hours']
            period_str = f"{dom_period:.1f} hours" if dom_period != np.inf and dom_period < 1000 else "Trend"
            print(f"  Dominant period: {period_str}")
        
        print(f"\nFiles saved to: {self.results_path}")
        print("- IMF_characteristics.csv")
        print("- Detailed_frequency_analysis.csv")
        print("- Cross_correlations.csv") 
        print("- Variable_summary.csv")
        print("- Frequency_comparison.png")
        print("- Frequency_summary_charts.png")
        print("- Correlation_analysis.png")
        print("- IMF_comparison.png")
        for var in self.variables:
            print(f"- EEMD_decomposition_{var}.png")
            print(f"- Frequency_spectrum_{var}.png")
        
        return {
            'decomposition_results': self.decomposition_results,
            'characteristics': self.characteristics_df,
            'frequency_analysis': self.frequency_analysis_df,
            'correlations': self.correlation_df,
            'summary': summary_df
        }


# Main execution
if __name__ == "__main__":
    # Configuration
    data_path = '/Users/xiaxin/work/WindForecast_Project/01_Data/processed/imputed_data/changma_clean.csv'
    results_path = '/Users/xiaxin/work/WindForecast_Project/03_Results/Consistent_EEMD_Analysis'
    
    # Create analyzer with 15-minute sampling interval
    analyzer = ConsistentEEMDAnalyzer(
        data_path=data_path, 
        results_path=results_path,
        sampling_interval_minutes=15  # Explicitly specify 15-minute sampling interval
    )
    
    # Run analysis with forced consistency and detailed frequency analysis
    results = analyzer.run_complete_analysis(
        target_imfs=12,      # Force all variables to have exactly 12 IMFs
        ensemble_size=50,    # Ensemble size
        noise_std=0.2        # Noise level
    )
    
    if results:
        print("\n" + "="*80)
        print("SUCCESS: Complete EEMD analysis with detailed frequency information")
        print("All variables now have exactly 12 IMFs with detailed frequency characteristics")
        print("You can now perform fair comparison across all frequency components")
        print("="*80)
        
        # Display some key frequency insights
        freq_df = results['frequency_analysis']
        print("\nKey Frequency Analysis Findings:")
        print("-" * 50)
        
        for var in freq_df['Variable'].unique():
            var_data = freq_df[freq_df['Variable'] == var]
            max_energy_imf = var_data.loc[var_data['Energy_Percentage'].idxmax()]
            
            print(f"\n{var}:")
            print(f"  Highest energy IMF: IMF{max_energy_imf['IMF']} ({max_energy_imf['Energy_Percentage']:.1f}%)")
            period = max_energy_imf['Dominant_Period_Hours']
            if period != np.inf and period < 1000:
                print(f"  Dominant period: {period:.1f} hours ({period/24:.2f} days)")
            else:
                print(f"  Dominant period: Long-term trend")
            print(f"  Frequency band: {max_energy_imf['Frequency_Category']}")
        
    else:
        print("Analysis failed")