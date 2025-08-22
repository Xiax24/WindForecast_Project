import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import signal
from scipy.stats import pearsonr, spearmanr
from scipy.fft import fft, fftfreq
from scipy.signal import find_peaks, argrelextrema
from scipy.interpolate import interp1d
from scipy.ndimage import gaussian_filter1d
import warnings
import os
warnings.filterwarnings('ignore')

# Set plotting style
plt.rcParams['font.family'] = 'Arial'
plt.rcParams['axes.unicode_minus'] = False
sns.set_style("whitegrid")

# Try importing EMD library
EMD_AVAILABLE = False
EMD_LIBRARY = None

print("Checking for EMD libraries...")

try:
    from PyEMD import CEEMDAN
    print("✓ Successfully imported CEEMDAN from PyEMD")
    EMD_AVAILABLE = True
    EMD_LIBRARY = "PyEMD"
except ImportError as e:
    print(f"✗ PyEMD import failed: {e}")
    try:
        import emd
        print("✓ Successfully imported emd library")
        
        # Check available functions
        available_funcs = [attr for attr in dir(emd.sift) if 'sift' in attr.lower()]
        print(f"Available emd.sift functions: {available_funcs}")
        
        EMD_AVAILABLE = True
        EMD_LIBRARY = "emd"
    except ImportError as e2:
        print(f"✗ emd library import failed: {e2}")
        print("→ Will use simplified implementation")
        EMD_AVAILABLE = False

print(f"Final EMD status: Available={EMD_AVAILABLE}, Library={EMD_LIBRARY}")

class CompleteCEEMDANAnalyzer:
    """
    Complete CEEMDAN Analyzer with comprehensive analysis features
    """
    
    def __init__(self, data_path, results_path, sampling_interval_minutes=15):
        self.data_path = data_path
        self.results_path = results_path
        self.sampling_interval_minutes = sampling_interval_minutes
        self.sampling_freq = 1 / (sampling_interval_minutes * 60)
        # 修改变量列表，添加30m和50m风速
        self.variables = ['obs_wind_speed_10m', 'obs_wind_speed_30m', 'obs_wind_speed_50m', 'obs_wind_speed_70m', 'power']
        
        os.makedirs(results_path, exist_ok=True)
        self.decomposition_results = {}
    
    def load_data(self):
        """Load data"""
        print("Loading data...")
        
        self.df = pd.read_csv(self.data_path)
        self.df['datetime'] = pd.to_datetime(self.df['datetime'])
        self.df.set_index('datetime', inplace=True)
        
        print(f"Data shape: {self.df.shape}")
        print(f"Date range: {self.df.index.min()} to {self.df.index.max()}")
        print(f"Variables to analyze: {self.variables}")
        
        # 检查哪些变量实际存在于数据中
        available_vars = [var for var in self.variables if var in self.df.columns]
        missing_vars = [var for var in self.variables if var not in self.df.columns]
        
        if missing_vars:
            print(f"Warning: Missing variables in data: {missing_vars}")
            print(f"Available variables for analysis: {available_vars}")
            self.variables = available_vars
        
        return self.df
    
    def improved_simple_ceemdan(self, data, ensemble_size=50, max_imfs=12, noise_std=0.005):
        """
        Improved simplified CEEMDAN implementation - EXACTLY from working version
        """
        print(f"  Using improved simplified CEEMDAN (ensemble_size={ensemble_size})")
        
        # Set random seed for reproducibility
        np.random.seed(42)
        
        imfs = []
        residue = data.copy().astype(float)
        original_data = data.copy().astype(float)
        
        # Adaptive noise level
        base_noise_std = noise_std * np.std(original_data)
        
        for mode in range(max_imfs):
            print(f"    Processing IMF {mode + 1}...")
            
            if len(residue) < 20 or np.std(residue) < 1e-10:
                print(f"    Stopping: insufficient residue variation")
                break
            
            # CEEMDAN: different noise handling for each mode
            ensemble_imfs = []
            
            for ens in range(ensemble_size):
                # Set seed for each ensemble member
                np.random.seed(42 + mode * ensemble_size + ens)
                
                # CEEMDAN noise strategy: noise added to original for first mode, 
                # to residue for subsequent modes
                if mode == 0:
                    # First mode: add noise to original signal
                    noise = base_noise_std * np.random.randn(len(original_data))
                    noisy_signal = original_data + noise
                else:
                    # Subsequent modes: add mode-specific noise to residue
                    mode_noise_std = base_noise_std / (2 ** (mode - 1))
                    noise = mode_noise_std * np.random.randn(len(residue))
                    noisy_signal = residue + noise
                
                # Extract IMF through sifting
                try:
                    imf = self.extract_imf_sifting(noisy_signal)
                    if len(imf) == len(residue):
                        ensemble_imfs.append(imf)
                except Exception as e:
                    print(f"    Warning: Ensemble {ens} failed: {e}")
                    continue
            
            if len(ensemble_imfs) < ensemble_size // 2:
                print(f"    Warning: Only {len(ensemble_imfs)} successful ensemble members")
                if len(ensemble_imfs) == 0:
                    break
            
            # Average ensemble to get final IMF
            imf = np.mean(ensemble_imfs, axis=0)
            
            # Ensure IMF has reasonable properties
            if np.std(imf) < 1e-10 * np.std(original_data):
                print(f"    Stopping: IMF has negligible variation")
                break
            
            imfs.append(imf)
            residue = residue - imf
            
            # Check stopping criteria
            if mode >= 3:  # After several IMFs
                try:
                    # Check if residue is monotonic (trend-like)
                    max_indices = argrelextrema(residue, np.greater)[0]
                    if len(max_indices) < 3:
                        print(f"    Stopping: residue is monotonic")
                        break
                except:
                    pass
            
            # Energy-based stopping criterion
            residue_energy = np.sum(residue ** 2)
            original_energy = np.sum(original_data ** 2)
            if residue_energy < 0.01 * original_energy:
                print(f"    Stopping: residue energy too low")
                break
        
        # Add final residue as trend
        if len(residue) > 0:
            imfs.append(residue)
        
        print(f"  Completed: {len(imfs)} IMFs extracted")
        return np.array(imfs)
    
    def extract_imf_sifting(self, signal, max_sifts=15):
        """
        Extract single IMF through sifting process - EXACTLY from working version
        """
        h = signal.copy()
        
        for sift in range(max_sifts):
            try:
                # Find local maxima and minima
                max_indices = argrelextrema(h, np.greater)[0]
                min_indices = argrelextrema(h, np.less)[0]
                
                # Need at least 2 extrema of each type
                if len(max_indices) < 2 or len(min_indices) < 2:
                    break
                
                # Create envelopes with error handling
                try:
                    # Upper envelope
                    if len(max_indices) >= 2:
                        f_max = interp1d(max_indices, h[max_indices], 
                                       kind='cubic', fill_value='extrapolate',
                                       bounds_error=False)
                        upper_env = f_max(np.arange(len(h)))
                        
                        # Check for interpolation issues
                        if np.any(np.isnan(upper_env)) or np.any(np.isinf(upper_env)):
                            upper_env = np.full_like(h, np.max(h))
                    else:
                        upper_env = np.full_like(h, np.max(h))
                    
                    # Lower envelope
                    if len(min_indices) >= 2:
                        f_min = interp1d(min_indices, h[min_indices], 
                                       kind='cubic', fill_value='extrapolate',
                                       bounds_error=False)
                        lower_env = f_min(np.arange(len(h)))
                        
                        # Check for interpolation issues
                        if np.any(np.isnan(lower_env)) or np.any(np.isinf(lower_env)):
                            lower_env = np.full_like(h, np.min(h))
                    else:
                        lower_env = np.full_like(h, np.min(h))
                    
                except Exception as e:
                    # Fallback to simple envelopes
                    upper_env = np.full_like(h, np.max(h))
                    lower_env = np.full_like(h, np.min(h))
                
                # Calculate mean envelope
                mean_env = (upper_env + lower_env) / 2
                h_new = h - mean_env
                
                # Check stopping criterion
                if np.std(h_new - h) < 0.005 * np.std(h):
                    break
                
                # Update for next iteration
                h = h_new
                
            except Exception as e:
                print(f"      Sifting iteration {sift} error: {e}")
                break
        
        return h
    
    def perform_ceemdan_decomposition(self, data, variable_name, ensemble_size=100, noise_std=0.005):
        """
        Perform CEEMDAN decomposition - EXACTLY from working version with library support
        """
        print(f"\nPerforming CEEMDAN decomposition for {variable_name}...")
        
        # Clean data
        clean_data = data.dropna()
        
        if len(clean_data) < 100:
            print(f"Error: Insufficient data for {variable_name} ({len(clean_data)} points)")
            return None
        
        values = clean_data.values.astype(float)
        time_index = clean_data.index
        
        print(f"  Data points: {len(values)}")
        print(f"  Data range: {np.min(values):.3f} to {np.max(values):.3f}")
        print(f"  Ensemble size: {ensemble_size}")
        print(f"  Noise std: {noise_std}")
        print(f"  EMD library: {EMD_LIBRARY}")
        
        try:
            if EMD_LIBRARY == "PyEMD":
                print("  Using PyEMD CEEMDAN...")
                ceemdan = CEEMDAN()
                ceemdan.ensemble_size = ensemble_size
                ceemdan.noise_scale = noise_std
                ceemdan.S_number = 4
                
                IMFs = ceemdan(values)
                
            elif EMD_LIBRARY == "emd":
                print("  Using emd library...")
                import emd
                
                # Try complete_ensemble_sift first (CEEMDAN)
                if hasattr(emd.sift, 'complete_ensemble_sift'):
                    try:
                        print("    Trying complete_ensemble_sift (CEEMDAN)...")
                        imfs = emd.sift.complete_ensemble_sift(values, nensembles=ensemble_size)
                        IMFs = imfs.T
                        print("    Successfully used complete_ensemble_sift")
                    except Exception as e:
                        print(f"    complete_ensemble_sift failed: {e}")
                        print("    Falling back to ensemble_sift...")
                        imfs = emd.sift.ensemble_sift(values, nensembles=ensemble_size)
                        IMFs = imfs.T
                else:
                    print("    Using ensemble_sift...")
                    imfs = emd.sift.ensemble_sift(values, nensembles=ensemble_size)
                    IMFs = imfs.T
            
            else:
                print("  Using improved simplified CEEMDAN...")
                IMFs = self.improved_simple_ceemdan(values, ensemble_size, max_imfs=12, noise_std=noise_std)
            
            # Validate results
            if len(IMFs) == 0:
                raise ValueError("No IMFs generated")
            
            # Check reconstruction quality
            reconstructed = np.sum(IMFs, axis=0)
            reconstruction_error = np.mean((values - reconstructed) ** 2)
            
            if reconstruction_error > 0.1 * np.var(values):
                print(f"  Warning: High reconstruction error ({reconstruction_error:.6f})")
            
            result = {
                'original_data': clean_data,
                'IMFs': IMFs,
                'time_index': time_index,
                'n_imfs': len(IMFs),
                'reconstruction_error': reconstruction_error
            }
            
            print(f"  Success: {len(IMFs)} IMFs extracted")
            print(f"  Reconstruction RMSE: {np.sqrt(reconstruction_error):.6f}")
            
            return result
            
        except Exception as e:
            print(f"  Error during decomposition: {e}")
            import traceback
            traceback.print_exc()
            return None
    
    def calculate_cross_correlations(self):
        """Calculate cross-correlations - 修改为包含所有高度的风速"""
        print("\nCalculating cross-correlations...")
        
        correlations = []
        variables = list(self.decomposition_results.keys())
        
        if len(variables) < 2:
            return pd.DataFrame()
        
        # Find the minimum IMF count across all variables
        min_imfs = min([result['n_imfs'] for result in self.decomposition_results.values()])
        print(f"Analyzing correlations for {min_imfs} IMFs")
        
        for imf_idx in range(min_imfs):
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
            corr_path = os.path.join(self.results_path, 'CEEMDAN_Cross_correlations.csv')
            self.correlation_df.to_csv(corr_path, index=False)
            print(f"Cross-correlations saved to: {corr_path}")
        
        return self.correlation_df
    
    def get_color_for_variable_pair(self, var1, var2):
        """
        根据变量对的类型返回相应的颜色
        - 10m风速相关: 蓝色系渐变
        - 其他风速之间: 紫色系渐变  
        - 风速与功率: 橙色系渐变
        """
        # 获取seaborn色板
        blues = sns.color_palette("Purples", 6)  # 蓝色系
        purples = sns.color_palette("Blues", 6)  # 紫色系Blues
        oranges = sns.color_palette("Reds", 6)  # 橙色系
        
        # 判断变量对类型
        if ('wind_speed' in var1 and 'power' in var2) or ('wind_speed' in var2 and 'power' in var1):
            # 风速与功率：橙色系
            if 'obs_wind_speed_10m' in var1 or 'obs_wind_speed_10m' in var2:
                return oranges[5]  # 最深的橙色
            elif 'obs_wind_speed_30m' in var1 or 'obs_wind_speed_30m' in var2:
                return oranges[4]
            elif 'obs_wind_speed_50m' in var1 or 'obs_wind_speed_50m' in var2:
                return oranges[3]
            elif 'obs_wind_speed_70m' in var1 or 'obs_wind_speed_70m' in var2:
                return oranges[2]
            else:
                return oranges[1]
        
        elif 'wind_speed' in var1 and 'wind_speed' in var2:
            # 风速之间的相关性
            if 'obs_wind_speed_10m' in var1 or 'obs_wind_speed_10m' in var2:
                # 包含10m风速：蓝色系
                if ('obs_wind_speed_10m' in var1 and 'obs_wind_speed_30m' in var2) or ('obs_wind_speed_30m' in var1 and 'obs_wind_speed_10m' in var2):
                    return blues[5]  # 10m-30m
                elif ('obs_wind_speed_10m' in var1 and 'obs_wind_speed_50m' in var2) or ('obs_wind_speed_50m' in var1 and 'obs_wind_speed_10m' in var2):
                    return blues[4]  # 10m-50m
                elif ('obs_wind_speed_10m' in var1 and 'obs_wind_speed_70m' in var2) or ('obs_wind_speed_70m' in var1 and 'obs_wind_speed_10m' in var2):
                    return blues[3]  # 10m-70m
                else:
                    return blues[2]
            else:
                # 其他风速之间：紫色系
                if ('obs_wind_speed_30m' in var1 and 'obs_wind_speed_50m' in var2) or ('obs_wind_speed_50m' in var1 and 'obs_wind_speed_30m' in var2):
                    return purples[5]  # 30m-50m
                elif ('obs_wind_speed_30m' in var1 and 'obs_wind_speed_70m' in var2) or ('obs_wind_speed_70m' in var1 and 'obs_wind_speed_30m' in var2):
                    return purples[4]  # 30m-70m
                elif ('obs_wind_speed_50m' in var1 and 'obs_wind_speed_70m' in var2) or ('obs_wind_speed_70m' in var1 and 'obs_wind_speed_50m' in var2):
                    return purples[3]  # 50m-70m
                else:
                    return purples[2]
        
        # 默认颜色
        return 'gray'
    
    def plot_correlation_analysis_only(self):
        """只画相关性分析图，10m相关的用方形标记，其他用圆形标记，调整线条粗细和标记大小"""
        if not hasattr(self, 'correlation_df') or self.correlation_df.empty:
            print("No correlation data available")
            return
        
        # 创建一个大图，只显示相关性强度图
        fig, ax = plt.subplots(1, 1, figsize=(12, 8))
        
        # Correlation strength by IMF - 10m用方形，其他用圆形
        for (var1, var2), group in self.correlation_df.groupby(['Variable1', 'Variable2']):
            # 使用颜色分配函数
            color = self.get_color_for_variable_pair(var1, var2)
            
            # 判断是否包含10m风速
            is_10m_related = 'obs_wind_speed_10m' in var1 or 'obs_wind_speed_10m' in var2
            
            # 设置标记样式 - 调整参数：线条更细，标记更大
            if is_10m_related:
                marker = 's'  # 方形
                markersize = 8  # 从6增加到8
                linewidth = 1.5  # 从2.5减少到1.5
            else:
                marker = 'o'  # 圆形
                markersize = 6  # 从4增加到6
                linewidth = 1.2  # 从2减少到1.2
            
            # 创建更清晰的标签
            if 'wind_speed' in var1 and 'wind_speed' in var2:
                # 提取高度信息
                height1 = var1.split('_')[-1].replace('m', '')
                height2 = var2.split('_')[-1].replace('m', '')
                label = f'Wind {height1}m vs {height2}m'
            elif 'wind_speed' in var1 and 'power' in var2:
                height = var1.split('_')[-1].replace('m', '')
                label = f'Wind {height}m vs Power'
            elif 'wind_speed' in var2 and 'power' in var1:
                height = var2.split('_')[-1].replace('m', '')
                label = f'Wind {height}m vs Power'
            else:
                label = f'{var1} vs {var2}'
            
            ax.plot(group['IMF'], np.abs(group['Pearson_Correlation']), 
                   marker=marker, linestyle='-', label=label, 
                   linewidth=linewidth, markersize=markersize,
                   color=color, markerfacecolor=color, markeredgecolor='white', markeredgewidth=0.5)
        
        ax.set_title('CEEMDAN IMF Correlation Strength Analysis\n(Squares: 10m wind related, Circles: Others)', 
                    fontsize=14, fontweight='bold')
        ax.set_xlabel('IMF Number', fontsize=12)
        ax.set_ylabel('|Pearson Correlation Coefficient|', fontsize=12)
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=10)
        ax.grid(True, alpha=0.3)
        ax.set_ylim(0, 1)  # 设置y轴范围为0-1
        
        # 美化图表
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['left'].set_linewidth(0.5)
        ax.spines['bottom'].set_linewidth(0.5)
        
        plt.tight_layout()
        save_path = os.path.join(self.results_path, 'CEEMDAN_correlation_analysis_only.png')
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
        print(f"Correlation analysis plot saved to: {save_path}")
    
    def run_correlation_only_analysis(self, ensemble_size=100, noise_std=0.005):
        """
        运行只生成相关性分析图的分析
        """
        print("Starting CEEMDAN Analysis - Correlation Plot Only")
        print("Generating correlation analysis plot with custom markers")
        print("=" * 80)
        
        # Load data
        self.load_data()
        
        # Analyze each variable (decomposition only, no plots)
        for var in self.variables:
            print(f"\n{'='*60}")
            print(f"Analyzing variable: {var}")
            print(f"{'='*60}")
            
            if var not in self.df.columns:
                print(f"Warning: {var} not found in data")
                continue
            
            var_data = self.df[var]
            print(f"Data points: {len(var_data)}")
            print(f"Missing values: {var_data.isna().sum()}")
            print(f"Value range: {var_data.min():.3f} to {var_data.max():.3f}")
            
            # Perform decomposition (no plotting)
            result = self.perform_ceemdan_decomposition(var_data, var, ensemble_size, noise_std)
            
            if result is not None:
                self.decomposition_results[var] = result
                print(f"  Decomposition completed for {var}")
            else:
                print(f"Failed to decompose {var}")
        
        if not self.decomposition_results:
            print("No successful decompositions completed")
            return None
        
        # Calculate cross-correlations
        print("\nCalculating cross-correlations...")
        self.calculate_cross_correlations()
        
        # Generate ONLY the correlation analysis plot
        if hasattr(self, 'correlation_df') and not self.correlation_df.empty:
            self.plot_correlation_analysis_only()
        
        print(f"\nCorrelation-only analysis completed!")
        print(f"Generated files:")
        print("- CEEMDAN_correlation_analysis_only.png (10m wind: squares, others: circles)")
        print("- CEEMDAN_Cross_correlations.csv")
        
        return self.decomposition_results


# Main execution
if __name__ == "__main__":
    # Set paths
    data_path = '/Users/xiaxin/work/WindForecast_Project/01_Data/processed/imputed_data/changma_clean.csv'
    results_path = '/Users/xiaxin/work/WindForecast_Project/03_Results/Complete_CEEMDAN_Analysis'
    
    # Create analyzer
    analyzer = CompleteCEEMDANAnalyzer(
        data_path=data_path,
        results_path=results_path,
        sampling_interval_minutes=15
    )
    
    # Run correlation-only analysis
    results = analyzer.run_correlation_only_analysis(
        ensemble_size=100,    # Proven working parameter
        noise_std=0.005       # Proven working parameter
    )
    
    if results:
        print("\n" + "="*80)
        print("SUCCESS: CEEMDAN correlation analysis finished")
        print("Generated correlation plot with optimized styling:")
        print("- Square markers (■): 10m wind related - size 8, linewidth 1.5")
        print("- Circle markers (●): Others - size 6, linewidth 1.2")
        print("- Color scheme maintained: Blue(10m), Purple(other winds), Orange(wind-power)")
        print("="*80)
    else:
        print("Analysis failed")