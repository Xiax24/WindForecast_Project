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

class AdaptiveCEEMDANAnalyzer:
    """
    自适应CEEMDAN分析器 - 每个变量根据自身特性确定最优IMF数量
    """
    
    def __init__(self, data_path, results_path, sampling_interval_minutes=15):
        self.data_path = data_path
        self.results_path = results_path
        self.sampling_interval_minutes = sampling_interval_minutes
        self.sampling_freq = 1 / (sampling_interval_minutes * 60)
        # 包含所有高度的风速
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
    
    def improved_simple_ceemdan_adaptive(self, data, ensemble_size=50, noise_std=0.005):
        """
        自适应CEEMDAN实现 - 根据数据特性动态确定IMF数量
        """
        print(f"  Using adaptive CEEMDAN (ensemble_size={ensemble_size})")
        
        # Set random seed for reproducibility
        np.random.seed(42)
        
        imfs = []
        residue = data.copy().astype(float)
        original_data = data.copy().astype(float)
        
        # Adaptive noise level
        base_noise_std = noise_std * np.std(original_data)
        original_energy = np.sum(original_data ** 2)
        
        # 动态最大IMF数量：基于数据长度
        max_imfs = min(int(np.log2(len(data))), 15)  # 更合理的上限
        print(f"    Maximum IMFs for this data: {max_imfs}")
        
        for mode in range(max_imfs):
            print(f"    Processing IMF {mode + 1}...")
            
            # 更严格的停止条件
            if len(residue) < 20:
                print(f"    Stopping: insufficient data length ({len(residue)})")
                break
                
            if np.std(residue) < 1e-8 * np.std(original_data):  # 更严格的标准偏差阈值
                print(f"    Stopping: negligible residue variation")
                break
            
            # 检查残差的能量
            residue_energy = np.sum(residue ** 2)
            if residue_energy < 0.005 * original_energy:  # 更严格的能量阈值
                print(f"    Stopping: residue energy too low ({residue_energy/original_energy:.6f})")
                break
            
            # 检查极值点数量 - 更严格的条件
            try:
                max_indices = argrelextrema(residue, np.greater)[0]
                min_indices = argrelextrema(residue, np.less)[0]
                total_extrema = len(max_indices) + len(min_indices)
                
                if total_extrema < 6:  # 至少需要6个极值点
                    print(f"    Stopping: insufficient extrema ({total_extrema})")
                    break
                    
                # 检查是否接近单调
                if len(max_indices) <= 2 or len(min_indices) <= 2:
                    print(f"    Stopping: residue becoming monotonic")
                    break
            except:
                print(f"    Stopping: extrema detection failed")
                break
            
            # CEEMDAN ensemble processing
            ensemble_imfs = []
            
            for ens in range(ensemble_size):
                np.random.seed(42 + mode * ensemble_size + ens)
                
                if mode == 0:
                    noise = base_noise_std * np.random.randn(len(original_data))
                    noisy_signal = original_data + noise
                else:
                    mode_noise_std = base_noise_std / (2 ** (mode - 1))
                    noise = mode_noise_std * np.random.randn(len(residue))
                    noisy_signal = residue + noise
                
                try:
                    imf = self.extract_imf_sifting_adaptive(noisy_signal, mode)
                    if len(imf) == len(residue) and np.std(imf) > 1e-10:
                        ensemble_imfs.append(imf)
                except Exception as e:
                    continue
            
            if len(ensemble_imfs) < ensemble_size // 3:  # 至少需要1/3成功
                print(f"    Stopping: insufficient ensemble success ({len(ensemble_imfs)}/{ensemble_size})")
                break
            
            # Average ensemble
            imf = np.mean(ensemble_imfs, axis=0)
            
            # IMF质量检查
            imf_energy = np.sum(imf ** 2)
            if imf_energy < 1e-6 * original_energy:
                print(f"    Stopping: IMF energy too low")
                break
            
            # 检查IMF是否有意义（不是噪声）
            imf_std = np.std(imf)
            if imf_std < 1e-8 * np.std(original_data):
                print(f"    Stopping: IMF variation negligible")
                break
            
            imfs.append(imf)
            residue = residue - imf
            
            # 连续IMF检查：如果连续两个IMF都很相似，可能过度分解了
            if len(imfs) >= 2:
                correlation = np.corrcoef(imfs[-1], imfs[-2])[0, 1]
                if abs(correlation) > 0.95:
                    print(f"    Warning: High correlation with previous IMF ({correlation:.3f})")
                    if len(imfs) >= 4:  # 如果已经有足够的IMF，就停止
                        print(f"    Stopping: potential over-decomposition detected")
                        break
        
        # Add final residue as trend
        if len(residue) > 0:
            final_residue_energy = np.sum(residue ** 2)
            if final_residue_energy > 1e-10 * original_energy:
                imfs.append(residue)
            else:
                print(f"    Final residue energy negligible, not adding as trend")
        
        print(f"  Completed: {len(imfs)} IMFs extracted (adaptive)")
        
        # 质量检查
        if len(imfs) > 0:
            reconstructed = np.sum(imfs, axis=0)
            reconstruction_error = np.mean((original_data - reconstructed) ** 2)
            print(f"  Reconstruction RMSE: {np.sqrt(reconstruction_error):.8f}")
            
            # 能量分布检查
            total_energy = sum(np.sum(imf**2) for imf in imfs)
            for i, imf in enumerate(imfs):
                energy_ratio = np.sum(imf**2) / total_energy
                print(f"    IMF{i+1} energy ratio: {energy_ratio:.4f}")
        
        return np.array(imfs)
    
    def extract_imf_sifting_adaptive(self, signal, mode, max_sifts=20):
        """
        自适应筛选过程 - 根据IMF阶数调整参数
        """
        h = signal.copy()
        
        # 根据IMF阶数调整停止阈值
        if mode <= 2:
            stop_threshold = 0.001  # 前几个IMF要求更严格
        else:
            stop_threshold = 0.005  # 后面的IMF可以放松一些
        
        for sift in range(max_sifts):
            try:
                max_indices = argrelextrema(h, np.greater)[0]
                min_indices = argrelextrema(h, np.less)[0]
                
                # 需要足够的极值点
                if len(max_indices) < 2 or len(min_indices) < 2:
                    break
                
                # 极值点分布检查
                if len(h) > 100:  # 对于长序列，检查极值点分布
                    extrema_spacing = np.diff(np.sort(np.concatenate([max_indices, min_indices])))
                    if np.mean(extrema_spacing) > len(h) / 4:  # 极值点太稀疏
                        break
                
                try:
                    # 包络插值
                    if len(max_indices) >= 2:
                        f_max = interp1d(max_indices, h[max_indices], 
                                       kind='cubic', fill_value='extrapolate',
                                       bounds_error=False)
                        upper_env = f_max(np.arange(len(h)))
                        
                        if np.any(np.isnan(upper_env)) or np.any(np.isinf(upper_env)):
                            upper_env = np.full_like(h, np.max(h))
                    else:
                        upper_env = np.full_like(h, np.max(h))
                    
                    if len(min_indices) >= 2:
                        f_min = interp1d(min_indices, h[min_indices], 
                                       kind='cubic', fill_value='extrapolate',
                                       bounds_error=False)
                        lower_env = f_min(np.arange(len(h)))
                        
                        if np.any(np.isnan(lower_env)) or np.any(np.isinf(lower_env)):
                            lower_env = np.full_like(h, np.min(h))
                    else:
                        lower_env = np.full_like(h, np.min(h))
                    
                except Exception:
                    upper_env = np.full_like(h, np.max(h))
                    lower_env = np.full_like(h, np.min(h))
                
                mean_env = (upper_env + lower_env) / 2
                h_new = h - mean_env
                
                # 自适应停止准则
                if np.std(h_new - h) < stop_threshold * np.std(h):
                    break
                
                h = h_new
                
            except Exception as e:
                break
        
        return h
    
    def estimate_mean_frequency(self, imf):
        """
        估计IMF的平均频率
        """
        try:
            # 通过零交叉点估计频率
            zero_crossings = np.where(np.diff(np.signbit(imf)))[0]
            if len(zero_crossings) > 1:
                mean_period = 2 * np.mean(np.diff(zero_crossings))  # 周期（样本点）
                mean_freq = 1.0 / (mean_period * self.sampling_interval_minutes / 60.0)  # 转换为Hz
                return mean_freq
            else:
                return 0.0
        except:
            return 0.0
    
    def analyze_imf_frequencies(self, imf, imf_index):
        """
        详细分析IMF的频率特性
        """
        try:
            # 1. 零交叉频率
            zero_crossings = np.where(np.diff(np.signbit(imf)))[0]
            if len(zero_crossings) > 1:
                mean_period_samples = 2 * np.mean(np.diff(zero_crossings))
                mean_period_hours = mean_period_samples * self.sampling_interval_minutes / 60.0
                zero_crossing_freq = 1.0 / mean_period_hours  # Hz
            else:
                zero_crossing_freq = 0.0
                mean_period_hours = np.inf
            
            # 2. FFT主频率
            fft_vals = np.abs(fft(imf))
            freqs = fftfreq(len(imf), d=self.sampling_interval_minutes/60.0)  # Hz
            
            # 找到主频率（排除DC分量）
            positive_freqs = freqs[freqs > 0]
            positive_fft = fft_vals[freqs > 0]
            
            if len(positive_fft) > 0:
                dominant_freq_idx = np.argmax(positive_fft)
                dominant_freq = positive_freqs[dominant_freq_idx]
                dominant_period_hours = 1.0 / dominant_freq if dominant_freq > 0 else np.inf
            else:
                dominant_freq = 0.0
                dominant_period_hours = np.inf
            
            # 3. 转换为实际时间尺度
            period_interpretation = self.interpret_period(mean_period_hours)
            
            return {
                'zero_crossing_freq_hz': zero_crossing_freq,
                'dominant_freq_hz': dominant_freq,
                'mean_period_hours': mean_period_hours,
                'dominant_period_hours': dominant_period_hours,
                'period_interpretation': period_interpretation,
                'energy_ratio': np.var(imf)
            }
        except Exception as e:
            return {
                'zero_crossing_freq_hz': 0.0,
                'dominant_freq_hz': 0.0,
                'mean_period_hours': np.inf,
                'dominant_period_hours': np.inf,
                'period_interpretation': 'Unknown',
                'energy_ratio': 0.0
            }
    
    def interpret_period(self, period_hours):
        """
        将周期转换为可理解的时间尺度描述
        """
        if period_hours == np.inf or period_hours <= 0:
            return "Trend/DC"
        elif period_hours < 1:
            minutes = period_hours * 60
            if minutes < 30:
                return f"~{minutes:.1f}min (Sub-hourly)"
            else:
                return f"~{minutes:.0f}min (Sub-hourly)"
        elif period_hours < 6:
            return f"~{period_hours:.1f}h (Hourly scale)"
        elif period_hours < 18:
            return f"~{period_hours:.1f}h (Semi-diurnal)"
        elif 18 <= period_hours <= 30:
            return f"~{period_hours:.1f}h (Diurnal)"
        elif period_hours < 120:  # < 5 days
            days = period_hours / 24
            return f"~{days:.1f}d (Multi-day)"
        elif period_hours < 720:  # < 30 days
            days = period_hours / 24
            return f"~{days:.1f}d (Weekly-Monthly)"
        else:
            days = period_hours / 24
            return f"~{days:.0f}d (Long-term trend)"
    
    def perform_ceemdan_decomposition_adaptive(self, data, variable_name, ensemble_size=100, noise_std=0.005):
        """
        执行自适应CEEMDAN分解
        """
        print(f"\nPerforming adaptive CEEMDAN decomposition for {variable_name}...")
        
        clean_data = data.dropna()
        
        if len(clean_data) < 100:
            print(f"Error: Insufficient data for {variable_name} ({len(clean_data)} points)")
            return None
        
        values = clean_data.values.astype(float)
        time_index = clean_data.index
        
        print(f"  Data points: {len(values)}")
        print(f"  Data range: {np.min(values):.3f} to {np.max(values):.3f}")
        print(f"  Data std: {np.std(values):.6f}")
        
        try:
            if EMD_LIBRARY == "PyEMD":
                print("  Using PyEMD CEEMDAN...")
                ceemdan = CEEMDAN()
                ceemdan.ensemble_size = ensemble_size
                ceemdan.noise_scale = noise_std
                # 让PyEMD自动决定IMF数量
                IMFs = ceemdan(values)
                
            elif EMD_LIBRARY == "emd":
                print("  Using emd library...")
                import emd
                
                if hasattr(emd.sift, 'complete_ensemble_sift'):
                    try:
                        imfs = emd.sift.complete_ensemble_sift(values, nensembles=ensemble_size)
                        IMFs = imfs.T
                    except Exception:
                        imfs = emd.sift.ensemble_sift(values, nensembles=ensemble_size)
                        IMFs = imfs.T
                else:
                    imfs = emd.sift.ensemble_sift(values, nensembles=ensemble_size)
                    IMFs = imfs.T
            
            else:
                print("  Using adaptive simplified CEEMDAN...")
                IMFs = self.improved_simple_ceemdan_adaptive(values, ensemble_size, noise_std)
            
            if len(IMFs) == 0:
                raise ValueError("No IMFs generated")
            
            # 验证重构质量
            reconstructed = np.sum(IMFs, axis=0)
            reconstruction_error = np.mean((values - reconstructed) ** 2)
            relative_error = reconstruction_error / np.var(values)
            
            print(f"  IMFs extracted: {len(IMFs)}")
            print(f"  Reconstruction RMSE: {np.sqrt(reconstruction_error):.8f}")
            print(f"  Relative error: {relative_error:.6f}")
            
            if relative_error > 0.01:  # 1%的相对误差
                print(f"  Warning: High reconstruction error!")
            
            # 分析每个IMF的特性
            print("  IMF frequency analysis:")
            total_var = np.var(values)
            imf_frequency_info = []
            
            for i, imf in enumerate(IMFs):
                var_ratio = np.var(imf) / total_var
                freq_analysis = self.analyze_imf_frequencies(imf, i)
                
                print(f"    IMF{i+1}: var_ratio={var_ratio:.4f}, {freq_analysis['period_interpretation']}")
                print(f"           Period: {freq_analysis['mean_period_hours']:.2f}h, Freq: {freq_analysis['zero_crossing_freq_hz']:.6f}Hz")
                
                # 保存详细信息
                imf_frequency_info.append({
                    'IMF': i+1,
                    'variance_ratio': var_ratio,
                    'mean_period_hours': freq_analysis['mean_period_hours'],
                    'zero_crossing_freq_hz': freq_analysis['zero_crossing_freq_hz'],
                    'dominant_freq_hz': freq_analysis['dominant_freq_hz'],
                    'period_interpretation': freq_analysis['period_interpretation']
                })
            
            result = {
                'original_data': clean_data,
                'IMFs': IMFs,
                'time_index': time_index,
                'n_imfs': len(IMFs),
                'reconstruction_error': reconstruction_error,
                'relative_error': relative_error,
                'frequency_analysis': imf_frequency_info
            }
            
            return result
            
        except Exception as e:
            print(f"  Error during decomposition: {e}")
            import traceback
            traceback.print_exc()
            return None
    
    def calculate_cross_correlations_adaptive(self):
        """
        自适应计算交叉相关性 - 处理不同变量IMF数量不同的情况
        """
        print("\nCalculating adaptive cross-correlations...")
        
        correlations = []
        variables = list(self.decomposition_results.keys())
        
        if len(variables) < 2:
            print("Error: Need at least 2 variables for correlation analysis")
            return pd.DataFrame()
        
        # 显示每个变量的IMF数量
        print("IMF counts per variable:")
        for var in variables:
            n_imfs = self.decomposition_results[var]['n_imfs']
            print(f"  {var}: {n_imfs} IMFs")
        
        # 对每对变量，计算它们共同拥有的IMF的相关性
        for i, var1 in enumerate(variables):
            for j, var2 in enumerate(variables):
                if i < j:  # 避免重复计算
                    n_imfs_var1 = self.decomposition_results[var1]['n_imfs']
                    n_imfs_var2 = self.decomposition_results[var2]['n_imfs']
                    
                    # 取两个变量IMF数量的最小值
                    max_comparable_imfs = min(n_imfs_var1, n_imfs_var2)
                    
                    print(f"\nComparing {var1} ({n_imfs_var1} IMFs) vs {var2} ({n_imfs_var2} IMFs)")
                    print(f"  → Will compare first {max_comparable_imfs} IMFs")
                    
                    # 计算可比较的IMF的相关性
                    for imf_idx in range(max_comparable_imfs):
                        try:
                            imf1 = self.decomposition_results[var1]['IMFs'][imf_idx]
                            imf2 = self.decomposition_results[var2]['IMFs'][imf_idx]
                            
                            # 确保两个IMF长度一致
                            min_len = min(len(imf1), len(imf2))
                            imf1 = imf1[:min_len]
                            imf2 = imf2[:min_len]
                            
                            if len(imf1) > 1 and np.std(imf1) > 1e-10 and np.std(imf2) > 1e-10:
                                pearson_corr, pearson_p = pearsonr(imf1, imf2)
                                spearman_corr, spearman_p = spearmanr(imf1, imf2)
                                
                                correlations.append({
                                    'IMF': imf_idx + 1,
                                    'Variable1': var1,
                                    'Variable2': var2,
                                    'Variable1_Total_IMFs': n_imfs_var1,
                                    'Variable2_Total_IMFs': n_imfs_var2,
                                    'Max_Comparable_IMFs': max_comparable_imfs,
                                    'Pearson_Correlation': pearson_corr,
                                    'Pearson_P_Value': pearson_p,
                                    'Spearman_Correlation': spearman_corr,
                                    'Spearman_P_Value': spearman_p,
                                    'IMF1_Variance': np.var(imf1),
                                    'IMF2_Variance': np.var(imf2),
                                    # 添加频率信息
                                    'IMF1_Period_Hours': self.decomposition_results[var1]['frequency_analysis'][imf_idx]['mean_period_hours'],
                                    'IMF2_Period_Hours': self.decomposition_results[var2]['frequency_analysis'][imf_idx]['mean_period_hours'],
                                    'IMF1_Interpretation': self.decomposition_results[var1]['frequency_analysis'][imf_idx]['period_interpretation'],
                                    'IMF2_Interpretation': self.decomposition_results[var2]['frequency_analysis'][imf_idx]['period_interpretation'],
                                    'Average_Period_Hours': (self.decomposition_results[var1]['frequency_analysis'][imf_idx]['mean_period_hours'] + 
                                                           self.decomposition_results[var2]['frequency_analysis'][imf_idx]['mean_period_hours']) / 2
                                })
                                
                                print(f"    IMF{imf_idx+1}: r={pearson_corr:.3f} (p={pearson_p:.3f})")
                            else:
                                print(f"    IMF{imf_idx+1}: Skipped (insufficient variation)")
                        
                        except Exception as e:
                            print(f"    IMF{imf_idx+1}: Error - {e}")
                            continue
        
        self.correlation_df = pd.DataFrame(correlations)
        
        if not self.correlation_df.empty:
            # 保存结果
            corr_path = os.path.join(self.results_path, 'CEEMDAN_Cross_correlations_adaptive.csv')
            self.correlation_df.to_csv(corr_path, index=False)
            print(f"\nCross-correlations saved to: {corr_path}")
            
            # 打印统计信息
            print("\nCorrelation analysis summary:")
            for (var1, var2), group in self.correlation_df.groupby(['Variable1', 'Variable2']):
                max_imf = group['Max_Comparable_IMFs'].iloc[0]
                actual_correlations = len(group)
                print(f"  {var1} vs {var2}: {actual_correlations}/{max_imf} correlations calculated")
        
        return self.correlation_df
    
    def get_color_for_variable_pair(self, var1, var2):
        """
        根据变量对的类型返回相应的颜色
        """
        # 获取seaborn色板
        blues = sns.color_palette("Purples", 6)  # 蓝色系
        purples = sns.color_palette("Blues", 6)  # 紫色系
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
    
    def plot_correlation_analysis_adaptive(self):
        """
        绘制自适应相关性分析图 - 处理不同长度的IMF序列
        """
        if not hasattr(self, 'correlation_df') or self.correlation_df.empty:
            print("No correlation data available")
            return
        
        # 创建图形 - 增加第三个子图显示频率信息
        fig = plt.figure(figsize=(16, 14))
        ax1 = plt.subplot(3, 1, 1)
        ax2 = plt.subplot(3, 1, 2)
        ax3 = plt.subplot(3, 1, 3)
        
        # 上图：相关性强度随IMF变化
        print("Plotting correlation strength analysis...")
        
        for (var1, var2), group in self.correlation_df.groupby(['Variable1', 'Variable2']):
            # 获取颜色和标记样式
            color = self.get_color_for_variable_pair(var1, var2)
            is_10m_related = 'obs_wind_speed_10m' in var1 or 'obs_wind_speed_10m' in var2
            
            if is_10m_related:
                marker = 's'  # 方形
                markersize = 8
                linewidth = 1.5
                alpha = 0.8
            else:
                marker = 'o'  # 圆形
                markersize = 6
                linewidth = 1.2
                alpha = 0.7
            
            # 创建标签
            if 'wind_speed' in var1 and 'wind_speed' in var2:
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
            
            # 获取总IMF数量信息
            total_imfs_1 = group['Variable1_Total_IMFs'].iloc[0]
            total_imfs_2 = group['Variable2_Total_IMFs'].iloc[0]
            max_comparable = group['Max_Comparable_IMFs'].iloc[0]
            
            # 更新标签包含IMF信息
            label_with_info = f'{label} (max {max_comparable}/{total_imfs_1},{total_imfs_2})'
            
            # 绘制相关性强度
            x_values = group['IMF'].values
            y_values = np.abs(group['Pearson_Correlation'].values)
            
            ax1.plot(x_values, y_values, 
                    marker=marker, linestyle='-', label=label_with_info,
                    linewidth=linewidth, markersize=markersize, alpha=alpha,
                    color=color, markerfacecolor=color, markeredgecolor='white', 
                    markeredgewidth=0.5)
            
            # 在最后一个点添加终止标记
            if len(x_values) > 0:
                last_x, last_y = x_values[-1], y_values[-1]
                ax1.scatter(last_x, last_y, s=markersize*3, marker='x', 
                           color='red', linewidth=2, alpha=0.8, zorder=10)
        
        ax1.set_title('CEEMDAN IMF Correlation Strength Analysis (Adaptive)\n' + 
                     'Red X marks: End of comparable IMFs | Squares: 10m wind related', 
                     fontsize=13, fontweight='bold')
        ax1.set_xlabel('IMF Number', fontsize=11)
        ax1.set_ylabel('|Pearson Correlation Coefficient|', fontsize=11)
        ax1.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=9)
        ax1.grid(True, alpha=0.3)
        ax1.set_ylim(0, 1)
        
        # 下图：显示每个变量对的可比较IMF范围
        print("Plotting IMF availability heatmap...")
        
        # 创建热力图数据
        variable_pairs = []
        max_imf_overall = 0
        
        for (var1, var2), group in self.correlation_df.groupby(['Variable1', 'Variable2']):
            if 'wind_speed' in var1 and 'wind_speed' in var2:
                height1 = var1.split('_')[-1].replace('m', '')
                height2 = var2.split('_')[-1].replace('m', '')
                pair_name = f'{height1}m-{height2}m'
            elif 'wind_speed' in var1 and 'power' in var2:
                height = var1.split('_')[-1].replace('m', '')
                pair_name = f'{height}m-Power'
            elif 'wind_speed' in var2 and 'power' in var1:
                height = var2.split('_')[-1].replace('m', '')
                pair_name = f'{height}m-Power'
            else:
                pair_name = f'{var1}-{var2}'
            
            variable_pairs.append({
                'pair_name': pair_name,
                'max_comparable': group['Max_Comparable_IMFs'].iloc[0],
                'var1_total': group['Variable1_Total_IMFs'].iloc[0],
                'var2_total': group['Variable2_Total_IMFs'].iloc[0],
                'actual_correlations': len(group)
            })
            
            max_imf_overall = max(max_imf_overall, 
                                group['Variable1_Total_IMFs'].iloc[0],
                                group['Variable2_Total_IMFs'].iloc[0])
        
        # 创建热力图矩阵
        heatmap_data = np.zeros((len(variable_pairs), max_imf_overall))
        pair_labels = []
        
        for i, pair_info in enumerate(variable_pairs):
            pair_labels.append(f"{pair_info['pair_name']} ({pair_info['actual_correlations']}/{pair_info['max_comparable']})")
            
            # 填充可比较的IMF区域
            for j in range(pair_info['max_comparable']):
                heatmap_data[i, j] = 2  # 可比较且已计算
            
            # 标记不可比较的区域
            for j in range(pair_info['max_comparable'], max_imf_overall):
                if j < pair_info['var1_total'] or j < pair_info['var2_total']:
                    heatmap_data[i, j] = 1  # 其中一个变量有此IMF但不可比较
                else:
                    heatmap_data[i, j] = 0  # 两个变量都没有此IMF
        
        # 绘制热力图
        im = ax2.imshow(heatmap_data, cmap='RdYlBu_r', aspect='auto', alpha=0.8)
        
        # 设置刻度和标签
        ax2.set_xticks(range(max_imf_overall))
        ax2.set_xticklabels([f'IMF{i+1}' for i in range(max_imf_overall)], rotation=45)
        ax2.set_yticks(range(len(pair_labels)))
        ax2.set_yticklabels(pair_labels, fontsize=9)
        
        ax2.set_title('IMF Availability and Correlation Coverage\n' +
                     'Dark Red: Calculated | Orange: Available but not comparable | Blue: Not available',
                     fontsize=12, fontweight='bold')
        ax2.set_xlabel('IMF Number', fontsize=11)
        ax2.set_ylabel('Variable Pairs (calculated/max_comparable)', fontsize=11)
        
        # 添加颜色条
        cbar = plt.colorbar(im, ax=ax2, shrink=0.6)
        cbar.set_ticks([0, 1, 2])
        cbar.set_ticklabels(['Not Available', 'Available Only', 'Calculated'])
        
        # 第三图：频率-相关性分析
        print("Plotting frequency-correlation analysis...")
        
        # 为每个变量对绘制频率vs相关性
        for (var1, var2), group in self.correlation_df.groupby(['Variable1', 'Variable2']):
            color = self.get_color_for_variable_pair(var1, var2)
            is_10m_related = 'obs_wind_speed_10m' in var1 or 'obs_wind_speed_10m' in var2
            
            if is_10m_related:
                marker = 's'
                markersize = 8
            else:
                marker = 'o'
                markersize = 6
            
            # 创建标签
            if 'wind_speed' in var1 and 'wind_speed' in var2:
                height1 = var1.split('_')[-1].replace('m', '')
                height2 = var2.split('_')[-1].replace('m', '')
                label = f'{height1}m-{height2}m'
            elif 'wind_speed' in var1 and 'power' in var2:
                height = var1.split('_')[-1].replace('m', '')
                label = f'{height}m-Power'
            elif 'wind_speed' in var2 and 'power' in var1:
                height = var2.split('_')[-1].replace('m', '')
                label = f'{height}m-Power'
            else:
                label = f'{var1}-{var2}'
            
            # 使用平均周期作为x轴
            periods = group['Average_Period_Hours'].values
            correlations_abs = np.abs(group['Pearson_Correlation'].values)
            
            # 过滤掉无穷大的周期值
            valid_mask = np.isfinite(periods) & (periods > 0)
            if np.any(valid_mask):
                periods_valid = periods[valid_mask]
                correlations_valid = correlations_abs[valid_mask]
                
                ax3.scatter(periods_valid, correlations_valid, 
                           marker=marker, s=markersize*8, alpha=0.7,
                           color=color, label=label, edgecolors='white', linewidth=0.5)
                
                # 为每个点添加IMF标号
                imf_numbers = group['IMF'].values[valid_mask]
                for i, (period, corr, imf_num) in enumerate(zip(periods_valid, correlations_valid, imf_numbers)):
                    ax3.annotate(f'{imf_num}', (period, corr), 
                               xytext=(2, 2), textcoords='offset points',
                               fontsize=8, alpha=0.8)
        
        ax3.set_xscale('log')
        ax3.set_xlabel('Period (Hours)', fontsize=11)
        ax3.set_ylabel('|Correlation Coefficient|', fontsize=11)
        ax3.set_title('IMF Correlations vs Temporal Scales\n' +
                     'Numbers show IMF index | Squares: 10m wind related',
                     fontsize=12, fontweight='bold')
        ax3.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=9)
        ax3.grid(True, alpha=0.3)
        ax3.set_ylim(0, 1)
        
        # 添加时间尺度参考线
        reference_periods = [
            (0.25, '15min'),
            (1, '1h'),
            (6, '6h'),
            (24, '1d'),
            (24*7, '1w'),
            (24*30, '1m')
        ]
        
        for period, label in reference_periods:
            if period >= ax3.get_xlim()[0] and period <= ax3.get_xlim()[1]:
                ax3.axvline(x=period, color='gray', linestyle='--', alpha=0.5, linewidth=1)
                ax3.text(period, ax3.get_ylim()[1]*0.95, label, 
                        rotation=90, verticalalignment='top', horizontalalignment='right',
                        fontsize=8, alpha=0.7)
        
        # 美化图表
        for ax in [ax1, ax2, ax3]:
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            ax.spines['left'].set_linewidth(0.5)
            ax.spines['bottom'].set_linewidth(0.5)
        
        plt.tight_layout()
        save_path = os.path.join(self.results_path, 'CEEMDAN_correlation_analysis_adaptive.png')
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
        print(f"Adaptive correlation analysis plot saved to: {save_path}")
    
    def run_adaptive_correlation_analysis(self, ensemble_size=100, noise_std=0.005):
        """
        运行自适应相关性分析
        """
        print("Starting Adaptive CEEMDAN Correlation Analysis")
        print("Each variable will have its optimal number of IMFs")
        print("Correlations calculated only for comparable IMF pairs")
        print("=" * 80)
        
        # Load data
        self.load_data()
        
        # Analyze each variable with adaptive decomposition
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
            
            # 使用自适应分解
            result = self.perform_ceemdan_decomposition_adaptive(var_data, var, ensemble_size, noise_std)
            
            if result is not None:
                self.decomposition_results[var] = result
                print(f"  ✓ Decomposition completed: {result['n_imfs']} IMFs")
            else:
                print(f"  ✗ Failed to decompose {var}")
        
        if not self.decomposition_results:
            print("No successful decompositions completed")
            return None
        
        # 显示分解结果摘要（包含频率信息）
        print(f"\n{'='*80}")
        print("Decomposition Summary with Frequency Analysis:")
        print(f"{'='*80}")
        for var, result in self.decomposition_results.items():
            print(f"\n{var}: {result['n_imfs']} IMFs (RMSE: {np.sqrt(result['reconstruction_error']):.6f})")
            print("  IMF Frequency Breakdown:")
            for freq_info in result['frequency_analysis']:
                print(f"    IMF{freq_info['IMF']}: {freq_info['period_interpretation']} "
                      f"(Period: {freq_info['mean_period_hours']:.2f}h, Var: {freq_info['variance_ratio']:.4f})")
        
        print(f"\n{'='*60}")
        print("Frequency Scale Interpretation:")
        print(f"{'='*60}")
        print("• Sub-hourly (<1h): High-frequency fluctuations, turbulence")
        print("• Hourly scale (1-6h): Atmospheric boundary layer processes")
        print("• Semi-diurnal (6-18h): Twice-daily patterns, land-sea breeze")
        print("• Diurnal (18-30h): Daily cycles, solar heating patterns")
        print("• Multi-day (1-5d): Weather system passages")
        print("• Weekly-Monthly (5-30d): Synoptic patterns, seasonal transitions")
        print("• Long-term trend (>30d): Climate variability, seasonal trends")
        
        # 计算自适应交叉相关性
        self.calculate_cross_correlations_adaptive()
        
        # 生成自适应相关性分析图
        if hasattr(self, 'correlation_df') and not self.correlation_df.empty:
            self.plot_correlation_analysis_adaptive()
        
        print(f"\n{'='*80}")
        print("Adaptive CEEMDAN Analysis Completed!")
        print("Generated files:")
        print("- CEEMDAN_correlation_analysis_adaptive.png (3-panel analysis)")
        print("- CEEMDAN_Cross_correlations_adaptive.csv (with frequency info)")
        print("Key features:")
        print("• Each variable has optimal IMF count with frequency analysis")
        print("• Panel 1: Correlation strength vs IMF number")
        print("• Panel 2: IMF availability heatmap")
        print("• Panel 3: Correlation vs temporal scales (NEW!)")
        print("• Detailed frequency interpretation for each IMF")
        print("• Time scale references: 15min, 1h, 6h, 1d, 1w, 1m")
        print(f"{'='*80}")
        
        return self.decomposition_results


# Main execution
if __name__ == "__main__":
    # Set paths - 修改为您的实际路径
    data_path = '/Users/xiaxin/work/WindForecast_Project/01_Data/processed/imputed_data/changma_clean.csv'
    results_path = '/Users/xiaxin/work/WindForecast_Project/03_Results/Adaptive_CEEMDAN_Analysis'
    
    # Create analyzer
    analyzer = AdaptiveCEEMDANAnalyzer(
        data_path=data_path,
        results_path=results_path,
        sampling_interval_minutes=15
    )
    
    # Run adaptive correlation analysis
    print("🚀 Starting Adaptive CEEMDAN Analysis")
    print("📊 This analysis will:")
    print("   • Automatically determine optimal IMF count for each variable")
    print("   • Analyze frequency characteristics of each IMF")
    print("   • Calculate correlations only for comparable IMF pairs")
    print("   • Generate 3-panel visualization:")
    print("     - Panel 1: Correlation strength vs IMF number")
    print("     - Panel 2: IMF availability heatmap")
    print("     - Panel 3: Correlation vs temporal scales (NEW!)")
    print("   • Provide detailed frequency interpretation")
    print()
    
    results = analyzer.run_adaptive_correlation_analysis(
        ensemble_size=100,    # Proven working parameter
        noise_std=0.005       # Proven working parameter
    )
    
    if results:
        print("\n" + "🎉" + "="*78 + "🎉")
        print("SUCCESS: Adaptive CEEMDAN correlation analysis completed!")
        print()
        print("📈 Key improvements:")
        print("   ✓ Each variable has its natural IMF count (no forced decomposition)")
        print("   ✓ Detailed frequency analysis for each IMF:")
        print("     • Sub-hourly: Turbulence, high-frequency fluctuations")
        print("     • Hourly: Boundary layer processes")
        print("     • Semi-diurnal: Land-sea breeze, twice-daily patterns")
        print("     • Diurnal: Daily solar heating cycles")
        print("     • Multi-day: Weather system passages")
        print("     • Weekly-Monthly: Synoptic patterns")
        print("     • Long-term: Seasonal trends")
        print("   ✓ NEW: Frequency vs correlation scatter plot")
        print("   ✓ Time scale reference lines (15min to 1 month)")
        print("   ✓ IMF annotations show which frequency bands correlate most")
        print()
        print("📂 Output files:")
        print("   • CEEMDAN_correlation_analysis_adaptive.png (3-panel plot with frequencies)")
        print("   • CEEMDAN_Cross_correlations_adaptive.csv (includes frequency columns)")
        print()
        print("🔍 How to interpret the results:")
        print("   • IMF1-3: Usually sub-hourly to hourly (turbulence, fast dynamics)")
        print("   • IMF4-6: Often hourly to semi-diurnal (boundary layer, local effects)")
        print("   • IMF7-9: Typically diurnal to multi-day (weather patterns)")
        print("   • IMF10+: Usually longer-term trends (seasonal, climate)")
        print("   • Panel 3 shows which time scales have strongest correlations")
        print("="*80)
    else:
        print("❌ Analysis failed - please check data path and format")