import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import warnings
import os
warnings.filterwarnings('ignore')

# Try importing EMD library
EMD_AVAILABLE = False
EMD_LIBRARY = None

try:
    from PyEMD import CEEMDAN
    print("Successfully imported CEEMDAN from PyEMD (EMD-signal package)")
    EMD_AVAILABLE = True
    EMD_LIBRARY = "PyEMD"
except ImportError:
    try:
        import emd
        print("Successfully imported emd library")
        
        # Check available functions
        print(f"emd.sift functions: {[attr for attr in dir(emd.sift) if 'sift' in attr.lower()]}")
        
        # Try to get function signature
        try:
            import inspect
            # Check if complete_ensemble_sift is available (CEEMDAN equivalent)
            if hasattr(emd.sift, 'complete_ensemble_sift'):
                sig = inspect.signature(emd.sift.complete_ensemble_sift)
                print(f"complete_ensemble_sift parameters: {list(sig.parameters.keys())}")
            else:
                print("complete_ensemble_sift not available, will use ensemble_sift")
        except:
            print("Could not inspect function parameters")
        
        EMD_AVAILABLE = True
        EMD_LIBRARY = "emd"
    except ImportError:
        print("No EMD library found. Using simplified implementation.")
        EMD_AVAILABLE = False

# Set paths
data_path = '/Users/xiaxin/work/WindForecast_Project/01_Data/processed/imputed_data/changma_clean.csv'
results_path = '/Users/xiaxin/work/WindForecast_Project/03_Results/08时间序列分解/CEEMDAN'

# Create results directory if it doesn't exist
os.makedirs(results_path, exist_ok=True)

# Load data
print("\nLoading data...")
df = pd.read_csv(data_path)

# Convert datetime column
df['datetime'] = pd.to_datetime(df['datetime'])
df.set_index('datetime', inplace=True)

# Select variables for analysis
variables = ['obs_wind_speed_10m', 'obs_wind_speed_70m', 'power']

print(f"Data shape: {df.shape}")
print(f"Date range: {df.index.min()} to {df.index.max()}")
print(f"Variables to analyze: {variables}")

# Simplified EMD implementation as backup
def simple_ceemdan(data, ensemble_size=50, max_imfs=10):
    """
    Simplified CEEMDAN implementation when libraries are not available
    This is a basic version for demonstration purposes
    """
    imfs = []
    residue = data.copy()
    
    noise_std = 0.2 * np.std(data)
    
    for mode in range(max_imfs):
        if len(residue) < 10:
            break
        
        # CEEMDAN: use noise in specific way
        ensemble_imfs = []
        
        for ens in range(ensemble_size):
            # Add noise to original signal for first mode, to residue for others
            if mode == 0:
                noisy_signal = data + noise_std * np.random.randn(len(data))
            else:
                noisy_signal = residue + noise_std * np.random.randn(len(residue))
            
            # Simple sifting process
            h = noisy_signal.copy()
            
            for sift in range(10):  # Limited sifting
                try:
                    from scipy.signal import argrelextrema
                    from scipy.interpolate import interp1d
                    
                    max_indices = argrelextrema(h, np.greater)[0]
                    min_indices = argrelextrema(h, np.less)[0]
                    
                    if len(max_indices) < 2 or len(min_indices) < 2:
                        break
                    
                    # Interpolate envelopes
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
            
            ensemble_imfs.append(h[:len(residue)])
        
        # Average ensemble to get IMF
        if ensemble_imfs:
            imf = np.mean(ensemble_imfs, axis=0)
            imfs.append(imf)
            residue = residue - imf
        
        # Check if residue is monotonic
        try:
            from scipy.signal import argrelextrema
            if len(argrelextrema(residue, np.greater)[0]) < 2:
                break
        except:
            break
    
    # Add final residue
    imfs.append(residue)
    return np.array(imfs)

# Function to perform CEEMDAN decomposition
def perform_ceemdan_decomposition(data, variable_name, ensemble_size=50, noise_std=0.2):
    """
    Perform CEEMDAN decomposition on time series data
    """
    print(f"\nPerforming CEEMDAN decomposition for {variable_name}...")
    
    # Remove missing values
    clean_data = data.dropna()
    
    if len(clean_data) < 100:
        print(f"Warning: Not enough data for {variable_name}")
        return None
    
    # Prepare data
    values = clean_data.values
    time_index = clean_data.index
    
    print(f"  Data points: {len(values)}")
    print(f"  Ensemble size: {ensemble_size}")
    print(f"  Noise std: {noise_std}")
    print(f"  Using EMD library: {EMD_LIBRARY}")
    
    try:
        if EMD_LIBRARY == "PyEMD":
            # Use PyEMD CEEMDAN if available
            ceemdan = CEEMDAN()
            ceemdan.ensemble_size = ensemble_size
            ceemdan.noise_std = noise_std
            
            print("  Running CEEMDAN decomposition with PyEMD...")
            IMFs = ceemdan.ceemdan(values)
            
        elif EMD_LIBRARY == "emd":
            # Use emd library
            print("  Running CEEMDAN decomposition with emd library...")
            
            import emd
            
            # Check if complete_ensemble_sift is available (CEEMDAN equivalent)
            if hasattr(emd.sift, 'complete_ensemble_sift'):
                try:
                    imfs = emd.sift.complete_ensemble_sift(values, nensembles=ensemble_size)
                    print("  Used complete_ensemble_sift (CEEMDAN)")
                except Exception as e1:
                    try:
                        # Try without noise parameter
                        imfs = emd.sift.complete_ensemble_sift(values, nensembles=ensemble_size)
                        print("  Used complete_ensemble_sift without noise parameter")
                    except Exception as e2:
                        print(f"  complete_ensemble_sift failed, falling back to ensemble_sift")
                        imfs = emd.sift.ensemble_sift(values, nensembles=ensemble_size)
            else:
                print("  complete_ensemble_sift not available, using ensemble_sift")
                imfs = emd.sift.ensemble_sift(values, nensembles=ensemble_size)
            
            # Convert to the format we expect (transpose)
            IMFs = imfs.T
            
        else:
            # Use simplified implementation
            print("  Running simplified CEEMDAN decomposition...")
            print("  Note: This is a basic implementation. For best results, install PyEMD.")
            
            IMFs = simple_ceemdan(values, ensemble_size)
        
        # Create result dictionary
        result = {
            'original_data': clean_data,
            'IMFs': IMFs,
            'time_index': time_index,
            'n_imfs': len(IMFs)
        }
        
        print(f"  Decomposition completed: {len(IMFs)} IMFs generated")
        return result
        
    except Exception as e:
        print(f"  Error during CEEMDAN decomposition: {e}")
        import traceback
        traceback.print_exc()
        return None

# Function to plot CEEMDAN results
def plot_ceemdan_results(ceemdan_result, variable_name, save_path):
    """
    Plot CEEMDAN decomposition results
    """
    if ceemdan_result is None:
        return
    
    IMFs = ceemdan_result['IMFs']
    time_index = ceemdan_result['time_index']
    original_data = ceemdan_result['original_data']
    n_imfs = len(IMFs)
    
    # Create subplot layout
    fig, axes = plt.subplots(n_imfs + 1, 1, figsize=(15, 2 * (n_imfs + 1)))
    fig.suptitle(f'CEEMDAN Decomposition - {variable_name}', fontsize=16, y=0.98)
    
    # Plot original data
    axes[0].plot(time_index, original_data.values, 'b-', linewidth=0.8)
    axes[0].set_title('Original Data')
    axes[0].set_ylabel('Value')
    axes[0].grid(True, alpha=0.3)
    
    # Plot IMFs
    for i, imf in enumerate(IMFs):
        axes[i + 1].plot(time_index, imf, linewidth=0.8)
        
        # Different colors for different IMF types
        if i < 3:
            axes[i + 1].set_title(f'IMF {i+1} (High Frequency)')
            color = 'red'
        elif i < n_imfs - 3:
            axes[i + 1].set_title(f'IMF {i+1} (Medium Frequency)')
            color = 'green'
        else:
            axes[i + 1].set_title(f'IMF {i+1} (Low Frequency/Trend)')
            color = 'purple'
        
        axes[i + 1].plot(time_index, imf, color=color, linewidth=0.8)
        axes[i + 1].set_ylabel(f'IMF {i+1}')
        axes[i + 1].grid(True, alpha=0.3)
    
    axes[-1].set_xlabel('Date')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()

# Function to analyze IMF characteristics
def analyze_imf_characteristics(ceemdan_result, variable_name):
    """
    Analyze characteristics of each IMF
    """
    if ceemdan_result is None:
        return None
    
    IMFs = ceemdan_result['IMFs']
    n_imfs = len(IMFs)
    
    characteristics = []
    
    for i, imf in enumerate(IMFs):
        # Calculate basic statistics
        mean_val = np.mean(imf)
        std_val = np.std(imf)
        energy = np.sum(imf ** 2)
        
        # Estimate dominant frequency
        zero_crossings = np.sum(np.diff(np.sign(imf)) != 0)
        estimated_period = len(imf) / (zero_crossings / 2) if zero_crossings > 0 else np.inf
        
        # Convert to time units (15-minute intervals)
        period_hours = estimated_period * 0.25
        
        char = {
            'Variable': variable_name,
            'IMF': i + 1,
            'Mean': mean_val,
            'Std': std_val,
            'Energy': energy,
            'Energy_Ratio': energy / np.sum([np.sum(imf_i ** 2) for imf_i in IMFs]) * 100,
            'Zero_Crossings': zero_crossings,
            'Estimated_Period_Points': estimated_period,
            'Estimated_Period_Hours': period_hours,
            'Interpretation': get_imf_interpretation(i, n_imfs, period_hours)
        }
        
        characteristics.append(char)
    
    return characteristics

# Function to interpret IMF meaning
def get_imf_interpretation(imf_index, total_imfs, period_hours):
    """
    Provide interpretation for each IMF based on its characteristics
    """
    if imf_index < 2:
        return "High-frequency noise/measurement errors"
    elif period_hours < 6:
        return "Short-term fluctuations (< 6 hours)"
    elif 6 <= period_hours <= 30:
        return "Daily cycles and sub-daily patterns"
    elif 30 < period_hours <= 200:
        return "Weekly and multi-day patterns"
    elif 200 < period_hours <= 2000:
        return "Monthly and seasonal patterns"
    else:
        return "Long-term trends and climate patterns"

# Function to calculate reconstruction quality
def calculate_reconstruction_quality(ceemdan_result, variable_name):
    """
    Calculate how well IMFs reconstruct the original signal
    """
    if ceemdan_result is None:
        return None
    
    original = ceemdan_result['original_data'].values
    IMFs = ceemdan_result['IMFs']
    
    # Reconstruct signal
    reconstructed = np.sum(IMFs, axis=0)
    
    # Calculate metrics
    mse = np.mean((original - reconstructed) ** 2)
    rmse = np.sqrt(mse)
    mae = np.mean(np.abs(original - reconstructed))
    
    # Correlation
    correlation = np.corrcoef(original, reconstructed)[0, 1]
    
    # Variance explained
    explained_variance = 1 - (np.var(original - reconstructed) / np.var(original))
    
    quality = {
        'Variable': variable_name,
        'RMSE': rmse,
        'MAE': mae,
        'Correlation': correlation,
        'Explained_Variance': explained_variance * 100,
        'Max_Reconstruction_Error': np.max(np.abs(original - reconstructed))
    }
    
    return quality

# Main analysis
decomposition_results = {}
all_characteristics = []
quality_metrics = []

for var in variables:
    print(f"\n{'='*60}")
    print(f"Analyzing variable: {var}")
    print(f"{'='*60}")
    
    # Check data availability
    if var not in df.columns:
        print(f"Warning: {var} not found in data")
        continue
    
    # Get data for this variable
    var_data = df[var]
    print(f"Data points: {len(var_data)}")
    print(f"Missing values: {var_data.isna().sum()}")
    print(f"Value range: {var_data.min():.2f} to {var_data.max():.2f}")
    
    # Perform CEEMDAN decomposition
    ceemdan_result = perform_ceemdan_decomposition(var_data, var)
    
    if ceemdan_result is not None:
        # Store result
        decomposition_results[var] = ceemdan_result
        
        # Plot results
        plot_path = os.path.join(results_path, f'CEEMDAN_decomposition_{var}.png')
        plot_ceemdan_results(ceemdan_result, var, plot_path)
        
        # Analyze IMF characteristics
        characteristics = analyze_imf_characteristics(ceemdan_result, var)
        if characteristics:
            all_characteristics.extend(characteristics)
        
        # Calculate reconstruction quality
        quality = calculate_reconstruction_quality(ceemdan_result, var)
        if quality:
            quality_metrics.append(quality)
        
        print(f"\nDecomposition completed for {var}")
        print(f"Number of IMFs: {ceemdan_result['n_imfs']}")
        print(f"Plot saved to: {plot_path}")

# Save detailed characteristics
if all_characteristics:
    char_df = pd.DataFrame(all_characteristics)
    
    # Round numerical columns
    numerical_cols = char_df.select_dtypes(include=[np.number]).columns
    char_df[numerical_cols] = char_df[numerical_cols].round(4)
    
    char_path = os.path.join(results_path, 'CEEMDAN_IMF_characteristics.csv')
    char_df.to_csv(char_path, index=False)
    
    print(f"\n{'='*60}")
    print("CEEMDAN IMF CHARACTERISTICS")
    print(f"{'='*60}")
    
    # Display summary
    for var in variables:
        if var in decomposition_results:
            var_chars = char_df[char_df['Variable'] == var]
            print(f"\n{var}:")
            for _, row in var_chars.iterrows():
                print(f"  IMF {row['IMF']:2d}: {row['Interpretation']:<35} "
                      f"(Period: {row['Estimated_Period_Hours']:.1f}h, "
                      f"Energy: {row['Energy_Ratio']:.1f}%)")

# Save quality metrics
if quality_metrics:
    quality_df = pd.DataFrame(quality_metrics)
    
    # Round numerical columns
    numerical_cols = quality_df.select_dtypes(include=[np.number]).columns
    quality_df[numerical_cols] = quality_df[numerical_cols].round(4)
    
    quality_path = os.path.join(results_path, 'CEEMDAN_reconstruction_quality.csv')
    quality_df.to_csv(quality_path, index=False)
    
    print(f"\n{'='*60}")
    print("CEEMDAN RECONSTRUCTION QUALITY")
    print(f"{'='*60}")
    print(quality_df.to_string(index=False))

# Analysis summary
print(f"\n{'='*60}")
print("CEEMDAN DECOMPOSITION ANALYSIS COMPLETE")
print(f"{'='*60}")
print(f"Total variables analyzed: {len(decomposition_results)}")
print(f"Results saved to: {results_path}")
print("\nKey advantages of CEEMDAN over EEMD:")
print("1. Better reconstruction accuracy")
print("2. Reduced mode mixing")
print("3. More consistent IMF extraction")
print("4. Improved noise handling")
print("\nFiles generated:")
for var in decomposition_results.keys():
    print(f"  - CEEMDAN_decomposition_{var}.png")
print("  - CEEMDAN_IMF_characteristics.csv")
print("  - CEEMDAN_reconstruction_quality.csv")