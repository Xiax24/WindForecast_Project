import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import warnings
warnings.filterwarnings('ignore')

# Set matplotlib style
plt.style.use('default')
plt.rcParams['font.family'] = 'Arial'
plt.rcParams['font.size'] = 11

# Load data
data_path = '/Users/xiaxin/work/WindForecast_Project/01_Data/processed/imputed_data/changma_clean.csv'
df = pd.read_csv(data_path)

# Convert datetime column
df['datetime'] = pd.to_datetime(df['datetime'])
df.set_index('datetime', inplace=True)

# Select wind speed variables at 4 heights
wind_variables = ['obs_wind_speed_10m', 'obs_wind_speed_30m', 'obs_wind_speed_50m', 'obs_wind_speed_70m']
heights = [10, 30, 50, 70]  # Heights in meters

# Extract wind speed data for EOF analysis
wind_data = df[wind_variables].copy()

# Remove rows with any missing values
wind_data_clean = wind_data.dropna()
print(f"Data points for EOF analysis: {len(wind_data_clean)}")

# Standardize the data
wind_data_std = (wind_data_clean - wind_data_clean.mean()) / wind_data_clean.std()

# Perform PCA (EOF analysis)
pca = PCA(n_components=4)
pca.fit(wind_data_std)

# Get EOF patterns and explained variance
eof_patterns = pca.components_
explained_variance = pca.explained_variance_ratio_

print("Explained variance ratios:")
for i, var in enumerate(explained_variance):
    print(f"EOF {i+1}: {var*100:.2f}%")

# Function to interpret EOF patterns
def interpret_eof_pattern(pattern):
    """Provide physical interpretation of EOF patterns"""
    if np.all(pattern > 0) or np.all(pattern < 0):
        return "Uniform\n(All heights\nvary together)"
    
    correlation_with_height = np.corrcoef(pattern, heights)[0, 1]
    
    if correlation_with_height > 0.7:
        return "Shear increase\n(Upper levels\nstronger)"
    elif correlation_with_height < -0.7:
        return "Shear decrease\n(Lower levels\nstronger)"
    else:
        return "Mixed pattern\n(Complex\nvertical structure)"

# Create the EOF patterns plot
fig, axes = plt.subplots(1, 4, figsize=(16, 6))
# fig.suptitle('EOF Patterns (Vertical Wind Speed Profiles)', fontsize=16, fontweight='bold', y=0.95)

colors = ['blue', 'blue', 'blue', 'blue']  # Different colors for each EOF

for i in range(4):
    # Plot EOF pattern as vertical profile
    axes[i].plot(eof_patterns[i], heights, 'o-', linewidth=3, markersize=10, 
                color=colors[i], markerfacecolor='blue', markeredgecolor=colors[i], 
                markeredgewidth=2)
    
    # Add zero reference line
    axes[i].axvline(x=0, color='black', linestyle='--', alpha=0.6, linewidth=1)
    
    # Set labels and title
    axes[i].set_title(f'EOF {i+1}', fontsize=23, fontweight='normal')
    axes[i].set_xlabel('EOF Loading', fontsize=23)
    if i == 0:
        axes[i].set_ylabel('Height (m)', fontsize=23)
    
    # Set tick label size
    axes[i].tick_params(axis='y', which='major', labelsize=22)
    axes[i].tick_params(axis='x', which='major', labelsize=22)
    # Set grid
    axes[i].grid(True, alpha=0.3, linestyle='-', linewidth=0.5)
    

    # Set y-axis limits and ticks
    axes[i].set_ylim(0, 80)
    axes[i].set_yticks([10, 30, 50, 70])
    axes[i].set_yticklabels(['10m', '30m', '50m', '70m'])
    # Add variance percentage text in top-left corner
    axes[i].text(0.05, 0.95, f'{explained_variance[i]*100:.1f}%', 
                transform=axes[i].transAxes, 
                verticalalignment='top', horizontalalignment='left',
                fontsize=18, fontweight='normal')
    
    # Beautify axes
    for spine in axes[i].spines.values():
        spine.set_linewidth(1.2)
    
    # Set x-axis range for better visualization
    x_range = np.abs(eof_patterns[i]).max() * 1.2
    axes[i].set_xlim(-x_range, x_range)
    # axes[i].set_xlim(-0.6, 0.6)
    # axes[i].set_xticks([-0.6, -0.5,-0.4, -0.3,-0.2, -0.1, 0, 0.1, 0.2,0.3,0.4,0.5,0.6]) 

# Adjust layout
plt.tight_layout()
plt.subplots_adjust(top=0.85)

# Save the figure
save_path = '/Users/xiaxin/work/WindForecast_Project/03_Results/figures/3.1results/EOF_patterns_clean.png'
plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white', edgecolor='none')

save_path_pdf = '/Users/xiaxin/work/WindForecast_Project/03_Results/figures/3.1results/EOF_patterns_clean.pdf'
plt.savefig(save_path_pdf, bbox_inches='tight', facecolor='white', edgecolor='none')

plt.show()

print("\nEOF Patterns plot saved successfully!")
print(f"PNG: {save_path}")
print(f"PDF: {save_path_pdf}")

# Display the actual loading coefficients for verification
print(f"\nEOF Loading Coefficients:")
print("Height    EOF1     EOF2     EOF3     EOF4")
print("-" * 40)
for i, h in enumerate(heights):
    print(f"{h:3d}m   {eof_patterns[0,i]:6.3f}  {eof_patterns[1,i]:6.3f}  {eof_patterns[2,i]:6.3f}  {eof_patterns[3,i]:6.3f}")

# Verify first mode statistics mentioned in text
first_mode_loadings = eof_patterns[0]
loading_range = np.max(first_mode_loadings) - np.min(first_mode_loadings)

print(f"\nFirst EOF Mode Analysis:")
print(f"Explained variance: {explained_variance[0]*100:.2f}%")
print(f"Loading coefficients: {first_mode_loadings}")
print(f"Range of loadings: {loading_range:.3f}")
print(f"All positive? {np.all(first_mode_loadings > 0)}")