import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

# Set style for academic publication
plt.rcParams.update({
    'font.size': 10,
    'axes.titlesize': 11,
    'axes.labelsize': 10,
    'xtick.labelsize': 9,
    'ytick.labelsize': 9,
    'legend.fontsize': 7,
    'figure.titlesize': 13,
    'font.family': 'Arial'
})

# Define file paths
data_path = '/Users/xiaxin/work/WindForecast_Project/03_Results/03-09/Adaptive_CEEMDAN_Analysis/CEEMDAN_Cross_correlations_adaptive.csv'
output_dir = '/Users/xiaxin/work/WindForecast_Project/03_Results/figures/'

# Create output directory if it doesn't exist
os.makedirs(output_dir, exist_ok=True)

# Load data
data = pd.read_csv(data_path)
data['pair_id'] = data['Variable1'] + '-' + data['Variable2']

# Color scheme matching your original figure
color_mapping = {
    'obs_wind_speed_30m-obs_wind_speed_50m': '#2ca02c',  # Green
    'obs_wind_speed_30m-obs_wind_speed_70m': '#98df8a',  # Light green
    'obs_wind_speed_50m-obs_wind_speed_70m': '#8c564b',  # Brown
    'obs_wind_speed_10m-obs_wind_speed_30m': '#9467bd',  # Purple
    'obs_wind_speed_10m-obs_wind_speed_50m': '#aec7e8',  # Light blue
    'obs_wind_speed_10m-obs_wind_speed_70m': '#c5b0d5',  # Orange
    'obs_wind_speed_10m-power': '#d62728',  # Red - highlighted
    'obs_wind_speed_30m-power': '#ff9896',  # Light red
    'obs_wind_speed_50m-power': '#ff7f0e',  # Light purple #c5b0d5
    'obs_wind_speed_70m-power': '#e377c2'   # Pink
}

# Create figure with 2 subplots
fig = plt.figure(figsize=(14, 7))
gs = fig.add_gridspec(1, 2, width_ratios=[4, 1.5], wspace=0.25)

# Subplot 1: Period vs Correlation scatter plot (left side - main plot)
ax1 = fig.add_subplot(gs[0, 0])

valid_period_data = data.dropna(subset=['Average_Period_Hours'])

# Define specific vertical lines with labels
time_markers = {
    1: '1h',      # 1 hour
    6: '6h',      # 6 hours  
    24: '1d',     # 1 day
    168: '1w',    # 1 week (7*24 hours)
    720: '1m'     # 1 month (30*24 hours)
}

# Plot vertical dashed lines with annotations
for period_hours, label in time_markers.items():
    ax1.axvline(x=period_hours, color='gray', linestyle='--', alpha=0.6, linewidth=1.2)
    # Add label at the top of each line
    ax1.text(period_hours, 0.918, label, rotation=90, ha='center', va='bottom', 
             fontsize=18, color='gray', fontweight='bold',
             transform=ax1.get_xaxis_transform())

# Then plot the scatter points with connecting lines
for pair_id in color_mapping.keys():
    subset = valid_period_data[valid_period_data['pair_id'] == pair_id].copy()
    if len(subset) > 0:
        # Sort by period for proper line connection
        subset = subset.sort_values('Average_Period_Hours')
        
        if '10m' in pair_id:
            marker = 's'  # Square for 10m wind
            size = 80 if 'power' in pair_id else 50
        else:
            marker = 'o'  # Circle for others  
            size = 60 if 'power' in pair_id else 60
        
        alpha = 1.0 if 'power' in pair_id else 0.7
        linewidth = 1.5 if 'power' in pair_id else 1.0
        
        # Create label
        if 'power' in pair_id:
            height = pair_id.split('-')[0].replace('obs_wind_speed_', '').replace('_', '')
            label = f'{height}-Power'
        else:
            var1 = pair_id.split('-')[0].replace('obs_wind_speed_', '').replace('_', '')
            var2 = pair_id.split('-')[1].replace('obs_wind_speed_', '').replace('_', '')
            label = f'{var1}-{var2}'
        
        # Plot connecting line first (lower z-order)
        ax1.plot(subset['Average_Period_Hours'], subset['Pearson_Correlation'],
                color=color_mapping[pair_id], alpha=alpha*0.6, linewidth=linewidth, 
                linestyle='-', zorder=2)
        
        # Plot scatter points on top
        scatter = ax1.scatter(subset['Average_Period_Hours'], subset['Pearson_Correlation'],
                            c=color_mapping[pair_id], marker=marker, s=size, alpha=alpha, 
                            label=label, edgecolors='black', linewidth=0.5, zorder=3)
        
        # Add IMF numbers for power correlations
        if '70m-power' in pair_id:
            for _, row in subset.iterrows():
                ax1.annotate(f'IMF{int(row.IMF)}', 
                           (row['Average_Period_Hours'], row['Pearson_Correlation']),
                           xytext=(-16, -38), textcoords='offset points', 
                           fontsize=14, alpha=0.8, zorder=4)

ax1.set_xlabel('Period (Hours)', fontsize=16, fontfamily='Arial')
ax1.set_ylabel('Correlation Coefficient', fontsize=16, fontfamily='Arial')
ax1.set_title('(a)',fontsize=16, fontfamily='Arial')
ax1.set_xscale('log')
ax1.legend(fontsize=16, loc='lower right', ncol=2)
ax1.grid(False)
ax1.set_ylim(0, 1.05)
ax1.set_yticks(np.arange(0, 1.1, 0.1))
# 设置刻度标签的字体大小
ax1.tick_params(axis='both', which='major', labelsize=16)

# Subplot 2: Trend component correlation (right side - vertical bar chart)
ax2 = fig.add_subplot(gs[0, 1])

# Extract trend component correlations (IMF12 - the last/trend component)
trend_pairs = ['obs_wind_speed_10m-power', 'obs_wind_speed_30m-power', 
               'obs_wind_speed_50m-power', 'obs_wind_speed_70m-power']
heights = ['10m', '30m', '50m', '70m']
height_colors = ['#d62728', '#ff9896', '#c5b0d5', '#e377c2']

# Get trend component correlations (usually the highest IMF number)
max_imf = data['IMF'].max()  # This should be IMF12 (trend component)
trend_correlations = []

for pair in trend_pairs:
    subset = data[(data['pair_id'] == pair) & (data['IMF'] == max_imf)]
    if len(subset) > 0:
        trend_corr = subset['Pearson_Correlation'].iloc[0]
        trend_correlations.append(trend_corr)
    else:
        trend_correlations.append(0)

# Create horizontal bar chart for trend correlations
bars = ax2.barh(range(len(heights)), trend_correlations, 
                color=height_colors, alpha=0.85, height=0.2)

# Add value labels at the end of bars
for i, (bar, val) in enumerate(zip(bars, trend_correlations)):
    width_val = bar.get_width()
    if width_val >= 0:
        ax2.text(- 0.056-0.02, bar.get_y() + bar.get_height()/2.,
                 f'{val:.3f}', ha='right', va='center', fontsize=14, fontweight='normal')
    else:
        ax2.text(width_val - 0.02, bar.get_y() + bar.get_height()/2.,
                 f'{val:.3f}', ha='right', va='center', fontsize=14, fontweight='normal')

# Highlight the best performer in trend component
if trend_correlations and max(trend_correlations) > 0:
    best_idx = np.argmax(trend_correlations)
    bars[best_idx].set_color('#b91c1c')  # Darker red
    # bars[best_idx].set_alpha(1.0)
    # bars[best_idx].set_edgecolor('black')
    # bars[best_idx].set_linewidth(1.5)

ax2.set_yticks(range(len(heights)))
ax2.tick_params(axis='both', which='major', labelsize=14)
ax2.set_yticklabels([f'WS {h}' for h in heights], fontsize=14, fontfamily='Arial')
ax2.set_xlabel(f'Trend Component Correlation', fontsize=14, fontfamily='Arial')
ax2.set_title('(b)',fontsize=14, fontfamily='Arial')

# Add vertical line at x=0 (center of the plot)
ax2.axvline(x=0, color='black', linewidth=1.2, alpha=0.8)
ax2.grid(False)
ax2.set_axisbelow(True)

if trend_correlations:
    # Center the x-axis around 0 with better spacing
    max_abs_val = max(abs(min(trend_correlations)), abs(max(trend_correlations)))
    ax2.set_xlim(-max_abs_val * 1.3, max_abs_val * 1.3)
    ax2.set_ylim(-0.4, len(heights) - 0.6)  # Better vertical spacing
    
    # Add annotation for best trend performer
    # if max(trend_correlations) > 0:
    #     best_idx = np.argmax(trend_correlations)
    #     ax2.annotate('Strongest\nTrend Link', 
    #                  xy=(trend_correlations[best_idx], best_idx), 
    #                  xytext=(trend_correlations[best_idx] + max_abs_val * 0.15, best_idx + 0.25),
    #                  arrowprops=dict(arrowstyle='->', color='red', lw=1.8),
    #                  fontsize=9, fontweight='bold', color='red',
    #                  bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8, edgecolor="red"))

plt.tight_layout()

# Add main title
# fig.suptitle('EEMD Cross-Correlation Analysis: Multi-Height Wind Speed and Power Relationships', 
#              fontsize=14, fontweight='bold', y=0.95)

# Save the figure
try:
    png_path = os.path.join(output_dir, 'EEMD_frequency_domain_analysis.png')
    pdf_path = os.path.join(output_dir, 'EEMD_frequency_domain_analysis.pdf')
    
    plt.savefig(png_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.savefig(pdf_path, bbox_inches='tight', facecolor='white')
    
    print(f"Figures saved successfully:")
    print(f"PNG: {png_path}")  
    print(f"PDF: {pdf_path}")
    
except Exception as e:
    print(f"Error saving figures: {e}")

plt.show()

print(f"\n=== Analysis Summary ===")
if trend_correlations and max(trend_correlations) > 0:
    best_trend_idx = np.argmax(trend_correlations)
    print(f"Best trend component predictor: Wind {heights[best_trend_idx]} (trend r={max(trend_correlations):.3f})")
else:
    print("No trend data available")
print(f"Total variable pairs analyzed: {len(color_mapping)}")
print(f"Trend component: IMF{max_imf} (long-term variations)")