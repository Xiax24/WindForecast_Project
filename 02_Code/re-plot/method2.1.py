import pandas as pd
import numpy as np

csv_file = '/Users/xiaxin/work/WindForecast_Project/01_Data/processed/matched_data/changma_matched.csv'
heights = [10, 30, 50, 70]

df = pd.read_csv(csv_file)

data = {}
for h in heights:
    data[f'ws_{h}m'] = df[f'obs_wind_speed_{h}m'].values
    data[f'wd_{h}m'] = df[f'obs_wind_direction_{h}m'].values

# 所有高度风向都有效的mask
mask_all = np.ones(len(df), dtype=bool)
for h in heights:
    mask_all = mask_all & ~np.isnan(data[f'wd_{h}m'])

# 严格筛选：所有高度都在指定风向区间
def strict_direction_mask(data, heights, direction_range):
    min_deg, max_deg = direction_range
    mask = np.ones(len(data[f'wd_{heights[0]}m']), dtype=bool)
    for h in heights:
        wd = data[f'wd_{h}m']
        if min_deg > max_deg:
            h_mask = (wd >= min_deg) | (wd <= max_deg)
        else:
            h_mask = (wd >= min_deg) & (wd <= max_deg)
        mask = mask & h_mask & ~np.isnan(wd)
    return mask

mask_west = strict_direction_mask(data, heights, (225, 315))
mask_east = strict_direction_mask(data, heights, (45, 135))

n_total = np.sum(mask_all)
n_west = np.sum(mask_west)
n_east = np.sum(mask_east)

pct_west = n_west / n_total * 100
pct_east = n_east / n_total * 100

print(f"Total valid samples: {n_total:,}")
print(f"Free-stream (Westerly 225°-315°): {n_west:,} ({pct_west:.1f}%)")
print(f"Wake (Easterly 45°-135°): {n_east:,} ({pct_east:.1f}%)")
print(f"Other directions: {n_total - n_west - n_east:,} ({100 - pct_west - pct_east:.1f}%)")