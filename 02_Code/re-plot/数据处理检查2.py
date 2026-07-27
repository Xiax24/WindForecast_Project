import numpy as np, pandas as pd
df = pd.read_csv('/Users/xiaxin/work/WindForecast_Project/01_Data/processed/matched_data/changma_matched.csv')
ws70 = df['obs_wind_speed_70m'].values
for h in [10,30,50]:
    v = ((df[f'obs_wind_direction_{h}m'].values - df['obs_wind_direction_70m'].values + 180) % 360) - 180
    strong = ws70 > 6          # 强风时风向定义最可靠
    print(f'{h}m vs 70m | 全样本 中位 {np.nanmedian(v):6.1f}  |  强风(>6m/s) 中位 {np.nanmedian(v[strong]):6.1f}  众数附近 {np.nanpercentile(v[strong],[25,50,75]).round(1)}')
# 各高度风向直方图的峰位
for h in [10,30,50,70]:
    wd = df[f'obs_wind_direction_{h}m'].values
    hist,edges = np.histogram(wd[~np.isnan(wd)], bins=72, range=(0,360))
    print(f'{h}m 主峰方向 = {edges[hist.argmax()]:.0f}°')