import numpy as np, pandas as pd
df = pd.read_csv('/Users/xiaxin/work/WindForecast_Project/01_Data/processed/matched_data/changma_matched.csv')
t = pd.to_datetime(df.iloc[:,0])

print('=== 卡值检查：各高度风向精确等于 0 的条数 ===')
for h in [10,30,50,70]:
    wd = df[f'obs_wind_direction_{h}m'].values
    n0 = (wd==0).sum(); nv = (~np.isnan(wd)).sum()
    print(f'{h}m: ==0 精确 {n0:6d} ({100*n0/nv:5.2f}%)   <1° {(wd<1).sum():6d}   有效 {nv}')

print('\n=== 10m 风向：0 值时刻的其他信息 ===')
wd10 = df['obs_wind_direction_10m'].values
z = (wd10==0)
print('10m WD==0 时: 10m风速 中位 %.2f, 70m风速 中位 %.2f, 70m风向 中位 %.0f'%(
    np.nanmedian(df['obs_wind_speed_10m'].values[z]),
    np.nanmedian(df['obs_wind_speed_70m'].values[z]),
    np.nanmedian(df['obs_wind_direction_70m'].values[z])))
print('0 值的时间跨度: %s → %s'%(t[z].min(), t[z].max()))
print('0 值按月分布:'); print(t[z].dt.to_period('M').value_counts().sort_index())

print('\n=== 连续卡值段（10m WD 连续不变）===')
d = np.abs(np.diff(wd10)); stuck = (d==0)
runs=[]; c=0
for s in stuck:
    c = c+1 if s else 0
    if c: runs.append(c)
import itertools
print('最长连续不变步数:', max(runs) if runs else 0, '(15min/步)')

print('\n=== 偏角随时间是否稳定（月度中位）===')
ws70 = df['obs_wind_speed_70m'].values
for h in [10,30,50]:
    v = ((df[f'obs_wind_direction_{h}m'].values - df['obs_wind_direction_70m'].values + 180)%360)-180
    g = pd.DataFrame({'m':t.dt.to_period('M'),'v':np.where(ws70>6,v,np.nan)}).groupby('m').v.median()
    print(f'{h}m: ', g.round(1).to_dict())