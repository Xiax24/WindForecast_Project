import numpy as np, pandas as pd
df = pd.read_csv('/Users/xiaxin/work/WindForecast_Project/01_Data/processed/matched_data/changma_matched.csv')
t = pd.to_datetime(df.iloc[:,0])
wd10 = df['obs_wind_direction_10m'].values

print('=== 10m WD 最高频取值 top20 ===')
print(pd.Series(wd10).round(1).value_counts().head(20))

print('\n=== 0-2° 内的精细分布 ===')
sub = wd10[(wd10>=0)&(wd10<2)]
print(pd.Series(sub).round(2).value_counts().head(15))
print('该区间总数 %d, 唯一值个数 %d'%(len(sub), len(np.unique(sub))))

print('\n=== 卡死段清单（连续不变 >8 步 = 2小时）===')
s = pd.Series(wd10)
grp = (s != s.shift()).cumsum()
runs = s.groupby(grp).agg(val='first', n='size')
runs = runs[(runs.n>8) & runs.val.notna()]
print(runs.sort_values('n', ascending=False).head(15))
print('卡死段总样本数: %d (%.1f%%)'%(runs.n.sum(), 100*runs.n.sum()/len(df)))

print('\n=== 2021-11 的 30m 异常 ===')
m = t.dt.to_period('M')=='2021-11'
for h in [10,30,50,70]:
    print(f'  {h}m 主峰 = %d°, 有效 %d'%(
        np.histogram(df[f'obs_wind_direction_{h}m'].values[m & df[f'obs_wind_direction_{h}m'].notna()], bins=72, range=(0,360))[1][
        np.histogram(df[f'obs_wind_direction_{h}m'].values[m & df[f'obs_wind_direction_{h}m'].notna()], bins=72, range=(0,360))[0].argmax()],
        (m & df[f'obs_wind_direction_{h}m'].notna()).sum()))

print('\n=== 70m 绝对定向核验：obs vs WRF ===')
for c in df.columns:
    if 'wind_direction_70m' in c and 'obs' not in c:
        v = ((df[c].values - df['obs_wind_direction_70m'].values + 180)%360)-180
        strong = df['obs_wind_speed_70m'].values>6
        print(f'  {c}: 中位偏差 %.1f° (强风 %.1f°)'%(np.nanmedian(v), np.nanmedian(v[strong])))
from scipy.stats import spearmanr
wd70 = df['obs_wind_direction_70m'].values
pw = df['power'].values
for name,lo,hi in [('WEST',225,315),('EAST',45,135)]:
    m = (wd70>=lo)&(wd70<=hi)
    print(f'\n{name}  N={m.sum()}  白天占比 {((t.dt.hour>=8)&(t.dt.hour<18))[m].mean():.2f}')
    for h in [10,30,50,70]:
        d = pd.DataFrame({'w':df[f'obs_wind_speed_{h}m'],'p':pw})[m].dropna()
        print('  %2dm  r=%.3f  (n=%d)'%(h, spearmanr(d.w,d.p)[0], len(d)))