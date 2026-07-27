import pandas as pd, numpy as np

df = pd.read_csv('/Users/xiaxin/work/WindForecast_Project/01_Data/processed/matched_data/changma_matched.csv')
H = [10,30,50,70]
wd = {h: df[f'obs_wind_direction_{h}m'].values for h in H}
ws = {h: df[f'obs_wind_speed_{h}m'].values for h in H}
pw = df['power'].values

def insec(x, lo, hi): return (x>=lo)&(x<=hi)

print('N_raw                    =', len(df))
ok4 = np.ones(len(df), bool)
for h in H: ok4 &= ~np.isnan(wd[h])
print('N_all4_wd_valid          =', ok4.sum())

for name, lo, hi in [('WEST(225-315)',225,315), ('EAST(45-135)',45,135)]:
    s70 = insec(wd[70], lo, hi) & ok4          # 仅按 70 m 判（= 风玫瑰口径）
    s4  = ok4.copy()
    for h in H: s4 &= insec(wd[h], lo, hi)
    print(f'\n{name}')
    print('  70m-only in sector     =', s70.sum())
    print('  all-4  in sector       =', s4.sum())
    print('  retention of 4-height  = %.1f%%'%(100*s4.sum()/max(s70.sum(),1)))
    # veer 分布（保留样本 vs 仅70m通过但被剔除的样本）
    v = ((wd[70]-wd[10]+180)%360)-180
    print('  veer|70-10| kept  : med %.1f  p95 %.1f'%(np.nanmedian(np.abs(v[s4])), np.nanpercentile(np.abs(v[s4]),95)))
    rej = s70 & ~s4
    print('  veer|70-10| reject: med %.1f  p95 %.1f  (n=%d)'%(np.nanmedian(np.abs(v[rej])), np.nanpercentile(np.abs(v[rej]),95), rej.sum()))
    # 日内分布
    t = pd.to_datetime(df.iloc[:,0]) if df.columns[0].lower().startswith(('time','date')) else None
    if t is not None:
        print('  kept   daytime(08-18) frac = %.2f'%(((t.dt.hour>=8)&(t.dt.hour<18))[s4].mean()))
        print('  reject daytime(08-18) frac = %.2f'%(((t.dt.hour>=8)&(t.dt.hour<18))[rej].mean()))