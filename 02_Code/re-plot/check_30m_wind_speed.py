#!/usr/bin/env python3
"""
30 m wind-SPEED health check.

Motivation: under the 70 m-only sectoring the correlation profile is
non-monotonic in the free-stream sector (10 m 0.825 -> 30 m 0.800 -> 50 m 0.831
-> 70 m 0.842). A dip at 30 m is not physical in a surface layer. The 30 m VANE
already has a +12.2 deg offset and a 2021-11 failure, so the 30 m ANEMOMETER
needs to be cleared before "redundancy within the rotor-swept layer" (Sec 4.3)
can be attributed to physics rather than to a bad sensor.
"""
import numpy as np, pandas as pd

CSV = '/Users/xiaxin/work/WindForecast_Project/01_Data/processed/matched_data/changma_matched.csv'
H = [10, 30, 50, 70]

df = pd.read_csv(CSV)
t = pd.to_datetime(df['datetime'])
ws = {h: df[f'obs_wind_speed_{h}m'].values for h in H}
wd70 = df['obs_wind_direction_70m'].values

print('=== 1. Shear ratios, monthly medians (strong wind only, ws70 > 6) ===')
strong = ws[70] > 6
for a, b in [(30, 10), (50, 30), (70, 50), (70, 10)]:
    r = np.where(strong, ws[a] / ws[b], np.nan)
    g = pd.DataFrame({'m': t.dt.to_period('M'), 'r': r}).groupby('m').r.median()
    print(f'  ws{a}/ws{b}: ', {str(k): round(v, 3) for k, v in g.items()})

print('\n=== 2. Monotonicity violations (profile should increase with height) ===')
for a, b in [(30, 10), (50, 30), (70, 50)]:
    v = (ws[a] < ws[b]) & strong
    print(f'  ws{a} < ws{b}: {v.sum()} ({100*v.mean():.1f}% of strong-wind records)')

print('\n=== 3. Power-law exponent per layer (strong wind median) ===')
for a, b in [(30, 10), (50, 30), (70, 50), (70, 10)]:
    al = np.log(ws[a] / ws[b]) / np.log(a / b)
    print(f'  alpha({b}-{a} m): median {np.nanmedian(al[strong]):.3f}'
          f'  IQR [{np.nanpercentile(al[strong],25):.3f}, {np.nanpercentile(al[strong],75):.3f}]')

print('\n=== 4. Mast shadowing: ws ratio vs 70 m wind direction (10 deg bins) ===')
b = np.arange(0, 361, 10)
idx = np.clip(np.digitize(wd70, b) - 1, 0, len(b) - 2)
out = pd.DataFrame({'bin': b[:-1][idx]})
for h in [10, 30, 50]:
    out[f'r{h}'] = np.where(strong, ws[h] / ws[70], np.nan)
g = out.groupby('bin').agg(['median', 'size'])
print(g.round(3).to_string())
print('  -> a narrow direction band where one height dips sharply = tower shadow')

print('\n=== 5. Stuck / flatline segments in wind speed ===')
for h in H:
    s = pd.Series(ws[h])
    grp = (s != s.shift()).cumsum()
    runs = s.groupby(grp).agg(val='first', n='size')
    runs = runs[(runs.n > 8) & runs.val.notna() & (runs.val > 0.5)]
    print(f'  {h}m: {len(runs)} segments >2h, total {runs.n.sum()} rows'
          f' ({100*runs.n.sum()/len(df):.2f}%), longest {runs.n.max() if len(runs) else 0} steps')

print('\n=== 6. 2021-11 wind speed sanity ===')
m = (t.dt.to_period('M').astype(str) == '2021-11').values
for h in H:
    print(f'  {h}m: median {np.nanmedian(ws[h][m]):.2f} (rest of record'
          f' {np.nanmedian(ws[h][~m]):.2f}), NaN {np.isnan(ws[h][m]).sum()}')

print('\n=== 7. Does the 30 m dip survive excluding 2021-11? ===')
from scipy.stats import spearmanr
pw = df['power'].values
for tag, keep in [('all', np.ones(len(df), bool)), ('excl 2021-11', ~m)]:
    for nm, lo, hi in [('WEST', 225, 315), ('EAST', 45, 135)]:
        sel = keep & (wd70 >= lo) & (wd70 <= hi) & (pw >= 0)
        rs = []
        for h in H:
            d = pd.DataFrame({'w': ws[h], 'p': pw})[sel].dropna()
            rs.append(spearmanr(d.w, d.p)[0])
        print(f'  {tag:14s} {nm}: ' + '  '.join(f'{h}m={r:.3f}' for h, r in zip(H, rs)))
