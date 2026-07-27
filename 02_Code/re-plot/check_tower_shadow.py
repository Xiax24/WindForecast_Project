#!/usr/bin/env python3
"""
Tower / boom shadow diagnostic.

Problem: ws_h/ws70 varies with wind direction, but that ratio responds to BOTH
  (a) stability (shear), which changes the whole profile coherently, and
  (b) flow distortion by the lattice mast / booms, which hits ONE height only.

Separation strategy:
  1. Restrict to near-neutral conditions (bulk 10-70 m shear exponent in a
     narrow band) -> removes most of (a).
  2. Express each ratio as a log-anomaly from its own record median, then
     divide by ln(70/z). Under pure shear all three heights collapse onto ONE
     curve. A height that departs from the others is flow-distorted.
  3. Split the record in half. A boom shadow is fixed in space -> identical in
     both halves. Stability/wake artefacts drift.
"""
import numpy as np, pandas as pd

CSV = '/Users/xiaxin/work/WindForecast_Project/01_Data/processed/matched_data/changma_matched.csv'
H = [10, 30, 50]

df = pd.read_csv(CSV)
t = pd.to_datetime(df['datetime'])
ws = {h: df[f'obs_wind_speed_{h}m'].values for h in H + [70]}
wd70 = df['obs_wind_direction_70m'].values

# bulk shear exponent 10-70 m
with np.errstate(all='ignore'):
    alpha = np.log(ws[70] / ws[10]) / np.log(70 / 10)

strong = ws[70] > 6
neutral = strong & (alpha > 0.08) & (alpha < 0.14)      # near-neutral band
print(f'strong-wind records: {strong.sum()};  near-neutral subset: {neutral.sum()}')

def shadow_curve(mask, bw=5):
    b = np.arange(0, 361, bw)
    idx = np.clip(np.digitize(wd70, b) - 1, 0, len(b) - 2)
    rows = {'bin': b[:-1][idx]}
    for h in H:
        with np.errstate(all='ignore'):
            r = np.log(ws[h] / ws[70]) / np.log(70 / h)   # shear-normalised
        rows[f's{h}'] = np.where(mask, r, np.nan)
    d = pd.DataFrame(rows)
    g = d.groupby('bin').median()
    n = d.groupby('bin').size()
    # anomaly from each height's own overall median
    for h in H:
        g[f's{h}'] = g[f's{h}'] - np.nanmedian(d[f's{h}'])
    g['n'] = n
    return g

print('\n=== Shear-normalised anomaly by direction (near-neutral, 5 deg bins) ===')
print('   All three columns should be ~0 and MOVE TOGETHER if only stability varies.')
print('   A single column dipping alone = flow distortion at that height.\n')
g = shadow_curve(neutral)
print(g[g.n >= 30].round(3).to_string())

print('\n=== Split-half stability of the pattern (a real shadow is identical) ===')
mid = t.iloc[len(t) // 2]
h1 = neutral & (t < mid).values
h2 = neutral & (t >= mid).values
g1, g2 = shadow_curve(h1), shadow_curve(h2)
cmp = pd.DataFrame({f's{h}_1st': g1[f's{h}'] for h in H})
for h in H:
    cmp[f's{h}_2nd'] = g2[f's{h}']
ok = (g1.n >= 15) & (g2.n >= 15)
print(cmp[ok].round(3).to_string())

print('\n=== Deepest single-height notches (candidate boom shadows) ===')
gg = g[g.n >= 30]
for h in H:
    others = [f's{k}' for k in H if k != h]
    excess = gg[f's{h}'] - gg[others].mean(axis=1)   # this height vs the others
    worst = excess.nsmallest(4)
    print(f'  {h} m: ' + ', '.join(f'{int(d)}deg {v:+.3f}' for d, v in worst.items()))
print('\n  (negative = this height reads low relative to the other two, after')
print('   removing the common shear signal -> shadow band. Boom points ~180 deg away.)')
