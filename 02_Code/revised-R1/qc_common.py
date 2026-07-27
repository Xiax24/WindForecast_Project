#!/usr/bin/env python3
"""
qc_common.py  —  Shared QC / sector-masking for the 70m-only re-analysis.
v2: flatline (stuck-sensor) removal added; month exclusion disabled.

Diagnosed instrument issues (for SI):
  WIND VANES
  * 10 m: quantised to 16 discrete bins (23.93 deg apart); ~30% of records
    pinned at 0/359 deg -> unusable as a feature or as a screening variable.
  * 30 m: ~+12.2 deg installation offset (monthly median stable to <4 deg over
    18 months); 2021-11 corrupted (monthly median offset 120.5 deg).
  * 50 m: ~-6.5 deg installation offset.
  * 70 m: clean (modal direction 85 deg, no pinning, no quantisation) and is the
    variable the operational WDA strategy already uses -> adopt as the single
    sector-defining variable.

  ANEMOMETERS
  * Monthly median shear ratios stable across the record at all heights;
    2021-11 wind SPEED is normal -> no month exclusion needed.
  * 10 m: ~1,282 records (2.45%) in flatline segments >2 h, longest 200 steps
    (50 h) at non-zero speed -> stuck cup, removed here. The other heights show
    only one 181-step segment, common to all four levels (mast outage).
"""

import numpy as np
import pandas as pd

# ----------------------------------------------------------------------------
# Configuration
# ----------------------------------------------------------------------------
SECTOR_FREE = (225, 315)   # westerly  -> free-stream
SECTOR_WAKE = (45, 135)    # easterly  -> wake-influenced

WD_SECTOR_COL = 'obs_wind_direction_70m'

WD_BAD_COLS = ['obs_wind_direction_10m',
               'obs_wind_direction_30m',
               'obs_wind_direction_50m']

WS_COLS = ['obs_wind_speed_10m', 'obs_wind_speed_30m',
           'obs_wind_speed_50m', 'obs_wind_speed_70m']

# 2021-11 vane failure only; wind speed is clean -> keep the month.
EXCLUDE_MONTHS = ['2021-11']
APPLY_MONTH_EXCLUSION = False

# Flatline detection
FLATLINE_MIN_STEPS = 12      # >2 h at 15-min sampling
FLATLINE_MIN_SPEED = 0.5    # ignore genuine calms

DATETIME_COL = 'datetime'


# ----------------------------------------------------------------------------
# Sector mask (70 m only)
# ----------------------------------------------------------------------------
def sector_mask(data, direction_range, wd_col=WD_SECTOR_COL):
    """Sector membership from a SINGLE height (70 m).

    Replaces the former `strict_direction_mask`, which required all four
    heights to fall inside the sector simultaneously. That criterion was
    dominated by instrument defects rather than meteorology: it discarded ~80%
    of records even though the two sectors together contain 87% of the 70 m
    wind rose, and it produced regime subsets with markedly different diurnal
    composition (22% vs 49% daytime). Under 70 m sectoring the two regimes are
    balanced (43% vs 44%).
    """
    lo, hi = direction_range
    wd = data[wd_col].values if hasattr(data, 'columns') else np.asarray(data)
    if lo > hi:
        m = (wd >= lo) | (wd <= hi)
    else:
        m = (wd >= lo) & (wd <= hi)
    return m & ~np.isnan(wd)


# ----------------------------------------------------------------------------
# Flatline (stuck sensor) removal
# ----------------------------------------------------------------------------
def flatline_mask(series, min_steps=FLATLINE_MIN_STEPS, min_val=FLATLINE_MIN_SPEED):
    """True where a value repeats identically for >= min_steps consecutive rows."""
    s = pd.Series(np.asarray(series))
    grp = (s != s.shift()).cumsum()
    size = s.groupby(grp).transform('size')
    return ((size >= min_steps) & s.notna() & (s > min_val)).values


def remove_flatlines(df, cols=None, verbose=True):
    cols = WS_COLS if cols is None else cols
    bad = np.zeros(len(df), bool)
    for c in cols:
        if c not in df.columns:
            continue
        m = flatline_mask(df[c].values)
        if verbose and m.sum():
            print(f"  [QC] flatline {c}: {m.sum()} rows")
        bad |= m
    if verbose:
        print(f"  [QC] flatline union: {bad.sum()} rows removed")
    return df[~bad].copy()


def apply_month_exclusion(df, months=None, dt_col=DATETIME_COL, verbose=True):
    months = EXCLUDE_MONTHS if months is None else months
    if not APPLY_MONTH_EXCLUSION or not months or dt_col not in df.columns:
        return df
    t = pd.to_datetime(df[dt_col])
    drop = t.dt.to_period('M').astype(str).isin(months)
    if verbose:
        print(f"  [QC] month exclusion {months}: dropped {drop.sum()} rows")
    return df[~drop].copy()


# ----------------------------------------------------------------------------
# QC entry points
# ----------------------------------------------------------------------------
def qc_power_only(df, dt_col=DATETIME_COL, verbose=True):
    """QC for CORRELATION comparisons (Figure 2).

    Height-neutral by design: NO wind-speed range filter. A 3-25 m/s cut
    imposed on 70 m truncates the 70 m range directly but the 10 m range only
    indirectly (r ~ 0.87), i.e. range restriction that systematically depresses
    the 70 m correlation and inflates the very 10 m vs 70 m contrast under test.
    """
    n0 = len(df)
    d = df[df['power'] >= 0].copy()
    d = d.dropna(subset=['power'])
    d = remove_flatlines(d, verbose=verbose)
    d = apply_month_exclusion(d, dt_col=dt_col, verbose=verbose)
    if verbose:
        print(f"  [QC power-only] {n0} -> {len(d)}")
    return d


def qc_operating_range(df, ws_col='obs_wind_speed_70m', lo=3.0, hi=25.0,
                       dt_col=DATETIME_COL, verbose=True):
    """QC for the RECONSTRUCTION / FORECAST experiments (Figures 3-4).

    Here the 3-25 m/s cut on 70 m is legitimate: it defines the turbine
    operating envelope (cut-in / cut-out) and is applied identically to the HH,
    SR and ER configurations, which are compared against each other rather than
    against individual heights.
    """
    n0 = len(df)
    d = df[df['power'] >= 0].copy()
    d = d.dropna(subset=['power', ws_col])
    d = remove_flatlines(d, verbose=verbose)
    d = d[(d[ws_col] >= lo) & (d[ws_col] <= hi)].copy()
    d = apply_month_exclusion(d, dt_col=dt_col, verbose=verbose)
    if verbose:
        print(f"  [QC operating-range {lo}-{hi} @ {ws_col}] {n0} -> {len(d)}")
    return d


def report_sectors(df, tag=''):
    mf = sector_mask(df, SECTOR_FREE)
    mw = sector_mask(df, SECTOR_WAKE)
    n = len(df)
    print(f"  [sectors {tag}] N={n} | free={mf.sum()} ({100*mf.sum()/n:.1f}%)"
          f" | wake={mw.sum()} ({100*mw.sum()/n:.1f}%)")
    if DATETIME_COL in df.columns:
        t = pd.to_datetime(df[DATETIME_COL])
        day = (t.dt.hour >= 8) & (t.dt.hour < 18)
        print(f"      daytime fraction: free={day[mf].mean():.2f}"
              f"  wake={day[mw].mean():.2f}")
    return mf, mw
