"""Build complete paper-schema examples from uploaded hourly local data."""
import io

import numpy as np
import pandas as pd

from .models import LEADS

TEMPLATE = ('time_utc,surge_m,tide_m,wind_u_m_s,wind_v_m_s,pressure_hpa\n')


def hourly_to_training(data, features):
    raw = pd.read_csv(io.BytesIO(data), compression='gzip' if data[:2] == b'\x1f\x8b' else None, nrows=25100)
    if len(raw) > 25096:
        raise ValueError('The free app accepts up to 25,096 raw hourly records (25,000 complete forecast windows).')
    if 'time_utc' not in raw:
        raise ValueError('Hourly data need a time_utc column.')
    t = pd.to_datetime(raw.pop('time_utc'), utc=True, errors='coerce').dt.tz_localize(None)
    if t.isna().any() or t.duplicated().any() or (t.dt.minute.ne(0) | t.dt.second.ne(0)).any():
        raise ValueError('Use unique, whole-hour UTC timestamps.')
    raw.index = pd.DatetimeIndex(t)
    raw = raw.sort_index().apply(pd.to_numeric, errors='coerce')
    if 'surge_m' not in raw and {'water_level_m', 'tide_m'}.issubset(raw):
        raw['surge_m'] = raw.water_level_m-raw.tide_m
    if {'wind_u_m_s', 'wind_v_m_s'}.issubset(raw):
        speed = np.sqrt(raw.wind_u_m_s**2+raw.wind_v_m_s**2)
        # Match atan2(u,v), including the original calm-wind convention.
        angle = np.arctan2(raw.wind_u_m_s, raw.wind_v_m_s)
        raw['wind_speed'] = speed
        raw['wind_sin'], raw['wind_cos'] = np.sin(angle), np.cos(angle)
    elif {'wind_speed_m_s', 'wind_direction_from_deg'}.issubset(raw):
        angle = np.radians(raw.wind_direction_from_deg)
        raw['wind_speed'] = raw.wind_speed_m_s
        raw['wind_sin'], raw['wind_cos'] = -np.sin(angle), -np.cos(angle)
    else:
        raise ValueError('Supply u/v wind components, or wind_speed_m_s and meteorological wind_direction_from_deg.')
    required = ['surge_m', 'tide_m', 'wind_speed', 'wind_sin', 'wind_cos', 'pressure_hpa']
    if set(required)-set(raw):
        raise ValueError('Hourly data need surge_m (or water_level_m minus tide_m), tide_m and pressure_hpa.')
    if len(raw) < 500:
        raise ValueError('At least 500 hourly observations are needed to construct a useful training/validation record.')
    if (raw.index.max()-raw.index.min()).total_seconds()/3600 > 50000:
        raise ValueError('Choose a smaller, reasonably continuous fitting budget.')
    raw = raw.reindex(pd.date_range(raw.index.min(), raw.index.max(), freq='h'))
    if ((raw.pressure_hpa.dropna() < 800) | (raw.pressure_hpa.dropna() > 1100)).any():
        raise ValueError('Pressure must be in hPa (800–1100), not Pa.')
    if ((raw.wind_speed.dropna() < 0) | (raw.wind_speed.dropna() > 120)).any():
        raise ValueError('Wind speed must be in m/s.')
    values = {}
    for i in range(25):
        lag = 24-i
        values[f'surge_lb{i:02d}'] = raw.surge_m.shift(lag)
        for key, column in [('wind_speed', 'wind_speed'), ('wind_sin', 'wind_sin'), ('wind_cos', 'wind_cos'), ('pressure', 'pressure_hpa')]:
            values[f'{key}_lb{i:02d}'] = raw[column].shift(lag)
    for h in range(1, 73):
        for key, column in [('wind_speed', 'wind_speed'), ('wind_sin', 'wind_sin'), ('wind_cos', 'wind_cos'), ('pressure', 'pressure_hpa')]:
            values[f'{key}_fc{h:02d}'] = raw[column].shift(-h)
        values[f'tide_fc{h:02d}'] = raw.tide_m.shift(-h)
    values['month_sin'] = np.sin(2*np.pi*raw.index.month/12)
    values['month_cos'] = np.cos(2*np.pi*raw.index.month/12)
    for h in LEADS:
        values[f'surge_{h}h'] = raw.surge_m.shift(-h)
    frame = pd.DataFrame(values, index=raw.index)
    # A full observed target window must also be complete, not just the five lead values.
    complete_observed = raw.surge_m.notna().rolling(73, min_periods=73).sum().shift(-72).eq(73)
    frame = frame.replace([np.inf, -np.inf], np.nan).dropna()
    frame = frame.loc[complete_observed.reindex(frame.index).fillna(False)]
    if set(features)-set(frame):
        raise ValueError('Hourly inputs do not construct the required feature schema.')
    frame = frame[[*features, *[f'surge_{h}h' for h in LEADS]]].astype(np.float32)
    frame.insert(0, 't0', frame.index)
    if len(frame) > 25000:
        raise ValueError('More than 25,000 complete examples were constructed.')
    return frame.reset_index(drop=True)
