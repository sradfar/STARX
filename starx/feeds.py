"""NOAA observations and tides plus GFS atmospheric forcing, in the paper's schema."""
from datetime import datetime, timezone
import math
import time

import numpy as np
import requests

from .models import STATIONS

NOAA = 'https://api.tidesandcurrents.noaa.gov/api/prod/datagetter'
WEATHER = 'https://api.open-meteo.com/v1/forecast'


def timestamp(value):
    return int(datetime.fromisoformat(value.replace('Z', '+00:00')).replace(tzinfo=timezone.utc).timestamp())


def noaa(station, product, start, end):
    params = dict(station=station, product=product, datum='MLLW', time_zone='gmt', units='metric',
                  format='json', application='STARX',
                  begin_date=datetime.fromtimestamp(start, timezone.utc).strftime('%Y%m%d %H:%M'),
                  end_date=datetime.fromtimestamp(end, timezone.utc).strftime('%Y%m%d %H:%M'))
    if product == 'predictions':
        params['interval'] = 'h'
    response = requests.get(NOAA, params=params, timeout=40)
    response.raise_for_status()
    payload = response.json()
    if 'error' in payload:
        raise ValueError(f'NOAA {product}: {payload["error"].get("message")}')
    return payload.get('predictions' if product == 'predictions' else 'data', [])


def qc_observations(water, tides, now):
    bins = {}
    for row in water:
        try:
            t, value = timestamp(row['t']), float(row['v'])
            flags = [v.strip() for v in row.get('f', '').split(',') if v.strip()]
            quality = row.get('q')
            if not math.isfinite(value) or len(flags) != 4 or quality not in ('p', 'v'):
                continue
            if any(v != '0' for v in flags[1:]) or int(flags[0]) < 0 or (quality == 'v' and flags[0] != '0'):
                continue
            bins.setdefault(t//3600*3600, {})[t] = value
        except (KeyError, ValueError, TypeError):
            continue
    out = {}
    for hour, values in bins.items():
        required = [hour+360*i for i in range(10)]
        if hour+3600 > now or hour not in tides or any(t not in values for t in required):
            continue
        surge = sum(values[t] for t in required)/10 - tides[hour]
        if -2 <= surge <= 6 and math.isfinite(surge):
            out[hour] = surge
    return out


def build_vector(columns, origin, residual, weather, tides):
    values = {}
    for step in range(25):
        t = origin+(step-24)*3600
        if t not in residual:
            raise ValueError('NOAA has an incomplete or flagged surge-history hour. Refresh when complete data arrive.')
        values[f'surge_lb{step:02d}'] = residual[t]
        for key in ['wind_speed', 'wind_sin', 'wind_cos', 'pressure']:
            values[f'{key}_lb{step:02d}'] = weather.get(t, {}).get(key)
    for step in range(1, 73):
        t = origin+step*3600
        for key in ['wind_speed', 'wind_sin', 'wind_cos', 'pressure']:
            values[f'{key}_fc{step:02d}'] = weather.get(t, {}).get(key)
        values[f'tide_fc{step:02d}'] = tides.get(t)
    angle = 2*math.pi*datetime.fromtimestamp(origin, timezone.utc).month/12
    values.update(month_sin=math.sin(angle), month_cos=math.cos(angle))
    if set(values) != set(columns) or len(columns) != 487:
        raise ValueError('Forecast inputs do not match the saved 487-predictor contract.')
    if any(values[k] is None or not math.isfinite(values[k]) for k in columns):
        raise ValueError('Atmospheric or tidal inputs are incomplete; no prediction was filled with defaults.')
    return np.asarray([values[k] for k in columns], dtype=np.float32)


def collect(station, columns, now=None):
    now = float(now if now is not None else time.time())
    latest = int(now)//3600*3600-3600
    start = latest-8*86400
    tides = {timestamp(r['t']): float(r['v']) for r in noaa(station, 'predictions', start, latest+73*3600)}
    residual = qc_observations(noaa(station, 'water_level', start, latest+3600), tides, now)
    origins = [t for t in residual if latest-3*3600 <= t <= latest and all(t-i*3600 in residual for i in range(25))]
    if not origins:
        raise ValueError('No recent complete 25-hour NOAA history is available. Forecasting will resume when the input record is complete.')
    origin = max(origins)
    lat, lon = STATIONS[station]['lat'], STATIONS[station]['lon']
    r = requests.get(WEATHER, params=dict(latitude=lat, longitude=lon, hourly='wind_speed_10m,wind_direction_10m,pressure_msl',
                    wind_speed_unit='ms', timezone='GMT', past_days=2, forecast_days=5,
                    models='gfs_global', cell_selection='nearest'), timeout=40)
    r.raise_for_status()
    payload = r.json()
    if payload.get('hourly_units', {}).get('wind_speed_10m') != 'm/s' or payload.get('hourly_units', {}).get('pressure_msl') != 'hPa':
        raise ValueError('Unexpected GFS units.')
    hourly = payload['hourly']
    weather = {}
    for i, label in enumerate(hourly['time']):
        try:
            speed, direction, pressure = [float(hourly[k][i]) for k in ['wind_speed_10m', 'wind_direction_10m', 'pressure_msl']]
            if not (0 <= speed <= 120 and 0 <= direction <= 360 and 800 <= pressure <= 1100):
                continue
            angle = math.radians(direction)
            weather[timestamp(label)] = dict(wind_speed=speed, wind_sin=-math.sin(angle), wind_cos=-math.cos(angle), pressure=pressure)
        except (ValueError, TypeError, IndexError):
            continue
    return dict(station=station, origin=origin, fetched_at=now, residual=residual, tides=tides, weather=weather,
                X=build_vector(columns, origin, residual, weather, tides),
                provenance=dict(observations='NOAA CO-OPS', forcing='NOAA GFS via Open-Meteo',
                                tide_datum='MLLW', wind_units='m/s', pressure_units='hPa',
                                grid_latitude=payload.get('latitude'), grid_longitude=payload.get('longitude')))
