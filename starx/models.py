"""Portable XGBoost bundles: native UBJ trees and JSON scalers, never uploaded pickle."""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import io
import json
from pathlib import Path
import zipfile

import numpy as np
import xgboost as xgb

ROOT = Path(__file__).resolve().parents[1]
LEADS = [6, 12, 24, 48, 72]
STATIONS = {
    '8771450': {'name': 'Galveston', 'lat': 29.310, 'lon': -94.79331, 'coast': 'Gulf'},
    '8638610': {'name': 'Sewells Point', 'lat': 36.94667, 'lon': -76.330, 'coast': 'East'},
    '8518750': {'name': 'The Battery', 'lat': 40.70060, 'lon': -74.01420, 'coast': 'East'},
    '8443970': {'name': 'Boston', 'lat': 42.35389, 'lon': -71.05028, 'coast': 'East'},
    '8534720': {'name': 'Atlantic City', 'lat': 39.355, 'lon': -74.41830, 'coast': 'East'},
    '8726520': {'name': 'St. Petersburg', 'lat': 27.76060, 'lon': -82.62690, 'coast': 'Gulf'},
}


@dataclass
class Model:
    meta: dict
    boosters: list

    @property
    def features(self):
        return self.meta['features']

    def scaled(self, X):
        a = np.asarray(X, dtype=np.float32)
        if a.ndim != 2 or a.shape[1] != len(self.features) or not np.isfinite(a).all():
            raise ValueError('Inputs must contain all 487 finite predictors in the saved order.')
        # StandardScaler preserves float32 and rounds after each in-place step.
        # The two-step order matters for values close to a tree split threshold.
        a = a.copy(order='K')
        a -= np.asarray(self.meta['x_mean'], dtype=np.float32)
        a /= np.asarray(self.meta['x_scale'], dtype=np.float32)
        return a

    def predict(self, X, rounds=None):
        dm = xgb.DMatrix(self.scaled(X))
        p = np.column_stack([b.predict(dm, iteration_range=(0, int(rounds)) if rounds else (0, 0))
                             for b in self.boosters])
        p *= np.asarray(self.meta['y_scale'], dtype=np.float32)
        p += np.asarray(self.meta['y_mean'], dtype=np.float32)
        return p.astype(np.float32)

    def clone(self):
        return Model(json.loads(json.dumps(self.meta)), [b.copy() for b in self.boosters])

    def explain(self, X):
        dm = xgb.DMatrix(self.scaled(X))
        values, bases = [], []
        for i, booster in enumerate(self.boosters):
            v = booster.predict(dm, pred_contribs=True, approx_contribs=False)
            values.append(v[:, :-1] * self.meta['y_scale'][i])
            bases.append(v[:, -1] * self.meta['y_scale'][i] + self.meta['y_mean'][i])
        return np.asarray(values), np.asarray(bases)


def validate_meta(meta):
    required = {'features', 'targets', 'x_mean', 'x_scale', 'y_mean', 'y_scale', 'parameters', 'rounds'}
    if not required.issubset(meta):
        raise ValueError('Incomplete STARX model manifest.')
    if meta.get('format') != 'starx-native-v1' or len(meta['features']) != 487 or len(set(meta['features'])) != 487:
        raise ValueError('Unsupported STARX model schema.')
    if meta['targets'] != [f'surge_{h}h' for h in LEADS]:
        raise ValueError('The five target horizons do not match STARX.')
    for key, length in [('x_mean', 487), ('x_scale', 487), ('y_mean', 5), ('y_scale', 5)]:
        a = np.asarray(meta[key], dtype=float)
        if a.shape != (length,) or not np.isfinite(a).all() or ('scale' in key and np.any(a <= 0)):
            raise ValueError(f'Invalid scaler: {key}.')


def load_model(model_id):
    if not model_id.replace('_', '').isalnum():
        raise ValueError('Invalid model identifier.')
    folder = ROOT / 'assets' / 'models' / model_id
    meta = json.loads((folder / 'manifest.json').read_text())
    validate_meta(meta)
    boosters = []
    for h in LEADS:
        b = xgb.Booster(params={'nthread': 2})
        b.load_model(bytearray((folder / f'{h}.ubj').read_bytes()))
        boosters.append(b)
    return Model(meta, boosters)


def dump_model(model):
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, 'w', zipfile.ZIP_DEFLATED) as z:
        z.writestr('manifest.json', json.dumps(model.meta, indent=2, default=str))
        for h, b in zip(LEADS, model.boosters):
            z.writestr(f'{h}.ubj', bytes(b.save_raw(raw_format='ubj')))
    return buffer.getvalue()


def read_model(data):
    if len(data) > 75 * 1024**2:
        raise ValueError('Model archive exceeds 75 MB.')
    with zipfile.ZipFile(io.BytesIO(data)) as z:
        expected = {'manifest.json', *[f'{h}.ubj' for h in LEADS]}
        if set(z.namelist()) != expected or len(z.infolist()) != 6:
            raise ValueError('Upload a STARX model ZIP with exactly one manifest and five UBJ models.')
        if sum(x.file_size for x in z.infolist()) > 150 * 1024**2:
            raise ValueError('Model archive is too large when expanded.')
        meta = json.loads(z.read('manifest.json'))
        validate_meta(meta)
        boosters = []
        for h in LEADS:
            raw = z.read(f'{h}.ubj')
            if meta.get('model_sha256', {}).get(str(h)) not in (None, hashlib.sha256(raw).hexdigest()):
                raise ValueError('Model checksum mismatch.')
            b = xgb.Booster(params={'nthread': 2})
            b.load_model(bytearray(raw))
            if b.num_features() != 487 or b.num_boosted_rounds() > 10000:
                raise ValueError('Model dimensions exceed the STARX limits.')
            boosters.append(b)
    return Model(meta, boosters)


def group_for_feature(name):
    if name.startswith('surge_'):
        return 'Recent surge'
    if name.startswith('month_'):
        return 'Seasonality'
    phase = 'history' if '_lb' in name else 'future'
    if name.startswith('tide_'):
        return 'Astronomical tide'
    if name.startswith('wind_speed_'):
        return f'Wind speed · {phase}'
    if name.startswith(('wind_sin_', 'wind_cos_')):
        return f'Wind direction · {phase}'
    return f'Pressure · {phase}'
