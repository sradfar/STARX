"""Chronological selection and continuation using the paper's WQE objective."""
from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import io
import json
import time

import numpy as np
import pandas as pd
import xgboost as xgb

from .models import Model, LEADS

LOCAL_CONFIGS = {
    'Published local': dict(max_depth=6, learning_rate=.05, subsample=.8, colsample_bytree=.8,
                            min_child_weight=1, reg_alpha=0., reg_lambda=1.),
    'Source style · depth 5': dict(max_depth=5, learning_rate=.03, subsample=.8, colsample_bytree=.7,
                                 min_child_weight=5, reg_alpha=.1, reg_lambda=1.),
    'Source style · depth 6': dict(max_depth=6, learning_rate=.04, subsample=.8, colsample_bytree=.7,
                                 min_child_weight=5, reg_alpha=.1, reg_lambda=1.),
}


def read_training_csv(data, features):
    if len(data) > 75 * 1024**2:
        raise ValueError('Target record exceeds 75 MB.')
    compressed = data[:2] == b'\x1f\x8b'
    frame = pd.read_csv(io.BytesIO(data), compression='gzip' if compressed else None, nrows=25001)
    if len(frame) > 25000:
        raise ValueError('The free app accepts at most 25,000 training origins per update.')
    required = ['t0', *features, *[f'surge_{h}h' for h in LEADS]]
    missing = set(required) - set(frame.columns)
    if missing:
        raise ValueError(f'Missing {len(missing)} columns; first: {sorted(missing)[:5]}. Use the downloadable schema.')
    frame['t0'] = pd.to_datetime(frame.t0, utc=True, errors='coerce').dt.tz_localize(None)
    if frame.t0.isna().any() or frame.t0.duplicated().any():
        raise ValueError('Forecast origins must be valid, unique UTC timestamps.')
    if (frame.t0.dt.minute.ne(0) | frame.t0.dt.second.ne(0)).any():
        raise ValueError('Forecast origins must be hourly timestamps.')
    numeric = frame[required[1:]].apply(pd.to_numeric, errors='coerce')
    if not np.isfinite(numeric.to_numpy()).all():
        raise ValueError('All predictors and future observed targets must be finite. Incomplete windows cannot be fitted.')
    frame[required[1:]] = numeric.astype(np.float32)
    if frame.t0.max() + pd.Timedelta(hours=73) > pd.Timestamp.now(tz='UTC').tz_localize(None):
        raise ValueError('The uploaded record includes outcomes that have not yet completed.')
    return frame.sort_values('t0').reset_index(drop=True)


def split_record(frame, start, end, cutoff):
    start, end, cutoff = map(pd.Timestamp, [start, end, cutoff])
    if not start < cutoff < end:
        raise ValueError('Choose a validation boundary strictly inside the fitting budget.')
    full = frame.loc[(frame.t0 - pd.Timedelta(hours=24) >= start)
                     & (frame.t0 + pd.Timedelta(hours=72) < end)].copy()
    train = full.loc[full.t0 + pd.Timedelta(hours=72) < cutoff].copy()
    valid = full.loc[full.t0 >= cutoff].copy()
    if len(train) < 300 or len(valid) < 100:
        raise ValueError(f'Need ≥300 training and ≥100 validation origins after complete-window filtering; found {len(train)} and {len(valid)}.')
    return full, train, valid


def sklearn_objective(y_true, y_pred, sample_weight=None):
    eps = y_true-y_pred
    under = eps >= 0
    tau1, tau2, w1, w2 = .8, .7, .5, .5
    grad_phi1 = np.where(under, -tau1, 1.0-tau1)
    grad_phi2 = np.where(under, -2.0*tau2*eps, -2.0*(1.0-tau2)*eps)
    hess_phi2 = np.where(under, 2.0*tau2, 2.0*(1.0-tau2))
    return (w1*grad_phi1+w2*grad_phi2).astype(np.float32), (w1*1e-4+w2*hess_phi2).astype(np.float32)


def fit_model(schema, frame, rounds, params=None, source=None, threads=2):
    features = schema['features']
    X = frame[features].to_numpy(np.float32)
    y = frame[[f'surge_{h}h' for h in LEADS]].to_numpy(np.float32)
    if source is None:
        xm, xs = X.astype(float).mean(0), X.astype(float).std(0)
        ym, ys = y.astype(float).mean(0), y.astype(float).std(0)
        xs[xs == 0] = 1.
        ys[ys == 0] = 1.
        meta = dict(format='starx-native-v1', features=features, targets=[f'surge_{h}h' for h in LEADS],
                    x_mean=xm.tolist(), x_scale=xs.tolist(), y_mean=ym.tolist(), y_scale=ys.tolist(),
                    parameters=dict(params), rounds=[rounds]*5, kind='local', source_id=None,
                    objective=dict(tau1=.8, tau2=.7, w1=.5, w2=.5))
    else:
        meta = json.loads(json.dumps(source.meta))
        meta.pop('model_sha256', None)
        meta['kind'] = 'adapted'
        meta['source_id'] = source.meta.get('source_id', source.meta.get('target_id'))
    scaled_X, scaled_y = X.copy(order='K'), y.copy(order='K')
    scaled_X -= np.asarray(meta['x_mean'], dtype=np.float32)
    scaled_X /= np.asarray(meta['x_scale'], dtype=np.float32)
    scaled_y -= np.asarray(meta['y_mean'], dtype=np.float32)
    scaled_y /= np.asarray(meta['y_scale'], dtype=np.float32)
    if 'sample_weight' in frame:
        weights = frame.sample_weight.to_numpy(np.float32, copy=True)
        weights[~np.isfinite(weights) | (weights <= 0)] = 1.
    else:
        # Same capped exponential peak-surge weighting as the feature builder.
        peak = np.maximum(y.max(axis=1), 0.)
        weights = np.exp(np.minimum(3.*peak, np.log(50.))).astype(np.float32)
        weights = np.clip(weights, .2, 50.)
    weights /= weights.mean()
    parameters = {**meta['parameters'], 'tree_method': 'hist', 'random_state': 42, 'n_jobs': threads,
                  'objective': sklearn_objective, 'verbosity': 0, 'n_estimators': int(rounds)}
    boosters = []
    for i in range(5):
        warm = None
        if source:
            warm = source.boosters[i].copy()
            # Force an explicit seed transition: XGBoost otherwise can inherit
            # thread-local RNG state when continuing an already configured booster.
            warm.set_param({'seed': 0, 'random_state': 0, 'nthread': threads})
            warm.predict(xgb.DMatrix(scaled_X[:1]))
        estimator = xgb.XGBRegressor(**parameters)
        estimator.fit(scaled_X, scaled_y[:, i], sample_weight=weights, verbose=False,
                      xgb_model=warm)
        boosters.append(estimator.get_booster())
    meta['rounds'] = [b.num_boosted_rounds() for b in boosters]
    meta['fitted_through_utc'] = str(frame.t0.max() + pd.Timedelta(hours=72))
    meta['fit_first_origin_utc'] = str(frame.t0.min())
    return Model(meta, boosters)


def metrics(y, p, threshold):
    y, p = np.asarray(y, float), np.asarray(p, float)
    rmse = np.sqrt(np.mean((y-p)**2, axis=0))
    high = y >= threshold
    pred_high = p >= threshold
    hs = [float(np.sqrt(np.mean((y[high[:, i], i]-p[high[:, i], i])**2)))
          if high[:, i].any() else None for i in range(5)]
    tp = (high & pred_high).sum(0)
    fp = (~high & pred_high).sum(0)
    fn = (high & ~pred_high).sum(0)
    f1 = 2*tp / np.maximum(2*tp+fp+fn, 1)
    return dict(rmse_m=float(rmse.mean()), lead_rmse_m=rmse.tolist(),
                high_surge_rmse_m=float(np.mean([v for v in hs if v is not None])) if any(v is not None for v in hs) else None,
                high_surge_rmse_by_lead=hs, warning_f1=float(f1.mean()), high_surge_counts=high.sum(0).tolist())


def train_selected(frame, schema, sources, target_id, start, end, cutoff, progress=None):
    full, train, valid = split_record(frame, start, end, cutoff)
    threshold = np.quantile(train[[f'surge_{h}h' for h in LEADS]], .95, axis=0)
    Xv = valid[schema['features']].to_numpy(np.float32)
    yv = valid[[f'surge_{h}h' for h in LEADS]].to_numpy(np.float32)
    records = []
    best = {}
    begin = time.perf_counter()
    # Train each maximum-length ensemble once; score prefixes for the candidate rounds.
    for name, params in LOCAL_CONFIGS.items():
        if progress:
            progress(f'Fitting local configuration: {name}')
        model = fit_model(schema, train, 600, params=params)
        for rounds in [200, 400, 600]:
            score = metrics(yv, model.predict(Xv, rounds), threshold)
            records.append(dict(strategy='Local-only', candidate=name, rounds=rounds, **score))
            if 'local' not in best or score['rmse_m'] < best['local']['rmse_m']:
                best['local'] = records[-1]
    for sid, source in sources.items():
        if sid == target_id:
            continue
        if source.features != schema['features']:
            raise ValueError('Source and target feature contracts differ.')
        if progress:
            progress(f'Validating source {sid}: 25, 50 and 100 added trees')
        model = fit_model(schema, train, 100, source=source)
        initial = source.boosters[0].num_boosted_rounds()
        for rounds in [25, 50, 100]:
            score = metrics(yv, model.predict(Xv, initial+rounds), threshold)
            records.append(dict(strategy='Adapted', candidate=sid, rounds=rounds, **score))
            if 'adapted' not in best or score['rmse_m'] < best['adapted']['rmse_m']:
                best['adapted'] = records[-1]
    selection_seconds = time.perf_counter()-begin
    if progress:
        progress('Refitting the selected strategies on the complete target budget')
    t = time.perf_counter()
    local = fit_model(schema, full, best['local']['rounds'], params=LOCAL_CONFIGS[best['local']['candidate']])
    local_seconds = time.perf_counter()-t
    t = time.perf_counter()
    adapted = fit_model(schema, full, best['adapted']['rounds'], source=sources[best['adapted']['candidate']])
    adapted_seconds = time.perf_counter()-t
    report = dict(target_id=target_id, created_utc=datetime.now(timezone.utc).isoformat(),
                  budget_start=str(start), budget_end_exclusive=str(end), validation_start=str(cutoff),
                  full_rows=len(full), train_rows=len(train), validation_rows=len(valid),
                  last_training_target=str(train.t0.max()+pd.Timedelta(hours=72)),
                  first_validation_origin=str(valid.t0.min()),
                  threshold_m=threshold.tolist(), choices=best,
                  selection_seconds=selection_seconds, local_final_fit_seconds=local_seconds,
                  adaptation_final_fit_seconds=adapted_seconds,
                  candidate_records=records,
                  record_sha256=hashlib.sha256(pd.util.hash_pandas_object(full, index=False).values.tobytes()).hexdigest(),
                  performance_scope='Target-record validation used for selection; not an independent test estimate.')
    for name, model in [('local', local), ('adapted', adapted)]:
        model.meta.update(target_id=target_id, selection=report,
                          id=f'user_{name}_{report["record_sha256"][:12]}')
    return local, adapted, report


def continuation_check(incumbent, frame, schema, progress=None):
    """Fresh live evidence: select continuation on validation; gate on untouched last 20%."""
    if len(frame) < 2000:
        raise ValueError('At least 2,000 completed issued forecasts are needed for a monitored update.')
    frame = frame.sort_values('t0').reset_index(drop=True)
    first_valid = frame.t0.iloc[int(len(frame)*.6)]
    first_audit = frame.t0.iloc[int(len(frame)*.8)]
    trained_to = pd.Timestamp(incumbent.meta.get('fitted_through_utc', '2015-12-31 23:00'))
    if first_valid <= trained_to:
        raise ValueError('Validation overlaps data already used by the current model.')
    train = frame.loc[frame.t0+pd.Timedelta(hours=72) < first_valid]
    valid = frame.loc[(frame.t0 >= first_valid) & (frame.t0+pd.Timedelta(hours=72) < first_audit)]
    audit = frame.loc[frame.t0 >= first_audit]
    if min(len(train), len(valid), len(audit)) < 100:
        raise ValueError('Not enough separated evidence after the two 72-hour purges.')
    threshold = np.quantile(train[[f'surge_{h}h' for h in LEADS]], .95, axis=0)
    if progress:
        progress('Selecting continuation length on chronological validation')
    model = fit_model(schema, train, 100, source=incumbent)
    initial = incumbent.boosters[0].num_boosted_rounds()
    Xv, yv = valid[schema['features']].to_numpy(np.float32), valid[[f'surge_{h}h' for h in LEADS]].to_numpy(np.float32)
    chosen = min([25, 50, 100], key=lambda r: metrics(yv, model.predict(Xv, initial+r), threshold)['rmse_m'])
    candidate = Model(json.loads(json.dumps(model.meta)), [b[:initial+chosen] for b in model.boosters])
    candidate.meta['rounds'] = [initial+chosen]*5
    candidate.meta['kind'] = incumbent.meta['kind']
    Xa, ya = audit[schema['features']].to_numpy(np.float32), audit[[f'surge_{h}h' for h in LEADS]].to_numpy(np.float32)
    old, new = metrics(ya, incumbent.predict(Xa), threshold), metrics(ya, candidate.predict(Xa), threshold)
    normal = ya < threshold
    normal_old = np.sqrt(np.mean((ya[normal]-incumbent.predict(Xa)[normal])**2)) if normal.any() else None
    normal_new = np.sqrt(np.mean((ya[normal]-candidate.predict(Xa)[normal])**2)) if normal.any() else None
    enough_high = min(new['high_surge_counts']) >= 5
    accepted = bool(enough_high and normal_old is not None and old['high_surge_rmse_m'] is not None
                    and new['high_surge_rmse_m'] <= old['high_surge_rmse_m']*.99
                    and new['rmse_m'] <= old['rmse_m']*1.005 and normal_new <= normal_old*1.01)
    report = dict(accepted=accepted, selected_added_rounds=chosen, incumbent=old, candidate=new,
                  training_rows=len(train), validation_rows=len(valid), audit_rows=len(audit),
                  validation_start=str(first_valid), audit_start=str(first_audit),
                  consumed_through_utc=str(frame.t0.max()),
                  reason='Improved high-surge error with overall and normal-condition checks passed.' if accepted
                  else 'Keep the current model: acceptance checks or high-surge evidence were insufficient.')
    candidate.meta.update(id=f'live_{datetime.now(timezone.utc).strftime("%Y%m%d%H%M%S")}', update_audit=report)
    # Deploy the tested model as-is. Refitting on the audit would invalidate the gate.
    return candidate, report
