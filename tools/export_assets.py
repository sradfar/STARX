"""One-time export from trusted local paper files; original files stay unchanged."""
import hashlib
import json
from pathlib import Path
import pickle
import shutil
import sys

import numpy as np
import pandas as pd

APP = Path(__file__).resolve().parents[1]
WORK = APP.parent
sys.path.insert(0, str(WORK))
sys.path.insert(0, str(APP))
from transferability_common_v1 import install_pickle_compatibility_classes, predict_bundle, read_feature_period, load_config
from starx.models import Model

RUN = WORK / 'npj_runs/results/reviewer_selected_adaptation_clean_v1'
OUT = APP / 'assets'
OUT.mkdir(exist_ok=True)
install_pickle_compatibility_classes()
catalog = []


def export(path, mid, kind, sid, year=None):
    with path.open('rb') as f:
        bundle = pickle.load(f)
    folder = OUT / 'models' / mid
    folder.mkdir(parents=True, exist_ok=True)
    scaler = bundle['scaler']
    parameters = bundle['model'].base_model.get_params()
    keys = ['max_depth', 'learning_rate', 'subsample', 'colsample_bytree', 'min_child_weight', 'reg_alpha', 'reg_lambda']
    meta = dict(format='starx-native-v1', id=mid, kind=kind, target_id=sid, target_year=year,
                features=list(bundle['feature_cols']), targets=list(bundle['target_cols']),
                x_mean=scaler.X_scaler.mean_.tolist(), x_scale=scaler.X_scaler.scale_.tolist(),
                y_mean=scaler.y_scaler.mean_.tolist(), y_scale=scaler.y_scaler.scale_.tolist(),
                parameters={k: parameters[k] for k in keys},
                rounds=[e.get_booster().num_boosted_rounds() for e in bundle['model'].estimators_],
                objective=dict(tau1=.8, tau2=.7, w1=.5, w2=.5),
                selection=bundle.get('selection_provenance'), model_sha256={},
                source_file_sha256=hashlib.sha256(path.read_bytes()).hexdigest())
    if kind == 'source':
        meta['source_id'] = sid
    elif kind == 'adapted':
        meta['source_id'] = meta['selection']['adapted']['source_id']
    boosters = []
    for h, e in zip([6, 12, 24, 48, 72], bundle['model'].estimators_):
        b = e.get_booster().copy()
        b.set_param({'nthread': 2})
        b.save_model(folder / f'{h}.ubj')
        meta['model_sha256'][str(h)] = hashlib.sha256((folder / f'{h}.ubj').read_bytes()).hexdigest()
        boosters.append(b)
    (folder / 'manifest.json').write_text(json.dumps(meta, indent=2, default=str))
    # A conversion must preserve the complete preprocessing-and-prediction pipeline.
    X = np.tile(np.asarray(meta['x_mean'], dtype=np.float32), (3, 1))
    X[1] += np.asarray(meta['x_scale'], dtype=np.float32) * .1
    X[2] -= np.asarray(meta['x_scale'], dtype=np.float32) * .1
    error = float(np.max(np.abs(predict_bundle(bundle, X) - Model(meta, boosters).predict(X))))
    if error > 2e-6:
        raise AssertionError((mid, error))
    catalog.append(dict(id=mid, kind=kind, target_id=sid, year=year,
                        source_id=meta.get('source_id'), conversion_max_error_m=error))
    print(f'Exported {mid}: parity error {error:.2g} m', flush=True)


stations = ['8771450', '8638610', '8518750', '8443970', '8534720', '8726520']
for sid in stations:
    export(RUN / f'source_{sid}.pkl', f'source_{sid}', 'source', sid)
    for year in [2016, 2018]:
        for name in ['local', 'adapted']:
            export(RUN / f'{year}_{sid}/selected_{name}.pkl', f'{name}_{year}_{sid}', name, sid, year)

data = OUT / 'data'
data.mkdir(exist_ok=True)
for name in ['aggregate_performance.csv', 'aggregate_timing.csv', 'station_performance.csv',
             'test_metrics.csv', 'timings.csv', 'validation_candidates.csv', 'revision_facts.json', 'source_audit.json']:
    shutil.copy2(RUN / name, data / name)
for name in ['candidate_metrics.csv', 'reference_metrics.csv', 'transfer_skill_retention_pairs.csv',
             'shap_groups.csv', 'shap_alignment.csv', 'pair_diagnostics_summary.csv',
             'diagnostic_correlations.json', 'domain_distance.csv', 'budget_summary.csv',
             'episode_summary.csv', 'episode_coverage.csv', 'case_study.json', 'case_trace.csv',
             'case_group_profiles.csv', 'calibration_metrics.csv']:
    shutil.copy2(RUN / 'supporting' / name, data / name)
for year in [2016, 2018]:
    for sid in stations:
        folder = RUN / f'{year}_{sid}'
        for name in ['thresholds.csv', 'choices.json', 'chronology.json', 'test_predictions.csv.gz']:
            shutil.copy2(folder / name, data / f'{year}_{sid}_{name}')

# A real, small target-record example: the first 3,000 complete 2016 Boston origins.
# These are training-year observations, never the paper's independent test inputs.
schema = json.loads((OUT / 'models/source_8443970/manifest.json').read_text())
cols = ['t0', *schema['features'], *schema['targets'], 'sample_weight']
config, _ = load_config(WORK / 'transferability_config_v1.json')
frame = read_feature_period(WORK, config, '8443970', cols[1:], start_year=2016, end_year=2016, max_rows=3200)
frame['t0'] = pd.to_datetime(frame.t0)
frame = frame.loc[(frame.t0 - pd.Timedelta(hours=24) >= pd.Timestamp('2016-01-01'))
                 & (frame.t0 + pd.Timedelta(hours=72) < pd.Timestamp('2017-01-01'))].head(3000)
frame.to_csv(data / 'example_boston_2016.csv.gz', index=False, compression='gzip')
(OUT / 'catalog.json').write_text(json.dumps(catalog, indent=2))
(OUT / 'schema.json').write_text(json.dumps(dict(features=schema['features'], targets=schema['targets']), indent=2))
print(f'{len(catalog)} models; example {len(frame)} rows; assets {sum(p.stat().st_size for p in OUT.rglob("*") if p.is_file()) / 1e6:.1f} MB')
