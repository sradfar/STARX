"""Author-only verification against the manuscript's original Python pipeline."""
import copy
import json
from pathlib import Path
import pickle
import sys

import numpy as np
import xgboost as xgb

APP = Path(__file__).resolve().parents[1]
WORK = APP.parent
sys.path.insert(0, str(APP))
sys.path.insert(0, str(WORK))
from transferability_common_v1 import install_pickle_compatibility_classes, predict_bundle, load_config
from transferability_02_low_data_adaptation_v1 import adapt_source_bundle
from starx.models import load_model
from starx.adaptation import fit_model, read_training_csv

install_pickle_compatibility_classes()
with (WORK/'npj_runs/results/reviewer_selected_adaptation_clean_v1/source_8534720.pkl').open('rb') as f:
    original = pickle.load(f)
source = load_model('source_8534720')
schema = json.loads((APP/'assets/schema.json').read_text())
frame = read_training_csv((APP/'assets/data/example_boston_2016.csv.gz').read_bytes(), schema['features'])
small = frame.iloc[:350]
X = small[schema['features']].to_numpy(np.float32)
y = small[schema['targets']].to_numpy(np.float32)
cfg, _ = load_config(WORK/'transferability_config_v1.json')
cfg = copy.deepcopy(cfg)
cfg['xgboost'].update(continuation_rounds=3, n_jobs=2)
for estimator in original['model'].estimators_:
    b = estimator.get_booster()
    b.set_param({'seed': 0, 'random_state': 0, 'nthread': 2})
    b.predict(xgb.DMatrix(original['scaler'].transform_X(X[:1])))
paper = adapt_source_bundle(original, X, y, None, cfg)
paper_repeat = adapt_source_bundle(original, X, y, None, cfg)
portable = fit_model(schema, small, 3, source=source)
portable_repeat = fit_model(schema, small, 3, source=source)
Xt = frame[schema['features']].iloc[500:550].to_numpy(np.float32)
error = float(np.abs(predict_bundle(paper, Xt)-portable.predict(Xt)).max())
assert np.array_equal(portable.predict(Xt), portable_repeat.predict(Xt)), 'Repeated app updates differ'
if error >= 2e-5:
    diagnostic = dict(source_prediction_error=float(np.abs(predict_bundle(original, Xt)-source.predict(Xt)).max()),
                      scaled_input_error=float(np.abs(original['scaler'].transform_X(X)-source.scaled(X)).max()),
                      objective_parameters=vars(original['model'].wqe_obj),
                      paper_parameters=paper['model'].estimators_[0].get_params(),
                      paper_config=json.loads(paper['model'].estimators_[0].get_booster().save_config()),
                      native_config=json.loads(portable.boosters[0].save_config()),
                      paper_tree=json.loads(paper['model'].estimators_[0].get_booster().get_dump(dump_format='json')[600]),
                      native_tree=json.loads(portable.boosters[0].get_dump(dump_format='json')[600]))
    (APP/'test-output/parity_diagnostic.json').write_text(json.dumps(diagnostic, indent=2, default=str))
    print({k: diagnostic[k] for k in ['source_prediction_error', 'scaled_input_error']})
    print('objective:', diagnostic['objective_parameters'])
    def flatten(value, prefix=''):
        output = {}
        if isinstance(value, dict):
            for key, item in value.items():
                output.update(flatten(item, prefix+'/'+key))
        else:
            output[prefix] = value
        return output
    old, new = flatten(diagnostic['paper_config']), flatten(diagnostic['native_config'])
    print('config differences:', [(k, old[k], new.get(k)) for k in old if old[k] != new.get(k)])
    print('first added splits:', [(diagnostic[k].get('split'), diagnostic[k].get('split_condition')) for k in ['paper_tree', 'native_tree']])
    ys = y.copy()
    ys -= np.asarray(source.meta['y_mean'], dtype=np.float32)
    ys /= np.asarray(source.meta['y_scale'], dtype=np.float32)
    print('target scale error:', np.abs(ys-original['scaler'].transform_y(y)).max())
    print('training input prediction error:', np.abs(source.predict(X)-predict_bundle(original,X)).max())
    print('original training repeat error:', np.abs(predict_bundle(paper,Xt)-predict_bundle(paper_repeat,Xt)).max())
    from starx.adaptation import sklearn_objective
    pp = np.linspace(-3,3,350).astype(np.float32)
    ga, ha = original['model'].wqe_obj(ys[:,0], pp)
    gb, hb = sklearn_objective(ys[:,0], pp)
    print('gradient/hessian mismatch:', np.abs(ga-gb).max(), np.abs(ha-hb).max())
assert error < 2e-5, error
print(json.dumps(dict(training_pipeline_max_difference_m=error, retained_trees=600, added_rounds=3)))
