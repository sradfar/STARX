"""Execute the complete candidate-selection path on the included real example."""
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from starx.models import ROOT, STATIONS, load_model, dump_model
from starx.adaptation import read_training_csv, train_selected

schema = json.loads((ROOT/'assets/schema.json').read_text())
frame = read_training_csv((ROOT/'assets/data/example_boston_2016.csv.gz').read_bytes(), schema['features'])
sources = {sid: load_model(f'source_{sid}') for sid in STATIONS if sid != '8443970'}
local, adapted, report = train_selected(frame, schema, sources, '8443970', '2016-01-01', '2017-01-01', '2016-04-01', print)
out = ROOT/'test-output'
out.mkdir(exist_ok=True)
(out/'training_audit.json').write_text(json.dumps(report, indent=2, default=str))
(out/'trained_example.zip').write_bytes(dump_model(adapted))
assert len(report['candidate_records']) == 24
assert report['last_training_target'] < report['first_validation_origin']
print(json.dumps({k: report[k] for k in ['train_rows', 'validation_rows', 'selection_seconds', 'local_final_fit_seconds', 'adaptation_final_fit_seconds']}, indent=2))
