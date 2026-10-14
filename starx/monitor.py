"""Session-isolated issued-forecast ledger with portable checkpoints."""
from datetime import datetime, timezone
import io
import json
import time
import zipfile

import numpy as np
import pandas as pd

from .models import LEADS, dump_model, read_model


def issue(ledger, feed, model, now=None):
    issued_at = float(time.time() if now is None else now)
    if feed['origin']+min(LEADS)*3600 <= issued_at:
        return  # Never backdate issuance when a visitor reopens an old cached view.
    key = f'{feed["station"]}:{feed["origin"]}:{model.meta["id"]}'
    if any(r['key'] == key for r in ledger):
        return
    ledger.append(dict(key=key, station=feed['station'], origin=int(feed['origin']),
                       issued_at=issued_at, model_id=model.meta['id'],
                       X=np.asarray(feed['X'], float).tolist(),
                       prediction=model.predict(np.asarray(feed['X']).reshape(1, -1))[0].tolist(),
                       truth=None, provenance=feed['provenance']))


def verify(ledger, feed):
    for record in ledger:
        if record['station'] != feed['station'] or record['truth'] is not None:
            continue
        times = [record['origin']+h*3600 for h in LEADS]
        # Only observations whose valid times are later than actual issuance count.
        if min(times) <= record['issued_at'] or max(times)+3600 > feed['fetched_at']:
            continue
        if all(t in feed['residual'] for t in times):
            record['truth'] = [feed['residual'][t] for t in times]
    return ledger


def completed_frame(ledger, station, features, after=None):
    records = [r for r in ledger if r['station'] == station and r['truth'] is not None
               and (after is None or r['origin'] > after)]
    if not records:
        return pd.DataFrame()
    # One immutable observation/input set per origin, even if several strategies were issued.
    unique = {r['origin']: r for r in records}
    rows = []
    for r in sorted(unique.values(), key=lambda r: r['origin']):
        rows.append(dict(t0=pd.Timestamp(r['origin'], unit='s'), **dict(zip(features, r['X'])),
                         **{f'surge_{h}h': r['truth'][i] for i, h in enumerate(LEADS)}))
    return pd.DataFrame(rows)


def checkpoint(model, ledger, audits):
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, 'w', zipfile.ZIP_DEFLATED) as z:
        z.writestr('model.zip', dump_model(model))
        z.writestr('ledger.json', json.dumps(ledger, allow_nan=False))
        z.writestr('audits.json', json.dumps(audits, default=str, allow_nan=False))
        z.writestr('checkpoint.json', json.dumps(dict(format='starx-checkpoint-v1',
                    created_utc=datetime.now(timezone.utc).isoformat())))
    return buf.getvalue()


def restore(data):
    if len(data) > 75*1024**2:
        raise ValueError('Checkpoint exceeds 75 MB.')
    with zipfile.ZipFile(io.BytesIO(data)) as z:
        if set(z.namelist()) != {'model.zip', 'ledger.json', 'audits.json', 'checkpoint.json'}:
            raise ValueError('This is not a STARX checkpoint.')
        if sum(v.file_size for v in z.infolist()) > 200*1024**2:
            raise ValueError('Checkpoint expands beyond the app limit.')
        if json.loads(z.read('checkpoint.json')).get('format') != 'starx-checkpoint-v1':
            raise ValueError('Unsupported checkpoint format.')
        model = read_model(z.read('model.zip'))
        ledger = json.loads(z.read('ledger.json'))
        audits = json.loads(z.read('audits.json'))
    if not isinstance(ledger, list) or len(ledger) > 25000 or not isinstance(audits, list):
        raise ValueError('Checkpoint ledger is invalid.')
    for row in ledger:
        if len(row.get('X', [])) != 487 or len(row.get('prediction', [])) != 5:
            raise ValueError('Checkpoint has an invalid forecast shape.')
        if not np.isfinite(row['X']).all() or not np.isfinite(row['prediction']).all():
            raise ValueError('Checkpoint contains nonfinite data.')
        if row.get('truth') is not None and (len(row['truth']) != 5 or not np.isfinite(row['truth']).all()):
            raise ValueError('Checkpoint has invalid observed outcomes.')
    return model, ledger, audits
