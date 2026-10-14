import io
import json
from pathlib import Path
import sys
import unittest
import zipfile

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from starx.models import ROOT, LEADS, load_model, dump_model, read_model
from starx.adaptation import read_training_csv, split_record, fit_model, metrics
from starx.feeds import qc_observations, build_vector
from starx.monitor import issue, verify, checkpoint, restore
from starx.features import hourly_to_training


class ModelAndDataTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.model = load_model('source_8443970')
        cls.schema = json.loads((ROOT/'assets/schema.json').read_text())
        cls.frame = read_training_csv((ROOT/'assets/data/example_boston_2016.csv.gz').read_bytes(), cls.schema['features'])

    def test_native_archive_roundtrip(self):
        X = self.frame[self.model.features].iloc[:10].to_numpy(np.float32)
        restored = read_model(dump_model(self.model))
        np.testing.assert_array_equal(self.model.predict(X), restored.predict(X))

    def test_continuation_retains_source_and_scalers(self):
        source_prediction = self.model.predict(self.frame[self.model.features].iloc[:5])
        updated = fit_model(self.schema, self.frame.iloc[:350], 3, source=self.model)
        self.assertEqual(updated.meta['rounds'], [603]*5)
        self.assertEqual(updated.meta['x_mean'], self.model.meta['x_mean'])
        self.assertEqual(updated.meta['y_scale'], self.model.meta['y_scale'])
        np.testing.assert_allclose(updated.predict(self.frame[self.model.features].iloc[:5], rounds=600), source_prediction, atol=2e-7)
        np.testing.assert_array_equal(self.model.predict(self.frame[self.model.features].iloc[:5]), source_prediction)

    def test_exact_tree_shap_additivity(self):
        X = self.frame[self.model.features].iloc[:2].to_numpy(np.float32)
        values, bases = self.model.explain(X)
        np.testing.assert_allclose((values.sum(2)+bases).T, self.model.predict(X), atol=2e-5)

    def test_complete_windows_and_purge(self):
        full, train, valid = split_record(self.frame, '2016-01-01', '2017-01-01', '2016-04-01')
        self.assertGreaterEqual(full.t0.min()-pd.Timedelta(hours=24), pd.Timestamp('2016-01-01'))
        self.assertLess(train.t0.max()+pd.Timedelta(hours=72), valid.t0.min())
        self.assertLess(full.t0.max()+pd.Timedelta(hours=72), pd.Timestamp('2017-01-01'))

    def test_bad_training_columns_rejected(self):
        with self.assertRaises(ValueError):
            read_training_csv(b't0,surge_6h\n2016-01-01,0.1\n', self.schema['features'])

    def test_non_model_zip_rejected(self):
        b = io.BytesIO()
        with zipfile.ZipFile(b, 'w') as z:
            z.writestr('../escape.pkl', b'not a model')
        with self.assertRaises(ValueError):
            read_model(b.getvalue())

    def test_no_incomplete_noaa_hours(self):
        water = [dict(t=f'2026-01-01 00:{6*i:02d}', v='1.0', f='0,0,0,0', q='p') for i in range(10)]
        hour = int(pd.Timestamp('2026-01-01').timestamp())
        residual = qc_observations(water, {hour: .4}, hour+3600)
        self.assertAlmostEqual(residual[hour], .6)
        self.assertEqual(qc_observations(water[:-1], {hour: .4}, hour+3600), {})
        water[0]['f'] = '0,1,0,0'
        self.assertEqual(qc_observations(water, {hour: .4}, hour+3600), {})

    def test_threshold_inclusive_and_counts(self):
        y = np.array([[.1]*5, [.2]*5, [.3]*5])
        score = metrics(y, y, np.repeat(.2, 5))
        self.assertEqual(score['high_surge_counts'], [2]*5)
        self.assertEqual(score['rmse_m'], 0.)
        self.assertEqual(score['warning_f1'], 1.)

    def test_hourly_builder_rejects_gaps_and_preserves_alignment(self):
        t = pd.date_range('2016-01-01', periods=700, freq='h')
        df = pd.DataFrame(dict(time_utc=t, surge_m=np.arange(700)/1000, tide_m=.2,
                               wind_u_m_s=3., wind_v_m_s=4., pressure_hpa=1010.))
        frame = hourly_to_training(df.to_csv(index=False).encode(), self.schema['features'])
        self.assertEqual(len(frame), 604)
        first = frame.iloc[0]
        self.assertAlmostEqual(first.surge_lb00, 0.)
        self.assertAlmostEqual(first.surge_lb24, .024)
        self.assertAlmostEqual(first.surge_72h, .096)
        self.assertAlmostEqual(first.wind_speed_fc72, 5.)
        self.assertAlmostEqual(first.wind_sin_fc72, .6)
        self.assertAlmostEqual(first.wind_cos_fc72, .8)
        gap = hourly_to_training(df.drop(index=350).to_csv(index=False).encode(), self.schema['features'])
        self.assertEqual(len(gap), len(frame)-97)

    def test_verification_and_checkpoint(self):
        ledger = []
        origin = int(pd.Timestamp('2026-01-01').timestamp())
        feed = dict(station='8443970', origin=origin, fetched_at=origin+3600,
                    X=self.frame[self.model.features].iloc[0].to_numpy(np.float32), provenance={})
        issue(ledger, feed, self.model, now=origin+3600)
        issue(ledger, feed, self.model, now=origin+3600)
        self.assertEqual(len(ledger), 1)
        next_feed = dict(station='8443970', fetched_at=origin+73*3600,
                         residual={origin+h*3600: .1 for h in LEADS})
        verify(ledger, next_feed)
        self.assertEqual(ledger[0]['truth'], [.1]*5)
        restored, restored_ledger, audits = restore(checkpoint(self.model, ledger, []))
        self.assertEqual(restored_ledger, ledger)
        self.assertEqual(restored.meta['id'], self.model.meta['id'])


class AppTests(unittest.TestCase):
    def test_local_activation_and_rollback_keep_strategy_labels(self):
        from streamlit.testing.v1 import AppTest
        app = AppTest.from_file(str(ROOT/'app.py'), default_timeout=60).run()
        app.session_state['active_8771450'] = load_model('local_2016_8771450')
        app.session_state['active_strategy_8771450'] = 'Local-only'
        app.session_state['previous_models'] = {'8771450': load_model('adapted_2016_8771450')}
        app.run()
        self.assertTrue(any('Your updated local-only model' in c.value for c in app.caption))
        app.radio(key='workspace').set_value('Verify & monitor').run()
        rollback = next(b for b in app.button if b.label == 'Roll back to the previous model')
        rollback.click().run()
        self.assertFalse(list(app.exception), str(app.exception))
        self.assertEqual(app.session_state['active_strategy_8771450'], 'Adapted')
        self.assertEqual(app.session_state['active_8771450'].meta['kind'], 'adapted')

    def test_every_page_renders(self):
        from streamlit.testing.v1 import AppTest
        app = AppTest.from_file(str(ROOT/'app.py'), default_timeout=60).run()
        self.assertFalse(list(app.exception), str(app.exception))
        pages = ['Transfer explorer', 'Update models', 'Verify & monitor', 'Explain forecasts', 'Paper evidence', 'Methods & downloads']
        for page in pages:
            with self.subTest(page=page):
                app.radio(key='workspace').set_value(page).run()
                self.assertFalse(list(app.exception), str(app.exception))
        app.radio(key='workspace').set_value('Explain forecasts').run()
        app.radio(key='attribution_view').set_value('Historical model profiles').run()
        self.assertFalse(list(app.exception), str(app.exception))


if __name__ == '__main__':
    unittest.main(verbosity=2)
