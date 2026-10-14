# STARX — coastal forecast decision support

**Storm-surge Transfer and Adaptation with Response eXplainability**

[Open STARX](https://starx-coastal.streamlit.app) · [Developer: Dr. Soheil Radfar](https://sradfar.github.io)

An executable Streamlit application for the six-gauge study by Soheil Radfar and
Ning Lin at Princeton University. The app runs the trained XGBoost models, uses
NOAA observations and tides with GFS forcing, compares transfer strategies, and
supports local updating and forecast verification.

## Included workspaces

1. **Live twin:** execute five forecast leads, review threshold
   exceedances and their valid times, compare models, inspect tide-plus-surge
   water levels, and download forecasts and a decision brief.
2. **Transfer explorer:** all 30 directed routes, source/target RMSE matrix,
   before/after adaptation, local comparators, environmental separation and
   attribution mismatch.
3. **Adaptation studio:** upload target observations; select from nine local and
   fifteen adaptation candidates with chronological validation and a 72-hour
   purge; refit, activate, and download the selected models and complete audit.
4. **Model monitor:** preserve actual issued forecasts, verify matured NOAA
   outcomes, evaluate continued updates, roll back, and save/restore checkpoints.
5. **Explanation studio:** exact native TreeSHAP for executed forecasts and the
   paper's historical physical-group profiles.
6. **Paper evidence:** the 2016/2018 comparisons, short-record budgets, observed
   storm episodes, retained-source case study, and computational measurements.
7. **Methods & downloads:** forecasting and selection methods, definitions,
   provenance, workflow and downloadable scientific evidence.

## Run locally

Use Python 3.12:

```text
python -m pip install -r requirements.txt
python -m streamlit run app.py
```

The `assets` directory includes native XGBoost UBJ files and JSON preprocessing
manifests; no manuscript, credentials, private configuration, or pickle files
are required. No API keys are required for the included NOAA/Open-Meteo feeds.
Free Open-Meteo use is subject to its noncommercial-use limits; requests are
cached by station for five minutes.

## Deploy free to Streamlit Community Cloud

1. Create `sradfar/STARX` on GitHub and push the contents of this directory.
2. At https://share.streamlit.io choose **Create app → Deploy a public app from
   GitHub**, repository `sradfar/STARX`, branch `main`, entrypoint `app.py`.
3. Choose Python **3.12** under Advanced settings. No secrets are required.
4. Request an available application name, for example `starx-coastal`.
5. Verify the public URL from a signed-out browser: live input retrieval,
   predictions, all workspaces, model updating and downloads.

GitHub accepts the individual native model files without Git LFS. Each file is
below the per-file limit. Models are loaded on demand and the shared cache is
bounded; visitor updates are held in separate session state.

## Model updating

For raw hourly inputs, use `time_utc`, `surge_m`, `tide_m`, `wind_u_m_s`,
`wind_v_m_s`, and `pressure_hpa`. You may supply `water_level_m` in the same
datum as tide instead of surge. Meteorological wind speed/direction columns
(`wind_speed_m_s`, `wind_direction_from_deg`) can replace u/v components.
STARX constructs the complete feature and target windows without interpolation.

For already prepared examples, input CSV columns are `t0` in UTC, the ordered 487 predictor names listed in
`assets/schema.json`, and observed `surge_6h`, `surge_12h`, `surge_24h`,
`surge_48h`, `surge_72h`. Supplied sample weights are normalized within the fitting
subset. Otherwise the paper's capped exponential peak-surge weights are generated
from the training targets. Complete input histories and future targets must be available within
the chosen fitting budget. A real 3,000-origin Boston training-year example is
included; it is not the full-year paper experiment.

The original source ensembles stay unchanged. Updating retains their trees,
scalers and WQE objective; chronological validation selects the continuation
length (25, 50 or 100 rounds). Local-only training is evaluated under the same
target data budget. Scores used for selection are labeled validation results.
The packaged paper evidence remains the independent 2020–2025 evaluation.

Monitored continuation additionally uses a second, untouched acceptance block.
It requires at least 2,000 unique completed origins, improved high-surge error,
bounded normal/overall degradation, and enough observed high-surge samples.
The exact tested candidate is deployed without refitting on the acceptance data.

## Session persistence

Free Community Cloud can sleep or restart. The app does not claim continuous
unattended collection or permanent local storage. Download a model ZIP and a
workspace checkpoint to resume later. Uploaded models must use the STARX
native format; executable pickle uploads are not accepted. Session uploads and
updates do not modify the public paper models or another visitor's workspace.

## Scientific interpretation

- Formal baseline name: **long-record local**.
- Unadapted transfer is a comparison baseline.
- High-surge RMSE uses hourly observed targets at or above the pre-2020
  station/lead 95th percentile; observed-episode peak errors are separate.
- Forecast values are issued at five discrete leads. Their maximum is not an
  estimate of the exact storm peak. Model disagreement is not a calibrated
  uncertainty interval.
- Water levels are in meters MLLW; surge review thresholds are not official
  inundation stages. The app supports forecast review and model selection;
  local flood decisions also use official guidance and compatible local data.
- Source-candidate diagnostics (150 pair/lead cases) remain separate from the
  selected deployment comparison (30 station/lead cases per target year).

## Tests

```text
python -m unittest discover -s tests -v
```

The author-only exporter `tools/export_assets.py` regenerates deployable native
models from trusted local paper artifacts and checks prediction parity. It is
not needed for public deployment.

## Attribution

Developer: Dr. Soheil Radfar.
[sradfar.github.io](https://sradfar.github.io)
Research: Soheil Radfar and Ning Lin, Princeton University.
Data: NOAA CO-OPS and GFS through Open-Meteo (CC BY 4.0 attribution applies to
Open-Meteo weather data). Third-party packages retain their respective licenses.
