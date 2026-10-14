"""STARX: public model execution, short-record adaptation and coastal decision support."""
from __future__ import annotations

from datetime import date, datetime, timezone
import io
import json
from pathlib import Path
import time
import threading

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from starx.models import ROOT, STATIONS, LEADS, load_model, read_model, dump_model, group_for_feature
from starx.adaptation import read_training_csv, split_record, train_selected, continuation_check
from starx.feeds import collect
from starx.monitor import issue, verify, completed_frame, checkpoint, restore
from starx.features import hourly_to_training, TEMPLATE

st.set_page_config(page_title='STARX · Coastal decision support', page_icon='🌊', layout='wide')
DATA = ROOT / 'assets/data'
COLORS = {'Adapted': '#147D92', 'Local-only': '#D17839', 'Long-record local': '#596B7B', 'Zero-shot reference': '#9F6A97'}
LABELS = {'validation_selected_adaptation': 'Adapted', 'validation_selected_local': 'Local-only',
          'source_candidate_adaptation': 'Source-candidate adapted', 'full_local': 'Long-record local', 'zero_shot': 'Zero-shot reference'}

st.markdown('''<style>
.block-container {padding-top:1.7rem; padding-bottom:3rem; max-width:1500px}
h1 {font-size:2.55rem!important; letter-spacing:-.055rem}
h2 {font-size:1.6rem!important; margin-top:.8rem}
[data-testid="stMetric"] {background:white; border:1px solid #DCE5EB; border-radius:12px; padding:15px}
[data-testid="stSidebar"] {border-right:1px solid #DCE5EB}
.eyebrow {font-size:.8rem; color:#157A86; font-weight:700; letter-spacing:.12rem}
.intro {font-size:1.06rem; color:#557080; max-width:900px}
</style>''', unsafe_allow_html=True)


@st.cache_data
def table(name):
    df = pd.read_csv(DATA / name)
    for col in ['source_id', 'target_id', 'station_id']:
        if col in df:
            df[col] = df[col].astype(str)
    return df


@st.cache_resource(max_entries=8)
def model(mid):
    return load_model(mid)


@st.cache_resource
def update_slot():
    return threading.BoundedSemaphore(1)


@st.cache_data(ttl=300, max_entries=6)
def live_inputs(sid):
    return collect(sid, schema['features'])


def name(sid):
    return STATIONS.get(str(sid), {}).get('name', str(sid))


def plot(fig, key=None):
    fig.update_layout(template='plotly_white', font=dict(family='Arial', size=14, color='#173347'),
                      margin=dict(l=12, r=20, t=35, b=20), legend_title_text='',
                      paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='white',
                      legend=dict(orientation='h', y=-.22, x=0))
    st.plotly_chart(fig, width='stretch', key=key)


def download_csv(df, label, filename, key):
    st.download_button(label, df.to_csv(index=False).encode(), filename, 'text/csv', key=key)


def models_for_station():
    adapted = model(f'adapted_{year}_{station}')
    local = st.session_state.get(f'local_{station}', model(f'local_{year}_{station}'))
    active = st.session_state.get(f'active_{station}')
    if active is not None:
        if active.meta['kind'] == 'local':
            local = active
        else:
            adapted = active
    sid = adapted.meta.get('source_id') or model(f'adapted_{year}_{station}').meta['source_id']
    return {'Adapted': adapted, 'Local-only': local, 'Long-record local': model(f'source_{station}'),
            'Zero-shot reference': model(f'source_{sid}')}


def choice_caption():
    active = st.session_state.get(f'active_{station}')
    if active is not None:
        if active.meta['kind'] == 'local':
            return f"Your updated local-only model · {active.meta['rounds'][0]} rounds · fitted through {active.meta.get('fitted_through_utc', 'the supplied target record')}"
        return f"Your adapted model · source: {name(active.meta['source_id'])} · {active.meta['rounds'][0]} total rounds · fitted through {active.meta.get('fitted_through_utc', 'the supplied target record')}"
    choice = json.loads((DATA / f'{year}_{station}_choices.json').read_text())
    return f"Paper selection · {year} target record · source: {name(choice['adapted']['source_id'])} · {choice['adapted']['rounds']} added rounds"


if not (ROOT / 'assets/schema.json').exists():
    st.error('The model assets have not been prepared. Run tools/export_assets.py using the author workspace before deployment.')
    st.stop()
schema = json.loads((ROOT / 'assets/schema.json').read_text())
st.session_state.setdefault('ledger', [])
st.session_state.setdefault('update_audits', [])
st.session_state.setdefault('previous_models', {})

with st.sidebar:
    st.markdown('## 🌊 STARX')
    st.caption('Storm-surge Transfer and Adaptation with Response eXplainability')
    page = st.radio('Workspace', ['Forecast & insights', 'Transfer explorer', 'Update models',
                                'Verify & monitor', 'Explain forecasts', 'Paper evidence', 'Methods & downloads'], key='workspace')
    st.divider()
    station = st.selectbox('Target station', list(STATIONS), format_func=name)
    year = st.selectbox('Paper target record', [2016, 2018], help='Selects the published, validation-selected model set. Your session updates remain separate.')
    st.caption(f'NOAA {station} · {STATIONS[station]["coast"]} Coast')
    if f'active_{station}' in st.session_state:
        st.success('Your updated model is active for this station.')
    st.divider()
    st.caption('Inputs: NOAA observations and tides · GFS atmospheric forecasts. All times UTC; water levels in meters MLLW.')
    st.caption('Decision support for model evaluation and forecast review. Local flood decisions should also use official bulletins and site-specific thresholds.')


def forecast_page():
    st.markdown('<div class="eyebrow">OBSERVE → FORECAST → REVIEW</div>', unsafe_allow_html=True)
    st.title('Coastal outlook')
    st.markdown('<div class="intro">See when elevated surge is forecast, compare model strategies, and carry the evidence into your planning discussion.</div>', unsafe_allow_html=True)
    st.caption(choice_caption())
    a, b = st.columns([1, 3])
    refresh = a.button('Fetch current conditions', type='primary', width='stretch')
    auto = b.toggle('Refresh every 5 minutes while this page is open', value=False)
    if refresh:
        live_inputs.clear()
        try:
            with st.spinner('Collecting NOAA observations, tides and GFS forcing…'):
                st.session_state[f'feed_{station}'] = live_inputs(station)
        except Exception as exc:
            st.error(f'Live inputs unavailable: {exc}')
    feed = st.session_state.get(f'feed_{station}')
    if feed is None:
        st.info('Fetch current conditions to execute the trained models. No sample or simulated forecast is shown as live.')
        draw_station_map()
        st.subheader('What STARX helps you decide')
        for heading, text in [('When to review conditions', 'Identify which forecast leads exceed the chosen surge threshold.'),
                              ('Which model to use', 'Compare validation-selected adaptation against a local-only alternative.'),
                              ('When to update', 'Verify issued forecasts and check new model candidates using separated observations.')]:
            st.markdown(f'**{heading}.** {text}')
        return
    if auto:
        @st.fragment(run_every='5m')
        def auto_refresh():
            try:
                previous_time = st.session_state[f'feed_{station}']['fetched_at']
                refreshed = live_inputs(station)
                st.session_state[f'feed_{station}'] = refreshed
                if refreshed['fetched_at'] != previous_time:
                    st.rerun(scope='app')
            except Exception as exc:
                st.info(f'Waiting for complete live inputs: {exc}')
        auto_refresh()
        feed = st.session_state[f'feed_{station}']
    strategies = models_for_station()
    default_strategy = st.session_state.get(f'active_strategy_{station}', 'Adapted')
    selected = st.selectbox('Forecast strategy', list(strategies), index=list(strategies).index(default_strategy))
    current = strategies[selected]
    X = np.asarray(feed['X']).reshape(1, -1)
    predictions = {label: m.predict(X)[0] for label, m in strategies.items()}
    for m in strategies.values():
        issue(st.session_state.ledger, feed, m)
    verify(st.session_state.ledger, feed)
    threshold_rows = table(f'{year}_{station}_thresholds.csv').sort_values('lead_h')
    thresholds = threshold_rows.threshold_m.to_numpy()
    custom = st.checkbox('Use a site-specific surge review threshold')
    if custom:
        threshold = st.number_input('Surge above astronomical tide (m)', min_value=0., max_value=6., value=.3, step=.05)
        thresholds = np.repeat(threshold, 5)
    st.caption('Default thresholds are the station- and lead-specific pre-2020 95th-percentile surge values used in the paper. They are high-surge review thresholds, not official inundation stages.')
    p = predictions[selected]
    frame = pd.DataFrame(dict(lead_h=LEADS, valid_time_utc=pd.to_datetime([feed['origin']+h*3600 for h in LEADS], unit='s', utc=True),
                              surge_m=p, astronomical_tide_m=[feed['tides'][feed['origin']+h*3600] for h in LEADS],
                              review_threshold_m=thresholds))
    frame['total_water_level_m_MLLW'] = frame.surge_m + frame.astronomical_tide_m
    frame['threshold_exceeded'] = frame.surge_m >= frame.review_threshold_m
    peak = frame.iloc[int(np.argmax(p))]
    exceed = frame[frame.threshold_exceeded]
    age = (time.time()-feed['origin'])/3600
    cards = st.columns(4)
    cards[0].metric('Current observed surge', f"{feed['residual'][feed['origin']]:.3f} m")
    cards[1].metric('Largest predicted surge¹', f'{p.max():.3f} m', f'+{int(peak.lead_h)} h', delta_color='off')
    cards[2].metric('Leads over threshold', f'{len(exceed)} / 5')
    cards[3].metric('Observation origin age', f'{age:.1f} h')
    if len(exceed):
        st.warning(f"Review elevated surge at {', '.join('+'+str(v)+' h' for v in exceed.lead_h)}. The earliest sampled exceedance is {exceed.iloc[0].valid_time_utc:%d %b %H:%M} UTC. Check local water-level guidance and official forecasts for the affected period.")
    else:
        st.success('None of the five predicted surge values exceeds the chosen review threshold. Continue monitoring subsequent forecasts and local guidance.')
    if age > 3:
        st.warning('The observation origin is more than three hours old. Use the timestamp when interpreting the outlook.')
    if selected == 'Zero-shot reference':
        st.info('The unadapted source model is included as a comparison baseline. Use local validation to choose a deployment model.')
    left, right = st.columns([1.65, 1])
    with left:
        fig = go.Figure()
        for label, values in predictions.items():
            fig.add_scatter(x=frame.valid_time_utc, y=values, name=label, mode='lines+markers',
                            line=dict(color=COLORS[label], width=3 if label == selected else 1.4,
                                      dash='dot' if label == 'Zero-shot reference' else 'solid'))
        fig.add_scatter(x=frame.valid_time_utc, y=thresholds, name='Review threshold', mode='lines', line=dict(color='#B64A40', dash='dash'))
        fig.update_layout(yaxis_title='Storm-surge residual (m)', xaxis_title='Forecast valid time (UTC)', height=390)
        plot(fig)
        st.caption('¹ Maximum across the five evaluated leads, not an estimate of the exact event peak. Lines connect discrete lead predictions. Model spread is not a calibrated probability interval.')
    with right:
        st.subheader('Model agreement')
        deployed = np.stack([predictions['Adapted'], predictions['Local-only']])
        spread = np.ptp(deployed, axis=0)
        disagreement = np.any((deployed >= thresholds).any(0) & ~(deployed >= thresholds).all(0))
        st.metric('Largest adapted/local difference', f'{spread.max():.3f} m')
        st.write('The two short-record strategies disagree about a threshold exceedance. Review both forecasts and recent verification.' if disagreement
                 else 'The two short-record strategies agree on which sampled leads exceed the review threshold.')
        st.write(f"Active model: **{selected}**. Forecast origin: **{datetime.fromtimestamp(feed['origin'], timezone.utc):%d %b %Y %H:%M} UTC**.")
        st.caption(f"Generated {datetime.fromtimestamp(feed['fetched_at'], timezone.utc):%H:%M} UTC · {len(current.features)} predictors · model {current.meta['id']}")
        if not current.meta['id'].startswith(('user_', 'live_')):
            historical = table('test_metrics.csv')
            code = 'validation_selected_adaptation' if selected == 'Adapted' else 'validation_selected_local'
            if selected in ('Adapted', 'Local-only'):
                hist = historical[(historical.target_id == station) & (historical.target_year == year) & (historical.strategy == code)]
                st.caption(f'Historical evidence for this packaged model: overall RMSE {hist.rmse_m.mean():.3f} m; high-surge RMSE {hist.extreme_rmse_m.mean():.3f} m; warning F1 {hist.event_f1.mean():.3f}. These summarize the independent historical test, not a confidence interval for this forecast.')
    st.subheader('Water-level planning table')
    st.dataframe(frame, hide_index=True, width='stretch', column_config={
        'lead_h': 'Lead (h)', 'valid_time_utc': 'Valid time (UTC)',
        'surge_m': st.column_config.NumberColumn('Surge (m)', format='%.3f'),
        'astronomical_tide_m': st.column_config.NumberColumn('Tide (m MLLW)', format='%.3f'),
        'review_threshold_m': st.column_config.NumberColumn('Surge threshold (m)', format='%.3f'),
        'total_water_level_m_MLLW': st.column_config.NumberColumn('Total level (m MLLW)', format='%.3f'),
        'threshold_exceeded': 'Above threshold'})
    st.caption('Total water level = predicted surge + astronomical tide, both in MLLW. Local terrain or asset elevations must use a compatible datum before comparison.')
    c1, c2 = st.columns(2)
    with c1:
        download_csv(frame, 'Download forecast table', f'STARX_{station}_forecast.csv', 'forecast_csv')
    brief = f"# STARX forecast review\n\nStation: {name(station)} ({station})\n\nModel: {selected} ({current.meta['id']})\n\nOrigin: {datetime.fromtimestamp(feed['origin'], timezone.utc).isoformat()}\n\nInputs: NOAA observations and tides; GFS atmospheric forcing.\n\nLeads above surge review threshold: {len(exceed)} of 5.\n\nLargest sampled surge: {p.max():.3f} m at +{int(peak.lead_h)} h.\n\n" + frame.to_csv(index=False)
    c2.download_button('Download decision brief', brief, f'STARX_{station}_brief.md', 'text/markdown')


def draw_station_map():
    df = pd.DataFrame([dict(station=name(s), lat=v['lat'], lon=v['lon'], coast=v['coast']) for s, v in STATIONS.items()])
    fig = px.scatter_geo(df, lat='lat', lon='lon', hover_name='station', color='coast',
                         color_discrete_map={'East': '#157A86', 'Gulf': '#D17839'}, scope='usa')
    fig.update_geos(projection_type='albers usa', fitbounds='locations', visible=False, showcountries=True, showsubunits=True, subunitcolor='#CAD6DF')
    fig.update_traces(marker_size=13)
    fig.update_layout(height=300)
    plot(fig, 'network')


def transfer_page():
    st.markdown('<div class="eyebrow">COMPARE → SELECT → ADAPT</div>', unsafe_allow_html=True)
    st.title('Transfer explorer')
    st.write('Examine directional transfer performance and the added value of local updating. These are independent 2020–2025 historical evaluations.')
    reference = table('reference_metrics.csv')
    zero = reference[reference.strategy == 'zero_shot']
    full = reference[reference.strategy == 'full_local']
    matrix = zero.groupby(['target_id', 'source_id']).rmse_m.mean().unstack()
    for sid in STATIONS:
        matrix.loc[sid, sid] = full[full.target_id == sid].rmse_m.mean()
    matrix = matrix.reindex(index=list(STATIONS), columns=list(STATIONS))
    fig = px.imshow(matrix.to_numpy(), x=[name(s) for s in matrix.columns], y=[name(s) for s in matrix.index],
                    text_auto='.3f', color_continuous_scale='OrRd', labels=dict(x='Source station', y='Target station', color='RMSE (m)'), aspect='auto')
    fig.update_layout(height=420)
    plot(fig)
    st.caption('Rows = target, columns = source. Off-diagonal cells are unadapted transfer; the diagonal is the long-record local reference. Values average five forecast leads.')
    sid = st.selectbox('Inspect a source → target route', [s for s in STATIONS if s != station], format_func=lambda s: f'{name(s)} → {name(station)}')
    candidate = table('candidate_metrics.csv')
    route = candidate[(candidate.target_id == station) & (candidate.source_id == sid)].sort_values('lead_time_h')
    z = zero[(zero.target_id == station) & (zero.source_id == sid)].sort_values('lead_time_h')
    primary = table('test_metrics.csv')
    local = primary[(primary.target_id == station) & (primary.target_year == 2016) & (primary.strategy == 'validation_selected_local')]
    lf = full[full.target_id == station]
    series = pd.concat([z.assign(label='Zero-shot reference'), route.assign(label='Adapted'),
                        local.assign(label='Local-only'), lf.assign(label='Long-record local')])
    cards = st.columns(3)
    cards[0].metric('Zero-shot RMSE', f'{z.rmse_m.mean():.3f} m')
    cards[1].metric('Source-candidate adapted RMSE', f'{route.rmse_m.mean():.3f} m')
    cards[2].metric('Error reduction after updating', f'{100*(1-route.rmse_m.mean()/z.rmse_m.mean()):.1f}%')
    plot(px.line(series, x='lead_time_h', y='rmse_m', color='label', markers=True, color_discrete_map=COLORS,
                 labels={'lead_time_h': 'Forecast lead (h)', 'rmse_m': 'RMSE (m)'}))
    selected = bool(route.chosen_for_deployment.iloc[0])
    st.info(f"This source used {int(route.selected_rounds.iloc[0])} added rounds selected within 2016. " +
            ('It was the target-validation-selected deployment source.' if selected else 'It is a route diagnostic; another source had the lowest target validation RMSE.'))
    st.caption('Route diagnostics use the 2016 target record. The sidebar year controls deployed models and the separate paper-evidence comparisons.')
    diagnostics = table('pair_diagnostics_summary.csv')
    r = diagnostics[(diagnostics.target_id == station) & (diagnostics.source_id == sid)].iloc[0]
    reverse = diagnostics[(diagnostics.target_id == sid) & (diagnostics.source_id == station)].iloc[0]
    st.subheader('Why the route deserves local evaluation')
    st.write(f"The environmental distance is **{r.domain_distance:.3f}** in both directions, but the RMSE penalties are **{r.penalty:.3f} m** for this route and **{reverse.penalty:.3f} m** in reverse. Environmental similarity alone does not select a reliable source.")
    st.write(f"Mean attribution mismatch changes from **{r.zero_shot_tv_distance:.3f}** to **{r.adapted_tv_distance:.3f}** after updating. This describes fitted-model behavior; source selection uses target validation error.")
    st.dataframe(series[['label', 'lead_time_h', 'rmse_m', 'extreme_rmse_m', 'event_f1']], hide_index=True,
                 column_config={'extreme_rmse_m': 'High-surge RMSE (m)', 'event_f1': 'Threshold-warning F1'})
    download_csv(series, 'Download route metrics', f'STARX_{sid}_to_{station}.csv', 'route_csv')


def update_page():
    st.markdown('<div class="eyebrow">LOCAL EVIDENCE → VALIDATION → MODEL UPDATE</div>', unsafe_allow_html=True)
    st.title('Update models with local observations')
    st.write('Fit the paper’s two short-record strategies using the same target observations. Source trees and scalers are retained during adaptation; local-only models learn their scalers from their training subset.')
    with st.expander('Required data and selection procedure', expanded=False):
        st.write('Upload hourly surge, tide and atmospheric data and let STARX construct the predictors, or upload prepared forecast origins with 487 predictors and five observed future surge targets. The app retains complete 24-hour histories and 72-hour target windows inside the fitting budget. Training targets end before the validation boundary, giving a 72-hour purge. The lowest mean validation RMSE across five leads selects each strategy. Selected models are then refitted on the full target budget.')
        st.write('Local search: three configurations × 200/400/600 rounds. Adaptation search: the other five sources × 25/50/100 added rounds. Uploaded data stay within your app session; the shared paper models are unchanged.')
    c1, c2, c3 = st.columns(3)
    c1.download_button('Download predictor schema', (ROOT/'assets/schema.json').read_bytes(), 'STARX_schema.json', 'application/json')
    c2.download_button('Download real training example', (DATA/'example_boston_2016.csv.gz').read_bytes(), 'Boston_2016_example.csv.gz', 'application/gzip')
    c3.caption('Example: 3,000 complete Boston origins from 2016. It is a learning example, not the full-year paper experiment.')
    input_kind = st.radio('Input format', ['Hourly observations and forcing', 'Prepared 487-predictor examples'], horizontal=True, key='input_kind')
    if input_kind == 'Hourly observations and forcing':
        st.caption('Columns: time_utc, surge_m, tide_m, wind_u_m_s, wind_v_m_s, pressure_hpa. You may supply water_level_m instead of surge_m; water level and tide must share a datum. Use the same atmospheric inputs as your intended forecasting workflow.')
        st.download_button('Download hourly CSV template', TEMPLATE, 'STARX_hourly_template.csv', 'text/csv')
    uploaded = st.file_uploader('Target record (.csv or .csv.gz)', type=['csv', 'gz'], key=f'train_upload_{station}')
    use_example = st.checkbox('Use the included Boston training example', disabled=station != '8443970')
    raw = uploaded.getvalue() if uploaded else (DATA/'example_boston_2016.csv.gz').read_bytes() if use_example else None
    if raw is not None:
        try:
            frame = (hourly_to_training(raw, schema['features']) if uploaded and input_kind == 'Hourly observations and forcing'
                     else read_training_csv(raw, schema['features']))
            if frame.empty or frame.t0.max()+pd.Timedelta(hours=73) > pd.Timestamp.now(tz='UTC').tz_localize(None):
                raise ValueError('No complete, already-observed target windows are available in this record.')
        except Exception as exc:
            st.error(str(exc))
            return
        st.write(f'{len(frame):,} supplied origins · {frame.t0.min():%d %b %Y} to {frame.t0.max():%d %b %Y}')
        first = (frame.t0.min()-pd.Timedelta(hours=24)).date()
        last = (frame.t0.max()+pd.Timedelta(hours=73)).date()
        default_cut = frame.t0.iloc[int(len(frame)*.75)].date()
        a, b, c = st.columns(3)
        start = a.date_input('Budget begins (inclusive UTC)', value=first)
        end = b.date_input('Budget ends (exclusive UTC)', value=last+pd.Timedelta(days=1))
        cutoff = c.date_input('Validation begins (UTC)', value=default_cut)
        try:
            full, training, validation = split_record(frame, start, end, cutoff)
            st.caption(f'{len(full):,} complete-budget origins · {len(training):,} inner training · {len(validation):,} validation. Selection uses observed target data only.')
        except ValueError as exc:
            st.error(str(exc))
            return
        if pd.Timestamp(start) < pd.Timestamp('2016-01-01'):
            st.error('The supplied source models were fitted through 2015. Use a target record beginning in 2016 or later for chronological selection.')
            return
        if st.button('Select and fit both strategies', type='primary'):
            slot = update_slot()
            if not slot.acquire(blocking=False):
                st.info('Another model update is running. Try again in a few minutes.')
                return
            try:
                with st.status('Selecting source and local configurations…', expanded=True) as status:
                    sources = {s: model(f'source_{s}') for s in STATIONS if s != station}
                    local, adapted, report = train_selected(frame, schema, sources, station, start, end, cutoff,
                                                            progress=lambda text: status.update(label=text))
                    st.session_state[f'training_{station}'] = (local, adapted, report)
                    status.update(label='Both selected strategies fitted', state='complete')
            except Exception as exc:
                st.error(f'Update could not be completed: {exc}')
            finally:
                slot.release()
    result = st.session_state.get(f'training_{station}')
    if result:
        local, adapted, report = result
        st.subheader('What the local validation says')
        a, b = report['choices']['local'], report['choices']['adapted']
        cols = st.columns(3)
        cols[0].metric('Local-only validation RMSE', f'{a["rmse_m"]:.3f} m')
        cols[1].metric('Adapted validation RMSE', f'{b["rmse_m"]:.3f} m')
        cols[2].metric('Adaptation validation gain', f'{100*(1-b["rmse_m"]/a["rmse_m"]):.1f}%')
        winner = 'Adapted' if b['rmse_m'] < a['rmse_m'] else 'Local-only'
        st.info(f"{winner} had the lower overall validation RMSE. The selected source was {name(b['candidate'])}, with {b['rounds']} added rounds. Review high-surge error and threshold-warning F1 alongside overall error before choosing an active strategy.")
        st.caption('These validation scores selected the candidates. Independent future verification is available in Verify & monitor; these values are not the paper’s held-out test performance.')
        st.caption('For this new fitting exercise, high-surge validation thresholds come only from the inner training targets. No validation or future test observations set the thresholds.')
        candidates = pd.DataFrame(report['candidate_records'])
        st.dataframe(candidates.drop(columns=['lead_rmse_m', 'high_surge_rmse_by_lead', 'high_surge_counts']), hide_index=True)
        a1, a2, a3 = st.columns(3)
        a1.download_button('Download adapted model', dump_model(adapted), f'STARX_{station}_adapted.zip', 'application/zip')
        a2.download_button('Download local-only model', dump_model(local), f'STARX_{station}_local.zip', 'application/zip')
        a3.download_button('Download update audit', json.dumps(report, indent=2), 'STARX_update_audit.json', 'application/json')
        deployment = st.radio('Use in this session', ['Adapted', 'Local-only'], horizontal=True)
        if st.button('Activate selected model for this station'):
            previous = models_for_station()[st.session_state.get(f'active_strategy_{station}', 'Adapted')]
            st.session_state.previous_models[station] = previous
            st.session_state[f'active_{station}'] = adapted if deployment == 'Adapted' else local
            st.session_state[f'active_strategy_{station}'] = deployment
            st.session_state[f'local_{station}'] = local
            st.session_state.update_audits.append(report)
            st.success(f'{deployment} is active in your session. Download it to preserve the update across restarts.')
    st.divider()
    st.subheader('Restore an existing model')
    saved = st.file_uploader('STARX model archive (.zip)', type=['zip'], key='model_restore')
    if saved and st.button('Load this model'):
        try:
            restored = read_model(saved.getvalue())
            if restored.meta.get('target_id') != station:
                raise ValueError('The saved target station does not match the selected station.')
            if restored.meta['kind'] not in ('adapted', 'local'):
                raise ValueError('Restore a target-fitted adapted or local-only model for deployment.')
            st.session_state.previous_models[station] = models_for_station()[st.session_state.get(f'active_strategy_{station}', 'Adapted')]
            st.session_state[f'active_{station}'] = restored
            st.session_state[f'active_strategy_{station}'] = 'Local-only' if restored.meta['kind'] == 'local' else 'Adapted'
            st.success('Saved model restored for this session.')
        except Exception as exc:
            st.error(str(exc))


def monitor_page():
    st.title('Verify forecasts and manage updates')
    st.write('Forecasts issued in your session are stored with their exact inputs, model version, origin and issuance time. Subsequent NOAA observations verify predictions after all five leads have matured.')
    st.info('Free hosting runs while the app is active. This session does not collect forecasts while it is asleep. Save a checkpoint to retain your models and verification record between visits.')
    strategies = models_for_station()
    active_strategy = st.session_state.get(f'active_strategy_{station}', 'Adapted')
    active = strategies[active_strategy]
    feed = st.session_state.get(f'feed_{station}')
    if st.button('Refresh observations and verify pending forecasts'):
        try:
            live_inputs.clear()
            feed = live_inputs(station)
            st.session_state[f'feed_{station}'] = feed
            verify(st.session_state.ledger, feed)
            issue(st.session_state.ledger, feed, active)
        except Exception as exc:
            st.error(str(exc))
    records = [r for r in st.session_state.ledger if r['station'] == station]
    completed = [r for r in records if r['truth'] is not None]
    unique = completed_frame(st.session_state.ledger, station, schema['features'])
    cols = st.columns(3)
    cols[0].metric('Issued model forecasts', f'{len(records):,}')
    cols[1].metric('Completed model forecasts', f'{len(completed):,}')
    cols[2].metric('Unique completed origins', f'{len(unique):,}')
    if completed:
        rows = []
        for r in completed:
            for i, h in enumerate(LEADS):
                rows.append(dict(model=r['model_id'], lead_h=h, prediction_m=r['prediction'][i], observed_m=r['truth'][i], origin=pd.Timestamp(r['origin'], unit='s')))
        scored = pd.DataFrame(rows)
        scored['error_m'] = scored.prediction_m-scored.observed_m
        scored['squared_error'] = scored.error_m**2
        performance = scored.groupby(['model', 'lead_h']).agg(n=('error_m', 'size'), mse=('squared_error', 'mean'), bias_m=('error_m', 'mean')).reset_index()
        performance['rmse_m'] = np.sqrt(performance.pop('mse'))
        st.dataframe(performance, hide_index=True)
        download_csv(scored, 'Download verified forecasts', f'STARX_{station}_verification.csv', 'verify_csv')
    else:
        st.caption('Verification begins when previously issued forecasts reach their valid times. Historical observations are never relabeled as forecasts issued by this session.')
    st.subheader('Check whether updating is supported')
    st.write('At least 2,000 unique completed origins are required. Continuation length is selected on chronological validation; a second, untouched block tests acceptance. High-surge RMSE must improve by ≥1%, overall RMSE may increase by at most 0.5%, and normal-condition RMSE by at most 1%. At least five high-surge observations per lead are required.')
    last_consumed = st.session_state.get(f'consumed_{station}')
    fresh = unique if last_consumed is None or unique.empty else unique[unique.t0 > pd.Timestamp(last_consumed)]
    if st.button('Evaluate a monitored update', disabled=len(fresh) < 2000):
        slot = update_slot()
        if not slot.acquire(blocking=False):
            st.info('Another model update is running. Try again in a few minutes.')
            return
        try:
            with st.status('Checking a model continuation…') as status:
                candidate, report = continuation_check(active, fresh, schema, progress=lambda text: status.update(label=text))
                st.session_state[f'consumed_{station}'] = report['consumed_through_utc']
                st.session_state.update_audits.append(report)
                if report['accepted']:
                    st.session_state.previous_models[station] = active
                    st.session_state[f'active_{station}'] = candidate
                    st.session_state[f'active_strategy_{station}'] = 'Local-only' if candidate.meta['kind'] == 'local' else 'Adapted'
                st.session_state[f'monitor_report_{station}'] = report
                status.update(label='Update checks complete', state='complete')
        except Exception as exc:
            st.error(str(exc))
        finally:
            slot.release()
    st.caption(f'{len(fresh):,} fresh completed origins available. A continuous hourly record needs about 86 days to accumulate 2,000 completed 72-hour forecasts.')
    if f'monitor_report_{station}' in st.session_state:
        report = st.session_state[f'monitor_report_{station}']
        st.success(report['reason']) if report['accepted'] else st.info(report['reason'])
        st.json(report, expanded=False)
    if station in st.session_state.previous_models and st.button('Roll back to the previous model'):
        previous = st.session_state.previous_models.pop(station)
        st.session_state[f'active_{station}'] = previous
        st.session_state[f'active_strategy_{station}'] = 'Local-only' if previous.meta['kind'] == 'local' else 'Adapted'
        st.success('Previous model restored. The audit history is retained.')
    st.subheader('Preserve or restore your workspace')
    current_active = models_for_station()[st.session_state.get(f'active_strategy_{station}', 'Adapted')]
    st.download_button('Download model + monitoring checkpoint', checkpoint(current_active, st.session_state.ledger, st.session_state.update_audits),
                       f'STARX_{station}_checkpoint.zip', 'application/zip')
    up = st.file_uploader('Restore a STARX checkpoint', type=['zip'], key='checkpoint_restore')
    if up and st.button('Restore checkpoint'):
        try:
            m, ledger, audits = restore(up.getvalue())
            if m.meta.get('target_id') != station:
                raise ValueError('Select the checkpoint’s target station before restoring it.')
            if m.meta['kind'] not in ('adapted', 'local'):
                raise ValueError('The active checkpoint model must be target-fitted.')
            st.session_state[f'active_{station}'] = m
            st.session_state[f'active_strategy_{station}'] = 'Local-only' if m.meta['kind'] == 'local' else 'Adapted'
            st.session_state.ledger, st.session_state.update_audits = ledger, audits
            if audits and 'consumed_through_utc' in audits[-1]:
                st.session_state[f'consumed_{station}'] = audits[-1]['consumed_through_utc']
            st.success('Checkpoint restored. Refresh observations to verify any pending forecasts.')
        except Exception as exc:
            st.error(str(exc))


def explain_page():
    st.title('Explain model behavior')
    st.write('Inspect the physical input groups behind a prediction and how local updating changes their contributions. SHAP describes fitted-model behavior; it does not establish physical causation.')
    mode = st.radio('Attribution view', ['Current forecast', 'Historical model profiles'], horizontal=True, key='attribution_view')
    h = st.select_slider('Forecast lead (h)', options=LEADS, value=24)
    if mode == 'Current forecast':
        feed = st.session_state.get(f'feed_{station}')
        if feed is None:
            st.info('Fetch current conditions on Forecast & insights to explain an executed forecast.')
            return
        if st.button('Calculate exact TreeSHAP', type='primary'):
            active = models_for_station()['Adapted']
            X = np.asarray(feed['X']).reshape(1, -1)
            with st.spinner('Calculating contributions from the retained and added trees…'):
                values, bases = active.explain(X)
            st.session_state[f'explanation_{station}'] = (values, bases, active.meta['features'], active.predict(X)[0], active.meta['id'], feed['origin'])
        result = st.session_state.get(f'explanation_{station}')
        if result:
            values, bases, features, prediction, mid, origin = result
            i = LEADS.index(h)
            grouped = pd.DataFrame(dict(group=[group_for_feature(f) for f in features], contribution_m=values[i, 0])).groupby('group', as_index=False).contribution_m.sum()
            plot(px.bar(grouped.sort_values('contribution_m'), x='contribution_m', y='group', orientation='h',
                        color='contribution_m', color_continuous_scale='Tealrose', labels={'contribution_m': 'Signed contribution (m)', 'group': ''}))
            error = abs(float(bases[i, 0] + values[i, 0].sum())-float(prediction[i]))
            st.caption(f'Model {mid} · origin {pd.Timestamp(origin, unit="s")} UTC. Reference {bases[i, 0]:.3f} m + contributions = {prediction[i]:.3f} m. Additivity check: {error:.2g} m.')
            download_csv(grouped, 'Download contributions', 'STARX_SHAP_groups.csv', 'shap_csv')
    else:
        df = table('shap_groups.csv')
        df = df[(df.target_id == station) & (df.lead_time_h == h) & (df.strategy == 'full_local')]
        per = st.checkbox('Normalize importance by the number of predictors in each group')
        col = 'per_feature_mean_abs_shap_m' if per else 'grouped_signed_mean_abs_shap_m'
        plot(px.bar(df.sort_values(col), x=col, y='group', orientation='h', labels={col: 'Mean absolute contribution (m)', 'group': ''}, color_discrete_sequence=['#157A86']))
        st.caption('Long-record local model, 25,000 fixed historical test inputs. The default sums signed contributions within each physical group before taking absolute values. Per-predictor sensitivity divides summed feature importance by group size.')
        alignment = table('shap_alignment.csv')
        subset = alignment[(alignment.target_id == station) & (alignment.lead_time_h == h)].copy()
        subset['Source'] = subset.source_id.map(name)
        st.dataframe(subset[['Source', 'zero_shot_tv_distance', 'adapted_tv_distance', 'full_test_rmse_recovery_m']], hide_index=True)
        st.caption('Transfer-profile comparisons use identical 5,000 target inputs. Total-variation distance measures attribution mismatch; smaller is closer to the target long-record local profile.')


def evidence_page():
    st.markdown('<div class="eyebrow">INDEPENDENT HISTORICAL EVALUATION · 2020–2025</div>', unsafe_allow_html=True)
    st.title('Evidence from the paper')
    st.write('Short-Record Adaptation Substantially Improves Transferability of Storm Surge Forecasting Models · Soheil Radfar and Ning Lin')
    aggregate = table('aggregate_performance.csv')
    rows = aggregate[aggregate.target_year == year].set_index('strategy')
    a, l = rows.loc['validation_selected_adaptation'], rows.loc['validation_selected_local']
    cs = st.columns(4)
    cs[0].metric('Overall RMSE reduction', f'{100*(1-a.rmse_m/l.rmse_m):.1f}%')
    cs[1].metric('High-surge RMSE reduction', f'{100*(1-a.extreme_rmse_m/l.extreme_rmse_m):.1f}%')
    cs[2].metric('Adapted warning F1', f'{a.event_f1:.3f}')
    cs[3].metric('Stations with lower overall RMSE', '5 / 6')
    st.caption('Adaptation and local-only training use identical local observations and chronological validation. Summaries average 30 station–lead cases per strategy; hourly predictions are not independent replicates.')
    station_metrics = table('station_performance.csv')
    pivot = station_metrics[station_metrics.target_year == year].pivot(index='target_name', columns='strategy', values='rmse_m')
    pivot['Adaptation gain (%)'] = 100*(1-pivot.validation_selected_adaptation/pivot.validation_selected_local)
    fig = px.bar(pivot.reset_index(), x='target_name', y='Adaptation gain (%)', color='Adaptation gain (%)', color_continuous_scale='Tealrose', labels={'target_name': 'Target station'})
    plot(fig)
    st.write('Adaptation offers the strongest advantage with very short local records and during high-surge conditions. St. Petersburg shows why a locally trained comparator remains useful: it had the lower overall RMSE in both one-year comparisons.')
    lead_metrics = table('test_metrics.csv')
    sub = lead_metrics[(lead_metrics.target_year == year) & (lead_metrics.target_id == station)].copy()
    sub['Strategy'] = sub.strategy.map(LABELS)
    st.subheader(f'{name(station)} · accuracy by forecast lead')
    plot(px.line(sub, x='lead_time_h', y='rmse_m', color='Strategy', markers=True, color_discrete_map=COLORS,
                 labels={'lead_time_h': 'Forecast lead (h)', 'rmse_m': 'RMSE (m)'}))
    st.dataframe(sub[['Strategy', 'lead_time_h', 'rmse_m', 'extreme_rmse_m', 'event_precision', 'event_recall', 'event_f1']], hide_index=True)
    st.subheader('How the local data budget changes the benefit')
    budget = table('budget_summary.csv').copy()
    budget['Strategy'] = budget.strategy.map(LABELS)
    plot(px.line(budget, x='target_months', y='rmse_m', color='Strategy', markers=True, color_discrete_map=COLORS,
                 labels={'target_months': 'Months of target observations', 'rmse_m': 'Overall RMSE (m)'}))
    st.subheader('Observed storm episodes')
    eps = table('episode_summary.csv')
    eps = eps[eps.target_year == year].copy()
    eps['Strategy'] = eps.strategy.map(LABELS).fillna(eps.strategy)
    st.dataframe(eps[['Strategy', 'subset', 'n_episodes', 'mean_peak_abs_error_m', 'median_peak_abs_timing_error_h']], hide_index=True)
    st.caption('Observed episodes use the paper’s separate declustered 24-hour episode definition. High-surge RMSE elsewhere uses hourly observations at or above the pre-2020 95th percentile; these are different evaluations.')
    st.subheader('Source reuse and the added local adjustment')
    trace = table('case_trace.csv')
    trace['valid_time'] = pd.to_datetime(trace.valid_time)
    trace_long = trace.melt(id_vars='valid_time', var_name='Strategy', value_name='surge_m')
    trace_long['Strategy'] = trace_long.Strategy.map({'observed': 'Observed', 'zero_shot': 'Zero-shot reference', 'adapted': 'Adapted', 'local': 'Local-only', 'full_local': 'Long-record local'})
    plot(px.line(trace_long, x='valid_time', y='surge_m', color='Strategy',
                 color_discrete_map={**COLORS, 'Observed': '#1F2F3B'},
                 labels={'valid_time': 'Observed episode valid time (UTC)', 'surge_m': 'Surge (m)'}))
    case = json.loads((DATA/'case_study.json').read_text())
    st.write(f"Atlantic City → Boston at 24 h retained 600 source trees and added {case['added_rounds']} validation-selected trees. Test RMSE was {case['test_rmse24_m']['zero_shot']:.3f} m before updating, {case['test_rmse24_m']['adapted']:.3f} m after updating, and {case['test_rmse24_m']['local']:.3f} m for local-only training.")
    st.subheader('Computational cost')
    st.dataframe(table('aggregate_timing.csv'), hide_index=True)
    st.caption('Selection, final fitting, source pretraining, and prediction are separate measurements on the paper’s benchmark computer. These are not cloud latency guarantees.')
    if (DATA/'geographic_mechanisms.csv').exists():
        st.subheader('Coastal descriptors and geographic sensitivity')
        st.write('Tidal and geographic information does not consistently remove cross-location error. The separate pooled geographic experiments identify where descriptors help or hurt; these findings support local validation when choosing a forecasting model.')
        st.dataframe(table('geographic_mechanisms.csv'), hide_index=True)
        st.caption('Gain = 100 × (primary RMSE − extended-model RMSE) / primary RMSE. Counterfactual settings are retrospective diagnostics selected using historical errors; they are not deployment selections.')
    gallery = ROOT/'assets/paper/figures.json'
    if gallery.exists():
        with st.expander('Explore the full paper figures and explanations'):
            figures = json.loads(gallery.read_text())
            figure = st.selectbox('Paper figure', figures, format_func=lambda v: f"Figure {v['number']} · {v['caption'].split('. ')[1]}")
            st.image(str(gallery.parent/figure['file']), width='stretch')
            st.write(figure['caption'])
    download_csv(sub, 'Download station evidence', f'STARX_{year}_{station}_evidence.csv', 'paper_csv')


def methods_page():
    st.title('Methods, data and reproducibility')
    st.write('STARX connects coastal observations, forecasting models, short-record updating, verification, and explanation in one workflow.')
    st.graphviz_chart('''digraph STARX {rankdir=LR; graph [bgcolor="transparent"]; node [shape=box,style="rounded,filled",fillcolor="#E6F2F4",fontname="Arial"]; "NOAA observations\nand tides" -> "Complete 487-predictor\nforecast window"; "GFS wind\nand pressure" -> "Complete 487-predictor\nforecast window"; "Complete 487-predictor\nforecast window" -> "Selected local or\nadapted XGBoost model"; "Selected local or\nadapted XGBoost model" -> "6–72 h forecasts\nthreshold review"; "Issued forecasts +\nnew observations" -> "Verification and\nupdate evaluation"; "Verification and\nupdate evaluation" -> "Selected local or\nadapted XGBoost model"; "Selected local or\nadapted XGBoost model" -> "SHAP and\nmodel comparison";}''')
    with st.expander('Forecast inputs and outputs', expanded=True):
        st.write('Each example uses observed surge and atmospheric history from t−24 h through t, future wind speed/direction and pressure from t+1 through t+72 h, astronomical tide, and cyclic month predictors: 487 inputs. Five direct XGBoost regressors predict surge at +6, +12, +24, +48 and +72 h. Live atmospheric forcing is GFS. Water-level and tide values use MLLW; all timestamps are UTC.')
        st.write('NOAA history uses hourly means from ten quality-controlled six-minute observations, less predicted astronomical tide. Missing or flagged windows block forecasting. Meteorological wind direction is converted to the training vector convention.')
    with st.expander('Four deployment strategies and model selection'):
        st.write('Long-record local: fitted on the target’s 2000–2015 record. Zero-shot: a source model evaluated at a different station, included only as a transfer baseline. Local-only: fitted from scratch on the target record. Adapted: retains the source trees and scalers and adds target-trained trees.')
        st.write('For each 2016 or 2018 target year, January–September origins train candidates whose targets end before October. October–December origins validate complete target windows. The lowest average validation RMSE across all five leads selects a local configuration/tree count or source/continuation length. Both selected strategies are refitted on identical full-year target examples: 8,688 in 2016 and 8,664 in 2018. Independent testing uses 52,536 origins in 2020–2025 per station.')
        st.write('Sources have 600 rounds, depth 6, learning rate 0.04, row/column sampling 0.8/0.7, minimum child weight 5, L1/L2 regularization 0.1/1.0 and seed 42. Local candidates use the three documented configurations at 200/400/600 rounds; adaptation uses 25/50/100 added rounds for each of the five other sources.')
    with st.expander('Objective and evaluation metrics'):
        st.latex(r'L=0.5\rho_{0.8}(e)+0.5|0.7-\mathbf{1}(e<0)|e^2,\quad e=y-\hat y')
        st.write('The weighted quantile–quadratic objective penalizes underprediction asymmetrically. Metrics include overall RMSE, MAE, signed bias, high-surge RMSE, threshold precision/recall/F1, skill against persistence, episode peak amplitude and timing errors, and computational cost.')
        st.latex(r'\mathrm{Gain}(\%)=100\frac{\mathrm{RMSE}_{\mathrm{local}}-\mathrm{RMSE}_{\mathrm{adapted}}}{\mathrm{RMSE}_{\mathrm{local}}}')
        st.latex(r'\mathrm{TSR}=\frac{1-(E_{\mathrm{transfer}}/E_{\mathrm{persistence}})^2}{1-(E_{\mathrm{local}}/E_{\mathrm{persistence}})^2}')
        st.caption('TSR follows the saved paper definition; see the downloadable metric documentation for its precise interpretation.')
        st.write('High-surge RMSE evaluates observed hourly targets at or above the pre-2020 station/lead 95th percentile. An exceedance at any of five forecast leads is not an independent storm. Episode metrics use a separate observed-episode definition.')
    with st.expander('Attribution, environmental separation and geographic experiments'):
        st.write('Exact tree-path-dependent TreeSHAP decomposes predictions into a reference and additive contributions. Nine physical groups are used. Long-record profiles use 25,000 inputs; transfer comparisons use a common subset of 5,000 target inputs. Contributions are converted to meters using each model’s target scale. Total-variation distance between normalized profiles quantifies attribution mismatch.')
        st.write('Environmental distance summarizes differences in the mean and variability of the paper’s atmospheric, surge and tidal inputs. It is symmetric; directional performance need not be. Pooled geographic experiments examine tidal and coastal descriptors separately from the six-gauge transfer-selection experiment. Downloadable evidence retains these distinctions.')
    with st.expander('Model updating and free-hosting behavior'):
        st.write('Target-record updating selects among candidate sources and continuation lengths. Monitored continuation uses only completed issued forecasts, two chronological purges, a validation block for choosing 25/50/100 added rounds, and an untouched acceptance block. Accepted models retain their tested trees; no refit on the acceptance block is performed. Rollback restores the previous session model.')
        st.write('Each visitor’s uploads, updated models and forecast ledger are isolated in that session. The app does not promise collection while Community Cloud sleeps. Downloadable model and workspace checkpoints preserve your work; restore them on a later visit. Shared packaged paper models and scientific results are read-only.')
    st.subheader('Download the evidence')
    files = ['aggregate_performance.csv', 'test_metrics.csv', 'validation_candidates.csv', 'source_audit.json',
             'candidate_metrics.csv', 'shap_alignment.csv', 'domain_distance.csv', 'budget_summary.csv',
             'episode_summary.csv', 'calibration_metrics.csv', 'aggregate_timing.csv', 'revision_facts.json']
    files += [v for v in ['station_context.csv', 'geographic_mechanisms.csv'] if (DATA/v).exists()]
    chosen = st.selectbox('Evidence file', files)
    st.download_button('Download selected evidence', (DATA/chosen).read_bytes(), chosen)
    st.markdown('[Source code and deployment](https://github.com/sradfar/STARX) · [NOAA CO-OPS](https://api.tidesandcurrents.noaa.gov/api/prod/) · [GFS data via Open-Meteo](https://open-meteo.com/en/docs/gfs-api)')


PAGES = {'Forecast & insights': forecast_page, 'Transfer explorer': transfer_page, 'Update models': update_page,
         'Verify & monitor': monitor_page, 'Explain forecasts': explain_page, 'Paper evidence': evidence_page,
         'Methods & downloads': methods_page}
PAGES[page]()
st.divider()
st.caption('STARX · Princeton University research · Soheil Radfar & Ning Lin · Open model evaluation and coastal forecast decision support')
