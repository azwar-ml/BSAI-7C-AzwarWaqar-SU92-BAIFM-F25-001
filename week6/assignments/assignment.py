"""Reproducible Week 6/7 assignment pipeline; raw inputs are never modified."""
from pathlib import Path
import contextlib
import io
import json
import textwrap
import shutil
import importlib.metadata
import importlib.util
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import vizlib
import tslib
import geolib

HERE = Path(__file__).resolve().parent
WEEK = 6
DATA = HERE / 'data'
BLUE, ORANGE, GREEN = vizlib.PALETTE['series']
MODES = ['Metro', 'BRT', 'Bus', 'Ferry']
SOURCE = f'Source: synthetic CDB601220 Week {WEEK}; local operator returns, not real-world evidence.'
vizlib.style_defaults()


def write(name, text):
    (HERE / name).write_text('\n'.join(line.rstrip() for line in text.splitlines()).rstrip() + '\n', encoding='utf-8')


def save(fig, name, caption):
    fig.text(.02, .025, caption + '\n' + SOURCE, fontsize=8, va='bottom')
    fig.savefig(HERE / name, dpi=300, bbox_inches='tight')
    plt.close(fig)


def dates(s):
    """ISO main format, explicitly day-first slash returns; no inferred ambiguity."""
    slash = s.str.contains('/')
    result = pd.Series(pd.NaT, index=s.index, dtype='datetime64[ns]')
    result.loc[slash] = pd.to_datetime(s[slash], format='%d/%m/%Y', errors='raise')
    result.loc[~slash] = pd.to_datetime(s[~slash], format='%Y-%m-%d', errors='raise')
    return result


def process():
    decisions = []
    def decision(what, detection, rule, n):
        decisions.append([what, detection, rule, int(n)])
        print(f'{what}: {rule}; rows affected={n}')
    stream = io.StringIO()
    with contextlib.redirect_stdout(stream):
        names = ['district_grid', 'grid_daily', 'generation_mix'] if WEEK == 6 else ['district_transit', 'routes', 'ridership_daily', 'fare_revenue']
        raw = {n: pd.read_csv(DATA / (n + '.csv')) for n in names}
        feats = geolib.read_geojson(DATA / 'regions.geojson')
        codes = {f['code'] for f in feats}
        for name, frame in raw.items():
            print('\nRAW', name, 'rows', len(frame))
            print(frame.dtypes.to_string())
            print('Numeric ranges and denominator audit:\n', frame.describe().to_string())
            print('Missing per column:', frame.isna().sum().to_dict())
            print('Exact duplicate rows:', frame.duplicated().sum())
        district = raw[names[0]].copy()
        print('Register duplicate keys:', district.code.duplicated().sum())
        print('GeoJSON key matches as filed:', len(codes & set(district.code)))
        if WEEK == 6:
            # Canonical district names in GeoJSON disambiguate the conflicting D11 entry.
            canonical = {f['code']: f['name'] for f in feats}
            bad = district.district.ne(district.code.map(canonical))
            decision('Conflicting district duplicate', 'Name differs from GeoJSON canonical name', 'Drop noncanonical duplicate; retain canonical Kotla', bad.sum())
            district = district[~bad].copy()
            bad = district.population.div(district.households).gt(20)
            decision('Household scale', 'Persons per household exceeds 20; value recorded in thousands', 'Multiply household entry by 1000, consistent with register unit', bad.sum())
            district.loc[bad, 'households'] *= 1000
            pct = district.electrification_pct.astype('string')
            decision('Percent suffix', 'Literal percent sign', 'Strip suffix and parse numeric; preserve genuine blanks', pct.str.contains('%', na=False).sum())
            district.electrification_pct = pd.to_numeric(pct.str.rstrip('%'))
            bad = district.outage_minutes.lt(0)
            decision('Duration sentinel', 'Negative minutes', 'Replace -1 with missing, never zero', bad.sum())
            district.loc[bad, 'outage_minutes'] = np.nan
        else:
            repaired = district.code.str.upper().str.replace(' ', '', regex=False)
            decision('Register join keys', 'Case and internal-space mismatch', 'Uppercase and remove spaces', repaired.ne(district.code).sum())
            district.code = repaired
            bad = ~district.code.isin(codes)
            decision('Outside federation', 'Key absent from all 24 boundary features', 'Exclude D00 overlapping capital territory from federation sum', bad.sum())
            district = district[~bad].copy()
            bad = district.boardings_annual.div(district.population).lt(1)
            decision('Boarding scale', 'Under one boarding per resident, isolated thousand-unit entries', 'Multiply three thousand-unit entries by 1000; rounding uncertainty retained', bad.sum())
            district.loc[bad, 'boardings_annual'] *= 1000
            bad = district.fare_concession_pct.between(0, 1)
            decision('Concession fraction', 'Fraction among percentage entries', 'Multiply fractional entries by 100', bad.sum())
            district.loc[bad, 'fare_concession_pct'] *= 100
            bad = district.road_km.le(0)
            decision('Road sentinel', 'Nonpositive road length', 'Replace with missing', bad.sum())
            district.loc[bad, 'road_km'] = np.nan
        assert district.code.is_unique
        assert len(district) == 24
        assert district.population.sum() == 57857000
        assert district.households.sum() == 8908512
        print('GeoJSON join: matched', len(codes & set(district.code)), 'unmatched', len(codes - set(district.code)))
        print('Clean denominator ranges:', district[['population', 'households', 'area_km2']].agg(['min', 'max']).to_dict())
        daily_name = 'grid_daily' if WEEK == 6 else 'ridership_daily'
        daily = raw[daily_name].copy()
        n_slash = daily.date.str.contains('/').sum()
        decision('Mixed date formats', 'Slash vs ISO strings', 'Parse slash dates as DD/MM/YYYY and ISO as YYYY-MM-DD explicitly', n_slash)
        if WEEK == 7:
            slash = daily.loc[daily.date.str.contains('/'), 'date']
            wrong = pd.to_datetime(slash, format='%m/%d/%Y', errors='coerce')
            print('Month-first failures', wrong.isna().sum(), 'silent reinterpretations', ((wrong.notna()) & wrong.ne(dates(slash))).sum())
        daily.date = dates(daily.date)
        n = daily.duplicated().sum()
        decision('Exact re-submissions', 'All columns identical', 'Keep one identical submission', n)
        daily = daily.drop_duplicates().copy()
        group = 'zone' if WEEK == 6 else 'mode'
        if WEEK == 6:
            n = daily.duplicated(['date', group]).sum()
            decision('Conflicting daily submissions', 'Same date-zone and meter, different peak and duration', 'One zone-day: retain common meter and average disputed measurements; not a known true value', n)
            daily = daily.groupby(['date', group], as_index=False).mean(numeric_only=True)
            bad = daily.peak_load_mw.le(0)
            decision('Peak sentinel', 'Nonpositive peak MW', 'Replace with missing', bad.sum())
            daily.loc[bad, 'peak_load_mw'] = np.nan
        else:
            bad = (daily.trips_operated.gt(daily.trips_scheduled) | daily.trips_on_time.gt(daily.trips_operated) | daily[['trips_scheduled', 'trips_operated', 'trips_on_time', 'boardings', 'delay_minutes']].lt(0).any(axis=1))
            decision('Impossible daily counts', 'Negative counts or on-time > operated or operated > scheduled', 'Preserve filed slot; mask all measurement columns on impossible row (no invented repairs)', bad.sum())
            daily['invalid_return'] = bad
            daily.loc[bad, ['trips_scheduled', 'trips_operated', 'trips_on_time', 'boardings', 'delay_minutes']] = np.nan
        daily = daily.sort_values([group, 'date']).reset_index(drop=True)
        slots = pd.MultiIndex.from_product([pd.date_range(daily.date.min(), daily.date.max()), sorted(daily[group].unique())], names=['date', group])
        filed = daily.set_index(['date', group])
        missing = slots.difference(filed.index)
        print('Filed rows', len(daily), 'span', daily.date.min(), daily.date.max(), 'slots', len(slots), 'never filed', len(missing))
        print('Missing by category:', pd.Series(missing.get_level_values(group)).value_counts().to_dict())
        for label, block in daily.groupby(group):
            print('Calendar audit', label, tslib.audit_index(block.set_index('date'), 'D'))
        assert len(daily) == (3046 if WEEK == 6 else 2129)
        assert len(missing) == (128 if WEEK == 6 else 63)
        clean = {names[0]: district, daily_name: daily}
        if WEEK == 6:
            daily['meter_change_mwh'] = daily.groupby('zone').cumulative_mwh.diff()
            daily['interval_days'] = daily.groupby('zone').date.diff().dt.days
            daily['energy_mwh'] = daily.meter_change_mwh.where(daily.interval_days.eq(1) & daily.meter_change_mwh.ge(0))
            decision('Counter reset', 'Negative consecutive meter difference', 'Flag reset interval as unknown; do not sum meter levels or bridge gaps', daily.meter_change_mwh.lt(0).sum())
            decision('Absent zone-days', 'Full date-zone calendar minus submissions', 'Keep absent slots missing; invalidate energy differences spanning gaps', len(missing))
            generation = raw['generation_mix']
            parts = generation[generation.subsource.ne('Total')].copy()
            totals = generation[generation.subsource.eq('Total')].set_index('month').mwh
            diff = parts.groupby('month').mwh.sum() - totals
            decision('Published generation totals', 'All sources / Total duplicates sum of children', 'Exclude total rows from composition; retain separate summaries', len(generation) - len(parts))
            print('Parts', len(parts), 'max reconciliation error MWh', diff.abs().max(), 'all-row percent', generation.mwh.sum() / parts.mwh.sum() * 100)
            assert len(parts) == 144 and diff.abs().max() <= 2
            clean.update(generation_mix=parts, generation_summary=generation[generation.subsource.eq('Total')])
        else:
            routes = raw['routes'].copy()
            replacement = routes['mode'].replace({'brt': 'BRT', 'Bus Rapid Transit': 'BRT'})
            decision('Route mode labels', 'Six spellings for four modes', 'Normalize brt and Bus Rapid Transit to BRT', routes['mode'].ne(replacement).sum())
            routes['mode'] = replacement
            for col, bad in [('punctuality_pct', ~routes.punctuality_pct.between(0, 100)), ('cost_recovery_pct', routes.cost_recovery_pct.le(0))]:
                decision(col + ' sentinel', 'Out-of-domain operator code', 'Mask code as missing; do not impute', bad.sum())
                routes.loc[bad, col] = np.nan
            import dashlib
            print('ROUTE OVERLAP', dashlib.overlap_factor(routes, 'route_id', 'districts_served', 'boardings_annual'))
            print('Route join misses:', sum(c not in codes for s in routes.districts_served for c in s.split(';')))
            fare = raw['fare_revenue']
            parts = fare[fare.ticket_type.ne('All tickets')].copy()
            for value in ['trips', 'revenue_pkr']:
                result = dashlib.drilldown_check(fare, 'ticket_type', value, total_label='All tickets', within=['month', 'mode'])
                print('FARE RECONCILIATION', value, result)
                assert result['reconciles']
            decision('Fare summary rows', 'All tickets equals three fare-product children', 'Exclude summaries from parts; retain summary separately', len(fare) - len(parts))
            print('MODE COUNTS', routes['mode'].value_counts().to_dict())
            clean.update(routes=routes, fare_revenue=parts, fare_summary=fare[fare.ticket_type.eq('All tickets')])
            plan = [{'panel': 'punctuality', 'x': 'period', 'y': 'on_time_pct'}, {'panel': 'decomposition', 'x': 'component', 'y': 'percentage_points'}, {'panel': 'map', 'lightness': 'boardings_per_1000_residents'}, {'panel': 'fare', 'x': 'month', 'y': 'index'}]
            print('CHANNEL PLAN', dashlib.channel_conflicts(plan))
            assert dashlib.channel_conflicts(plan)['ok']
        print('STATIC CHART PALETTE', vizlib.cvd_report([BLUE, ORANGE, GREEN]))
        from matplotlib.colors import to_hex
        shades = [to_hex(plt.get_cmap(vizlib.PALETTE['sequential'])(.18 + .78 * i / 4)) for i in range(5)]
        print('SCREEN MAP PALETTE actually used:', shades)
        print('SCREEN MAP CVD', vizlib.cvd_report(shades))
        pair = vizlib.worst_pair(shades, 'deutan')
        print('SCREEN MAP worst all-pairs CVD distance:', pair[-1])
        assert pair[-1] >= 8
        print('Screen line/bar ink:', BLUE, '; series use styles/text, no categorical hue mapping.')
        (HERE / 'clean').mkdir(exist_ok=True)
        for name, frame in clean.items():
            frame.to_csv(HERE / 'clean' / (name + '.csv'), index=False)
        shutil.copyfile(DATA / 'regions.geojson', HERE / 'clean' / 'regions.geojson')
        assert (district[['population', 'households', 'area_km2']] > 0).all().all()
        print('Raw inputs preserved. Cleaning complete.')
    audit = stream.getvalue()
    print(audit)
    write('partA_audit.txt', audit)
    write('partA_decisions.md', '| What | How detected | Rule applied | Rows affected |\n|---|---|---|---:|\n' + '\n'.join('| ' + ' | '.join(map(str, row)) + ' |' for row in decisions) + '\n')
    return clean


def load():
    if not (HERE / 'clean').exists():
        process()
    out = {p.stem: pd.read_csv(p) for p in (HERE / 'clean').glob('*.csv')}
    for df in out.values():
        if 'date' in df:
            df.date = pd.to_datetime(df.date)
    return out


def check(name, actual, expected, tolerance=0):
    good = abs(actual - expected) <= tolerance
    print(f'{"PASS" if good else "FAIL"} {name}: {actual} (expected {expected}, tolerance {tolerance})')
    assert good, name


def analysis():
    data = load()
    stream = io.StringIO()
    with contextlib.redirect_stdout(stream):
        d = data['district_grid' if WEEK == 6 else 'district_transit']
        value, den = ('outages', 'households') if WEEK == 6 else ('boardings_annual', 'population')
        rate = d[value] / d[den] * 1000
        check('districts', len(d), 24)
        check('population', d.population.sum(), 57857000)
        check('households', d.households.sum(), 8908512)
        count_top = d.nlargest(3, value).code.tolist()
        rate_top = d.loc[rate.nlargest(3).index, 'code'].tolist()
        assert count_top == (['D17', 'D13', 'D24'] if WEEK == 6 else ['D17', 'D09', 'D13'])
        assert rate_top == (['D12', 'D11', 'D13'] if WEEK == 6 else ['D12', 'D21', 'D09'])
        check('top-list intersection', len(set(count_top) & set(rate_top)), 1)
        print('Raw-count top three:', count_top)
        print('Rate top three:', d.loc[rate.nlargest(3).index, 'code'].tolist())
        check('count-population correlation', d[value].corr(d[den]), .991 if WEEK == 6 else .945, .0005)
        check('rate-population correlation', rate.corr(d[den]), .175 if WEEK == 6 else .249, .0005)
        print('Rate range', rate.min(), rate.max(), 'pooled', d[value].sum() / d[den].sum() * 1000)
        big = d.nlargest(5, 'population')
        check('top-five population share', big.population.sum() / d.population.sum() * 100, 59.9, .05)
        check('top-five area share', big.area_km2.sum() / d.area_km2.sum() * 100, 12.6, .05)
        if WEEK == 6:
            daily = data['grid_daily']
            check('interruptions', d.outages.sum(), 388879)
            pph = d.population / d.households
            check('persons per household min', pph.min(), 5.70, .005)
            check('persons per household max', pph.max(), 7.36, .005)
            check('electrification min', d.electrification_pct.min(), 66.6, .001)
            check('electrification max', d.electrification_pct.max(), 90.3, .001)
            duration = d.outage_minutes / d.outages
            check('mean duration min', duration.min(), 77, .5)
            check('mean duration max', duration.max(), 214, .5)
            check('electrification missing', d.electrification_pct.isna().sum(), 3)
            check('duration usable', d.outage_minutes.notna().sum(), 22)
            check('peak unusable', daily.peak_load_mw.isna().sum(), 46)
            check('meter resets', daily.meter_change_mwh.lt(0).sum(), 1)
            june = daily[(daily.zone == 'P1') & daily.date.between('2026-06-01', '2026-06-30')]
            p1 = daily[daily.zone.eq('P1')].set_index('date').cumulative_mwh
            brief_energy = p1.loc['2026-07-01'] - p1.loc['2026-06-01']
            check('brief June anchor (June 1 to July 1 readings)', brief_energy, 1200239, 1)
            print('Dictionary end-of-day June energy (May 31 to June 30):', p1.loc['2026-06-30'] - p1.loc['2026-05-31'])
            print('Known consecutive-day June energy only:', june.energy_mwh.sum(), '; one two-day interval excluded from daily allocations')
            check('incorrect meter level sum', june.cumulative_mwh.sum(), 574845315, 1)
            gen = data['generation_mix'].pivot(index='month', columns='subsource', values='mwh')
            total = gen.sum(axis=1)
            check('generation change pct', (total.iloc[-1] / total.iloc[0] - 1) * 100, -5, .05)
            check('solar MWh change pct', (gen.Solar.iloc[-1] / gen.Solar.iloc[0] - 1) * 100, 82.1, .05)
            print('Solar share endpoints:', (gen.Solar / total * 100).iloc[[0, -1]].tolist())
            freq = np.linspace(0, .5, 501)
            response = pd.DataFrame({'cycles_per_day': freq})
            for w in (7, 28, 91):
                response[f'gain_{w}d'] = np.abs(np.sinc(w * freq) / np.sinc(freq))
            response.to_csv(HERE / 'partB_filter_response.csv', index=False)
            series = daily[daily.zone.eq('P1')].set_index('date').peak_load_mw.asfreq('D')
            print('WINDOW SENSITIVITY', tslib.window_report(series, windows=(7, 28, 91)))
            for window in [7, 28, 91]:
                print('Moving-average filter', window, 'days: DC gain=1; first zero cycles/day=', 1 / window, '; full-window required, gaps never bridged')
        else:
            import dashlib
            routes, daily, fare = data['routes'], data['ridership_daily'], data['fare_revenue']
            check('register annual boardings', d.boardings_annual.sum(), 873564130)
            check('route annual boardings', routes.boardings_annual.sum(), 873564820)
            overlap = dashlib.overlap_factor(routes, 'route_id', 'districts_served', 'boardings_annual')
            check('attributed route boardings', overlap['attributed_total'], 2704739571)
            check('overlap', overlap['factor'], 3.096, .0005)
            check('punctuality usable', routes.punctuality_pct.notna().sum(), 33)
            check('road usable', d.road_km.notna().sum(), 22)
            check('fare journeys', fare.trips.sum(), 934051014)
            check('fare revenue', fare.revenue_pkr.sum(), 36669953119)
            annual = daily[daily.date.between('2025-09-01', '2026-08-31')].boardings.sum()
            check('route mode count', routes['mode'].nunique(), 4)
            assert routes['mode'].value_counts().to_dict() == {'Bus':22, 'BRT':6, 'Metro':4, 'Ferry':4}
            check('concession min', d.fare_concession_pct.min(), 20.9, .001)
            check('concession max', d.fare_concession_pct.max(), 42.6, .001)
            check('fare mean PKR', fare.revenue_pkr.sum() / fare.trips.sum(), 39.3, .05)
            print('Annual boardings under impossible-row masking policy:', annual, '; coverage %', annual / d.boardings_annual.sum() * 100)
            # The published anchor includes all filed boarding counts, including impossible operational returns.
            filed = pd.read_csv(DATA / 'ridership_daily.csv')
            filed.date = dates(filed.date)
            filed = filed.drop_duplicates()
            published = filed[filed.date.between('2025-09-01', '2026-08-31')].boardings.sum()
            check('filed annual boardings anchor (before invalid masking)', published, 845531940)
            print('Excluded impossible-row boardings, annual:', published - annual)
            ratio = daily.boardings.sum() / fare.trips.sum()
            check('boardings per journey', ratio, 1.303, .0021)
            dec = decomposition(daily)
            for key, expected in [('rate_before', 78.87), ('rate_after', 80.60), ('change', 1.73), ('within', -1.10), ('mix', 2.82)]:
                check(key, dec[key], expected, .005)
            assert dec['reversal']
            print(dec['groups'].to_string())
            for months in [3, 6]:
                before = daily[daily.date.between('2025-04-01', '2025-06-30' if months == 3 else '2025-09-30')]
                after = daily[daily.date.between('2026-06-01' if months == 3 else '2026-03-01', '2026-08-31')]
                print('Sensitivity first/last complete months', months, dashlib.rate_decomposition(before, after, 'mode', 'trips_on_time', 'trips_operated'))
            monthly = fare.groupby('month')[['trips', 'revenue_pkr']].sum().loc[:'2026-08']
            for col, expected in [('trips', 12.1), ('revenue_pkr', 26.9)]:
                check(col + ' growth pct', (monthly[col].iloc[-1] / monthly[col].iloc[0] - 1) * 100, expected, .05)
            mode = fare.groupby(['month', 'mode'])[['trips', 'revenue_pkr']].sum()
            a, b = mode.loc['2025-04'], mode.loc['2026-08']
            print('Journey growth by mode %:', (b.trips / a.trips - 1).mul(100).to_dict())
            print('Final shares journeys:', b.trips.div(b.trips.sum()).mul(100).to_dict())
            print('Final shares revenue:', b.revenue_pkr.div(b.revenue_pkr.sum()).mul(100).to_dict())
            # Exact symmetric decomposition of average fare: within-mode yield vs mix.
            wa, wb = a.trips / a.trips.sum(), b.trips / b.trips.sum()
            ra, rb = a.revenue_pkr / a.trips, b.revenue_pkr / b.trips
            print('Average fare change PKR:', (wb * rb).sum() - (wa * ra).sum(), 'within-mode fare/product-yield', (((wa + wb) / 2) * (rb - ra)).sum(), 'mode mix', (((ra + rb) / 2) * (wb - wa)).sum())
            print('Yield includes ticket-product mix; these files do not establish a causal tariff increase.')
    report = stream.getvalue()
    print(report)
    write('partB_metrics.txt', report)


def decomposition(daily):
    import dashlib
    a = daily[daily.date.between('2025-04-01', '2025-06-30')]
    b = daily[daily.date.between('2026-04-01', '2026-06-30')]
    return dashlib.rate_decomposition(a, b, 'mode', 'trips_on_time', 'trips_operated')


def map_panel(ax, d):
    col, den = ('outages', 'households') if WEEK == 6 else ('boardings_annual', 'population')
    values = dict(zip(d.code, d[col] / d[den] * 1000))
    feats = geolib.read_geojson(DATA / 'regions.geojson')
    edges = geolib.choropleth(ax, geolib.read_geojson(DATA / 'regions.geojson'), values, scheme='quantile', k=5, legend_title='Interruptions per 1,000 households' if WEEK == 6 else 'Boardings per 1,000 residents')
    lat_ts = np.mean([lat for f in feats for lon, lat in f['ring']])
    for f in feats:
        x, y = geolib.equal_area_xy(*geolib.representative_point(f['ring']), lat_ts=lat_ts)
        ax.text(float(x), float(y), f['code'], ha='center', va='center', fontsize=6, color='black', bbox={'facecolor':'white', 'alpha':.75, 'edgecolor':'none', 'pad':.5})
    ax.set_title('D12 leads on household interruption rate' if WEEK == 6 else 'D12 has the highest boardings per resident', fontsize=11)
    ax.text(0, -.13, 'Equal-area; 5 quantile classes; edges: ' + ', '.join(f'{v:,.1f}' for v in edges), transform=ax.transAxes, fontsize=7, wrap=True)


def ranking(ax, d):
    col, den = ('outages', 'households') if WEEK == 6 else ('boardings_annual', 'population')
    s = pd.Series((d[col] / d[den] * 1000).values, index=d.code).sort_values()
    ax.barh(s.index, s.values, color=BLUE)
    ax.set_xlim(left=0)
    ax.set_xlabel('Interruptions per 1,000 households' if WEEK == 6 else 'Annual boardings per 1,000 residents')
    ax.set_title('Prioritize D12, D11 and D13' if WEEK == 6 else 'D12, D21 and D09 lead on use per resident', fontsize=11)
    ax.tick_params(axis='y', labelsize=7)


def trend(ax, data):
    daily = data['grid_daily' if WEEK == 6 else 'ridership_daily']
    if WEEK == 6:
        s = daily[daily.zone.eq('P1')].set_index('date').peak_load_mw.asfreq('D')
        ax.plot(s.index, s, color=BLUE, alpha=.4, lw=.7, label='Daily peak')
        ax.plot(s.index, s.rolling(28, min_periods=28).mean(), color=ORANGE, label='28-day trailing mean')
        ax.set_ylabel('Peak demand (MW; level, never summed)')
        ax.set_title('P1 peak demand rises into the summer', fontsize=11)
    else:
        # A gap or invalid return in any mode makes the network daily total unknown.
        table = daily.pivot(index='date', columns='mode', values='boardings').asfreq('D')
        s = table.sum(axis=1, min_count=4)
        ax.plot(s.index, s / 1e6, color=BLUE, lw=.8, label='All four modes reported')
        ax.axvspan(pd.Timestamp('2026-02-09'), pd.Timestamp('2026-02-22'), color='.8', alpha=.5, label='14-day mode reporting gap')
        ax.axvspan(pd.Timestamp('2026-09-01'), pd.Timestamp('2026-09-30'), color='.9', label='Final period incomplete')
        ax.set_ylabel('Daily boardings (millions; not journeys)')
        ax.set_title('Reporting gaps prevent a continuous network total', fontsize=11)
    ax.legend(fontsize=7)
    ax.tick_params(axis='x', rotation=25)
    ax.set_xlabel('Reporting date; gaps remain missing')


def composition(ax, data):
    if WEEK == 6:
        p = data['generation_mix'].pivot_table(index='month', columns='subsource', values='mwh', aggfunc='sum')
        solar = p.Solar
        s = pd.DataFrame({'Solar': solar, 'Other generation': p.sum(axis=1) - solar})
        title = 'Solar rises 82.1% in MWh while total generation falls 5.0%'
        unit = 'Share of generated MWh (%)'
    else:
        fare = data['fare_revenue']
        p = fare[fare.month.le('2026-08')].pivot_table(index='month', columns='mode', values='trips', aggfunc='sum')
        s = pd.DataFrame({'Metro': p.Metro, 'Other modes': p.sum(axis=1) - p.Metro})
        title = 'Metro journey share rises as total journeys grow 12.1%'
        unit = 'Share of journeys (%; not revenue)'
    s = s.div(s.sum(axis=1), axis=0) * 100
    x = pd.to_datetime(s.index)
    ax.stackplot(x, s.iloc[:, 0], s.iloc[:, 1], colors=[BLUE, GREEN], labels=s.columns)
    ax.set_ylim(0, 100)
    ax.set_ylabel(unit)
    ax.set_title(title, fontsize=10)
    ax.legend(fontsize=7, loc='upper left')
    ax.tick_params(axis='x', rotation=25)


def relationship(ax, data):
    r = data['routes'].dropna(subset=['punctuality_pct', 'crowding_index'])
    ax.scatter(r.crowding_index, r.punctuality_pct, color=BLUE, s=24)
    ax.set_xlabel('Peak passengers / vehicle capacity (ratio)')
    ax.set_ylabel('On-time departures (%)')
    ax.set_title('Route crowding and reliability require separate scrutiny', fontsize=10)
    ax.text(.02, .03, f'n={len(r)} routes; 3 sentinel punctualities excluded\nNo causal model inferred from this cross-section.', transform=ax.transAxes, fontsize=8)


def charts():
    data = load()
    d = data['district_grid' if WEEK == 6 else 'district_transit']
    funcs = [lambda ax: ranking(ax, d), lambda ax: trend(ax, data), lambda ax: composition(ax, data)] if WEEK == 6 else [lambda ax: trend(ax, data), lambda ax: composition(ax, data), lambda ax: map_panel(ax, d), lambda ax: relationship(ax, data)]
    captions = (['Decision: allocate reliability support by household rate, not district size; 12-month register, 24 districts.', 'Decision: plan P1 peak capacity; daily level and 28-day trailing full-window mean; missing peaks retained.', 'Decision: distinguish solar growth from total contraction; Apr 2025-Sep 2026; totals excluded; Other = all non-solar.'] if WEEK == 6 else ['Decision: repair reporting before claiming service growth; Apr 2025-Sep 2026; absent/invalid days not zero.', 'Decision: plan journey capacity; Apr 2025-Aug 2026 complete months; Other = BRT+Bus+Ferry; summaries excluded.', 'Decision: compare corridor use per resident; 12 months ending Aug 2026; 24 districts, no suppression.', 'Decision: review crowded routes; annual route register; complete cases only; unweighted route observations.'])
    for i, (draw, caption) in enumerate(zip(funcs, captions), 1):
        fig, ax = plt.subplots(figsize=(10, 7))
        fig.subplots_adjust(bottom=.23, top=.9)
        draw(ax)
        save(fig, f'partB_{i}.png', caption)
    if WEEK == 6:
        fig, axes = plt.subplots(1, 2, figsize=(12, 8), gridspec_kw={'width_ratios': [1.4, 1]})
        fig.subplots_adjust(bottom=.22, wspace=.55, top=.88)
        map_panel(axes[0], d)
        ranking(axes[1], d)
        fig.suptitle('Reliability funding should follow household exposure', x=.05, ha='left', weight='bold')
        save(fig, 'partC_figure.png', 'Board reader: National Grid Authority budget committee. Annual register; no district suppressed; missing durations do not affect interruption counts.')
    notes()


def punctuality(ax, daily):
    dec = decomposition(daily)
    g = dec['groups'].reindex(MODES)
    labels = ['Network'] + MODES
    a = [dec['rate_before']] + g.rate_before.tolist()
    b = [dec['rate_after']] + g.rate_after.tolist()
    for i, (x, y) in enumerate(zip(a, b)):
        ax.plot([x, y], [i, i], color=BLUE, marker='o')
        ax.text(max(x, y) + .4, i, f'{y-x:+.2f} pts', fontsize=8, va='center')
    ax.set_yticks(range(5), labels)
    ax.invert_yaxis()
    ax.set_xlim(68, 103)
    ax.set_xlabel('On-time / operated departures (%); points label change')
    ax.set_title('Network improves; every mode deteriorates', fontsize=11)
    ax.text(.01, -.22, '2025 Q2 to 2026 Q2; ratios of summed counts; endpoints linked.', transform=ax.transAxes, fontsize=8)


def fare_panel(ax, data):
    f = data['fare_revenue']
    s = f[f.month.le('2026-08')].groupby('month')[['trips', 'revenue_pkr']].sum()
    for col, style, label in [('trips', '-', 'Journeys'), ('revenue_pkr', '--', 'Revenue PKR')]:
        index = s[col] / s[col].iloc[0] * 100
        ax.plot(pd.to_datetime(s.index), index, style, color=BLUE, label=label)
    ax.set_title('Revenue outgrows journeys: yield and mix both matter', fontsize=10)
    ax.set_ylabel('Index (Apr 2025 = 100; separate units normalized)')
    ax.tick_params(axis='x', rotation=25)
    ax.legend(fontsize=8)
    ax.text(.01, -.25, 'Apr 2025-Aug 2026; fare products only; Sep excluded.\nAverage fare +4.80 PKR: within-mode yield +0.24; mode mix +4.56.', transform=ax.transAxes, fontsize=7)


def screen():
    data = load()
    if WEEK == 6:
        fig, axes = plt.subplots(2, 2, figsize=(14, 12))
        fig.subplots_adjust(hspace=.6, wspace=.45, bottom=.13, top=.89)
        map_panel(axes[0, 0], data['district_grid'])
        ranking(axes[0, 1], data['district_grid'])
        trend(axes[1, 0], data)
        composition(axes[1, 1], data)
        fig.suptitle('Prioritize household reliability; solar growth survives a shrinking total', x=.05, ha='left', weight='bold')
        save(fig, 'partD_figure.png', 'Annual district rates; P1 daily peak with 28-day trailing mean; generation Apr 2025-Sep 2026. See panel labels for denominators and exclusions.')
    else:
        import dashlib
        fig = plt.figure(figsize=(15, 11))
        ax = dashlib.dashboard_grid(fig, [['service', 'decomp'], ['map', 'fare']], headline='Punctuality rises through the mode mix; every mode runs less reliably', standfirst='2025 Q2 vs 2026 Q2: on-time / operated departures. Corridor: boardings per 1,000 residents, 12mo to Aug 2026. Fare: journeys.', hspace=.7, wspace=.45, top=.84)
        punctuality(ax['service'], data['ridership_daily'])
        dec = decomposition(data['ridership_daily'])
        values = [dec['within'], dec['mix'], dec['change']]
        ax['decomp'].bar(['Within modes', 'Mode mix', 'Net change'], values, color=BLUE)
        ax['decomp'].axhline(0, color=vizlib.PALETTE['ink'], lw=.8)
        ax['decomp'].set_ylim(-1.8, 3.5)
        for i, v in enumerate(values):
            ax['decomp'].text(i, v + (.1 if v >= 0 else -.28), f'{v:+.2f}', ha='center', fontsize=10)
        ax['decomp'].set_ylabel('Percentage-point contribution (Kitagawa)')
        ax['decomp'].set_title('More Metro departures outweigh worse service', fontsize=11)
        map_panel(ax['map'], data['district_transit'])
        fare_panel(ax['fare'], data)
        fig.subplots_adjust(bottom=.16)
        save(fig, 'partD_screen.png', 'Read: service -> explanation -> corridor use -> fare box. Missing returns excluded, never zero; no district suppression.\nHue is constant blue ink; map lightness alone encodes boardings per resident. All periods and units are panel-specific.')
        shutil.copyfile(HERE / 'partD_screen.png', HERE / 'partD_figure.png')
    notes()


def interactive():
    import plotly.graph_objects as go
    data = load()
    if WEEK == 6:
        d = data['grid_daily']
        labels, group, col, unit = sorted(d.zone.unique()), 'zone', 'peak_load_mw', 'Peak demand (MW; daily level)'
    else:
        d = data['ridership_daily']
        labels, group, col, unit = MODES, 'mode', 'boardings', 'Daily boardings (not journeys)'
    fig = go.Figure()
    for label in labels:
        series = d[d[group].eq(label)].set_index('date')[col].asfreq('D')
        # Keep final incomplete reporting period out of selectable domain in Week 7.
        if WEEK == 7:
            series = series.loc[:'2026-08-31']
        fig.add_trace(go.Scatter(x=series.index, y=series.where(series.notna(), None).tolist(), name=label, mode='lines', connectgaps=False, line={'color': BLUE, 'dash': ['solid', 'dash', 'dot', 'dashdot', 'longdash', 'longdashdot'][labels.index(label)]}, visible=True))
    buttons = [dict(label='Reset: all', method='update', args=[{'visible': [True] * len(labels)}])]
    buttons += [dict(label=label, method='update', args=[{'visible': [x == label for x in labels]}]) for label in labels]
    fig.update_layout(title=f'Week {WEEK}: filed daily returns; gaps are unknown, never zero', yaxis_title=unit, xaxis_title='Date (daily; no resampling)', template='plotly_white', updatemenus=[dict(buttons=buttons, x=1.02, y=1)], annotations=[dict(text=SOURCE + '<br>Select one category by menu or legend; Reset: all restores traces; double-click axes restores date range.', x=0, y=-.3, xref='paper', yref='paper', showarrow=False)], margin={'b': 170}, legend_title='Directly named traces (shared blue ink)')
    fig.update_xaxes(rangeslider_visible=True)
    if WEEK == 7:
        fig.add_vrect(x0='2026-02-09', x1='2026-02-22', fillcolor='gray', opacity=.25, line_width=0, annotation_text='Missing mode returns')
        fig.add_annotation(text='Sep 2026 excluded: incomplete reporting. No district filter: routes overlap 3.096x.', x=0, y=1.12, xref='paper', yref='paper', showarrow=False)
    assert len(fig.data) == len(labels)
    assert all(t.connectgaps is False for t in fig.data)
    assert buttons[0]['args'][0]['visible'] == [True] * len(labels)
    assert all(sum(b['args'][0]['visible']) == 1 for b in buttons[1:])
    if WEEK == 7:
        assert max(max(t.x) for t in fig.data) == pd.Timestamp('2026-08-31')
    print('PASS interaction state: all-category reset, single-category controls, gaps, date domain')
    fig.write_html(HERE / ('partA_dashboard.html' if WEEK == 6 else 'partC_interactive.html'), include_plotlyjs=True)
    # Reproducible static default, with same all-category domain and daily level.
    static, ax = plt.subplots(figsize=(11, 6))
    static.subplots_adjust(bottom=.23, right=.83, top=.86)
    for i, label in enumerate(labels):
        s = d[d[group].eq(label)].set_index('date')[col].asfreq('D')
        if WEEK == 7:
            s = s.loc[:'2026-08-31']
        ax.plot(s.index, s, color=BLUE, lw=.8, linestyle=['-', '--', ':', '-.', (0, (5, 1)), (0, (3, 1, 1, 1, 1, 1))][i], label=label)
    ax.legend(bbox_to_anchor=(1.02, 1), loc='upper left')
    ax.set_ylabel(unit)
    ax.set_title(f'Week {WEEK}: filed daily returns; gaps are unknown, never zero', fontsize=12)
    if WEEK == 7:
        ax.axvspan(pd.Timestamp('2026-02-09'), pd.Timestamp('2026-02-22'), color='gray', alpha=.25)
    save(static, 'partC_default.png', 'Default: all categories; hover/legend/menu/zoom in HTML; missing slots break lines; no overlapping district totals.\n' + ('Sep 2026 excluded; grey = 14-day reporting gap.' if WEEK == 7 else 'Daily peak is a level; do not sum across days.'))
    notes()


def experiment():
    """Automated checks only. Never misrepresent synthetic responses as human trials."""
    data = load()
    rows = []
    daily = data['grid_daily' if WEEK == 6 else 'ridership_daily']
    group = 'zone' if WEEK == 6 else 'mode'
    rows.append({'check': 'unique_date_category', 'passed': not daily.duplicated(['date', group]).any(), 'kind': 'automated; not a reader trial'})
    if WEEK == 7:
        import dashlib
        rows.append({'check': 'all_modes_decline_aggregate_rises', 'passed': decomposition(daily)['reversal'], 'kind': 'automated; not a reader trial'})
        r = data['routes']
        mask = dashlib.select_in_rect(r, 'crowding_index', 'subsidy_pkr_per_trip', (0, 0, 2, 100))
        rows.append({'check': 'selection_contains_all_routes', 'passed': bool(mask.all()), 'kind': 'automated; not a reader trial'})
    results = pd.DataFrame(rows)
    results.to_csv(HERE / 'partC_results.csv', index=False)
    assert results.passed.all()
    print(results.to_string(index=False))
    write('partC_experiment_note.md', 'No reader responses were collected or fabricated. Week 6 explicitly requires no reader experiment. Week 7 Task C requires an interaction contract, not participant trials. The CSV contains automated checks only; optional Week 7 F2 (five real readers) is not claimed.\n')


def notes():
    if WEEK == 6:
        note = """# Reliability budget decision

For the National Grid Authority budget committee: prioritize D12, D11 and D13 on interruptions per 1,000 billed domestic households, rather than the largest district totals. A household denominator expresses service exposure and avoids treating population scale as poor reliability. It is a feeder-interruption count per connection, not a measured probability that a household lost power.

The map uses an equal-area projection because degrees are not area, five quantile classes to compare relative position across 24 districts, and explicit numeric edges below the panel. The companion ranking preserves the exact order hidden by classes. Bars start at zero; no district is suppressed or merged (0). Missing interruption duration in two districts is retained as missing; it does not remove their valid interruption counts. No uncertainty intervals are claimed without a sampling model.

The raw-count counterfactual prioritizes D17, D13 and D24. It would divert attention from D12 and D11. D13 appears on both lists because it has both a large connection base and a high interruption rate. Counts correlate with households at +0.991, but normalized rates at +0.175. The five largest districts hold 59.9% of residents on 12.6% of land, so the ranking accompanies the map to counter geographic ink bias.

Daily peak demand is a level: the trend uses P1 daily MW and a 28-day trailing full-window mean, which reduces noise and breaks at missing readings. Daily meter energy is differenced only across consecutive days, with resets unknown. The brief June anchor uses June 1 to July 1 readings (1,200,239 MWh); the dictionary defines end-of-day readings, making calendar June May 31 to June 30 (1,199,207 MWh). Both are reported rather than silently redefining the month. Generation composition excludes published total rows; Other combines all non-solar parts. April 2025 to September 2026 solar MWh rises 82.1% and share rises 4.1% to 7.9%, while the total falls 5.0%. This supports a real solar increase without concealing denominator contraction.
"""
        write('partC_note.md', note)
        write('partD_interaction.md', 'Hover exposes a dated zone peak and zoom isolates demand episodes that a static chart cannot resolve. A reader could cherry-pick a short interval or sum peak MW; axis wording, missing-line gaps and a resettable date range discourage those errors.\n')
    else:
        note = """# Transit board interpretation

For the Secretary, finance director and journalist: network punctuality improves through a shift toward Metro, while every mode becomes less punctual. Read service first, its decomposition beside it, then corridor use and the fare box. This order prevents an improving headline from being quoted as improved operations.

Use complete 2025 Q2 and 2026 Q2 for service, summing on-time and operated departures before division. The symmetric Kitagawa decomposition separates within-mode decline (about -1.10 points) from mode mix (+2.82). Invalid operational returns are masked and missing days are not zero. This policy slightly changes published anchors that retain impossible rows; the metrics report states the difference. Calendar gaps remain visible and the incomplete September 2026 period is excluded from fare comparisons and interaction.

The corridor map uses annual district-attributed boardings per 1,000 residents, five quantile classes with printed edges, and an equal-area projection. It describes use, not unmet demand or a causal case for construction. Raw boardings would prioritize D17, D09 and D13; rates prioritize D12, D21 and D09. A count-based allocation would take attention away from D12 and D21. D09 combines scale with high use per resident. Count-population correlation is +0.945 versus +0.249 for rates. No district is suppressed or merged (0); ranking should accompany geographic interpretation because 59.9% of people occupy 12.6% of land.

Fare comparisons use journeys, not boardings, in complete April 2025-August 2026 months. Revenue and journeys are indexed separately to 100; they are never added. Revenue grows 26.9% against 12.1% journeys; average yield and mode mix both contribute. Within-mode yield may include ticket-product mix and does not prove a tariff rise. Route district memberships overlap 3.096 times, so no district-sum filter is offered. Hue is constant blue ink, line styles and text identify series, and map lightness encodes only use per resident.
"""
        write('partE_note.md', note)
        write('partC_contract.md', """# Interaction contract

Default: all four modes show daily filed boardings from April 2025 through August 2026; gaps are unknown and not zero. Direct trace names, hover values and line patterns in the static default prevent reliance on hue identity. The static PNG uses the same date domain and measures.

The menu chooses one mode; Reset: all restores every trace. Legend clicks hide or restore modes, and double-clicking an axis restores the complete date extent. The range slider and drag zoom change the visible dates only, never the data, denominator, or aggregation. Hover reports the exact date, mode and boarding count. Daily counts remain daily at every zoom level. The missing February 9-22, 2026 mode-reporting interval is shaded and missing values break lines; impossible returns also remain gaps. September 2026 is outside the selectable data domain because two modes stopped reporting.

A short zoom must not invite a claim that missing service was zero or that a selected period was complete. Persistent warnings and broken lines guard that inference. No district filter is offered: summing whole-route boardings over served districts would inflate the network total 3.096 times. Modes are exclusive; their valid daily counts may be summed only when all four have reported. Reset restores visibility and the axis reset restores the original date window. The HTML embeds Plotly for offline use. No human-reader trials or optional five-person bonus are claimed.
""")
    write('partE_design.txt', note)


def pdf():
    from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
    from reportlab.lib.styles import getSampleStyleSheet
    from reportlab.lib import colors
    notes()
    styles = getSampleStyleSheet()
    styles['BodyText'].fontSize = 10
    styles['BodyText'].leading = 14
    styles['BodyText'].textColor = colors.HexColor(vizlib.PALETTE['ink'])
    paragraphs = (HERE / 'partE_design.txt').read_text(encoding='utf-8').split('\n\n')
    story = [Paragraph(f'Week {WEEK} | Visual design justification', styles['Title']), Spacer(1, 14)]
    for p in paragraphs[1:]:
        story += [Paragraph(p.strip().replace('\n', ' '), styles['BodyText']), Spacer(1, 10)]
    story += [Paragraph(SOURCE, styles['Italic'])]
    doc = SimpleDocTemplate(str(HERE / 'partE_justification.pdf'), pagesize=(595, 842), rightMargin=42, leftMargin=42, topMargin=38, bottomMargin=38)
    doc.build(story)
    from pypdf import PdfReader
    assert len(PdfReader(HERE / 'partE_justification.pdf').pages) == 1
    print('PASS one-page PDF')


def metadata():
    packages = ['numpy', 'pandas', 'matplotlib', 'scipy', 'seaborn', 'statsmodels', 'plotly', 'reportlab', 'pypdf', 'pymupdf']
    write('requirements.txt', '\n'.join(f'{p}=={importlib.metadata.version(p)}' for p in packages if importlib.util.find_spec(p if p != 'pymupdf' else 'fitz')) + '\n')
    commands = ['partA_process.py', 'partB_charts.py', 'partB_analytics.py' if WEEK == 6 else 'partB_analysis.py', 'partA_dashboard.py' if WEEK == 6 else 'partC_interactive.py', 'partC_experiment.py', 'partD_figure.py', 'generate_partE_pdf.py']
    lines = [f'# Week {WEEK} reproducible submission', 'Install the pinned requirements using the project virtual environment.', 'Run from this folder or pass each script path from the repository root.']
    lines += [f'{i+1}. Run `{c}`: ' + desc for i, (c, desc) in enumerate(zip(commands, ['audit raw inputs and write clean data, decisions and audit.', 'export required static charts and notes.', 'assert numerical anchors and export sensitivity metrics.', 'write offline interactive HTML and default PNG.', 'export explicitly labeled automated checks; no fabricated trials.', 'export final 300 DPI dashboard.', 'write the one-page justification PDF.']))]
    write('README.md', '\n'.join(lines) + '\n')


def main(action):
    globals()[action]()
