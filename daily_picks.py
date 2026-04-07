"""
daily_picks.py
--------------
Run daily in AWS SageMaker.
  1. Loads full historical dataset from S3 or local CSV.
  2. Retrains Logistic Regression on ALL available data.
  3. Loads or builds Neural Network; incrementally trains on yesterday's games.
  4. Runs 5-fold CV threshold tuning (LR) to find today's confidence threshold.
  5. Fetches today's games + features from live sources.
  6. Applies both models → outputs confident picks (LR, NN, consensus).
  7. Delivers picks via SNS and/or S3 JSON.

Environment variables:
  NRFI_DATA_PATH       — S3 URI or local path to training CSV
                         (default: 'data/NRFI_all.csv')
  NRFI_OUTPUT_BUCKET   — S3 bucket to write picks JSON (optional)
  NRFI_SNS_TOPIC_ARN   — SNS topic ARN for pick notifications (optional)
"""

import os
import re
import json
import tempfile
import requests
import pandas as pd
import numpy as np
import statsapi
import tensorflow as tf
from datetime import date, datetime, timedelta
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score
from unidecode import unidecode
import subprocess
import warnings
warnings.filterwarnings('ignore')
os.environ.setdefault('TF_CPP_MIN_LOG_LEVEL', '3')

# Inject AWS credentials from CLI (handles SSO/Identity Center; no-op in SageMaker)
try:
    _raw = subprocess.check_output(
        ['aws', 'configure', 'export-credentials', '--format', 'env-no-export'],
        text=True, stderr=subprocess.DEVNULL
    )
    for _line in _raw.splitlines():
        if '=' in _line:
            _k, _v = _line.split('=', 1)
            os.environ.setdefault(_k.strip(), _v.strip())
except Exception:
    pass

np.random.seed(42)
tf.random.set_seed(42)

# ── Config ────────────────────────────────────────────────────────────────────
DATA_PATH      = os.environ.get('NRFI_DATA_PATH',     'data/NRFI_all.csv')
NN_MODEL_PATH  = os.environ.get('NRFI_NN_MODEL_PATH', 's3://nrfi-store/models/nn_model.keras')
SESSION        = os.environ.get('SESSION', 'all')   # 'afternoon' | 'evening' | 'all'
TODAY          = date.today()
YESTERDAY      = TODAY - timedelta(days=1)

# Cutoff for afternoon vs evening: 5pm ET = 21:00 UTC (EDT, April-October)
AFTERNOON_CUTOFF_UTC_HOUR = 21
MIN_COVERAGE       = 0.10
MAX_COVERAGE       = 0.25
UNIT               = 10    # dollars per unit
RECENCY_HALF_LIFE  = 365   # days; games 1yr old carry ~37% weight, 2yr ~14%, 3yr ~5%

# Park factors (2025 — update at start of each season)
PARK_FACTORS = {
    'COL':115,'CIN':106,'BOS':104,'LAA':101,'PHI':103,'KC':101,
    'CWS':103,'LAD':96, 'BAL':99, 'ARI':100,'PIT':99, 'MIL':97,
    'SF':96,  'ATL':100,'WAS':100,'CLE':97, 'TOR':99, 'MIA':94,
    'TEX':99, 'NYY':97, 'CHC':103,'HOU':100,'MIN':100,'DET':95,
    'TB':97,  'NYM':99, 'STL':100,'OAK':96, 'SEA':96, 'SD':96,
}

# Stadium coordinates (lat, lon) for Open-Meteo weather API
STADIUM_COORDS = {
    'ARI': (33.4453, -112.0667), 'ATL': (33.8908,  -84.4681),
    'BAL': (39.2839,  -76.6218), 'BOS': (42.3467,  -71.0972),
    'CHC': (41.9484,  -87.6553), 'CWS': (41.8299,  -87.6338),
    'CIN': (39.0979,  -84.5082), 'CLE': (41.4962,  -81.6852),
    'COL': (39.7559, -104.9942), 'DET': (42.3390,  -83.0485),
    'HOU': (29.7572,  -95.3556), 'KC':  (39.0517,  -94.4803),
    'LAA': (33.8003, -117.8827), 'LAD': (34.0739, -118.2400),
    'MIA': (25.7781,  -80.2197), 'MIL': (43.0280,  -87.9712),
    'MIN': (44.9817,  -93.2776), 'NYM': (40.7571,  -73.8458),
    'NYY': (40.8296,  -73.9262), 'OAK': (37.7516, -122.2005),
    'PHI': (39.9061,  -75.1665), 'PIT': (40.4469,  -80.0057),
    'SD':  (32.7076, -117.1570), 'SF':  (37.7786, -122.3893),
    'SEA': (47.5914, -122.3325), 'STL': (38.6226,  -90.1928),
    'TB':  (27.7683,  -82.6534), 'TEX': (32.7473,  -97.0845),
    'TOR': (43.6414,  -79.3894), 'WAS': (38.8730,  -77.0074),
}

# teamrankings name → abbreviation
TR_TO_ABBV = {
    'Philadelphia':'PHI','SF Giants':'SF',  'Texas':'TEX',    'Boston':'BOS',
    'Kansas City':'KC',  'Detroit':'DET',   'NY Yankees':'NYY','Tampa Bay':'TB',
    'Toronto':'TOR',     'Pittsburgh':'PIT','Sacramento':'OAK','Baltimore':'BAL',
    'Washington':'WAS',  'NY Mets':'NYM',   'Minnesota':'MIN', 'Chi Sox':'CWS',
    'Seattle':'SEA',     'Cleveland':'CLE', 'Chi Cubs':'CHC',  'St. Louis':'STL',
    'Miami':'MIA',       'Atlanta':'ATL',   'Milwaukee':'MIL', 'Arizona':'ARI',
    'Houston':'HOU',     'LA Angels':'LAA', 'San Diego':'SD',  'LA Dodgers':'LAD',
    'Cincinnati':'CIN',  'Colorado':'COL',
}

# ── Helpers ───────────────────────────────────────────────────────────────────
def pct_to_float(val):
    try:
        s = str(val).strip().rstrip('%')
        return None if s in ('--', 'nan', '') else round(float(s) / 100, 4)
    except Exception:
        return None

def edge_score(acc, cov):
    return (acc - 0.5) * cov

def ev_per_unit(prob_win, odds):
    """Expected value in units per 1 unit staked given American odds."""
    payout = (100 / abs(odds)) if odds < 0 else (odds / 100)
    return prob_win * payout - (1 - prob_win)

def confident_metrics(probs, actuals, low, high, boundary):
    mask = (probs < low) | (probs > high)
    n = mask.sum()
    if n == 0:
        return None, 0, 0.0
    preds = (probs[mask] > boundary).astype(int)
    acc   = accuracy_score(actuals[mask], preds)
    return acc, int(n), n / len(actuals)

# ── Data loading (local CSV or S3) ────────────────────────────────────────────
def load_data(path):
    if path.startswith('s3://'):
        import boto3, io
        bucket, key = path[5:].split('/', 1)
        obj = boto3.client('s3').get_object(Bucket=bucket, Key=key)
        return pd.read_csv(io.BytesIO(obj['Body'].read()))
    return pd.read_csv(path)


# ── Weather (Open-Meteo, free, no API key required) ───────────────────────────
def fetch_weather(abbv, target_date):
    """Return (temp_F, rain) for a team's home stadium on target_date."""
    coords = STADIUM_COORDS.get(abbv)
    if coords is None:
        return 65, 0
    lat, lon = coords
    try:
        url = (
            f'https://api.open-meteo.com/v1/forecast'
            f'?latitude={lat}&longitude={lon}'
            f'&daily=temperature_2m_max,precipitation_sum'
            f'&temperature_unit=fahrenheit'
            f'&timezone=auto'
            f'&start_date={target_date}&end_date={target_date}'
        )
        data = requests.get(url, timeout=10).json()['daily']
        temp = round(data['temperature_2m_max'][0])
        rain = 1 if data['precipitation_sum'][0] > 0.5 else 0
        return temp, rain
    except Exception:
        return 65, 0

# ── Odds scraping (The Odds API — totals_1st_1_innings) ──────────────────────
# Market: Over/Under 0.5 first-inning runs. Over = YRFI, Under = NRFI.
# Free tier: 500 requests/month. ~15 games/day ≈ 450 requests/month.
# Env var: ODDS_API_KEY

_ODDS_API_TEAM_MAP = {
    'arizona diamondbacks': 'ARI', 'atlanta braves': 'ATL', 'baltimore orioles': 'BAL',
    'boston red sox': 'BOS', 'chicago cubs': 'CHC', 'chicago white sox': 'CWS',
    'cincinnati reds': 'CIN', 'cleveland guardians': 'CLE', 'colorado rockies': 'COL',
    'detroit tigers': 'DET', 'houston astros': 'HOU', 'kansas city royals': 'KC',
    'los angeles angels': 'LAA', 'los angeles dodgers': 'LAD', 'miami marlins': 'MIA',
    'milwaukee brewers': 'MIL', 'minnesota twins': 'MIN', 'new york mets': 'NYM',
    'new york yankees': 'NYY', 'oakland athletics': 'OAK', 'philadelphia phillies': 'PHI',
    'pittsburgh pirates': 'PIT', 'san diego padres': 'SD', 'san francisco giants': 'SF',
    'seattle mariners': 'SEA', 'st. louis cardinals': 'STL', 'tampa bay rays': 'TB',
    'texas rangers': 'TEX', 'toronto blue jays': 'TOR', 'washington nationals': 'WAS',
    'athletics': 'OAK',
}

def _decimal_to_american(dec):
    """Convert decimal odds to American integer."""
    if dec >= 2.0:
        return int(round((dec - 1) * 100))
    else:
        return int(round(-100 / (dec - 1)))

def fetch_odds():
    """
    Fetch NRFI/YRFI odds via The Odds API.

    Primary market: totals_1st_1_innings at point=0.5 only.
      Over 0.5 = YRFI, Under 0.5 = NRFI. Lines at any other point are ignored
      for NRFI/YRFI odds (a line at 1.5 is a different bet).

    Also fetches alternate_totals_1st_1_innings to detect strong YRFI signals:
      If any book posts Over >1 at even money or better (American >= -110),
      the market is pricing 2+ first-inning runs as ~50%+, an extreme YRFI signal.

    Returns:
      odds        — dict: 'AWAY@HOME' -> (nrfi_odds, yrfi_odds) American ints
      alt_signals — dict: 'AWAY@HOME' -> {'point': float, 'best_over_american': int}
                    only populated when Over >1 is at >= -110
    Falls back to Bovada scrape if API key not set (no alt_signals in that case).
    """
    api_key = os.environ.get('ODDS_API_KEY')
    if not api_key:
        return _fetch_odds_bovada_fallback()

    odds        = {}
    alt_signals = {}
    try:
        events_resp = requests.get(
            'https://api.the-odds-api.com/v4/sports/baseball_mlb/events',
            params={'apiKey': api_key, 'daysFrom': 1}, timeout=15
        )
        if events_resp.status_code != 200:
            return _fetch_odds_bovada_fallback(), {}
        events = events_resp.json()

        for event in events:
            away_full = event.get('away_team', '')
            home_full = event.get('home_team', '')
            away_abbv = _ODDS_API_TEAM_MAP.get(away_full.lower())
            home_abbv = _ODDS_API_TEAM_MAP.get(home_full.lower())
            if not away_abbv or not home_abbv:
                continue

            r = requests.get(
                f'https://api.the-odds-api.com/v4/sports/baseball_mlb/events/{event["id"]}/odds',
                params={'apiKey': api_key, 'regions': 'us',
                        'markets': 'totals_1st_1_innings,alternate_totals_1st_1_innings'},
                timeout=15
            )
            if r.status_code != 200:
                continue

            matchup_key = f'{away_abbv}@{home_abbv}'
            best_yrfi = best_nrfi = None
            # best Over at each point > 0.5: point -> best American odds
            alt_over: dict[float, int] = {}

            for bk in r.json().get('bookmakers', []):
                for mkt in bk.get('markets', []):
                    for outcome in mkt.get('outcomes', []):
                        dec   = outcome.get('price')
                        point = outcome.get('point')
                        if dec is None or point is None:
                            continue
                        american = _decimal_to_american(dec)
                        name = outcome['name']

                        if point == 0.5:
                            # Standard NRFI/YRFI line
                            if name == 'Over' and (best_yrfi is None or american > best_yrfi):
                                best_yrfi = american
                            elif name == 'Under' and (best_nrfi is None or american > best_nrfi):
                                best_nrfi = american
                        elif name == 'Over' and point > 0.5:
                            # Alternate over: track best odds per point
                            if point not in alt_over or american > alt_over[point]:
                                alt_over[point] = american

            if best_nrfi is not None and best_yrfi is not None:
                odds[matchup_key] = (best_nrfi, best_yrfi)

            # Flag strong YRFI signal: any Over at point > 1 priced at -110 or better
            for point, american in alt_over.items():
                if point > 1.0 and american >= -110:
                    # Keep the highest point that still clears -110 (most extreme signal)
                    existing = alt_signals.get(matchup_key)
                    if existing is None or point > existing['point']:
                        alt_signals[matchup_key] = {
                            'point': point,
                            'best_over_american': american,
                        }

    except Exception as ex:
        print(f'  WARNING: Odds API fetch failed ({ex}) — trying Bovada fallback')
        return _fetch_odds_bovada_fallback()

    return odds, alt_signals


def _fetch_odds_bovada_fallback():
    """Bovada public API fallback — no key required, partial coverage. Returns (odds, {})."""
    _BOVADA_TEAM_MAP = {
        'arizona diamondbacks': 'ARI', 'atlanta braves': 'ATL', 'baltimore orioles': 'BAL',
        'boston red sox': 'BOS', 'chicago cubs': 'CHC', 'chicago white sox': 'CWS',
        'cincinnati reds': 'CIN', 'cleveland guardians': 'CLE', 'colorado rockies': 'COL',
        'detroit tigers': 'DET', 'houston astros': 'HOU', 'kansas city royals': 'KC',
        'los angeles angels': 'LAA', 'los angeles dodgers': 'LAD', 'miami marlins': 'MIA',
        'milwaukee brewers': 'MIL', 'minnesota twins': 'MIN', 'new york mets': 'NYM',
        'new york yankees': 'NYY', 'oakland athletics': 'OAK', 'philadelphia phillies': 'PHI',
        'pittsburgh pirates': 'PIT', 'san diego padres': 'SD', 'san francisco giants': 'SF',
        'seattle mariners': 'SEA', 'st. louis cardinals': 'STL', 'tampa bay rays': 'TB',
        'texas rangers': 'TEX', 'toronto blue jays': 'TOR', 'washington nationals': 'WAS',
        'athletics': 'OAK', 'sacramento athletics': 'OAK',
    }
    odds = {}
    try:
        url = ('https://www.bovada.lv/services/sports/event/v2/events/A/description/baseball/mlb')
        resp = requests.get(url, headers={'User-Agent': 'Mozilla/5.0'}, timeout=15)
        if resp.status_code != 200:
            return odds
        for event_group in resp.json():
            for event in event_group.get('events', []):
                competitors = event.get('competitors', [])
                home = next((c['name'] for c in competitors if c.get('home')), None)
                away = next((c['name'] for c in competitors if not c.get('home')), None)
                if not home or not away:
                    continue
                home_abbv = _BOVADA_TEAM_MAP.get(home.lower())
                away_abbv = _BOVADA_TEAM_MAP.get(away.lower())
                if not home_abbv or not away_abbv:
                    continue
                for grp in event.get('displayGroups', []):
                    for mkt in grp.get('markets', []):
                        if mkt.get('description') != 'Will there be a run scored in the 1st Inning':
                            continue
                        nrfi_odds = yrfi_odds = None
                        for outcome in mkt.get('outcomes', []):
                            american = outcome.get('price', {}).get('american')
                            if american is None:
                                continue
                            if outcome.get('description') == 'No':
                                nrfi_odds = int(american)
                            elif outcome.get('description') == 'Yes':
                                yrfi_odds = int(american)
                        if nrfi_odds and yrfi_odds:
                            odds[f'{away_abbv}@{home_abbv}'] = (nrfi_odds, yrfi_odds)
    except Exception:
        pass
    return odds, {}

# ── Output delivery (SNS + S3 JSON) ──────────────────────────────────────────
def deliver_picks(picks_rows, date_str, threshold, cv_acc, cv_cov):
    """Write picks JSON to S3 and/or publish to SNS if env vars are configured."""
    payload = {
        'date':      date_str,
        'threshold': {'low': round(1 - threshold, 3), 'high': threshold},
        'cv_acc':    round(cv_acc, 4),
        'cv_cov':    round(cv_cov, 4),
        'picks':     picks_rows,
        'generated_at': datetime.utcnow().isoformat() + 'Z',
    }

    s3_bucket = os.environ.get('NRFI_OUTPUT_BUCKET')
    if s3_bucket:
        try:
            import boto3
            suffix = f'-{SESSION}' if SESSION != 'all' else ''
            key = f'picks/{TODAY.year}/{date_str}{suffix}.json'
            boto3.client('s3').put_object(
                Bucket=s3_bucket, Key=key,
                Body=json.dumps(payload, indent=2),
                ContentType='application/json',
            )
            print(f'  Picks written to s3://{s3_bucket}/{key}')
        except Exception as ex:
            print(f'  WARNING: S3 write failed ({ex})')

    # SNS plain-text fallback (kept for backwards compat; SES is primary)
    sns_topic = os.environ.get('NRFI_SNS_TOPIC_ARN')
    if sns_topic:
        try:
            import boto3
            unit_size = picks_rows[0]['unit_size'] if picks_rows else UNIT
            lines = [f'NRFI Picks — {date_str}  [1u = ${unit_size}]', '']
            for p in picks_rows:
                ev_str = (f'  EV: {p["ev_units"]:+.3f}u (${p["ev_dollars"]:+.2f})'
                          if p.get('ev_units') is not None else '')
                cons_str = '  *** CONSENSUS' if p.get('consensus') else ''
                lines.append(
                    f'[{p.get("model","?")}] {p["matchup"]}  →  {p["prediction"]}'
                    f'  ({p["confidence"]:.1%}){ev_str}{cons_str}'
                )
                lines.append(f'  {p["away_pitcher"]} vs {p["home_pitcher"]}')
                lines.append('')
            boto3.client('sns').publish(
                TopicArn=sns_topic,
                Subject=f'NRFI Picks {date_str}',
                Message='\n'.join(lines),
            )
            print('  Picks published to SNS')
        except Exception as ex:
            print(f'  WARNING: SNS publish failed ({ex})')

# ── SES HTML email ───────────────────────────────────────────────────────────
# Env vars:
#   NRFI_SES_FROM  — verified SES sender address
#   NRFI_SES_TO    — comma-separated list of recipient addresses
# Region: us-east-1

def _odds_str(val):
    if val is None: return '—'
    return f'+{val}' if val > 0 else str(val)

def _ev_str(units, dollars):
    if units is None: return '—'
    sign = '+' if units >= 0 else ''
    return f'{sign}{units:.3f}u (${sign}{dollars:.2f})'

def _pick_pl(correct, pred, nrfi_odds, yrfi_odds):
    """Actual P/L for one graded bet. Uses real odds when available, -110 fallback."""
    if correct is None or pd.isna(correct):
        return None
    raw = nrfi_odds if pred == 'NRFI' else yrfi_odds
    odds = int(raw) if raw is not None and not pd.isna(raw) else -110
    if int(correct) == 1:
        return round(UNIT * (100 / abs(odds) if odds < 0 else odds / 100), 2)
    return -float(UNIT)

def _pl_str(pl):
    if pl is None: return '—'
    return f'<span style="color:{"#34d399" if pl >= 0 else "#f87171"};font-weight:700">{"+" if pl>=0 else ""}${pl:.2f}</span>'

def build_email_html(date_str, picks_rows, yesterday_rows, ytd_df, today_df_all,
                     lr_threshold, nn_threshold, cv_acc, cv_cov, alt_signals=None):
    """
    Clean white email:
      - Yesterday's results table
      - YTD performance summary
      - Today's picks (consensus first, then LR-only, then NN-only — no duplicates)
      - All-games table
    """
    # colour palette — clean light theme
    G = '#2563eb'   # blue   (NRFI)
    R = '#7c3aed'   # purple (YRFI)
    MUT = '#6b7280' # muted grey
    BDR = '#e5e7eb' # border/divider
    TXT = '#111827' # near-black

    def section(title, content):
        return (f'<div style="margin-bottom:28px">'
                f'<div style="font-size:11px;font-weight:700;letter-spacing:1.5px;'
                f'text-transform:uppercase;color:{MUT};border-bottom:2px solid {TXT};'
                f'padding-bottom:6px;margin-bottom:16px">{title}</div>'
                f'{content}</div>')

    def th(label, right=False):
        align = 'right' if right else 'left'
        return (f'<th style="padding:6px 10px;text-align:{align};font-size:11px;'
                f'font-weight:600;color:{MUT};border-bottom:1px solid {BDR};'
                f'white-space:nowrap">{label}</th>')

    def td(val, right=False, bold=False, color=None):
        align = 'right' if right else 'left'
        fw = '600' if bold else '400'
        col = f'color:{color};' if color else ''
        return (f'<td style="padding:7px 10px;text-align:{align};font-size:13px;'
                f'{col}font-weight:{fw};border-bottom:1px solid {BDR};'
                f'white-space:nowrap">{val}</td>')

    def ev_display(ev_units, ev_dollars):
        """Return EV string or em-dash, never nan."""
        try:
            if ev_units is None or (isinstance(ev_units, float) and np.isnan(ev_units)):
                return '—'
            sign = '+' if ev_units >= 0 else ''
            col  = G if ev_units >= 0 else R
            return f'<span style="color:{col}">{sign}{ev_units:.3f}u (${ev_dollars:+.2f})</span>'
        except Exception:
            return '—'

    # ── Yesterday's results ───────────────────────────────────────────────────
    if yesterday_rows is not None and not yesterday_rows.empty:
        conf = yesterday_rows[yesterday_rows['lr_confident'] | yesterday_rows['nn_confident']]
        rows_html = ''; total_pl = 0.0
        for _, r in conf.sort_values('lr_conf', ascending=False).iterrows():
            result_label = result_color = None
            pl_val = None
            models_used = []
            for m, pred_col, correct_col in [('LR','lr_pred','lr_correct'),('NN','nn_pred','nn_correct')]:
                if not r.get(f'{m.lower()}_confident', False): continue
                models_used.append(m)
                correct = r[correct_col]
                if pd.notna(correct):
                    pl_val = _pick_pl(correct, r[pred_col], r.get('nrfi_odds'), r.get('yrfi_odds'))
                    if pl_val is not None: total_pl += pl_val
                    result_label = 'WIN' if int(correct) == 1 else 'LOSS'
                    result_color = G if int(correct) == 1 else R
                else:
                    result_label = 'Pending'; result_color = MUT
            if result_label is None: continue
            actual = ('YRFI' if int(r['actual_yrfi'])==1 else 'NRFI') if pd.notna(r.get('actual_yrfi')) else '—'
            pred   = r.get('lr_pred','—')
            pred_color = G if pred == 'NRFI' else R
            nrfi_o = _odds_str(r.get('nrfi_odds')); yrfi_o = _odds_str(r.get('yrfi_odds'))
            pl_str = _pl_str(pl_val) if pl_val is not None else '—'
            rows_html += (
                f'<tr>'
                + td(r['matchup'], bold=True)
                + td('/'.join(models_used), color=MUT)
                + td(f'<span style="color:{pred_color};font-weight:600">{pred}</span>')
                + td(f'{r.get("lr_conf",0):.1%}', right=True)
                + td(f'{nrfi_o} / {yrfi_o}', right=True, color=MUT)
                + td(actual)
                + td(f'<span style="color:{result_color};font-weight:700">{result_label}</span>', right=True)
                + td(pl_str, right=True)
                + '</tr>'
            )
        yest_tbl = (
            f'<table style="width:100%;border-collapse:collapse">'
            f'<tr>{th("Matchup")}{th("Model")}{th("Pick")}{th("Conf",True)}'
            f'{th("NRFI / YRFI Odds",True)}{th("Actual")}{th("Result",True)}{th("P/L",True)}</tr>'
            + rows_html
            + f'<tr><td colspan="7" style="padding:6px 10px;font-size:12px;color:{MUT}">Total</td>'
            + td(_pl_str(total_pl), right=True) + '</tr></table>'
        )
        odds_note = (f'<div style="font-size:11px;color:{MUT};margin-top:8px">'
                     f'* P/L uses actual odds when available, -110 otherwise</div>'
                     if any(pd.isna(r.get('nrfi_odds')) for _,r in conf.iterrows()) else '')
        yest_section = section(
            f"Yesterday's Results &mdash; {YESTERDAY.strftime('%B')} {YESTERDAY.day}",
            yest_tbl + odds_note
        )
    else:
        yest_section = section("Yesterday's Results",
            f'<span style="color:{MUT}">No picks on record for yesterday.</span>')

    # ── YTD performance ───────────────────────────────────────────────────────
    if ytd_df is not None and not ytd_df.empty:
        lr_ytd  = ytd_df[ytd_df['lr_confident'] & ytd_df['lr_correct'].notna()]
        nn_ytd  = ytd_df[ytd_df['nn_confident'] & ytd_df['nn_correct'].notna()]
        con_ytd = ytd_df[ytd_df['consensus']    & ytd_df['lr_correct'].notna()]

        def ytd_row(label, subset, correct_col, pred_col):
            if subset.empty:
                return f'<tr>{td(label,bold=True)}{td("0-0")}{td("—")}{td("—")}{td("—")}</tr>'
            w = int(subset[correct_col].sum()); l = len(subset)-w
            pct = w/(w+l)
            pl_sum = sum(_pick_pl(r[correct_col],r[pred_col],r.get('nrfi_odds'),r.get('yrfi_odds'))
                         for _,r in subset.iterrows() if pd.notna(r[correct_col]))
            pl_col = G if pl_sum>=0 else R
            acc_col = G if pct>0.5 else R if pct<0.5 else MUT
            return (f'<tr>'
                    + td(label, bold=True)
                    + td(f'{w}-{l}')
                    + td(f'<span style="color:{acc_col};font-weight:600">{pct:.1%}</span>', right=True)
                    + td(f'{len(subset)/max(len(ytd_df),1):.1%}', right=True, color=MUT)
                    + td(f'<span style="color:{pl_col};font-weight:600">{"+" if pl_sum>=0 else ""}${pl_sum:.2f}</span>', right=True)
                    + '</tr>')

        ytd_tbl = (f'<table style="width:100%;border-collapse:collapse">'
                   f'<tr>{th("Model")}{th("Record")}{th("Acc",True)}{th("Cov",True)}{th("P/L",True)}</tr>'
                   + ytd_row('LR', lr_ytd, 'lr_correct', 'lr_pred')
                   + ytd_row('NN', nn_ytd, 'nn_correct', 'nn_pred')
                   + ytd_row('Consensus', con_ytd, 'lr_correct', 'lr_pred')
                   + '</table>')
        ytd_section = section(f'{TODAY.year} Season', ytd_tbl)
    else:
        ytd_section = ''

    # ── Alt signals — compact single line per game ────────────────────────────
    if alt_signals:
        parts = []
        for matchup, sig in sorted(alt_signals.items()):
            am = sig['best_over_american']
            am_str = f'+{am}' if am>=0 else str(am)
            parts.append(
                f'<span style="margin-right:18px;white-space:nowrap">'
                f'<b>{matchup}</b> Over {sig["point"]} '
                f'<span style="color:#b45309;font-weight:700">{am_str}</span></span>'
            )
        alt_signals_section = section(
            'Market YRFI Signals (Over 1.5 at near-even money)',
            f'<div style="font-size:13px;color:{TXT};line-height:2">{"".join(parts)}</div>'
            f'<div style="font-size:11px;color:{MUT};margin-top:6px">'
            f'Book pricing 1.5+ first-inning runs at ~50%+ implied. Independent YRFI signal.</div>'
        )
    else:
        alt_signals_section = ''

    # ── Helper: format pitcher name as "F. Last" ──────────────────────────────
    def fmt_pitcher(name):
        if not name or str(name).strip() in ('', 'TBD', 'nan', 'None'):
            return '—'
        parts = str(name).strip().split()
        if len(parts) == 1:
            return parts[0]
        return f'{parts[0][0]}. {" ".join(parts[1:])}'

    # ── Helper: build a 2-row game block (away row + home row) ────────────────
    def game_rows(r, picked=False, pick_info=None):
        """
        Two <tr> rows per game — away on top, home below.
        Columns: Team | Pitcher | NN % | LR % | Odds
        pick_info: dict with keys model_badge, lr_ev, nn_ev (for picked games only)
        """
        away, home = r['matchup'].split('@', 1)
        lrc  = G if r['lr_pred'] == 'NRFI' else R
        nnc  = G if r['nn_pred'] == 'NRFI' else R
        lrfw = '700' if r['lr_confident'] else '400'
        nnfw = '700' if r['nn_confident'] else '400'

        bo = get_odds(r['matchup'])
        yrfi_odds_s = _odds_str(bo[1]) if bo else '—'
        nrfi_odds_s = _odds_str(bo[0]) if bo else '—'

        a_pitcher = fmt_pitcher(r.get('away_pitcher', ''))
        h_pitcher = fmt_pitcher(r.get('home_pitcher', ''))

        # model badge for picked games
        if picked and pick_info:
            badge = pick_info.get('badge', '')
            badge_html = (f'&nbsp;<span style="font-size:10px;background:#f3f4f6;'
                          f'padding:1px 5px;border-radius:3px;font-weight:700;'
                          f'color:#374151">{badge}</span>') if badge else ''
        else:
            badge_html = ''

        # away row — YRFI side (top of scoreboard)
        away_row = (
            f'<tr>'
            + td(f'<span style="font-weight:{"700" if picked else "400"}">{away}</span>{badge_html}')
            + td(f'<span style="color:{MUT};font-style:italic;font-size:12px">{a_pitcher}</span>')
            + td(f'<span style="color:{nnc};font-weight:{nnfw}">{r["nn_prob_yrfi"]:.0%}</span>',
                 right=True)
            + td(f'<span style="color:{lrc};font-weight:{lrfw}">{r["lr_prob_yrfi"]:.0%}</span>',
                 right=True)
            + f'<td style="padding:7px 10px 2px 10px;text-align:right;font-size:12px;'
              f'color:{MUT};white-space:nowrap;border-bottom:none">'
              f'<span style="color:{R}">YRFI</span> {yrfi_odds_s}</td>'
            + '</tr>'
        )
        # home row — NRFI side (bottom of scoreboard), with bottom border
        home_row = (
            f'<tr style="border-bottom:2px solid {BDR}">'
            + td(f'<span style="color:{MUT}">{home}</span>')
            + td(f'<span style="color:{MUT};font-style:italic;font-size:12px">{h_pitcher}</span>')
            + td(f'<span style="color:{MUT};font-size:12px">{r["nn_prob_nrfi"]:.0%}</span>',
                 right=True)
            + td(f'<span style="color:{MUT};font-size:12px">{r["lr_prob_nrfi"]:.0%}</span>',
                 right=True)
            + f'<td style="padding:2px 10px 7px 10px;text-align:right;font-size:12px;'
              f'color:{MUT};white-space:nowrap;border-bottom:2px solid {BDR}">'
              f'<span style="color:{G}">NRFI</span> {nrfi_odds_s}</td>'
            + '</tr>'
        )
        spacer = f'<tr><td colspan="5" style="padding:4px 0;border:none"></td></tr>'
        return away_row + home_row + spacer

    def games_table(rows_iter, header=True):
        hdr = (f'<tr>{th("Team")}{th("Starter")}'
               f'{th("NN YRFI%", True)}{th("LR YRFI%", True)}{th("Odds", True)}</tr>'
               ) if header else ''
        return (f'<table style="width:100%;border-collapse:collapse">'
                + hdr + rows_iter + '</table>')

    # ── Build picked-matchup lookup from picks_rows ───────────────────────────
    # picks_rows may have LR + NN entries for same matchup
    pick_meta = {}  # matchup -> {badge, lr_ev, nn_ev}
    for p in picks_rows:
        m = p['matchup']
        if m not in pick_meta:
            pick_meta[m] = {'badge': '', 'models': []}
        pick_meta[m]['models'].append(p.get('model', ''))
    for m, info in pick_meta.items():
        models = info['models']
        if 'LR' in models and 'NN' in models:
            info['badge'] = 'CON'
        elif 'LR' in models:
            info['badge'] = 'LR'
        else:
            info['badge'] = 'NN'

    # ── Split today_df_all into picked vs not-picked ──────────────────────────
    if today_df_all is not None and not today_df_all.empty:
        df_sorted   = today_df_all.sort_values('lr_conf', ascending=False)
        df_picked   = df_sorted[df_sorted['lr_confident'] | df_sorted['nn_confident']]
        df_unpicked = df_sorted[~(df_sorted['lr_confident'] | df_sorted['nn_confident'])]

        picked_rows_html = ''.join(
            game_rows(r, picked=True, pick_info=pick_meta.get(r['matchup']))
            for _, r in df_picked.iterrows()
        )
        unpicked_rows_html = ''.join(
            game_rows(r, picked=False)
            for _, r in df_unpicked.iterrows()
        )

        if picked_rows_html:
            picks_section = section(
                "Today's Picks",
                games_table(picked_rows_html)
            )
        else:
            picks_section = section("Today's Picks",
                f'<span style="color:{MUT}">No games cleared the threshold today.</span>')

        not_picked_section = section(
            'Not Picked',
            games_table(unpicked_rows_html)
        ) if unpicked_rows_html else ''
    else:
        picks_section     = ''
        not_picked_section = ''

    # ── Assemble ──────────────────────────────────────────────────────────────
    n_games = len(today_df_all) if today_df_all is not None else 0
    body = f"""<!DOCTYPE html>
<html><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<style>
  body {{ margin:0;padding:0;background:#ffffff;
         font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',Arial,sans-serif;
         color:{TXT}; }}
  * {{ box-sizing:border-box; }}
</style></head>
<body>
<div style="max-width:680px;margin:0 auto;padding:28px 20px">

  <!-- Header -->
  <div style="border-bottom:3px solid {TXT};padding-bottom:14px;margin-bottom:28px">
    <div style="font-size:11px;font-weight:700;letter-spacing:2px;text-transform:uppercase;color:{MUT}">NRFI Daily</div>
    <div style="font-size:26px;font-weight:800;margin-top:4px">{TODAY.strftime('%A, %B')} {TODAY.day}</div>
    <div style="font-size:13px;color:{MUT};margin-top:6px">
      {n_games} games &nbsp;&nbsp;|&nbsp;&nbsp; threshold &lt;{round(1-lr_threshold,3)} / &gt;{round(lr_threshold,3)}
      &nbsp;&nbsp;|&nbsp;&nbsp; CV {cv_acc:.1%} acc &nbsp; {cv_cov:.1%} cov
    </div>
  </div>

  {yest_section}
  {ytd_section}
  {alt_signals_section}
  {picks_section}
  {not_picked_section}

  <div style="margin-top:32px;padding-top:12px;border-top:1px solid {BDR};
              font-size:11px;color:{MUT};text-align:center">
    Generated {datetime.utcnow().strftime('%H:%M UTC')} &nbsp;&middot;&nbsp;
    LR daily retrain + NN incremental &nbsp;&middot;&nbsp; 1u = ${UNIT}
  </div>
</div>
</body></html>"""
    return body


def send_email(html_body, subject, date_str):
    """
    Send daily picks email via AWS SES.

    Recipients: NRFI_SES_TO env var, comma-separated for multiple addresses.
      e.g. NRFI_SES_TO="alice@example.com,bob@example.com"
    """
    ses_from = os.environ.get('NRFI_SES_FROM')
    ses_to   = os.environ.get('NRFI_SES_TO', '')
    if not ses_from or not ses_to:
        print('  Email skipped: NRFI_SES_FROM / NRFI_SES_TO not set')
        return
    recipients = [addr.strip() for addr in ses_to.split(',') if addr.strip()]
    if not recipients:
        return
    try:
        import boto3
        boto3.client('ses', region_name='us-east-1').send_email(
            Source=ses_from,
            Destination={'ToAddresses': recipients},
            Message={
                'Subject': {'Data': subject},
                'Body':    {'Html': {'Data': html_body, 'Charset': 'UTF-8'}},
            },
        )
        print(f'  Email sent to {", ".join(recipients)}')
    except Exception as ex:
        print(f'  WARNING: SES send failed ({ex})')


# ── Grade yesterday's picks ───────────────────────────────────────────────────
def grade_yesterday():
    """
    Load yesterday's game log and Lambda results file from S3.
    Fill in actual_yrfi, lr_correct, nn_correct on every game.
    Append completed rows to results/results.csv.
    Prints a W/L summary for confident picks.
    Silently skips if either file is missing.
    """
    s3_bucket = os.environ.get('NRFI_OUTPUT_BUCKET')
    if not s3_bucket:
        return None, None

    import boto3, io
    s3 = boto3.client('s3')
    ystr = YESTERDAY.strftime('%Y-%m-%d')

    # Load yesterday's full game log
    log_key = f'game_log/{YESTERDAY.year}/{ystr}.csv'
    try:
        obj = s3.get_object(Bucket=s3_bucket, Key=log_key)
        log_df = pd.read_csv(io.BytesIO(obj['Body'].read()))
    except Exception:
        print(f'Grade: no game log found for {ystr} — skipping')
        return None, None

    # Load yesterday's Lambda results file
    results_key = f'data/{YESTERDAY.year}/{YESTERDAY.month}/{YESTERDAY.day}.txt'
    try:
        obj = s3.get_object(Bucket=s3_bucket, Key=results_key)
        results_df = pd.read_csv(io.BytesIO(obj['Body'].read()))
    except Exception:
        print(f'Grade: no Lambda results file for {ystr} — skipping')
        return None, None

    # Build matchup -> YRFI lookup (Lambda id: YYYY-MM-DD-AWAY@HOME)
    def parse_matchup(game_id):
        parts = str(game_id).split('-', 3)
        return parts[3] if len(parts) == 4 else None

    results_df['matchup'] = results_df['id'].apply(parse_matchup)
    result_map = dict(zip(results_df['matchup'], results_df['YRFI']))

    # Fill actuals into game log
    log_df['actual_yrfi'] = log_df['matchup'].map(result_map)
    log_df['lr_correct'] = log_df.apply(
        lambda r: (
            None if pd.isna(r['actual_yrfi']) else
            int(('YRFI' if r['actual_yrfi'] == 1 else 'NRFI') == r['lr_pred'])
        ), axis=1
    )
    log_df['nn_correct'] = log_df.apply(
        lambda r: (
            None if pd.isna(r['actual_yrfi']) else
            int(('YRFI' if r['actual_yrfi'] == 1 else 'NRFI') == r['nn_pred'])
        ), axis=1
    )

    # Print summary for confident picks
    print(f'\n{"=" * 60}')
    print(f'YESTERDAY\'S RESULTS — {ystr}')
    print(f'{"=" * 60}')
    lr_conf = log_df[log_df['lr_confident'] == True]
    nn_conf = log_df[log_df['nn_confident'] == True]
    for label, subset, pred_col, correct_col, conf_col in [
        ('LR', lr_conf, 'lr_pred', 'lr_correct', 'lr_conf'),
        ('NN', nn_conf, 'nn_pred', 'nn_correct', 'nn_conf'),
    ]:
        if subset.empty:
            print(f'  [{label}] No confident picks')
            continue
        for _, r in subset.sort_values(conf_col, ascending=False).iterrows():
            actual_yrfi = r['actual_yrfi']
            correct     = r[correct_col]
            if pd.isna(actual_yrfi):
                char, status = '?', 'NO RESULT'
            else:
                actual = 'YRFI' if int(actual_yrfi) == 1 else 'NRFI'
                char   = 'W' if correct == 1 else 'L'
                status = f'{"WIN" if correct == 1 else "LOSS"}  (actual: {actual})'
            odds_str = f'  odds: {r["nrfi_odds"] if r[pred_col]=="NRFI" else r["yrfi_odds"]}' \
                       if pd.notna(r.get('nrfi_odds')) else ''
            print(f'  [{char}][{label}] {r["matchup"]:14}  {r[pred_col]}  '
                  f'({r[conf_col]:.1%}){odds_str}  -> {status}')

    for label, subset, correct_col in [('LR', lr_conf, 'lr_correct'), ('NN', nn_conf, 'nn_correct')]:
        graded = subset[subset[correct_col].notna()]
        if not graded.empty:
            w = int(graded[correct_col].sum())
            l = len(graded) - w
            print(f'\n  {label} record: {w}-{l}  ({w/len(graded):.1%})')
    print()

    # Append to running results CSV (all games, not just picks)
    results_log_key = 'results/results.csv'
    combined = pd.DataFrame()
    try:
        try:
            obj = s3.get_object(Bucket=s3_bucket, Key=results_log_key)
            existing = pd.read_csv(io.BytesIO(obj['Body'].read()))
            # Drop any prior rows for this date (idempotent re-runs)
            existing = existing[existing['date'] != ystr]
        except Exception:
            existing = pd.DataFrame()
        combined = pd.concat([existing, log_df], ignore_index=True)
        buf = io.BytesIO()
        combined.to_csv(buf, index=False)
        s3.put_object(Bucket=s3_bucket, Key=results_log_key,
                      Body=buf.getvalue(), ContentType='text/csv')
        print(f'  Results log updated: s3://{s3_bucket}/{results_log_key}')
    except Exception as ex:
        print(f'  WARNING: could not update results log ({ex})')

    ytd_df = (combined[combined['date'].str.startswith(str(TODAY.year))]
              if not combined.empty else pd.DataFrame())
    return log_df, ytd_df

yesterday_log_df, ytd_df = grade_yesterday()

# ══════════════════════════════════════════════════════════════════════════════
# PART 1 — RETRAIN ON ALL HISTORICAL DATA
# ══════════════════════════════════════════════════════════════════════════════
print('=' * 60)
print(f'DAILY PICKS  {TODAY}')
print('=' * 60)

_df_raw = load_data(DATA_PATH)

# Exclude pre-April-15 games from training: pitcher RA and YRFI pct are
# unreliable before mid-April (Fangraphs splits need innings to accumulate;
# teamrankings YRFI% is based on 0-2 games and often reads as 0%).
df = _df_raw[~((_df_raw['month'] < 4) | ((_df_raw['month'] == 4) & (_df_raw['day'] < 15)))].copy()
print(f'Training set after April-15 filter: {len(df)} games '
      f'(dropped {len(_df_raw) - len(df)} pre-Apr-15 rows)')

league_avg_ra   = df[df['away_pitcher_ra'] > 0]['away_pitcher_ra'].median()
league_avg_whip = df[df['home_whip'] > 0]['home_whip'].median()
league_avg_yrfi = df[df['home_yrfi_pct'] > 0]['home_yrfi_pct'].mean()
RA_CAP = 1.5  # cap extreme small-sample RA values before imputing zeros
df['away_pitcher_ra'] = df['away_pitcher_ra'].clip(upper=RA_CAP).replace(0, league_avg_ra)
df['home_pitcher_ra'] = df['home_pitcher_ra'].clip(upper=RA_CAP).replace(0, league_avg_ra)
df['away_whip']       = df['away_whip'].replace(0, league_avg_whip)
df['home_whip']       = df['home_whip'].replace(0, league_avg_whip)
df['home_yrfi_pct']   = df['home_yrfi_pct'].replace(0, league_avg_yrfi)
df['away_yrfi_pct']   = df['away_yrfi_pct'].replace(0, league_avg_yrfi)

FEATURES = ['away_ops', 'home_ops', 'home_yrfi_pct', 'away_yrfi_pct',
            'home_pitcher_ra', 'home_whip', 'away_pitcher_ra', 'away_whip',
            'park_factor', 'temp', 'rain']

def make_features(d):
    e = pd.DataFrame(index=d.index)
    e['away_ops']        = d['away_ops']
    e['home_ops']        = d['home_ops']
    e['home_yrfi_pct']   = d['home_yrfi_pct']
    e['away_yrfi_pct']   = d['away_yrfi_pct']
    e['home_pitcher_ra'] = d['home_pitcher_ra']
    e['home_whip']       = d['home_whip']
    e['away_pitcher_ra'] = d['away_pitcher_ra']
    e['away_whip']       = d['away_whip']
    e['park_factor']     = d['park_factor']
    e['temp']            = d['temp']
    e['rain']            = d['rain']
    return e[FEATURES]

X_raw = make_features(df).values
y     = df['YRFI'].values

# Recency weights: exp(-age / half_life), normalized so mean weight = 1
_game_dates    = pd.to_datetime(df[['year', 'month', 'day']])
_age_days      = (pd.Timestamp(TODAY) - _game_dates).dt.days.values
sample_weights = np.exp(-_age_days / RECENCY_HALF_LIFE)
sample_weights = sample_weights / sample_weights.mean()

scaler   = StandardScaler()
X_scaled = scaler.fit_transform(X_raw)
BOUNDARY = y.mean()

lr = LogisticRegression(max_iter=500)
lr.fit(X_scaled, y, sample_weight=sample_weights)
print(f'Trained LR on {len(df)} games  (YRFI base rate: {BOUNDARY:.4f})')

# ── Neural Network (load from S3 or train from scratch, then increment) ───────
def _s3_model_exists(s3_path):
    import boto3
    bucket, key = s3_path[5:].split('/', 1)
    try:
        boto3.client('s3').head_object(Bucket=bucket, Key=key)
        return True
    except Exception:
        return False

def _load_nn_from_s3(s3_path):
    import boto3
    bucket, key = s3_path[5:].split('/', 1)
    with tempfile.NamedTemporaryFile(suffix='.keras', delete=False) as f:
        boto3.client('s3').download_fileobj(bucket, key, f)
        return tf.keras.models.load_model(f.name)

def _save_nn_to_s3(model, s3_path):
    import boto3
    bucket, key = s3_path[5:].split('/', 1)
    with tempfile.NamedTemporaryFile(suffix='.keras', delete=False) as f:
        model.save(f.name)
        boto3.client('s3').upload_file(f.name, bucket, key)

def _build_nn(input_dim):
    """Best architecture from hyperparam gridsearch (Apr 2026): 8->8->8->1, dropout=0.1, lr=0.005, bs=64, l2=0.001.
    Trained on 2021-2025 data: conf_acc=54.2%, coverage=39.0%, edge=0.0163."""
    reg = tf.keras.regularizers.l2(0.001)
    model = tf.keras.Sequential([
        tf.keras.layers.Dense(8, activation='relu', input_shape=(input_dim,), kernel_regularizer=reg),
        tf.keras.layers.Dropout(0.1),
        tf.keras.layers.Dense(8, activation='relu', kernel_regularizer=reg),
        tf.keras.layers.Dropout(0.1),
        tf.keras.layers.Dense(8, activation='relu', kernel_regularizer=reg),
        tf.keras.layers.Dropout(0.1),
        tf.keras.layers.Dense(1, activation='sigmoid'),
    ])
    model.compile(optimizer=tf.keras.optimizers.Adam(0.005),
                  loss='binary_crossentropy')
    return model

nn_scaler = StandardScaler()
X_nn_all  = nn_scaler.fit_transform(X_raw)
nn_boundary = y.mean()

uses_s3 = NN_MODEL_PATH.startswith('s3://')

nn = None
if uses_s3 and _s3_model_exists(NN_MODEL_PATH):
    print(f'\nLoading NN from {NN_MODEL_PATH}...')
    try:
        nn = _load_nn_from_s3(NN_MODEL_PATH)
    except Exception as _load_err:
        print(f'  WARNING: could not load saved NN ({_load_err}) — will retrain from scratch')
        nn = None

if nn is not None:
    # Incremental train: 5 epochs on yesterday's S3 batch
    yesterday_path = (f's3://nrfi-store/data/{YESTERDAY.year}/'
                      f'{YESTERDAY.month}/{YESTERDAY.day}.txt')
    try:
        batch_df = load_data(yesterday_path)
        for col in ['away_pitcher_ra', 'home_pitcher_ra', 'away_whip', 'home_whip',
                    'home_yrfi_pct', 'away_yrfi_pct']:
            batch_df[col] = batch_df[col].replace(0, df[col].median())
            batch_df[col] = batch_df[col].fillna(df[col].median())
        feat_batch = make_features(batch_df)
        # drop rows with any NaN features to prevent weight explosion
        valid_mask = feat_batch.notna().all(axis=1)
        feat_batch = feat_batch[valid_mask]
        y_batch = batch_df['YRFI'].values[valid_mask]
        if len(feat_batch) == 0:
            print(f'  WARNING: batch has no clean rows after NaN drop — skipping increment')
        else:
            X_batch = nn_scaler.transform(feat_batch.values)
            nn.fit(X_batch, y_batch, epochs=5, batch_size=64, verbose=0)
            print(f'  Incremental train: 5 epochs on {len(feat_batch)} games from {YESTERDAY}')
    except Exception as ex:
        print(f'  WARNING: could not load yesterday batch ({ex}) — skipping increment')

    _save_nn_to_s3(nn, NN_MODEL_PATH)
    print(f'  NN saved to {NN_MODEL_PATH}')
else:
    print('\nNo saved NN found — training from scratch on full dataset...')
    nn = _build_nn(X_nn_all.shape[1])
    es = tf.keras.callbacks.EarlyStopping(
        monitor='val_loss', patience=10, restore_best_weights=True, verbose=0
    )
    nn.fit(X_nn_all, y, epochs=150, batch_size=64,
           validation_split=0.1, shuffle=True, callbacks=[es], verbose=0)
    print(f'  Trained from scratch on {len(y)} games')
    if uses_s3:
        _save_nn_to_s3(nn, NN_MODEL_PATH)
        print(f'  NN saved to {NN_MODEL_PATH}')

# CV threshold tuning for NN (same sweep as LR)
nn_cv_probs_all, nn_cv_y_all = [], []
kf_nn = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
for tr, vl in kf_nn.split(X_raw, y):
    sc_nn = StandardScaler()
    X_tr_nn = sc_nn.fit_transform(X_raw[tr])
    X_vl_nn = sc_nn.transform(X_raw[vl])
    m_nn = _build_nn(X_tr_nn.shape[1])
    m_nn.fit(X_tr_nn, y[tr], epochs=60, batch_size=64,
             validation_split=0.1, verbose=0)
    nn_cv_probs_all.append(m_nn.predict(X_vl_nn, verbose=0).flatten())
    nn_cv_y_all.append(y[vl])
    tf.keras.backend.clear_session()

nn_cv_probs   = np.concatenate(nn_cv_probs_all)
nn_cv_y       = np.concatenate(nn_cv_y_all)
nn_cv_boundary = nn_cv_y.mean()

nn_eligible = []
for t in np.round(np.arange(0.52, 0.631, 0.005), 3):
    acc, n, cov = confident_metrics(nn_cv_probs, nn_cv_y, round(1-t, 3), t, nn_cv_boundary)
    if acc is not None and MIN_COVERAGE <= cov <= MAX_COVERAGE:
        nn_eligible.append((t, acc, n, cov))

nn_best   = max(nn_eligible, key=lambda r: edge_score(r[1], r[3])) if nn_eligible else None
nn_high   = nn_best[0] if nn_best else 0.545
nn_low    = round(1 - nn_high, 3)
nn_meta   = {'boundary': nn_cv_boundary}
print(f'NN CV threshold:  <{nn_low} / >{nn_high}'
      + (f'  (acc={nn_best[1]:.1%}, cov={nn_best[3]:.1%})' if nn_best else '  (fallback)'))

# ══════════════════════════════════════════════════════════════════════════════
# PART 2 — CV THRESHOLD TUNING
# ══════════════════════════════════════════════════════════════════════════════
kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
cv_probs_all, cv_y_all = [], []
for tr, vl in kf.split(X_raw, y):
    sc = StandardScaler()
    m  = LogisticRegression(max_iter=500).fit(sc.fit_transform(X_raw[tr]), y[tr], sample_weight=sample_weights[tr])
    cv_probs_all.append(m.predict_proba(sc.transform(X_raw[vl]))[:, 1])
    cv_y_all.append(y[vl])

cv_probs    = np.concatenate(cv_probs_all)
cv_y        = np.concatenate(cv_y_all)
cv_boundary = cv_y.mean()

sweep   = np.round(np.arange(0.52, 0.631, 0.005), 3)
cv_rows = []
for t in sweep:
    acc, n, cov = confident_metrics(cv_probs, cv_y, round(1 - t, 3), t, cv_boundary)
    if acc is not None:
        cv_rows.append((t, acc, n, cov))

eligible  = [(t, a, n, c) for t, a, n, c in cv_rows if MIN_COVERAGE <= c <= MAX_COVERAGE]
best      = max(eligible, key=lambda r: edge_score(r[1], r[3])) if eligible else None
THRESHOLD = best[0] if best else 0.545

print(f'CV threshold:  <{round(1 - THRESHOLD, 3)} / >{THRESHOLD}  '
      f'(acc={best[1]:.1%}, cov={best[3]:.1%}, edge={edge_score(best[1], best[3]):.4f})')

# ══════════════════════════════════════════════════════════════════════════════
# PART 3 — FETCH TODAY'S GAMES
# ══════════════════════════════════════════════════════════════════════════════
print(f'\nFetching schedule for {TODAY}...')
sched = statsapi.schedule(date=TODAY.strftime('%m/%d/%Y'))

games = []
for g in sched:
    if g.get('status') in ('Postponed', 'Cancelled'):
        continue
    games.append({
        'game_id':      g['game_id'],
        'away_abbv':    g['away_name'],
        'home_abbv':    g['home_name'],
        'away_id':      g['away_id'],
        'home_id':      g['home_id'],
        'away_pitcher': g.get('away_probable_pitcher', 'TBD'),
        'home_pitcher': g.get('home_probable_pitcher', 'TBD'),
        'venue':        g.get('venue_name', ''),
        'game_time':    g.get('game_datetime', ''),
    })

name_to_abbv = {
    'Pittsburgh Pirates':'PIT', 'New York Mets':'NYM',     'Chicago White Sox':'CWS',
    'Milwaukee Brewers':'MIL',  'Washington Nationals':'WAS','Chicago Cubs':'CHC',
    'Minnesota Twins':'MIN',    'Baltimore Orioles':'BAL',  'Boston Red Sox':'BOS',
    'Cincinnati Reds':'CIN',    'Los Angeles Angels':'LAA', 'Houston Astros':'HOU',
    'Detroit Tigers':'DET',     'San Diego Padres':'SD',    'Texas Rangers':'TEX',
    'Philadelphia Phillies':'PHI','Tampa Bay Rays':'TB',    'St. Louis Cardinals':'STL',
    'Arizona Diamondbacks':'ARI','Los Angeles Dodgers':'LAD',
    'Cleveland Guardians':'CLE','Seattle Mariners':'SEA',
    'New York Yankees':'NYY',   'San Francisco Giants':'SF','Oakland Athletics':'OAK',
    'Toronto Blue Jays':'TOR',  'Atlanta Braves':'ATL',     'Colorado Rockies':'COL',
    'Miami Marlins':'MIA',      'Kansas City Royals':'KC',
}
for g in games:
    g['away_abbv'] = name_to_abbv.get(g['away_abbv'], g['away_abbv'])
    g['home_abbv'] = name_to_abbv.get(g['home_abbv'], g['home_abbv'])

print(f'Found {len(games)} games')

def _is_afternoon(game_time_utc: str) -> bool:
    """True if game starts before 5pm ET (21:00 UTC during EDT)."""
    if not game_time_utc:
        return False
    try:
        from datetime import timezone
        dt = datetime.fromisoformat(game_time_utc.replace('Z', '+00:00'))
        return dt.hour < AFTERNOON_CUTOFF_UTC_HOUR
    except Exception:
        return False

if SESSION == 'afternoon':
    games = [g for g in games if _is_afternoon(g['game_time'])]
    print(f'  Session=afternoon: {len(games)} games before 5pm ET')
elif SESSION == 'evening':
    games = [g for g in games if not _is_afternoon(g['game_time'])]
    print(f'  Session=evening: {len(games)} games at 5pm ET or later')

# ══════════════════════════════════════════════════════════════════════════════
# PART 4 — FETCH FEATURES
# ══════════════════════════════════════════════════════════════════════════════

# ── 4a. YRFI pct from teamrankings (current + prior-year fallback) ─────────────
print(f'Fetching YRFI pct from teamrankings ({YESTERDAY})...')
LEAGUE_YRFI = BOUNDARY

def scrape_yrfi_table(date_str):
    """Scrape teamrankings YRFI table. Returns (home_dict, away_dict, overall_dict)."""
    url = f'https://www.teamrankings.com/mlb/stat/yes-run-first-inning-pct?date={date_str}'
    tbl = pd.read_html(url)[0]
    tbl['abbv'] = tbl['Team'].map(TR_TO_ABBV)
    tbl = tbl[pd.notna(tbl['abbv'])]
    year_cols   = [c for c in tbl.columns if str(c).isdigit()]
    overall_col = year_cols[0] if year_cols else None
    h = {row['abbv']: pct_to_float(row['Home']) for _, row in tbl.iterrows()}
    a = {row['abbv']: pct_to_float(row['Away']) for _, row in tbl.iterrows()}
    o = ({row['abbv']: pct_to_float(row[overall_col]) for _, row in tbl.iterrows()}
         if overall_col else {})
    return h, a, o

yrfi_home, yrfi_away, yrfi_overall   = {}, {}, {}
yrfi_home_prev, yrfi_away_prev       = {}, {}

try:
    yrfi_home, yrfi_away, yrfi_overall = scrape_yrfi_table(str(YESTERDAY))
    print(f'  Loaded YRFI pct for {len(yrfi_home)} teams (current season)')
except Exception as ex:
    print(f'  WARNING: current-year teamrankings fetch failed ({ex})')

# Prior-year splits as fallback when current season has insufficient data
try:
    prior_date = f'{TODAY.year - 1}-10-01'
    yrfi_home_prev, yrfi_away_prev, _ = scrape_yrfi_table(prior_date)
    print(f'  Loaded prior-year splits for {len(yrfi_home_prev)} teams ({prior_date})')
except Exception as ex:
    print(f'  WARNING: prior-year teamrankings fetch failed ({ex})')

def get_yrfi(abbv, split_curr, split_prev, overall, fallback=LEAGUE_YRFI):
    """current split → prior-year split → overall year rate → league average."""
    v = split_curr.get(abbv)
    if v is not None:
        return v
    v = split_prev.get(abbv)
    if v is not None:
        return v
    v = overall.get(abbv)
    if v is not None:
        return v
    return fallback

# ── 4b. Pitcher + team stats (MLB Stats API — 30/60-day rolling windows) ─────
# Fangraphs blocked by Cloudflare; MLB Stats API supports identical date ranges
# with no auth and no rate limiting.
print('Fetching pitcher/team stats (MLB Stats API)...')
PITCHER_RA, PITCHER_WHIP, TEAM_OPS = {}, {}, {}

MLB_TEAM_ABBV = {
    'Arizona Diamondbacks': 'ARI', 'Athletics': 'OAK', 'Atlanta Braves': 'ATL',
    'Baltimore Orioles': 'BAL', 'Boston Red Sox': 'BOS', 'Chicago Cubs': 'CHC',
    'Chicago White Sox': 'CWS', 'Cincinnati Reds': 'CIN', 'Cleveland Guardians': 'CLE',
    'Colorado Rockies': 'COL', 'Detroit Tigers': 'DET', 'Houston Astros': 'HOU',
    'Kansas City Royals': 'KC', 'Los Angeles Angels': 'LAA', 'Los Angeles Dodgers': 'LAD',
    'Miami Marlins': 'MIA', 'Milwaukee Brewers': 'MIL', 'Minnesota Twins': 'MIN',
    'New York Mets': 'NYM', 'New York Yankees': 'NYY', 'Oakland Athletics': 'OAK',
    'Philadelphia Phillies': 'PHI', 'Pittsburgh Pirates': 'PIT', 'San Diego Padres': 'SD',
    'San Francisco Giants': 'SF', 'Seattle Mariners': 'SEA', 'St. Louis Cardinals': 'STL',
    'Tampa Bay Rays': 'TB', 'Texas Rangers': 'TEX', 'Toronto Blue Jays': 'TOR',
    'Washington Nationals': 'WAS',
}

try:
    # Pitcher RA + WHIP: 60-day rolling window, starters only (gamesStarted > 0)
    # RA = runs / inningsPitched — per-inning scale matches training data (~0.47 median)
    d60 = YESTERDAY - timedelta(days=59)
    pit_url = (
        f'https://statsapi.mlb.com/api/v1/stats'
        f'?stats=season&group=pitching&season={TODAY.year}'
        f'&startDate={d60}&endDate={YESTERDAY}'
        f'&playerPool=All&limit=5000'
        f'&fields=stats,splits,stat,whip,era,inningsPitched,runs,gamesStarted,player,fullName'
    )
    pit_resp = requests.get(pit_url, timeout=30)
    for split in pit_resp.json()['stats'][0]['splits']:
        stat = split['stat']
        gs   = int(stat.get('gamesStarted', 0) or 0)
        if gs == 0:
            continue
        name = unidecode(split['player']['fullName'].strip())
        try:    PITCHER_WHIP[name] = float(stat['whip'])
        except: pass
        try:
            ip = float(stat.get('inningsPitched') or 0)
            if ip > 0:
                PITCHER_RA[name] = float(stat['runs']) / ip
        except: pass
    print(f'  Loaded 60-day WHIP for {len(PITCHER_WHIP)} starters, '
          f'RA for {len(PITCHER_RA)} starters')
except Exception as ex:
    print(f'  WARNING: MLB Stats API pitcher fetch failed ({ex})')

# Individual batter OPS: 30-day rolling — matches training data window.
# Used for lineup-level OPS lookup; team average is the fallback.
BATTER_OPS   = {}   # cleaned_name -> OPS
TEAM_AVG_OPS = {}   # abbv -> mean OPS of players on that team (30-day)

try:
    d30 = YESTERDAY - timedelta(days=29)
    bat_url = (
        f'https://statsapi.mlb.com/api/v1/stats'
        f'?stats=season&group=hitting&season={TODAY.year}'
        f'&startDate={d30}&endDate={YESTERDAY}'
        f'&playerPool=All&limit=5000'
        f'&fields=stats,splits,stat,ops,obp,slg,atBats,player,fullName,team,name'
    )
    bat_resp = requests.get(bat_url, timeout=30)
    team_ops_lists = {}  # abbv -> [ops, ...]
    for split in bat_resp.json()['stats'][0]['splits']:
        stat = split['stat']
        ab   = int(stat.get('atBats', 0) or 0)
        if ab < 10:  # skip tiny samples to avoid extreme OPS skewing team averages
            continue
        ops_val = stat.get('ops')
        if ops_val is None:
            continue
        try:
            ops_f = float(ops_val)
        except (ValueError, TypeError):
            continue
        name = unidecode(split['player']['fullName'].strip())
        BATTER_OPS[name] = ops_f
        team_name = split.get('team', {}).get('name', '')
        abbv = MLB_TEAM_ABBV.get(team_name)
        if abbv:
            team_ops_lists.setdefault(abbv, []).append(ops_f)
    TEAM_AVG_OPS = {abbv: round(sum(vals)/len(vals), 3)
                    for abbv, vals in team_ops_lists.items() if vals}
    print(f'  Loaded 30-day OPS for {len(BATTER_OPS)} batters ({len(TEAM_AVG_OPS)} teams)')
except Exception as ex:
    print(f'  WARNING: MLB Stats API batting fetch failed ({ex})')

LEAGUE_RA   = league_avg_ra
LEAGUE_WHIP = df['home_whip'].median()
LEAGUE_OPS  = df['home_ops'].median()

def get_pitcher_ra(name):
    return PITCHER_RA.get(unidecode(name), LEAGUE_RA)

def get_pitcher_whip(name):
    return PITCHER_WHIP.get(unidecode(name), LEAGUE_WHIP)

def _lineup_ops(player_names):
    """Average OPS of top-4 found batters in lineup. None if fewer than 2 found."""
    vals = [BATTER_OPS[unidecode(n)] for n in player_names
            if unidecode(n) in BATTER_OPS][:4]
    return round(sum(vals) / len(vals), 3) if len(vals) >= 2 else None

def _fetch_lineup(game_id):
    """Fetch lineup for a game. Returns (away_names, home_names) or ([], [])."""
    try:
        r = requests.get(
            f'https://statsapi.mlb.com/api/v1/game/{game_id}/lineups',
            timeout=10
        )
        if r.status_code == 200:
            data = r.json()
            return (
                [p['fullName'] for p in data.get('awayPlayers', [])],
                [p['fullName'] for p in data.get('homePlayers', [])],
            )
    except Exception:
        pass
    return [], []

# Build yesterday's team_id -> game_id map for lineup fallback.
# Used when today's lineup isn't posted yet (typical at 11 AM ET).
YESTERDAY_GAME_BY_TEAM = {}  # team_id (int) -> game_id
try:
    yest_sched = statsapi.schedule(date=YESTERDAY.strftime('%m/%d/%Y'))
    for yg in yest_sched:
        if yg.get('status') not in ('Final', 'Game Over', 'Completed Early'):
            continue
        YESTERDAY_GAME_BY_TEAM[yg['away_id']] = yg['game_id']
        YESTERDAY_GAME_BY_TEAM[yg['home_id']] = yg['game_id']
    print(f'  Cached yesterday lineups for {len(YESTERDAY_GAME_BY_TEAM)} team slots')
except Exception as ex:
    print(f'  WARNING: Could not fetch yesterday schedule for lineup fallback ({ex})')

def fetch_game_ops(game_id, away_abbv, home_abbv, away_team_id=None, home_team_id=None):
    """
    Return (away_ops, home_ops) using the best available lineup:
      1. Today's announced lineup (statsapi /lineups)
      2. Yesterday's lineup for each side independently
      3. 30-day team average OPS
    """
    today_away, today_home = _fetch_lineup(game_id)
    away_ops = _lineup_ops(today_away)
    home_ops = _lineup_ops(today_home)

    # Fill in missing sides from yesterday's lineup
    if away_ops is None and away_team_id and away_team_id in YESTERDAY_GAME_BY_TEAM:
        yest_game_id = YESTERDAY_GAME_BY_TEAM[away_team_id]
        yest_away, yest_home = _fetch_lineup(yest_game_id)
        # Use whichever side of yesterday's game this team was on
        away_ops = _lineup_ops(yest_away) or _lineup_ops(yest_home)

    if home_ops is None and home_team_id and home_team_id in YESTERDAY_GAME_BY_TEAM:
        yest_game_id = YESTERDAY_GAME_BY_TEAM[home_team_id]
        yest_away, yest_home = _fetch_lineup(yest_game_id)
        home_ops = _lineup_ops(yest_home) or _lineup_ops(yest_away)

    # Final fallback: 30-day team average
    if away_ops is None:
        away_ops = TEAM_AVG_OPS.get(away_abbv, LEAGUE_OPS)
    if home_ops is None:
        home_ops = TEAM_AVG_OPS.get(home_abbv, LEAGUE_OPS)

    return away_ops, home_ops

# ── 4c. Weather (Open-Meteo) ──────────────────────────────────────────────────
print('Fetching weather from Open-Meteo...')
WEATHER_CACHE = {}
for g in games:
    home = g['home_abbv']
    if home not in WEATHER_CACHE:
        WEATHER_CACHE[home] = fetch_weather(home, str(TODAY))
live_count = sum(1 for v in WEATHER_CACHE.values() if v != (65, 0))
print(f'  Fetched weather for {live_count}/{len(WEATHER_CACHE)} stadiums')

# ── 4d. Odds (BettingPros) ────────────────────────────────────────────────────
print('Fetching odds (The Odds API / Bovada fallback)...')
GAME_ODDS, ALT_YRFI_SIGNALS = fetch_odds()
using_real_odds = bool(GAME_ODDS)
if using_real_odds:
    print(f'  Loaded odds for {len(GAME_ODDS)} games')
else:
    print(f'  WARNING: No odds available — EV will not be computed')
if ALT_YRFI_SIGNALS:
    for mk, sig in ALT_YRFI_SIGNALS.items():
        am = sig['best_over_american']
        print(f'  *** STRONG YRFI SIGNAL: {mk}  Over {sig["point"]} at '
              f'{"+" if am>=0 else ""}{am} (market pricing {sig["point"]}+ runs at ~50%+)')

def get_odds(matchup_key):
    return GAME_ODDS.get(matchup_key)  # None if not available

# ══════════════════════════════════════════════════════════════════════════════
# PART 5 — BUILD FEATURE ROWS FOR TODAY'S GAMES
# ══════════════════════════════════════════════════════════════════════════════
print('\nBuilding feature rows...')
rows = []
for g in games:
    home = g['home_abbv']
    away = g['away_abbv']
    hp   = g['home_pitcher']
    ap   = g['away_pitcher']

    home_ra   = get_pitcher_ra(hp)
    home_whip = get_pitcher_whip(hp)
    away_ra   = get_pitcher_ra(ap)
    away_whip = get_pitcher_whip(ap)
    away_ops, home_ops = fetch_game_ops(g['game_id'], away, home, g.get('away_id'), g.get('home_id'))
    park      = PARK_FACTORS.get(home, 100)
    temp, rain = WEATHER_CACHE.get(home, (65, 0))

    home_yrfi = get_yrfi(home, yrfi_home, yrfi_home_prev, yrfi_overall)
    away_yrfi = get_yrfi(away, yrfi_away, yrfi_away_prev, yrfi_overall)

    rows.append({
        'matchup':                f'{away}@{home}',
        'home_pitcher':           hp,
        'away_pitcher':           ap,
        'away_ops':        away_ops,
        'home_ops':        home_ops,
        'home_yrfi_pct':   home_yrfi,
        'away_yrfi_pct':   away_yrfi,
        'home_pitcher_ra': home_ra,
        'home_whip':       home_whip,
        'away_pitcher_ra': away_ra,
        'away_whip':       away_whip,
        'park_factor':            park,
        'temp':                   temp,
        'rain':                   rain,
        # display-only
        '_home_ra':   home_ra,
        '_home_whip': home_whip,
        '_away_ra':   away_ra,
        '_away_whip': away_whip,
        '_home_ops':  home_ops,
        '_away_ops':  away_ops,
        '_temp':      temp,
        '_rain':      rain,
        '_home_yrfi': home_yrfi,
        '_away_yrfi': away_yrfi,
        '_park':      park,
    })

today_df = pd.DataFrame(rows)

# ══════════════════════════════════════════════════════════════════════════════
# PART 6 — APPLY MODELS
# ══════════════════════════════════════════════════════════════════════════════
X_today    = scaler.transform(today_df[FEATURES].values)
X_today_nn = nn_scaler.transform(today_df[FEATURES].values)

# LR
lr_probs = lr.predict_proba(X_today)[:, 1]
today_df['lr_prob_yrfi'] = lr_probs
today_df['lr_prob_nrfi'] = 1 - lr_probs
today_df['lr_pred']      = np.where(lr_probs > BOUNDARY, 'YRFI', 'NRFI')
today_df['lr_conf']      = np.where(lr_probs > BOUNDARY, lr_probs, 1 - lr_probs)

# NN
nn_probs = nn.predict(X_today_nn, verbose=0).flatten()
today_df['nn_prob_yrfi'] = nn_probs
today_df['nn_prob_nrfi'] = 1 - nn_probs
today_df['nn_pred']      = np.where(nn_probs > nn_meta['boundary'], 'YRFI', 'NRFI')
today_df['nn_conf']      = np.where(nn_probs > nn_meta['boundary'], nn_probs, 1 - nn_probs)

LOW, HIGH = round(1 - THRESHOLD, 3), THRESHOLD
today_df['lr_confident'] = (lr_probs < LOW) | (lr_probs > HIGH)
today_df['nn_confident'] = (nn_probs < nn_low) | (nn_probs > nn_high)
today_df['consensus']    = today_df['lr_confident'] & today_df['nn_confident'] \
                           & (today_df['lr_pred'] == today_df['nn_pred'])

# EV — only when real odds available; computed for the prediction each model makes
def compute_ev(probs_yrfi, preds, matchup_series):
    ev = []
    for prob, pred, matchup in zip(probs_yrfi, preds, matchup_series):
        odds = get_odds(matchup)
        if odds is None:
            ev.append(None)
        else:
            nrfi_odds, yrfi_odds = odds
            prob_win = (1 - prob) if pred == 'NRFI' else prob
            ev.append(round(ev_per_unit(prob_win, nrfi_odds if pred == 'NRFI' else yrfi_odds), 4))
    return ev

today_df['lr_ev'] = compute_ev(lr_probs, today_df['lr_pred'], today_df['matchup'])
today_df['nn_ev'] = compute_ev(nn_probs, today_df['nn_pred'], today_df['matchup'])

# ══════════════════════════════════════════════════════════════════════════════
# PART 7 — OUTPUT
# ══════════════════════════════════════════════════════════════════════════════
odds_note = '' if using_real_odds else '  (no odds — EV unavailable)'
header_note = (f'LR: <{LOW} / >{HIGH}  '
               f'NN: <{nn_low} / >{nn_high}  [1u = ${UNIT}]{odds_note}')

# ── All-games table ───────────────────────────────────────────────────────────
print('\n' + '=' * 80)
print(f'ALL GAMES  ({header_note})')
print('=' * 80)
print(f'  {"Matchup":<14}  {"LR NRFI%":>9} {"LR YRFI%":>9} {"LR":>5} {"LR EV/u":>8}'
      f'  {"NN NRFI%":>9} {"NN YRFI%":>9} {"NN":>5} {"NN EV/u":>8}  {"Flags"}')
print(f'  {"-" * 76}')
for _, r in today_df.sort_values('lr_prob_nrfi', ascending=False).iterrows():
    lr_ev  = f'{r["lr_ev"]:>+7.3f}u' if r['lr_ev'] is not None else f'{"N/A":>8}'
    nn_ev  = f'{r["nn_ev"]:>+7.3f}u' if r['nn_ev'] is not None else f'{"N/A":>8}'
    flags  = []
    if r['lr_confident']: flags.append('LR')
    if r['nn_confident']: flags.append('NN')
    if r['consensus']:    flags[-1] += '*'  # * marks consensus
    print(f'  {r["matchup"]:<14}  {r["lr_prob_nrfi"]:>8.1%} {r["lr_prob_yrfi"]:>9.1%}'
          f' {r["lr_pred"]:>5} {lr_ev}'
          f'  {r["nn_prob_nrfi"]:>8.1%} {r["nn_prob_yrfi"]:>9.1%}'
          f' {r["nn_pred"]:>5} {nn_ev}'
          f'  {" ".join(flags)}')

def print_picks_section(mask, model_label, pred_col, conf_col, ev_col, prob_nrfi_col, prob_yrfi_col):
    section_picks = today_df[mask].copy()
    n_total = len(today_df)
    print(f'\n{"=" * 80}')
    print(f'{model_label} PICKS  ({len(section_picks)} of {n_total} games — '
          f'{len(section_picks)/max(n_total,1):.0%} coverage)  [1u = ${UNIT}]')
    print(f'{"=" * 80}')
    if section_picks.empty:
        print('  No games cleared the confidence threshold.')
        return []
    payload = []
    for _, r in section_picks.sort_values(conf_col, ascending=False).iterrows():
        odds     = get_odds(r['matchup'])
        has_odds = odds is not None
        ev       = r[ev_col]
        if has_odds:
            nrfi_odds, yrfi_odds = odds
            disp_odds  = nrfi_odds if r[pred_col] == 'NRFI' else yrfi_odds
            ev_dollars = round(ev * UNIT, 2)
        consensus_tag = '  *** CONSENSUS ***' if r['consensus'] else ''
        print(f'\n  {r["matchup"]}  ->  {r[pred_col]}{consensus_tag}')
        print(f'    Probability:  NRFI {r[prob_nrfi_col]:.1%}  /  YRFI {r[prob_yrfi_col]:.1%}')
        if has_odds:
            print(f'    Confidence:   {r[conf_col]:.1%}   '
                  f'EV: {ev:+.3f}u (${ev_dollars:+.2f})  Odds: {disp_odds:+d}')
        else:
            print(f'    Confidence:   {r[conf_col]:.1%}   (no odds — EV unavailable)')
        print(f'    Starters:     {r["away_pitcher"]} (away)  vs  {r["home_pitcher"]} (home)')
        print(f'    Home RA/WHIP: {r["_home_ra"]:.2f} / {r["_home_whip"]:.2f}  '
              f'OPS: {r["_home_ops"]:.3f}  Park: {r["_park"]}')
        print(f'    Away OPS:     {r["_away_ops"]:.3f}  '
              f'Temp: {r["_temp"]}°F  Rain: {"Yes" if r["_rain"] else "No"}')
        print(f'    YRFI pct:     home {r["_home_yrfi"]:.1%} (Home split)  '
              f'away {r["_away_yrfi"]:.1%} (Away split)')
        payload.append({
            'model':        model_label,
            'matchup':      r['matchup'],
            'prediction':   r[pred_col],
            'prob_nrfi':    round(r[prob_nrfi_col], 4),
            'prob_yrfi':    round(r[prob_yrfi_col], 4),
            'confidence':   round(r[conf_col], 4),
            'consensus':    bool(r['consensus']),
            'ev_units':     ev,
            'ev_dollars':   round(ev * UNIT, 2) if ev is not None else None,
            'unit_size':    UNIT,
            'odds':         disp_odds if has_odds else None,
            'home_pitcher': r['home_pitcher'],
            'away_pitcher': r['away_pitcher'],
            'temp':         r['_temp'],
            'rain':         bool(r['_rain']),
        })
    return payload

lr_payload = print_picks_section(
    today_df['lr_confident'], 'LR',
    'lr_pred', 'lr_conf', 'lr_ev', 'lr_prob_nrfi', 'lr_prob_yrfi',
)
nn_payload = print_picks_section(
    today_df['nn_confident'], 'NN',
    'nn_pred', 'nn_conf', 'nn_ev', 'nn_prob_nrfi', 'nn_prob_yrfi',
)

picks_payload = lr_payload + nn_payload

deliver_picks(
    picks_payload, str(TODAY), THRESHOLD,
    best[1] if best else 0.0,
    best[3] if best else 0.0,
)

# ── Save full game log (all games, both models) ───────────────────────────────
def save_game_log(df, date_str, lr_threshold, nn_threshold_low, nn_threshold_high,
                  lr_boundary, nn_boundary, cv_acc, cv_cov):
    """
    Save a detailed per-game snapshot to S3 for later result grading.
    Includes raw model outputs, features, odds, and thresholds used.
    """
    s3_bucket = os.environ.get('NRFI_OUTPUT_BUCKET')
    if not s3_bucket:
        return
    log_rows = []
    for _, r in df.iterrows():
        bo = get_odds(r['matchup'])
        nrfi_odds_val = bo[0] if bo else None
        yrfi_odds_val = bo[1] if bo else None
        log_rows.append({
            'date':               date_str,
            'matchup':            r['matchup'],
            'home_pitcher':       r['home_pitcher'],
            'away_pitcher':       r['away_pitcher'],
            # LR outputs
            'lr_prob_nrfi':       round(r['lr_prob_nrfi'], 4),
            'lr_prob_yrfi':       round(r['lr_prob_yrfi'], 4),
            'lr_pred':            r['lr_pred'],
            'lr_conf':            round(r['lr_conf'], 4),
            'lr_confident':       bool(r['lr_confident']),
            'lr_ev':              r['lr_ev'],
            'lr_threshold_low':   round(1 - lr_threshold, 3),
            'lr_threshold_high':  round(lr_threshold, 3),
            'lr_boundary':        round(lr_boundary, 4),
            # NN outputs
            'nn_prob_nrfi':       round(r['nn_prob_nrfi'], 4),
            'nn_prob_yrfi':       round(r['nn_prob_yrfi'], 4),
            'nn_pred':            r['nn_pred'],
            'nn_conf':            round(r['nn_conf'], 4),
            'nn_confident':       bool(r['nn_confident']),
            'nn_ev':              r['nn_ev'],
            'nn_threshold_low':   round(nn_threshold_low, 3),
            'nn_threshold_high':  round(nn_threshold_high, 3),
            'nn_boundary':        round(nn_boundary, 4),
            # Consensus
            'consensus':          bool(r['consensus']),
            # CV stats for this run
            'cv_acc':             round(cv_acc, 4),
            'cv_cov':             round(cv_cov, 4),
            # Odds
            'nrfi_odds':          nrfi_odds_val,
            'yrfi_odds':          yrfi_odds_val,
            # Features
            'home_yrfi_pct':      r['_home_yrfi'],
            'away_yrfi_pct':      r['_away_yrfi'],
            'home_ra':            r['_home_ra'],
            'home_whip':          r['_home_whip'],
            'home_ops':           r['_home_ops'],
            'away_ops':           r['_away_ops'],
            'park_factor':        r['_park'],
            'temp':               r['_temp'],
            'rain':               int(r['_rain']),
            # Actuals (filled in next day by grade_yesterday)
            'actual_yrfi':        None,
            'lr_correct':         None,
            'nn_correct':         None,
        })
    import io
    log_df = pd.DataFrame(log_rows)
    buf = io.BytesIO()
    log_df.to_csv(buf, index=False)
    key = f'game_log/{TODAY.year}/{date_str}.csv'
    try:
        import boto3
        boto3.client('s3').put_object(
            Bucket=s3_bucket, Key=key,
            Body=buf.getvalue(), ContentType='text/csv',
        )
        print(f'  Game log written to s3://{s3_bucket}/{key}')
    except Exception as ex:
        print(f'  WARNING: game log write failed ({ex})')

save_game_log(
    today_df, str(TODAY), THRESHOLD,
    LOW, HIGH,
    BOUNDARY, nn_meta['boundary'],
    best[1] if best else 0.0,
    best[3] if best else 0.0,
)

# ── SES email notification ────────────────────────────────────────────────────
# Build email subject — include yesterday's record if available
_yest_summary = ''
if yesterday_log_df is not None and not yesterday_log_df.empty:
    _conf = yesterday_log_df[yesterday_log_df['lr_confident'] | yesterday_log_df['nn_confident']]
    _graded = _conf[_conf['lr_correct'].notna() | _conf['nn_correct'].notna()]
    if not _graded.empty:
        _lr_g = _conf[_conf['lr_confident'] & _conf['lr_correct'].notna()]
        _w = int(_lr_g['lr_correct'].sum()) if not _lr_g.empty else 0
        _l = len(_lr_g) - _w
        _pl = sum(
            _pick_pl(r['lr_correct'], r['lr_pred'], r.get('nrfi_odds'), r.get('yrfi_odds'))
            for _, r in _lr_g.iterrows() if pd.notna(r['lr_correct'])
        )
        _sign = '+' if _pl >= 0 else ''
        _yest_summary = f' | Yesterday {_w}-{_l} ({_sign}${_pl:.2f})'

_n_picks = len([p for p in picks_payload if not p.get('consensus')])
_n_cons  = len([p for p in picks_payload if p.get('consensus')])
_picks_summary = f'{_n_picks} pick{"s" if _n_picks != 1 else ""}'
if _n_cons:
    _picks_summary += f', {_n_cons} consensus'
_session_label = {'afternoon': ' (Afternoon)', 'evening': ' (Evening)', 'all': ''}.get(SESSION, '')
email_subject = f'NRFI {str(TODAY)}{_session_label} — {_picks_summary}{_yest_summary}'

email_html = build_email_html(
    date_str=str(TODAY),
    picks_rows=picks_payload,
    yesterday_rows=yesterday_log_df,
    ytd_df=ytd_df,
    today_df_all=today_df,
    lr_threshold=THRESHOLD,
    nn_threshold=nn_high,
    cv_acc=best[1] if best else 0.0,
    cv_cov=best[3] if best else 0.0,
    alt_signals=ALT_YRFI_SIGNALS,
)
send_email(email_html, email_subject, str(TODAY))
