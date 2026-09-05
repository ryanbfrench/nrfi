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

# ── Utils (structured logging, P/L calc, email chart) ────────────────────────
try:
    from utils.logger import log as _log, metric as cw_metric
    from utils.pl_calc import compute_pl
    from utils.email_charts import build_threshold_timeline
    from utils.email_html import build_email_html, send_email
except ImportError:
    def _log(level, msg, **ctx): print(f'[{level}] {msg}')
    def cw_metric(*args, **kwargs): pass
    def compute_pl(correct, pred, nrfi_odds, yrfi_odds, unit=10):
        """Fallback: no P/L without real odds."""
        if correct is None or (isinstance(correct, float) and __import__('math').isnan(correct)):
            return None
        raw = nrfi_odds if pred == 'NRFI' else yrfi_odds
        if raw is None or (isinstance(raw, float) and __import__('math').isnan(raw)):
            return None
        odds = int(raw)
        if int(correct) == 1:
            return round(unit * (100 / abs(odds) if odds < 0 else odds / 100), 2)
        return -float(unit)
    def build_threshold_timeline(*args, **kwargs): return None
    def build_email_html(*args, **kwargs): return '<html><body>Email unavailable</body></html>'
    def send_email(*args, **kwargs): pass

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
# Weights-only numpy persistence (see _save_nn_to_s3). Default key is .npz; the old .keras full-model file is abandoned (version-fragile). A path override is stored as whatever bytes we write regardless of extension.
NN_MODEL_PATH  = os.environ.get('NRFI_NN_MODEL_PATH', 's3://nrfi-store/models/nn_weights.npz')
SESSION        = os.environ.get('SESSION', 'all')   # 'afternoon' | 'evening' | 'all'
TODAY          = date.today()
YESTERDAY      = TODAY - timedelta(days=1)
SEASON_START   = date(2026, 4, 15)   # first date counted in YTD stats — based on training data cutoff

# Cutoff for afternoon vs evening: 5pm ET = 21:00 UTC (EDT, April-October)
AFTERNOON_CUTOFF_UTC_HOUR = 21
MIN_COVERAGE       = 0.10
MAX_COVERAGE       = 0.15
# LR confidence band: percentile of recent LR outputs, BOUNDARY UNMOVED (2026-08-25) — tails are independent allocations (SUM = coverage target, SPLIT = directional tilt) so the YRFI lean survives a coverage fix; 30% at 3:1 tuned on a full-season 2026 walk-forward, CV margin sweep below is the small-pool fallback. See DECISIONS.md 2026-08-25 / 2026-08-26.
LR_YRFI_TAIL       = 0.225  # fraction of pool above `high` -> YRFI picks
LR_NRFI_TAIL       = 0.075  # fraction of pool below `low`  -> NRFI picks
LR_POOL_DAYS       = 45
LR_POOL_MIN        = 100
# Hard floor: the 2026-08-14 corpus un-freeze shifted the LR output distribution, so pooling across it would size the band on a distribution the model no longer produces — set to None once the break is older than LR_POOL_DAYS.
LR_POOL_NOT_BEFORE = date(2026, 8, 15)

# ── Blend model (third pick set, 2026-08-26) ──────────────────────────────────
# LR shrunk toward the de-vigged market consensus; which source is better flips mid-season, so the weight is set ADAPTIVELY from each one's trailing realised edge. See DECISIONS.md 2026-08-26.
BLEND_ENABLED       = os.environ.get('NRFI_BLEND_ENABLED', '1') != '0'
BLEND_LOOKBACK_DAYS = 60
BLEND_MIN_GRADED    = 150
BLEND_DEFAULT_W     = 0.50
BLEND_W_MIN         = 0.20
BLEND_W_MAX         = 0.90
# NN confidence band is set by PERCENTILE of recent NN outputs (rolling window), not a fixed CV margin (2026-06-07). A margin tuned on historical CV undershot deployment coverage badly (2.8% vs 15% target) because the live output distribution is shifted/ compressed. Percentile calibration on a recent window hits the coverage target on the distribution actually being scored. See DECISIONS.md 2026-06-07.
NN_COVERAGE_TARGET = 0.15   # fraction of games to flag confident (split across both tails)
NN_POOL_DAYS       = 45     # rolling window of recent game_logs to calibrate the band
NN_POOL_MIN        = 100    # min pooled outputs required; else fall back to margin band
UNIT               = 10    # dollars per unit
HIST_ODDS_API_KEY  = os.environ.get('HISTORICAL_ODDS_API_KEY')
RECENCY_HALF_LIFE  = 365   # days; games 1yr old carry ~37% weight, 2yr ~14%, 3yr ~5%

# Online (incremental) NN update — methods A + B + C (see _incremental_update):   A streaming SGD,  B experience replay,  C L2-SP anchor to previous weights.
NN_ONLINE_LR       = 3e-4  # A: low LR — each daily batch nudges weights only slightly
NN_ONLINE_MOMENTUM = 0.9   # A: SGD momentum
NN_REPLAY_N        = 256   # B: games sampled UNIFORMLY from all history per update
NN_L2SP_LAMBDA     = 1e-3  # C: strength of the anchor toward pre-update weights

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
    # 'OAK' = Sutter Health Park, West Sacramento
    'NYY': (40.8296,  -73.9262), 'OAK': (38.5802, -121.5133),
    'PHI': (39.9061,  -75.1665), 'PIT': (40.4469,  -80.0057),
    'SD':  (32.7076, -117.1570), 'SF':  (37.7786, -122.3893),
    'SEA': (47.5914, -122.3325), 'STL': (38.6226,  -90.1928),
    'TB':  (27.7683,  -82.6534), 'TEX': (32.7473,  -97.0845),
    'TOR': (43.6414,  -79.3894), 'WAS': (38.8730,  -77.0074),
}

# Climate-controlled parks: roof shut all season, so the boxscore reports a near-constant indoor temp and outdoor weather is the wrong variable — feeding it ran +16 to +36F wrong on `temp`, the LR's strongest feature. Values are the 2021-2025 boxscore means (sd 0.8-2.7F). TOR/MIL/SEA are deliberately excluded: their roofs track ambient (error <2F). See DECISIONS.md 2026-08-25.
DOME_TEMPS = {'HOU': 73.0, 'TB': 72.0, 'MIA': 72.5, 'TEX': 74.0, 'ARI': 78.5}

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
# Must return FIRST-PITCH temperature, not the daily maximum. The training corpus takes `temp`
# from the MLB boxscore "Weather" line (lambda_function.parse_weather) = the temperature actually
# reported at the park at game time. Requesting `daily=temperature_2m_max` fed a different
# variable: measured against the Open-Meteo archive over 6 parks (Aug 5-11 2026), daily max runs
# +6.7F above the 19:00 local temperature (LAD +12.6, NYY +7.7, BOS +6.1, COL +3.8). Because the
# diurnal swing widens through the summer, the live-vs-training gap grew Apr +5.3 -> Aug +9.2F,
# which pushed every August game's YRFI probability up and inflated LR coverage to 46%. Hourly
# temperature at the game's local start hour is the like-for-like analogue of the boxscore value.
def fetch_weather(abbv, target_date, first_pitch_utc=None):
    """Return (temp_F, rain, source) for a team's home stadium at first pitch on target_date. `first_pitch_utc` is an ISO-8601 UTC string (statsapi `game_datetime`); when it is missing we fall back to 19:00 stadium-local, the modal MLB start. `rain` is precipitation over the 3h game window — kept for the game_log/DQ only, it is NOT a model feature (see FEATURES). source is 'api' on a live fetch, 'dome' for a climate-controlled park (see DOME_TEMPS — constant indoor temp, no API call), 'default' on any failure."""
    # Roof closed all season -> indoor temp is a constant and outdoor weather is the wrong variable; return before the API call, rain is 0 by construction under a closed roof.
    if abbv in DOME_TEMPS:
        return DOME_TEMPS[abbv], 0, 'dome'

    coords = STADIUM_COORDS.get(abbv)
    if coords is None:
        return 65, 0, 'default'
    lat, lon = coords
    try:
        # Pull a 2-day hourly window so a late local start that rolls past midnight still resolves.
        end_date = (datetime.fromisoformat(str(target_date)).date() + timedelta(days=1)).isoformat()
        url = (
            f'https://api.open-meteo.com/v1/forecast'
            f'?latitude={lat}&longitude={lon}'
            f'&hourly=temperature_2m,precipitation'
            f'&temperature_unit=fahrenheit'
            f'&timezone=auto'
            f'&start_date={target_date}&end_date={end_date}'
        )
        payload = requests.get(url, timeout=10).json()
        hourly  = payload['hourly']
        times   = hourly['time']                       # ISO strings in STADIUM-local time
        temps   = hourly['temperature_2m']
        precip  = hourly['precipitation']

        # Convert first pitch to stadium-local using the offset Open-Meteo resolved for this lat/lon,
        # so we never need a local tz database.
        offset = int(payload.get('utc_offset_seconds', 0))
        if first_pitch_utc:
            local_dt = (datetime.fromisoformat(str(first_pitch_utc).replace('Z', '+00:00'))
                        + timedelta(seconds=offset))
            stamp = local_dt.strftime('%Y-%m-%dT%H:00')
        else:
            stamp = f'{target_date}T19:00'
        idx = times.index(stamp) if stamp in times else times.index(f'{target_date}T19:00')

        temp = round(temps[idx])
        # Rain over first pitch + 3h (covers well beyond the 1st inning this model cares about).
        window = [p for p in precip[idx:idx + 3] if p is not None]
        rain = 1 if sum(window) > 0.5 else 0
        return temp, rain, 'api'
    except Exception:
        return 65, 0, 'default'

# ── Odds scraping (The Odds API — totals_1st_1_innings) ────────────────────── Market: Over/Under 0.5 first-inning runs. Over = YRFI, Under = NRFI. Free tier: 500 requests/month. ~15 games/day ≈ 450 requests/month. Env var: ODDS_API_KEY

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
    """Fetch NRFI/YRFI odds via The Odds API. Primary market: totals_1st_1_innings at point=0.5 only. Over 0.5 = YRFI, Under 0.5 = NRFI. Lines at any other point are ignored for NRFI/YRFI odds (a line at 1.5 is a different bet). Requests American-format odds directly (same as backfill_odds_2025.py) and saves a raw snapshot to s3://nrfi-store/odds/{year}/{date}.json so historical odds are stored in a consistent format for future analysis. Returns: odds — dict: 'AWAY@HOME' -> (nrfi_odds, yrfi_odds) American ints Falls back to Bovada scrape if API key not set."""
    api_key = os.environ.get('ODDS_API_KEY')
    if not api_key:
        return _fetch_odds_bovada_fallback()

    odds      = {}
    raw_events = []   # collect for S3 snapshot
    try:
        events_resp = requests.get(
            'https://api.the-odds-api.com/v4/sports/baseball_mlb/events',
            params={'apiKey': api_key, 'daysFrom': 1}, timeout=15
        )
        if events_resp.status_code != 200:
            return _fetch_odds_bovada_fallback()
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
                        'markets': 'totals_1st_1_innings',
                        'oddsFormat': 'american'},
                timeout=15
            )
            if r.status_code != 200:
                continue

            event_data = r.json()
            raw_events.append(event_data)   # accumulate for S3

            matchup_key = f'{away_abbv}@{home_abbv}'
            best_yrfi = best_nrfi = None

            for bk in event_data.get('bookmakers', []):
                for mkt in bk.get('markets', []):
                    for outcome in mkt.get('outcomes', []):
                        american = outcome.get('price')   # already American int
                        point    = outcome.get('point')
                        if american is None or point is None:
                            continue
                        name = outcome['name']
                        if point == 0.5:
                            if name == 'Over' and (best_yrfi is None or american > best_yrfi):
                                best_yrfi = american
                            elif name == 'Under' and (best_nrfi is None or american > best_nrfi):
                                best_nrfi = american

            if best_nrfi is not None and best_yrfi is not None:
                odds[matchup_key] = (best_nrfi, best_yrfi)

    except Exception as ex:
        print(f'  WARNING: Odds API fetch failed ({ex}) — trying Bovada fallback')
        return _fetch_odds_bovada_fallback()

    # Save raw snapshot to S3 — same path/format as backfill_odds_2025.py s3://nrfi-store/odds/{year}/{date}.json
    if raw_events:
        _odds_s3_key = f'odds/{TODAY.year}/{TODAY.isoformat()}.json'
        try:
            import boto3
            boto3.client('s3').put_object(
                Bucket='nrfi-store',
                Key=_odds_s3_key,
                Body=json.dumps(raw_events, separators=(',', ':')),
                ContentType='application/json',
            )
            print(f'  Odds snapshot saved to s3://nrfi-store/{_odds_s3_key}')
        except Exception as _ex:
            print(f'  WARNING: could not save odds snapshot to S3 ({_ex})')

    return odds


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
    return odds

# ── Output delivery (SNS + S3 JSON) ──────────────────────────────────────────
def deliver_picks(picks_rows, date_str, band_low, band_high, cv_acc, cv_cov):
    """Write picks JSON to S3 and/or publish to SNS if env vars are configured."""
    payload = {
        'date':      date_str,
        'threshold': {'low': round(band_low, 3), 'high': round(band_high, 3)},
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

# ── SES HTML email ─────────────────────────────────────────────────────────── build_email_html and send_email live in utils/email_html.py

def _odds_str(val):
    if val is None: return '—'
    return f'+{val}' if val > 0 else str(val)

def _ev_str(units, dollars):
    if units is None: return '—'
    sign = '+' if units >= 0 else ''
    return f'{sign}{units:.3f}u (${sign}{dollars:.2f})'


# ── Grade yesterday's picks ───────────────────────────────────────────────────
def grade_yesterday():
    """Load yesterday's game log and Lambda results file from S3. Fill in actual_yrfi, lr_correct, nn_correct on every game. Append completed rows to results/results.csv. Prints a W/L summary for confident picks. Silently skips if either file is missing."""
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

    # blend_correct: absent in logs written before the blend rollout, so guard the column.
    if 'blend_pred' in log_df.columns:
        log_df['blend_correct'] = log_df.apply(
            lambda r: (
                None if pd.isna(r['actual_yrfi']) or pd.isna(r.get('blend_pred')) else
                int(('YRFI' if r['actual_yrfi'] == 1 else 'NRFI') == r['blend_pred'])
            ), axis=1
        )
    else:
        log_df['blend_correct'] = None

    # Persist the graded log back to its own key — missing until 2026-08-30, actuals were filled into log_df in memory then discarded, so game_log/ rows kept actual_yrfi = NaN forever and silently diverged from results.csv (1,062/2,019 graded vs 1,877), which is how the first P/L-repair dry-run reported a flattering 65.4% off a half-season; results.csv remains authoritative, this keeps game_log a faithful copy.
    try:
        _gbuf = io.BytesIO()
        log_df.to_csv(_gbuf, index=False)
        s3.put_object(Bucket=s3_bucket, Key=log_key,
                      Body=_gbuf.getvalue(), ContentType='text/csv')
        _ngraded = int(log_df['actual_yrfi'].notna().sum())
        print(f'  Game log graded in place: {_ngraded}/{len(log_df)} rows -> {log_key}')
    except Exception as _gex:
        print(f'  WARNING: could not write graded game log back ({_gex})')

    # Print summary for confident picks
    print(f'\n{"=" * 60}')
    print(f'YESTERDAY\'S RESULTS — {ystr}')
    print(f'{"=" * 60}')
    lr_conf = log_df[log_df['lr_confident'] == True]
    nn_conf = log_df[log_df['nn_confident'] == True]
    bl_conf = (log_df[log_df['blend_confident'] == True]
               if 'blend_confident' in log_df.columns else log_df.iloc[0:0])
    for label, subset, pred_col, correct_col, conf_col in [
        ('LR', lr_conf, 'lr_pred', 'lr_correct', 'lr_conf'),
        ('NN', nn_conf, 'nn_pred', 'nn_correct', 'nn_conf'),
        ('BLEND', bl_conf, 'blend_pred', 'blend_correct', 'blend_conf'),
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

    for label, subset, correct_col in [('LR', lr_conf, 'lr_correct'), ('NN', nn_conf, 'nn_correct'),
                                       ('BLEND', bl_conf, 'blend_correct')]:
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

        # Backfill missing odds for confident picks across all historical dates
        try:
            from utils.odds_backfill import backfill_missing_odds
            combined = backfill_missing_odds(combined, s3, s3_bucket, HIST_ODDS_API_KEY)
        except Exception as _bf_ex:
            print(f'  WARNING: odds backfill failed ({_bf_ex})')

        buf = io.BytesIO()
        combined.to_csv(buf, index=False)
        s3.put_object(Bucket=s3_bucket, Key=results_log_key,
                      Body=buf.getvalue(), ContentType='text/csv')
        print(f'  Results log updated: s3://{s3_bucket}/{results_log_key}')

        # Emit yesterday's performance metrics (use combined so backfilled odds are included)
        _yest_rows = combined[(combined['date'] == ystr) & combined['lr_confident'] & combined['lr_correct'].notna()]
        if not _yest_rows.empty:
            _w = int(_yest_rows['lr_correct'].sum())
            _l = len(_yest_rows) - _w
            cw_metric('YesterdayWins',   _w)
            cw_metric('YesterdayLosses', _l)
            _pl_vals = [compute_pl(r['lr_correct'], r['lr_pred'], r.get('nrfi_odds'), r.get('yrfi_odds'))
                        for _, r in _yest_rows.iterrows()]
            cw_metric('YesterdayPL', sum(v for v in _pl_vals if v is not None), unit='None')

        # YTD P/L metric
        _ytd_rows = combined[
            combined['date'].str.startswith(str(TODAY.year)) &
            combined['lr_confident'] &
            combined['lr_correct'].notna()
        ] if not combined.empty else pd.DataFrame()
        if not _ytd_rows.empty:
            _ytd_pl_vals = [compute_pl(r['lr_correct'], r['lr_pred'], r.get('nrfi_odds'), r.get('yrfi_odds'))
                            for _, r in _ytd_rows.iterrows()]
            cw_metric('YTDProfitLoss', sum(v for v in _ytd_pl_vals if v is not None), unit='None')

    except Exception as ex:
        print(f'  WARNING: could not update results log ({ex})')

    ytd_df = (combined[combined['date'] >= SEASON_START.isoformat()]
              if not combined.empty else pd.DataFrame())

    _analyze_dq_outcomes(combined)
    return log_df, ytd_df


def _analyze_dq_outcomes(df):
    """Compare model accuracy on high-data-quality vs imputed games. Printed to console; only runs when dq_n_imputed column is present and we have at least 20 graded LR confident picks to draw from."""
    if df.empty or 'dq_n_imputed' not in df.columns:
        return

    graded = df[
        df['lr_confident'].fillna(False).astype(bool) &
        df['lr_correct'].notna() &
        df['dq_n_imputed'].notna()
    ].copy()

    if len(graded) < 20:
        return  # not enough data yet

    graded['dq_n_imputed'] = graded['dq_n_imputed'].astype(int)
    overall_acc = graded['lr_correct'].mean()
    n_total = len(graded)

    print('\n' + '=' * 60)
    print('DATA QUALITY OUTCOME ANALYSIS  (LR confident picks, YTD)')
    print('=' * 60)
    print(f'  Overall: {overall_acc:.1%} accuracy  ({n_total} graded picks)\n')

    # Accuracy by imputation level
    for label, mask in [
        ('0 features imputed (clean)',  graded['dq_n_imputed'] == 0),
        ('1 feature imputed',           graded['dq_n_imputed'] == 1),
        ('2 features imputed',          graded['dq_n_imputed'] == 2),
        ('3+ features imputed',         graded['dq_n_imputed'] >= 3),
    ]:
        subset = graded[mask]
        if len(subset) < 3:
            continue
        acc = subset['lr_correct'].mean()
        diff = acc - overall_acc
        sign = '+' if diff >= 0 else ''
        print(f'  {label:<35} n={len(subset):>3}  acc={acc:.1%}  ({sign}{diff:+.1%} vs avg)')

    # Per-flag accuracy (only flags with enough data)
    print()
    flag_checks = [
        ('Home RA imputed',   'dq_home_ra_imp',   True),
        ('Away RA imputed',   'dq_away_ra_imp',   True),
        ('Home WHIP imputed', 'dq_home_whip_imp', True),
        ('Away WHIP imputed', 'dq_away_whip_imp', True),
        ('Weather defaulted', 'dq_weather_src',   'default'),
    ]
    for label, col, imp_val in flag_checks:
        if col not in graded.columns:
            continue
        imp_mask = graded[col] == imp_val
        imp_sub  = graded[imp_mask]
        live_sub = graded[~imp_mask]
        if len(imp_sub) < 5:
            continue
        print(f'  {label:<30} imputed={imp_sub["lr_correct"].mean():.1%} (n={len(imp_sub)})'
              f'  live={live_sub["lr_correct"].mean():.1%} (n={len(live_sub)})')

    # OPS source accuracy
    for side, col in [('Away OPS', 'dq_away_ops_src'), ('Home OPS', 'dq_home_ops_src')]:
        if col not in graded.columns:
            continue
        for src in ['lineup', 'yesterday', 'team_avg', 'league']:
            sub = graded[graded[col] == src]
            if len(sub) < 5:
                continue
            print(f'  {side} src={src:<12} acc={sub["lr_correct"].mean():.1%}  (n={len(sub)})')

    print()

yesterday_log_df, ytd_df = grade_yesterday()

# ══════════════════════════════════════════════════════════════════════════════ PART 1 — RETRAIN ON ALL HISTORICAL DATA ══════════════════════════════════════════════════════════════════════════════
print('=' * 60)
print(f'DAILY PICKS  {TODAY}')
print('=' * 60)

def load_season_games(year, bucket='nrfi-store'):
    """Read every Lambda daily file under data/{year}/ and return them as training rows.

    These are the SAME rows the historical corpus is built from — lambda_function writes the full
    training schema post-game, with `temp` off the boxscore Weather line and YRFI off the real
    linescore — so they can be concatenated with DATA_PATH directly.

    Rebuilt from S3 on every run rather than appended to a running file on purpose: there is no
    accumulator state to corrupt, so a bad or missing day self-heals on the next run instead of
    poisoning the corpus permanently (the failure mode that silently emptied the pitcher_ra file).

    Dates come from `id` (always the game date), NOT the stored year/month/day: the collector runs
    the morning after and stamped those with the RUN date until 2026-08-14, leaving existing rows
    one day ahead. Returns an empty frame on any failure — a season append must never be fatal.
    """
    try:
        import boto3, io
        s3 = boto3.client('s3')
        keys = []
        paginator = s3.get_paginator('list_objects_v2')
        for page in paginator.paginate(Bucket=bucket, Prefix=f'data/{year}/'):
            keys += [o['Key'] for o in page.get('Contents', []) if o['Key'].endswith('.txt')]
        frames = []
        for k in keys:
            try:
                frames.append(pd.read_csv(io.BytesIO(
                    s3.get_object(Bucket=bucket, Key=k)['Body'].read())))
            except Exception:
                continue
        if not frames:
            return pd.DataFrame()
        season = pd.concat(frames, ignore_index=True)
        season = season.drop_duplicates(subset='id', keep='last')
        gd = pd.to_datetime(season['id'].astype(str).str.slice(0, 10), errors='coerce')
        season = season[gd.notna()].copy()
        gd = gd[gd.notna()]
        season['year'], season['month'], season['day'] = gd.dt.year, gd.dt.month, gd.dt.day
        return season
    except Exception as ex:
        print(f'  WARNING: could not load {year} season games ({ex}) — training on base corpus only')
        return pd.DataFrame()


_df_raw = load_data(DATA_PATH)
_base_n = len(_df_raw)

# Un-freeze the training corpus (2026-08-14). DATA_PATH is a static 2021-2025 file, so until now
# the LR was refit every day on a corpus that never grew — it never saw a single 2026 game, and
# its scaler/coefficients stayed calibrated to a distribution the live feed had drifted away from.
# (The NN was fine: it gets the A+B+C incremental update off the same daily files.)
_season = load_season_games(TODAY.year)
if not _season.empty:
    _cols = [c for c in _df_raw.columns if c in _season.columns]
    _df_raw = pd.concat([_df_raw, _season[_cols]], ignore_index=True)
    _df_raw = _df_raw.drop_duplicates(subset='id', keep='last')
    print(f'Training corpus: {_base_n} base + {len(_df_raw) - _base_n} from {TODAY.year} '
          f'= {len(_df_raw)} games')
else:
    print(f'Training corpus: {_base_n} games (no {TODAY.year} season files found)')

# Exclude pre-April-15 games from training: pitcher RA and YRFI pct are unreliable before mid-April (Fangraphs splits need innings to accumulate; teamrankings YRFI% is based on 0-2 games and often reads as 0%).
df = _df_raw[~((_df_raw['month'] < 4) | ((_df_raw['month'] == 4) & (_df_raw['day'] < 15)))].copy()
print(f'Training set after April-15 filter: {len(df)} games '
      f'(dropped {len(_df_raw) - len(df)} pre-Apr-15 rows)')

league_avg_ra   = df[df['away_pitcher_ra'] > 0]['away_pitcher_ra'].median()
league_avg_whip = df[df['home_whip'] > 0]['home_whip'].median()
league_avg_yrfi = df[df['home_yrfi_pct'] > 0]['home_yrfi_pct'].mean()
league_avg_ops  = df['home_ops'].median()
RA_CAP = 1.5  # cap extreme small-sample RA values before imputing zeros
# NaN means the same thing as the 0 sentinel here ('no data'). The 2021-2025 base corpus has no
# nulls, so `.replace(0, ...)` alone sufficed — but the Lambda daily files write NaN when a stat
# is unavailable (~30% of 2026 rows have a null pitcher_ra), and those rows are now part of
# training, so an unfilled NaN would reach LogisticRegression.fit and raise. Mirrors
# picks_engine.impute_training.
df['away_pitcher_ra'] = df['away_pitcher_ra'].clip(upper=RA_CAP).replace(0, league_avg_ra).fillna(league_avg_ra)
df['home_pitcher_ra'] = df['home_pitcher_ra'].clip(upper=RA_CAP).replace(0, league_avg_ra).fillna(league_avg_ra)
df['away_whip']       = df['away_whip'].replace(0, league_avg_whip).fillna(league_avg_whip)
df['home_whip']       = df['home_whip'].replace(0, league_avg_whip).fillna(league_avg_whip)
df['home_yrfi_pct']   = df['home_yrfi_pct'].replace(0, league_avg_yrfi).fillna(league_avg_yrfi)
df['away_yrfi_pct']   = df['away_yrfi_pct'].replace(0, league_avg_yrfi).fillna(league_avg_yrfi)
df['away_ops']        = df['away_ops'].replace(0, league_avg_ops).fillna(league_avg_ops)
df['home_ops']        = df['home_ops'].replace(0, league_avg_ops).fillna(league_avg_ops)

# `rain` was dropped 2026-08-14. Live set it from Open-Meteo `precipitation_sum > 0.5` — total mm over the whole CALENDAR DAY — while the training corpus set it from the boxscore Weather string ("rain"/"drizzle"/"shower"), i.e. the game actually being played in rain. Result: 20.7% of live games flagged vs 1.0% of training rows, with the outcome sign flipped (training rain -> 47.7% YRFI, live -> 52.6%). The two sides were not the same variable, so the coefficient was meaningless.
FEATURES = ['away_ops', 'home_ops', 'home_yrfi_pct', 'away_yrfi_pct',
            'home_pitcher_ra', 'home_whip', 'away_pitcher_ra', 'away_whip',
            'park_factor', 'temp']

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

def _load_nn_from_s3(s3_path, input_dim):
    """Rebuild the architecture from _build_nn and load WEIGHTS ONLY (plain numpy). We deliberately do NOT use tf.keras.models.load_model: full-model serialization embeds the architecture config, whose schema changed between Keras 2 and 3 (InputLayer 'batch_input_shape' -> 'batch_shape'), so a model saved in one env fails to load in another and silently falls back to scratch-retraining -- meaning the incremental A+B+C updates never actually persist. get_weights/set_weights round a flat list of numpy arrays, which is version-independent as long as the layer shapes from _build_nn match (they're deterministic here)."""
    import boto3, io
    bucket, key = s3_path[5:].split('/', 1)
    obj  = boto3.client('s3').get_object(Bucket=bucket, Key=key)
    data = np.load(io.BytesIO(obj['Body'].read()), allow_pickle=False)
    weights = [data[f'arr_{i}'] for i in range(len(data.files))]
    # Explicit feature-count guard. set_weights would raise on its own, but with an opaque
    # layer-shape message; a FEATURES change (e.g. dropping `rain`, 2026-08-14) legitimately
    # invalidates the persisted weights and must fall through to a scratch retrain, so say so.
    saved_dim = weights[0].shape[0] if weights else None
    if saved_dim is not None and saved_dim != input_dim:
        raise ValueError(
            f'saved NN expects {saved_dim} features but FEATURES now has {input_dim} — '
            f'feature set changed, weights are not transferable')
    model = _build_nn(input_dim)
    model.set_weights(weights)
    return model

def _save_nn_to_s3(model, s3_path):
    """Save WEIGHTS ONLY as a numpy .npz (see _load_nn_from_s3 for why not load_model)."""
    try:
        import boto3, io
        bucket, key = s3_path[5:].split('/', 1)
        buf = io.BytesIO()
        np.savez(buf, *model.get_weights())
        boto3.client('s3').put_object(Bucket=bucket, Key=key, Body=buf.getvalue())
    except Exception as ex:
        print(f'  WARNING: could not save NN to S3 ({ex})')

def _nn_retrain_date_key(s3_path):
    """Sidecar key next to the weights recording when the NN was last refit from scratch."""
    bucket, key = s3_path[5:].split('/', 1)
    return bucket, key.rsplit('.', 1)[0] + '.retrained_on'

def _save_nn_retrain_date(s3_path, d):
    import boto3
    try:
        bucket, key = _nn_retrain_date_key(s3_path)
        boto3.client('s3').put_object(Bucket=bucket, Key=key, Body=d.isoformat().encode())
    except Exception as ex:
        print(f'  WARNING: could not record NN retrain date ({ex})')

def _load_nn_retrain_date(s3_path):
    """Date of the last from-scratch refit, or None. Used to keep the band-calibration pool from reaching back past a model discontinuity."""
    import boto3
    try:
        bucket, key = _nn_retrain_date_key(s3_path)
        body = boto3.client('s3').get_object(Bucket=bucket, Key=key)['Body'].read()
        return date.fromisoformat(body.decode().strip())
    except Exception:
        return None

def _recent_pool(end_date, n_days, column, not_before=None):
    """Pool `column` from the last `n_days` of game_logs ending before `end_date`. Used to calibrate a confidence band by percentile on the live output distribution, which is the only distribution the band is ever applied to. Returns a 1-D np.array (possibly empty).

    `not_before` hard-floors the window at a known model discontinuity — a from-scratch refit (NN) or a training-corpus change (LR) is a break, not drift, and outputs written before it came from a model that no longer exists. Without the floor the band silently describes the dead model, which ran NN at 30% coverage against a 15% target for ten days after the 2026-08-14 refit."""
    bucket = os.environ.get('NRFI_OUTPUT_BUCKET')
    if not bucket:
        return np.array([])
    import boto3, io
    s3 = boto3.client('s3')
    probs = []
    for i in range(1, n_days + 1):
        d = end_date - timedelta(days=i)
        if not_before is not None and d < not_before:
            continue
        key = f'game_log/{d.year}/{d.isoformat()}.csv'
        try:
            obj = s3.get_object(Bucket=bucket, Key=key)
            gl  = pd.read_csv(io.BytesIO(obj['Body'].read()))
            if column in gl.columns:
                probs.append(pd.to_numeric(gl[column], errors='coerce').dropna().values)
        except Exception:
            continue
    return np.concatenate(probs) if probs else np.array([])

def _recent_nn_pool(end_date, n_days, not_before=None):
    return _recent_pool(end_date, n_days, 'nn_prob_yrfi', not_before=not_before)

def _build_nn(input_dim):
    """8->8->8->1, lr=0.005, bs=64. Regularization dialed back to l2=1e-5, dropout=0 (2026-06-07): the prior l2=1e-3 on all three layers compressed the sigmoid output asymmetrically -- floor ~0.495, no left tail -- so the NN could NEVER produce a confident NRFI pick (0% below the NRFI band on train AND 2026 data). Light reg restores a symmetric ~25% NRFI tail (std~0.038) matching the LR. The kernel regularizer only affects the from-scratch fit; the incremental update applies its own L2-SP anchor in a custom loop, so this change is safe for daily stability."""
    reg = tf.keras.regularizers.l2(1e-5)
    model = tf.keras.Sequential([
        tf.keras.layers.Dense(8, activation='relu', input_shape=(input_dim,), kernel_regularizer=reg),
        tf.keras.layers.Dropout(0.0),
        tf.keras.layers.Dense(8, activation='relu', kernel_regularizer=reg),
        tf.keras.layers.Dropout(0.0),
        tf.keras.layers.Dense(8, activation='relu', kernel_regularizer=reg),
        tf.keras.layers.Dropout(0.0),
        tf.keras.layers.Dense(1, activation='sigmoid'),
    ])
    model.compile(optimizer=tf.keras.optimizers.Adam(0.005),
                  loss='binary_crossentropy')
    return model

def _incremental_update(model, X_new, y_new, X_hist, y_hist, *,
                        replay_n=NN_REPLAY_N, lr=NN_ONLINE_LR,
                        momentum=NN_ONLINE_MOMENTUM, l2sp=NN_L2SP_LAMBDA,
                        epochs=1, seed=None):
    """Stable online NN update — methods A + B + C. A (streaming SGD): a single epoch of SGD+momentum at a low LR, so the day's batch nudges the weights only slightly — a true online step, not the old 5-epoch refit that memorized each day's ~15 games. B (experience replay): mix the new games with `replay_n` games sampled UNIFORMLY from all history, so every update keeps re-seeing the full distribution and cannot drift toward a constant output. C (L2-SP): penalize ||w - w_prev||^2 — anchor to the PRE-UPDATE weights — instead of ||w||^2. The model stays near what it already knew rather than decaying toward zero (the original collapse driver). Trains the model in place. Deliberately excludes the model's baked-in l2(->0) kernel regularizers (`model.losses`) so only the L2-SP anchor acts during the increment. Returns the number of rows trained on."""
    rng = np.random.default_rng(seed)
    if replay_n and len(y_hist) > 0:
        idx   = rng.choice(len(y_hist), size=min(replay_n, len(y_hist)), replace=False)
        X_mix = np.concatenate([X_hist[idx], X_new], axis=0)
        y_mix = np.concatenate([np.asarray(y_hist)[idx], np.asarray(y_new)], axis=0)
    else:
        X_mix, y_mix = X_new, np.asarray(y_new)

    X_t = tf.constant(X_mix, dtype=tf.float32)
    y_t = tf.constant(y_mix.reshape(-1, 1), dtype=tf.float32)

    kernels = [v for v in model.trainable_variables if 'kernel' in v.name]
    anchors = [tf.constant(k.numpy()) for k in kernels]   # C: w_prev snapshot

    opt = tf.keras.optimizers.SGD(learning_rate=lr, momentum=momentum)
    bce = tf.keras.losses.BinaryCrossentropy()
    n, bs = int(X_t.shape[0]), 64
    for _ in range(epochs):
        perm = tf.random.shuffle(tf.range(n))
        for s in range(0, n, bs):
            b      = perm[s:s + bs]
            xb, yb = tf.gather(X_t, b), tf.gather(y_t, b)
            with tf.GradientTape() as tape:
                loss = bce(yb, model(xb, training=True))
                loss = loss + l2sp * tf.add_n(
                    [tf.reduce_sum(tf.square(k - a)) for k, a in zip(kernels, anchors)])
            grads = tape.gradient(loss, model.trainable_variables)
            opt.apply_gradients(zip(grads, model.trainable_variables))
    return n

nn_scaler = StandardScaler()
X_nn_all  = nn_scaler.fit_transform(X_raw)
nn_boundary = y.mean()

uses_s3 = NN_MODEL_PATH.startswith('s3://')

nn = None
_nn_from_scratch = False   # set when the persisted weights are unusable and we refit from zero
if uses_s3 and _s3_model_exists(NN_MODEL_PATH):
    print(f'\nLoading NN from {NN_MODEL_PATH}...')
    try:
        nn = _load_nn_from_s3(NN_MODEL_PATH, X_nn_all.shape[1])
        print('  Loaded NN weights (incremental state preserved)')
    except Exception as _load_err:
        print(f'  WARNING: could not load saved NN ({_load_err}) — will retrain from scratch')
        nn = None

if nn is not None:
    _NN_SEASON_START = date(TODAY.year, 5, 1)
    if YESTERDAY < _NN_SEASON_START:
        # Early-season freeze: pitcher RA and YRFI splits are unreliable before May 1 (tiny sample sizes create extreme feature values that are OOD vs training data). Also, the lambda saves RA as R/G vs training R/IP — wrong scale before this is resolved. Skip the increment entirely; save model unchanged so weights don't drift.
        print(f'  Skipping incremental NN update: {YESTERDAY} is pre-May-1 (early-season OOD window)')
        _save_nn_to_s3(nn, NN_MODEL_PATH)
        print(f'  NN saved to {NN_MODEL_PATH} (unchanged)')
    else:
        # Incremental train: A+B+C update (_incremental_update) on yesterday's S3 batch
        yesterday_path = (f's3://nrfi-store/data/{YESTERDAY.year}/'
                          f'{YESTERDAY.month}/{YESTERDAY.day}.txt')
        try:
            batch_df = load_data(yesterday_path)
            # Quality gate: skip the increment when the batch's key features are largely degenerate. The upstream stat source writes 0 OR NaN on failure; both get imputed to the league median downstream, which destroys feature variance and drifts the NN toward a constant output over successive days. isna() alone misses the imputed-to-zero case, so count nulls AND zero-sentinels across RA, WHIP, and OPS.
            _dq_cols  = [c for c in ['away_pitcher_ra', 'home_pitcher_ra',
                                     'away_whip', 'home_whip', 'away_ops', 'home_ops']
                         if c in batch_df.columns]
            _dq_vals  = batch_df[_dq_cols].apply(pd.to_numeric, errors='coerce')
            _dq_bad_rate = float((_dq_vals.isna() | (_dq_vals == 0)).values.mean())
            if _dq_bad_rate > 0.5:
                print(f'  WARNING: {_dq_bad_rate:.0%} of batch RA/WHIP/OPS values are missing or '
                      f'zero ({YESTERDAY}) — skipping increment to prevent feature collapse')
                _save_nn_to_s3(nn, NN_MODEL_PATH)
                print(f'  NN saved to {NN_MODEL_PATH} (unchanged)')
            else:
                for col in ['away_pitcher_ra', 'home_pitcher_ra', 'away_whip', 'home_whip',
                            'home_yrfi_pct', 'away_yrfi_pct']:
                    batch_df[col] = batch_df[col].replace(0, df[col].median())
                    batch_df[col] = batch_df[col].fillna(df[col].median())
                # Apply same preprocessing as training data (RA cap, whip cap)
                for col in ['away_pitcher_ra', 'home_pitcher_ra']:
                    batch_df[col] = batch_df[col].clip(upper=RA_CAP)
                feat_batch = make_features(batch_df)
                # drop rows with any NaN features to prevent weight explosion
                valid_mask = feat_batch.notna().all(axis=1)
                feat_batch = feat_batch[valid_mask]
                y_batch = batch_df['YRFI'].values[valid_mask]
                if len(feat_batch) == 0:
                    print(f'  WARNING: batch has no clean rows after NaN drop — skipping increment')
                else:
                    X_batch = nn_scaler.transform(feat_batch.values)
                    _n_upd  = _incremental_update(
                        nn, X_batch, y_batch, X_nn_all, y,
                        seed=int(YESTERDAY.strftime('%Y%m%d')))
                    print(f'  Incremental update (A+B+C): {len(feat_batch)} new games '
                          f'+ {min(NN_REPLAY_N, len(y))} replay = {_n_upd} rows, '
                          f'1-epoch SGD(lr={NN_ONLINE_LR}, mom={NN_ONLINE_MOMENTUM}) '
                          f'+ L2-SP(λ={NN_L2SP_LAMBDA}) from {YESTERDAY}')
                _save_nn_to_s3(nn, NN_MODEL_PATH)
                print(f'  NN saved to {NN_MODEL_PATH}')
        except Exception as ex:
            print(f'  WARNING: could not load yesterday batch ({ex}) — skipping increment')
            _save_nn_to_s3(nn, NN_MODEL_PATH)
            print(f'  NN saved to {NN_MODEL_PATH} (unchanged)')
else:
    print('\nNo saved NN found — training from scratch on full dataset...')
    _nn_from_scratch = True
    nn = _build_nn(X_nn_all.shape[1])
    es = tf.keras.callbacks.EarlyStopping(
        monitor='val_loss', patience=10, restore_best_weights=True, verbose=0
    )
    nn.fit(X_nn_all, y, epochs=50, batch_size=64,
           validation_split=0.1, shuffle=True, callbacks=[es], verbose=0)
    print(f'  Trained from scratch on {len(y)} games')
    if uses_s3:
        _save_nn_to_s3(nn, NN_MODEL_PATH)
        _save_nn_retrain_date(NN_MODEL_PATH, TODAY)
        print(f'  NN saved to {NN_MODEL_PATH}')

# Calibrate the direction boundary from the production model's actual output distribution. LR matches y.mean() by construction; the NN's sigmoid outputs can sit systematically above or below the base rate. Pred direction and the agreement check both use this single boundary so a marginal NN output is classified consistently.
nn_calibrated_boundary = float(nn.predict(X_nn_all, verbose=0).mean())
nn_meta   = {'boundary': nn_calibrated_boundary}

# NN confidence band: PERCENTILE of recent live outputs (rolling NN_POOL_DAYS window), targeting NN_COVERAGE_TARGET coverage on the distribution actually being scored. A fixed CV margin undershot deployment coverage badly (the live outputs are shifted/ compressed vs historical CV). If the rolling pool is too small (early season / no OUTPUT_BUCKET), fall back to a symmetric margin around the calibrated boundary.
_nn_retrained_on = TODAY if _nn_from_scratch else (_load_nn_retrain_date(NN_MODEL_PATH) if uses_s3 else None)
_nn_pool = _recent_nn_pool(TODAY, NN_POOL_DAYS, not_before=_nn_retrained_on)
_tail    = NN_COVERAGE_TARGET / 2.0
if _nn_retrained_on and not _nn_from_scratch:
    _days_since = (TODAY - _nn_retrained_on).days
    if _days_since < NN_POOL_DAYS:
        print(f'  NN pool floored at the {_nn_retrained_on} refit ({_days_since}d of history '
              f'available, not {NN_POOL_DAYS}) — pre-refit outputs came from a different model.')

if len(_nn_pool) >= NN_POOL_MIN:
    nn_low  = round(float(np.percentile(_nn_pool, _tail * 100)), 4)
    nn_high = round(float(np.percentile(_nn_pool, (1 - _tail) * 100)), 4)
    _cov_chk = float((_nn_pool < nn_low).mean() + (_nn_pool > nn_high).mean())
    print(f'NN band (percentile of {len(_nn_pool)} recent outputs):  <{nn_low} / >{nn_high}  '
          f'(cov {_cov_chk:.1%}, target {NN_COVERAGE_TARGET:.0%}, boundary {nn_calibrated_boundary:.4f})')
else:
    # Too little post-refit live history — fall back to the model's OWN training-set percentiles rather than a fixed margin, since same model lands far closer to the live band.
    _self = nn.predict(X_nn_all, verbose=0).ravel()
    nn_low  = round(float(np.percentile(_self, _tail * 100)), 4)
    nn_high = round(float(np.percentile(_self, (1 - _tail) * 100)), 4)
    print(f'NN band (percentile of the model\'s own {len(_self)} training outputs; only '
          f'{len(_nn_pool)} pooled live outputs < {NN_POOL_MIN}):  <{nn_low} / >{nn_high}  '
          f'(target {NN_COVERAGE_TARGET:.0%}, boundary {nn_calibrated_boundary:.4f})')

# ══════════════════════════════════════════════════════════════════════════════ PART 2 — CV THRESHOLD TUNING ══════════════════════════════════════════════════════════════════════════════
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

# Sweep a band half-width (margin); the band is boundary ± margin so that the confidence band shares ONE reference point with the prediction direction (cv_boundary == y.mean() here, since CV covers every row exactly once).
sweep   = np.round(np.arange(0.02, 0.131, 0.005), 3)
cv_rows = []
for mgn in sweep:
    acc, n, cov = confident_metrics(
        cv_probs, cv_y,
        round(cv_boundary - mgn, 4), round(cv_boundary + mgn, 4),
        cv_boundary)
    if acc is not None:
        cv_rows.append((mgn, acc, n, cov))

eligible  = [(mgn, a, n, c) for mgn, a, n, c in cv_rows if MIN_COVERAGE <= c <= MAX_COVERAGE]
best      = max(eligible, key=lambda r: edge_score(r[1], r[3])) if eligible else None
MARGIN    = best[0] if best else 0.045

if best:
    print(f'LR CV sweep: boundary {BOUNDARY:.4f} ± {MARGIN:.3f}  '
          f'(acc={best[1]:.1%}, in-sample cov={best[3]:.1%}, edge={edge_score(best[1], best[3]):.4f})')
else:
    print(f'LR CV sweep: boundary {BOUNDARY:.4f} ± {MARGIN:.3f}  (fallback)')

# The CV sweep measures coverage IN-SAMPLE, but the band is only ever applied to the LIVE output distribution, and the two diverged in both directions during 2026 (40% live coverage in July, 5% in August) — size the band on the live pool, keeping BOUNDARY put; the CV margin is the cold-pool fallback. See DECISIONS.md 2026-08-25.
_lr_pool = _recent_pool(TODAY, LR_POOL_DAYS, 'lr_prob_yrfi', not_before=LR_POOL_NOT_BEFORE)
if len(_lr_pool) >= LR_POOL_MIN:
    LOW  = round(float(np.percentile(_lr_pool, LR_NRFI_TAIL * 100)), 4)
    HIGH = round(float(np.percentile(_lr_pool, (1 - LR_YRFI_TAIL) * 100)), 4)
    _lr_cov = float((_lr_pool < LOW).mean() + (_lr_pool > HIGH).mean())
    print(f'LR band (percentile of {len(_lr_pool)} recent outputs):  <{LOW} / >{HIGH}  '
          f'(cov {_lr_cov:.1%}, target {LR_YRFI_TAIL + LR_NRFI_TAIL:.0%} = '
          f'{LR_YRFI_TAIL:.1%} YRFI / {LR_NRFI_TAIL:.1%} NRFI, boundary {BOUNDARY:.4f} unmoved)')
else:
    LOW  = round(BOUNDARY - MARGIN, 3)
    HIGH = round(BOUNDARY + MARGIN, 3)
    _lr_cov = None
    print(f'LR band (FALLBACK to CV margin; only {len(_lr_pool)} pooled outputs < {LR_POOL_MIN}):  '
          f'<{LOW} / >{HIGH}')
cw_metric('LRBandMargin', MARGIN,                unit='None')
cw_metric('LRCVAccuracy', best[1] if best else 0.0, unit='None')
if _lr_cov is not None:
    cw_metric('LRPoolCoverage', _lr_cov, unit='None')

# ══════════════════════════════════════════════════════════════════════════════ PART 3 — FETCH TODAY'S GAMES ══════════════════════════════════════════════════════════════════════════════
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
    'Athletics':'OAK',
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

# ══════════════════════════════════════════════════════════════════════════════ PART 4 — FETCH FEATURES ══════════════════════════════════════════════════════════════════════════════

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
    """Return (value, source) where source is one of: 'current', 'prior', 'overall', 'league'. Before May 1: prior-year split → current split → overall → league average. Current-year home/away splits are based on ~6 games before May; those tiny samples produce extreme values (0-100%) that land 4-9 std deviations outside the training distribution, making both models overconfident. After May 1: current split → prior-year split → overall → league average."""
    use_prior_first = TODAY < date(TODAY.year, 5, 1)
    ordered = (
        [('prior', split_prev), ('current', split_curr), ('overall', overall)]
        if use_prior_first else
        [('current', split_curr), ('prior', split_prev), ('overall', overall)]
    )
    for src_name, src in ordered:
        v = src.get(abbv)
        if v is not None:
            return v, src_name
    return fallback, 'league'

# ── 4b. Pitcher + team stats (MLB Stats API — 30/60-day rolling windows) ───── Fangraphs blocked by Cloudflare; MLB Stats API supports identical date ranges with no auth and no rate limiting.
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
    # Pitcher RA + WHIP: 60-day rolling window, starters only (gamesStarted > 0) RA = runs / inningsPitched — per-inning scale matches training data (~0.47 median)
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

# Individual batter OPS: 30-day rolling — matches training data window. Used for lineup-level OPS lookup; team average is the fallback.
BATTER_OPS      = {}   # cleaned_name -> OPS
BATTER_OPS_BY_ID = {}  # personId -> OPS (lineup match: the boxscore gives short names, not full)
TEAM_AVG_OPS    = {}   # abbv -> mean OPS of the team's top-4 hitters (30-day)

try:
    d30 = YESTERDAY - timedelta(days=29)
    bat_url = (
        f'https://statsapi.mlb.com/api/v1/stats'
        f'?stats=season&group=hitting&season={TODAY.year}'
        f'&startDate={d30}&endDate={YESTERDAY}'
        f'&playerPool=All&limit=5000'
        f'&fields=stats,splits,stat,ops,obp,slg,atBats,player,fullName,id,team,name'
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
        pid = split['player'].get('id')
        if pid is not None:
            BATTER_OPS_BY_ID[pid] = ops_f
        team_name = split.get('team', {}).get('name', '')
        abbv = MLB_TEAM_ABBV.get(team_name)
        if abbv:
            team_ops_lists.setdefault(abbv, []).append(ops_f)
    # Team OPS fallback (used when no lineup is posted yet) = mean of the team's top-4 hitters by OPS — a proxy for the 1-4 batting order the model actually trains on (lambda's get_lineup caps at the first 4 hitters). The OLD fallback was a whole-roster mean over everyone with >=10 AB (~0.67), which biased every team toward a weak offense vs the ~0.79 training scale — the -1.3sigma OPS drift the data-quality report flagged. NOTE: top-4-BY-OPS (~0.85) runs a touch above top-4-BY-ORDER (~0.79), since the 4 best hitters aren't always the 4 who bat 1-4; this is a fallback estimate only and is superseded whenever a real (today/yesterday) lineup is available.
    TEAM_AVG_OPS = {abbv: round(float(np.mean(sorted(vals, reverse=True)[:4])), 3)
                    for abbv, vals in team_ops_lists.items() if vals}
    print(f'  Loaded 30-day OPS for {len(BATTER_OPS)} batters ({len(TEAM_AVG_OPS)} teams)')
except Exception as ex:
    print(f'  WARNING: MLB Stats API batting fetch failed ({ex})')

LEAGUE_RA   = league_avg_ra
LEAGUE_WHIP = df['home_whip'].median()
LEAGUE_OPS  = df['home_ops'].median()

def get_pitcher_ra(name):
    """Return (ra, imputed). imputed=True when pitcher not found in 60-day MLB API window."""
    cleaned = unidecode(name)
    if cleaned in PITCHER_RA:
        return min(PITCHER_RA[cleaned], RA_CAP), False
    return LEAGUE_RA, True

def get_pitcher_whip(name):
    """Return (whip, imputed). imputed=True when pitcher not found in 60-day MLB API window."""
    cleaned = unidecode(name)
    if cleaned in PITCHER_WHIP:
        return PITCHER_WHIP[cleaned], False
    return LEAGUE_WHIP, True

def _lineup_ops(batter_ids):
    """Mean OPS of the top-4 batters (by personId) in the lineup — matches the training signal, which lambda's get_lineup builds from exactly the first 4 non-substitution hitters (`while hitters < 4`). The 1st inning is decided by the top of the order, so only those 4 are used. Matching by personId (not name) because the boxscore returns short names ("Alvarez, Y") that don't match the full-name OPS keys. None if fewer than 2 found."""
    vals = [BATTER_OPS_BY_ID[i] for i in batter_ids if i in BATTER_OPS_BY_ID][:4]
    return round(sum(vals) / len(vals), 3) if len(vals) >= 2 else None

def _fetch_lineup(game_id):
    """Return (away_ids, home_ids): personIds of the first 4 non-substitution hitters in each side's boxscore batting order. The old /api/v1/game/{id}/lineups endpoint was retired (404), which silently killed the lineup path and forced every game onto the team_avg fallback; the boxscore battingOrder is the live source and matches lambda's get_lineup. Empty lists when the order isn't posted yet (pre-game) → caller falls back to yesterday's lineup, then team_avg."""
    try:
        box = statsapi.boxscore_data(game_id)
    except Exception:
        return [], []
    def top4(side):
        batters = box.get(side + 'Batters', [])
        ids, i, n = [], 1, 0     # index 0 is the header row (matches lambda's i=1 start)
        while n < 4 and i < len(batters):
            b = batters[i]
            if isinstance(b, dict) and not b.get('substitution', True) and b.get('personId'):
                ids.append(b['personId']); n += 1
            i += 1
        return ids
    return top4('away'), top4('home')

# Build yesterday's team_id -> game_id map for lineup fallback. Used when today's lineup isn't posted yet (typical at 11 AM ET).
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
    """Return (away_ops, home_ops, away_src, home_src) using the best available lineup: 1. Today's announced lineup → source='lineup' 2. Yesterday's lineup for each side independently → source='yesterday' 3. 30-day team average OPS → source='team_avg' 4. League average → source='league'"""
    today_away, today_home = _fetch_lineup(game_id)
    away_ops = _lineup_ops(today_away)
    home_ops = _lineup_ops(today_home)
    away_src = 'lineup' if away_ops is not None else None
    home_src = 'lineup' if home_ops is not None else None

    # Fill in missing sides from yesterday's lineup
    if away_ops is None and away_team_id and away_team_id in YESTERDAY_GAME_BY_TEAM:
        yest_game_id = YESTERDAY_GAME_BY_TEAM[away_team_id]
        yest_away, yest_home = _fetch_lineup(yest_game_id)
        away_ops = _lineup_ops(yest_away) or _lineup_ops(yest_home)
        if away_ops is not None:
            away_src = 'yesterday'

    if home_ops is None and home_team_id and home_team_id in YESTERDAY_GAME_BY_TEAM:
        yest_game_id = YESTERDAY_GAME_BY_TEAM[home_team_id]
        yest_away, yest_home = _fetch_lineup(yest_game_id)
        home_ops = _lineup_ops(yest_home) or _lineup_ops(yest_away)
        if home_ops is not None:
            home_src = 'yesterday'

    # Final fallback: 30-day team average, then league average
    if away_ops is None:
        team_val = TEAM_AVG_OPS.get(away_abbv)
        if team_val is not None:
            away_ops, away_src = team_val, 'team_avg'
        else:
            away_ops, away_src = LEAGUE_OPS, 'league'
    if home_ops is None:
        team_val = TEAM_AVG_OPS.get(home_abbv)
        if team_val is not None:
            home_ops, home_src = team_val, 'team_avg'
        else:
            home_ops, home_src = LEAGUE_OPS, 'league'

    return away_ops, home_ops, away_src, home_src

# ── 4c. Weather (Open-Meteo) ──────────────────────────────────────────────────
print('Fetching weather from Open-Meteo...')
# Keyed by (stadium, first pitch) rather than stadium alone: temperature is now read at the game's
# start hour, so the two halves of a doubleheader legitimately get different values.
WEATHER_CACHE = {}
for g in games:
    key = (g['home_abbv'], g.get('game_time', ''))
    if key not in WEATHER_CACHE:
        WEATHER_CACHE[key] = fetch_weather(g['home_abbv'], str(TODAY), g.get('game_time'))
live_count = sum(1 for v in WEATHER_CACHE.values() if v[2] == 'api')
dome_count = sum(1 for v in WEATHER_CACHE.values() if v[2] == 'dome')
print(f'  Fetched first-pitch weather for {live_count}/{len(WEATHER_CACHE)} games '
      f'({dome_count} climate-controlled, {len(WEATHER_CACHE) - live_count - dome_count} defaulted)')

# ── 4d. Odds (BettingPros) ────────────────────────────────────────────────────
print('Fetching odds (The Odds API / Bovada fallback)...')
GAME_ODDS = fetch_odds()
using_real_odds = bool(GAME_ODDS)
if using_real_odds:
    print(f'  Loaded odds for {len(GAME_ODDS)} games')
else:
    print(f'  WARNING: No odds available — EV will not be computed')

def get_odds(matchup_key):
    return GAME_ODDS.get(matchup_key)  # None if not available

def _implied(american):
    """American odds -> implied probability (vig included)."""
    a = float(american)
    return 100.0 / (a + 100.0) if a > 0 else abs(a) / (abs(a) + 100.0)

def _devig(nrfi_odds, yrfi_odds):
    """Two-way de-vigged P(YRFI) from an American price pair; nan if either side is missing."""
    if nrfi_odds is None or yrfi_odds is None:
        return float('nan')
    try:
        n, y = _implied(nrfi_odds), _implied(yrfi_odds)
    except Exception:
        return float('nan')
    return float(y / (y + n)) if (y + n) > 0 else float('nan')

def _blend_weight_from_history():
    """Market weight from each source's trailing realised edge (AUC - 0.5), over the last BLEND_LOOKBACK_DAYS of graded rows in results/results.csv that have both a stored lr_prob_yrfi and a priced market. A source decayed to chance contributes ~0 edge and is weighted down on its own — the LR earns its weight in April and loses it by August with no constant to edit. Returns (w, lr_auc, mkt_auc, n)."""
    bucket = os.environ.get('NRFI_OUTPUT_BUCKET')
    if not bucket:
        return BLEND_DEFAULT_W, None, None, 0
    import boto3, io as _io
    try:
        obj = boto3.client('s3').get_object(Bucket=bucket, Key='results/results.csv')
        h = pd.read_csv(_io.BytesIO(obj['Body'].read()))
    except Exception:
        return BLEND_DEFAULT_W, None, None, 0
    try:
        h['date'] = pd.to_datetime(h['date'], errors='coerce')
        h = h[h['date'] >= pd.Timestamp(TODAY) - pd.Timedelta(days=BLEND_LOOKBACK_DAYS)]
        h = h[h['actual_yrfi'].notna() & h['lr_prob_yrfi'].notna()
              & h['nrfi_odds'].notna() & h['yrfi_odds'].notna()]
        if len(h) < BLEND_MIN_GRADED:
            return BLEND_DEFAULT_W, None, None, len(h)
        mkt = np.array([_devig(n, y) for n, y in zip(h['nrfi_odds'], h['yrfi_odds'])])
        lrp = pd.to_numeric(h['lr_prob_yrfi'], errors='coerce').values
        act = pd.to_numeric(h['actual_yrfi'], errors='coerce').values
        ok  = ~(np.isnan(mkt) | np.isnan(lrp) | np.isnan(act))
        if ok.sum() < BLEND_MIN_GRADED or len(np.unique(act[ok])) < 2:
            return BLEND_DEFAULT_W, None, None, int(ok.sum())
        from sklearn.metrics import roc_auc_score
        a_lr, a_mk = roc_auc_score(act[ok], lrp[ok]), roc_auc_score(act[ok], mkt[ok])
        e_lr, e_mk = max(a_lr - 0.5, 1e-4), max(a_mk - 0.5, 1e-4)
        w = min(max(e_mk / (e_lr + e_mk), BLEND_W_MIN), BLEND_W_MAX)
        return float(w), float(a_lr), float(a_mk), int(ok.sum())
    except Exception:
        return BLEND_DEFAULT_W, None, None, 0

def _recent_blend_pool(end_date, n_days, w, not_before=None):
    """Reconstruct the blended-probability pool from recent game_logs. The blend isn't persisted historically, but every row carries lr_prob_yrfi and both prices, so it can be rebuilt at TODAY's weight — which is what the band must be calibrated on."""
    bucket = os.environ.get('NRFI_OUTPUT_BUCKET')
    if not bucket:
        return np.array([])
    import boto3, io as _io
    s3 = boto3.client('s3')
    out = []
    for i in range(1, n_days + 1):
        d = end_date - timedelta(days=i)
        if not_before is not None and d < not_before:
            continue
        try:
            obj = s3.get_object(Bucket=bucket, Key=f'game_log/{d.year}/{d.isoformat()}.csv')
            gl  = pd.read_csv(_io.BytesIO(obj['Body'].read()))
            if not {'lr_prob_yrfi', 'nrfi_odds', 'yrfi_odds'}.issubset(gl.columns):
                continue
            lrp = pd.to_numeric(gl['lr_prob_yrfi'], errors='coerce').values
            mk  = np.array([_devig(n, y) for n, y in zip(gl['nrfi_odds'], gl['yrfi_odds'])])
            bl  = np.where(np.isnan(mk), lrp, (1 - w) * lrp + w * mk)
            out.append(bl[~np.isnan(bl)])
        except Exception:
            continue
    return np.concatenate(out) if out else np.array([])

# ── 4e. NRFI juice tracking ───────────────────────────────────────────────────
# Books shade NRFI toward the public side, so the two sides do not carry symmetric vig — that asymmetry is the structural edge the YRFI-heavy tilt rides, so it is measured daily rather than assumed; `NrfiJuiceAsymmetry` = mean NRFI implied - mean YRFI implied in probability points, positive means NRFI is the pricier side — if it decays toward 0, revisit LR_NRFI_TAIL. See DECISIONS.md 2026-08-25.
if using_real_odds:
    _imp_n = [_implied(o[0]) for o in GAME_ODDS.values() if o and o[0] is not None]
    _imp_y = [_implied(o[1]) for o in GAME_ODDS.values() if o and o[1] is not None]
    if _imp_n and _imp_y:
        _mn, _my = float(np.mean(_imp_n)), float(np.mean(_imp_y))
        _vig     = _mn + _my - 1.0
        _asym    = _mn - _my
        print(f'  Juice: NRFI implied {_mn:.4f} / YRFI implied {_my:.4f}  '
              f'(vig {_vig:+.2%}, NRFI dearer by {_asym * 100:+.2f}pp)')
        cw_metric('NrfiImpliedProb',      _mn,   unit='None')
        cw_metric('YrfiImpliedProb',      _my,   unit='None')
        cw_metric('TwoWayVig',            _vig,  unit='None')
        cw_metric('NrfiJuiceAsymmetry',   _asym, unit='None')

# ══════════════════════════════════════════════════════════════════════════════ PART 5 — BUILD FEATURE ROWS FOR TODAY'S GAMES ══════════════════════════════════════════════════════════════════════════════
print('\nBuilding feature rows...')
rows = []
for g in games:
    home = g['home_abbv']
    away = g['away_abbv']
    hp   = g['home_pitcher']
    ap   = g['away_pitcher']

    home_ra,   home_ra_imp   = get_pitcher_ra(hp)
    home_whip, home_whip_imp = get_pitcher_whip(hp)
    away_ra,   away_ra_imp   = get_pitcher_ra(ap)
    away_whip, away_whip_imp = get_pitcher_whip(ap)
    away_ops, home_ops, away_ops_src, home_ops_src = fetch_game_ops(
        g['game_id'], away, home, g.get('away_id'), g.get('home_id'))
    park            = PARK_FACTORS.get(home, 100)
    temp, rain, weather_src = WEATHER_CACHE.get((home, g.get('game_time', '')), (65, 0, 'default'))

    home_yrfi, home_yrfi_src = get_yrfi(home, yrfi_home, yrfi_home_prev, yrfi_overall)
    away_yrfi, away_yrfi_src = get_yrfi(away, yrfi_away, yrfi_away_prev, yrfi_overall)

    # Count imputed features: pitcher missing from API, OPS not from any lineup, YRFI/weather defaulted
    _dq_n_imputed = sum([
        home_ra_imp, away_ra_imp,
        home_whip_imp, away_whip_imp,
        home_ops_src in ('team_avg', 'league'),
        away_ops_src in ('team_avg', 'league'),
        home_yrfi_src == 'league',
        away_yrfi_src == 'league',
        weather_src == 'default',
    ])

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
        # data quality flags (prefix _ so they don't enter model features)
        '_dq_home_ra_imp':   home_ra_imp,
        '_dq_away_ra_imp':   away_ra_imp,
        '_dq_home_whip_imp': home_whip_imp,
        '_dq_away_whip_imp': away_whip_imp,
        '_dq_home_ops_src':  home_ops_src,
        '_dq_away_ops_src':  away_ops_src,
        '_dq_home_yrfi_src': home_yrfi_src,
        '_dq_away_yrfi_src': away_yrfi_src,
        '_dq_weather_src':   weather_src,
        '_dq_n_imputed':     _dq_n_imputed,
    })

today_df = pd.DataFrame(rows)


def print_data_quality_report(rows_list):
    """Print a per-feature imputation summary for today's games."""
    n = len(rows_list)
    print('\n' + '=' * 60)
    print(f'DATA QUALITY — {TODAY}  ({n} games)')
    print('=' * 60)
    if n == 0:
        print('  No games.')
        return

    def _pct(cnt): return f'{cnt}/{n} ({cnt/n:.0%})'

    # Pitcher RA / WHIP
    for label, col in [
        ('Home starter RA',   '_dq_home_ra_imp'),
        ('Away starter RA',   '_dq_away_ra_imp'),
        ('Home starter WHIP', '_dq_home_whip_imp'),
        ('Away starter WHIP', '_dq_away_whip_imp'),
    ]:
        cnt = sum(1 for r in rows_list if r.get(col, False))
        flag = '  *** TBD starters' if cnt == n else ''
        print(f'  {label:<22} imputed: {_pct(cnt)}{flag}')

    # OPS source breakdown
    for label, col in [('Away OPS', '_dq_away_ops_src'), ('Home OPS', '_dq_home_ops_src')]:
        counts = {}
        for r in rows_list:
            src = r.get(col, 'unknown')
            counts[src] = counts.get(src, 0) + 1
        src_str = '  '.join(f'{k}={v}' for k, v in
                            sorted(counts.items(), key=lambda x: ['lineup','yesterday','team_avg','league','unknown'].index(x[0])
                                   if x[0] in ['lineup','yesterday','team_avg','league','unknown'] else 99))
        print(f'  {label:<22} {src_str}')

    # YRFI% source breakdown
    for label, col in [('Home YRFI%', '_dq_home_yrfi_src'), ('Away YRFI%', '_dq_away_yrfi_src')]:
        counts = {}
        for r in rows_list:
            src = r.get(col, 'unknown')
            counts[src] = counts.get(src, 0) + 1
        src_str = '  '.join(f'{k}={v}' for k, v in
                            sorted(counts.items(), key=lambda x: ['current','prior','overall','league','unknown'].index(x[0])
                                   if x[0] in ['current','prior','overall','league','unknown'] else 99))
        print(f'  {label:<22} {src_str}')

    # Weather
    api_cnt = sum(1 for r in rows_list if r.get('_dq_weather_src') == 'api')
    print(f'  {"Weather":<22} api={api_cnt}  default={n - api_cnt}')

    # High-imputation games
    high_imp = [r['matchup'] for r in rows_list if r.get('_dq_n_imputed', 0) >= 3]
    if high_imp:
        print(f'\n  *** High-imputation (>=3 features): {", ".join(high_imp)}')
        print(f'      Predictions for these games lean heavily on league averages.')
    print()


print_data_quality_report(rows)

# ══════════════════════════════════════════════════════════════════════════════ PART 6 — APPLY MODELS ══════════════════════════════════════════════════════════════════════════════
X_today    = scaler.transform(today_df[FEATURES].values)
X_today_nn = nn_scaler.transform(today_df[FEATURES].values)

nn_boundary_today = nn_meta['boundary']

# LR — direction set by LR's own boundary (matches the band center).
lr_probs = lr.predict_proba(X_today)[:, 1]
today_df['lr_prob_yrfi'] = lr_probs
today_df['lr_prob_nrfi'] = 1 - lr_probs
today_df['lr_pred']      = np.where(lr_probs > BOUNDARY, 'YRFI', 'NRFI')
today_df['lr_conf']      = np.where(lr_probs > BOUNDARY, lr_probs, 1 - lr_probs)

# NN — direction set by the NN's calibrated boundary (matches the band center).
nn_probs = nn.predict(X_today_nn, verbose=0).flatten()
today_df['nn_prob_yrfi'] = nn_probs
today_df['nn_prob_nrfi'] = 1 - nn_probs
today_df['nn_pred']      = np.where(nn_probs > nn_boundary_today, 'YRFI', 'NRFI')
today_df['nn_conf']      = np.where(nn_probs > nn_boundary_today, nn_probs, 1 - nn_probs)

# Each model's pick is recommended INDEPENDENTLY: a model is "confident" when its probability falls outside its own boundary-centered band. Cross-model agreement is NOT required for an individual pick — only for the consensus tag.
today_df['lr_confident'] = (lr_probs < LOW) | (lr_probs > HIGH)
today_df['nn_confident'] = (nn_probs < nn_low) | (nn_probs > nn_high)

# Consensus = both models confident AND agreeing on direction. Each model's direction is measured against its OWN boundary, identical to how that model's prediction and band are defined (one reference point per model).
_models_agree_direction = (lr_probs > BOUNDARY) == (nn_probs > nn_boundary_today)
today_df['consensus']    = today_df['lr_confident'] & today_df['nn_confident'] & _models_agree_direction

# ── BLEND — LR shrunk toward the de-vigged market, scored as its own pick set ──
# Nothing is fit here — it is the LR with a market prior; its band is calibrated like the others (percentile of its own recent outputs, rebuilt at today's weight) and uses the same tail split, so its coverage and tilt are directly comparable to the LR's.
blend_w, _b_lr_auc, _b_mk_auc, _b_n = (_blend_weight_from_history() if BLEND_ENABLED
                                       else (0.0, None, None, 0))
market_probs = np.array([_devig(*(get_odds(mu) or (None, None))) for mu in today_df['matchup']])
blend_probs_today = np.where(np.isnan(market_probs), lr_probs,
                             (1 - blend_w) * lr_probs + blend_w * market_probs)
today_df['market_prob_yrfi'] = market_probs
today_df['blend_prob_yrfi']  = blend_probs_today
today_df['blend_prob_nrfi']  = 1 - blend_probs_today

if BLEND_ENABLED:
    if _b_lr_auc is not None:
        print(f'\nBlend weight: {blend_w:.2f} on market  (trailing {BLEND_LOOKBACK_DAYS}d, n={_b_n}: '
              f'LR AUC {_b_lr_auc:.4f}, market AUC {_b_mk_auc:.4f})')
    else:
        print(f'\nBlend weight: {blend_w:.2f} on market  (default — only {_b_n} graded rows '
              f'< {BLEND_MIN_GRADED})')
    _bl_pool = _recent_blend_pool(TODAY, LR_POOL_DAYS, blend_w, not_before=LR_POOL_NOT_BEFORE)
    if len(_bl_pool) >= LR_POOL_MIN:
        blend_low  = round(float(np.percentile(_bl_pool, LR_NRFI_TAIL * 100)), 4)
        blend_high = round(float(np.percentile(_bl_pool, (1 - LR_YRFI_TAIL) * 100)), 4)
        _bl_cov = float((_bl_pool < blend_low).mean() + (_bl_pool > blend_high).mean())
        print(f'BLEND band (percentile of {len(_bl_pool)} rebuilt outputs):  '
              f'<{blend_low} / >{blend_high}  (cov {_bl_cov:.1%})')
    else:
        blend_low, blend_high = LOW, HIGH
        print(f'BLEND band (FALLBACK to LR band; only {len(_bl_pool)} pooled < {LR_POOL_MIN}):  '
              f'<{blend_low} / >{blend_high}')
    # The blend mixes an LR centered on BOUNDARY with a market centered on ~0.5, so its own centre moves with the weight — split it at the same mix rather than reusing BOUNDARY.
    blend_boundary = (1 - blend_w) * BOUNDARY + blend_w * 0.5
else:
    blend_low, blend_high, blend_boundary = LOW, HIGH, BOUNDARY

today_df['blend_pred']      = np.where(blend_probs_today > blend_boundary, 'YRFI', 'NRFI')
today_df['blend_conf']      = np.where(blend_probs_today > blend_boundary,
                                       blend_probs_today, 1 - blend_probs_today)
today_df['blend_confident'] = (((blend_probs_today < blend_low) | (blend_probs_today > blend_high))
                               & BLEND_ENABLED)

cw_metric('LRPickCount',        int(today_df['lr_confident'].sum()))
cw_metric('NNPickCount',        int(today_df['nn_confident'].sum()))
cw_metric('BlendPickCount',     int(today_df['blend_confident'].sum()))
cw_metric('BlendWeight',        blend_w, unit='None')
cw_metric('ConsensusPickCount', int(today_df['consensus'].sum()))

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
today_df['blend_ev'] = compute_ev(blend_probs_today, today_df['blend_pred'], today_df['matchup'])

# ══════════════════════════════════════════════════════════════════════════════ PART 7 — OUTPUT ══════════════════════════════════════════════════════════════════════════════
odds_note = '' if using_real_odds else '  (no odds — EV unavailable)'
header_note = (f'LR: <{LOW} / >{HIGH}  '
               f'NN: <{nn_low} / >{nn_high}  '
               + (f'BLEND(w={blend_w:.2f}): <{blend_low} / >{blend_high}  ' if BLEND_ENABLED else '')
               + f'[1u = ${UNIT}]{odds_note}')

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
blend_payload = print_picks_section(
    today_df['blend_confident'], 'BLEND',
    'blend_pred', 'blend_conf', 'blend_ev', 'blend_prob_nrfi', 'blend_prob_yrfi',
) if BLEND_ENABLED else []

# Drop any NN pick that contradicts LR's direction for the same game. Without this, the email merges them into "NN,LR → YRFI" even when NN said NRFI.
_lr_directions = {p['matchup']: p['prediction'] for p in lr_payload}
_nn_filtered   = [p for p in nn_payload
                  if p['matchup'] not in _lr_directions
                  or p['prediction'] == _lr_directions[p['matchup']]]
picks_payload = lr_payload + _nn_filtered

deliver_picks(
    picks_payload, str(TODAY), LOW, HIGH,
    best[1] if best else 0.0,
    best[3] if best else 0.0,
)

# ── Save full game log (all games, both models) ───────────────────────────────
def save_game_log(df, date_str, lr_threshold_low, lr_threshold_high,
                  nn_threshold_low, nn_threshold_high,
                  lr_boundary, nn_boundary, cv_acc, cv_cov):
    """Save a detailed per-game snapshot to S3 for later result grading. Includes raw model outputs, features, odds, and thresholds used."""
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
            'lr_threshold_low':   round(lr_threshold_low, 3),
            'lr_threshold_high':  round(lr_threshold_high, 3),
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
            # BLEND (LR shrunk toward the de-vigged market — third pick set, 2026-08-26)
            'market_prob_yrfi':   (round(r['market_prob_yrfi'], 4)
                                   if pd.notna(r.get('market_prob_yrfi')) else None),
            'blend_prob_yrfi':    round(r['blend_prob_yrfi'], 4),
            'blend_prob_nrfi':    round(r['blend_prob_nrfi'], 4),
            'blend_pred':         r['blend_pred'],
            'blend_conf':         round(r['blend_conf'], 4),
            'blend_confident':    bool(r['blend_confident']),
            'blend_ev':           r['blend_ev'],
            'blend_threshold_low':  round(blend_low, 4),
            'blend_threshold_high': round(blend_high, 4),
            'blend_boundary':     round(blend_boundary, 4),
            'blend_weight':       round(blend_w, 4),
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
            'away_ra':            r['_away_ra'],
            'away_whip':          r['_away_whip'],
            'home_ops':           r['_home_ops'],
            'away_ops':           r['_away_ops'],
            'park_factor':        r['_park'],
            'temp':               r['_temp'],
            'rain':               int(r['_rain']),
            # Data quality flags
            'dq_home_ra_imp':     bool(r.get('_dq_home_ra_imp', False)),
            'dq_away_ra_imp':     bool(r.get('_dq_away_ra_imp', False)),
            'dq_home_whip_imp':   bool(r.get('_dq_home_whip_imp', False)),
            'dq_away_whip_imp':   bool(r.get('_dq_away_whip_imp', False)),
            'dq_home_ops_src':    r.get('_dq_home_ops_src', 'unknown'),
            'dq_away_ops_src':    r.get('_dq_away_ops_src', 'unknown'),
            'dq_home_yrfi_src':   r.get('_dq_home_yrfi_src', 'unknown'),
            'dq_away_yrfi_src':   r.get('_dq_away_yrfi_src', 'unknown'),
            'dq_weather_src':     r.get('_dq_weather_src', 'unknown'),
            'dq_n_imputed':       int(r.get('_dq_n_imputed', 0)),
            # Actuals (filled in next day by grade_yesterday)
            'actual_yrfi':        None,
            'lr_correct':         None,
            'nn_correct':         None,
            'blend_correct':      None,
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
    today_df, str(TODAY), LOW, HIGH,
    nn_low, nn_high,
    BOUNDARY, nn_meta['boundary'],
    best[1] if best else 0.0,
    best[3] if best else 0.0,
)

# ── Load today's game log + picks from S3 if available ─────────────────────── Prefer SageMaker's run (authoritative lineups/features) over local recompute
try:
    import boto3, io as _io
    _s3 = boto3.client('s3')
    _gl_key = f'game_log/{TODAY.year}/{TODAY.isoformat()}.csv'
    _gl_obj = _s3.get_object(Bucket='nrfi-store', Key=_gl_key)
    today_df = pd.read_csv(_io.BytesIO(_gl_obj['Body'].read()))
    print(f'  Loaded today\'s game log from S3 ({len(today_df)} games)')
    # Also load picks JSON so picks_payload reflects SageMaker's picks
    _pk_key = f'picks/{TODAY.year}/{TODAY.isoformat()}.json'
    try:
        _pk_obj = _s3.get_object(Bucket='nrfi-store', Key=_pk_key)
        picks_payload = json.loads(_pk_obj['Body'].read()).get('picks', picks_payload)
        print(f'  Loaded {len(picks_payload)} picks from S3')
    except Exception:
        pass  # keep locally computed picks_payload
except Exception as _ex:
    print(f'  Using locally computed game data (S3 unavailable: {_ex})')

# ── SES email notification ──────────────────────────────────────────────────── Build email subject — include yesterday's record if available
_yest_summary = ''
if yesterday_log_df is not None and not yesterday_log_df.empty:
    _conf = yesterday_log_df[yesterday_log_df['lr_confident'] | yesterday_log_df['nn_confident']]
    _graded = _conf[_conf['lr_correct'].notna() | _conf['nn_correct'].notna()]
    if not _graded.empty:
        # Count all picks (LR or NN); use each game's relevant model result
        _corrects, _pl_vals = [], []
        for _, _r in _graded.iterrows():
            if _r.get('lr_confident') and pd.notna(_r.get('lr_correct')):
                _corrects.append(int(_r['lr_correct']))
                _v = compute_pl(_r['lr_correct'], _r['lr_pred'], _r.get('nrfi_odds'), _r.get('yrfi_odds'))
            elif _r.get('nn_confident') and pd.notna(_r.get('nn_correct')):
                _corrects.append(int(_r['nn_correct']))
                _v = compute_pl(_r['nn_correct'], _r['nn_pred'], _r.get('nrfi_odds'), _r.get('yrfi_odds'))
            else:
                _v = None
            if _v is not None:
                _pl_vals.append(_v)
        _w = sum(_corrects)
        _l = len(_corrects) - _w
        _pl   = sum(_pl_vals)
        _sign = '+' if _pl >= 0 else ''
        _yest_summary = f' | Yesterday {_w}-{_l} ({_sign}${_pl:.2f})'

_n_picks = len([p for p in picks_payload if not p.get('consensus')])
_n_cons  = len([p for p in picks_payload if p.get('consensus')])
_picks_summary = f'{_n_picks} pick{"s" if _n_picks != 1 else ""}'
if _n_cons:
    _picks_summary += f', {_n_cons} consensus'
_session_label = {'afternoon': ' (Afternoon)', 'evening': ' (Evening)', 'all': ''}.get(SESSION, '')
_date_label = (TODAY.strftime('%b') + ' ' + str(TODAY.day)) if hasattr(TODAY, 'strftime') else str(TODAY)
email_subject = f'NRFI {_date_label}{_session_label} | {_picks_summary}{_yest_summary}'

email_html = build_email_html(
    date_str=str(TODAY),
    picks_rows=picks_payload,
    yesterday_rows=yesterday_log_df,
    ytd_df=ytd_df,
    today_df_all=today_df,
    lr_band=(LOW, HIGH),
    nn_band=(nn_low, nn_high),
    cv_acc=best[1] if best else 0.0,
    cv_cov=best[3] if best else 0.0,
    yesterday=YESTERDAY,
    today=TODAY,
    unit=UNIT,
    get_odds_fn=get_odds,
)

# Build 7-day confidence band timeline chart for email. Load the past 6 game_log CSVs directly from S3 — these are written unconditionally by save_game_log() and exist even when Lambda results are absent (which would cause grade_yesterday() to return ytd_df=None, leaving the chart with only 1 day of data).
_chart_bytes = None
try:
    _hist_dfs = []
    _s3_bucket_chart = os.environ.get('NRFI_OUTPUT_BUCKET')
    _loaded_from_s3 = False
    if _s3_bucket_chart:
        try:
            import boto3 as _boto3c, io as _cioc
            _cs3 = _boto3c.client('s3')
            for _d_offset in range(1, 7):
                _d = TODAY - timedelta(days=_d_offset)
                _gl_key = f'game_log/{_d.year}/{_d.isoformat()}.csv'
                try:
                    _obj = _cs3.get_object(Bucket=_s3_bucket_chart, Key=_gl_key)
                    _ddf = pd.read_csv(_cioc.BytesIO(_obj['Body'].read()))
                    if not _ddf.empty:
                        _ddf = _ddf.rename(columns={'date': 'game_date'})
                        _hist_dfs.append(_ddf)
                        _loaded_from_s3 = True
                except Exception:
                    pass  # day missing or no data — skip
        except Exception as _s3_chart_ex:
            print(f'  WARNING: chart history S3 load failed ({_s3_chart_ex})')
    if not _loaded_from_s3 and ytd_df is not None and not ytd_df.empty:
        _cutoff = (TODAY - timedelta(days=6)).isoformat()
        _past = ytd_df[ytd_df['date'] >= _cutoff].copy()
        _past = _past.rename(columns={'date': 'game_date'})
        _hist_dfs.append(_past)
    if today_df is not None and not today_df.empty:
        _today_chart = today_df.copy()
        _today_chart['game_date'] = str(TODAY)
        if 'lr_threshold_low' not in _today_chart.columns:
            _today_chart['lr_threshold_low']  = LOW
            _today_chart['lr_threshold_high'] = HIGH
        if 'nn_threshold_low' not in _today_chart.columns:
            _today_chart['nn_threshold_low']  = nn_low
            _today_chart['nn_threshold_high'] = nn_high
        _hist_dfs.append(_today_chart)
    _history_combined = pd.concat(_hist_dfs, ignore_index=True) if _hist_dfs else pd.DataFrame()

    # game_log CSVs have lr_correct=None; merge actuals from ytd_df so past dots show green/red.
    if ytd_df is not None and not ytd_df.empty and not _history_combined.empty:
        _ytd_outcomes = (ytd_df[['date', 'matchup', 'lr_correct', 'nn_correct']]
                         .rename(columns={'date': 'game_date'}))
        for _col in ['lr_correct', 'nn_correct']:
            if _col in _history_combined.columns:
                _history_combined = _history_combined.drop(columns=[_col])
        _history_combined = _history_combined.merge(
            _ytd_outcomes, on=['game_date', 'matchup'], how='left'
        )

    _chart_bytes = build_threshold_timeline(_history_combined)
except Exception as _chart_ex:
    print(f'  WARNING: chart generation failed ({_chart_ex})')

send_email(email_html, email_subject, str(TODAY), chart_bytes=_chart_bytes)
