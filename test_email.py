"""
test_email.py — end-to-end email test using real S3 data.
Pulls results.csv (7-day history + thresholds) and today's game log,
builds the full email HTML, and sends via SES.

Run from the project root: python test_email.py
"""

import io, json, os, sys
import pandas as pd
from datetime import date, timedelta
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

_env = Path(__file__).parent / '.env'
if _env.exists():
    for _line in _env.read_text().splitlines():
        _line = _line.strip()
        if _line and not _line.startswith('#') and '=' in _line:
            _k, _v = _line.split('=', 1)
            os.environ.setdefault(_k.strip(), _v.strip())

from utils.email_charts import build_threshold_timeline
from utils.email_html   import build_email_html, send_email

import subprocess, boto3

# Mirror daily_picks.py credential chain (handles SSO/Identity Center locally)
try:
    _raw = subprocess.check_output(
        ['aws', 'configure', 'export-credentials', '--format', 'env-no-export'],
        text=True, stderr=subprocess.DEVNULL,
    )
    for _line in _raw.splitlines():
        if '=' in _line:
            _k, _v = _line.split('=', 1)
            os.environ.setdefault(_k.strip(), _v.strip())
except Exception:
    pass

s3        = boto3.client('s3')
BUCKET    = os.environ.get('NRFI_OUTPUT_BUCKET', 'nrfi-store')
TODAY     = date.today()
YESTERDAY = TODAY - timedelta(days=1)
UNIT      = int(os.environ.get('NRFI_UNIT', 10))

def s3_csv(key):
    try:
        obj = s3.get_object(Bucket=BUCKET, Key=key)
        return pd.read_csv(io.BytesIO(obj['Body'].read()))
    except Exception as ex:
        print(f'  S3 miss: {key} ({ex})')
        return None

# ── Load results.csv (season history with per-day thresholds + outcomes) ──────
print('Loading results.csv ...')
results_df = s3_csv('results/results.csv')
if results_df is None or results_df.empty:
    print('ERROR: results.csv not found — cannot build chart or yesterday section.')
    results_df = pd.DataFrame()

cutoff   = (TODAY - timedelta(days=6)).isoformat()
ytd_df   = results_df if not results_df.empty else pd.DataFrame()
hist_df  = results_df[results_df['date'] >= cutoff].copy() if not results_df.empty else pd.DataFrame()
yest_df  = results_df[results_df['date'] == YESTERDAY.isoformat()].copy() if not results_df.empty else pd.DataFrame()

# ── Load today's game log ─────────────────────────────────────────────────────
print(f'Loading today\'s game log ({TODAY}) ...')
today_key = f'game_log/{TODAY.year}/{TODAY.isoformat()}.csv'
today_df  = s3_csv(today_key)
if today_df is not None:
    print(f'  {len(today_df)} games loaded')
else:
    print('  No game log for today yet — chart will show history only')
    today_df = pd.DataFrame()

# ── Load today's picks JSON (for picks_rows) ──────────────────────────────────
picks_rows = []
try:
    pk_key = f'picks/{TODAY.year}/{TODAY.isoformat()}.json'
    pk_obj = s3.get_object(Bucket=BUCKET, Key=pk_key)
    picks_rows = json.loads(pk_obj['Body'].read()).get('picks', [])
    print(f'  {len(picks_rows)} picks loaded from picks JSON')
except Exception:
    print('  No picks JSON for today — picks section will be empty')

# ── Build chart history (ytd past days + today) ───────────────────────────────
chart_dfs = []
if not hist_df.empty:
    past = hist_df.copy()
    past['game_date'] = past['date']
    chart_dfs.append(past)
if not today_df.empty:
    td = today_df.copy()
    td['game_date'] = str(TODAY)
    if 'lr_threshold_low' not in td.columns:
        td['lr_threshold_low']  = 0.450
        td['lr_threshold_high'] = 0.550
    if 'nn_threshold_low' not in td.columns:
        td['nn_threshold_low']  = 0.450
        td['nn_threshold_high'] = 0.550
    chart_dfs.append(td)

history_combined = pd.concat(chart_dfs, ignore_index=True) if chart_dfs else pd.DataFrame()
chart_bytes = build_threshold_timeline(history_combined)
print(f'Chart: {len(chart_bytes) if chart_bytes else 0} bytes')

# ── Pull today's threshold from game log (most recent cv_* columns) ───────────
lr_threshold = 0.550; nn_threshold = 0.550
cv_acc = 0.0;         cv_cov = 0.0
if not today_df.empty and 'lr_threshold_high' in today_df.columns:
    lr_threshold = float(today_df['lr_threshold_high'].iloc[0])
    nn_threshold = float(today_df.get('nn_threshold_high', today_df['lr_threshold_high']).iloc[0])
if not today_df.empty and 'cv_acc' in today_df.columns:
    cv_acc = float(today_df['cv_acc'].iloc[0])
    cv_cov = float(today_df['cv_cov'].iloc[0])

# ── Build full email HTML ─────────────────────────────────────────────────────
email_html = build_email_html(
    date_str=TODAY.isoformat(),
    picks_rows=picks_rows,
    yesterday_rows=yest_df if not yest_df.empty else None,
    ytd_df=ytd_df if not ytd_df.empty else None,
    today_df_all=today_df if not today_df.empty else None,
    lr_threshold=lr_threshold,
    nn_threshold=nn_threshold,
    cv_acc=cv_acc,
    cv_cov=cv_cov,
    yesterday=YESTERDAY,
    today=TODAY,
    unit=UNIT,
    get_odds_fn=None,
)

# ── Subject ───────────────────────────────────────────────────────────────────
n_picks    = len(picks_rows)
n_consensus = sum(1 for p in picks_rows if p.get('consensus'))
_picks_summary = f'{n_picks} pick{"s" if n_picks != 1 else ""}' + (f', {n_consensus} consensus' if n_consensus else '')
email_subject = f'NRFI TEST {TODAY.strftime("%b")} {TODAY.day} | {_picks_summary}'
print(f'Subject: {email_subject}')

send_email(email_html, email_subject, TODAY.isoformat(), chart_bytes=chart_bytes)
