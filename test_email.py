"""
test_email.py
-------------
Send a test email using existing S3 data — no model retraining.
Renders the full production email (same HTML as the daily run).

Requires .env with SES_FROM and SES_TO.

Usage:
    python test_email.py
"""

import io, json, os, subprocess
from datetime import date, timedelta
from pathlib import Path

import boto3
import pandas as pd
from dotenv import load_dotenv

load_dotenv(Path(__file__).parent / '.env')

# ── AWS creds from CLI ────────────────────────────────────────────────────────
try:
    _raw = subprocess.check_output(
        ['aws', 'configure', 'export-credentials', '--format', 'env-no-export'],
        text=True, stderr=subprocess.DEVNULL,
    )
    for _line in _raw.splitlines():
        if '=' in _line:
            k, v = _line.split('=', 1)
            os.environ.setdefault(k.strip(), v.strip())
except Exception:
    pass

# Override SES env vars from .env so send_email picks them up
os.environ['NRFI_SES_FROM'] = os.environ['SES_FROM']
os.environ['NRFI_SES_TO']   = os.environ['SES_TO']

from utils.email_charts import build_threshold_timeline
from utils.email_html   import build_email_html, send_email
from utils.pl_calc      import compute_pl

TODAY     = date.today()
YESTERDAY = TODAY - timedelta(days=1)
BUCKET    = 'nrfi-store'
UNIT      = 10

s3 = boto3.client('s3')


def s3_csv(key):
    try:
        obj = s3.get_object(Bucket=BUCKET, Key=key)
        df  = pd.read_csv(io.BytesIO(obj['Body'].read()))
        print(f'  loaded {key} ({len(df)} rows)')
        return df
    except Exception as e:
        print(f'  missing {key}: {e}')
        return None


# ── Load data from S3 ─────────────────────────────────────────────────────────
print('Loading S3 data...')
today_df     = s3_csv(f'game_log/{TODAY.year}/{TODAY.isoformat()}.csv')
yesterday_df = s3_csv(f'game_log/{YESTERDAY.year}/{YESTERDAY.isoformat()}.csv')
results_df   = s3_csv('results/results.csv')

picks_payload = []
try:
    obj = s3.get_object(Bucket=BUCKET, Key=f'picks/{TODAY.year}/{TODAY.isoformat()}.json')
    picks_payload = json.loads(obj['Body'].read()).get('picks', [])
    print(f'  loaded {len(picks_payload)} picks')
except Exception as e:
    print(f'  missing picks JSON: {e}')

# ── YTD df (season slice of results) ─────────────────────────────────────────
ytd_df = None
if results_df is not None and not results_df.empty:
    season_start = f'{TODAY.year}-04-15'
    ytd_df = results_df[results_df['date'] >= season_start].copy()

# ── Merge yesterday actuals into yesterday_df ─────────────────────────────────
if yesterday_df is not None and results_df is not None:
    res = results_df[['date', 'matchup', 'lr_correct', 'nn_correct', 'actual_yrfi']].copy()
    res = res[res['date'] == str(YESTERDAY)].drop(columns='date')
    yesterday_df = yesterday_df.drop(columns=['lr_correct', 'nn_correct', 'actual_yrfi'], errors='ignore')
    yesterday_df = yesterday_df.merge(res, on='matchup', how='left')

# ── 7-day history for chart ───────────────────────────────────────────────────
print('Loading 7-day history for chart...')
hist_dfs = []
for d_off in range(6, -1, -1):
    hd  = TODAY - timedelta(days=d_off)
    hdf = s3_csv(f'game_log/{hd.year}/{hd.isoformat()}.csv')
    if hdf is not None:
        hdf['game_date'] = str(hd)
        hist_dfs.append(hdf)

if hist_dfs:
    history = pd.concat(hist_dfs, ignore_index=True)
elif today_df is not None:
    history = today_df.copy()
    history['game_date'] = str(TODAY)
else:
    history = pd.DataFrame()

# Merge lr_correct/nn_correct from results (game logs save these as None at pick time)
if results_df is not None and not history.empty:
    res = results_df[['date', 'matchup', 'lr_correct', 'nn_correct']].rename(columns={'date': 'game_date'})
    history = history.drop(columns=['lr_correct', 'nn_correct'], errors='ignore')
    history = history.merge(res, on=['game_date', 'matchup'], how='left')

# ── Build chart ───────────────────────────────────────────────────────────────
print('Building chart...')
chart_bytes = build_threshold_timeline(history)
print(f'Chart: {len(chart_bytes)} bytes' if chart_bytes else 'Chart: None')

# ── Email subject ─────────────────────────────────────────────────────────────
yest_summary = ''
if yesterday_df is not None and not yesterday_df.empty:
    conf   = yesterday_df[yesterday_df['lr_confident'] | yesterday_df['nn_confident']]
    graded = conf[conf['lr_correct'].notna() | conf['nn_correct'].notna()]
    if not graded.empty:
        corrects, pl_vals = [], []
        for _, r in graded.iterrows():
            if r.get('lr_confident') and pd.notna(r.get('lr_correct')):
                corrects.append(int(r['lr_correct']))
                v = compute_pl(r['lr_correct'], r['lr_pred'], r.get('nrfi_odds'), r.get('yrfi_odds'))
            elif r.get('nn_confident') and pd.notna(r.get('nn_correct')):
                corrects.append(int(r['nn_correct']))
                v = compute_pl(r['nn_correct'], r['nn_pred'], r.get('nrfi_odds'), r.get('yrfi_odds'))
            else:
                v = None
            if v is not None:
                pl_vals.append(v)
        w, l = sum(corrects), len(corrects) - sum(corrects)
        pl   = sum(pl_vals)
        yest_summary = f' | Yesterday {w}-{l} ({("+" if pl >= 0 else "")}${pl:.2f})'

n_picks = len([p for p in picks_payload if not p.get('consensus')])
n_cons  = len([p for p in picks_payload if p.get('consensus')])
picks_s = f'{n_picks} pick{"s" if n_picks != 1 else ""}' + (f', {n_cons} consensus' if n_cons else '')
subject = f'[TEST] NRFI {TODAY} — {picks_s}{yest_summary}'
print(f'Subject: {subject}')

# ── Build and send full production email ─────────────────────────────────────
email_html = build_email_html(
    date_str=str(TODAY),
    picks_rows=picks_payload,
    yesterday_rows=yesterday_df,
    ytd_df=ytd_df,
    today_df_all=today_df,
    lr_threshold=0.545,
    nn_threshold=0.545,
    cv_acc=0.0,
    cv_cov=0.0,
    yesterday=YESTERDAY,
    today=TODAY,
    unit=UNIT,
    get_odds_fn=None,   # uses stored odds from game log; no live scrape
)

send_email(email_html, subject, str(TODAY), chart_bytes=chart_bytes)
