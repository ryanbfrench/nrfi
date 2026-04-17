"""
utils/odds_backfill.py
----------------------
Targeted historical odds backfill for confident picks missing real odds.

Called from grade_yesterday() in daily_picks.py after grading actuals.
Only fetches dates where confident picks have null nrfi_odds/yrfi_odds.
Uses S3 odds cache (odds/{year}/{date}.json) — same format as daily
fetch_odds() and backfill_odds_2025.py. Never re-fetches a cached date.

API: The Odds API historical endpoint (separate key from live key).
Env var: HISTORICAL_ODDS_API_KEY

Key behaviors:
  - Skips dates already cached in S3
  - Stops early if x-requests-remaining drops below STOP_THRESHOLD
  - Handles doubleheaders by assigning odds in commence_time order
  - Fails silently: errors are logged, original df returned unchanged
"""

import json
import time

import numpy as np
import pandas as pd
import requests

EVENTS_URL     = 'https://api.the-odds-api.com/v4/historical/sports/baseball_mlb/events'
ODDS_URL       = 'https://api.the-odds-api.com/v4/historical/sports/baseball_mlb/events/{event_id}/odds'
SNAPSHOT_HOUR  = '14:00:00Z'   # morning lines — consistent with backfill_odds_2025.py
STOP_THRESHOLD = 200            # abort if remaining quota drops below this
SLEEP_BETWEEN  = 1.0            # seconds between API calls

# Team name → abbreviation (must match fetch_odds() in daily_picks.py)
_TEAM_MAP = {
    'arizona diamondbacks': 'ARI', 'atlanta braves': 'ATL',       'baltimore orioles': 'BAL',
    'boston red sox': 'BOS',       'chicago cubs': 'CHC',          'chicago white sox': 'CWS',
    'cincinnati reds': 'CIN',      'cleveland guardians': 'CLE',   'colorado rockies': 'COL',
    'detroit tigers': 'DET',       'houston astros': 'HOU',        'kansas city royals': 'KC',
    'los angeles angels': 'LAA',   'los angeles dodgers': 'LAD',   'miami marlins': 'MIA',
    'milwaukee brewers': 'MIL',    'minnesota twins': 'MIN',       'new york mets': 'NYM',
    'new york yankees': 'NYY',     'oakland athletics': 'OAK',     'philadelphia phillies': 'PHI',
    'pittsburgh pirates': 'PIT',   'san diego padres': 'SD',       'san francisco giants': 'SF',
    'seattle mariners': 'SEA',     'st. louis cardinals': 'STL',   'tampa bay rays': 'TB',
    'texas rangers': 'TEX',        'toronto blue jays': 'TOR',     'washington nationals': 'WAS',
    'athletics': 'OAK',
}


def _s3_key(date_str):
    """S3 key for a date's odds cache. date_str: 'YYYY-MM-DD'."""
    year = date_str[:4]
    return f'odds/{year}/{date_str}.json'


def _load_cached(s3_client, bucket, date_str):
    """Load cached events list from S3. Returns list or None if not cached."""
    try:
        import io
        obj = s3_client.get_object(Bucket=bucket, Key=_s3_key(date_str))
        return json.loads(obj['Body'].read())
    except Exception:
        return None


def _fetch_and_cache(s3_client, bucket, date_str, api_key):
    """
    Fetch NRFI/YRFI odds for date_str from the historical API and cache to S3.

    Returns (events_list, last_requests_remaining).
    events_list may be empty on API failure — the empty list is still cached
    to skip this date on future runs.
    """
    snapshot       = f'{date_str}T{SNAPSHOT_HOUR}'
    last_remaining = None
    results        = []

    # Step 1: get all MLB events for the date
    try:
        r = requests.get(EVENTS_URL, params={
            'apiKey': api_key,
            'date':   snapshot,
            'sport':  'baseball_mlb',
        }, timeout=20)
        last_remaining = int(r.headers.get('x-requests-remaining', -1))
        if r.status_code != 200:
            return [], last_remaining
        events_data = r.json()
        events = (events_data.get('data', events_data)
                  if isinstance(events_data, dict) else events_data)
    except Exception:
        return [], last_remaining

    time.sleep(SLEEP_BETWEEN)

    # Step 2: fetch NRFI market odds per event
    for event in events:
        if last_remaining is not None and last_remaining < STOP_THRESHOLD:
            break
        event_id = event.get('id')
        if not event_id:
            continue
        try:
            r = requests.get(ODDS_URL.format(event_id=event_id), params={
                'apiKey':     api_key,
                'regions':    'us',
                'markets':    'totals_1st_1_innings',
                'oddsFormat': 'american',
                'date':       snapshot,
            }, timeout=20)
            last_remaining = int(r.headers.get('x-requests-remaining', -1))
            if r.status_code == 200:
                odds_data = r.json()
                data = (odds_data.get('data', odds_data)
                        if isinstance(odds_data, dict) else odds_data)
                if isinstance(data, list):
                    results.extend(data)
                elif data:
                    results.append(data)
        except Exception:
            pass
        time.sleep(SLEEP_BETWEEN)

    # Cache result to S3 (even if empty, to prevent future re-fetches)
    try:
        s3_client.put_object(
            Bucket=bucket,
            Key=_s3_key(date_str),
            Body=json.dumps(results, separators=(',', ':')),
            ContentType='application/json',
        )
    except Exception:
        pass  # Cache failure is non-fatal

    return results, last_remaining


def _parse_odds_from_events(events):
    """
    Parse a list of event odds dicts into: {'AWAY@HOME': (nrfi_odds, yrfi_odds)}.

    Doubleheader handling: if two events share the same AWAY@HOME matchup key,
    the one with the earlier commence_time is treated as game 1 and assigned
    first. Game 2 is skipped (odds for doubleheader game 2 are rare/unreliable).
    """
    if not events:
        return {}

    # Build sorted candidate list: (matchup_key, commence_time, nrfi, yrfi)
    candidates = []
    for event in events:
        away_full = event.get('away_team', '')
        home_full = event.get('home_team', '')
        away_abbv = _TEAM_MAP.get(away_full.lower())
        home_abbv = _TEAM_MAP.get(home_full.lower())
        if not away_abbv or not home_abbv:
            continue

        best_yrfi = best_nrfi = None
        for bk in event.get('bookmakers', []):
            for mkt in bk.get('markets', []):
                for outcome in mkt.get('outcomes', []):
                    american = outcome.get('price')
                    point    = outcome.get('point')
                    if american is None or point is None:
                        continue
                    if point == 0.5:
                        name = outcome['name']
                        if name == 'Over' and (best_yrfi is None or american > best_yrfi):
                            best_yrfi = american
                        elif name == 'Under' and (best_nrfi is None or american > best_nrfi):
                            best_nrfi = american

        if best_nrfi is not None and best_yrfi is not None:
            matchup_key = f'{away_abbv}@{home_abbv}'
            commence    = event.get('commence_time', '')
            candidates.append((matchup_key, commence, best_nrfi, best_yrfi))

    # Sort by commence_time ascending — game 1 of a doubleheader gets priority
    candidates.sort(key=lambda x: x[1])
    odds = {}
    for matchup_key, _, nrfi, yrfi in candidates:
        if matchup_key not in odds:   # skip game 2 of doubleheaders
            odds[matchup_key] = (nrfi, yrfi)

    return odds


def backfill_missing_odds(results_df, s3_client, bucket, hist_api_key):
    """
    For confident picks in results_df with null nrfi_odds, fetch real odds
    from S3 cache (or historical API if not cached) and fill them in.

    Args:
        results_df:    The combined results DataFrame from grade_yesterday()
        s3_client:     boto3 S3 client
        bucket:        S3 bucket name (e.g. 'nrfi-store')
        hist_api_key:  The Odds API historical key
                       (env var: HISTORICAL_ODDS_API_KEY)

    Returns:
        Updated DataFrame with nrfi_odds/yrfi_odds filled where found.
        Returns original df unchanged on any error.

    Behaviors:
        - S3-cached dates are always used regardless of hist_api_key
        - hist_api_key is only needed to fetch dates not already in S3 cache
        - If API quota drops below STOP_THRESHOLD, stops processing further dates
        - Remaining unfilled rows keep null odds (P/L excluded until next run)
        - Never raises an exception
    """
    from utils.logger import log

    if results_df is None or results_df.empty:
        return results_df

    if 'nrfi_odds' not in results_df.columns:
        return results_df

    # Find confident picks with null odds
    lr_conf = results_df['lr_confident'] if 'lr_confident' in results_df.columns else pd.Series(False, index=results_df.index)
    nn_conf = results_df['nn_confident'] if 'nn_confident' in results_df.columns else pd.Series(False, index=results_df.index)
    needs_backfill = results_df[(lr_conf | nn_conf) & results_df['nrfi_odds'].isna()]

    if needs_backfill.empty:
        log('INFO', 'odds_backfill: no missing odds to backfill')
        return results_df

    dates_needed = sorted(needs_backfill['date'].dropna().unique())
    log('INFO', f'odds_backfill: {len(needs_backfill)} rows need odds across {len(dates_needed)} date(s)')

    df            = results_df.copy()
    last_remaining = None

    for date_str in dates_needed:
        if last_remaining is not None and last_remaining < STOP_THRESHOLD:
            log('WARN', f'odds_backfill: quota low ({last_remaining} remaining) — stopping early')
            break

        # Try S3 cache first
        events = _load_cached(s3_client, bucket, date_str)
        if events is not None:
            log('INFO', f'odds_backfill: {date_str} loaded from S3 cache ({len(events)} events)')
        elif hist_api_key:
            log('INFO', f'odds_backfill: fetching {date_str} from historical API')
            events, last_remaining = _fetch_and_cache(s3_client, bucket, date_str, hist_api_key)
            n_events = len(events) if events else 0
            log('INFO', f'odds_backfill: {date_str} fetched ({n_events} events, {last_remaining} remaining)')
            if last_remaining is not None and last_remaining < STOP_THRESHOLD:
                log('WARN', f'odds_backfill: quota low after fetching {date_str} — will stop after this date')
        else:
            log('INFO', f'odds_backfill: {date_str} not in S3 cache and no historical API key — skipping')
            events = []

        odds_for_date = _parse_odds_from_events(events)
        if not odds_for_date:
            log('INFO', f'odds_backfill: no NRFI odds found in {date_str} response')
            continue

        # Fill into df for this date where odds are still missing
        date_mask = (df['date'] == date_str) & df['nrfi_odds'].isna()
        filled = 0
        for idx in df[date_mask].index:
            matchup = df.at[idx, 'matchup']
            if matchup in odds_for_date:
                nrfi_o, yrfi_o = odds_for_date[matchup]
                df.at[idx, 'nrfi_odds'] = nrfi_o
                df.at[idx, 'yrfi_odds'] = yrfi_o
                filled += 1

        total_missing = int(date_mask.sum())
        log('INFO', f'odds_backfill: {date_str} filled {filled}/{total_missing} rows')

    return df
