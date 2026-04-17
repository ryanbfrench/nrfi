"""
utils/pl_calc.py
----------------
P/L calculation utilities for the NRFI pipeline.

Policy: P/L is ONLY computed when real odds are available.
No -110 fallback. Games without real odds are counted as 'pending'
in summaries and shown as '—' in the email.
"""

import pandas as pd
import numpy as np

UNIT = 10  # dollars per unit — must match daily_picks.py


def compute_pl(correct, pred, nrfi_odds, yrfi_odds, unit=UNIT):
    """
    Compute actual P/L for one graded bet.

    Returns float P/L, or None if:
      - correct is None/NaN (not yet graded)
      - nrfi_odds or yrfi_odds is None/NaN (no real odds — no P/L without real odds)

    No -110 fallback. Caller is responsible for handling None.
    """
    if correct is None or (isinstance(correct, float) and np.isnan(correct)):
        return None
    raw = nrfi_odds if pred == 'NRFI' else yrfi_odds
    if raw is None or (isinstance(raw, float) and np.isnan(raw)):
        return None
    odds = int(raw)
    if int(correct) == 1:
        return round(unit * (100 / abs(odds) if odds < 0 else odds / 100), 2)
    return -float(unit)


def summarize_pl(results_df, model='lr', unit=UNIT):
    """
    Summarize P/L for a model over graded games with real odds.

    Only includes rows where:
      - model was confident (lr_confident / nn_confident == True)
      - game is graded (lr_correct / nn_correct is not null)
      - real odds available (nrfi_odds is not null)

    pending_count: confident picks that are graded but have no real odds yet
                  (pick was made; result known; waiting for odds backfill to count P/L)

    Returns dict:
      wins, losses, pl, roi_pct, graded_count, pending_count
    """
    conf_col    = f'{model}_confident'
    correct_col = f'{model}_correct'
    pred_col    = f'{model}_pred'

    empty = dict(wins=0, losses=0, pl=0.0, roi_pct=0.0, graded_count=0, pending_count=0)

    if results_df is None or results_df.empty:
        return empty
    if conf_col not in results_df.columns:
        return empty

    conf = results_df[results_df[conf_col] == True]
    if conf.empty:
        return empty

    has_odds = conf['nrfi_odds'].notna() if 'nrfi_odds' in conf.columns else pd.Series(False, index=conf.index)
    is_graded = conf[correct_col].notna() if correct_col in conf.columns else pd.Series(False, index=conf.index)

    graded_with_odds = conf[is_graded & has_odds]
    graded_no_odds   = conf[is_graded & ~has_odds]

    pending_count = len(graded_no_odds)

    if graded_with_odds.empty:
        return {**empty, 'pending_count': pending_count}

    wins = 0; losses = 0; pl_total = 0.0
    for _, r in graded_with_odds.iterrows():
        pl = compute_pl(r[correct_col], r[pred_col], r.get('nrfi_odds'), r.get('yrfi_odds'), unit)
        if pl is not None:
            pl_total += pl
            if pl > 0:
                wins += 1
            else:
                losses += 1

    graded_count = wins + losses
    roi_pct = (pl_total / (graded_count * unit) * 100) if graded_count > 0 else 0.0

    return dict(
        wins=wins,
        losses=losses,
        pl=round(pl_total, 2),
        roi_pct=round(roi_pct, 1),
        graded_count=graded_count,
        pending_count=pending_count,
    )


def running_pl_by_date(results_df, model='lr', unit=UNIT):
    """
    Returns list of (date_str, cumulative_pl) tuples for graded games with real odds.
    Ordered by date ascending. Excludes games without real odds.
    """
    conf_col    = f'{model}_confident'
    correct_col = f'{model}_correct'
    pred_col    = f'{model}_pred'

    if results_df is None or results_df.empty:
        return []
    if conf_col not in results_df.columns:
        return []

    conf = results_df[results_df[conf_col] == True]
    if conf.empty:
        return []

    has_odds  = conf['nrfi_odds'].notna() if 'nrfi_odds' in conf.columns else pd.Series(False, index=conf.index)
    is_graded = conf[correct_col].notna() if correct_col in conf.columns else pd.Series(False, index=conf.index)
    graded    = conf[is_graded & has_odds].copy()

    if graded.empty:
        return []

    graded = graded.sort_values('date')
    result = []
    cumulative = 0.0
    for _, r in graded.iterrows():
        pl = compute_pl(r[correct_col], r[pred_col], r.get('nrfi_odds'), r.get('yrfi_odds'), unit)
        if pl is not None:
            cumulative += pl
            result.append((str(r['date']), round(cumulative, 2)))

    return result
