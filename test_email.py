"""
test_email.py — send a test email to verify the new subject format and chart layout.
Run from the project root: python test_email.py
"""

import os, sys
import numpy as np
import pandas as pd
import random
from datetime import date, timedelta
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Load .env
_env = Path(__file__).parent / '.env'
if _env.exists():
    for _line in _env.read_text().splitlines():
        _line = _line.strip()
        if _line and not _line.startswith('#') and '=' in _line:
            _k, _v = _line.split('=', 1)
            os.environ.setdefault(_k.strip(), _v.strip())

from utils.email_charts import build_threshold_timeline
from utils.email_html   import send_email

random.seed(42)
np.random.seed(42)

# ── Fake 7-day history ───────────────────────────────────────────────────────
today = date.today()
dates = [(today - timedelta(days=i)).isoformat() for i in range(6, -1, -1)]

MATCHUPS = [
    'NYY @ BOS', 'LAD @ SF', 'HOU @ TEX', 'ATL @ NYM',
    'CHC @ MIL', 'MIN @ DET', 'SEA @ OAK', 'CLE @ KC',
    'TOR @ TB',  'PHI @ WSH', 'CIN @ PIT', 'COL @ ARI',
    'STL @ MIA', 'BAL @ DET', 'SD @ LAA',
]

rows = []
for d in dates:
    n_games = random.randint(10, 15)
    for matchup in random.sample(MATCHUPS, min(n_games, len(MATCHUPS))):
        lr_prob = round(random.uniform(0.38, 0.62), 4)
        nn_prob = round(random.uniform(0.38, 0.62), 4)
        tl, th  = 0.455, 0.545
        lr_conf = lr_prob < tl or lr_prob > th
        nn_conf = nn_prob < tl or nn_prob > th
        graded  = d < today.isoformat()
        rows.append({
            'game_date':          d,
            'matchup':            matchup,
            'lr_prob_yrfi':       lr_prob,
            'lr_threshold_low':   tl,
            'lr_threshold_high':  th,
            'lr_confident':       lr_conf,
            'lr_correct':         float(random.random() > 0.44) if (lr_conf and graded) else float('nan'),
            'nn_prob_yrfi':       nn_prob,
            'nn_threshold_low':   tl,
            'nn_threshold_high':  th,
            'nn_confident':       nn_conf,
            'nn_correct':         float(random.random() > 0.44) if (nn_conf and graded) else float('nan'),
        })

history_df = pd.DataFrame(rows)

# ── Generate chart ───────────────────────────────────────────────────────────
chart_bytes = build_threshold_timeline(history_df)
print(f'Chart: {len(chart_bytes) if chart_bytes else 0} bytes')

# ── Subject (mirrors daily_picks.py logic) ────────────────────────────────────
_date_label    = today.strftime('%b') + ' ' + str(today.day)
_picks_summary = '3 picks, 1 consensus'
_yest_summary  = ' | Yesterday 5-0 (+$39.82)'
email_subject  = f'NRFI {_date_label} | {_picks_summary}{_yest_summary}'
print(f'Subject: {email_subject}')

# ── Simple HTML body ──────────────────────────────────────────────────────────
html_body = f"""
<html><body style="font-family:sans-serif;max-width:680px;margin:0 auto;padding:24px;color:#111827">
  <h2 style="margin-bottom:4px">NRFI Test Email</h2>
  <p style="color:#6b7280;margin-top:0">Subject format + chart layout verification &mdash; {today}</p>
  <p><strong>Subject line:</strong><br>
     <code style="background:#f3f4f6;padding:4px 8px;border-radius:4px">{email_subject}</code></p>
  <p>This email verifies:
    <ul>
      <li>New subject format: <code>NRFI Apr 26 | 3 picks | Yesterday 5-0 (+$39.82)</code></li>
      <li>New chart: 2 rows &times; 7 columns (one column per day)</li>
      <li>Row titles: &ldquo;Logistic Regression&rdquo; and &ldquo;Neural Network&rdquo;</li>
    </ul>
  </p>
  <!-- THRESHOLD_CHART_PLACEHOLDER -->
</body></html>
"""

send_email(html_body, email_subject, today.isoformat(), chart_bytes=chart_bytes)
