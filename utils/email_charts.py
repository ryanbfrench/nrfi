"""
utils/email_charts.py
---------------------
7-day confidence band chart embedded in the daily email.

Layout: 2 rows × 7 columns.
  - Row 0: Logistic Regression
  - Row 1: Neural Network
  - Each column: one day's games

Within each cell:
  - Horizontal shaded bands: NRFI (blue), no-pick (grey), YRFI (purple)
  - Threshold boundary lines (horizontal)
  - One dot per game; confident picks filled, unconfident hollow
    - Green: confident + correct
    - Red:   confident + incorrect
    - Blue/purple: confident + ungraded (today)
    - Hollow: no pick

history_df must contain: game_date (YYYY-MM-DD str), matchup,
lr_prob_yrfi, lr_threshold_low, lr_threshold_high, lr_confident,
nn_prob_yrfi, nn_threshold_low, nn_threshold_high, nn_confident.
Optional for result coloring: lr_correct, nn_correct (0/1/NaN).

Returns PNG bytes for inline CID attachment, or None on failure.
"""

import io
from datetime import datetime

COLOR_NRFI = '#2563eb'
COLOR_YRFI = '#7c3aed'
COLOR_NONE = '#9ca3af'
COLOR_BAND = '#f3f4f6'
COLOR_TEXT = '#374151'
COLOR_WIN  = '#16a34a'
COLOR_LOSS = '#dc2626'


def build_threshold_timeline(history_df):
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        import numpy as np
        import pandas as pd
    except ImportError:
        return None

    if history_df is None or history_df.empty:
        return None
    if 'game_date' not in history_df.columns:
        return None

    models = [
        ('Logistic Regression', 'lr_prob_yrfi', 'lr_threshold_low', 'lr_threshold_high', 'lr_confident', 'lr_correct'),
        ('Neural Network',      'nn_prob_yrfi', 'nn_threshold_low', 'nn_threshold_high', 'nn_confident', 'nn_correct'),
    ]
    required = [col for _, p, lo, hi, c, _ in models for col in (p, lo, hi, c)]
    if not all(c in history_df.columns for c in required) or 'matchup' not in history_df.columns:
        return None

    dates  = sorted(history_df['game_date'].unique())
    n_days = len(dates)

    def fmt_date(d):
        try:
            dt = datetime.strptime(str(d), '%Y-%m-%d')
            return dt.strftime('%b') + ' ' + str(dt.day)
        except Exception:
            return str(d)[-5:]

    try:
        fig, axes = plt.subplots(
            2, n_days,
            figsize=(max(2.2 * n_days, 8), 6),
            facecolor='white',
            gridspec_kw={'hspace': 0.30, 'wspace': 0.10},
        )
        fig.subplots_adjust(left=0.07, right=0.98, top=0.88, bottom=0.06)

        # Ensure axes is always shape (2, n_days)
        if n_days == 1:
            axes = np.array([[axes[0]], [axes[1]]])

        for row_idx, (label, prob_col, low_col, high_col, conf_col, correct_col) in enumerate(models):
            all_rows = history_df[history_df[prob_col].notna()].copy()
            has_correct = correct_col in all_rows.columns

            for col_idx, date in enumerate(dates):
                ax = axes[row_idx, col_idx]
                ax.set_facecolor('white')
                for spine in ('top', 'right', 'bottom'):
                    ax.spines[spine].set_visible(False)
                ax.spines['left'].set_color('#e5e7eb')

                day_rows = all_rows[all_rows['game_date'] == date].copy()

                # Threshold for this day
                if not day_rows.empty:
                    tl = float(day_rows[low_col].iloc[0])
                    th = float(day_rows[high_col].iloc[0])
                else:
                    tl, th = 0.455, 0.545

                # Horizontal bands
                ax.axhspan(0.0, tl,  color=COLOR_NRFI, alpha=0.13, zorder=0)
                ax.axhspan(tl,  th,  color=COLOR_BAND, alpha=0.65, zorder=0)
                ax.axhspan(th,  1.0, color=COLOR_YRFI, alpha=0.13, zorder=0)
                ax.axhline(tl, color=COLOR_NRFI, lw=1.2, alpha=0.8, zorder=2)
                ax.axhline(th, color=COLOR_YRFI, lw=1.2, alpha=0.8, zorder=2)

                # Game dots
                for g_idx, (_, row) in enumerate(day_rows.iterrows()):
                    prob      = float(row[prob_col])
                    confident = bool(row.get(conf_col, False))
                    correct   = row[correct_col] if has_correct else float('nan')

                    if confident:
                        if pd.notna(correct):
                            dot_color = COLOR_WIN if int(correct) == 1 else COLOR_LOSS
                        else:
                            dot_color = (COLOR_NRFI if prob < tl else
                                         COLOR_YRFI if prob > th else COLOR_NONE)
                        ax.scatter([g_idx], [prob], c=dot_color, s=50, zorder=5, linewidths=0)
                    else:
                        zone_col = (COLOR_NRFI if prob < tl else
                                    COLOR_YRFI if prob > th else COLOR_NONE)
                        ax.scatter([g_idx], [prob], c='none', edgecolors=[zone_col],
                                   s=22, linewidths=1.0, zorder=3, alpha=0.5)

                # Column date header (top row only, centered)
                if row_idx == 0:
                    ax.set_title(fmt_date(date), fontsize=8, color=COLOR_TEXT, pad=4, loc='center')

                # Row label above leftmost column (left-aligned, doesn't overwrite centered date)
                if col_idx == 0:
                    ax.set_title(label, fontsize=9, color=COLOR_TEXT, pad=4, loc='left',
                                 fontweight='semibold')

                # Y-axis: ticks only on leftmost column
                if col_idx == 0:
                    yticks = sorted({tl, 0.5, th})
                    ax.set_yticks(yticks)
                    ax.set_yticklabels([f'{v:.3f}' for v in yticks], fontsize=6.5)
                    ax.tick_params(axis='y', length=3, pad=2)
                else:
                    ax.set_yticks([])

                ax.set_xticks([])
                ax.set_ylim(0.35, 0.65)
                n_games = max(len(day_rows), 1)
                ax.set_xlim(-0.7, n_games - 0.3)

        buf = io.BytesIO()
        fig.savefig(buf, format='png', dpi=130, bbox_inches='tight', facecolor='white')
        plt.close(fig)
        buf.seek(0)
        return buf.read()

    except Exception:
        try:
            plt.close('all')
        except Exception:
            pass
        return None
