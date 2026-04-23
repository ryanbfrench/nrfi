"""
utils/email_charts.py
---------------------
7-day confidence band timeline chart embedded in the daily email.

Design: two panels (LR top, NN bottom). X-axis = dates (7 days), Y-axis = YRFI probability.
Continuous shaded bands run left-to-right across all days:
  - Blue   (NRFI zone): prob < threshold_low
  - Grey   (no pick):   threshold_low ≤ prob ≤ threshold_high
  - Purple (YRFI zone): prob > threshold_high

Threshold boundary lines are drawn as continuous lines that shift day-to-day.
Game dots per day:
  - Green  filled: confident pick, correct
  - Red    filled: confident pick, incorrect
  - Blue/purple filled: confident pick, ungraded (today)
  - Hollow (small):  no pick

history_df must contain: game_date (YYYY-MM-DD str), matchup,
lr_prob_yrfi, lr_threshold_low, lr_threshold_high, lr_confident,
nn_prob_yrfi, nn_threshold_low, nn_threshold_high, nn_confident.
Optional for result coloring: lr_correct, nn_correct (0/1/NaN).

Returns PNG bytes for inline CID attachment, or None on failure.
"""

import io
from datetime import datetime

COLOR_NRFI    = '#2563eb'   # blue
COLOR_YRFI    = '#7c3aed'   # purple
COLOR_NONE    = '#9ca3af'   # grey
COLOR_BAND    = '#f3f4f6'   # light grey no-pick zone
COLOR_TEXT    = '#374151'
COLOR_WIN     = '#16a34a'   # green
COLOR_LOSS    = '#dc2626'   # red


def build_threshold_timeline(history_df):
    """
    Generate 7-day confidence band timeline for LR and NN models.
    X-axis: game dates; Y-axis: YRFI probability.
    Continuous bands + lines across all days; dots colored by result.
    """
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        import matplotlib.ticker as mticker
        import numpy as np
        import pandas as pd
    except ImportError:
        return None

    if history_df is None or history_df.empty:
        return None

    if 'game_date' not in history_df.columns:
        return None

    models = [
        ('LR', 'lr_prob_yrfi', 'lr_threshold_low', 'lr_threshold_high', 'lr_confident', 'lr_correct'),
        ('NN', 'nn_prob_yrfi', 'nn_threshold_low', 'nn_threshold_high', 'nn_confident', 'nn_correct'),
    ]
    required = [col for _, p, lo, hi, c, _ in models for col in (p, lo, hi, c)]
    if not all(c in history_df.columns for c in required) or 'matchup' not in history_df.columns:
        return None

    dates   = sorted(history_df['game_date'].unique())
    n_days  = len(dates)
    date_xi = {d: i for i, d in enumerate(dates)}

    def fmt_date(d):
        try:
            dt = datetime.strptime(str(d), '%Y-%m-%d')
            return dt.strftime('%b %-d')
        except Exception:
            return str(d)[-5:]

    try:
        fig, axes = plt.subplots(2, 1, figsize=(10, 5.5), facecolor='white',
                                 gridspec_kw={'hspace': 0.55})
        fig.subplots_adjust(left=0.07, right=0.97, top=0.90, bottom=0.10)

        for ax_idx, (label, prob_col, low_col, high_col, conf_col, correct_col) in enumerate(models):
            ax = axes[ax_idx]
            ax.set_facecolor('white')

            rows = history_df[history_df[prob_col].notna()].copy()
            if rows.empty:
                ax.set_title(f'{label} Model — no data', fontsize=9, loc='left',
                             color=COLOR_TEXT, pad=6)
                ax.set_xticks(range(n_days))
                ax.set_xticklabels([fmt_date(d) for d in dates], fontsize=7.5)
                continue

            # Build per-day threshold arrays for continuous band/line rendering
            x_vals   = []
            low_vals = []
            hi_vals  = []
            for date in dates:
                xi       = date_xi[date]
                day_rows = rows[rows['game_date'] == date]
                if day_rows.empty:
                    # Gap day: carry forward last known threshold or skip
                    if x_vals:
                        x_vals.append(xi)
                        low_vals.append(low_vals[-1])
                        hi_vals.append(hi_vals[-1])
                    continue
                x_vals.append(xi)
                low_vals.append(float(day_rows[low_col].iloc[0]))
                hi_vals.append(float(day_rows[high_col].iloc[0]))

            if not x_vals:
                continue

            x_arr  = np.array(x_vals,   dtype=float)
            lo_arr = np.array(low_vals,  dtype=float)
            hi_arr = np.array(hi_vals,   dtype=float)

            # Extend bands to full x-range edges so shading looks continuous
            x_full  = np.concatenate([[x_arr[0] - 0.5], x_arr, [x_arr[-1] + 0.5]])
            lo_full = np.concatenate([[lo_arr[0]], lo_arr, [lo_arr[-1]]])
            hi_full = np.concatenate([[hi_arr[0]], hi_arr, [hi_arr[-1]]])

            # ── Continuous shaded bands ───────────────────────────────────────
            ax.fill_between(x_full, 0.0,     lo_full, color=COLOR_NRFI, alpha=0.13, zorder=0)
            ax.fill_between(x_full, lo_full, hi_full, color=COLOR_BAND, alpha=0.65, zorder=0)
            ax.fill_between(x_full, hi_full, 1.0,     color=COLOR_YRFI, alpha=0.13, zorder=0)

            # ── Threshold boundary lines ──────────────────────────────────────
            x_line  = np.concatenate([[x_arr[0] - 0.5], x_arr, [x_arr[-1] + 0.5]])
            ax.plot(x_line, lo_full, color=COLOR_NRFI, lw=1.6, alpha=0.8, zorder=2)
            ax.plot(x_line, hi_full, color=COLOR_YRFI, lw=1.6, alpha=0.8, zorder=2)

            # ── Game dots ─────────────────────────────────────────────────────
            has_correct = correct_col in rows.columns

            for date in dates:
                xi       = date_xi[date]
                day_rows = rows[rows['game_date'] == date]
                for _, row in day_rows.iterrows():
                    prob      = float(row[prob_col])
                    confident = bool(row.get(conf_col, False))
                    correct   = row[correct_col] if has_correct else float('nan')

                    if confident:
                        if pd.notna(correct):
                            dot_color = COLOR_WIN if int(correct) == 1 else COLOR_LOSS
                        else:
                            # Ungraded (today): use zone color
                            dot_color = COLOR_NRFI if prob < float(row[low_col]) else (
                                        COLOR_YRFI if prob > float(row[high_col]) else COLOR_NONE)
                        ax.scatter([xi], [prob], c=dot_color, s=55, zorder=5, linewidths=0)
                    else:
                        zone_col = (COLOR_NRFI if prob < float(row[low_col]) else
                                    COLOR_YRFI if prob > float(row[high_col]) else COLOR_NONE)
                        ax.scatter([xi], [prob], c='none', edgecolors=[zone_col],
                                   s=22, linewidths=1.0, zorder=3, alpha=0.5)

            # ── Axes ──────────────────────────────────────────────────────────
            all_probs = rows[prob_col].astype(float)
            y_lo = max(0.0, min(all_probs.min(), lo_arr.min()) - 0.04)
            y_hi = min(1.0, max(all_probs.max(), hi_arr.max()) + 0.04)
            if y_hi - y_lo < 0.20:
                mid = (y_lo + y_hi) / 2
                y_lo, y_hi = max(0.0, mid - 0.10), min(1.0, mid + 0.10)

            # Three fixed Y ticks: threshold_low, 0.5, threshold_high
            last_rows = rows[rows['game_date'] == dates[-1]]
            tl = float(last_rows[low_col].iloc[0])  if not last_rows.empty else lo_arr[-1]
            th = float(last_rows[high_col].iloc[0]) if not last_rows.empty else hi_arr[-1]
            yticks = sorted({tl, 0.5, th})
            ax.set_yticks(yticks)
            ax.set_yticklabels([f'{v:.3f}' for v in yticks], fontsize=7)

            ax.set_xlim(x_full[0], x_full[-1])
            ax.set_ylim(y_lo, y_hi)
            ax.set_xticks(range(n_days))
            ax.set_xticklabels([fmt_date(d) for d in dates], fontsize=7.5)
            ax.tick_params(axis='y', length=3, pad=3)
            ax.tick_params(axis='x', labelsize=7.5, length=3, pad=3)
            ax.set_ylabel('YRFI prob', fontsize=7.5, labelpad=4)
            ax.grid(axis='y', color='#e5e7eb', linewidth=0.5, zorder=0)
            ax.set_title(label, fontsize=9, loc='left', color=COLOR_TEXT, pad=6)

            for spine in ('top', 'right'):
                ax.spines[spine].set_visible(False)
            ax.spines['bottom'].set_color('#e5e7eb')
            ax.spines['left'].set_color('#e5e7eb')

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
