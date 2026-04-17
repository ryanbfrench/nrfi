"""
utils/email_charts.py
---------------------
Generates the 7-day threshold timeline chart for the daily email.

Chart design: two subplots (LR top, NN bottom), time on X-axis (7 days),
probability on Y-axis (0–1). Each game's predicted YRFI probability is
plotted as a dot for that day, color-coded by zone:
  - Blue  (NRFI pick):  prob_yrfi < threshold_low
  - Grey  (no pick):    threshold_low ≤ prob_yrfi ≤ threshold_high
  - Purple (YRFI pick): prob_yrfi > threshold_high

A shaded horizontal band marks the no-pick zone. Dashed lines mark
the threshold values. Per-day pick counts are annotated below each column.

Data source: game_log/{year}/{date}.csv files in S3 — written daily by
save_game_log() in daily_picks.py. Contains all model outputs and thresholds.
No separate distribution snapshot file needed.

Returns PNG bytes (for inline CID email attachment), or None on any failure.
Email sends HTML-only as graceful fallback when None is returned.
"""

import io

COLOR_NRFI = '#2563eb'   # blue   — matches email palette
COLOR_YRFI = '#7c3aed'   # purple — matches email palette
COLOR_NONE = '#9ca3af'   # grey
COLOR_BAND = '#f3f4f6'   # light grey no-pick band
COLOR_TEXT = '#374151'   # near-black text


def _load_game_log(s3_client, bucket, date_str):
    """Load a game_log CSV from S3. Returns DataFrame or None."""
    try:
        import boto3_io_wrapper  # not a real module — just clarifying intent below
    except ImportError:
        pass
    try:
        import io as _io
        import pandas as pd
        year = date_str[:4]
        obj  = s3_client.get_object(Bucket=bucket, Key=f'game_log/{year}/{date_str}.csv')
        return pd.read_csv(_io.BytesIO(obj['Body'].read()))
    except Exception:
        return None


def build_threshold_timeline(s3_client, bucket, today_date):
    """
    Generate a 7-day threshold timeline PNG for both LR and NN models.

    Args:
        s3_client:  boto3 S3 client
        bucket:     S3 bucket name (e.g. 'nrfi-store')
        today_date: datetime.date — the 7th (most recent) day in the chart

    Returns:
        PNG bytes if successful, None on any failure.
    """
    try:
        import matplotlib
        matplotlib.use('Agg')   # non-interactive — must be set before pyplot import
        import matplotlib.pyplot as plt
        import matplotlib.patches as mpatches
        import numpy as np
        import pandas as pd
        from datetime import timedelta
    except ImportError:
        return None

    try:
        # ── Load last 7 days of game logs ─────────────────────────────────────
        days_data = []   # list of (date_str, df or None)
        for i in range(6, -1, -1):   # 6 days ago → today
            d        = today_date - timedelta(days=i)
            date_str = d.isoformat()
            df       = _load_game_log(s3_client, bucket, date_str)
            days_data.append((date_str, df))

        available = [(ds, df) for ds, df in days_data if df is not None]
        if len(available) < 2:
            return None   # not enough data to draw a meaningful chart

        # ── Figure: 2 rows (LR, NN) × full-width timeline ────────────────────
        fig, axes = plt.subplots(
            2, 1, figsize=(10, 5.5), facecolor='white',
            gridspec_kw={'hspace': 0.55}
        )
        fig.subplots_adjust(left=0.07, right=0.97, top=0.93, bottom=0.18)

        n_days = len(days_data)

        for ax_idx, (model_label, prob_col, thr_low_col, thr_high_col) in enumerate([
            ('LR', 'lr_prob_yrfi', 'lr_threshold_low', 'lr_threshold_high'),
            ('NN', 'nn_prob_yrfi', 'nn_threshold_low', 'nn_threshold_high'),
        ]):
            ax = axes[ax_idx]
            ax.set_facecolor('white')
            ax.set_xlim(-0.5, n_days - 0.5)
            ax.set_ylim(-0.18, 1.06)

            # Compute median thresholds across available days for the band
            thr_lows  = []
            thr_highs = []
            for _, df in available:
                if thr_low_col in df.columns and not df[thr_low_col].empty:
                    thr_lows.append(float(df[thr_low_col].iloc[0]))
                if thr_high_col in df.columns and not df[thr_high_col].empty:
                    thr_highs.append(float(df[thr_high_col].iloc[0]))

            med_low  = float(np.median(thr_lows))  if thr_lows  else 0.455
            med_high = float(np.median(thr_highs)) if thr_highs else 0.545

            # No-pick zone shaded band
            ax.axhspan(med_low, med_high, color=COLOR_BAND, alpha=0.9, zorder=0, linewidth=0)

            # Threshold dashed lines
            ax.axhline(med_low,  color=COLOR_NRFI, linestyle='--', linewidth=1.2,
                       alpha=0.75, zorder=1)
            ax.axhline(med_high, color=COLOR_YRFI,  linestyle='--', linewidth=1.2,
                       alpha=0.75, zorder=1)

            # Threshold value labels on the left
            ax.text(-0.48, med_low,  f'{med_low:.3f}',  fontsize=6.5,
                    color=COLOR_NRFI, va='center', ha='right')
            ax.text(-0.48, med_high, f'{med_high:.3f}', fontsize=6.5,
                    color=COLOR_YRFI,  va='center', ha='right')

            x_ticks  = []
            x_labels = []
            rng      = np.random.default_rng(seed=0)   # deterministic jitter

            for x_idx, (date_str, df) in enumerate(days_data):
                # X-axis label (abbreviated date)
                from datetime import date as _date
                d_obj = _date.fromisoformat(date_str)
                label = d_obj.strftime('%b %-d')   # e.g. "Apr 4"  (Linux/SageMaker)
                x_ticks.append(x_idx)
                x_labels.append(label)

                if df is None or prob_col not in df.columns:
                    # No data — draw placeholder text
                    ax.text(x_idx, 0.5, 'no\ndata', ha='center', va='center',
                            fontsize=6.5, color=COLOR_NONE, style='italic')
                    continue

                probs = df[prob_col].dropna().values
                if len(probs) == 0:
                    continue

                # Jitter x within ±0.22 so dots don't stack on a vertical line
                jitter      = rng.uniform(-0.22, 0.22, size=len(probs))
                x_positions = x_idx + jitter

                # Color by zone
                dot_colors = [
                    COLOR_NRFI if p < med_low else (COLOR_YRFI if p > med_high else COLOR_NONE)
                    for p in probs
                ]

                ax.scatter(x_positions, probs, c=dot_colors, s=16,
                           alpha=0.75, zorder=2, linewidths=0)

                # Per-day count labels below the x-axis
                n_nrfi = int((probs < med_low).sum())
                n_yrfi = int((probs > med_high).sum())
                n_none = len(probs) - n_nrfi - n_yrfi

                # Three-part label: NRFI:N | —:N | YRFI:N
                label_parts = [
                    (f'N:{n_nrfi}',    COLOR_NRFI),
                    (f'—:{n_none}',    COLOR_NONE),
                    (f'Y:{n_yrfi}',    COLOR_YRFI),
                ]
                # Draw each part at progressively lower y positions
                for li, (txt, col) in enumerate(label_parts):
                    ax.text(x_idx, -0.07 - li * 0.035, txt,
                            ha='center', va='top', fontsize=6.0,
                            color=col, fontweight='600',
                            transform=ax.transData)

            # Axes styling
            ax.set_xticks(x_ticks)
            ax.set_xticklabels(x_labels, fontsize=8)
            ax.set_yticks([0, 0.25, 0.5, 0.75, 1.0])
            ax.set_yticklabels(['0%', '25%', '50%', '75%', '100%'], fontsize=7)
            ax.tick_params(axis='x', length=0, pad=14)   # pad makes room for count labels
            ax.tick_params(axis='y', length=2)
            ax.set_ylabel('YRFI prob.', fontsize=8, labelpad=6)
            ax.set_title(
                f'{model_label} Model — Confidence Distribution',
                fontsize=9, fontweight='bold', loc='left', pad=5, color=COLOR_TEXT
            )
            for spine in ('top', 'right'):
                ax.spines[spine].set_visible(False)
            ax.spines['left'].set_color('#e5e7eb')
            ax.spines['bottom'].set_color('#e5e7eb')

        # Legend
        handles = [
            mpatches.Patch(color=COLOR_NRFI, label='NRFI pick'),
            mpatches.Patch(color=COLOR_YRFI,  label='YRFI pick'),
            mpatches.Patch(color=COLOR_NONE, label='No pick'),
        ]
        fig.legend(handles=handles, loc='upper right', fontsize=7.5,
                   frameon=False, bbox_to_anchor=(0.97, 1.0), ncol=3)

        # Render to PNG bytes
        buf = io.BytesIO()
        fig.savefig(buf, format='png', dpi=130, bbox_inches='tight', facecolor='white')
        plt.close(fig)
        buf.seek(0)
        return buf.read()

    except Exception:
        # Never crash the pipeline over a chart
        try:
            plt.close('all')
        except Exception:
            pass
        return None
