"""
utils/email_charts.py
---------------------
Generates the confidence threshold chart embedded in the daily email,
placed directly above the TODAY'S PICKS table.

Chart design: two panels (LR top, NN bottom), YRFI probability on X-axis.
Each game is plotted as a dot:
  - Blue   (NRFI pick):  prob < threshold_low
  - Grey   (no pick):    threshold_low ≤ prob ≤ threshold_high
  - Purple (YRFI pick):  prob > threshold_high

Picked games are large filled dots labeled with their matchup (AWAY@HOME).
Unpicked games are small hollow dots (no label — clutter reduction).
Threshold lines are bold and labeled with their exact values so the reader
immediately understands the criteria used in the table below.

Args: today_df — game log DataFrame produced by save_game_log().
      Required columns: lr_prob_yrfi, lr_threshold_low, lr_threshold_high,
      lr_confident, nn_prob_yrfi, nn_threshold_low, nn_threshold_high,
      nn_confident, matchup.

Returns PNG bytes for inline CID attachment, or None on any failure.
Email falls back to plain HTML when None is returned.
"""

import io

COLOR_NRFI = '#2563eb'   # blue
COLOR_YRFI = '#7c3aed'   # purple
COLOR_NONE = '#9ca3af'   # grey
COLOR_BAND = '#f3f4f6'   # light grey no-pick zone
COLOR_TEXT = '#374151'


def build_threshold_timeline(today_df):
    """
    Generate today's confidence threshold chart for LR and NN models.

    Placed directly above the TODAY'S PICKS table in the email so the reader
    can see exactly which probability zones trigger a pick.
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

    if today_df is None or today_df.empty:
        return None

    models = [
        ('LR', 'lr_prob_yrfi', 'lr_threshold_low', 'lr_threshold_high', 'lr_confident'),
        ('NN', 'nn_prob_yrfi', 'nn_threshold_low', 'nn_threshold_high', 'nn_confident'),
    ]
    required = [col for _, p, lo, hi, c in models for col in (p, lo, hi, c)]
    if not all(c in today_df.columns for c in required) or 'matchup' not in today_df.columns:
        return None

    try:
        fig, axes = plt.subplots(2, 1, figsize=(10, 4.2), facecolor='white',
                                 gridspec_kw={'hspace': 0.82})
        fig.subplots_adjust(left=0.04, right=0.97, top=0.88, bottom=0.10)

        for ax_idx, (label, prob_col, low_col, high_col, conf_col) in enumerate(models):
            ax = axes[ax_idx]
            ax.set_facecolor('white')

            rows = today_df[today_df[prob_col].notna()].copy()
            if rows.empty:
                ax.set_title(f'{label} Model — no data today', fontsize=9, loc='left',
                             color=COLOR_TEXT, pad=6)
                ax.set_yticks([])
                ax.set_xticks([])
                continue

            probs    = rows[prob_col].astype(float).values
            thr_low  = float(rows[low_col].iloc[0])
            thr_high = float(rows[high_col].iloc[0])

            # X range: cover all dots + thresholds with comfortable padding
            pad   = 0.055
            x_min = max(0.0, min(probs.min(), thr_low)  - pad)
            x_max = min(1.0, max(probs.max(), thr_high) + pad)
            if x_max - x_min < 0.25:
                mid   = (x_min + x_max) / 2
                x_min = max(0.0, mid - 0.125)
                x_max = min(1.0, mid + 0.125)

            ax.set_xlim(x_min, x_max)
            ax.set_ylim(-0.6, 1.6)
            ax.set_yticks([])

            # ── Coloured background zones ─────────────────────────────────────
            ax.axvspan(x_min,    thr_low,  color=COLOR_NRFI, alpha=0.07, zorder=0)
            ax.axvspan(thr_low,  thr_high, color=COLOR_BAND, alpha=0.55, zorder=0)
            ax.axvspan(thr_high, x_max,    color=COLOR_YRFI, alpha=0.07, zorder=0)

            # ── Zone label strip at the top of the band ───────────────────────
            zone_y = 1.42
            for x_left, x_right, txt, col in [
                (x_min,    thr_low,  'NRFI picks', COLOR_NRFI),
                (thr_low,  thr_high, 'no pick',    COLOR_NONE),
                (thr_high, x_max,    'YRFI picks',  COLOR_YRFI),
            ]:
                mid_x = (x_left + x_right) / 2
                # Only draw if zone is wide enough to fit the label
                zone_px = (x_right - x_left) / (x_max - x_min)
                if zone_px > 0.05:
                    ax.text(mid_x, zone_y, txt, ha='center', va='bottom',
                            fontsize=7.5, color=col, style='italic',
                            transform=ax.transData)

            # ── Threshold lines ───────────────────────────────────────────────
            for thr, col in ((thr_low, COLOR_NRFI), (thr_high, COLOR_YRFI)):
                ax.axvline(thr, color=col, linewidth=1.8, linestyle='--',
                           zorder=2, alpha=0.90)
                # Value label on the line, above zone labels
                ax.text(thr, 1.55, f'{thr:.3f}', ha='center', va='bottom',
                        fontsize=8.5, color=col, fontweight='bold',
                        transform=ax.transData)

            # ── Game dots ────────────────────────────────────────────────────
            Y_DOT = 0.5   # all dots on one horizontal strip
            picked_items   = []   # (prob, matchup, color) for label pass
            unpicked_probs = []
            unpicked_cols  = []

            for _, row in rows.iterrows():
                prob       = float(row[prob_col])
                confident  = bool(row.get(conf_col, False))
                matchup    = str(row.get('matchup', ''))

                if prob < thr_low:
                    dot_color = COLOR_NRFI
                elif prob > thr_high:
                    dot_color = COLOR_YRFI
                else:
                    dot_color = COLOR_NONE

                if confident:
                    picked_items.append((prob, matchup, dot_color))
                else:
                    unpicked_probs.append(prob)
                    unpicked_cols.append(dot_color)

            # Unpicked: small hollow circles
            if unpicked_probs:
                ax.scatter(unpicked_probs, [Y_DOT] * len(unpicked_probs),
                           c='none', edgecolors=unpicked_cols, s=28,
                           linewidths=1.2, zorder=3, alpha=0.6)

            # Picked: large filled circles
            if picked_items:
                px = [p for p, _, _ in picked_items]
                pc = [c for _, _, c in picked_items]
                ax.scatter(px, [Y_DOT] * len(picked_items),
                           c=pc, s=90, zorder=5, linewidths=0)

            # ── Matchup labels for picked games ───────────────────────────────
            # Sort by prob; alternate labels above/below the dot strip to
            # reduce overlap when two picks land close together.
            picked_items.sort(key=lambda x: x[0])
            for i, (prob, matchup, col) in enumerate(picked_items):
                above = (i % 2 == 0)
                y_off = 11 if above else -16
                va    = 'bottom' if above else 'top'
                ax.annotate(
                    matchup,
                    xy=(prob, Y_DOT),
                    xytext=(0, y_off),
                    textcoords='offset points',
                    ha='center', va=va,
                    fontsize=7.0, color=col, fontweight='700',
                )

            # ── Title ─────────────────────────────────────────────────────────
            n_nrfi = sum(1 for p, _, _ in picked_items if p < thr_low)
            n_yrfi = sum(1 for p, _, _ in picked_items if p > thr_high)
            pick_str = []
            if n_nrfi:
                pick_str.append(f'{n_nrfi} NRFI')
            if n_yrfi:
                pick_str.append(f'{n_yrfi} YRFI')
            pick_summary = f'  ({", ".join(pick_str)})' if pick_str else '  (no picks today)'

            ax.set_title(
                f'{label} Model — NRFI < {thr_low:.3f}  |  YRFI > {thr_high:.3f}{pick_summary}',
                fontsize=9, fontweight='bold', loc='left',
                color=COLOR_TEXT, pad=6,
            )

            # ── X-axis ────────────────────────────────────────────────────────
            ax.xaxis.set_major_formatter(mticker.FuncFormatter(
                lambda v, _: f'{v:.2f}'
            ))
            ax.xaxis.set_major_locator(mticker.MultipleLocator(0.05))
            ax.tick_params(axis='x', labelsize=7, length=3, pad=3)
            ax.set_xlabel('YRFI probability', fontsize=7.5, labelpad=3)

            for spine in ('top', 'right', 'left'):
                ax.spines[spine].set_visible(False)
            ax.spines['bottom'].set_color('#e5e7eb')

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
