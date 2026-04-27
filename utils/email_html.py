"""
utils/email_html.py
-------------------
build_email_html  — full daily email HTML
send_email        — SES delivery (with optional inline chart image)
"""

import os
from datetime import datetime, timedelta

import numpy as np
import pandas as pd

from utils.pl_calc import compute_pl


def _odds_str(val):
    if val is None: return '—'
    return f'+{val}' if val > 0 else str(val)

def _pl_str(pl):
    if pl is None: return '—'
    col  = '#16a34a' if pl >= 0 else '#dc2626'
    sign = '+' if pl >= 0 else ''
    return f'<span style="color:{col};font-weight:700">{sign}${pl:.2f}</span>'


def build_email_html(date_str, picks_rows, yesterday_rows, ytd_df, today_df_all,
                     lr_threshold, nn_threshold, cv_acc, cv_cov,
                     *, yesterday, today, unit, get_odds_fn=None):
    """
    Full daily email HTML.

    Extra keyword args (must be passed explicitly):
      yesterday    — date object for yesterday
      today        — date object for today
      unit         — dollar value per unit (e.g. 10)
      get_odds_fn  — callable(matchup) -> (nrfi_odds, yrfi_odds) | None
                     Pass None to suppress live odds lookup (shows — in table)
    """
    G    = '#2563eb'
    R    = '#7c3aed'
    WIN  = '#16a34a'
    LOSS = '#dc2626'
    MUT  = '#6b7280'
    BDR  = '#e5e7eb'
    TXT  = '#111827'

    def section(title, content):
        return (f'<div style="margin-bottom:28px">'
                f'<div style="font-size:11px;font-weight:700;letter-spacing:1.5px;'
                f'text-transform:uppercase;color:{MUT};border-bottom:2px solid {TXT};'
                f'padding-bottom:6px;margin-bottom:16px">{title}</div>'
                f'{content}</div>')

    def th(label, right=False):
        align = 'right' if right else 'left'
        return (f'<th style="padding:6px 10px;text-align:{align};font-size:11px;'
                f'font-weight:600;color:{MUT};border-bottom:1px solid {BDR};'
                f'white-space:nowrap">{label}</th>')

    def td(val, right=False, bold=False, color=None):
        align = 'right' if right else 'left'
        fw    = '600' if bold else '400'
        col   = f'color:{color};' if color else ''
        return (f'<td style="padding:7px 10px;text-align:{align};font-size:13px;'
                f'{col}font-weight:{fw};border-bottom:1px solid {BDR};'
                f'white-space:nowrap">{val}</td>')

    def ev_display(ev_units, ev_dollars):
        try:
            if ev_units is None or (isinstance(ev_units, float) and np.isnan(ev_units)):
                return '—'
            sign = '+' if ev_units >= 0 else ''
            col  = WIN if ev_units >= 0 else LOSS
            return f'<span style="color:{col}">{sign}{ev_units:.3f}u (${ev_dollars:+.2f})</span>'
        except Exception:
            return '—'

    # ── Yesterday's results ───────────────────────────────────────────────────
    if yesterday_rows is not None and not yesterday_rows.empty:
        conf = yesterday_rows[yesterday_rows['lr_confident'] | yesterday_rows['nn_confident']]
        rows_html = ''; total_pl = 0.0
        for _, r in conf.sort_values('lr_conf', ascending=False).iterrows():
            result_label = result_color = None
            pl_val = None
            models_used = []
            for m, pred_col, correct_col in [('LR','lr_pred','lr_correct'),('NN','nn_pred','nn_correct')]:
                if not r.get(f'{m.lower()}_confident', False): continue
                models_used.append(m)
                correct = r[correct_col]
                if pd.notna(correct):
                    pl_val = compute_pl(correct, r[pred_col], r.get('nrfi_odds'), r.get('yrfi_odds'))
                    if pl_val is not None: total_pl += pl_val
                    result_label = 'WIN' if int(correct) == 1 else 'LOSS'
                    result_color = WIN if int(correct) == 1 else LOSS
                else:
                    result_label = 'Pending'; result_color = MUT
            if result_label is None: continue
            pred       = r.get('lr_pred','—') if 'LR' in models_used else r.get('nn_pred','—')
            pred_color = G if pred == 'NRFI' else R
            nn_first   = sorted(models_used, key=lambda x: 0 if x == 'NN' else 1)
            model_str  = ','.join(nn_first)
            picked_odds = _odds_str(r.get('nrfi_odds') if pred == 'NRFI' else r.get('yrfi_odds'))
            pl_str      = _pl_str(pl_val) if pl_val is not None else '—'
            matchup_disp = r['matchup'].replace('Athletics', 'ATH')
            rows_html += (
                f'<tr>'
                + td(matchup_disp, bold=True)
                + td(f'<span style="color:{result_color};font-weight:700">{result_label}</span>')
                + td(f'<span style="color:{pred_color};font-weight:600">{pred}</span>')
                + td(model_str, color=MUT)
                + td(f'{r.get("lr_conf",0):.1%}', right=True)
                + td(picked_odds, right=True, color=MUT)
                + td(pl_str, right=True)
                + '</tr>'
            )
        yest_tbl = (
            f'<table style="width:100%;border-collapse:collapse">'
            f'<tr>{th("Matchup")}{th("Result")}{th("Pick")}{th("Model")}'
            f'{th("Conf",True)}{th("Odds",True)}{th("P/L",True)}</tr>'
            + rows_html
            + f'<tr><td colspan="6" style="padding:6px 10px;font-size:12px;color:{MUT}">Total</td>'
            + td(_pl_str(total_pl), right=True) + '</tr></table>'
        )
        _n_pending = sum(1 for _, r in conf.iterrows() if pd.isna(r.get('nrfi_odds')))
        odds_note = (f'<div style="font-size:11px;color:{MUT};margin-top:8px">'
                     f'* {_n_pending} pick{"s" if _n_pending!=1 else ""} missing odds — '
                     f'P/L excluded until backfilled</div>'
                     if _n_pending else '')
        yest_section = section(
            f"Yesterday's Results &mdash; {yesterday.strftime('%B')} {yesterday.day}",
            yest_tbl + odds_note
        )
    else:
        yest_section = section("Yesterday's Results",
            f'<span style="color:{MUT}">No picks on record for yesterday.</span>')

    # ── YTD performance ───────────────────────────────────────────────────────
    if ytd_df is not None and not ytd_df.empty:
        lr_ytd  = ytd_df[ytd_df['lr_confident'] & ytd_df['lr_correct'].notna()]
        nn_ytd  = ytd_df[ytd_df['nn_confident'] & ytd_df['nn_correct'].notna()]
        con_ytd = ytd_df[ytd_df['consensus']    & ytd_df['lr_correct'].notna()]

        def ytd_row(label, subset, correct_col, pred_col):
            if subset.empty:
                return f'<tr>{td(label,bold=True)}{td("0-0")}{td("—")}{td("—")}{td("—")}</tr>'
            w = int(subset[correct_col].sum()); l = len(subset) - w
            pct = w / (w + l)
            pl_vals = [
                compute_pl(r[correct_col], r[pred_col], r.get('nrfi_odds'), r.get('yrfi_odds'))
                for _, r in subset.iterrows() if pd.notna(r[correct_col])
            ]
            graded_pl = [v for v in pl_vals if v is not None]
            pending   = len(pl_vals) - len(graded_pl)
            pl_sum    = sum(graded_pl) if graded_pl else 0.0
            pl_col    = WIN if pl_sum >= 0 else LOSS
            acc_col   = WIN if pct > 0.5 else LOSS if pct < 0.5 else MUT
            pl_disp   = f'<span style="color:{pl_col};font-weight:600">{"+" if pl_sum>=0 else ""}${pl_sum:.2f}</span>'
            if pending:
                pl_disp += f'<span style="color:{MUT};font-size:11px"> ({pending} pending)</span>'
            return (f'<tr>'
                    + td(label, bold=True)
                    + td(f'{w}-{l}')
                    + td(f'<span style="color:{acc_col};font-weight:600">{pct:.1%}</span>', right=True)
                    + td(f'{len(subset)/max(len(ytd_df),1):.1%}', right=True, color=MUT)
                    + td(pl_disp, right=True)
                    + '</tr>')

        ytd_tbl = (f'<table style="width:100%;border-collapse:collapse">'
                   f'<tr>{th("Model")}{th("Record")}{th("Acc",True)}{th("Cov",True)}{th("P/L",True)}</tr>'
                   + ytd_row('LR',        lr_ytd,  'lr_correct', 'lr_pred')
                   + ytd_row('NN',        nn_ytd,  'nn_correct', 'nn_pred')
                   + ytd_row('Consensus', con_ytd, 'lr_correct', 'lr_pred')
                   + '</table>')
        ytd_section = section(f'{today.year} Season', ytd_tbl)
    else:
        ytd_section = ''

    # ── Helper: format pitcher name as "F. Last" ─────────────────────────────
    def fmt_pitcher(name):
        if not name or str(name).strip() in ('', 'TBD', 'nan', 'None'):
            return '—'
        parts = str(name).strip().split()
        if len(parts) == 1:
            return parts[0]
        return f'{parts[0][0]}. {" ".join(parts[1:])}'

    # ── Helper: build a single-row game block ─────────────────────────────────
    def game_rows(r, picked=False, pick_info=None):
        matchup = r['matchup'].replace('Athletics', 'ATH')
        lrc  = G if r['lr_pred'] == 'NRFI' else R
        nnc  = G if r['nn_pred'] == 'NRFI' else R
        lrfw = '700' if r['lr_confident'] else '400'
        nnfw = '700' if r['nn_confident'] else '400'

        if get_odds_fn is not None:
            bo = get_odds_fn(matchup)
            yrfi_odds_s = _odds_str(bo[1]) if bo else '—'
            nrfi_odds_s = _odds_str(bo[0]) if bo else '—'
        else:
            nrfi_odds_s = _odds_str(r.get('nrfi_odds'))
            yrfi_odds_s = _odds_str(r.get('yrfi_odds'))
        odds_cell = (f'<span style="color:{G}">NRFI</span> {nrfi_odds_s}'
                     f' / <span style="color:{R}">YRFI</span> {yrfi_odds_s}')

        a_pitcher   = fmt_pitcher(r.get('away_pitcher', ''))
        h_pitcher   = fmt_pitcher(r.get('home_pitcher', ''))
        starter_str = f'{a_pitcher} v. {h_pitcher}'

        if picked and pick_info:
            pick      = pick_info.get('pick', '—')
            model_str = pick_info.get('model_str', '—')
            pick_col  = G if pick == 'NRFI' else R
            pick_html = f'<span style="color:{pick_col};font-weight:700">{pick}</span>'
        else:
            pick_html = f'<span style="color:{MUT}">—</span>'
            model_str = '—'

        return (
            f'<tr style="border-bottom:1px solid {BDR}">'
            + td(f'<span style="font-weight:{"700" if picked else "400"}">{matchup}</span>')
            + td(pick_html)
            + td(model_str, color=MUT)
            + td(f'<span style="color:{MUT};font-style:italic;font-size:12px">{starter_str}</span>')
            + td(f'<span style="color:{nnc};font-weight:{nnfw}">{r["nn_prob_yrfi"]:.0%}</span>', right=True)
            + td(f'<span style="color:{lrc};font-weight:{lrfw}">{r["lr_prob_yrfi"]:.0%}</span>', right=True)
            + f'<td style="padding:7px 10px;text-align:right;font-size:12px;'
              f'color:{MUT};white-space:nowrap;border-bottom:1px solid {BDR}">'
              f'{odds_cell}</td>'
            + '</tr>'
        )

    def games_table(rows_html, header=True):
        hdr = (f'<tr>{th("Matchup")}{th("Pick")}{th("Model")}{th("Starter")}'
               f'{th("NN YRFI%", True)}{th("LR YRFI%", True)}{th("Odds", True)}</tr>'
               ) if header else ''
        return f'<table style="width:100%;border-collapse:collapse">{hdr}{rows_html}</table>'

    # ── Build picked-matchup lookup from picks_rows ───────────────────────────
    pick_meta = {}
    for p in picks_rows:
        m = p['matchup']
        if m not in pick_meta:
            pick_meta[m] = {'models': [], 'pick': p.get('prediction', '—')}
        pick_meta[m]['models'].append(p.get('model', ''))
    for m, info in pick_meta.items():
        models = info['models']
        if 'NN' in models and 'LR' in models:
            info['model_str'] = 'NN,LR'
        elif 'NN' in models:
            info['model_str'] = 'NN'
        else:
            info['model_str'] = 'LR'

    # ── Split today_df_all into picked vs not-picked ──────────────────────────
    if today_df_all is not None and not today_df_all.empty:
        df_sorted  = today_df_all.sort_values('lr_conf', ascending=False)
        df_picked  = df_sorted[df_sorted['lr_confident'] | df_sorted['nn_confident']]
        df_unpicked = df_sorted[~(df_sorted['lr_confident'] | df_sorted['nn_confident'])]

        picked_rows_html = ''.join(
            game_rows(r, picked=True, pick_info=pick_meta.get(r['matchup']))
            for _, r in df_picked.iterrows()
        )
        unpicked_rows_html = ''.join(
            game_rows(r, picked=False)
            for _, r in df_unpicked.iterrows()
        )

        if picked_rows_html:
            picks_section = section("Today's Picks", games_table(picked_rows_html))
        else:
            picks_section = section("Today's Picks",
                f'<span style="color:{MUT}">No games cleared the threshold today.</span>')

        not_picked_section = (section('Not Picked', games_table(unpicked_rows_html))
                              if unpicked_rows_html else '')
    else:
        picks_section      = ''
        not_picked_section = ''

    # ── Assemble ──────────────────────────────────────────────────────────────
    u = datetime.utcnow()
    offset  = -4 if 3 <= u.month <= 10 else -5
    tz_name = 'EDT' if offset == -4 else 'EST'
    gen_time = (u + timedelta(hours=offset)).strftime('%H:%M')

    body = f"""<!DOCTYPE html>
<html><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<style>
  body {{ margin:0;padding:0;background:#ffffff;
         font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',Arial,sans-serif;
         color:{TXT}; }}
  * {{ box-sizing:border-box; }}
</style></head>
<body>
<div style="max-width:680px;margin:0 auto;padding:28px 20px">

  <div style="padding-bottom:14px;margin-bottom:28px">
    <div style="font-size:11px;font-weight:700;letter-spacing:2px;text-transform:uppercase;color:{MUT}">NRFI Daily</div>
    <div style="font-size:26px;font-weight:800;margin-top:4px">{today.strftime('%A, %B')} {today.day}</div>
  </div>

  {yest_section}
  {ytd_section}
  <!-- THRESHOLD_CHART_PLACEHOLDER -->
  {picks_section}
  {not_picked_section}

  <div style="margin-top:32px;padding-top:12px;border-top:1px solid {BDR};
              font-size:11px;color:{MUT};text-align:center">
    Generated {gen_time} {tz_name} &nbsp;&middot;&nbsp; 1u = ${unit}
  </div>
</div>
</body></html>"""
    return body


def send_email(html_body, subject, date_str, chart_bytes=None):
    """Send daily picks email via SES. Recipients from NRFI_SES_FROM / NRFI_SES_TO env vars."""
    ses_from = os.environ.get('NRFI_SES_FROM') or os.environ.get('SES_FROM')
    ses_to   = os.environ.get('NRFI_SES_TO')  or os.environ.get('SES_TO', '')
    if not ses_from or not ses_to:
        print('  Email skipped: SES_FROM / SES_TO not set')
        return
    recipients = [a.strip() for a in ses_to.split(',') if a.strip()]
    if not recipients:
        return
    try:
        import json, subprocess
        import boto3
        from email.mime.image     import MIMEImage
        from email.mime.multipart import MIMEMultipart
        from email.mime.text      import MIMEText
        # Try standard credential chain first; fall back to aws cli credential process
        _session = boto3.Session()
        if _session.get_credentials() is None:
            _r = subprocess.run(
                ['aws', 'configure', 'export-credentials', '--format', 'process'],
                capture_output=True, text=True, timeout=10,
            )
            _c = json.loads(_r.stdout)
            _session = boto3.Session(
                aws_access_key_id=_c['AccessKeyId'],
                aws_secret_access_key=_c['SecretAccessKey'],
                aws_session_token=_c.get('SessionToken'),
                region_name='us-east-1',
            )
        ses = _session.client('ses', region_name='us-east-1')
        if chart_bytes:
            chart_img_html = (
                '<div style="margin-bottom:28px">'
                '<img src="cid:threshold_chart" alt="7-day threshold timeline" '
                'style="max-width:100%;height:auto;display:block"></div>'
            )
            html_with_chart = html_body.replace('<!-- THRESHOLD_CHART_PLACEHOLDER -->', chart_img_html)
            msg = MIMEMultipart('related')
            msg['Subject'] = subject
            msg['From']    = ses_from
            msg['To']      = ', '.join(recipients)
            alt = MIMEMultipart('alternative')
            alt.attach(MIMEText(html_with_chart, 'html', 'utf-8'))
            msg.attach(alt)
            img = MIMEImage(chart_bytes, 'png')
            img.add_header('Content-ID', '<threshold_chart>')
            img.add_header('Content-Disposition', 'inline', filename='threshold_chart.png')
            msg.attach(img)
            ses.send_raw_email(Source=ses_from, Destinations=recipients,
                               RawMessage={'Data': msg.as_bytes()})
        else:
            html_clean = html_body.replace('<!-- THRESHOLD_CHART_PLACEHOLDER -->', '')
            ses.send_email(
                Source=ses_from,
                Destination={'ToAddresses': recipients},
                Message={'Subject': {'Data': subject},
                         'Body':    {'Html': {'Data': html_clean, 'Charset': 'UTF-8'}}},
            )
        print(f'  Email sent to {", ".join(recipients)}')
    except Exception as ex:
        print(f'  WARNING: SES send failed ({ex})')
