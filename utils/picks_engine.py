"""Single source of truth for the NRFI model + pick-decision logic, imported by both daily_picks.py (live) and scripts/backfill_picks.py (repair) so they can't drift. Pure/offline only: feature prep, LR/NN training + bands, and the decision layer (prob -> pred/confidence/consensus/EV), which is separable from the probabilities so most pick bugs are fixable from stored probs with no retrain."""

import numpy as np
import pandas as pd

# ── Modeling constants (single source — daily_picks.py and the backfill import these) ──
FEATURES = ['away_ops', 'home_ops', 'home_yrfi_pct', 'away_yrfi_pct',
            'home_pitcher_ra', 'home_whip', 'away_pitcher_ra', 'away_whip',
            'park_factor', 'temp']

RECENCY_HALF_LIFE  = 365     # days; games 1yr old carry ~37% weight
RA_CAP             = 1.5     # cap extreme small-sample RA before imputing zeros
MIN_COVERAGE       = 0.10
MAX_COVERAGE       = 0.15
MARGIN_SWEEP       = np.round(np.arange(0.02, 0.131, 0.005), 3)

# LR confidence band: percentile of recent live outputs, falling back to weighted boundary ± CV margin. See DECISIONS.md 2026-06-20 (the "only YRFI lately" fix).
LR_COVERAGE_TARGET = 0.15
LR_POOL_DAYS       = 45
LR_POOL_MIN        = 100

# NN confidence band: percentile of recent live outputs. See DECISIONS.md 2026-06-07.
NN_COVERAGE_TARGET = 0.15
NN_POOL_DAYS       = 45
NN_POOL_MIN        = 100
NN_FALLBACK_MARGIN = 0.045

# NN online (incremental) update — methods A + B + C (see incremental_update).
NN_ONLINE_LR       = 3e-4
NN_ONLINE_MOMENTUM = 0.9
NN_REPLAY_N        = 256
NN_L2SP_LAMBDA     = 1e-3


# ── Small pure helpers ────────────────────────────────────────────────────────
def edge_score(acc, cov):
    return (acc - 0.5) * cov

def ev_per_unit(prob_win, odds):
    """Expected value in units per 1 unit staked given American odds. None if no odds."""
    if odds is None or (isinstance(odds, float) and np.isnan(odds)):
        return None
    try:
        odds = float(odds)
    except (TypeError, ValueError):
        return None
    payout = (100 / abs(odds)) if odds < 0 else (odds / 100)
    return prob_win * payout - (1 - prob_win)

def confident_metrics(probs, actuals, low, high, boundary):
    """Accuracy / count / coverage of the games whose prob falls outside [low, high]."""
    from sklearn.metrics import accuracy_score
    probs   = np.asarray(probs)
    actuals = np.asarray(actuals)
    mask = (probs < low) | (probs > high)
    n = int(mask.sum())
    if n == 0:
        return None, 0, 0.0
    preds = (probs[mask] > boundary).astype(int)
    return accuracy_score(actuals[mask], preds), n, n / len(actuals)

def make_features(d):
    """Select the model FEATURES (in order) from a frame that already has those columns."""
    e = pd.DataFrame(index=d.index)
    for col in FEATURES:
        e[col] = d[col]
    return e[FEATURES]

def load_data(path):
    """Load a CSV from a local path or an s3:// URI."""
    if str(path).startswith('s3://'):
        import boto3, io
        bucket, key = path[5:].split('/', 1)
        obj = boto3.client('s3').get_object(Bucket=bucket, Key=key)
        return pd.read_csv(io.BytesIO(obj['Body'].read()))
    return pd.read_csv(path)


# ── Training data prep ─────────────────────────────────────────────────────────
def april15_filter(df_raw):
    """Drop pre-April-15 games (pitcher RA + YRFI pct unreliable that early)."""
    return df_raw[~((df_raw['month'] < 4) |
                    ((df_raw['month'] == 4) & (df_raw['day'] < 15)))].copy()

def impute_training(df, *, ra_cap=RA_CAP):
    """Impute missing values (returns the frame) and the league averages used. RA=0 / WHIP=0 / YRFI%=0 mean 'no data' (debut / early season) — fill with league medians/means. RA is also capped before the zero-fill so extreme small-sample values don't dominate. Mirrors daily_picks.py PART 1 exactly.

    NaN is treated as the same 'no data' signal as the 0 sentinel. The 2021-2025 base corpus has no NaNs, so a `.replace(0, ...)` alone was sufficient for years — but the Lambda daily files write NaN (not 0) when a stat is unavailable, and ~30% of 2026 rows have a null pitcher_ra. Once the current season is appended to training (2026-08-14) an un-filled NaN reaches LogisticRegression.fit and raises."""
    league = {
        'ra':   float(df[df['away_pitcher_ra'] > 0]['away_pitcher_ra'].median()),
        'whip': float(df[df['home_whip'] > 0]['home_whip'].median()),
        'yrfi': float(df[df['home_yrfi_pct'] > 0]['home_yrfi_pct'].mean()),
        'ops':  float(df['home_ops'].median()),
    }
    df['away_pitcher_ra'] = df['away_pitcher_ra'].clip(upper=ra_cap).replace(0, league['ra']).fillna(league['ra'])
    df['home_pitcher_ra'] = df['home_pitcher_ra'].clip(upper=ra_cap).replace(0, league['ra']).fillna(league['ra'])
    df['away_whip']       = df['away_whip'].replace(0, league['whip']).fillna(league['whip'])
    df['home_whip']       = df['home_whip'].replace(0, league['whip']).fillna(league['whip'])
    df['home_yrfi_pct']   = df['home_yrfi_pct'].replace(0, league['yrfi']).fillna(league['yrfi'])
    df['away_yrfi_pct']   = df['away_yrfi_pct'].replace(0, league['yrfi']).fillna(league['yrfi'])
    df['away_ops']        = df['away_ops'].replace(0, league['ops']).fillna(league['ops'])
    df['home_ops']        = df['home_ops'].replace(0, league['ops']).fillna(league['ops'])
    return df, league

def recency_weights(df, today, *, half_life=RECENCY_HALF_LIFE):
    """exp(-age/half_life), normalized so mean weight = 1. Mirrors daily_picks.py."""
    game_dates = pd.to_datetime(df[['year', 'month', 'day']])
    age_days   = (pd.Timestamp(today) - game_dates).dt.days.values
    w = np.exp(-age_days / half_life)
    return w / w.mean()

def prepare_training(df_raw, today, *, half_life=RECENCY_HALF_LIFE, ra_cap=RA_CAP):
    """Full training prep: April-15 filter → impute → features → recency weights. Returns a dict with df, X_raw, y, sample_weights, league averages, base_rate, boundary. base_rate = UNWEIGHTED y.mean() — the true league YRFI rate (feature-fill default). boundary = recency-WEIGHTED y.mean() — the decision split (the LR is fit with the same weights, so its outputs center here, not on the unweighted rate)."""
    df = april15_filter(df_raw)
    df, league = impute_training(df, ra_cap=ra_cap)
    X_raw = make_features(df).values
    y     = df['YRFI'].values
    sw    = recency_weights(df, today, half_life=half_life)
    return {
        'df': df, 'X_raw': X_raw, 'y': y, 'sample_weights': sw, 'league': league,
        'base_rate': float(y.mean()),
        'boundary':  float(np.average(y, weights=sw)),
    }


# ── Logistic Regression ────────────────────────────────────────────────────────
def train_lr(X_raw, y, sample_weights):
    """Fit StandardScaler + recency-weighted LogisticRegression. Returns (scaler, lr)."""
    from sklearn.preprocessing import StandardScaler
    from sklearn.linear_model import LogisticRegression
    scaler = StandardScaler()
    Xs = scaler.fit_transform(X_raw)
    lr = LogisticRegression(max_iter=500)
    lr.fit(Xs, y, sample_weight=sample_weights)
    return scaler, lr

def lr_cv_margin(X_raw, y, sample_weights, *, sweep=MARGIN_SWEEP,
                 min_cov=MIN_COVERAGE, max_cov=MAX_COVERAGE):
    """5-fold CV edge-score sweep → best band half-width (margin). Mirrors daily_picks PART 2. Returns (margin, cv_acc, cv_cov, cv_boundary). margin defaults to 0.045 if no band lands inside the coverage window."""
    from sklearn.preprocessing import StandardScaler
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import StratifiedKFold
    kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    cv_probs_all, cv_y_all = [], []
    for tr, vl in kf.split(X_raw, y):
        sc = StandardScaler()
        m  = LogisticRegression(max_iter=500).fit(
            sc.fit_transform(X_raw[tr]), y[tr], sample_weight=sample_weights[tr])
        cv_probs_all.append(m.predict_proba(sc.transform(X_raw[vl]))[:, 1])
        cv_y_all.append(y[vl])
    cv_probs    = np.concatenate(cv_probs_all)
    cv_y        = np.concatenate(cv_y_all)
    cv_boundary = float(cv_y.mean())
    rows = []
    for mgn in sweep:
        acc, n, cov = confident_metrics(
            cv_probs, cv_y, round(cv_boundary - mgn, 4), round(cv_boundary + mgn, 4), cv_boundary)
        if acc is not None:
            rows.append((mgn, acc, n, cov))
    eligible = [(mgn, a, n, c) for mgn, a, n, c in rows if min_cov <= c <= max_cov]
    best = max(eligible, key=lambda r: edge_score(r[1], r[3])) if eligible else None
    margin = best[0] if best else 0.045
    return float(margin), (best[1] if best else 0.0), (best[3] if best else 0.0), cv_boundary

def lr_band(boundary, margin, lr_pool, *, coverage_target=LR_COVERAGE_TARGET,
            pool_min=LR_POOL_MIN):
    """Recenter the LR boundary + band on the LIVE output distribution (the YRFI-only fix). When ≥ pool_min recent live `lr_prob_yrfi` are pooled, override: boundary = pool median, low/high = P(tail)/P(1-tail) at coverage_target. Otherwise fall back to the static weighted boundary ± CV margin. Returns dict(boundary, low, high, regime, coverage, n_pool). Mirrors daily_picks PART 2."""
    pool = np.asarray(lr_pool, dtype=float)
    pool = pool[~np.isnan(pool)]
    tail = coverage_target / 2.0
    if len(pool) >= pool_min:
        b   = float(np.median(pool))
        low  = round(float(np.percentile(pool, tail * 100)), 4)
        high = round(float(np.percentile(pool, (1 - tail) * 100)), 4)
        cov  = float((pool < low).mean() + (pool > high).mean())
        return {'boundary': b, 'low': low, 'high': high,
                'regime': 'percentile', 'coverage': cov, 'n_pool': len(pool)}
    return {'boundary': float(boundary),
            'low':  round(boundary - margin, 3),
            'high': round(boundary + margin, 3),
            'regime': 'static', 'coverage': None, 'n_pool': len(pool)}


# ── Neural Network ─────────────────────────────────────────────────────────────
def build_nn(input_dim):
    """8→8→8→1, sigmoid out, Adam lr=0.005. l2=1e-5, dropout=0 (see DECISIONS 2026-06-07). Must match daily_picks.py._build_nn exactly."""
    import tensorflow as tf
    reg = tf.keras.regularizers.l2(1e-5)
    model = tf.keras.Sequential([
        tf.keras.layers.Dense(8, activation='relu', input_shape=(input_dim,), kernel_regularizer=reg),
        tf.keras.layers.Dropout(0.0),
        tf.keras.layers.Dense(8, activation='relu', kernel_regularizer=reg),
        tf.keras.layers.Dropout(0.0),
        tf.keras.layers.Dense(8, activation='relu', kernel_regularizer=reg),
        tf.keras.layers.Dropout(0.0),
        tf.keras.layers.Dense(1, activation='sigmoid'),
    ])
    model.compile(optimizer=tf.keras.optimizers.Adam(0.005), loss='binary_crossentropy')
    return model

def incremental_update(model, X_new, y_new, X_hist, y_hist, *,
                       replay_n=NN_REPLAY_N, lr=NN_ONLINE_LR,
                       momentum=NN_ONLINE_MOMENTUM, l2sp=NN_L2SP_LAMBDA,
                       epochs=1, seed=None):
    """A+B+C online update: streaming SGD + uniform experience replay + L2-SP anchor. Trains in place; returns rows trained on. Must match daily_picks.py._incremental_update."""
    import tensorflow as tf
    rng = np.random.default_rng(seed)
    if replay_n and len(y_hist) > 0:
        idx   = rng.choice(len(y_hist), size=min(replay_n, len(y_hist)), replace=False)
        X_mix = np.concatenate([X_hist[idx], X_new], axis=0)
        y_mix = np.concatenate([np.asarray(y_hist)[idx], np.asarray(y_new)], axis=0)
    else:
        X_mix, y_mix = X_new, np.asarray(y_new)
    X_t = tf.constant(X_mix, dtype=tf.float32)
    y_t = tf.constant(y_mix.reshape(-1, 1), dtype=tf.float32)
    kernels = [v for v in model.trainable_variables if 'kernel' in v.name]
    anchors = [tf.constant(k.numpy()) for k in kernels]
    opt = tf.keras.optimizers.SGD(learning_rate=lr, momentum=momentum)
    bce = tf.keras.losses.BinaryCrossentropy()
    n, bs = int(X_t.shape[0]), 64
    for _ in range(epochs):
        perm = tf.random.shuffle(tf.range(n))
        for s in range(0, n, bs):
            b      = perm[s:s + bs]
            xb, yb = tf.gather(X_t, b), tf.gather(y_t, b)
            with tf.GradientTape() as tape:
                loss = bce(yb, model(xb, training=True))
                loss = loss + l2sp * tf.add_n(
                    [tf.reduce_sum(tf.square(k - a)) for k, a in zip(kernels, anchors)])
            grads = tape.gradient(loss, model.trainable_variables)
            opt.apply_gradients(zip(grads, model.trainable_variables))
    return n

def nn_band(nn_pool, calibrated_boundary, *, coverage_target=NN_COVERAGE_TARGET,
            pool_min=NN_POOL_MIN, fallback_margin=NN_FALLBACK_MARGIN):
    """Percentile band on recent live NN outputs, else boundary ± fallback_margin. Returns dict(low, high, regime, coverage, n_pool). Mirrors daily_picks PART 1 NN band."""
    pool = np.asarray(nn_pool, dtype=float)
    pool = pool[~np.isnan(pool)]
    tail = coverage_target / 2.0
    if len(pool) >= pool_min:
        low  = round(float(np.percentile(pool, tail * 100)), 4)
        high = round(float(np.percentile(pool, (1 - tail) * 100)), 4)
        cov  = float((pool < low).mean() + (pool > high).mean())
        return {'low': low, 'high': high, 'regime': 'percentile',
                'coverage': cov, 'n_pool': len(pool)}
    return {'low':  round(calibrated_boundary - fallback_margin, 3),
            'high': round(calibrated_boundary + fallback_margin, 3),
            'regime': 'static', 'coverage': None, 'n_pool': len(pool)}


# ── Decision layer (the part the backfill recomputes from stored probabilities) ──
def decision_columns(probs_yrfi, boundary, low, high):
    """From P(YRFI) + the model's boundary/band, derive the per-game decision columns. Returns dict of equal-length arrays: prob_yrfi, prob_nrfi, pred ('YRFI'/'NRFI'), conf (winning-side prob), confident (bool). Direction is split at `boundary`; `confident` is prob outside [low, high]. This is the single definition used by BOTH the live run and the backfill."""
    p = np.asarray(probs_yrfi, dtype=float)
    pred      = np.where(p > boundary, 'YRFI', 'NRFI')
    conf      = np.where(p > boundary, p, 1 - p)
    confident = (p < low) | (p > high)
    return {'prob_yrfi': p, 'prob_nrfi': 1 - p, 'pred': pred,
            'conf': conf, 'confident': confident}

def consensus_mask(lr_probs, lr_boundary, lr_confident, nn_probs, nn_boundary, nn_confident):
    """Both models confident AND agreeing on direction (each vs its OWN boundary)."""
    lr_yrfi = np.asarray(lr_probs, dtype=float) > lr_boundary
    nn_yrfi = np.asarray(nn_probs, dtype=float) > nn_boundary
    return (np.asarray(lr_confident, dtype=bool)
            & np.asarray(nn_confident, dtype=bool)
            & (lr_yrfi == nn_yrfi))

def ev_for_rows(probs_yrfi, preds, nrfi_odds, yrfi_odds):
    """Per-row EV in units for the side each model predicts. None where odds are missing."""
    out = []
    for p, pred, no, yo in zip(probs_yrfi, preds, nrfi_odds, yrfi_odds):
        if pred == 'NRFI':
            ev = ev_per_unit(1 - p, no)
        else:
            ev = ev_per_unit(p, yo)
        out.append(None if ev is None else round(ev, 4))
    return out

def correctness(preds, actual_yrfi):
    """1 if pred matches the actual outcome, 0 if not, None when the actual is unknown."""
    out = []
    for pred, act in zip(preds, actual_yrfi):
        if act is None or (isinstance(act, float) and np.isnan(act)):
            out.append(None)
        else:
            out.append(int(('YRFI' if int(act) == 1 else 'NRFI') == pred))
    return out
