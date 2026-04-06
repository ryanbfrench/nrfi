import json
import re
import boto3
import datetime
import requests
import statsapi
import pandas as pd
from unidecode import unidecode


def lambda_handler(event, context):
    result = main()
    if isinstance(result, str):
        return {'statusCode': 200, 'body': json.dumps(result)}
    df, date = result
    csv_string = df.to_csv(index=False)
    bucket_name = "nrfi-store"
    s3_path = "data/" + str(date.year) + "/" + str(date.month) + "/" + str(date.day) + ".txt"
    s3 = boto3.resource("s3")
    s3.Bucket(bucket_name).put_object(Key=s3_path, Body=csv_string)
    return {
        'statusCode': 200,
        'body': json.dumps(f'{date} — {len(df)} games saved to s3://{bucket_name}/{s3_path}')
    }


def pct_to_float(val):
    try:
        s = str(val).strip().rstrip('%')
        return None if s in ('--', 'nan', '') else round(float(s) / 100, 4)
    except Exception:
        return None


def lineup_to_ops(lineup, ops_df):
    lineup_ops = []
    for batter in lineup:
        try:
            batter_ops = ops_df[ops_df['cleanedName'] == unidecode(batter)]['OPS'].item()
            lineup_ops.append(batter_ops)
        except Exception:
            continue
    return round(sum(lineup_ops) / len(lineup_ops), 3) if lineup_ops else None


def get_whip(pitcher_name, whip_df):
    whip = whip_df[whip_df['cleanedName'] == unidecode(pitcher_name)]['WHIP']
    return whip.item() if len(whip) == 1 else None


def get_pitcher_ra(pitcher_name, pitcher_runs_df):
    ra = pitcher_runs_df[pitcher_runs_df['cleanedName'] == unidecode(pitcher_name)]['RA']
    return ra.item() if len(ra) == 1 else None


def api_name(person_id):
    return statsapi.get('people', {'personIds': person_id})['people'][0]['fullName']


def get_lineup(boxscore, home_away):
    hitters, i, lineup = 0, 1, []
    batters = boxscore[home_away + 'Batters']
    while hitters < 4 and i < len(batters):
        if not batters[i].get('substitution', True):
            hitters += 1
            lineup.append(api_name(batters[i]['personId']))
        i += 1
    return lineup


def parse_weather(weather_str, first_pitch_str):
    ws = str(weather_str).lower()
    m = re.search(r'(\d+)\s+degree', ws)
    temp   = int(m.group(1)) if m else 65
    dome   = 1 if any(w in ws for w in ['dome', 'indoor', 'retractable', 'roof closed']) else 0
    rain   = 1 if any(w in ws for w in ['rain', 'drizzle', 'shower']) else 0
    clear  = 1 if any(w in ws for w in ['sunny', 'clear']) else 0
    cloudy = 1 if any(w in ws for w in ['cloud', 'overcast']) else 0
    day_night = 1
    tm = re.search(r'(\d+):(\d+)\s*(AM|PM)', str(first_pitch_str), re.IGNORECASE)
    if tm:
        hour, ampm = int(tm.group(1)), tm.group(3).upper()
        hour24 = (hour if hour == 12 else hour + 12) if ampm == 'PM' else (0 if hour == 12 else hour)
        day_night = 0 if hour24 < 17 else 1
    return temp, day_night, clear, cloudy, rain, dome


def get_yrfi_split(team_pct, team_name, split_col, year_col):
    row = team_pct[team_pct['Team'] == team_name]
    if row.empty:
        return None
    v = pct_to_float(row[split_col].item())
    if v is None and year_col in team_pct.columns:
        v = pct_to_float(row[year_col].item())
    return v


def main():
    team_abbvs = {
        'PHI': {'team_pct': 'Philadelphia', 'normal': 'PHI'},
        'SF':  {'team_pct': 'SF Giants',    'normal': 'SF'},
        'TEX': {'team_pct': 'Texas',         'normal': 'TEX'},
        'BOS': {'team_pct': 'Boston',        'normal': 'BOS'},
        'KC':  {'team_pct': 'Kansas City',   'normal': 'KC'},
        'DET': {'team_pct': 'Detroit',       'normal': 'DET'},
        'NYY': {'team_pct': 'NY Yankees',    'normal': 'NYY'},
        'TB':  {'team_pct': 'Tampa Bay',     'normal': 'TB'},
        'TOR': {'team_pct': 'Toronto',       'normal': 'TOR'},
        'PIT': {'team_pct': 'Pittsburgh',    'normal': 'PIT'},
        'OAK': {'team_pct': 'Sacramento',    'normal': 'OAK'},
        'ATH': {'team_pct': 'Sacramento',    'normal': 'OAK'},
        'BAL': {'team_pct': 'Baltimore',     'normal': 'BAL'},
        'WSH': {'team_pct': 'Washington',    'normal': 'WAS'},
        'NYM': {'team_pct': 'NY Mets',       'normal': 'NYM'},
        'MIN': {'team_pct': 'Minnesota',     'normal': 'MIN'},
        'CWS': {'team_pct': 'Chi Sox',       'normal': 'CHW'},
        'SEA': {'team_pct': 'Seattle',       'normal': 'SEA'},
        'CLE': {'team_pct': 'Cleveland',     'normal': 'CLE'},
        'CHC': {'team_pct': 'Chi Cubs',      'normal': 'CHC'},
        'STL': {'team_pct': 'St. Louis',     'normal': 'STL'},
        'MIA': {'team_pct': 'Miami',         'normal': 'MIA'},
        'ATL': {'team_pct': 'Atlanta',       'normal': 'ATL'},
        'MIL': {'team_pct': 'Milwaukee',     'normal': 'MIL'},
        'AZ':  {'team_pct': 'Arizona',       'normal': 'ARI'},
        'HOU': {'team_pct': 'Houston',       'normal': 'HOU'},
        'LAA': {'team_pct': 'LA Angels',     'normal': 'LAA'},
        'SD':  {'team_pct': 'San Diego',     'normal': 'SD'},
        'LAD': {'team_pct': 'LA Dodgers',    'normal': 'LAD'},
        'CIN': {'team_pct': 'Cincinnati',    'normal': 'CIN'},
        'COL': {'team_pct': 'Colorado',      'normal': 'COL'},
    }

    park_factor_dict = {
        'COL': 112, 'CIN': 111, 'BOS': 109, 'LAA': 104, 'PHI': 104,
        'KC':  103, 'CHW': 102, 'LAD': 102, 'BAL': 101, 'ARI': 101,
        'PIT': 100, 'MIL': 100, 'SF':  100, 'ATL': 100, 'WAS': 100,
        'CLE':  99, 'TOR':  99, 'MIA':  99, 'TEX':  99, 'NYY':  99,
        'CHC':  98, 'HOU':  98, 'MIN':  98, 'DET':  97, 'TB':   96,
        'NYM':  96, 'STL':  95, 'OAK':  94, 'SD':   94, 'SEA':  91,
    }

    date = datetime.date.today()
    yesterday = date - datetime.timedelta(1)
    y = yesterday.year

    all_star_breaks = [
        '2021-07-12', '2021-07-13', '2021-07-14', '2021-07-15',
        '2022-07-18', '2022-07-19', '2022-07-20', '2022-07-21',
        '2023-07-10', '2023-07-11', '2023-07-12', '2023-07-13',
    ]
    if str(date) in all_star_breaks:
        return 'All Star Break'

    date_minus_thirty_days = date - datetime.timedelta(30)
    date_minus_sixty_days  = date - datetime.timedelta(60)
    season_start_date = datetime.date(y, 3, 1)

    # ── Fetch external stats ──────────────────────────────────────────────────
    ops_url = (
        f'https://www.fangraphs.com/api/leaders/major-league/data'
        f'?age=&pos=all&stats=bat&lg=all&qual=1&season={y}&season1={y}'
        f'&startdate={date_minus_thirty_days}&enddate={yesterday}&month=1000'
        f'&hand=&team=0&pageitems=2000000000&pagenum=1&ind=0&rost=0&players='
        f'&type=8&postseason=&sortdir=default&sortstat=WAR'
    )
    ops_df = None
    try:
        r = requests.get(ops_url, timeout=30)
        ops_df = pd.json_normalize(r.json()['data'])[['PlayerName', 'OPS']]
        ops_df['cleanedName'] = ops_df['PlayerName'].apply(unidecode)
    except Exception as e:
        print(f'Could not load OPS df: {e}')

    whip_url = (
        f'https://www.fangraphs.com/api/leaders/major-league/data'
        f'?age=0&pos=all&stats=pit&lg=all&qual=1&season={y}&season1={y}'
        f'&startdate={date_minus_sixty_days}&enddate={yesterday}&month=1000'
        f'&hand=&team=0&pageitems=100000&pagenum=1&ind=0&rost=0&players=0'
        f'&type=1&postseason=&sortdir=default&sortstat=SIERA&qual=1'
    )
    whip_df = None
    try:
        r = requests.get(whip_url, timeout=30)
        whip_df = pd.json_normalize(r.json()['data'])[['PlayerName', 'WHIP']]
        whip_df['cleanedName'] = whip_df['PlayerName'].apply(unidecode)
    except Exception as e:
        print(f'Could not load WHIP df: {e}')

    payload = {
        'strPlayerId': 'all', 'strSplitArr': [44], 'strGroup': 'season',
        'strPosition': 'P', 'strType': '1',
        'strStartDate': str(season_start_date), 'strEndDate': str(yesterday),
        'strSplitTeams': False,
        'dctFilters': [],
        'strStatType': 'player', 'strAutoPt': 'true', 'arrPlayerId': [],
        'strSplitArrPitch': [], 'arrWxTemperature': None, 'arrWxPressure': None,
        'arrWxAirDensity': None, 'arrWxElevation': None, 'arrWxWindSpeed': None,
    }
    pitcher_runs_df = None
    try:
        r = requests.post(
            'https://www.fangraphs.com/api/leaders/splits/splits-leaders',
            json=payload, timeout=30
        )
        _pdf = pd.json_normalize(r.json()['data'])
        if 'R' in _pdf.columns and 'G' in _pdf.columns and 'playerName' in _pdf.columns:
            _pdf['RA'] = _pdf['R'] / _pdf['G']
            _pdf['cleanedName'] = _pdf['playerName'].apply(unidecode)
            pitcher_runs_df = _pdf
        else:
            print(f'Pitcher runs df missing expected columns: {list(_pdf.columns[:10])}')
    except Exception as e:
        print(f'Could not load pitcher runs df: {e}')

    team_pct = pd.read_html(
        f'https://www.teamrankings.com/mlb/stat/yes-run-first-inning-pct?date={yesterday}'
    )[0]
    year_col = str(y)

    # ── Game loop ─────────────────────────────────────────────────────────────
    INCOMPLETE_STATUSES = {'Preview', 'Pre-Game', 'Warmup', 'Postponed', 'Cancelled', 'Scheduled'}
    all_games = statsapi.schedule(start_date=yesterday.strftime('%m/%d/%Y'))
    games = [g for g in all_games if g.get('status') not in INCOMPLETE_STATUSES]
    if not games:
        return 'No completed games today'

    day_rows = []
    ids_seen = []

    for g in games:
        game_id = g['game_id']
        game_data = statsapi.get('game', params={'gamePk': game_id})['gameData']
        away_abbreviation = game_data['teams']['away']['abbreviation']
        home_abbreviation = game_data['teams']['home']['abbreviation']

        away_normal = team_abbvs[away_abbreviation]['normal']
        home_normal = team_abbvs[home_abbreviation]['normal']

        row_id = f'{yesterday}-{away_normal}@{home_normal}'
        if row_id in ids_seen:
            row_id += '-game2'
        ids_seen.append(row_id)

        linescore = statsapi.linescore(game_id)
        top1 = int(re.search(r'[0-9]+', linescore.split('\n')[1]).group())
        bot1 = int(re.search(r'[0-9]+', linescore.split('\n')[2]).group())
        first_inn_runs = top1 + bot1

        boxscore = statsapi.boxscore_data(game_id)

        away_lineup = get_lineup(boxscore, 'away')
        home_lineup = get_lineup(boxscore, 'home')
        away_ops = lineup_to_ops(away_lineup, ops_df) if ops_df is not None else None
        home_ops = lineup_to_ops(home_lineup, ops_df) if ops_df is not None else None

        if len(boxscore['awayPitchers']) < 2 or len(boxscore['homePitchers']) < 2:
            print(f'  Skipping {row_id}: pitcher data not available')
            ids_seen.remove(row_id)
            continue
        away_pitcher = api_name(boxscore['awayPitchers'][1]['personId'])
        home_pitcher = api_name(boxscore['homePitchers'][1]['personId'])

        if pitcher_runs_df is not None:
            away_ra = get_pitcher_ra(away_pitcher, pitcher_runs_df)
            home_ra = get_pitcher_ra(home_pitcher, pitcher_runs_df)
        else:
            away_ra = home_ra = None

        away_whip = get_whip(away_pitcher, whip_df) if whip_df is not None else None
        home_whip = get_whip(home_pitcher, whip_df) if whip_df is not None else None

        home_yrfi_pct = get_yrfi_split(team_pct, team_abbvs[home_abbreviation]['team_pct'], 'Home', year_col)
        away_yrfi_pct = get_yrfi_split(team_pct, team_abbvs[away_abbreviation]['team_pct'], 'Away', year_col)

        park_factor = park_factor_dict.get(home_normal, 100)

        weather_str, pitch_str = '', ''
        for line in boxscore.get('gameBoxInfo', []):
            if line['label'] == 'Weather':     weather_str = line['value']
            if line['label'] == 'First pitch': pitch_str   = line['value']
        temp, day_night, clear, cloudy, rain, dome = parse_weather(weather_str, pitch_str)

        day_rows.append({
            'id':              row_id,
            'year':            date.year,
            'month':           date.month,
            'day':             date.day,
            '1st_runs':        first_inn_runs,
            'YRFI':            1 if first_inn_runs > 0 else 0,
            'away_lineup':     away_lineup,
            'home_lineup':     home_lineup,
            'away_ops':        away_ops,
            'home_ops':        home_ops,
            'away_pitcher':    away_pitcher,
            'home_pitcher':    home_pitcher,
            'away_pitcher_ra': away_ra,
            'home_pitcher_ra': home_ra,
            'away_whip':       away_whip,
            'home_whip':       home_whip,
            'home_yrfi_pct':   home_yrfi_pct,
            'away_yrfi_pct':   away_yrfi_pct,
            'park_factor':     park_factor,
            'temp':            temp,
            'day0/night1':     day_night,
            'clear':           clear,
            'cloudy':          cloudy,
            'rain':            rain,
            'dome':            dome,
        })

    big_df = pd.DataFrame(day_rows).reset_index(drop=True)
    return big_df, yesterday
