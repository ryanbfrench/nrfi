"""One-time backfill: compute first-inning pitcher RA for the current season from statsapi linescore data and save to s3://nrfi-store/pitcher_ra/{year}.json. Run locally before the next lambda invocation: python utils/backfill_pitcher_ra.py The lambda will then load this file and only add yesterday's increment each day. RA definition: runs allowed by the starting pitcher in their half of the 1st inning, computed as total_1st_inn_runs / number_of_starts. - Home starter: faces away team in the TOP of the 1st (top1 = away team's runs) - Away starter: faces home team in the BOTTOM of the 1st (bot1 = home team's runs)"""

import json
import re
import datetime
import boto3
import statsapi

BUCKET      = 'nrfi-store'
SEASON_START = datetime.date(2026, 3, 27)
YESTERDAY   = datetime.date.today() - datetime.timedelta(1)
S3_KEY      = f'pitcher_ra/{YESTERDAY.year}.json'


def main():
    s3 = boto3.client('s3', region_name='us-east-1')

    # Load any existing accumulation so we can resume a partial run
    existing = {}
    try:
        obj = s3.get_object(Bucket=BUCKET, Key=S3_KEY)
        existing = json.loads(obj['Body'].read().decode('utf-8'))
        last_date_str = existing.pop('_last_date', None)
        if last_date_str:
            resume_from = datetime.date.fromisoformat(last_date_str) + datetime.timedelta(1)
            print(f'Resuming from {resume_from} (last saved: {last_date_str})')
        else:
            resume_from = SEASON_START
    except Exception:
        resume_from = SEASON_START
        print(f'No existing file found — starting from {SEASON_START}')

    if resume_from > YESTERDAY:
        print('Already up to date.')
        return

    # Fetch all completed games in the range
    start_str = resume_from.strftime('%m/%d/%Y')
    end_str   = YESTERDAY.strftime('%m/%d/%Y')
    print(f'Fetching schedule {start_str} to {end_str}...')
    all_games  = statsapi.schedule(start_date=start_str, end_date=end_str)
    completed  = [g for g in all_games if g.get('status') == 'Final']
    print(f'{len(completed)} completed games to process')

    ra = dict(existing)  # str(personId) -> {'R': int, 'G': int}
    errors = 0

    for i, g in enumerate(completed, 1):
        gid = g['game_id']
        try:
            box = statsapi.boxscore_data(gid)
            ls  = statsapi.linescore(gid)
            lines = ls.split('\n')
            top1 = int(re.search(r'[0-9]+', lines[1]).group())
            bot1 = int(re.search(r'[0-9]+', lines[2]).group())

            # Home starter faces away batters → their runs allowed = top1
            if len(box.get('homePitchers', [])) > 1:
                pid = str(box['homePitchers'][1]['personId'])
                if pid not in ra:
                    ra[pid] = {'R': 0, 'G': 0}
                ra[pid]['R'] += top1
                ra[pid]['G'] += 1

            # Away starter faces home batters → their runs allowed = bot1
            if len(box.get('awayPitchers', [])) > 1:
                pid = str(box['awayPitchers'][1]['personId'])
                if pid not in ra:
                    ra[pid] = {'R': 0, 'G': 0}
                ra[pid]['R'] += bot1
                ra[pid]['G'] += 1

        except Exception as e:
            errors += 1
            print(f'  [{i}/{len(completed)}] game {gid} skipped: {e}')
            continue

        if i % 50 == 0:
            print(f'  [{i}/{len(completed)}] processed ({errors} errors so far)')

    ra['_last_date'] = str(YESTERDAY)

    body = json.dumps(ra).encode('utf-8')

    # Save locally first as a backup
    local_path = f'pitcher_ra_{YESTERDAY.year}.json'
    with open(local_path, 'wb') as f:
        f.write(body)
    print(f'Saved locally to {local_path}')

    # Save to S3
    try:
        s3.put_object(Bucket=BUCKET, Key=S3_KEY, Body=body, ContentType='application/json')
        print(f'Saved to s3://{BUCKET}/{S3_KEY}')
    except Exception as e:
        print(f'S3 upload failed: {e}')
        print(f'Upload manually: aws s3 cp {local_path} s3://{BUCKET}/{S3_KEY}')

    pitcher_count = len(ra) - 1  # exclude _last_date key
    print(f'\nDone. {pitcher_count} pitchers, {errors} errors.')

    # Spot-check a few well-known pitchers
    print('\nSpot-check (verify these match what you expect):')
    known = {
        '660271': 'Shohei Ohtani',
        '694973': 'Paul Skenes',
        '519242': 'Chris Sale',
    }
    for pid, name in known.items():
        if pid in ra:
            s = ra[pid]
            print(f'  {name}: {s["R"]} runs / {s["G"]} starts = {s["R"]/s["G"]:.3f} RA')
        else:
            print(f'  {name}: not found')


if __name__ == '__main__':
    main()
