import json
import pandas as pd
import numpy as np
import io
import requests
import urllib.parse
from bs4 import BeautifulSoup, Tag
from datetime import date, timedelta, datetime
import re
import pymongo
from pymongoarrow.api import find_pandas_all
from os import path
from os import environ as env
import pickle

# Requests Session
def get(url: str) -> requests.Response:
    print('GET', url, end = '')
    req = requests.Response()
    global __SESSION__
    try:
        req = __SESSION__.get(url, timeout = None)
    except requests.exceptions.ConnectionError:
        __SESSION__ = requests.Session()
        __SESSION__.headers.update(__SESSION_HEADERS__)
        req = __SESSION__.get(url, timeout = None)
    print(f' ({req.status_code}) {round(req.elapsed.total_seconds(), 2)}s')
    return req

__SESSION_HEADERS__ = {
    'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/108.0.0.0 Safari/537.36',
    'X-Requested-With': 'XMLHttpRequest'
}
__SESSION__ = requests.Session()
__SESSION__.headers.update(__SESSION_HEADERS__)

# MongoDB
def get_mongodb_password():
    file = 'MongoDB.txt'
    if path.isfile(file):
        with open(file) as f:
            for line in f.read().split('\n'):
                env['MONGODB_PASSWORD'] = line
                print('Found MongoDB password in', file)
                return

get_mongodb_password()
__DB__ = pymongo.MongoClient(f'mongodb+srv://peteb206:{env.get("MONGODB_PASSWORD")}@btscluster.tp9p0.mongodb.net').get_database('bts')

__DB_SCHEMAS__ = {
    'atBats': {
        'game_pk': int, 'at_bat': int, 'batter': int, 'pitcher': int, 'xBA': float, 'home': bool, 'rhb': bool, 'rhp': bool, 'events': int
    },
    'games': {
        'game_date': datetime, 'game_pk': int, 'away_team': str, 'home_team': str, 'away_starter': int, 'home_starter': int, 'park_factor': int
    },
    'players': {
        'year': int, 'playerId': int, 'hp_to_1b': float, 'speed': float, 'bats': str, 'throws': str
    }
}

def update_db(year: int):
    # At Bats
    start_date = date(year, 7 if year == 2020 else 3, 23)
    end_date = min(date(year, 10, 6), date.today() - timedelta(days = 1))
    print('-' * 80)
    print('Adding/Updating at bats from', start_date.strftime('%Y-%m-%d'), 'to', end_date.strftime('%Y-%m-%d'))
    new_at_bat_df = StatcastData.get_statcast_csv(start_date = start_date, end_date = end_date) \
        .sort_values(by = ['game_pk', 'at_bat'], ignore_index = True)
    # Delete existing entries
    deleted_count = __DB__.atBats.delete_many({'game_pk': {'$in': new_at_bat_df['game_pk'].unique().tolist()}}).deleted_count
    print('Deleted', '{:,}'.format(deleted_count), 'at bats')
    # Add new entries
    new_at_bat_df['game_date'] = new_at_bat_df['game_date'].apply(lambda x: date_to_datetime(x))
    __DB__.atBats.insert_many(
        [{k: v for k, v in row.items() if pd.notnull(v)} for row in new_at_bat_df[list(__DB_SCHEMAS__['atBats'].keys())].to_dict('records')]
    )
    print('Added', '{:,}'.format(len(new_at_bat_df.index)), 'at bats')

    # Players
    print('-' * 80)
    print('Adding/Updating players for the', year, 'season')
    new_players_df = MLBData.get_stats_api_players(year).rename({'id': 'playerId', 'fullName': 'name'}, axis = 1) \
        .loc[:, ['year', 'playerId', 'name', 'bats', 'throws']] \
            .merge(StatcastData.get_sprint_speed_csv(year), how = 'outer', on = ['year', 'playerId'])
    # Delete existing entries
    print('Deleted', '{:,}'.format(__DB__.players.delete_many({'year': year}).deleted_count), 'players')
    # Add new entries
    __DB__.players.insert_many([{k: v for k, v in row.items() if pd.notnull(v)} for row in new_players_df.to_dict('records')])
    print('Added', '{:,}'.format(len(new_players_df.index)), 'players')

    # Games
    print('-' * 80)
    print('Adding/Updating games for the', year, 'season')
    game_pks = new_at_bat_df['game_pk'].unique().tolist()
    game_dates_df = new_at_bat_df[['game_date', 'game_pk', 'away_team', 'home_team']].drop_duplicates().sort_values(by = ['game_date', 'game_pk'])
    starting_pitchers_df = new_at_bat_df.sort_values(['game_pk', 'home', 'at_bat']).groupby(['game_pk', 'home']).pitcher.first() \
        .reset_index().pivot(index = 'game_pk', columns = 'home')['pitcher'] \
        .rename({False: 'home_starter', True: 'away_starter'}, axis = 1).reset_index()
    games_df = game_dates_df.merge(starting_pitchers_df, on = 'game_pk')
    games_df['park_factor'] = games_df.game_pk.map(MLBData.get_stats_api_game_venues(year)).map(StatcastData.get_park_factors(year))
    # Delete existing entries
    print('Deleted', '{:,}'.format(__DB__.games.delete_many({'game_pk': {'$in': game_pks}}).deleted_count), 'games')
    # Add new entries
    __DB__.games.insert_many([{k: v for k, v in row.items() if pd.notnull(v)} for row in games_df.to_dict('records')])
    print('Added', '{:,}'.format(len(games_df.index)), 'games')

def month_start_and_end(year: int, month: int) -> tuple[date, date]:
    return date(year = year, month = month, day = 1), date(year = year, month = month + 1, day = 1) - timedelta(days = 1)

def date_to_datetime(date: date) -> datetime:
    return datetime.combine(date, datetime.min.time())

def get_enhanced_at_bats(from_date: date | datetime) -> pd.DataFrame:
    games_df: pd.DataFrame = find_pandas_all(__DB__.games, {'game_date': {'$gte': from_date}}, projection = {'_id': False})
    at_bats_df: pd.DataFrame = find_pandas_all(__DB__.atBats, {'game_pk': {'$in': games_df.game_pk.unique().tolist()}}, projection = {'_id': False})
    players_df: pd.DataFrame = find_pandas_all(__DB__.players, {'year': {'$gte': from_date.year}}, projection = {'_id': False})

    games_df.set_index('game_pk', inplace = True)
    games_df.park_factor = games_df.park_factor.fillna(100).astype(int)
    at_bats_df.set_index(['game_pk', 'home', 'at_bat', 'batter', 'pitcher'], inplace = True)
    at_bats_df.sort_index(inplace = True)
    # Starting pitcher
    at_bats_df = at_bats_df.merge(games_df, left_index = True, right_index = True)
    at_bats_df['team'] = np.where(at_bats_df.index.get_level_values('home'), at_bats_df.home_team, at_bats_df.away_team)
    at_bats_df['opponent'] = np.where(at_bats_df.index.get_level_values('home'), at_bats_df.away_team, at_bats_df.home_team)
    at_bats_df['opp_sp'] = np.where(at_bats_df.index.get_level_values('home'), at_bats_df.away_starter, at_bats_df.home_starter)
    at_bats_df['year'] = at_bats_df.game_date.dt.year
    at_bats_df.set_index(['year', 'game_date', 'team', 'opponent'], append = True, inplace = True)

    # Player data
    players_df.set_index(['year', 'playerId'], inplace = True)
    at_bats_df = at_bats_df \
        .merge(players_df.rename_axis(index = {'playerId': 'batter'}).drop('throws', axis = 1), how = 'left', left_index = True, right_index = True) \
            .merge(players_df.rename_axis(index = {'playerId': 'pitcher'}).rename({'name': 'opp_sp_name'}, axis = 1)[['throws', 'opp_sp_name']],
                   how = 'left', left_index = True, right_index = True) \
                .droplevel('year')
    hp_to_1b_regression = pickle.load(open('models/hp_to_1b.pkl', 'rb'))
    null_hp_to_1b = at_bats_df['hp_to_1b'].isna() & ~at_bats_df['speed'].isna()
    at_bats_df.loc[null_hp_to_1b, 'hp_to_1b'] = hp_to_1b_regression.predict(at_bats_df.loc[null_hp_to_1b, ['rhb', 'speed']]).round(2)

    # Lineups
    lineups_df = at_bats_df.reset_index().sort_values(by = ['game_pk', 'at_bat']).drop_duplicates(subset = ['game_pk', 'batter'])
    lineups_df['lineup'] = lineups_df.groupby(['game_pk', 'home']).cumcount() + 1
    lineups_df = lineups_df.set_index(['game_pk', 'home', 'batter'])[['lineup']]
    at_bats_df = at_bats_df \
        .merge(lineups_df.loc[lineups_df.lineup < 10], how = 'left', left_index = True, right_index = True) \
            .reorder_levels(['game_date', 'game_pk', 'home', 'team', 'opponent', 'at_bat', 'batter', 'pitcher']) \
                .sort_index()

    # x stats calculations
    at_bats_df.events = at_bats_df.events.replace(dict(enumerate(StatcastData.__AT_BAT_FINAL_EVENTS__)))
    at_bats_df['xBA_med'] = at_bats_df.events.map(at_bats_df.groupby('events')['xBA'].median())
    at_bats_df['H'] = at_bats_df.events.isin(['single', 'double', 'triple', 'home_run'])
    at_bats_df['BIP'] = at_bats_df.xBA_med > 0
    at_bats_df['statcast_tracked'] = at_bats_df.index \
        .get_level_values('game_pk') \
            .map(at_bats_df.groupby('game_pk')['xBA'].sum() > 0)
    # print(at_bats_df.statcast_tracked.value_counts(normalize = True).mul(100).round(1)[True], '% of at bats were statcast tracked...', sep = '')
    return at_bats_df.drop(['speed', 'away_team', 'home_team', 'away_starter', 'home_starter', 'xBA_med'], axis = 1)

def get_todays_batters() -> pd.DataFrame:
    game_date = date.today()
    BTS_JSON_PATH = 'https://www.mlb.com/apps/beat-the-streak/game/json'
    venues_dict = MLBData.get_stats_api_game_venues()
    park_factors_dict = StatcastData.get_park_factors(game_date.year)

    games_df = pd.DataFrame(json.loads(get(f'{BTS_JSON_PATH}/units.json').text)['units'])
    games_df.lockDateTime = games_df.lockDateTime.apply(lambda x: datetime.strptime(x[:16], '%Y-%m-%dT%H:%M'))
    games_df = games_df.loc[
        games_df.lockDateTime.apply(lambda x: x.date()) == game_date,
        ['feedId', 'awaySquadId', 'homeSquadId', 'awayProbablePitcherId', 'homeProbablePitcherId', 'lockDateTime', 'lineups']
    ].rename({'feedId': 'game_pk'}, axis = 1)
    games_df['park_factor'] = games_df.game_pk.map(venues_dict).map(park_factors_dict).fillna(100)

    teams_df = pd.DataFrame(json.loads(get(f'{BTS_JSON_PATH}/squads.json').text)['squads'])[['id', 'abbreviation']]
    games_df = games_df \
        .merge(teams_df.rename({'id': 'awaySquadId', 'abbreviation': 'awayTeam'}, axis = 1), on = 'awaySquadId') \
            .merge(teams_df.rename({'id': 'homeSquadId', 'abbreviation': 'homeTeam'}, axis = 1), on = 'homeSquadId')

    todays_batters_df = games_df.explode('lineups').reset_index(drop = True)
    todays_batters_df['id'] = todays_batters_df.lineups.apply(lambda x: x['playerId'])
    todays_batters_df['lineup'] = todays_batters_df.lineups.apply(lambda x: x['lineupNumber'] / 100 if x['lineupNumber'] != None else np.nan)

    players_df = pd.DataFrame(json.loads(get(f'{BTS_JSON_PATH}/players.json').text)['players'])
    players_df.handedness = players_df.handedness.fillna('R').apply(lambda x: x[0].upper() if x[0] in ['l', 'r', 's'] else 'R')
    # players_df.name = players_df.apply(lambda row: f'{row["name"]} ({row["handedness"]})', axis = 1)

    todays_batters_df = todays_batters_df \
        .merge(players_df.loc[(players_df.status == 'active') & (players_df.position != 'pitcher'),
                              ['id', 'feedId', 'squadId', 'name', 'handedness', 'status']], on = 'id')

    todays_batters_df['home'] = todays_batters_df.squadId == todays_batters_df.homeSquadId
    todays_batters_df['team'] = np.where(todays_batters_df.home, todays_batters_df.homeTeam, todays_batters_df.awayTeam)
    todays_batters_df['opponent'] = np.where(todays_batters_df.home, todays_batters_df.awayTeam, todays_batters_df.homeTeam)
    todays_batters_df['spId'] = np.where(todays_batters_df.home, todays_batters_df.awayProbablePitcherId, todays_batters_df.homeProbablePitcherId)

    todays_batters_df = todays_batters_df \
        .merge(players_df[['id', 'feedId', 'name', 'handedness']], left_on = 'spId', right_on = 'id', suffixes = ('', '.sp')) \
            .rename({'lockDateTime': 'game_time', 'feedId': 'batter', 'handedness': 'bats', 'feedId.sp': 'opp_sp', 'name.sp': 'opp_sp_name',
                        'handedness.sp': 'throws'}, axis = 1)

    todays_batters_df = todays_batters_df \
        .merge((todays_batters_df.groupby(['game_pk', 'team']).lineup.max() > 0).reset_index().rename({'lineup': 'team_lineup_set'}, axis = 1))
    # 0 = TBD, 10 = OUT
    todays_batters_df.lineup = todays_batters_df \
        .apply(lambda row: int(row['lineup']) if row['lineup'] > 0 else 10 if row['team_lineup_set'] else 0, axis = 1)
    todays_batters_df['game_date'] = date_to_datetime(game_date)
    todays_batters_df = todays_batters_df \
        .set_index(['game_date', 'game_pk', 'home', 'team', 'opponent', 'batter']) \
            .loc[:, ['game_time', 'lineup', 'name', 'bats', 'opp_sp', 'opp_sp_name', 'throws', 'park_factor']]
    return todays_batters_df

class MLBData:
    @staticmethod
    def get_stats_api_game_venues(year: int | None = None) -> dict[int, int]:
        game_venue_dict = dict()
        url = 'https://statsapi.mlb.com/api/v1/schedule?lang=en&sportId=1&gameType=R'
        if year == None:
            url += f'&date={date.today().strftime("%Y-%m-%d")}'
        else:
            url += f'&season={year}'
        for game_dt in json.loads(get(url).text)['dates']:
            for game in game_dt['games']:
                game_venue_dict[game['gamePk']] = game['venue']['id']
        return game_venue_dict

    @staticmethod
    def get_stats_api_players(year: int):
        req = get(f'https://statsapi.mlb.com/api/v1/sports/1/players?lang=en&sportId=1&season={year}')
        people = json.loads(req.text)['people']
        player_df = pd.DataFrame(people)
        player_df['year'] = year
        player_df['position'] = player_df['primaryPosition'].apply(lambda x: x['abbreviation'])
        player_df['throws'] = player_df['pitchHand'].apply(lambda x: x['code'])
        player_df['bats'] = player_df['batSide'].apply(lambda x: x['code'])
        # player_df['team_id'] = player_df['currentTeam'].apply(lambda x: x['id'])
        return player_df[['year', 'id', 'fullName', 'position', 'throws', 'bats', 'active']]

class StatcastData(MLBData):
    __AT_BAT_FINAL_EVENTS__ = [
        'single', 'double', 'triple', 'home_run', 'field_out', 'strikeout', 'strikeout_double_play', 'walk', 'double_play', 'field_error',
        'grounded_into_double_play', 'fielders_choice', 'fielders_choice_out', 'batter_interference', 'catcher_interf', 'force_out', 'hit_by_pitch',
        'intent_walk', 'sac_bunt', 'sac_bunt_double_play', 'sac_fly', 'sac_fly_double_play', 'triple_play'
    ]

    @staticmethod
    def get_statcast_csv(start_date = date(2023, 4, 1), end_date = date(2023, 4, 1)) -> pd.DataFrame:
        end_date = min(end_date, date.today() - timedelta(days = 1))

        # Construct data splits for request to Baseball Savant
        i, split_size, date_splits, date_split_end = 0, 21, list(), start_date - timedelta(days = 1)
        while date_split_end < end_date:
            date_split_start = start_date + timedelta(days = i * split_size + i)
            date_split_end = min(end_date, date_split_start + timedelta(days = split_size))
            date_splits.append((date_split_start, date_split_end))
            i += 1

        cols = [
            'game_pk', 'game_date', 'at_bat_number', 'inning_topbot', 'batter', 'stand', 'pitcher', 'p_throws', 'away_team', 'home_team', 'events',
            'estimated_ba_using_speedangle'
        ]
        df = pd.DataFrame(columns = cols)
        print('-' * 50)
        for date_split in date_splits:
            # Filter pitches to only pitches that end at bats
            # Construct URL and send request
            print('Fetching at bats from ', date_split[0].strftime('%Y-%m-%d'), ' to ', date_split[1].strftime('%Y-%m-%d'), '...', sep = '')
            url_params = {
                'all': 'true', 'hfAB': urllib.parse.quote('|'.join(StatcastData.__AT_BAT_FINAL_EVENTS__).replace('_', '\\.' * 2) + '|'),
                'hfGT': 'R%7C', 'hfSea': urllib.parse.quote('|'.join({str(date_split[0].year), str(date_split[1].year)}) + '|'),
                'player_type': 'batter', 'game_date_lt': date_split[1].strftime('%Y-%m-%d'), 'game_date_gt': date_split[0].strftime('%Y-%m-%d'),
                'min_pitches': 0, 'min_results': 0, 'group_by': 'name', 'sort_col': 'pitches', 'player_event_sort': 'api_p_release_speed',
                'sort_order': 'desc', 'min_pas': '0', 'type': 'details'
            }
            url = f'https://baseballsavant.mlb.com/statcast_search/csv?{"".join([f"{k}={v}&" for k, v in url_params.items()])}'

            # Convert CSV to dataframe and filter to wanted columns
            df2, baseball_savant_response, attempt = pd.DataFrame(columns = cols), requests.Response(), 1
            while (len(df2.index) == 0) & (attempt <= 2):
                try:
                    baseball_savant_response = get(url)
                    df2 = pd.read_csv(io.StringIO(baseball_savant_response.content.decode('utf-8')), usecols = cols, engine = 'pyarrow')
                except:
                    if attempt == 1:
                        print('Retrying...')
                        attempt += 1
                    else:
                        raise Exception(f'Unsuccessful reading of Statcast CSV\n\nResponse from Baseball Savant:\n{baseball_savant_response.text}')
            print('{:,}'.format(len(df2.index)), 'total at bats')
            df = pd.concat([df, df2], ignore_index = True)
        print('-' * 50)
        df['home'] = df['inning_topbot'] == 'Bot'
        df['rhb'] = df['stand'] == 'R'
        df['rhp'] = df['p_throws'] == 'R'
        enumerated_events = {x: i for i, x in enumerate(StatcastData.__AT_BAT_FINAL_EVENTS__)}
        df['events'] = df['events'].apply(lambda x: enumerated_events[x])
        df.rename({'estimated_ba_using_speedangle': 'xBA', 'at_bat_number': 'at_bat'}, axis = 1, inplace = True)
        return df[list(__DB_SCHEMAS__['atBats'].keys()) + ['game_date', 'away_team', 'home_team']]

    @staticmethod
    def get_sprint_speed_csv(year = 2023) -> pd.DataFrame:
        url = f'https://baseballsavant.mlb.com/leaderboard/sprint_speed?min_season={year}&max_season={year}&position=&team=&min=0&csv=true'
        baseball_savant_response = get(url)
        df = pd.read_csv(io.StringIO(baseball_savant_response.content.decode('utf-8')), usecols = ['player_id', 'hp_to_1b', 'sprint_speed'])
        df['year'] = year
        return df.rename({'player_id': 'playerId', 'sprint_speed': 'speed'}, axis = 1)

    @staticmethod
    def get_park_factors(year: int) -> dict[int, float]:
        venues_dict = dict()
        baseball_savant_response = get('https://baseballsavant.mlb.com/leaderboard/statcast-park-factors?type=venue&stat=index_Hits')
        soup = BeautifulSoup(baseball_savant_response.text, 'html.parser')
        data_div = soup.find('div', {'class': 'article-template'})
        if data_div != None:
            script = data_div.find('script')
            if isinstance(script, Tag):
                search = re.search(r'var data = (\[.*\])', script.text)
                if search != None:
                    park_factors_json_str = search.group(1)
                    park_factors_json = json.loads(park_factors_json_str)
                    for park_dict in park_factors_json:
                        venue_id = int(park_dict['venue_id'])
                        park_factor = park_dict[f'metric_value_{year}']
                        if isinstance(park_factor, str):
                            venues_dict[venue_id] = int(park_factor)
        return venues_dict

if __name__ == '__main__':
    get_todays_batters()