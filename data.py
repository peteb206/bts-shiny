import json
import pandas as pd
import io
import requests
from datetime import date, timedelta, datetime
import pymongo
from pymongoarrow.api import find_pandas_all
from os import path
from os import environ as env

# Requests Session
__SESSION__ = requests.session()
__SESSION__.headers = {
    'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/108.0.0.0 Safari/537.36',
    'X-Requested-With': 'XMLHttpRequest'
}
def get(url: str) -> requests.Response:
    print('GET', url if len(url) <= 100 else f'{url[:97]}...', end = '')
    req = __SESSION__.get(url, timeout = None)
    print(f' ({req.status_code}) {round(req.elapsed.total_seconds(), 2)}s')
    return req

# MongoDB
if path.isfile('.env'):
    with open('.env') as f:
        for line in f.read().split('\n'):
            x = line.split('=')
            env[x[0]] = x[1]
__DB__ = pymongo.MongoClient(f'mongodb+srv://peteb206:{env.get("MONGODB_PASSWORD")}@btscluster.tp9p0.mongodb.net')['bts-hub']

def month_start_and_end(year: int = None, month: int = None) -> (date, date):
    return date(year = year, month = month, day = 1), date(year = year, month = month + 1, day = 1) - timedelta(days = 1)

def date_to_datetime(date: date) -> datetime:
    return datetime.combine(date, datetime.min.time())

class MLBData:

    @staticmethod
    def get_stats_api_games(year: int = None, date: date = None):
        req = get(
            'https://statsapi.mlb.com/api/v1/schedule?lang=en&sportId=1&gameType=R' \
                + (f'&season={year}' if date == None else f'&date={date.strftime("%Y-%m-%d")}') \
                + '&hydrate=team,probablePitcher,lineups' #,weather,linescore'
        )
        games_list = list()
        for game_date in json.loads(req.text)['dates']:
            for game in game_date['games']:
                utc_hour, minute = int(game['gameDate'].split('T')[1].split(':')[0]), int(game['gameDate'].split('T')[1].split(':')[1])
                hour = utc_hour - 5 if utc_hour > 5 else utc_hour + 19
                games_list.append({
                    'year': year,
                    'game_date': game_date['date'],
                    'game_time': f'{hour if hour < 13 else hour - 12}:{"0" if minute < 10 else ""}{minute} {"PM" if hour > 11 else "AM"}',
                    'game_pk': game['gamePk'],
                    # 'away_team_id': game['teams']['away']['team']['id'],
                    # 'home_team_id': game['teams']['home']['team']['id'],
                    'away_team': game['teams']['away']['team']['abbreviation'],
                    'home_team': game['teams']['home']['team']['abbreviation'],
                    'away_starter': game['teams']['away']['probablePitcher']['id'] if 'probablePitcher' in game['teams']['away'].keys() else 0,
                    'home_starter': game['teams']['home']['probablePitcher']['id'] if 'probablePitcher' in game['teams']['home'].keys() else 0,
                    'away_lineup': ([x['id'] for x in game['lineups']['awayPlayers']] if 'awayPlayers' in game['lineups'].keys() else []) \
                        if 'lineups' in game.keys() else [],
                    'home_lineup': ([x['id'] for x in game['lineups']['homePlayers']] if 'homePlayers' in game['lineups'].keys() else []) \
                        if 'lineups' in game.keys() else []
                })
        return pd.DataFrame(games_list)

    @staticmethod
    def get_stats_api_players(year: int = None):
        req = get(f'https://statsapi.mlb.com/api/v1/sports/1/players?lang=en&sportId=1&season={year}')
        people = json.loads(req.text)['people']
        player_df = pd.DataFrame(people)
        player_df['year'] = year
        player_df['position'] = player_df['primaryPosition'].apply(lambda x: x['abbreviation'])
        player_df['throws'] = player_df['pitchHand'].apply(lambda x: x['code'])
        player_df['bats'] = player_df['batSide'].apply(lambda x: x['code'])
        # player_df['team_id'] = player_df['currentTeam'].apply(lambda x: x['id'])
        return player_df[['year', 'id', 'fullName', 'position', 'throws', 'bats', 'active']]

class StatcastData:
    __AT_BAT_FINAL_EVENTS__ = [
        'single', 'double', 'triple', 'home_run', 'field_out', 'strikeout', 'strikeout_double_play', 'walk', 'double_play', 'field_error',
        'grounded_into_double_play', 'fielders_choice', 'fielders_choice_out', 'batter_interference', 'catcher_interf', 'force_out', 'hit_by_pitch',
        'intent_walk', 'sac_bunt', 'sac_bunt_double_play', 'sac_fly', 'sac_fly_double_play', 'triple_play'
    ]

    def __init__(self):
        self.__df__ = self.get_at_bats_from_db()

        # Replace missing xBA values with median for that event
        self.__df__ = self.__df__ \
            .merge(self.__df__.groupby('game_pk')['xBA'].sum().reset_index(), how = 'left', on = 'game_pk', suffixes = ['', '_game_sum']) \
            .merge(self.__df__.groupby('events')['xBA'].median().reset_index(), how = 'left', on = 'events', suffixes = ['', '_med']) \
            .fillna({'xBA_game_sum': 0, 'xBA_med': 0})
        self.__df__['xBA'] = self.__df__.apply(lambda row: row['xBA_med'] if row['xBA_game_sum'] == 0 else row['xBA'], axis = 1)
        self.__df__['H'] = self.__df__['events'] < 4
        self.__df__['BIP'] = self.__df__['xBA_med'] > 0
        self.__df__.drop(['events', 'xBA_med', 'xBA_game_sum'], axis = 1, inplace = True)

    def df(self):
        return self.__df__.copy()

    @staticmethod
    def get_at_bats_from_db(query: dict = dict()) -> pd.DataFrame:
        return find_pandas_all(__DB__.atBats, query, projection = {'_id': False})

    @staticmethod
    def update_db(year: int):
        # At Bats
        new_at_bat_df: pd.DataFrame = StatcastData \
            .add_at_bats_to_db(start_date = date(year, 7 if year == 2020 else 3, 23), end_date = date(year, 10, 6))
        # Sprint Speeds
        StatcastData.add_season_sprint_speeds_to_db(year)
        # Game Dates
        game_pks = new_at_bat_df['game_pk'].unique().tolist()
        games_df = MLBData.get_stats_api_games(year)
        print('-' * 50)
        new_game_dates_df = new_at_bat_df[['game_date', 'game_pk']].drop_duplicates().sort_values(by = ['game_date', 'game_pk'])
        print(f'Deleted {"{:,}".format(__DB__.gameDates.delete_many({"game_pk": {"$in": game_pks}}).deleted_count)} game dates')
        __DB__.gameDates.insert_many(new_game_dates_df.to_dict('records'))
        print('Added', '{:,}'.format(len(new_game_dates_df.index)), 'game dates')
        # Lineup Slots
        print('-' * 50)
        new_at_bat_df = new_at_bat_df.merge(games_df.drop('game_date', axis = 1), how = 'left', on = 'game_pk')
        new_at_bat_df['lineup'] = new_at_bat_df.apply(lambda row: row['home_lineup'] if row['home'] else row['away_lineup'], axis = 1)
        new_at_bat_df['lineup'] = new_at_bat_df \
            .apply(lambda row: row['lineup'].index(row['batter']) + 1 if row['batter'] in row['lineup'] else 10, axis = 1)
        new_lineups_df = new_at_bat_df[['game_pk', 'home', 'lineup', 'batter']].dropna().drop_duplicates() \
            .sort_values(by = ['game_pk', 'home', 'lineup'])
        print(f'Deleted {"{:,}".format(__DB__.lineupSlots.delete_many({"game_pk": {"$in": game_pks}}).deleted_count)} lineup slots')
        __DB__.lineupSlots.insert_many(new_lineups_df.to_dict('records'))
        print('Added', '{:,}'.format(len(new_lineups_df.index)), 'lineup slots')
        # Starting Pitchers
        print('-' * 50)
        new_at_bat_df['primary_pitcher'] = new_at_bat_df.apply(lambda row: row['pitcher'] in [row['away_starter'], row['home_starter']], axis = 1)
        new_starting_pitchers_df = new_at_bat_df[new_at_bat_df['primary_pitcher']][['game_pk', 'home', 'pitcher']].dropna().drop_duplicates() \
            .sort_values(by = ['game_pk', 'home'])
        print(f'Deleted {"{:,}".format(__DB__.opposingStartingPitchers.delete_many({"game_pk": {"$in": game_pks}}).deleted_count)} starting pitchers')
        __DB__.opposingStartingPitchers.insert_many(new_starting_pitchers_df.to_dict('records'))
        print('Added', '{:,}'.format(len(new_starting_pitchers_df.index)), 'starting pitchers')
        print('-' * 80)

    @staticmethod
    def add_at_bats_to_db(start_date = date(2023, 4, 1), end_date = date(2023, 4, 30)) -> pd.DataFrame:
        end_date = min(end_date, date.today() - timedelta(days = 1))
        print('-' * 80)
        print('Adding/Updating at bats from', start_date.strftime('%Y-%m-%d'), 'to', end_date.strftime('%Y-%m-%d'))
        new_at_bat_df = StatcastData.get_statcast_csv(start_date = start_date, end_date = end_date)
        # Delete existing entries
        deleted_count = __DB__.atBats.delete_many({'game_pk': {'$in': new_at_bat_df['game_pk'].unique().tolist()}}).deleted_count
        print('Deleted', '{:,}'.format(deleted_count), 'at bats')
        # Add new entries
        new_at_bat_df['game_date'] = new_at_bat_df['game_date'].apply(lambda x: date_to_datetime(x))
        __DB__.atBats.insert_many([{k: v for k, v in row.items() if pd.notnull(v)} for row in new_at_bat_df.drop(['game_date', 'home'], axis = 1) \
                                   .to_dict('records')])
        print('Added', '{:,}'.format(len(new_at_bat_df.index)), 'at bats')
        print('-' * 80)
        return new_at_bat_df

    @staticmethod
    def get_statcast_csv(start_date = date(2023, 4, 1), end_date = date(2023, 4, 30)) -> pd.DataFrame:
        end_date = min(end_date, date.today() - timedelta(days = 1))

        # Construct data splits for request to Baseball Savant
        i, split_size, date_splits, date_split_end = 0, 21, list(), start_date
        while date_split_end < end_date:
            date_split_start = start_date + timedelta(days = i * split_size + i)
            date_split_end = min(end_date, date_split_start + timedelta(days = split_size))
            date_splits.append((date_split_start, date_split_end))
            i += 1

        cols = ['game_pk', 'game_date', 'inning_topbot', 'batter', 'stand', 'pitcher', 'p_throws', 'estimated_ba_using_speedangle', 'events']
        df = pd.DataFrame(columns = cols)
        print('-' * 50)
        for date_split in date_splits:
            # Filter pitches to only pitches that end at bats
            # Construct URL and send request
            print('Fetching at bats from ', date_split[0].strftime('%Y-%m-%d'), ' to ', date_split[1].strftime('%Y-%m-%d'), '...', sep = '')
            url_params = {
                'all': 'true', 'hfAB': requests.utils.quote('|'.join(StatcastData.__AT_BAT_FINAL_EVENTS__).replace('_', '\\.' * 2) + '|'),
                'hfGT': 'R%7C', 'hfSea': requests.utils.quote('|'.join({str(date_split[0].year), str(date_split[1].year)}) + '|'),
                'player_type': 'batter', 'game_date_lt': date_split[1].strftime('%Y-%m-%d'), 'game_date_gt': date_split[0].strftime('%Y-%m-%d'),
                'min_pitches': 0, 'min_results': 0, 'group_by': 'name', 'sort_col': 'pitches', 'player_event_sort': 'api_p_release_speed',
                'sort_order': 'desc', 'min_pas': '0', 'type': 'details'
            }
            url = f'https://baseballsavant.mlb.com/statcast_search/csv?{"".join([f"{k}={v}&" for k, v in url_params.items()])}'

            # Convert CSV to dataframe and filter to wanted columns
            df2, baseball_savant_response, attempt = pd.DataFrame(columns = cols), None, 1
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
        return df \
            .rename({'estimated_ba_using_speedangle': 'xBA'}, axis = 1) \
            .sort_values(by = ['game_date', 'game_pk'], ignore_index = True) \
            .drop(['inning_topbot', 'stand', 'p_throws'], axis = 1)

    @staticmethod
    def add_season_sprint_speeds_to_db(year: int):
        print('-' * 80)
        print('Adding/Updating sprint speeds for the', year, 'season')
        new_sprint_speeds_df = StatcastData.get_sprint_speed_csv(year)
        # Delete existing entries
        print('Deleted', __DB__.sprintSpeeds.delete_many({'year': year}).deleted_count, 'sprint speeds')
        # Add new entries
        __DB__.sprintSpeeds.insert_many([{k: v for k, v in row.items() if pd.notnull(v)} for row in new_sprint_speeds_df.to_dict('records')])
        print('Added', len(new_sprint_speeds_df.index), 'sprint speeds')
        print('-' * 80)

    @staticmethod
    def get_sprint_speed_csv(year = 2023) -> pd.DataFrame:
        url = f'https://baseballsavant.mlb.com/leaderboard/sprint_speed?min_season={year}&max_season={year}&position=&team=&min=0&csv=true'
        baseball_savant_response = get(url)
        df = pd.read_csv(io.StringIO(baseball_savant_response.content.decode('utf-8')), usecols = ['player_id', 'hp_to_1b', 'sprint_speed'])
        df['year'] = year
        return df.rename({'player_id': 'batter', 'sprint_speed': 'speed'}, axis = 1)

    def batter_game_agg(self) -> pd.DataFrame:
        df = self.__df__.copy()
        df['pa'] = 1
        agg_df = df.groupby(['game_date', 'game_pk', 'home', 'batter'])[['pa', 'xBA', 'hit', 'bip']].sum()
        agg_df['G'] = 1
        agg_df['HG'] = agg_df['hit'] >= 1
        agg_df['xHG'] = agg_df['xBA'] >= 1
        return agg_df.rename({'xBA': 'xH'}, axis = 1).reset_index()

    def batter_season_agg(game_agg_df: pd.DataFrame):
        agg_df = game_agg_df.groupby('batter')[['G', 'HG', 'xHG', 'pa', 'hit', 'xH', 'bip']].sum()
        agg_df['H/PA'] = agg_df['hit'] / agg_df['pa']
        agg_df['xH/PA'] = agg_df['xH'] / agg_df['pa']
        agg_df['H/G'] = agg_df['hit'] / agg_df['G']
        agg_df['xH/G'] = agg_df['xH'] / agg_df['G']
        agg_df['H%'] = agg_df['HG'] / agg_df['G']
        agg_df['xH%'] = agg_df['xHG'] / agg_df['G']
        return agg_df[['G', 'pa', 'H/PA', 'xH/PA', 'H/G', 'xH/G', 'H%', 'xH%']].reset_index()

    def __enrich_with_stats_api_game_data__(self):
        games_df = pd.concat(
            [MLBData.get_stats_api_games(year = year) for year in self.__df__['game_date'].apply(lambda x: str(x).split('-')[0]).unique()],
            ignore_index = True
        )
        self.__df__ = self.__df__.merge(games_df.drop('game_date', axis = 1), how = 'left', on = 'game_pk')
        self.__df__['primary_pitcher'] = self.__df__.apply(lambda row: row['pitcher'] in [row['away_starter'], row['home_starter']], axis = 1)
        self.__df__['lineup'] = self.__df__.apply(lambda row: row['home_lineup'] if row['home'] else row['away_lineup'], axis = 1)
        self.__df__['lineup'] = self.__df__ \
            .apply(lambda row: row['lineup'].index(row['batter']) + 1 if row['batter'] in row['lineup'] else 10, axis = 1)
        self.__df__.drop(['away_starter', 'home_starter', 'away_lineup', 'home_lineup'], axis = 1, inplace = True)