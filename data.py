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
__DB__ = pymongo.MongoClient(f'mongodb+srv://peteb206:{env.get("MONGODB_PASSWORD")}@btscluster.tp9p0.mongodb.net').get_database('bts')

def month_start_and_end(year: int, month: int) -> tuple[date, date]:
    return date(year = year, month = month, day = 1), date(year = year, month = month + 1, day = 1) - timedelta(days = 1)

def date_to_datetime(date: date) -> datetime:
    return datetime.combine(date, datetime.min.time())

class MLBData:

    @staticmethod
    def get_stats_api_games(year: int, date: date):
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
                    'year': game['season'],
                    'game_date': game_date['date'],
                    'game_time': f'{hour if hour < 13 else hour - 12}:{"0" if minute < 10 else ""}{minute} {"PM" if hour > 11 else "AM"}',
                    'game_pk': game['gamePk'],
                    # 'away_team_id': game['teams']['away']['team']['id'],
                    # 'home_team_id': game['teams']['home']['team']['id'],
                    'away_team': game['teams']['away']['team']['abbreviation'],
                    'home_team': game['teams']['home']['team']['abbreviation'],
                    'away_starter': game['teams']['away']['probablePitcher']['id'] if 'probablePitcher' in game['teams']['away'].keys() else None,
                    'home_starter': game['teams']['home']['probablePitcher']['id'] if 'probablePitcher' in game['teams']['home'].keys() else None,
                    'away_lineup': ([x['id'] for x in game['lineups']['awayPlayers']] if 'awayPlayers' in game['lineups'].keys() else []) \
                        if 'lineups' in game.keys() else [],
                    'home_lineup': ([x['id'] for x in game['lineups']['homePlayers']] if 'homePlayers' in game['lineups'].keys() else []) \
                        if 'lineups' in game.keys() else [],
                    'status': game['status']['detailedState']
                })
        return pd.DataFrame(games_list)

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

class StatcastData:
    __AT_BAT_FINAL_EVENTS__ = [
        'single', 'double', 'triple', 'home_run', 'field_out', 'strikeout', 'strikeout_double_play', 'walk', 'double_play', 'field_error',
        'grounded_into_double_play', 'fielders_choice', 'fielders_choice_out', 'batter_interference', 'catcher_interf', 'force_out', 'hit_by_pitch',
        'intent_walk', 'sac_bunt', 'sac_bunt_double_play', 'sac_fly', 'sac_fly_double_play', 'triple_play'
    ]

    def __init__(self):
        at_bats_df: pd.DataFrame = find_pandas_all(__DB__.atBats, dict(), projection = {'_id': False})

        # Replace missing xBA values with median for that event
        at_bats_df = at_bats_df \
            .merge(at_bats_df.groupby('game_pk')['xBA'].sum().reset_index(), how = 'left', on = 'game_pk', suffixes = ('', '_game_sum')) \
            .merge(at_bats_df.groupby('events')['xBA'].median().reset_index(), how = 'left', on = 'events', suffixes = ('', '_med')) \
            .fillna({'xBA_game_sum': 0, 'xBA_med': 0})
        at_bats_df['xBA'] = at_bats_df.apply(lambda row: row['xBA_med'] if row['xBA_game_sum'] == 0 else row['xBA'], axis = 1)
        at_bats_df['H'] = at_bats_df['events'] < 4
        at_bats_df['BIP'] = at_bats_df['xBA_med'] > 0
        at_bats_df.drop(['events', 'xBA_med', 'xBA_game_sum'], axis = 1, inplace = True)

        # Combine with other collections
        ## At bat level
        self.games_df: pd.DataFrame = find_pandas_all(__DB__.games, dict(), projection = {'_id': False})
        self.at_bats_df = at_bats_df.merge(self.games_df[['game_pk', 'away_starter', 'home_starter']], on = 'game_pk')
        # assert len(self.at_bats_df.index) == len(at_bats_df.index)
        self.at_bats_df['starter'] = self.at_bats_df.apply(lambda row: row['pitcher'] in [row['away_starter'], row['home_starter']], axis = 1)
        self.at_bats_df.drop(['away_starter', 'home_starter'], axis = 1, inplace = True)

        ## Game level
        self.games_df['year'] = self.games_df['game_date'].apply(lambda x: x.year)
        lineup_slots_df: pd.DataFrame = find_pandas_all(__DB__.lineupSlots, dict(), projection = {'_id': False})
        self.lineup_slots_df = lineup_slots_df \
            .melt(id_vars = ['game_pk', 'lineup'], value_vars = ['away', 'home'], var_name = 'home', value_name = 'batter')
        self.lineup_slots_df['home'] = self.lineup_slots_df['home'] == 'home'
        batter_game_agg_df = self.batter_game_agg()
        self.batter_games_df = batter_game_agg_df \
            .merge(self.games_df, how = 'left', on = 'game_pk') \
            .merge(self.lineup_slots_df, how = 'left', on = ['game_pk', 'batter', 'home']) \
            .merge(find_pandas_all(__DB__.sprintSpeeds, dict(), projection = {'_id': False}), how = 'left', on = ['batter', 'year']) \
            .sort_values(by = ['game_date', 'game_pk', 'home', 'lineup'], ignore_index = True)
        self.batter_games_df['team'] = self.batter_games_df.apply(lambda row: row['home_team'] if row['home'] else row['away_team'], axis = 1)
        self.batter_games_df['opponent'] = self.batter_games_df.apply(lambda row: row['away_team'] if row['home'] else row['home_team'], axis = 1)
        self.batter_games_df.drop(['away_team', 'home_team', 'away_starter', 'home_starter'], axis = 1, inplace = True)
        # assert len(self.batter_games_df.index) == len(batter_game_agg_df.index)

    @staticmethod
    def update_db(year: int):
        # At Bats
        new_at_bat_df = StatcastData.add_at_bats_to_db(start_date = date(year, 7 if year == 2020 else 3, 23), end_date = date(year, 10, 6))
        # Sprint Speeds
        StatcastData.add_season_sprint_speeds_to_db(year)
        # Game dates and starting pitchers
        game_pks = new_at_bat_df['game_pk'].unique().tolist()
        print('-' * 50)
        game_dates_df = new_at_bat_df[['game_date', 'game_pk', 'away_team', 'home_team']].drop_duplicates().sort_values(by = ['game_date', 'game_pk'])
        starting_pitchers_df = new_at_bat_df.sort_values(['game_pk', 'home', 'at_bat']).groupby(['game_pk', 'home'])['pitcher'].first() \
            .reset_index().pivot(index = 'game_pk', columns = 'home')['pitcher'] \
            .rename({False: 'away_starter', True: 'home_starter'}, axis = 1).reset_index()
        games_df = game_dates_df.merge(starting_pitchers_df, on = 'game_pk')
        print(f'Deleted {"{:,}".format(__DB__.games.delete_many({"game_pk": {"$in": game_pks}}).deleted_count)} games')
        __DB__.games.insert_many(games_df.to_dict('records'))
        print('Added', '{:,}'.format(len(games_df.index)), 'games')
        # Lineup slots
        print('-' * 50)
        lineups_df = new_at_bat_df.sort_values(['game_pk', 'home', 'at_bat']).drop_duplicates(subset = ['game_pk', 'batter'])
        lineups_df['lineup'] = lineups_df.groupby(['game_pk', 'home']).cumcount() + 1
        lineups_df = lineups_df[lineups_df['lineup'] < 10].pivot(index = ['game_pk', 'lineup'], columns = 'home')['batter']
        lineups_df = lineups_df.rename({False: 'away', True: 'home'}, axis = 1).reset_index()
        print(f'Deleted {"{:,}".format(__DB__.lineupSlots.delete_many({"game_pk": {"$in": game_pks}}).deleted_count)} lineup slots')
        __DB__.lineupSlots.insert_many(lineups_df.to_dict('records'))
        print('Added', '{:,}'.format(len(lineups_df.index)), 'lineup slots')
        print('-' * 80)

    @staticmethod
    def add_at_bats_to_db(start_date = date(2023, 4, 1), end_date = date(2023, 4, 30)) -> pd.DataFrame:
        end_date = min(end_date, date.today() - timedelta(days = 1))
        print('-' * 80)
        print('Adding/Updating at bats from', start_date.strftime('%Y-%m-%d'), 'to', end_date.strftime('%Y-%m-%d'))
        new_at_bat_df = StatcastData.get_statcast_csv(start_date = start_date, end_date = end_date) \
            .sort_values(by = ['game_pk', 'at_bat'], ignore_index = True)
        # Delete existing entries
        deleted_count = __DB__.atBats.delete_many({'game_pk': {'$in': new_at_bat_df['game_pk'].unique().tolist()}}).deleted_count
        print('Deleted', '{:,}'.format(deleted_count), 'at bats')
        # Add new entries
        new_at_bat_df['game_date'] = new_at_bat_df['game_date'].apply(lambda x: date_to_datetime(x))
        __DB__.atBats.insert_many([{k: v for k, v in row.items() if pd.notnull(v)} for row in \
                                   new_at_bat_df.drop(['game_date', 'away_team', 'home_team'], axis = 1).to_dict('records')])
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

        cols = [
            'game_pk', 'game_date', 'at_bat_number', 'inning_topbot', 'batter', 'stand', 'pitcher', 'p_throws', 'away_team', 'home_team',
            'estimated_ba_using_speedangle', 'events'
        ]
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
            .rename({'estimated_ba_using_speedangle': 'xBA', 'at_bat_number': 'at_bat'}, axis = 1) \
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
        df = self.at_bats_df.copy()
        df['PA'] = 1
        agg_df = df.groupby(['game_pk', 'home', 'batter'])[['PA', 'xBA', 'H', 'BIP']].sum()
        agg_df['G'] = 1
        agg_df['HG'] = agg_df['H'] >= 1
        agg_df['xHG'] = agg_df['xBA'] >= 1
        return agg_df.rename({'xBA': 'xH'}, axis = 1).reset_index()

    @staticmethod
    def batter_span_agg(game_agg_df: pd.DataFrame):
        agg_df = game_agg_df.groupby('batter')[['G', 'HG', 'xHG', 'PA', 'H', 'xH', 'BIP']].sum()
        agg_df['H/PA'] = agg_df['H'] / agg_df['PA']
        agg_df['xH/PA'] = agg_df['xH'] / agg_df['PA']
        agg_df['H/G'] = agg_df['H'] / agg_df['G']
        agg_df['xH/G'] = agg_df['xH'] / agg_df['G']
        agg_df['H%'] = agg_df['HG'] / agg_df['G']
        agg_df['xH%'] = agg_df['xHG'] / agg_df['G']
        return agg_df[['G', 'PA', 'H/PA', 'xH/PA', 'H/G', 'xH/G', 'H%', 'xH%']].reset_index()

class GameDay(StatcastData):
    def __init__(self, game_date: date, statcast_data: StatcastData):
        print()

    def features_df(self):
        return pd.DataFrame()

    def labels_series(self):
        return pd.Series()

if __name__ == '__main__':
    statcast_data = StatcastData()
    print()