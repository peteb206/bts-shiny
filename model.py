# bts-shiny
import data

# third-party
import pandas as pd
import pickle
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

class BTSBatterClassifier:
    '''
    # Example

    ## Read data from MongoDB
    games_df: pd.DataFrame = find_pandas_all(__DB__.games, {'game_date': {'$gt': datetime(now.year - 2, 1, 1)}}, projection = {'_id': False})
    at_bats_df: pd.DataFrame = find_pandas_all(__DB__.atBats, {'game_pk': {'$in': games_df.game_pk.unique().tolist()}}, projection = {'_id': False})
    speed_df: pd.DataFrame = find_pandas_all(__DB__.sprintSpeeds, {'year': {'$gte': datetime.now().year}}, projection = {'_id': False})

    games_df.set_index('game_pk', inplace = True)
    at_bats_df.set_index(['game_pk', 'home', 'at_bat', 'batter', 'pitcher'], inplace = True)
    at_bats_df.sort_index(inplace = True)
    speed_df.set_index(['year', 'batter'], inplace = True)

    ## Build classifier
    enhanced_at_bats = enhance_at_bats(at_bats_df = at_bats_df, games_df = games_df, speed_df = speed_df)
    log_reg = BTSBatterClassifier(LogisticRegressionCV(cv = 10, random_state = 57), at_bats_df, 'log_reg')
    ### Fit classifier
    log_reg.fit_model(scale_features = True, perform_pca = True)
    ### Simulate results on test data
    log_reg.simulate_results()
    ### Get predictions for today
    log_reg.todays_predictions()
    '''
    SIGNIFICANT_GAMES, MINIMUM_GAMES = 50, 25
    SIGNIFICANT_PAS, MINIMUM_PAS = 200, 100
    PKL_DIR = '/Users/peteberryman/Desktop/bts-shiny/models'

    def __init__(self, clf, at_bats_df = pd.DataFrame(), pkl_name = ''):
        assert pkl_name != ''
        self.clf = clf
        self.at_bats_df = at_bats_df
        self.__added_todays_batters_to_at_bats__ = False
        # Train/test split
        self.X_train, self.X_test = pd.DataFrame(), pd.DataFrame()
        self.y_train, self.y_test = pd.Series(dtype = int), pd.Series(dtype = int)
        self.model_input_df = pd.DataFrame()
        self.pkl_name = pkl_name

    def build_model_input_df(self):
        game_days_df = self.at_bats_df \
            .groupby(['game_date', 'game_pk', 'home', 'team', 'opponent', 'batter']) \
                .agg({'opp_sp': ['count', 'first'], 'H': max, 'hp_to_1b': 'first'}) #, 'bats': 'first', 'lineup': 'first'
        game_days_df.columns = ['PA' if col[1] == 'count' else col[0] for col in game_days_df.columns]
        game_days_df.set_index('opp_sp', append = True, inplace = True)

        self.model_input_df = game_days_df \
            .merge(self.batter_per_game_agg(), left_index = True, right_index = True) \
                .merge(self.batter_per_game_agg(split_cols = ['home']), left_index = True, right_index = True, suffixes = ('', '_home_away')) \
                    .merge(self.batter_per_pa_agg(), left_index = True, right_index = True) \
                        .merge(self.batter_per_pa_agg(split_cols = ['rhp']), left_index = True, right_index = True, suffixes = ('', '_vs_hp')) \
                            .merge(self.pitcher_per_bf_agg().rename_axis(index = {'pitcher': 'opp_sp'}), left_index = True, right_index = True) \
                                .merge(self.pitcher_per_bf_agg(split_cols = ['rhb']).rename_axis(index = {'pitcher': 'opp_sp'}),
                                       left_index = True, right_index = True, suffixes = ('', '_vs_hb')) \
                                    .merge(self.bullpen_per_bf_agg(), left_index = True, right_index = True, suffixes = ('', '_bullpen')) \
                                        .merge(self.batter_per_pa_agg(split_cols = ['pitcher']), left_index = True, right_index = True,
                                               suffixes = ('', '_vs_opp_sp')) \
                                            .reorder_levels(['game_date', 'game_pk', 'home', 'team', 'opponent', 'opp_sp', 'batter']) \
                                                .sort_index() \
                                                    .query(f'''
                                                        PA >= 3 and G_last_{self.SIGNIFICANT_GAMES}G >= {self.MINIMUM_GAMES} and
                                                        PA_last_{self.SIGNIFICANT_PAS}PA >= {self.MINIMUM_PAS} and
                                                        BF_last_{self.SIGNIFICANT_PAS}BF >= {self.MINIMUM_PAS} and
                                                        BF_last_{self.SIGNIFICANT_PAS}BF_bullpen >= {self.MINIMUM_PAS}
                                                    '''.replace('\n', '')) \
                                                        .drop(['PA', f'G_last_{self.SIGNIFICANT_GAMES}G',
                                                               f'G_last_{self.SIGNIFICANT_GAMES}G_home_away', f'PA_last_{self.SIGNIFICANT_PAS}PA',
                                                               f'PA_last_{self.SIGNIFICANT_PAS}PA_vs_hp', f'BF_last_{self.SIGNIFICANT_PAS}BF',
                                                               f'BF_last_{self.SIGNIFICANT_PAS}BF_vs_hb',
                                                               f'BF_last_{self.SIGNIFICANT_PAS}BF_bullpen'], axis = 1) \
                                                            .dropna()

    def batter_per_game_agg(self, split_cols: list[str] = [], significant_games = SIGNIFICANT_GAMES) -> pd.DataFrame:
        group_by = ['batter'] + split_cols
        batter_games_df = self.at_bats_df \
            .groupby(['game_date', 'game_pk', 'home', 'batter']) \
                .agg({'xBA': sum, 'H': sum, 'BIP': sum, 'statcast_tracked': ['count', 'first']})
        batter_games_df.columns = ['PA' if (col[0] == 'statcast_tracked') & (col[1] == 'count') else col[0] for col in batter_games_df.columns]
        batter_games_df = batter_games_df.loc[batter_games_df.PA >= 3] # don't include partial games
        # display(batter_games_df[batter_games_df.index.isin([668804], level = 3)])
        batter_games_df['G'] = 1
        batter_games_df['HG'] = batter_games_df.H >= 1
        batter_games_df['xHG'] = batter_games_df.xBA >= 1
        batter_games_df = batter_games_df.astype({'BIP': int, 'statcast_tracked': int})
        batter_games_df = batter_games_df.astype(float).groupby(group_by).cumsum().groupby(group_by).shift(1).fillna(0)
        batter_games_df.rename({col: 'cumulG_statcast' if col == 'statcast_tracked' else f'cumul{col}' for col in batter_games_df.columns},
                               axis = 1, inplace = True)
        batter_games_df[f'G_last_{significant_games}G'] = \
            (batter_games_df.cumulG - batter_games_df.groupby(group_by).cumulG.shift(significant_games)) \
                .combine_first(batter_games_df.cumulG).astype(int)
        batter_games_df[f'statcast_G_last_{significant_games}G'] = \
            (batter_games_df.cumulG_statcast - batter_games_df.groupby(group_by).cumulG_statcast.shift(significant_games)) \
                .combine_first(batter_games_df.cumulG_statcast)
        batter_games_df[f'HG%_last_{significant_games}G'] = \
            (batter_games_df.cumulHG - batter_games_df.groupby(group_by).cumulHG.shift(significant_games)).combine_first(batter_games_df.cumulHG) \
                .div(batter_games_df[f'G_last_{significant_games}G'])
        batter_games_df[f'xHG%_last_{significant_games}G'] = \
            (batter_games_df.cumulxHG - batter_games_df.groupby(group_by).cumulxHG.shift(significant_games)).combine_first(batter_games_df.cumulxHG) \
                .div(batter_games_df[f'statcast_G_last_{significant_games}G'])
        batter_games_df[f'PA/G_last_{significant_games}G'] = \
            (batter_games_df.cumulPA - batter_games_df.groupby(group_by).cumulPA.shift(significant_games)).combine_first(batter_games_df.cumulPA) \
                .div(batter_games_df[f'G_last_{significant_games}G'])
        batter_games_df[f'BIP/G_last_{significant_games}G'] = \
            (batter_games_df.cumulBIP - batter_games_df.groupby(group_by).cumulBIP.shift(significant_games)).combine_first(batter_games_df.cumulBIP) \
                .div(batter_games_df[f'G_last_{significant_games}G'])
        batter_games_df[f'H/G_last_{significant_games}G'] = \
            (batter_games_df.cumulH - batter_games_df.groupby(group_by).cumulH.shift(significant_games)).combine_first(batter_games_df.cumulH) \
                .div(batter_games_df[f'G_last_{significant_games}G'])
        batter_games_df[f'xH/G_last_{significant_games}G'] = \
            (batter_games_df.cumulxBA - batter_games_df.groupby(group_by).cumulxBA.shift(significant_games)).combine_first(batter_games_df.cumulxBA) \
                .div(batter_games_df[f'statcast_G_last_{significant_games}G'])
        # display(batter_games_df[batter_games_df.index.isin([668804], level = 3)])
        batter_games_df.drop([col for col in batter_games_df.columns if (col.startswith('cumul')) | (col.startswith('statcast'))],
                             axis = 1, inplace = True)
        return batter_games_df.fillna(0)

    def batter_per_pa_agg(self, split_cols: list[str] = [], significant_pas = SIGNIFICANT_PAS) -> pd.DataFrame:
        group_by = ['batter'] + split_cols
        pas_df = self.at_bats_df.fillna({'xBA': 0})
        if len(split_cols) > 0:
            pas_df.set_index([col for col in split_cols if col in pas_df.columns], append = True, inplace = True)
        # display(pas_df[pas_df.index.get_level_values('batter') == 660670])
        pas_df['PA'] = 1
        pas_df = pas_df.loc[:, ['PA', 'xBA', 'H', 'BIP', 'statcast_tracked']].astype(float) \
            .groupby(group_by).cumsum().groupby(group_by).shift(1).fillna(0)
        pas_df.rename({col: 'cumulPA_statcast' if col == 'statcast_tracked' else f'cumul{col}' for col in pas_df.columns}, axis = 1, inplace = True)
        pas_df[f'PA_last_{significant_pas}PA'] = \
            (pas_df.cumulPA - pas_df.groupby(group_by).cumulPA.shift(significant_pas)).combine_first(pas_df.cumulPA).astype(int)
        pas_df[f'statcast_PA_last_{significant_pas}PA'] = \
            (pas_df.cumulPA_statcast - pas_df.groupby(group_by).cumulPA_statcast.shift(significant_pas)).combine_first(pas_df.cumulPA_statcast)
        pas_df[f'BIP/PA_last_{significant_pas}PA'] = \
            (pas_df.cumulBIP - pas_df.groupby(group_by).cumulBIP.shift(significant_pas)).combine_first(pas_df.cumulBIP) \
                .div(pas_df[f'PA_last_{significant_pas}PA'])
        pas_df[f'H/PA_last_{significant_pas}PA'] = \
            (pas_df.cumulH - pas_df.groupby(group_by).cumulH.shift(significant_pas)).combine_first(pas_df.cumulH) \
                .div(pas_df[f'PA_last_{significant_pas}PA'])
        pas_df[f'xH/PA_last_{significant_pas}PA'] = \
            (pas_df.cumulxBA - pas_df.groupby(group_by).cumulxBA.shift(significant_pas)).combine_first(pas_df.cumulxBA) \
                .div(pas_df[f'statcast_PA_last_{significant_pas}PA'])
        # display(pas_df[pas_df.index.get_level_values('batter') == 660670])
        pas_df.drop([col for col in pas_df.columns if (col.startswith('cumul')) | (col.startswith('statcast'))], axis = 1, inplace = True)
        return pas_df.fillna(0).groupby(['game_date', 'game_pk', 'batter']).first()

    def pitcher_per_bf_agg(self, split_cols: list[str] = [], significant_bfs = SIGNIFICANT_PAS):
        group_by = ['pitcher'] + split_cols
        bfs_df = self.at_bats_df[self.at_bats_df.index.get_level_values('pitcher') != 0].fillna({'xBA': 0})
        if len(split_cols) > 0:
            bfs_df.set_index([col for col in split_cols if col in bfs_df.columns], append = True, inplace = True)
        bfs_df['BF'] = 1
        bfs_df['K'] = bfs_df.events.isin(['strikeout', 'strikeout_double_play'])
        bfs_df['BB'] = bfs_df.events.isin(['walk', 'hit_by_pitch'])
        # display(bfs_df[bfs_df.index.get_level_values('pitcher') == 457435])
        bfs_df = bfs_df.loc[:, ['BF', 'xBA', 'H', 'K', 'BB', 'statcast_tracked']].astype(float) \
            .groupby(group_by).cumsum() \
                .groupby(group_by).shift(1).fillna(0)
        bfs_df.rename({col: 'cumulBF_statcast' if col == 'statcast_tracked' else f'cumul{col}' for col in bfs_df.columns}, axis = 1, inplace = True)
        bfs_df[f'BF_last_{significant_bfs}BF'] = \
            (bfs_df.cumulBF - bfs_df.groupby(group_by).cumulBF.shift(significant_bfs)).combine_first(bfs_df.cumulBF).astype(int)
        bfs_df[f'statcast_BF_last_{significant_bfs}BF'] = \
            (bfs_df.cumulBF_statcast - bfs_df.groupby(group_by).cumulBF_statcast.shift(significant_bfs)).combine_first(bfs_df.cumulBF_statcast)
        bfs_df[f'K%_last_{significant_bfs}BF'] = \
            (bfs_df.cumulK - bfs_df.groupby(group_by).cumulK.shift(significant_bfs)).combine_first(bfs_df.cumulK) \
                .div(bfs_df[f'BF_last_{significant_bfs}BF'])
        bfs_df[f'BB%_last_{significant_bfs}BF'] = \
            (bfs_df.cumulBB - bfs_df.groupby(group_by).cumulBB.shift(significant_bfs)).combine_first(bfs_df.cumulBB) \
                .div(bfs_df[f'BF_last_{significant_bfs}BF'])
        bfs_df[f'H/PA_last_{significant_bfs}BF'] = \
            (bfs_df.cumulH - bfs_df.groupby(group_by).cumulH.shift(significant_bfs)).combine_first(bfs_df.cumulH) \
                .div(bfs_df[f'BF_last_{significant_bfs}BF'])
        bfs_df[f'xH/PA_last_{significant_bfs}BF'] = \
            (bfs_df.cumulxBA - bfs_df.groupby(group_by).cumulxBA.shift(significant_bfs)).combine_first(bfs_df.cumulxBA) \
                .div(bfs_df[f'statcast_BF_last_{significant_bfs}BF'])
        # display(bfs_df[bfs_df.index.get_level_values('pitcher') == 457435])
        bfs_df.drop([col for col in bfs_df.columns if (col.startswith('cumul')) | (col.startswith('statcast'))], axis = 1, inplace = True)
        return bfs_df.fillna(0).groupby(['game_date', 'game_pk', 'pitcher']).first()

    def bullpen_per_bf_agg(self, significant_bfs = SIGNIFICANT_PAS):
        bfs_df = self.at_bats_df.loc[self.at_bats_df.opp_sp != self.at_bats_df.index.get_level_values('pitcher')].fillna({'xBA': 0})
        bfs_df['BF'] = 1
        bfs_df['K'] = bfs_df.events.isin(['strikeout', 'strikeout_double_play'])
        bfs_df['BB'] = bfs_df.events.isin(['walk', 'hit_by_pitch'])
        bfs_df = bfs_df.loc[:, ['BF', 'xBA', 'H', 'K', 'BB', 'statcast_tracked']].astype(float) \
            .groupby('opponent').cumsum(numeric_only = True) \
                .groupby('opponent').shift(1).fillna(0)
        bfs_df.rename({col: 'cumulBF_statcast' if col == 'statcast_tracked' else f'cumul{col}' for col in bfs_df.columns}, axis = 1, inplace = True)
        # display(bfs_df[bfs_df.index.get_level_values('opponent') == 'CHC'])
        bfs_df[f'BF_last_{significant_bfs}BF'] = \
            (bfs_df.cumulBF - bfs_df.groupby('opponent').cumulBF.shift(significant_bfs)).combine_first(bfs_df.cumulBF).astype(int)
        bfs_df[f'statcast_BF_last_{significant_bfs}BF'] = \
            (bfs_df.cumulBF_statcast - bfs_df.groupby('opponent').cumulBF_statcast.shift(significant_bfs)).combine_first(bfs_df.cumulBF_statcast)
        bfs_df[f'K%_last_{significant_bfs}BF'] = \
            (bfs_df.cumulK - bfs_df.groupby('opponent').cumulK.shift(significant_bfs)).combine_first(bfs_df.cumulK) \
                .div(bfs_df[f'BF_last_{significant_bfs}BF'])
        bfs_df[f'BB%_last_{significant_bfs}BF'] = \
            (bfs_df.cumulBB - bfs_df.groupby('opponent').cumulBB.shift(significant_bfs)).combine_first(bfs_df.cumulBB) \
                .div(bfs_df[f'BF_last_{significant_bfs}BF'])
        bfs_df[f'H/PA_last_{significant_bfs}BF'] = \
            (bfs_df.cumulH - bfs_df.groupby('opponent').cumulH.shift(significant_bfs)).combine_first(bfs_df.cumulH) \
                .div(bfs_df[f'BF_last_{significant_bfs}BF'])
        bfs_df[f'xH/PA_last_{significant_bfs}BF'] = \
            (bfs_df.cumulxBA - bfs_df.groupby('opponent').cumulxBA.shift(significant_bfs)).combine_first(bfs_df.cumulxBA) \
                .div(bfs_df[f'statcast_BF_last_{significant_bfs}BF'])
        # display(bfs_df[bfs_df.index.get_level_values('opponent') == 'CHC'])
        bfs_df.drop([col for col in bfs_df.columns if (col.startswith('cumul')) | (col.startswith('statcast'))], axis = 1, inplace = True)
        return bfs_df.fillna(0).groupby(['game_date', 'game_pk', 'opponent']).first()

    def fit_model(self, scale_features = True, perform_pca = False):
        self.build_model_input_df()

        # Correlation matrix
        '''
        correlation_matrix = self.model_input_df.loc[:, self.model_input_df.dtypes == float].corr()
        for row_num in range(len(correlation_matrix.index)):
            for col_num in range(len(correlation_matrix.columns)):
                if row_num <= col_num:
                    correlation_matrix.iloc[row_num, col_num] = None
        correlation_matrix.style.background_gradient(cmap = 'bwr', axis = None, vmin = -1, vmax = 1).highlight_null(color = '#f1f1f1').format(precision = 2) \
            .set_table_styles([
                {'selector': 'th.col_heading', 'props': [('text-align', 'left'), ('writing-mode', 'vertical-rl'), ('transform', 'rotateZ(192deg)')]},
                {'selector': 'td, th', 'props': [('border', '1px solid black !important')]}
            ])
        try:
            display(correlation_matrix)
        except:
            import IPython.display as display
            display.display_html(correlation_matrix)
        '''

        # Scale data https://stackoverflow.com/a/59164898
        features_df = self.model_input_df.drop('H', axis = 1)
        scaler = None
        if scale_features:
            scaler = StandardScaler().fit(features_df)
            features_df = pd.DataFrame(scaler.transform(features_df), index = features_df.index, columns = features_df.columns)

        # Train/test split
        train_game_dates, test_game_dates = train_test_split(features_df.index.get_level_values('game_date').unique(),
                                                             test_size = 0.3, random_state = 57)
        self.X_train = features_df.loc[features_df.index.get_level_values('game_date').isin(train_game_dates)]
        self.X_test = features_df.loc[features_df.index.get_level_values('game_date').isin(test_game_dates)]
        self.y_train = self.model_input_df.loc[self.model_input_df.index.get_level_values('game_date').isin(train_game_dates), 'H']
        self.y_test = self.model_input_df.loc[self.model_input_df.index.get_level_values('game_date').isin(test_game_dates), 'H']

        # PCA https://datascience.stackexchange.com/a/55080
        pca = None
        if perform_pca:
            assert scale_features, 'PCA cannot be performed if features are not normalized'
            pca = PCA(n_components = 0.99, svd_solver = 'full', random_state = 57).fit(self.X_train)
            print('Initial', len(self.X_train.columns), 'features:', ', '.join(self.X_train.columns))
            print('PCA: # of features reduced from', len(self.X_train.columns), 'to', pca.n_components_)
            self.X_train = pd.DataFrame(pca.transform(self.X_train), index = self.X_train.index,
                                        columns = [f'PC{f + 1}' for f in range(pca.n_components_)])
            self.X_test = pd.DataFrame(pca.transform(self.X_test), index = self.X_test.index,
                                       columns = [f'PC{f + 1}' for f in range(pca.n_components_)])

        # Logistic Regression
        self.clf.fit(self.X_train, self.y_train)
        print('Score on training data:', round(self.clf.score(self.X_train, self.y_train), 3))
        print('Score on testing data:', round(self.clf.score(self.X_test, self.y_test), 3))

        # Dump pickle file
        pickle.dump((scaler, pca, self.clf), open(f'{self.PKL_DIR}/{self.pkl_name}.pkl', 'wb'))
        # Confirm success
        # print(pickle.load(open(f'{self.PKL_DIR}/{self.pkl_name}.pkl', 'rb')))

    def __add_todays_batters_to_at_bats__(self, todays_batters_df: pd.DataFrame):
        # Add dummy at bats for today's hitters
        if not self.__added_todays_batters_to_at_bats__:
            for at_bat in range(1, 4):
                todays_batters_at_bats_df = todays_batters_df.drop(['name', 'opp_sp_name'], axis = 1)
                todays_batters_at_bats_df['at_bat'] = at_bat
                todays_batters_at_bats_df['pitcher'] = todays_batters_at_bats_df.opp_sp if at_bat < 3 else 0
                todays_batters_at_bats_df.set_index(['at_bat', 'pitcher'], append = True, inplace = True)
                todays_batters_at_bats_df = todays_batters_at_bats_df.reorder_levels(self.at_bats_df.index.names)
                todays_batters_at_bats_df[['bats', 'xBA', 'events', 'rhb', 'rhp', 'H', 'BIP', 'statcast_tracked']] = \
                    self.at_bats_df.tail(len(todays_batters_at_bats_df.index)) \
                        .loc[:, ['bats', 'xBA', 'events', 'rhb', 'rhp', 'H', 'BIP', 'statcast_tracked']]
                todays_batters_at_bats_df = todays_batters_at_bats_df.astype(self.at_bats_df.dtypes)
                self.at_bats_df = pd.concat([self.at_bats_df, todays_batters_at_bats_df])
            self.__added_todays_batters_to_at_bats__ = True

    def simulate_results(self):
        test_df = self.X_test.copy()
        test_df['H_prob'] = self.clf.predict_proba(test_df)[:, -1]
        test_df['H'] = self.y_test.copy()

        top_two_picks_by_day_df = test_df.sort_values(['game_date', 'H_prob'], ascending = [True, False]).groupby('game_date').head(2)

        streak, streaks = 0, list()
        for _, row in top_two_picks_by_day_df.groupby('game_date').H.agg(['count', 'min']).iterrows():
            result = row['count'] if row['min'] > 0 else 0
            if result == 0:
                if streak > 0:
                    streaks.append(streak)
                streak = 0
            else:
                streak += result
        if streak > 0:
            streaks.append(streak)

        title = ''
        try:
            title = self.clf.__str__()
        except:
            title = self.pkl_name
        plt.hist(streaks, bins = list(range(1, 58)))
        plt.title('\n'.join([
            title,
            f'{100 * round(top_two_picks_by_day_df.H.astype(bool).mean(), 2)}% Pick Accuracy (assuming double-down every day)',
            f'Best Streak: {max(streaks) if len(streaks) > 0 else 0}'
        ]))
        plt.show()

    def todays_predictions(self):
        todays_batters_df = data.get_todays_batters()
        todays_batters_df = todays_batters_df.loc[todays_batters_df.opp_sp != 0]
        todays_batters_df = todays_batters_df.merge(self.at_bats_df.groupby('batter').hp_to_1b.last(), how = 'left', left_index = True,
                                                    right_index = True) # most recent time

        self.__add_todays_batters_to_at_bats__(todays_batters_df)
        self.build_model_input_df()

        todays_options_df = todays_batters_df.loc[:, ['name', 'lineup', 'opp_sp_name',]] \
            .merge(self.model_input_df, left_index = True, right_index = True)

        scaler, pca, clf = pickle.load(open(f'{self.PKL_DIR}/{self.pkl_name}.pkl', 'rb'))
        if scaler != None:
            todays_options_df = pd.DataFrame(scaler.transform(todays_options_df.loc[:, scaler.feature_names_in_]), index = todays_options_df.index,
                                             columns = scaler.feature_names_in_)
        if pca != None:
            todays_options_df = pd.DataFrame(pca.transform(todays_options_df), index = todays_options_df.index,
                                             columns = [f'PC{f + 1}' for f in range(pca.n_components_)])
        todays_options_df.reset_index(level = 'opp_sp', drop = True, inplace = True)
        todays_options_df.loc[:, 'H%'] = clf.predict_proba(todays_options_df)[:, -1]
        return todays_batters_df.merge(todays_options_df[['H%']], left_index = True, right_index = True).sort_values(by = 'H%', ascending = False)