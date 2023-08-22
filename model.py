# bts-shiny
import data

# third-party
import pandas as pd
import pickle
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegressionCV
import matplotlib.pyplot as plt
# import IPython.display as display

class LogisticRegression1:
    SIGNIFICANT_GAMES, MINIMUM_GAMES = 50, 25
    SIGNIFICANT_PAS, MINIMUM_PAS = 200, 100
    PKL_PATH = '/Users/peteberryman/Desktop/bts-shiny/models/log_reg.pkl'

    def __init__(self, at_bats_df: pd.DataFrame):
        self.at_bats_df = at_bats_df
        self.added_todays_batters_to_at_bats = False

    def batter_per_game_agg(self, significant_games = SIGNIFICANT_GAMES) -> pd.DataFrame:
        batter_games_df = self.at_bats_df \
            .groupby(['game_date', 'game_pk', 'batter']) \
                .agg({'xBA': sum, 'H': sum, 'BIP': sum, 'statcast_tracked': ['count', 'first']})
        batter_games_df.columns = ['PA' if (col[0] == 'statcast_tracked') & (col[1] == 'count') else col[0] for col in batter_games_df.columns]
        batter_games_df = batter_games_df.loc[batter_games_df.PA >= 3] # don't include partial games
        # display(batter_games_df[batter_games_df.index.isin([668804], level = 3)])
        batter_games_df['G'] = 1
        batter_games_df['HG'] = batter_games_df.H >= 1
        batter_games_df['xHG'] = batter_games_df.xBA >= 1
        batter_games_df = batter_games_df.astype({'BIP': int, 'statcast_tracked': int})
        batter_games_df = batter_games_df.astype(float).groupby('batter').cumsum().groupby('batter').shift(1).fillna(0)
        batter_games_df.rename({col: 'cumulG_statcast' if col == 'statcast_tracked' else f'cumul{col}' for col in batter_games_df.columns},
                               axis = 1, inplace = True)
        batter_games_df[f'G_last_{significant_games}G'] = \
            (batter_games_df.cumulG - batter_games_df.groupby('batter').cumulG.shift(significant_games)) \
                .combine_first(batter_games_df.cumulG).astype(int)
        batter_games_df[f'statcast_G_last_{significant_games}G'] = \
            (batter_games_df.cumulG_statcast - batter_games_df.groupby('batter').cumulG_statcast.shift(significant_games)) \
                .combine_first(batter_games_df.cumulG_statcast)
        batter_games_df[f'HG%_last_{significant_games}G'] = \
            (batter_games_df.cumulHG - batter_games_df.groupby('batter').cumulHG.shift(significant_games)).combine_first(batter_games_df.cumulHG) \
                .div(batter_games_df[f'G_last_{significant_games}G'])
        batter_games_df[f'xHG%_last_{significant_games}G'] = \
            (batter_games_df.cumulxHG - batter_games_df.groupby('batter').cumulxHG.shift(significant_games)).combine_first(batter_games_df.cumulxHG) \
                .div(batter_games_df[f'statcast_G_last_{significant_games}G'])
        batter_games_df[f'PA/G_last_{significant_games}G'] = \
            (batter_games_df.cumulPA - batter_games_df.groupby('batter').cumulPA.shift(significant_games)).combine_first(batter_games_df.cumulPA) \
                .div(batter_games_df[f'G_last_{significant_games}G'])
        batter_games_df[f'BIP/G_last_{significant_games}G'] = \
            (batter_games_df.cumulBIP - batter_games_df.groupby('batter').cumulBIP.shift(significant_games)).combine_first(batter_games_df.cumulBIP) \
                .div(batter_games_df[f'G_last_{significant_games}G'])
        batter_games_df[f'H/G_last_{significant_games}G'] = \
            (batter_games_df.cumulH - batter_games_df.groupby('batter').cumulH.shift(significant_games)).combine_first(batter_games_df.cumulH) \
                .div(batter_games_df[f'G_last_{significant_games}G'])
        batter_games_df[f'xH/G_last_{significant_games}G'] = \
            (batter_games_df.cumulxBA - batter_games_df.groupby('batter').cumulxBA.shift(significant_games)).combine_first(batter_games_df.cumulxBA) \
                .div(batter_games_df[f'statcast_G_last_{significant_games}G'])
        # display(batter_games_df[batter_games_df.index.isin([668804], level = 3)])
        batter_games_df.drop([col for col in batter_games_df.columns if (col.startswith('cumul')) | (col.startswith('statcast'))],
                             axis = 1, inplace = True)
        return batter_games_df.fillna(0)

    def batter_per_pa_agg(self, significant_pas = SIGNIFICANT_PAS) -> pd.DataFrame:
        pas_df = self.at_bats_df.fillna({'xBA': 0})
        # display(pas_df[pas_df.index.get_level_values('batter') == 660670])
        pas_df['PA'] = 1
        pas_df = pas_df.loc[:, ['PA', 'xBA', 'H', 'BIP', 'statcast_tracked']].astype(float) \
            .groupby('batter').cumsum().groupby('batter').shift(1).fillna(0)
        pas_df.rename({col: 'cumulPA_statcast' if col == 'statcast_tracked' else f'cumul{col}' for col in pas_df.columns}, axis = 1, inplace = True)
        pas_df[f'PA_last_{significant_pas}PA'] = \
            (pas_df.cumulPA - pas_df.groupby('batter').cumulPA.shift(significant_pas)).combine_first(pas_df.cumulPA).astype(int)
        pas_df[f'statcast_PA_last_{significant_pas}PA'] = \
            (pas_df.cumulPA_statcast - pas_df.groupby('batter').cumulPA_statcast.shift(significant_pas)).combine_first(pas_df.cumulPA_statcast)
        pas_df[f'BIP/PA_last_{significant_pas}PA'] = \
            (pas_df.cumulBIP - pas_df.groupby('batter').cumulBIP.shift(significant_pas)).combine_first(pas_df.cumulBIP) \
                .div(pas_df[f'PA_last_{significant_pas}PA'])
        pas_df[f'H/PA_last_{significant_pas}PA'] = \
            (pas_df.cumulH - pas_df.groupby('batter').cumulH.shift(significant_pas)).combine_first(pas_df.cumulH) \
                .div(pas_df[f'PA_last_{significant_pas}PA'])
        pas_df[f'xH/PA_last_{significant_pas}PA'] = \
            (pas_df.cumulxBA - pas_df.groupby('batter').cumulxBA.shift(significant_pas)).combine_first(pas_df.cumulxBA) \
                .div(pas_df[f'statcast_PA_last_{significant_pas}PA'])
        # display(pas_df[pas_df.index.get_level_values('batter') == 660670])
        pas_df.drop([col for col in pas_df.columns if (col.startswith('cumul')) | (col.startswith('statcast'))], axis = 1, inplace = True)
        return pas_df.fillna(0).groupby(['game_date', 'game_pk', 'batter']).first()

    def pitcher_per_bf_agg(self, significant_bfs = SIGNIFICANT_PAS):
        bfs_df = self.at_bats_df[self.at_bats_df.index.get_level_values('pitcher') != 0].fillna({'xBA': 0})
        bfs_df['BF'] = 1
        bfs_df['K'] = bfs_df.events.isin(['strikeout', 'strikeout_double_play'])
        bfs_df['BB'] = bfs_df.events == 'walk'
        # display(bfs_df[bfs_df.index.get_level_values('pitcher') == 457435])
        bfs_df = bfs_df.loc[:, ['BF', 'xBA', 'H', 'K', 'BB', 'statcast_tracked']].astype(float) \
            .groupby('pitcher').cumsum() \
                .groupby('pitcher').shift(1).fillna(0)
        bfs_df.rename({col: 'cumulBF_statcast' if col == 'statcast_tracked' else f'cumul{col}' for col in bfs_df.columns}, axis = 1, inplace = True)
        bfs_df[f'BF_last_{significant_bfs}BF'] = \
            (bfs_df.cumulBF - bfs_df.groupby('pitcher').cumulBF.shift(significant_bfs)).combine_first(bfs_df.cumulBF).astype(int)
        bfs_df[f'statcast_BF_last_{significant_bfs}BF'] = \
            (bfs_df.cumulBF_statcast - bfs_df.groupby('pitcher').cumulBF_statcast.shift(significant_bfs)).combine_first(bfs_df.cumulBF_statcast)
        bfs_df[f'K%_last_{significant_bfs}BF'] = \
            (bfs_df.cumulK - bfs_df.groupby('pitcher').cumulK.shift(significant_bfs)).combine_first(bfs_df.cumulK) \
                .div(bfs_df[f'BF_last_{significant_bfs}BF'])
        bfs_df[f'BB%_last_{significant_bfs}BF'] = \
            (bfs_df.cumulBB - bfs_df.groupby('pitcher').cumulBB.shift(significant_bfs)).combine_first(bfs_df.cumulBB) \
                .div(bfs_df[f'BF_last_{significant_bfs}BF'])
        bfs_df[f'H/PA_last_{significant_bfs}BF'] = \
            (bfs_df.cumulH - bfs_df.groupby('pitcher').cumulH.shift(significant_bfs)).combine_first(bfs_df.cumulH) \
                .div(bfs_df[f'BF_last_{significant_bfs}BF'])
        bfs_df[f'xH/PA_last_{significant_bfs}BF'] = \
            (bfs_df.cumulxBA - bfs_df.groupby('pitcher').cumulxBA.shift(significant_bfs)).combine_first(bfs_df.cumulxBA) \
                .div(bfs_df[f'statcast_BF_last_{significant_bfs}BF'])
        # display(bfs_df[bfs_df.index.get_level_values('pitcher') == 457435])
        bfs_df.drop([col for col in bfs_df.columns if (col.startswith('cumul')) | (col.startswith('statcast'))], axis = 1, inplace = True)
        return bfs_df.fillna(0).groupby(['game_date', 'game_pk', 'pitcher']).first()

    def bullpen_per_bf_agg(self, significant_bfs = SIGNIFICANT_PAS):
        bfs_df = self.at_bats_df.loc[self.at_bats_df.opp_sp != self.at_bats_df.index.get_level_values('pitcher')].fillna({'xBA': 0})
        bfs_df['BF'] = 1
        bfs_df['K'] = bfs_df.events.isin(['strikeout', 'strikeout_double_play'])
        bfs_df['BB'] = bfs_df.events == 'walk'
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

    def aggregated_df(self, filtered = True):
        game_days_df = self.at_bats_df \
            .groupby(['game_date', 'game_pk', 'home', 'team', 'opponent', 'batter']) \
                .agg({'opp_sp': ['count', 'first'], 'H': max, 'hp_to_1b': 'first'}) #, 'bats': 'first', 'lineup': 'first'
        game_days_df.columns = ['PA' if col[1] == 'count' else col[0] for col in game_days_df.columns]
        game_days_df.set_index('opp_sp', append = True, inplace = True)

        game_days_df = game_days_df \
            .merge(self.batter_per_game_agg(), left_index = True, right_index = True) \
                .merge(self.batter_per_pa_agg(), left_index = True, right_index = True) \
                    .merge(self.pitcher_per_bf_agg().rename_axis(index = {'pitcher': 'opp_sp'}),
                           left_index = True, right_index = True) \
                        .merge(self.bullpen_per_bf_agg(), left_index = True, right_index = True, suffixes = ('', '_bullpen')) \
                            .reorder_levels(['game_date', 'game_pk', 'home', 'team', 'opponent', 'opp_sp', 'batter']) \
                                .sort_index()
        if filtered:
            game_days_df = game_days_df.loc[
                (game_days_df.PA >= 3) \
                    & (game_days_df[f'G_last_{self.SIGNIFICANT_GAMES}G'] >= self.MINIMUM_GAMES) \
                        & (game_days_df[f'PA_last_{self.SIGNIFICANT_PAS}PA'] >= self.MINIMUM_PAS) \
                            & (game_days_df[f'BF_last_{self.SIGNIFICANT_PAS}BF'] >= self.MINIMUM_PAS) \
                                & (game_days_df[f'BF_last_{self.SIGNIFICANT_PAS}BF_bullpen'] >= self.MINIMUM_PAS)] \
                                    .drop(['PA', f'G_last_{self.SIGNIFICANT_GAMES}G', f'PA_last_{self.SIGNIFICANT_PAS}PA',
                                           f'BF_last_{self.SIGNIFICANT_PAS}BF', f'BF_last_{self.SIGNIFICANT_PAS}BF_bullpen'], axis = 1) \
                                        .dropna()
        return game_days_df

    def build_model(self):
        filtered_game_days_df = self.aggregated_df()
        # Correlation matrix
        # correlation_matrix = filtered_game_days_df.loc[:, filtered_game_days_df.dtypes == float].corr()
        # for row_num in range(len(correlation_matrix.index)):
        #     for col_num in range(len(correlation_matrix.columns)):
        #         if row_num <= col_num:
        #             correlation_matrix.iloc[row_num, col_num] = None
        # correlation_matrix.style.background_gradient(cmap = 'bwr', axis = None, vmin = -1, vmax = 1).highlight_null(color = '#f1f1f1').format(precision = 2) \
        #     .set_table_styles([
        #         {'selector': 'th.col_heading', 'props': [('text-align', 'left'), ('writing-mode', 'vertical-rl'), ('transform', 'rotateZ(192deg)')]},
        #         {'selector': 'td, th', 'props': [('border', '1px solid black !important')]}
        #     ])
        # display.display_html(correlation_matrix)

        # Scale data https://stackoverflow.com/a/59164898
        features_df = filtered_game_days_df.select_dtypes(include = 'float64')
        scaler = StandardScaler().fit(features_df)
        scaled_features_df = pd.DataFrame(scaler.transform(features_df), index = features_df.index, columns = features_df.columns)

        # Train/test split
        train_game_dates, test_game_dates = train_test_split(features_df.index.get_level_values('game_date').unique(),
                                                             test_size = 0.3, random_state = 57)
        X_train = scaled_features_df.loc[scaled_features_df.index.get_level_values('game_date').isin(train_game_dates)]
        X_test = scaled_features_df.loc[scaled_features_df.index.get_level_values('game_date').isin(test_game_dates)]
        y_train = filtered_game_days_df.loc[filtered_game_days_df.index.get_level_values('game_date').isin(train_game_dates), 'H']
        y_test = filtered_game_days_df.loc[filtered_game_days_df.index.get_level_values('game_date').isin(test_game_dates), 'H']

        # PCA https://datascience.stackexchange.com/a/55080
        pca = PCA(n_components = 0.99, svd_solver = 'full', random_state = 57).fit(X_train)
        print('Initial', len(X_train.columns), 'features:', ', '.join(X_train.columns))
        print('PCA: # of features reduced from', len(X_train.columns), 'to', pca.n_components_)
        X_train = pd.DataFrame(pca.transform(X_train), index = X_train.index, columns = [f'PC{f + 1}' for f in range(pca.n_components_)])
        X_test = pd.DataFrame(pca.transform(X_test), index = X_test.index, columns = [f'PC{f + 1}' for f in range(pca.n_components_)])

        # Logistic Regression
        clf = LogisticRegressionCV(cv = 10, random_state = 57).fit(X_train, y_train)
        print('Score on training data:', round(clf.score(X_train, y_train), 3))
        print('Score on testing data:', round(clf.score(X_test, y_test), 3))

        # Predict
        filtered_game_days_df.loc[filtered_game_days_df.index.get_level_values('game_date').isin(test_game_dates), 'H_prob'] = \
            clf.predict_proba(X_test)[:, -1]
        filtered_game_days_df[~filtered_game_days_df.H_prob.isna()]

        # Simulate Results
        threshold = 0.73
        top_two_picks_by_day_df = filtered_game_days_df[filtered_game_days_df.H_prob >= threshold] \
            .sort_values(['game_date', 'H_prob'], ascending = [True, False]) \
                .groupby('game_date') \
                    .head(2)

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

        plt.hist(streaks, bins = list(range(1, 58)))
        plt.title('\n'.join([
            f'Best Streak: {max(streaks) if len(streaks) > 0 else 0}',
            f'Decision Threshold: {100 * threshold}%',
            f'{100 * round(top_two_picks_by_day_df.H.astype(bool).mean(), 2)}% Pick Accuracy'
        ]))
        plt.show()

        # Dump pickle file
        pickle.dump((scaler, pca, clf), open(self.PKL_PATH, 'wb'))
        # Confirm success
        print(pickle.load(open(self.PKL_PATH, 'rb')))

    def todays_predictions(self):
        todays_batters_df = data.get_todays_batters()
        todays_batters_df = todays_batters_df.loc[todays_batters_df.opp_sp != 0]
        todays_batters_df = todays_batters_df.merge(self.at_bats_df.groupby('batter').hp_to_1b.last(), how = 'left', left_index = True,
                                                    right_index = True) # most recent time

        # Add dummy at bats for today's hitters
        if not self.added_todays_batters_to_at_bats:
            for at_bat in range(1, 4):
                todays_batters_at_bats_df = todays_batters_df.drop(['name', 'opp_sp_name'], axis = 1)
                todays_batters_at_bats_df['at_bat'] = at_bat
                todays_batters_at_bats_df['pitcher'] = todays_batters_at_bats_df.opp_sp if at_bat < 3 else 0
                todays_batters_at_bats_df.set_index(['at_bat', 'pitcher'], append = True, inplace = True)
                todays_batters_at_bats_df = todays_batters_at_bats_df.reorder_levels(self.at_bats_df.index.names)
                todays_batters_at_bats_df[['bats', 'xBA', 'events', 'rhb', 'rhp', 'H', 'BIP', 'statcast_tracked']] = \
                    self.at_bats_df.tail(len(todays_batters_at_bats_df.index)).loc[:, ['bats', 'xBA', 'events', 'rhb', 'rhp', 'H', 'BIP', 'statcast_tracked']]
                todays_batters_at_bats_df = todays_batters_at_bats_df.astype(self.at_bats_df.dtypes)
                self.at_bats_df = pd.concat([self.at_bats_df, todays_batters_at_bats_df])
            self.added_todays_batters_to_at_bats = True

        game_days_df = self.aggregated_df()
        todays_options_df = todays_batters_df.loc[:, ['name', 'lineup', 'opp_sp_name',]].merge(game_days_df, left_index = True, right_index = True)

        scaler, pca, clf = pickle.load(open(self.PKL_PATH, 'rb'))
        todays_options_scaled_df = pd.DataFrame(scaler.transform(todays_options_df.loc[:, scaler.feature_names_in_]), index = todays_options_df.index,
                                                columns = scaler.feature_names_in_)
        todays_options_pcs_df = pd.DataFrame(pca.transform(todays_options_scaled_df), index = todays_options_scaled_df.index,
                                             columns = [f'PC{f + 1}' for f in range(pca.n_components_)])
        todays_options_df.loc[:, 'H%'] = clf.predict_proba(todays_options_pcs_df)[:, -1]
        return todays_options_df.sort_values(by = 'H%', ascending = False)