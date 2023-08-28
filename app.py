from shiny import App, render, ui
from data import get_enhanced_at_bats
from model import BTSBatterClassifier
from datetime import datetime

# Filter data to begin of 2 seasons ago
now = datetime.now()
print('Fetching data from DB...', end = '')
enhanced_at_bats = get_enhanced_at_bats(from_date = datetime(now.year - 2, 1, 1))
print(' complete after', round((datetime.now() - now).seconds, 1), 'seconds')

log_reg = BTSBatterClassifier(None, enhanced_at_bats, 'log_reg')

app_ui = ui.page_fluid(
    ui.tags.style(
        '''
        body {
            background-color: #53565B;
        }
        '''
        # div.card {
        #     background-color: green;
        # }
    ),
    # Reference this for adding JS (column widths, for example): https://shinylive.io/py/examples/#wordle
    ui.h2('Beat the Streak', {'style': 'color: white;'}),
    ui.navset_tab_card(
        ui.nav(
            'Home',
            # ui.input_date(
            #     'picksDate',
            #     'Date:',
            #     value = date.today(),
            #     format='M d, yyyy'
            # ),
            ui.h3('Today\'s Picks', {'style': 'color: green;'}),
            ui.output_data_frame('todays_picks')
        ),
        ui.nav('Player', '')
    ),
    title = 'Beat the Streak Hub',
    lang = 'en'
)

def server(input, output, session):
    # db_summary = list(__DB__.games.aggregate([
    #     {
    #         '$group': {
    #             '_id': None,
    #             'min': {'$min': '$game_date'},
    #             'max': {'$max': '$game_date'},
    #             'count': {'$sum': 1}
    #         }
    #     }, {
    #         '$unset': '_id'
    #     }
    # ]))[0]

    @output
    @render.data_frame
    def todays_picks():
        todays_predictions_df = log_reg.todays_predictions()
        df = todays_predictions_df.reset_index()
        df['game'] = df.apply(lambda row: f'{row["team"]} {"vs" if row["home"] else "@"} {row["opponent"]}', axis = 1)
        df['time'] = df.apply(lambda row: f'{row["game_date"].hour - 13}:' + (str(row["game_date"].minute) if row["game_date"].minute > 9 else f'0{row["game_date"].minute}') + f' {"P" if row["game_date"].hour > 12 else "A"}M', axis = 1)
        df.lineup = df.lineup.apply(lambda x: 'OUT' if x == 10 else 'TBD' if x == 0 else x)
        df['H%'] = df['H%'].apply(lambda x: f'{round(x * 100, 1)}%')
        df = df[['name', 'H%', 'game', 'time', 'opp_sp_name', 'lineup']]
        df.columns = ['Batter', 'H%', 'Game', 'Time', 'Starting Pitcher', 'Lineup']
        return render.DataGrid(df, summary = 'Viewing Batters {start} to {end} of {total} | All times Central', row_selection_mode = 'none')

app = App(app_ui, server)