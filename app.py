from shiny import App, render, ui
from data import get_enhanced_at_bats
from model import BTSBatterClassifier
from datetime import datetime

# Filter data to begin of 2 seasons ago
now = datetime.now()
print('Fetching and processing data from DB...', end = '')
enhanced_at_bats = get_enhanced_at_bats(from_date = datetime(now.year - 2, 1, 1))
print(' complete after', round((datetime.now() - now).seconds, 1), 'seconds')

log_reg = BTSBatterClassifier(None, enhanced_at_bats, 'log_reg')

grey_hex = '#D3D3D3'

app_ui = ui.page_fluid(
    ui.tags.style(
        f'''
        body {{
            background-color: {grey_hex};
        }}
        div.card-body {{
            background-color: {grey_hex};
        }}
        tbody {{
            background-color: white;
        }}
        div.shiny-data-grid-summary {{
            color: green;
        }}
        a.nav-link {{
            color: green !important;
        }}
        '''
    ),
    # Reference this for adding JS (column widths, for example): https://shinylive.io/py/examples/#wordle
    ui.h2('Beat the Streak', {'style': 'color: green;'}),
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
        # ui.nav('Player', '')
    ),
    title = 'Beat the Streak Hub',
    lang = 'en'
)

def server(input, output, session):
    def timestamp_to_str(timestamp):
        hour = str(timestamp.hour - (13 if timestamp.hour > 13 else 1))
        minute = str(timestamp.minute) if timestamp.minute > 9 else f'0{timestamp.minute}'
        am_pm = 'PM' if timestamp.hour > 12 else 'AM'
        return f'{hour}:{minute} {am_pm}'

    @output
    @render.data_frame
    def todays_picks():
        todays_predictions_df = log_reg.todays_predictions()
        df = todays_predictions_df.reset_index()
        df['game'] = df.apply(lambda row: f'{row["team"]} {"vs" if row["home"] else "@"} {row["opponent"]}', axis = 1)
        df['time'] = df.game_date.apply(lambda x: timestamp_to_str(x))
        df.lineup = df.lineup.apply(lambda x: 'OUT' if x == 10 else 'TBD' if x == 0 else x)
        df['H%'] = df['H%'].apply(lambda x: f'{round(x * 100, 1)}%')
        df = df[['name', 'H%', 'game', 'time', 'lineup', 'opp_sp_name']]
        df.columns = ['Batter', 'H%', 'Game', 'Time', 'Lineup', 'Opposing Starter']
        return render.DataGrid(df, summary = 'Viewing Batters {start} to {end} of {total} | All times Central', row_selection_mode = 'none')

app = App(app_ui, server)