from shiny import App, render, ui
from shinyswatch.theme import spacelab as theme # https://bootswatch.com/
from data import get_enhanced_at_bats
from model import BTSBatterClassifier
from datetime import datetime

# Filter data to begin of 2 seasons ago
now = datetime.now()
print('Fetching and processing data from DB...', end = '')
enhanced_at_bats = get_enhanced_at_bats(from_date = datetime(now.year - 2, 1, 1))
print(' complete after', round((datetime.now() - now).seconds, 1), 'seconds')

log_reg = BTSBatterClassifier(None, enhanced_at_bats, 'log_reg')
todays_predictions_df = log_reg.todays_predictions().query('`H%` >= 0.7')

app_ui = ui.page_fluid(
    theme(),
    ui.tags.script('''
        $(document).ready(function() {
            $('nav').toggleClass('bg-dark bg-primary');
        });
    '''),
    ui.page_navbar(
        ui.nav(
            '', # 'Home',
            # ui.input_date('picksDate', 'Date:', value = now.date(), max = now.date(), format = 'M d, yyyy'),
            ui.h4(f'Picks for {now.strftime("%B")} {now.day}, {now.year}'),
            ui.output_data_frame('todays_picks')
        ),
        # ui.nav('Player', ''),
        title = 'Beat the Streak Shiny App',
        inverse = True
    ),
    title = 'Beat the Streak Shiny App',
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
        df = todays_predictions_df.reset_index()
        df['game'] = df.apply(lambda row: f'{row["team"]} {"vs" if row["home"] else "@"} {row["opponent"]}', axis = 1)
        df['time'] = df.game_date.apply(lambda x: timestamp_to_str(x))
        df.lineup = df.lineup.apply(lambda x: 'OUT' if x == 10 else 'TBD' if x == 0 else x)
        df['H%'] = df['H%'].apply(lambda x: f'{round(x * 100, 1)}%')
        df = df[['name', 'H%', 'game', 'time', 'lineup', 'opp_sp_name']]
        df.columns = ['Batter', 'H%', 'Game', 'Time', 'Lineup', 'Opposing Starter']
        return render.DataGrid(df, summary = 'Viewing Batters {start} to {end} of {total} (>= 70%) | All times Central', row_selection_mode = 'none')

app = App(app_ui, server)