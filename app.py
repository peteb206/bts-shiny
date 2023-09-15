from shiny import App, Inputs, Outputs, Session, ui, render, reactive
from shinyswatch.theme import spacelab as theme # https://bootswatch.com/
from data import get_enhanced_at_bats
from model import BTSBatterClassifier
from datetime import datetime, date

# Filter data to begin of 2 seasons ago
now = datetime.now()
print('Fetching and processing data from DB...', end = '')
enhanced_at_bats = get_enhanced_at_bats(from_date = datetime(now.year - 1, 1, 1))
print(' complete after', round((datetime.now() - now).seconds, 1), 'seconds')

classifier = BTSBatterClassifier(None, enhanced_at_bats, 'log_reg')

game_date = now.date()
todays_predictions_df = classifier.todays_predictions(game_date)#.query('`H%` >= 0.7')

def timestamp_to_str(timestamp):
    hour = str(timestamp.hour - (13 if timestamp.hour > 13 else 1))
    minute = str(timestamp.minute) if timestamp.minute > 9 else f'0{timestamp.minute}'
    am_pm = 'PM' if timestamp.hour > 12 else 'AM'
    return f'{hour}:{minute} {am_pm}'

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
            ui.input_date('date', 'Date:', value = now.date(), min = date(now.year, 3, 1), max = now.date(), format = 'M d, yyyy'),
            # ui.h4(f'{now.strftime("%B")} {now.day}, {now.year}'),
            ui.output_ui('recommendations'),
            ui.output_data_frame('picks_dataframe')
        ),
        ui.nav_control(ui.a('MLB Play Link', href = 'https://www.mlb.com/apps/beat-the-streak/game', target = '_blank')),
        # ui.nav('Player', ''),
        title = 'Beat the Streak Shiny App',
        inverse = True
    ),
    title = 'Beat the Streak Shiny App',
    lang = 'en'
)

def server(input: Inputs, output: Outputs, session: Session):
    @reactive.Calc
    def updated_predictions():
        global game_date, todays_predictions_df
        if game_date != input.date():
            game_date = input.date()
            todays_predictions_df = classifier.todays_predictions(game_date)
        return todays_predictions_df

    @output
    @render.ui
    def recommendations():
        todays_recommendations_df = updated_predictions().head(2).query('`H%` >= 0.75').reset_index()
        cols = [
            ui.div(
                ui.row(
                    ui.column(
                        3,
                        ui.img(src = f'https://img.mlbstatic.com/mlb/images/players/head_shot/{row["batter"]}.jpg', width = '80px')
                    ),
                    ui.column(
                        9,
                        ui.row(ui.column(12, row['name'])),
                        ui.row(ui.column(12, f'{round(row["H%"] * 100, 1)}%')),
                        ui.row(ui.column(12, f'Lineup Slot: {"OUT" if row["lineup"] == 10 else "TBD" if row["lineup"] == 0 else str(int(row["lineup"]))}')),
                        ui.row(ui.column(12, f'{row["team"]} {"vs" if row["home"] else "@"} {row["opponent"]}')),
                        ui.row(ui.column(
                            12,
                            f'{timestamp_to_str(row["game_time"])} CDT' if 'game_time' in row.keys() else ui.p('✅' if row['H'] > 0 else '❌')
                        ))
                    )
                )
            ).add_class('col-6') for _, row in todays_recommendations_df.iterrows()
        ]
        recommendation = ''
        if len(cols) == 2:
            recommendation = f'Double down with {" and ".join(todays_recommendations_df.name.to_list())}.'
        elif len(cols) == 1:
            recommendation = f'Pick only {todays_recommendations_df.name.to_list()[0]}. No one else has a chance of 75% or higher to get a hit.'
        elif len(todays_predictions_df.index) > 0:
            recommendation = 'Skip today. No players have a chance of 75% or higher to get a hit.'
        else:
            recommendation = 'No games today.'
        return ui.div(
            ui.row(ui.column(12, ui.strong(f'Recommendation for {game_date.strftime("%B")} {game_date.day}, {game_date.year}'), ui.br(), recommendation)).add_style('padding-bottom: 10px;'),
            ui.row(*cols).add_style('padding-bottom: 10px; max-width: 750px;')
        )

    @output
    @render.data_frame
    def picks_dataframe():
        df = updated_predictions().reset_index()
        df['game'] = df.apply(lambda row: f'{row["team"]} {"vs" if row["home"] else "@"} {row["opponent"]}', axis = 1)
        df.lineup = df.lineup.apply(lambda x: 'OUT' if x == 10 else 'TBD' if x == 0 else str(int(x)))
        df['H%'] = df['H%'].apply(lambda x: f'{round(x * 100, 1)}%')
        if 'H' in df.columns: # Past day
            df = df[['name', 'H%', 'H', 'game', 'lineup', 'opp_sp_name']]
            df.columns = ['Batter', 'H%', 'H', 'Game', 'Lineup', 'Opposing Starter']
        elif 'game_time' in df.columns: # Today
            df['time'] = df.game_time.apply(lambda x: timestamp_to_str(x))
            df = df[['name', 'H%', 'game', 'time', 'lineup', 'opp_sp_name']]
            df.columns = ['Batter', 'H%', 'Game', 'Time (CDT)', 'Lineup', 'Opposing Starter']
        return render.DataGrid(df, summary = 'Viewing Batters {start} to {end} of {total}', row_selection_mode = 'none')

app = App(app_ui, server)