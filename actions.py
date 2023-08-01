from data import StatcastData
from datetime import date

if __name__ == '__main__':
    # Refresh this year's data
    today = date.today()
    StatcastData.update_db(today.year)

    # If Monday, refresh a past year's data, too
    if today.weekday() == 0:
        StatcastData.update_db(today.year - today.isocalendar().week % 7 - 1)