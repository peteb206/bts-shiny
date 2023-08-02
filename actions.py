from data import StatcastData
from datetime import date
import time

today = date.today()

def daily():
    # Refresh this year's data
    StatcastData.update_db(today.year)

    # If Monday, refresh a past year's data, too
    if today.weekday() == 0:
        print()
        StatcastData.update_db(today.year - today.isocalendar().week % 7 - 1)

def full_reset():
    # Refresh all years' data
    for year in range(2015, today.year):
        StatcastData.update_db(year)
        print()
        time.sleep(60)