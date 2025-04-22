import pandas as pd
from datetime import datetime, timedelta

def year_week_to_datetime(year, week):
    """Convert ISO year and week to a datetime object (first day of that week)"""
    if pd.isna(year) or pd.isna(week):
        return pd.NaT
    
    # Handle week 53 in non-leap years
    if week > 52 and not (year % 4 == 0 and (year % 100 != 0 or year % 400 == 0)):
        week = 52
    
    # The %G and %V directive in strftime/strptime would be ideal but isn't reliably supported
    # So we calculate manually: Jan 1 of that year + weeks
    jan1 = datetime(int(year), 1, 1)
    
    # Find the first Monday of the year (or Jan 1 if it's a Monday)
    daysToAdd = 0 - jan1.weekday()  # weekday() is 0 for Monday
    if daysToAdd <= 0:
        daysToAdd += 7
    
    # Get first Monday
    first_monday = jan1 + timedelta(days=daysToAdd)
    
    # Add the weeks (minus 1 because week 1 starts at first_monday)
    target_day = first_monday + timedelta(weeks=int(week)-1)
    
    return target_day

def add_year_week(
    df: pd.DataFrame,
    datetime_col: pd.Series
):
    """
    Add a new series 'year_week' to the DataFrame with .isocalendar()
    """    
    df[datetime_col] = pd.to_datetime(df[datetime_col])
    iso_calendar = df[datetime_col].dt.isocalendar()
    # Create the year_week column by combining the year and week number as a string (e.g. "2025-14")
    df['year_week'] = iso_calendar['year'].astype(str) + '_' + iso_calendar['week'].astype(str).str.zfill(2)
    return df