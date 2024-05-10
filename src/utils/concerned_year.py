import numpy as np
import pandas as pd


def _calculate_concerned_year(row: pd.Series) -> int:
    """
    Calculate concerned year based on the year in the title of the row
    """
    # if row['concerned_year'] is not empty
    if not np.isnan(row["concerned_year"]):
        return row["concerned_year"]

    # the title is a string, check if the last word is a year over 2018
    if row["title"].split()[-1].isdigit():
        year = int(row["title"].split()[-1])
        if year >= 2018:
            return year

    if row["type"] == "Årsrapport":
        if pd.to_datetime(row["published_at"]).month < 6:
            return pd.to_datetime(row["published_at"]).year - 1
        return pd.to_datetime(row["published_at"]).year

    if row["type"] == "Tildelingsbrev":
        if pd.to_datetime(row["published_at"]).month > 6:
            return pd.to_datetime(row["published_at"]).year + 1
        return pd.to_datetime(row["published_at"]).year

    return np.nan


def populate_concerned_year_(df: pd.DataFrame) -> None:
    df["concerned_year"] = df.apply(_calculate_concerned_year, axis=1)
