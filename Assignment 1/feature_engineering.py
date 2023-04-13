import numpy as np
import pandas as pd


def run_feature_engineering(df):

    add_experience_column(df)
    categorize_bedtime(df)


def add_experience_column(df):

    experience_values = []
    for index, row in df.iterrows():

        xp = 0

        xp += row["Machine learning"] == "yes"
        xp += row["Information retrieval"] == "1"
        xp += row["Statistics"] == "mu"
        xp += row["Databases"] == "ja"

        experience_values.append(xp)

    df["Experience"] = experience_values
    df["Experience"].apply(pd.to_numeric, errors="coerce")


def categorize_bedtime(df):

    hours = []
    categories = []
    for index, row in df.iterrows():

        if pd.isna(row["Bedtime"]):
            category = "Invalid"
        
        else:
            hour = int(row["Bedtime"][:2])
            
            
            if 0 <= hour < 2 or hour == 23:
                category = "Average"
            elif 2 <= hour < 8:
                category = "Nightowl"
            elif 8 <= hour < 19:
                category = "Day dreamers"
            elif 19 <= hour < 23:
                category = "Early bird"

        hours.append(hour)
        categories.append(category)

    df["Bedtime - hour"] = hours
    df["Sleep level"] = categories
    
    df["Bedtime - hour"].apply(pd.to_numeric, errors="coerce")
    df["Sleep level"] = df["Sleep level"].astype("category")

