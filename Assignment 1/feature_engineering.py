import numpy as np
import pandas as pd


def run_feature_engineering(df):

    add_experience_column(df)
    categorize_bedtime(df)
    clean_genders(df)


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
            category = "Average"
        
        else:
            hour = int(row["Bedtime"][:2])
            minute = int(row["Bedtime"][3:])
            
            if 0 <= hour < 1:
                category = "Average"
            elif 1 <= hour < 8:
                category = "Nightowl"
            elif 8 <= hour < 19:
                category = "Day dreamers"
            elif 19 <= hour < 24:
                category = "Early bird"
            else:
                raise Exception(f"{row['Bedtime']} does not fall in any sleep category.")

        hours.append(hour)
        categories.append(category)

    df["Bedtime - hour"] = hours
    df["Sleep level"] = categories
    
    df["Bedtime - hour"].apply(pd.to_numeric, errors="coerce")
    df["Sleep level"] = df["Sleep level"].astype("category")

def clean_genders(df):

    genders = []
    for gender in df["Gender"]:

        value = gender
        if value not in ["male", "female"]:
            value = "other"

        genders.append(value)

    df["Gender"] = genders
    df["Gender"] = df["Gender"].astype("category")
