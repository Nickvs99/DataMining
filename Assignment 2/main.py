import numpy as np
import pandas as pd
import warnings

from data_exploration import explore_df

warnings.simplefilter(action='ignore', category=FutureWarning)

def main():

    df = pd.read_csv("data/training_set_100000.csv", sep=",")

    df = set_df_types(df)

    explore_df(df, save_suffix="", show=False)

    print(df)
    print(df.dtypes)

def set_df_types(df):

    # Most types are automatically set by pandas, however it does not
    # detect boolean/categorical columns

    categorical_columns = [
        "srch_id",
        "site_id",
        "visitor_location_country_id",
        "prop_country_id",
        "prop_id",
        "prop_brand_bool",
        "srch_destination_id",
        "srch_saturday_night_bool",
        "random_bool",
        "click_bool",
        "booking_bool",
        "promotion_flag"
    ]   

    for i in range(1, 9):
        categorical_columns.append(f"comp{i}_rate")
        categorical_columns.append(f"comp{i}_inv")

    df[categorical_columns] = df[categorical_columns].astype("category")

    return df

if __name__ == "__main__":
    main()