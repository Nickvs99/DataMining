from logger import logger

def run_feature_engineering(df):

    logger.status("Feature engineering")

    df = add_relevance_column(df)

    # Drop attributes which have been used for feature engineering
    df.drop(columns=["click_bool", "booking_bool"], inplace=True)

    return df

def add_relevance_column(df):
    
    df["relevance"] = df.apply(lambda row: 
        calc_relevance(row),
        axis=1
    )

    return df

def calc_relevance(row):
    
    # Use the normalized values, instead of the absolute values
    if row["booking_bool"]:
        relevance = 1
    elif row["click_bool"]:
        relevance = 0.2
    else:
        relevance = 0

    return relevance
