

def run_feature_engineering(df):

    df = add_relevance_column(df)

    return df

def add_relevance_column(df):

    relevance_values = []
    for index, row in df.iterrows():

        # Use the normalized values, instead of the absolute values
        if row["booking_bool"]:
            relevance = 1
        elif row["click_bool"]:
            relevance = 0.2
        else:
            relevance = 0
        
        relevance_values.append(relevance)
    
    df["relevance"] = relevance_values

    return df
