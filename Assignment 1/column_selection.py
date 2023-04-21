import pandas as pd

from df_util import save_df

def select_columns(df, validator, mandatory=[], prefered=[]):

    columns = mandatory

    # Compute initial score given the mandatory columns
    max_score, max_score_error = validate(validator, df, columns)
    
    # Store progress of the selection procedure
    initial_row = {
        "variables": str(columns),
        "score": max_score,
        "error": max_score_error,
        "message": "Initial variables"
    }
    column_selection_df = pd.DataFrame(initial_row, index=[0])

    # Test if the addition of a column improves the performance of the validator
    # Start with the prefered columns and then go through the others
    for column in prefered + df.columns.tolist():

        if column in columns:
            continue

        validate_columns = columns + [column]

        score, std_error = validate(validator, df, validate_columns)

        # Update best score
        if score - std_error > max_score + max_score_error:
            max_score = score
            max_score_error = std_error
            
            message = "Accepted"
            columns.append(column)
        else:

            if score > max_score:
                message = "Rejected, due to insignificance"
            else:
                message = "Rejected"
        
        # Add data to column_selection_df
        new_row = {
            "variables": str(validate_columns),
            "score": score,
            "error": std_error,
            "message": message
        }

        temp_df = pd.DataFrame(new_row, index=[0])

        column_selection_df = pd.concat([column_selection_df, temp_df], ignore_index=True)

    save_df(column_selection_df, "tables/column_selection.tex")
    
    return columns

def validate(validator, df, columns):

    validate_df = df[columns]
    validator.df = validate_df

    return validator.validate()
