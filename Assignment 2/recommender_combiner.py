import math
import pandas as pd

from feature_engineering import add_relevance_column
from logger import logger
from recommender_scorer import RecommenderScorer

class RecommenderCombiner:

    def __init__(self, output_paths, weights=[]):

        self.output_paths = output_paths

        if len(weights) == 0:
            N = len(self.output_paths)
            weights = [1/N for i in range(N)]
        
        self.output_df = self.get_output_df(output_paths)
        self.weights = weights
    
    def get_output_df(self, output_paths):
        """
        Creates a df with all the output data from the output paths stored in a single df.
        """

        output_dfs = [pd.read_csv(output_path, sep=",") for output_path in output_paths]
        output_df = output_dfs[0]
        
        self.columns = output_df.columns.tolist()

        output_df.rename(columns={ output_df.columns[1]: "output0" }, inplace = True)

        for i in range(1, len(output_dfs)):
            df = output_dfs[i]
            df.rename(columns={ df.columns[1]: f"output{i}" }, inplace = True)

            output_df[f"output{i}"] = df[f"output{i}"]

        return output_df

    def combine(self, output_path="output/output.csv"):
        
        output = [f"{self.columns[0]},{self.columns[1]}"]

        for group_id, group in self.output_df.groupby(self.columns[0]):
            sorted_values = self.get_sorted_values(group)

            for value in sorted_values:
                output.append(f"{group_id},{value}")

        with open(output_path, "w") as f:
            for line in output:
                f.write(f"{line}\n")

    def get_sorted_values(self, group):

        score_map = {}
        for i in range(len(group.index)):
            for j in range(len(group.columns[1:])):

                # Skip the groupby column
                column_id = j + 1
                cell_value = group.iloc[i, column_id]

                if cell_value not in score_map:
                    score_map[cell_value] = 0

                score_map[cell_value] += self.ranking_value(i + 1, column_id - 1)

        # Sort the values based on their ranking score
        sorted_values = [i[0] for i in sorted(score_map.items(), key=lambda kvp: kvp[1], reverse=True)]
        
        return sorted_values

    def ranking_value(self, ranking, weight_index):
        return 1/math.log2(ranking + 1) * self.weights[weight_index]

def main():

    usecols = ["srch_id", "prop_id", "click_bool", "booking_bool"]
    df = pd.read_csv("data/training_set_100000.csv", sep=",", usecols=usecols)
    df = add_relevance_column(df)
    df.drop(columns=["click_bool", "booking_bool"], inplace=True)

    output_path = "output/test.csv"
    input_paths = ["output/svd.csv", "output/prop_review_score.csv"]
    combiner = RecommenderCombiner(input_paths)
    combiner.combine(output_path="output/test.csv")

    for path in input_paths + [output_path]:
        scorer = RecommenderScorer(path, df, "relevance")

        logger.info(f"Score {path}: {scorer.score()}")


def set_df_types(df):

    # Most types are automatically set by pandas, however it does not
    # detect boolean/categorical columns

    categorical_columns = [
        "srch_id",
        "prop_id",
        "click_bool",
        "booking_bool",
    ]   

    df[categorical_columns] = df[categorical_columns].astype("category")

    return df


if __name__ == "__main__":
    main()