from evaluators.evaluator import Evaluator
from predictors.numerical_predictor import NumericalPredictor

from recommender_scorer import RecommenderScorer
from logger import logger

class RecommenderEvaluator(Evaluator):

    def __init__(self, target, predictor, groupby_column, score_attribute):
        super().__init__(target, predictor)

        # Recommender evaluators only work with numerical predictors
        if not isinstance(predictor, NumericalPredictor):
            raise TypeError("Predictor is not a numerical predictor")

        self.groupby_column = groupby_column
        self.score_attribute = score_attribute

    def evaluate(self, test_df, output_path="output/output.csv"):
        
        logger.status("Evaluating recommender")
        
        self.write_recommendations(test_df, output_path)

        scorer = RecommenderScorer(output_path, test_df, self.score_attribute)

        return scorer.score()

    def write_recommendations(self, test_df, output_path):

        output = [f"{self.groupby_column},{self.target}"]
        
        groups = test_df.groupby(self.groupby_column)
        log_per_n_messages = min(groups.ngroups // 10, 1000)

        for i, (group_id, group) in enumerate(groups):

            if i % log_per_n_messages == 0:
                logger.progress(f"Creating recommendations {i / groups.ngroups * 100:.2f}%")
            
            if len(group.index) == 0:
                continue
    
            target_values = self.get_sorted_attribute_values(group, self.target)

            for target_value in target_values:
                output.append(f"{group_id},{target_value}")

        
        with open(output_path, 'w') as f:
            for line in output:
                f.write(f"{line}\n")


    def get_sorted_attribute_values(self, group, attribute):
        """
        Returns the attribute values after sorting the instances by the target attribute
        """
        
        indices, predictions = self.get_predictions(group)

        # Sort indices based on predictions
        sorted_ = sorted(list(zip(indices, predictions)), key=lambda x: x[1], reverse=True)
        indices_sorted = [index for index, prediction in sorted_]

        # Get the actual values
        attribute_values = [group.loc[index, attribute] for index in indices_sorted]
        
        return attribute_values

    def get_predictions(self, group):
        
        predictions = group.apply(lambda row:
            self.predictor.predict(row),
            axis=1
        )

        return group.index.values, predictions.values
    