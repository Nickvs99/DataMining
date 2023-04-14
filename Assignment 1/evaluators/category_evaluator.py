from evaluators.evaluator import Evaluator

class CategoryEvaluator(Evaluator):

    def __init__(self, target, predictor, test_df):
        
        super().__init__(target, predictor, test_df)

        if not test_df[target].dtype.name == 'category':
            raise Exception(f"{target} is not a categorical column.")

    def evalutate(self):

        correct_predictions = 0
        for i in range(len(self.test_df.index)):
            entity = self.test_df.iloc[[i]]

            prediction = self.predictor.predict(entity)
            actual = entity[self.target].values[0]

            correct_predictions += prediction == actual

        return correct_predictions / len(self.test_df.index)
