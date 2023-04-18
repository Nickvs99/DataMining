from validators.validator import Validator

class BasicValidator(Validator):

    def __init__(self, df, evaluator, predictor, validate_fraction=0.2):
        super().__init__(df, evaluator, predictor)

        self.validate_fraction = validate_fraction

    def validate(self):
        super().validate()
        
        n_validation_rows = int(len(self.df.index) * self.validate_fraction)

        validation_df = self.df[:n_validation_rows]
        training_df = self.df[n_validation_rows:]

        self.predictor.train(training_df)
        score = self.evaluator.evaluate(validation_df)

        self.actual_values = self.evaluator.actual_values
        self.prediction_values = self.evaluator.prediction_values

        return score, None
