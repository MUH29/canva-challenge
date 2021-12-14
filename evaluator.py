from dataclasses import asdict
from pprint import pprint
from typing import List

from evaluator_utils import (
    ConfusionMatrix,
    render_confusion_matrix,
    generate_data,
    Label,
    LabelMetrics,
)


class Evaluator:
    def __init__(self, actuals: List[Label], predictions: List[Label]):
        """ Initialize an Evaluator using two lists representing actual labels and predictions for some samples. """
        self.actuals = actuals
        self.predictions = predictions
        self.matrix = self.build_matrix(actuals, predictions)

    @staticmethod
    def build_matrix(actuals: List[Label], predictions: List[Label]) -> ConfusionMatrix:
        """ Builds and return the ConfusionMatrix. """
        # fmt: off
        assert len(actuals) == len(predictions), "Require same number of actuals and predictions."
        confusion_matrix = ConfusionMatrix()
        try:

            for la, lp in zip(actuals, predictions):

                confusion_matrix.add(la, lp)

        # fmt: on
        except:
            raise NotImplementedError("Your solution here!")

        return confusion_matrix

    def calculate_metrics(self) -> List[LabelMetrics]:
        """ Calculates and returns metrics. """

        """
        Note:
        
            loop through labels
            for each label calculate the precision, recall, and f1 score
            precision = tp/(tp + fp)
            recall = tp /(tp+fn)
            f1 = 2*precision / (precision + recall)
            
            tp = count of observations classified correctly for the given label
            fp =  count of observations predicted as the given label when their actual label is different 
            fn =  count of observation predicted as not the given label when they should be
        """
        # Initialize an empty list to hold LabelMetrics
        label_metrics = []
        # Loop through labels and get the corresponding counts
        for label in self.matrix.labels:
            tp = self.matrix.get(label, label)
            fp = sum([self.matrix.get(l, label) for l in self.matrix.labels.difference(set(label))])
            fn = sum([self.matrix.get(label, l) for l in self.matrix.labels.difference(set(label))])
            precision = tp / (tp+fp)
            recall = tp / (tp+fn)
            f1 = (2 * (precision*recall)) / (precision + recall)
            actual_samples = tp + fn

            label_metrics.append(LabelMetrics(label, actual_samples, precision, recall, f1))

        return label_metrics


def main():
    # Generate some fake data.
    actuals, predictions = generate_data()

    # Test Evaluator.
    evaluator = Evaluator(actuals, predictions)
    print(render_confusion_matrix(evaluator.matrix))
    for metric in evaluator.calculate_metrics():
        pprint(asdict(metric))


if __name__ == "__main__":

    main()
