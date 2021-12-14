import random
from collections import Counter
from dataclasses import dataclass
from typing import List, Tuple, Counter as CounterT

# Helper types and classes for the evaluator task.
Label = str


@dataclass(frozen=True)
class LabelMetrics:
    label: Label
    num_actual_samples: int
    precision: float
    recall: float
    f1: float


class ConfusionMatrix:
    """ Stores counts of true label and predicted label pairs. """

    def __init__(self):
        self.labels = set()
        self.data: CounterT[Tuple[Label, Label]] = Counter()

    def add(self, actual: Label, prediction: Label):
        """ Increments the count of label instances predicted as prediction. """
        key = (actual, prediction)
        self.data[key] += 1
        self.labels.update(key)

    def get(self, actual: Label, prediction: Label) -> int:
        """ Gets the count of label instances predicted as prediction. """
        return self.data.get((actual, prediction), 0)

    @property
    def all_labels(self) -> List[Label]:
        """ Returns a sorted list of the unique labels. """
        return sorted(self.labels)


"""
         888                     
         888                     
         888                     
.d8888b  888888 .d88b.  88888b.  
88K      888   d88""88b 888 "88b 
"Y8888b. 888   888  888 888  888 
     X88 Y88b. Y88..88P 888 d88P 
 88888P'  "Y888 "Y88P"  88888P"  
                        888      
                        888      
                        888      

Below here are functions and constants used for the evaluator task. 
No need to read or understand them!
"""


def generate_data(
    n_samples: int = 100, p_noise: float = 0.9
) -> Tuple[List[Label], List[Label]]:
    random.seed(1337)
    all_labels = ["🍎", "🍌", "🥑"]
    actuals = [random.choice(all_labels) for _ in range(n_samples)]
    predictions = []
    for actual in actuals:
        prediction = random.choice(all_labels) if random.random() > p_noise else actual
        predictions.append(prediction)
    return actuals, predictions


def render_confusion_matrix(matrix: ConfusionMatrix) -> str:
    lines: List[str] = []
    labels = matrix.all_labels
    lines.append("\t".join(labels + ["⬅️  predictions / ⬇️  actuals"]))
    for actual in labels:
        line = []
        for predicted in labels:
            count = matrix.get(actual, predicted)
            line.append(str(count))
        line.append(actual)
        lines.append("\t".join(line))
    return "\n".join(lines)
