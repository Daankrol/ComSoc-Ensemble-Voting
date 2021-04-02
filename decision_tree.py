from classifier import Classifier
import random
import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report, f1_score as sk_f1_score
from prepare_dataset import Dataset


class DecisionTree(Classifier):
    """A Decision tree classifier. Note: When using 'entropy' the output will be a one hot encoding.
    When using 'Gini' the output is a probability vector.
    """

    def __init__(self) -> None:
        super().__init__()
        self.criteria = random.choice(['gini', 'entropy'])
        self.max_depth = int(random.uniform(5., 25.))
        self.model = DecisionTreeClassifier(
            criterion=self.criteria,
            max_depth=self.max_depth
        )

    def __str__(self) -> str:
        return f'\nDT: \ncriteria: {self.criteria} \nmax depth: {self.max_depth}\nf1: {self.f1_score()}'
