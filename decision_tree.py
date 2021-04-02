import random
import numpy as np
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import classification_report, confusion_matrix
from prepare_dataset import Dataset


class DecisionTree():
    """A Decision tree classifier. Note: When using 'entropy' the output will be a one hot encoding.
    When using 'Gini' the output is a probability vector. 
    """

    def __init__(self) -> None:
        self.criteria = 'entropy'
        # random.choice(['gini', 'entropy'])
        self.max_depth = int(random.uniform(5., 25.))
        self.classifier = DecisionTreeClassifier(
            criterion=self.criteria,
            max_depth=self.max_depth
        )
        self.f1_scores = []

    def set_dataset(self, x_train, y_train, x_test, y_test):
        self.x_train, self.y_train = x_train, y_train
        self.x_test, self.y_test = x_test, y_test

    def fit(self):
        self.classifier.fit(self.x_train, self.y_train)

    def evaluate_and_save(self):
        f1 = self.f1_score()
        self.f1_scores.append(f1)
        return f1

    def f1_score(self):
        predictions = self.classifier.predict(self.x_test)
        report = classification_report(
            self.y_test, predictions, output_dict=True)

        weighted_f1 = report['weighted avg']['f1-score']
        return weighted_f1

    def average_f1(self):
        return np.mean(self.f1_scores)

    def predict(self, x):
        """Returns a probability vector over all classes."""
        return self.classifier.predict_proba(x)

    def plot(self):
        plot_tree(self.classifier)

    def __str__(self) -> str:
        return f'\nDT: \ncriteria: {self.criteria} \nmax depth: {self.max_depth}\nf1: {self.f1_score()}'
