import random
import numpy as np
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import classification_report, confusion_matrix
from prepare_dataset import Dataset


class DecisionTree():
    def __init__(self, dataset: Dataset) -> None:
        self.ds = dataset
        self.criteria = random.choice(['gini', 'entropy'])
        self.max_depth = int(random.uniform(5., 25.))
        self.classifier = DecisionTreeClassifier(
            criterion=self.criteria,
            max_depth=self.max_depth
        )

        self.classifier.fit(self.ds.train_x, self.ds.train_labels)
        self.evaluation = self.evaluate()

    def evaluate(self):
        predictions = self.classifier.predict(self.ds.test_x)
        report = classification_report(
            self.ds.test_labels, predictions, output_dict=True)
        weighted_f1 = report['weighted avg']['f1-score']
        return weighted_f1

    def f1_score(self):
        return self.evaluate()

    def predict(self, x):
        return self.classifier.predict(x)

    def plot(self):
        plot_tree(self.classifier)

    def __str__(self) -> str:
        return f'\nDT: \ncriteria: {self.criteria} \nmax depth: {self.max_depth}\nf1: {self.f1_score()}'
