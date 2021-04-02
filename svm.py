from prepare_dataset import Dataset
from sklearn import svm
from sklearn.metrics import f1_score
import random
import numpy as np


class SVM():
    """
    """

    def __init__(self) -> None:

        self.kernel = random.choice(['poly', 'rbf'])
        # Decreasing C corresponds to more regularization.
        self.C = random.uniform(2**-5, 2**5)
        self.coef = 0.0  # default
        self.gamma = 'scale'  # default
        if self.kernel == 'poly':
            self.coef = random.uniform(3, 5)
        else:
            self.gamma = 'auto'
        self.model = svm.SVC(
            C=self.C,
            kernel=self.kernel,
            coef0=self.coef,
            gamma=self.gamma,
            probability=True
        )
        self.f1_scores = []

    def set_dataset(self, x_train, y_train, x_test, y_test):
        self.x_train, self.y_train = x_train, y_train
        self.x_test, self.y_test = x_test, y_test

    def fit(self):
        self.model.fit(self.x_train, self.y_train)

    def f1_score(self):
        predictions = self.model.predict(self.x_test)
        f1 = f1_score(self.y_test, predictions, average='weighted')
        return f1

    def evaluate_and_save(self):
        f1 = self.f1_score()
        self.f1_scores.append(f1)
        return f1

    def average_f1(self):
        return np.mean(self.f1_scores)

    def predict(self, x):
        """Returns a probability vector over the classes. 
        Note: The probability model is created using cross validation, 
        so the results can be slightly different than those obtained by predict.
        """
        return self.model.predict_proba(x)

    def __str__(self) -> str:

        return f'\nSVM: \nkernel: {self.kernel} \nC: {self.C} \ngamma: {self.gamma} \
            \ncoef: {self.coef} \nf1: {self.evaluate()()}'
