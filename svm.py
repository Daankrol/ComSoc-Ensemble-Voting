from prepare_dataset import Dataset
from sklearn import svm
from sklearn.metrics import f1_score
import random


class SVM():
    def __init__(self, dataset: Dataset) -> None:
        self.ds = dataset
        self.kernel = random.choice(['poly', 'rbf'])
        self.C = random.uniform(2**-5, 2**5)
        self.coef = 0.0  # default
        self.gamma = 'scale'  # default
        if self.kernel == 'poly':
            self.coef = random.uniform(3, 5)
        else:
            self.gamma = 'auto'

        # train SVM
        self.model = svm.SVC(
            C=self.C,
            kernel=self.kernel,
            coef0=self.coef,
            gamma=self.gamma
        )
        self.model.fit(self.ds.train_x, self.ds.train_labels)

    def evaluate(self):
        predictions = self.model.predict(self.ds.test_x)
        f1 = f1_score(self.ds.test_labels, predictions, average='weighted')
        return f1

    def f1_score(self):
        return self.evaluate()

    def predict(self, x):
        # Returns class labels
        return self.model.predict(x)

    def __str__(self) -> str:

        return f'\nSVM: \nkernel: {self.kernel} \nC: {self.C} \ngamma: {self.gamma} \
            \ncoef: {self.coef} \nf1: {self.f1_score()}'
