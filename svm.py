from classifier import Classifier
from prepare_dataset import Dataset
from sklearn import svm
from sklearn.metrics import f1_score as sk_f1_score
import random
import numpy as np


class SVM(Classifier):
    def __init__(self) -> None:
        super().__init__()
        self.kernel = random.choice(['poly', 'rbf'])

        # Decreasing C corresponds to more regularization.
        c = random.uniform(2**-5, 2**5)
        self.C = int(np.exp(c))
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

    def __str__(self) -> str:

        return f'\nSVM: \nkernel: {self.kernel} \nC: {self.C} \ngamma: {self.gamma} \
            \ncoef: {self.coef} \nf1: {self.evaluate()()}'
