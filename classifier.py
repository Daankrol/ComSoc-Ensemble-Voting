import numpy as np
from sklearn.metrics import f1_score as sk_f1_score


class Classifier():
    def __init__(self) -> None:
        self.f1_scores = []

    def set_dataset(self, x_train, y_train, x_test, y_test):
        self.x_train, self.y_train = x_train, y_train
        self.x_test, self.y_test = x_test, y_test

    def average_f1(self):
        return np.mean(self.f1_scores)

    def evaluate_and_save(self):
        f1 = self.f1_score()
        self.f1_scores.append(f1)
        return f1

    def f1_score(self):
        predictions = self.model.predict(self.x_test)
        f1 = sk_f1_score(self.y_test, predictions, average='weighted')
        return f1

    def fit(self):
        self.model.fit(self.x_train, self.y_train)

    def get_ranking(self):
        prediction = self.predict(self.x_test)    # [.7, .1, .2]
        sorted = np.argsort(prediction)           # [1, 2, 0]
        sorted = np.flip(sorted, axis=-1)         # [0, 2, 1]
        return sorted

    def predict(self, x):
        return self.model.predict_proba(x)
