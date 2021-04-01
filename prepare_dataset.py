import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.python.ops.gen_array_ops import one_hot


class Dataset():
    def __init__(self, normalise=True) -> None:
        print('Constructing dataset')
        data = pd.read_csv("dataset/letter-recognition.data", header=None, names=[
            'label', 'x-box', 'y-box', 'width', 'height', 'onpix', 'x-bar', 'y-bar', 'x2bar', 'y2bar', 'xybar', 'x2ybr', 'xy2br', 'x-ege', 'xegvy', 'y-ege', 'yegvx'])

        letters_features = data.copy()

        letters_labels = letters_features.pop('label')
        if normalise:
            letters_features = (
                letters_features-letters_features.mean()) / letters_features.std()
        letters_features = np.array(letters_features)
        self.features = letters_features
        self.size = len(letters_labels)
        self.labels = np.array(letters_labels)

        # turn label chars into one hot encoding
        self.one_hot = np.array(pd.get_dummies(self.labels))
        self.n_classes = len(self.one_hot[0])

        # Split the dataset
        self.train_x = self.features[0:int(self.size * 0.7)]
        self.train_labels = self.labels[0:int(self.size * 0.7)]
        self.val_x = self.features[int(self.size * 0.7):int(self.size * 0.85)]
        self.val_labels = self.labels[int(
            self.size * 0.7):int(self.size * 0.85)]
        self.test_x = self.features[int(self.size * 0.85):]
        self.test_labels = self.labels[int(self.size * 0.85):]

        self.train_onehot = self.one_hot[0:int(self.size * 0.7)]
        self.val_onehot = self.one_hot[int(
            self.size * 0.7):int(self.size * 0.85)]
        self.test_onehot = self.one_hot[int(self.size * 0.85):]

    def __str__(self) -> str:
        return f'DS: \n{self.train_x[:3]} \n{self.train_labels[:3]}\nn_classes: {self.n_classes}'


if __name__ == '__main__':
    ds = Dataset()
    print(ds)
