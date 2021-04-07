import random
import sys
import getopt
from prepare_dataset import Dataset
from neural_net import NeuralNet
from decision_tree import DecisionTree
from svm import SVM
from sklearn.model_selection import KFold
from sklearn.metrics import f1_score
import numpy as np
import math
from voting_rules import *

n_kfold_splits = 10
dataset = Dataset()
crossVal = KFold(n_splits=n_kfold_splits, shuffle=False)
for train_index, test_index in crossVal.split(dataset.features):
    x_train, x_test = dataset.features[train_index], dataset.features[test_index]
    y_train_labels, y_test_labels = dataset.labels[train_index], dataset.labels[test_index]
    y_train_onehot, y_test_onehot = dataset.one_hot[train_index], dataset.one_hot[test_index]

    print(f'num training: {len(y_train_labels)}')
    print(f'num test: {len(y_test_labels)}')

    B = len(y_train_labels) / 100
    b_size = 2 ** (math.ceil(math.log2(B)))
    print(b_size)
    exit()


c1 = [[x for x in range(26)]]
c2 = [[x for x in range(26)]]
c3 = [[x for x in range(26)]]
c4 = [[x for x in range(26)]]
r = [c1, c2, c3, c4]

c2[0][0] = 1
c2[0][1] = 2
c2[0][2] = 0

c3[0][0] = 2
c3[0][1] = 1
c3[0][2] = 0

c4[0][0] = 3
c4[0][1] = 1
c4[0][2] = 2
c4[0][3] = 0

print(c1, '\n', c2, '\n')

print(r)
print(plurality(r))
print(STV(r))

c1 = [[x for x in range(26)]]
c2 = [[x for x in range(26)]]
c3 = [[x for x in range(26)]]

c1[0][0] = 2
c1[0][1] = 1
c1[0][2] = 0

c2[0][0] = 1
c2[0][1] = 0
c2[0][2] = 2

r = [c1, c2, c3]
print(plurality(r))
print(STV(r))
