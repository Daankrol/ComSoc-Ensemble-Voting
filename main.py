from numpy import testing
from prepare_dataset import Dataset
from neural_net import NeuralNet, reset_weights
from decision_tree import DecisionTree
from svm import SVM
from sklearn.model_selection import KFold
import numpy as np

dataset = Dataset()
# TODO: use MinMaxScaling to normalise al values between 0..1
crossVal = KFold(n_splits=2, shuffle=False)

svm = SVM()
nn = NeuralNet()
dp = DecisionTree()

classifiers = [svm, nn, dp]

fold = 0
for train_index, test_index in crossVal.split(dataset.features):
    fold += 1
    print(f'===== FOLD {fold} =====')
    x_train, x_test = dataset.features[train_index], dataset.features[test_index]
    y_train_labels, y_test_labels = dataset.labels[train_index], dataset.labels[test_index]
    y_train_onehot, y_test_onehot = dataset.one_hot[train_index], dataset.one_hot[test_index]

    nn.set_dataset(x_train, y_train_onehot, x_test, y_test_onehot)
    nn.compile_and_fit()
    print('NN: ', nn.evaluate_and_save())

    svm.set_dataset(x_train, y_train_labels, x_test, y_test_labels)
    svm.fit()
    print('SVM: ', svm.evaluate_and_save())

    dp.set_dataset(x_train, y_train_labels, x_test, y_test_labels)
    dp.fit()
    print('DP: ', dp.evaluate_and_save())

avg = [c.average_f1() for c in classifiers]
print(avg)
print('average over all classifiers:')
avg = np.mean(avg)
print(avg)
