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
from voting_rules import *

if len(sys.argv) != 5:
    print('main.py -n <num classifiers> -e <num experiments>')
    sys.exit(2)

try:
    opts, args = getopt.getopt(sys.argv[1:], "hn:e:")
except getopt.GetoptError:
    print('main.py -n <num classifiers> -e <num experiments>')
    sys.exit(2)
for opt, arg in opts:
    if opt == '-h':
        print('main.py -n <num classifiers> -e <num experiments>')
        sys.exit()
    elif opt == '-n':
        num_classifiers = int(arg)
    elif opt == '-e':
        n_experiments = int(arg)


n_kfold_splits = 10
exp_evaluation_plurality = []
exp_evaluation_svt = []
exp_evaluation_pos1 = []
exp_evaluation_pos2 = []
exp_evaluation_pos3 = []


w1 = [30.4, 5.51, 3.5, 2.67, 2.22, 1.92, 1.7, 1.53, 1.39, 1.28, 1.17, 1.08,
      0.99, 0.92, 0.85, 0.8, 0.75, 0.7, 0.65, 0.6, 0.56, 0.51, 0.48, 0.4, 0.43, 0.36]
w2 = [9.61, 1.74, 1.11, 0.85, 0.7, 0.61, 0.54, 0.48, 0.44, 0.4, 0.37, 0.34, 0.31,
      0.29, 0.27, 0.25, 0.24, 0.22, 0.2, 0.19, 0.18, 0.16, 0.15, 0.13, 0.14, 0.11]
w3 = [[6.83, 3.41, 2.51, 1.97, 1.6, 1.31, 1.06, 0.85, 0.66, 0.49, 0.31, 0.15, -0.01, -
       0.18, -0.32, -0.45, -0.59, -0.73, -0.87, -1.01, -1.18, -1.33, -1.48, -1.85, -1.69, -2.06]]

for exp in range(n_experiments):
    print(f'=== EXPERIMENT {exp} ===')
    dataset = Dataset()
    crossVal = KFold(n_splits=n_kfold_splits, shuffle=False)

    classifiers = []
    for i in range(num_classifiers):
        random_c = random.choice(['svm', 'nn', 'dp'])
        if random_c == 'svm':
            classifiers.append(SVM())
        elif random_c == 'nn':
            classifiers.append(NeuralNet())
        elif random_c == 'dp':
            classifiers.append(DecisionTree())

    fold = 0
    for train_index, test_index in crossVal.split(dataset.features):
        fold += 1
        print(f'= FOLD {fold} =')
        x_train, x_test = dataset.features[train_index], dataset.features[test_index]
        y_train_labels, y_test_labels = dataset.labels[train_index], dataset.labels[test_index]
        y_train_onehot, y_test_onehot = dataset.one_hot[train_index], dataset.one_hot[test_index]

        for classifier in classifiers:
            if isinstance(classifier, NeuralNet):
                classifier.set_dataset(
                    x_train, y_train_onehot, x_test, y_test_onehot)
            else:
                classifier.set_dataset(
                    x_train, y_train_labels, x_test, y_test_labels)
            classifier.fit()

        # compute f1 score based on voting rule
        rankings = np.array([c.get_ranking() for c in classifiers])
        outcome_plurality = plurality(rankings)
        outcome_SVT = STV(rankings)
        outcome_pos1 = positional_scoring(rankings, weights=w1)
        outcome_pos2 = positional_scoring(rankings, weights=w2)
        outcome_pos3 = positional_scoring(rankings, weights=w3)

        f1_plurality = f1_score(
            y_test_labels, outcome_plurality, average='weighted')
        f1_SVT = f1_score(y_test_labels, outcome_SVT, average='weighted')
        f1_pos1 = f1_score(y_test_labels, outcome_pos1, average='weighted')
        f1_pos2 = f1_score(y_test_labels, outcome_pos2, average='weighted')
        f1_pos3 = f1_score(y_test_labels, outcome_pos3, average='weighted')
        print('Plurality: ', f1_plurality)
        print('SVT: ', f1_SVT)
        print('POS1: ', f1_pos1)
        print('POS2: ', f1_pos2)
        print('POS3: ', f1_pos3)
        exp_evaluation_plurality.append(f1_plurality)
        exp_evaluation_svt.append(f1_SVT)
        exp_evaluation_pos1.append(f1_pos1)
        exp_evaluation_pos2.append(f1_pos2)
        exp_evaluation_pos3.append(f1_pos3)


final_f1_plurality = np.mean(exp_evaluation_plurality)
final_f1_SVT = np.mean(exp_evaluation_svt)
final_f1_pos1 = np.mean(exp_evaluation_pos1)
final_f1_pos2 = np.mean(exp_evaluation_pos2)
final_f1_pos3 = np.mean(exp_evaluation_pos3)
print(f'plurality final average f1: {final_f1_plurality}')
print(f'SVT final average f1: {final_f1_SVT}')
print(f'Pos f1: {final_f1_pos1}, with w={w1}')
print(f'Pos f1: {final_f1_pos2}, with w={w2}')
print(f'Pos f1: {final_f1_pos3}, with w={w3}')
print(
    f'Ran with n={num_classifiers}, folds={n_kfold_splits}, num_exp={n_experiments}')
