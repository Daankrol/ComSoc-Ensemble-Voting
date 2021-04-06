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


n_kfold_splits = 2
exp_evaluation_plurality = []
exp_evaluation_svt = []
exp_evaluation_pos = []

weights = []
for i in range(26):
    weights.append(26 - (i + 1))

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
            classifier.calculate_f1_score()

        # compute f1 score based on voting rule
        rankings = np.array([c.get_ranking() for c in classifiers])
        outcome_plurality = plurality(rankings)
        outcome_SVT = STV(rankings)
        outcome_pos = positional_scoring(rankings, weights=weights)

        # resolving ties
        outcome_plurality = solve_for_ties(outcome_plurality, classifiers)
        print('out', outcome_plurality)
        print('sysext')
        sys.exit(2)

        f1_plurality = f1_score(
            y_test_labels, outcome_plurality, average='weighted')
        f1_SVT = f1_score(y_test_labels, outcome_SVT, average='weighted')
        f1_pos = f1_score(y_test_labels, outcome_pos, average='weighted')
        print('Plurality: ', f1_plurality)
        print('SVT: ', f1_SVT)
        print('POS: ', f1_pos)
        exp_evaluation_plurality.append(f1_plurality)
        exp_evaluation_svt.append(f1_SVT)
        exp_evaluation_pos.append(f1_pos)


final_f1_plurality = np.mean(exp_evaluation_plurality)
final_f1_SVT = np.mean(exp_evaluation_svt)
final_f1_pos = np.mean(exp_evaluation_pos)
print(f'plurality final average f1: {final_f1_plurality}')
print(f'SVT final average f1: {final_f1_SVT}')
print(f'Positional scoring final average f1: {final_f1_pos}')
print(
    f'Ran with n={num_classifiers}, folds={n_kfold_splits}, num_exp={n_experiments}\n \
        using scoring vector:\n {weights}')
