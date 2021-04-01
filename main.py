from prepare_dataset import Dataset
from neural_net import NeuralNet
from decision_tree import DecisionTree
from svm import SVM

dataset = Dataset()

neural_net = NeuralNet(dataset)
print(neural_net)

tree = DecisionTree(dataset)
print(tree)

svm = SVM(dataset)
print(svm)
