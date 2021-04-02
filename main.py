from prepare_dataset import Dataset
from neural_net import NeuralNet
from decision_tree import DecisionTree
from svm import SVM

dataset = Dataset()

# neural_net = NeuralNet(dataset)
# print(neural_net)

# tree = DecisionTree(dataset)
# print(tree)
# print(tree.predict([dataset.test_x[3]]))

svm = SVM(dataset)
print(svm)
print(svm.predict(dataset.test_x[:2]))
