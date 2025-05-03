import sys
import numpy as np
import matplotlib.pyplot as plt

sys.path.append('SVM')
from linear_svm import linear_svm
from rbf_svm import rbf_svm

lsvm = linear_svm("Charge_EW\data\measurements_test.npy", "Charge_EW\data\labels_test.npy",
                  "Charge_EW\data\measurements_train.npy", "Charge_EW\data\labels_train.npy",
                  class_weights={-1: 1, 1: 5000}, C=10)
lsvm.print_accuracy()
#lsvm.pca_plot()
#print(lsvm.get_robustness(evaluate_train=False, p_min =0.5))

rsvm = rbf_svm("Charge_EW\data\measurements_test.npy", "Charge_EW\data\labels_test.npy","Charge_EW\data\measurements_train.npy", "Charge_EW\data\labels_train.npy",
               class_weights={-1: 1, 1: 1}, C=1, gamma=2)
rsvm.print_accuracy()


#l = np.load("Charge_EW\data\labels_train.npy")

#Count the number of 1s and -1s in the labels
#count_1s = np.sum(l == 1)
#count_neg1s = np.sum(l == -1)
#print(f"Number of 1s: {count_1s}")
#print(f"Number of -1s: {count_neg1s}")