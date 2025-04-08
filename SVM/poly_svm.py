import numpy as np
import matplotlib.pyplot as plt
from sklearn import svm
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.lines import Line2D

class poly_svm:
    def __init__(self,
                 X_train_path,
                 y_train_path,
                 X_test_path,
                 y_test_path,
                 class_weights={-1: 1, 1: 1000},
                 C = 1,
                 degree = 3):
        
        print("Loading data...")
        self.X_train, self.y_train, self.X_test, self.y_test = self.load_data(X_train_path, y_train_path, X_test_path, y_test_path)
        print("Normalizing data...")
        self.X_train_normalized, self.X_test_normalized, self.scaler = self.normalize_data()
        print("Training SVM...")
        self.clf = self.train_svm(class_weights, C, degree)


    def load_data(self, X_train_path, y_train_path, X_test_path, y_test_path):
        X_train = np.load(X_train_path)
        y_train = np.load(y_train_path)
        X_test = np.load(X_test_path)
        y_test = np.load(y_test_path)
        return X_train, y_train, X_test, y_test

    def normalize_data(self):
        scaler = StandardScaler()
        X_train_normalized = scaler.fit_transform(self.X_train)
        X_test_normalized = scaler.transform(self.X_test)
        return X_train_normalized, X_test_normalized, scaler

    def train_svm(self, class_weights, C, degree):
        clf = svm.SVC(kernel='poly', degree=degree, C=C, class_weight=class_weights)
        clf.fit(self.X_train_normalized, self.y_train)
        return clf

    def evaluate_model(self, evaluate_train):
        if evaluate_train:
            X = self.X_train_normalized
            y = self.y_train
        else:
            X = self.X_test_normalized
            y = self.y_test

        y_pred = self.clf.predict(X)
        accuracy = np.mean(y_pred == y)
        accuracy_sep = np.mean(y_pred[y == 1] == 1)
        accuracy_ent = np.mean(y_pred[y == -1] == -1)
        return accuracy, accuracy_sep, accuracy_ent, y_pred

    def print_accuracy(self):
        accuracy, accuracy_sep, accuracy_ent, y_pred = self.evaluate_model(evaluate_train=True)
        print(f"Training Accuracy: {accuracy}")
        print(f"Training Accuracy Separable: {accuracy_sep}")
        print(f"Training Accuracy Entangled: {accuracy_ent}")

        accuracy, accuracy_sep, accuracy_ent, y_pred = self.evaluate_model(evaluate_train=False)
        print(f"Test Accuracy: {accuracy}")
        print(f"Test Accuracy Separable: {accuracy_sep}")
        print(f"Test Accuracy Entangled: {accuracy_ent}")

    def pca_visualization(self,):
        pca = PCA(n_components=3)
        X_train_pca = pca.fit_transform(self.X_train_normalized)

        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection='3d')

        colors = ['red' if label == -1 else 'blue' for label in self.y_train]

        ax.scatter(X_train_pca[:, 0], X_train_pca[:, 1], X_train_pca[:, 2], 
                            c=colors, edgecolors='k')

        ax.set_xlabel('PC1')
        ax.set_ylabel('PC2')
        ax.set_zlabel('PC3')

        legend_elements = [
            Line2D([0], [0], marker='o', color='w', markerfacecolor='blue', markersize=10, label='Separable states'),
            Line2D([0], [0], marker='o', color='w', markerfacecolor='red', markersize=10, label='Werner states, p>0.5'),
        ]
        ax.legend(handles=legend_elements, loc='best')

        plt.show()
    
    def get_desicion_values_for_data(self, X_path):
        X = np.load(X_path)
        X_normalized = self.scaler.transform(X)
        y_pred = self.clf.decision_function(X_normalized)
        zero_idx = np.argmin(np.abs(y_pred))
        return y_pred, zero_idx

    def get_robustness(self, evaluate_train, p_min):
        X_ent = 0
        if evaluate_train:
            X_ent = self.X_train_normalized[self.y_train == -1]
        else:
            X_ent = self.X_test_normalized[self.y_test == -1]
        y_pred = self.clf.decision_function(X_ent)
        zero_idx = np.argmin(np.abs(y_pred))
        p_min = 1/2
        p_range = np.linspace(p_min, 1, X_ent.shape[0])
        return p_range[zero_idx]