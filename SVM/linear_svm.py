import numpy as np
import matplotlib.pyplot as plt
from sklearn.svm import LinearSVC
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.lines import Line2D

class linear_svm:
    def __init__(self,
                 X_train_path,
                 y_train_path,
                 X_test_path,
                 y_test_path):
        
        self.X_train, self.y_train, self.X_test, self.y_test = self.load_data(X_train_path, y_train_path, X_test_path, y_test_path)
        self.X_train_normalized, self.X_test_normalized, self.scaler = self.normalize_data(self.X_train, self.X_test)
        self.clf = self.train_svm()

    def load_data(self, train_data_path, train_labels_path, test_data_path, test_labels_path):
        train_measurements = np.load(train_data_path).real
        train_labels = np.load(train_labels_path)
        test_measurements = np.load(test_data_path).real
        test_labels = np.load(test_labels_path)
        return train_measurements, train_labels, test_measurements, test_labels

    def normalize_data(self, X_train, X_test):
        scaler = StandardScaler()
        X_train_normalized = scaler.fit_transform(X_train)
        X_test_normalized = scaler.transform(X_test)
        return X_train_normalized, X_test_normalized, scaler

    def train_svm(self, C_value=1):
        clf = LinearSVC(penalty='l2', class_weight={-1: 1, 1: 10000}, C=C_value)
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

    def get_weights(self):
        """
        Get weights and bias of the model in the normalized space
        """
        w_normalized = self.clf.coef_[0]
        b_normalized = self.clf.intercept_[0]
        return w_normalized, b_normalized

    def get_original_weights(self):
        """
        Get weights and bias of the model in the original space before normalization
        """
        w_normalized, b_normalized = self.get_weights()
        w = w_normalized / self.scaler.scale_
        b = b_normalized - np.dot(w_normalized, self.scaler.mean_ / self.scaler.scale_)
        return w, b

    def print_accuracy(self):
        accuracy, accuracy_sep, accuracy_ent, y_pred = self.evaluate_model(evaluate_train=True)
        print(f"Training Accuracy: {accuracy}")
        print(f"Training Accuracy Separable: {accuracy_sep}")
        print(f"Training Accuracy Entangled: {accuracy_ent}")

        accuracy, accuracy_sep, accuracy_ent, y_pred = self.evaluate_model(evaluate_train=False)
        print(f"Test Accuracy: {accuracy}")
        print(f"Test Accuracy Separable: {accuracy_sep}")
        print(f"Test Accuracy Entangled: {accuracy_ent}")

    def print_weights(self):
        w_normalized, b_normalized = self.get_weights()
        w_original, b_original = self.get_original_weights()

        print(f"Weight vector with normalized data: {w_normalized}")
        print(f"Bias with normalized data: {b_normalized}")
        print(f"Weight vector without normalization : {w_original}")
        print(f"Bias without normalization : {b_original}")

    def pca_plot(self, title=""):
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
        ax.set_title(title)

        legend_elements = [
            Line2D([0], [0], marker='o', color='w', markerfacecolor='blue', markersize=10, label='Separable states'),
            Line2D([0], [0], marker='o', color='w', markerfacecolor='red', markersize=10, label='Singlet Werner State, p>0.8'),
        ]
        ax.legend(handles=legend_elements, loc='best')

        plt.show()


    def measurement_plot(self, measurements_idx, measurement_labels, title="", ):
        
        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection='3d')

        colors = ['red' if label == -1 else 'blue' for label in self.y_train]

        ax.scatter(self.X_train[:, measurements_idx[0]], self.X_train[:, measurements_idx[1]], self.X_train[:, measurements_idx[2]], 
                            c=colors, edgecolors='k')

        ax.set_xlabel(measurement_labels[0])
        ax.set_ylabel(measurement_labels[1])
        ax.set_zlabel(measurement_labels[2])
        ax.set_title(title)

        legend_elements = [
            Line2D([0], [0], marker='o', color='w', markerfacecolor='blue', markersize=10, label='Separable states'),
            Line2D([0], [0], marker='o', color='w', markerfacecolor='red', markersize=10, label='Singlet Werner State, p>0.8'),
        ]
        ax.legend(handles=legend_elements, loc='best')

        plt.show()