import numpy as np
import matplotlib.pyplot as plt
from sklearn import svm
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.lines import Line2D

class pca_plot:
    def __init__(self,
                X_path,
                y_path,):
        self.X, self.y = self.load_data(X_path, y_path)
        self.X_normalized, self.scaler= self.normalize_data()


    def load_data(self, train_data_path, train_labels_path):
        X = np.load(train_data_path)
        y = np.load(train_labels_path)
        print(X.shape, y.shape)

        return X, y

    def normalize_data(self,):
        scaler = StandardScaler()
        X_normalized = scaler.fit_transform(self.X)
        return X_normalized, scaler

    def pca_plot_single_class(self): 
        pca = PCA(n_components=3)
        X_train_pca = pca.fit_transform(self.X_normalized)

        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection='3d')

        ax.scatter(X_train_pca[:, 0], X_train_pca[:, 1], X_train_pca[:, 2], edgecolors='k')

        # Add labels for the axes
        ax.set_xlabel('PC1')
        ax.set_ylabel('PC2')
        ax.set_zlabel('PC3')
        ax.set_title("All Werner States")

        plt.show()

    def pca_plot_two_classes(self):
        pca = PCA(n_components=3)
        X_pca = pca.fit_transform(self.X_normalized)

        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection='3d')

        colors = ['red' if label == -1 else 'blue' for label in self.y]

        ax.scatter(X_pca[:, 0], X_pca[:, 1], X_pca[:, 2], 
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

    def pca_plot_three_classes(self):
        pca = PCA(n_components=3)
        X_pca = pca.fit_transform(self.X_normalized)

        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection='3d')

        colors = ['red' if label == -1 else 'blue' if label == 1 else 'green' for label in self.y]

        ax.scatter(X_pca[:, 0], X_pca[:, 1], X_pca[:, 2], 
                c=colors, edgecolors='k')

        ax.set_xlabel('PC1')
        ax.set_ylabel('PC2')
        ax.set_zlabel('PC3')

        legend_elements = [
            Line2D([0], [0], marker='o', color='w', markerfacecolor='blue', markersize=10, label='Class 1'),
            Line2D([0], [0], marker='o', color='w', markerfacecolor='red', markersize=10, label='Class -1'),
            Line2D([0], [0], marker='o', color='w', markerfacecolor='green', markersize=10, label='Class 0'),
        ]
        ax.legend(handles=legend_elements, loc='best')

        plt.title("PCA Projection of Three Classes")
        plt.show()

    def plot_measurements(self, measurement_idx, measurement_labels, title=""):

        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection='3d')

        colors = ['red' if label == -1 else 'blue' if label == 1 else 'green' for label in self.y]

        ax.scatter(self.X[:, measurement_idx[0]], self.X[:, measurement_idx[1]], self.X[:, measurement_idx[2]], 
                c=colors, edgecolors='k')

        ax.set_xlabel(measurement_labels[0])
        ax.set_ylabel(measurement_labels[1])
        ax.set_zlabel(measurement_labels[2])

        from matplotlib.lines import Line2D
        legend_elements = [
            Line2D([0], [0], marker='o', color='w', markerfacecolor='blue', markersize=10, label='Class 1'),
            Line2D([0], [0], marker='o', color='w', markerfacecolor='red', markersize=10, label='Class -1'),
            Line2D([0], [0], marker='o', color='w', markerfacecolor='green', markersize=10, label='Class 0'),
        ]
        ax.legend(handles=legend_elements, loc='best')

        plt.title("PCA Projection of Three Classes")
        plt.show()