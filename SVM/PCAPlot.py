import numpy as np
import matplotlib.pyplot as plt
from sklearn import svm
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from mpl_toolkits.mplot3d import Axes3D

def load_data(train_data_path, train_labels_path):
    X = np.load(train_data_path)
    y = np.load(train_labels_path)
    print(X.shape, y.shape)

    return X, y

def normalize_data(X):
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    return X_scaled, scaler

def pca_plot_single_class(X): 
    pca = PCA(n_components=3)
    X_train_pca = pca.fit_transform(X)

    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')

    ax.scatter(X_train_pca[:, 0], X_train_pca[:, 1], X_train_pca[:, 2], edgecolors='k')

    # Add labels for the axes
    ax.set_xlabel('PC1')
    ax.set_ylabel('PC2')
    ax.set_zlabel('PC3')
    ax.set_title("All Werner States")

    plt.show()

def pca_plot_two_classes(X, y):
    pca = PCA(n_components=3)
    X_pca = pca.fit_transform(X)

    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')

    colors = ['red' if label == -1 else 'blue' for label in y]

    ax.scatter(X_pca[:, 0], X_pca[:, 1], X_pca[:, 2], 
                         c=colors, edgecolors='k')

    ax.set_xlabel('PC1')
    ax.set_ylabel('PC2')
    ax.set_zlabel('PC3')

    # Create a legend
    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], marker='o', color='w', markerfacecolor='blue', markersize=10, label='Separable states'),
        Line2D([0], [0], marker='o', color='w', markerfacecolor='red', markersize=10, label='Werner states, p>0.5'),
    ]
    ax.legend(handles=legend_elements, loc='best')

    plt.show()

def pca_plot_three_classes(X, y):
    pca = PCA(n_components=3)
    X_pca = pca.fit_transform(X)

    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')

    colors = ['red' if label == -1 else 'blue' if label == 1 else 'green' for label in y]

    ax.scatter(X_pca[:, 0], X_pca[:, 1], X_pca[:, 2], 
               c=colors, edgecolors='k')

    ax.set_xlabel('PC1')
    ax.set_ylabel('PC2')
    ax.set_zlabel('PC3')

    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], marker='o', color='w', markerfacecolor='blue', markersize=10, label='Class 1'),
        Line2D([0], [0], marker='o', color='w', markerfacecolor='red', markersize=10, label='Class -1'),
        Line2D([0], [0], marker='o', color='w', markerfacecolor='green', markersize=10, label='Class 0'),
    ]
    ax.legend(handles=legend_elements, loc='best')

    plt.title("PCA Projection of Three Classes")
    plt.show()

def plot_components(X, y, title=""):

    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')

    colors = ['red' if label == -1 else 'blue' if label == 1 else 'green' for label in y]

    ax.scatter(X[:, 0], X[:, 1], X[:, 2], 
               c=colors, edgecolors='k')

    ax.set_xlabel('measurement 1')
    ax.set_ylabel('measurement 2')
    ax.set_zlabel('measurement 3')

    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], marker='o', color='w', markerfacecolor='blue', markersize=10, label='Class 1'),
        Line2D([0], [0], marker='o', color='w', markerfacecolor='red', markersize=10, label='Class -1'),
        Line2D([0], [0], marker='o', color='w', markerfacecolor='green', markersize=10, label='Class 0'),
    ]
    ax.legend(handles=legend_elements, loc='best')

    plt.title("PCA Projection of Three Classes")
    plt.show()


def main():
    X, y = load_data(
        "Charge_EW\data\measurements_test.npy", "Charge_EW\data\labels_test.npy"
    )
    X_normalized, scaler= normalize_data(X)

    #pca_plot_single_class(X)
    #pca_plot_single_class(X_normalized)

    plot_components(X,y)
    pca_plot_three_classes(X, y)

if __name__ == "__main__":
    main()