import numpy as np
import matplotlib.pyplot as plt
from sklearn import svm
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.lines import Line2D


def load_data(train_data_path, train_labels_path, test_data_path, test_labels_path):
    X_train = np.load(train_data_path)
    y_train = np.load(train_labels_path)
    X_test = np.load(test_data_path)
    y_test = np.load(test_labels_path)
    print(X_train.shape, y_train.shape, X_test.shape, y_test.shape)

    return X_train, y_train, X_test, y_test

def normalize_data(X_train, X_test):
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    return X_train_scaled, X_test_scaled

def train_svm(X_train, y_train):
    class_weights={-1: 1, 1: 10000}
    clf = svm.SVC(kernel='rbf', class_weight=class_weights)
    clf.fit(X_train, y_train)
    return clf

def evaluate_model(clf, X, y):
    y_pred = clf.predict(X)
    accuracy = np.mean(y_pred == y)
    accuracy_sep = np.mean(y_pred[y == 1] == 1)
    accuracy_ent = np.mean(y_pred[y == -1] == -1)
    return accuracy, accuracy_sep, accuracy_ent

def print_accuracy(clf, X_train_scaled, y_train, X_test_scaled, y_test):
    accuracy, accuracy_sep, accuracy_ent = evaluate_model(clf, X_train_scaled, y_train)
    print(f"Training Accuracy: {accuracy}")
    print(f"Training Accuracy (Separable States): {accuracy_sep}")
    print(f"Training Accuracy (Entangled States): {accuracy_ent}")

    accuracy, accuracy_sep, accuracy_ent = evaluate_model(clf, X_test_scaled, y_test)
    print(f"Test Accuracy: {accuracy}")
    print(f"Test Accuracy (Separable States): {accuracy_sep}")
    print(f"Test Accuracy (Entangled States): {accuracy_ent}")


def pca_visualization(X_train_scaled, y_train):
    pca = PCA(n_components=3)
    X_train_pca = pca.fit_transform(X_train_scaled)

    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')

    colors = ['red' if label == -1 else 'blue' for label in y_train]

    scatter = ax.scatter(X_train_pca[:, 0], X_train_pca[:, 1], X_train_pca[:, 2], 
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


def main():
    X_train, y_train, X_test, y_test = load_data(
        "Charge_EW\data\measurements_train.npy", "Charge_EW\data\labels_train.npy", 
        "Charge_EW\data\measurements_test.npy", "Charge_EW\data\labels_test.npy")

    X_train_scaled, X_test_scaled = normalize_data(X_train, X_test)
    clf = train_svm(X_train_scaled, y_train)
    print_accuracy(clf, X_train_scaled, y_train, X_test_scaled, y_test)
    pca_visualization(X_train_scaled, y_train)

if __name__ == "__main__":
    main()