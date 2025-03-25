import numpy as np
import matplotlib.pyplot as plt
from sklearn import svm
from sklearn.preprocessing import StandardScaler

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
    class_weights={-1: 1, 1: 1000}
    clf = svm.SVC(kernel='rbf', class_weight=class_weights)
    clf.fit(X_train, y_train)
    return clf

def evaluate_model(clf, X, y):
    y_pred = clf.predict(X)
    accuracy = np.mean(y_pred == y)
    accuracy_sep = np.mean(y_pred[y == 1] == 1)
    accuracy_ent = np.mean(y_pred[y == -1] == -1)
    return accuracy, accuracy_sep, accuracy_ent

def plot_data(X, y, clf, title=""):
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')

    x1_min, x1_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    x2_min, x2_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    x3_min, x3_max = X[:, 2].min() - 1, X[:, 2].max() + 1

    x1, x2, x3 = np.meshgrid(
        np.linspace(x1_min, x1_max, 100),
        np.linspace(x2_min, x2_max, 100),
        np.linspace(x3_min, x3_max, 100),
    )

    #Compute desicion boundary
    grid_points = np.c_[x1.ravel(), x2.ravel(), x3.ravel()]
    decision_values = clf.decision_function(grid_points)
    boundary_points = grid_points[np.abs(decision_values) < 0.1]

    # Plot the decision boundary points
    ax.scatter(boundary_points[:, 0], boundary_points[:, 1], boundary_points[:, 2], 
            c='black', alpha=1, s=0.2, label='Decision Boundary')

    # Plot the measurements
    ax.scatter(X[:, 0], X[:, 1], X[:, 2], c=y, cmap=plt.cm.coolwarm, s=50, edgecolors='k')

    ax.set_xlim(x1_min, x1_max)
    ax.set_ylim(x2_min, x2_max)
    ax.set_zlim(x3_min, x3_max)
    ax.set_xlabel(r"X $\otimes$ X")
    ax.set_ylabel(r"Y $\otimes$ Y")
    ax.set_zlabel(r"Z $\otimes$ Z")
    ax.set_title("3D Decision Boundary for RBF Kernel SVM")
    ax.legend()
    plt.tight_layout()
    plt.show()

def print_accuracy(clf, X_train_scaled, y_train, X_test_scaled, y_test):
    accuracy, accuracy_sep, accuracy_ent = evaluate_model(clf, X_train_scaled, y_train)
    print(f"Training Accuracy: {accuracy}")
    print(f"Training Accuracy (Separable States): {accuracy_sep}")
    print(f"Training Accuracy (Entangled States): {accuracy_ent}")

    accuracy, accuracy_sep, accuracy_ent = evaluate_model(clf, X_test_scaled, y_test)
    print(f"Test Accuracy: {accuracy}")
    print(f"Test Accuracy (Separable States): {accuracy_sep}")
    print(f"Test Accuracy (Entangled States): {accuracy_ent}")

def main():
    X_train, y_train, X_test, y_test = load_data("Spin_EW\data\measurements_test.npy", "Spin_EW\data\labels_train.npy", "Spin_EW\data\measurements_train.npy", "Spin_EW\data\labels_test.npy")
    
    X_train_scaled, X_test_scaled = normalize_data(X_train, X_test)
    clf = train_svm(X_train_scaled, y_train)

    print_accuracy(clf, X_train_scaled, y_train, X_test_scaled, y_test)
    plot_data(X_train_scaled, y_train, clf)

if __name__ == "__main__":
    main()
