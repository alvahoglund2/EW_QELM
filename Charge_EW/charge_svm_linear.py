import numpy as np
import matplotlib.pyplot as plt
from sklearn.svm import LinearSVC
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from mpl_toolkits.mplot3d import Axes3D


def load_data(train_data_path, train_labels_path, test_data_path, test_labels_path):
    train_measurements = np.load(train_data_path).real
    train_labels = np.load(train_labels_path)
    test_measurements = np.load(test_data_path).real
    test_labels = np.load(test_labels_path)
    return train_measurements, train_labels, test_measurements, test_labels

def normalize_data(X_train, X_test):
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    return X_train_scaled, X_test_scaled, scaler

def train_svm(X_train_scaled, y_train, C_value=1):
    clf = LinearSVC(penalty='l2', class_weight={-1: 1, 1: 1000}, C=C_value)
    clf.fit(X_train_scaled, y_train)
    return clf

def evaluate_model(clf, X, y):
    y_pred = clf.predict(X)
    accuracy = np.mean(y_pred == y)
    accuracy_sep = np.mean(y_pred[y == 1] == 1)
    accuracy_ent = np.mean(y_pred[y == -1] == -1)
    return accuracy, accuracy_sep, accuracy_ent, y_pred

def rescale_weights(w_scaled, b_scaled, scaler):
    w = w_scaled / scaler.scale_
    b = b_scaled - np.dot(w_scaled, scaler.mean_ / scaler.scale_)
    return w, b

def print_accuracy(clf, X_train_scaled, y_train, X_test_scaled, y_test):
    accuracy, accuracy_sep, accuracy_ent, y_pred = evaluate_model(clf, X_train_scaled, y_train)
    print(f"Training Accuracy: {accuracy}")
    print(f"Training Accuracy Separable: {accuracy_sep}")
    print(f"Training Accuracy Entangled: {accuracy_ent}")

    accuracy, accuracy_sep, accuracy_ent, y_pred = evaluate_model(clf, X_test_scaled, y_test)
    print(f"Test Accuracy: {accuracy}")
    print(f"Test Accuracy Separable: {accuracy_sep}")
    print(f"Test Accuracy Entangled: {accuracy_ent}")

def print_weights(clf, scaler):
    w_normalized = clf.coef_[0]
    b_normalized = clf.intercept_[0]

    print(f"Weight vector with normalized data: {w_normalized}")
    print(f"Bias with normalized data: {b_normalized}")

    w_original, b_original = rescale_weights(w_normalized, b_normalized, scaler)

    print(f"Weight vector without normalization : {w_original}")
    print(f"Bias without normalization : {b_original}")

def pca_visualization(X_train_scaled, y_train, title=""):
    pca = PCA(n_components=3)
    X_train_pca = pca.fit_transform(X_train_scaled)

    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')

    colors = ['red' if label == -1 else 'blue' for label in y_train]

    scatter = ax.scatter(X_train_pca[:, 0], X_train_pca[:, 1], X_train_pca[:, 2], 
                         c=colors, edgecolors='k')

    # Add labels for the axes
    ax.set_xlabel('PC1')
    ax.set_ylabel('PC2')
    ax.set_zlabel('PC3')
    ax.set_title(title)

    # Create a legend
    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], marker='o', color='w', markerfacecolor='blue', markersize=10, label='Separable states'),
        Line2D([0], [0], marker='o', color='w', markerfacecolor='red', markersize=10, label='Singlet Werner State, p>0.8'),
    ]
    ax.legend(handles=legend_elements, loc='best')

    plt.show()


def main():
    X_train, y_train, X_test, y_test = load_data(
        "Charge_EW\data3\measurments_test.npy", "Charge_EW\data3\labels_train.npy", 
        "Charge_EW\data3\measurments_train.npy", "Charge_EW\data3\labels_test.npy")

    X_train_scaled, X_test_scaled, scaler = normalize_data(X_train, X_test)
    clf = train_svm(X_train_scaled, y_train)
    print_accuracy(clf, X_train_scaled, y_train, X_test_scaled, y_test)
    pca_visualization(X_train_scaled, y_train, "Normalized Data")
    pca_visualization(X_train, y_train, "Original Data")

if __name__ == "__main__":
    main()