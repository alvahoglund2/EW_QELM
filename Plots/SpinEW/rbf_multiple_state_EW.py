import sys
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.lines import Line2D

sys.path.append('SVM')
from linear_svm import linear_svm
from rbf_svm import rbf_svm

# Define the palette
palette = [
    (100/255, 143/255, 255/255),  # Blue 
    (120/255, 94/255, 240/255),  # Purple
    (220/255, 38/255, 127/255),  # Pink
    (254/255, 97/255, 0/255),    # Red-Orange
    (255/255, 176/255, 0/255),   # Orange
]

rsvm = rbf_svm("Plots/SpinEW/data/multiple_state_measurements_train.npy",
                   "Plots/SpinEW/data/multiple_state_labels_train.npy",
                   "Plots/SpinEW/data/multiple_state_measurements_test.npy",
                   "Plots/SpinEW/data/multiple_state_labels_test.npy",
                   class_weights={-1: 1, 1: 10^5},
                   C=1,)

rsvm.print_accuracy()

def plot_data(rsvm):
    fig = plt.figure(figsize=(7, 5))
    ax = fig.add_subplot(111, projection='3d')

    X = rsvm.X_train
    y = rsvm.y_train
    x1_min, x1_max = X[:, 0].min() - 0.5, X[:, 0].max() + 0.5
    x2_min, x2_max = X[:, 1].min() - 0.5, X[:, 1].max() + 0.5
    x3_min, x3_max = X[:, 2].min() - 0.5, X[:, 2].max() + 0.5

    x1, x2, x3 = np.meshgrid(
        np.linspace(x1_min-2, x1_max+2, 100),
        np.linspace(x2_min-2, x2_max+2, 100),
        np.linspace(x3_min-2, x3_max+2, 100),
    )

    grid_points = np.c_[x1.ravel(), x2.ravel(), x3.ravel()]
    decision_values = rsvm.clf.decision_function(grid_points)
    boundary_points_norm = grid_points[np.abs(decision_values) < 0.1]
    boundary_points = rsvm.scaler.inverse_transform(boundary_points_norm)

    class_1_label = 'Entangled States'
    class_2_label = 'Separable States'
    handle_class_1 = Line2D([0], [0], marker='o', color='w', markerfacecolor=palette[0], markersize=10, label=class_1_label)
    handle_class_2 = Line2D([0], [0], marker='o', color='w', markerfacecolor=palette[2], markersize=10, label=class_2_label)

    fig.legend(handles=[handle_class_1, handle_class_2], loc='upper right', fontsize=12)
    
    label_to_index = {label: i for i, label in enumerate(np.unique(y))}
    indices = np.array([label_to_index[label] for label in y])

    colors = np.array([palette[0], palette[2]])
    point_colors = colors[indices]

    ax.scatter(boundary_points[:, 0], boundary_points[:, 1], boundary_points[:, 2], 
            c='black', alpha=0.1, s=0.3)


    ax.scatter(X[:, 0], X[:, 1], X[:, 2], c=point_colors, s=50, edgecolors='k')

    ax.set_xlim(x1_min, x1_max)
    ax.set_ylim(x2_min, x2_max)
    ax.set_zlim(x3_min, x3_max)
    ax.set_xlabel(r"X $\otimes$ X")
    ax.set_ylabel(r"Y $\otimes$ Y")
    ax.set_zlabel(r"Z $\otimes$ Z")
    ax.set_title("Nonlinear Multiple State EW")
    plt.tight_layout()
    plt.show()


def plot_data_on_ax(rsvm, ax, title=""):
    X = rsvm.X_train
    y = rsvm.y_train
    x1_min, x1_max = X[:, 0].min() - 0.5, X[:, 0].max() + 0.5
    x2_min, x2_max = X[:, 1].min() - 0.5, X[:, 1].max() + 0.5
    x3_min, x3_max = X[:, 2].min() - 0.5, X[:, 2].max() + 0.5

    x1, x2, x3 = np.meshgrid(
        np.linspace(x1_min - 2, x1_max + 2, 50),
        np.linspace(x2_min - 2, x2_max + 2, 50),
        np.linspace(x3_min - 2, x3_max + 2, 50),
    )

    grid_points = np.c_[x1.ravel(), x2.ravel(), x3.ravel()]
    decision_values = rsvm.clf.decision_function(grid_points)
    boundary_points_norm = grid_points[np.abs(decision_values) < 0.1]
    boundary_points = rsvm.scaler.inverse_transform(boundary_points_norm)

    label_to_index = {label: i for i, label in enumerate(np.unique(y))}
    indices = np.array([label_to_index[label] for label in y])
    colors = np.array([palette[0], palette[2]])
    point_colors = colors[indices]

    if rsvm.clf._gamma <0.3:
        ax.scatter(boundary_points[:, 0], boundary_points[:, 1], boundary_points[:, 2], 
               c='black', alpha=0.5, s=0.3)
    else:
        ax.scatter(boundary_points[:, 0], boundary_points[:, 1], boundary_points[:, 2], 
               c='black', alpha=0.2, s=0.3)
    frac = 10    
    ax.scatter(X[::frac, 0], X[::frac, 1], X[::frac, 2], c=point_colors[::frac], s=30, edgecolors='k')

    ax.set_xlim(x1_min, x1_max)
    ax.set_ylim(x2_min, x2_max)
    ax.set_zlim(x3_min, x3_max)
    ax.set_xlabel(r"X $\otimes$ X")
    ax.set_ylabel(r"Y $\otimes$ Y")
    ax.set_zlabel(r"Z $\otimes$ Z")
    ax.set_xticks([-2, -1, 0, 1, 2])
    ax.set_yticks([-2, -1, 0, 1, 2])
    ax.set_zticks([-2, -1, 0, 1, 2])
    ax.set_title(title)


def plot_varying_gamma():
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    gammas = [0, 1.5]
    titles = [r"$\gamma$ = 0.33", r"$\gamma$ = 1.5"]
    
    fig = plt.figure(figsize=(8, 5))
    axs = [fig.add_subplot(1, 2, i + 1, projection='3d') for i in range(2)]

    for i, gamma in enumerate(gammas):  
        rsvm = rbf_svm("Plots/SpinEW/data/multiple_state_measurements_train.npy",
                       "Plots/SpinEW/data/multiple_state_labels_train.npy",
                       "Plots/SpinEW/data/multiple_state_measurements_test.npy",
                       "Plots/SpinEW/data/multiple_state_labels_test.npy",
                       class_weights={-1: 1, 1: 10**5},
                       C=1,
                       gamma=gamma)
        print(f"Trained with gamma = {rsvm.clf._gamma}")
        plot_data_on_ax(rsvm, axs[i], title=titles[i])
        print_robustness(rsvm)

    # Legend
    handle_class_1 = Line2D([0], [0], marker='o', color='w', markerfacecolor=palette[0], markersize=10, label='Entangled')
    handle_class_2 = Line2D([0], [0], marker='o', color='w', markerfacecolor=palette[2], markersize=10, label='Separable')
    fig.legend(handles=[handle_class_1, handle_class_2], loc='upper right', fontsize=12)

    plt.suptitle("Nonlinear EW with Varying Gamma", fontsize=16)
    plt.tight_layout()
    fig.subplots_adjust(wspace=0.3)
    plt.show()


def print_robustness(rsvm):
    dec_vals1, p_tol_idx1 = rsvm.get_desicion_values_for_data("Plots/SpinEW/data/singlet_state_measurements_werner.npy")
    dec_vals2, p_tol_idx2 = rsvm.get_desicion_values_for_data("Plots/SpinEW/data/triplet0_state_measurements_werner.npy")
    dec_vals3, p_tol_idx3 = rsvm.get_desicion_values_for_data("Plots/SpinEW/data/tripletp1_state_measurements_werner.npy")
    dec_vals4, p_tol_idx4 = rsvm.get_desicion_values_for_data("Plots/SpinEW/data/tripletn1_state_measurements_werner.npy")
    p_range = np.linspace(0, 1, len(dec_vals1))
    p_tol1 = p_range[p_tol_idx1]
    p_tol2 = p_range[p_tol_idx2]
    p_tol3 = p_range[p_tol_idx3]
    p_tol4 = p_range[p_tol_idx4]
    print(f"p_tol Singlet: {p_tol1}")
    print(f"p_tol Triplet0: {p_tol2}")
    print(f"p_tol Tripletp1: {p_tol3}")
    print(f"p_tol Tripletn1: {p_tol4}")

#plot_data(rsvm)
#print_robustness()
plot_varying_gamma()