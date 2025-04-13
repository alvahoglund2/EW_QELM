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
# Create four different linear_svm instances
rsvm1 = rbf_svm("Plots/SpinEW/data/singlet_state_measurements_train.npy",
                   "Plots/SpinEW/data/singlet_state_labels_train.npy",
                   "Plots/SpinEW/data/singlet_state_measurements_test.npy",
                   "Plots/SpinEW/data/singlet_state_labels_test.npy",
                   class_weights={-1: 1, 1: 1000},
                   C=1)

rsvm2 = rbf_svm("Plots/SpinEW/data/triplet0_state_measurements_train.npy",
                   "Plots/SpinEW/data/triplet0_state_labels_train.npy",
                   "Plots/SpinEW/data/triplet0_state_measurements_test.npy",
                   "Plots/SpinEW/data/triplet0_state_labels_test.npy",
                   class_weights={-1: 1, 1: 1000},
                   C=1)

rsvm3 = rbf_svm("Plots/SpinEW/data/tripletp1_state_measurements_train.npy",
                   "Plots/SpinEW/data/tripletp1_state_labels_train.npy",
                   "Plots/SpinEW/data/tripletp1_state_measurements_test.npy",
                   "Plots/SpinEW/data/tripletp1_state_labels_test.npy",
                   class_weights={-1: 1, 1: 1000},
                   C=1)

rsvm4 = rbf_svm("Plots/SpinEW/data/tripletn1_state_measurements_train.npy",
                   "Plots/SpinEW/data/tripletn1_state_labels_train.npy",
                   "Plots/SpinEW/data/tripletn1_state_measurements_test.npy",
                   "Plots/SpinEW/data/tripletn1_state_labels_test.npy",
                   class_weights={-1: 1, 1: 1000},
                   C=1)

rsvm1.print_accuracy()
rsvm2.print_accuracy()
rsvm3.print_accuracy()
rsvm4.print_accuracy()

def plot_spin_decision_boundary(ax, rsvm, title):
    X_test = rsvm.X_test

    
    x1_min, x1_max = X_test[:, 0].min() - 1, X_test[:, 0].max() + 1
    x2_min, x2_max = X_test[:, 1].min() - 1, X_test[:, 1].max() + 1
    x3_min, x3_max = X_test[:, 2].min() - 1, X_test[:, 2].max() + 1

    x1, x2, x3 = np.meshgrid(
        np.linspace(x1_min, x1_max, 100),
        np.linspace(x2_min, x2_max, 100),
        np.linspace(x3_min, x3_max, 100),
    )

    #Compute decision boundary
    grid_points = np.c_[x1.ravel(), x2.ravel(), x3.ravel()]
    decision_values = rsvm.clf.decision_function(grid_points)
    boundary_points = grid_points[np.abs(decision_values) < 0.1]

    # Plot the decision boundary points
    ax.scatter(boundary_points[:, 0], boundary_points[:, 1], boundary_points[:, 2], 
            c='black', alpha=0.1, s=0.2, label='Decision Boundary')

    # Plot the decision boundary with transparency and draw it first

    # Plot the measurements (points) after plotting the surface to make sure they appear on top
    frac = 10
    label_to_index = {label: i for i, label in enumerate(np.unique(rsvm.y_test))}
    indices = np.array([label_to_index[label] for label in rsvm.y_test[::frac]])

    colors = np.array([palette[0], palette[2]])
    point_colors = colors[indices]

    ax.scatter(rsvm.X_test_normalized[::frac, 0], rsvm.X_test_normalized[::frac, 1], rsvm.X_test_normalized[::frac, 2], 
               c=point_colors, s=50, edgecolors='k', marker='o')

    # Create one global legend
    class_1_label = title
    class_2_label = 'Separable States'
    handle_class_1 = Line2D([0], [0], marker='o', color='w', markerfacecolor=palette[0], markersize=10, label=class_1_label)
    handle_class_2 = Line2D([0], [0], marker='o', color='w', markerfacecolor=palette[2], markersize=10, label=class_2_label)

    ax.legend(handles=[handle_class_1, handle_class_2], loc='upper center')

    # Set limits and labels
    ax.set_xlim(-2, 2)
    ax.set_ylim(-2, 2)
    ax.set_zlim(-2, 2)
    ax.set_xticks(np.arange(-2, 3, 1))
    ax.set_yticks(np.arange(-2, 3, 1))
    ax.set_zticks(np.arange(-2, 3, 1))
    ax.set_xlabel(r"X $\otimes$ X")
    ax.set_ylabel(r"Y $\otimes$ Y")
    ax.set_zlabel(r"Z $\otimes$ Z")
    #witness_string = f"W = {round(w[0]/w[0], 1)}X$\\otimes $X + {round(w[1]/w[0], 1)}Y$\\otimes $Y + {round(w[2]/w[0], 1)}Z$\\otimes $Z + {round(b/w[0], 1)}I$\\otimes $I"
    #ax.set_title(title, fontsize=12)

# Create a 2x2 grid of subplots
fig, axs = plt.subplots(2, 2, figsize=(10, 7), subplot_kw={'projection': '3d'})

# Plot each decision boundary in a subplot
plot_spin_decision_boundary(axs[0, 0], rsvm1, r"Entangled states - $\Psi_p^-$")
plot_spin_decision_boundary(axs[0, 1], rsvm2, r"Entangled states - $\Psi^+_p$")
plot_spin_decision_boundary(axs[1, 0], rsvm3, r"Entangled states - $\Phi^-_p$")
plot_spin_decision_boundary(axs[1, 1], rsvm4, r"Entangled states - $\Phi^+_p$")

fig.suptitle("Implemented EW for different Werner states", fontsize=16)
plt.tight_layout()
plt.show()


