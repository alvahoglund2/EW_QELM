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
lsvm1 = linear_svm("Plots/SpinEW/data/singlet_state_measurements_train.npy",
                   "Plots/SpinEW/data/singlet_state_labels_train.npy",
                   "Plots/SpinEW/data/singlet_state_measurements_test.npy",
                   "Plots/SpinEW/data/singlet_state_labels_test.npy",
                   class_weights={-1: 1, 1: 1000},
                   C=1)

lsvm2 = linear_svm("Plots/SpinEW/data/triplet0_state_measurements_train.npy",
                   "Plots/SpinEW/data/triplet0_state_labels_train.npy",
                   "Plots/SpinEW/data/triplet0_state_measurements_test.npy",
                   "Plots/SpinEW/data/triplet0_state_labels_test.npy",
                   class_weights={-1: 1, 1: 1000},
                   C=1)

lsvm3 = linear_svm("Plots/SpinEW/data/tripletp1_state_measurements_train.npy",
                   "Plots/SpinEW/data/tripletp1_state_labels_train.npy",
                   "Plots/SpinEW/data/tripletp1_state_measurements_test.npy",
                   "Plots/SpinEW/data/tripletp1_state_labels_test.npy",
                   class_weights={-1: 1, 1: 1000},
                   C=1)

lsvm4 = linear_svm("Plots/SpinEW/data/tripletn1_state_measurements_train.npy",
                   "Plots/SpinEW/data/tripletn1_state_labels_train.npy",
                   "Plots/SpinEW/data/tripletn1_state_measurements_test.npy",
                   "Plots/SpinEW/data/tripletn1_state_labels_test.npy",
                   class_weights={-1: 1, 1: 1000},
                   C=1)

def plot_spin_decision_boundary(ax, lsvm, title):
    X_test = lsvm.X_test
    x1_min, x1_max = X_test[:, 0].min() - 1, X_test[:, 0].max() + 1
    x2_min, x2_max = X_test[:, 1].min() - 1, X_test[:, 1].max() + 1
    x1, x2 = np.meshgrid(np.linspace(x1_min, x1_max, 50), np.linspace(x2_min, x2_max, 50))

    w, b = lsvm.get_original_weights()

    if w[2] != 0:
        x3 = -(w[0] * x1 + w[1] * x2 + b) / w[2]
    else:
        raise ValueError("The weight corresponding to the third feature is zero, so the decision boundary cannot be plotted as a plane.")

    frac = 10
    label_to_index = {label: i for i, label in enumerate(np.unique(lsvm.y_test))}
    indices = np.array([label_to_index[label] for label in lsvm.y_test[::frac]])

    colors = np.array([palette[0], palette[2]])
    point_colors = colors[indices]

    ax.scatter(lsvm.X_test[::frac, 0], lsvm.X_test[::frac, 1], lsvm.X_test[::frac, 2], 
               c=point_colors, s=50, edgecolors='k', marker='o')

    ax.plot_surface(x1, x2, x3, color='lightblue', alpha=0.3)

    class_1_label = title
    class_2_label = 'Separable States'
    handle_class_1 = Line2D([0], [0], marker='o', color='w', markerfacecolor=palette[0], markersize=10, label=class_1_label)
    handle_class_2 = Line2D([0], [0], marker='o', color='w', markerfacecolor=palette[2], markersize=10, label=class_2_label)

    ax.legend(handles=[handle_class_1, handle_class_2], loc='upper center')

    ax.set_xlim(-2, 2)
    ax.set_ylim(-2, 2)
    ax.set_zlim(-2, 2)
    ax.set_xticks(np.arange(-2, 3, 1))
    ax.set_yticks(np.arange(-2, 3, 1))
    ax.set_zticks(np.arange(-2, 3, 1))
    ax.set_xlabel(r"X $\otimes$ X")
    ax.set_ylabel(r"Y $\otimes$ Y")
    ax.set_zlabel(r"Z $\otimes$ Z")

def print_decision_boundaries():
    w1, b1 = lsvm1.get_original_weights()
    w2, b2 = lsvm2.get_original_weights()
    w3, b3 = lsvm3.get_original_weights()
    w4, b4 = lsvm4.get_original_weights()

    print(f"Singlet state: wxx = {round(w1[0]/w1[0], 1)} wyy =  {round(w1[1]/w1[0], 1)}, wzz =  {round(w1[2]/w1[0],1)}  , b = {round(b1/w1[0],1)}")
    print(f"Triplet0 state: wxx = {round(w2[0]/w2[0], 1)} wyy =  {round(w2[1]/w2[0], 1)}, wzz =  {round(w2[2]/w2[0],1)}  , b = {round(b2/w2[0],1)}")
    print(f"Tripletp1 state: wxx = {round(w3[0]/w3[0], 1)} wyy =  {round(w3[1]/w3[0], 1)}, wzz =  {round(w3[2]/w3[0],1)}  , b = {round(b3/w3[0],1)}")
    print(f"Tripletn1 state: wxx = {round(w4[0]/w4[0], 1)} wyy =  {round(w4[1]/w4[0], 1)}, wzz =  {round(w4[2]/w4[0],1)}  , b = {round(b4/w4[0],1)}")

def print_accuracies():
    lsvm1.print_accuracy()
    lsvm2.print_accuracy()
    lsvm3.print_accuracy()
    lsvm4.print_accuracy()

def plot_single_state_EWs():
    fig, axs = plt.subplots(2, 2, figsize=(10, 7), subplot_kw={'projection': '3d'})

    plot_spin_decision_boundary(axs[0, 0], lsvm1, r"Entangled states - $\Psi_p^-$")
    plot_spin_decision_boundary(axs[0, 1], lsvm2, r"Entangled states - $\Psi^+_p$")
    plot_spin_decision_boundary(axs[1, 0], lsvm3, r"Entangled states - $\Phi^-_p$")
    plot_spin_decision_boundary(axs[1, 1], lsvm4, r"Entangled states - $\Phi^+_p$")
    fig.suptitle("Implemented EW for different Werner states", fontsize=16)

    plt.tight_layout()
    plt.show()

def print_robustness():
    dec_vals1, p_tol_idx1 = lsvm1.get_desicion_values_for_data("Plots/SpinEW/data/singlet_state_measurements_werner.npy")
    dec_vals2, p_tol_idx2 = lsvm2.get_desicion_values_for_data("Plots/SpinEW/data/triplet0_state_measurements_werner.npy")
    dec_vals3, p_tol_idx3 = lsvm3.get_desicion_values_for_data("Plots/SpinEW/data/tripletp1_state_measurements_werner.npy")
    dec_vals4, p_tol_idx4 = lsvm4.get_desicion_values_for_data("Plots/SpinEW/data/tripletn1_state_measurements_werner.npy")
    p_range = np.linspace(0, 1, len(dec_vals1))
    p_tol1 = p_range[p_tol_idx1]
    p_tol2 = p_range[p_tol_idx2]
    p_tol3 = p_range[p_tol_idx3]
    p_tol4 = p_range[p_tol_idx4]
    print(f"p_tol Singlet: {p_tol1}")
    print(f"p_tol Triplet0: {p_tol2}")
    print(f"p_tol Tripletp1: {p_tol3}")
    print(f"p_tol Tripletn1: {p_tol4}")

def plot_robustness():
    dec_vals1, p_tol_idx1 = lsvm1.get_desicion_values_for_data("Plots/SpinEW/data/singlet_state_measurements_werner.npy")
    dec_vals2, p_tol_idx2 = lsvm2.get_desicion_values_for_data("Plots/SpinEW/data/triplet0_state_measurements_werner.npy")
    dec_vals3, p_tol_idx3 = lsvm3.get_desicion_values_for_data("Plots/SpinEW/data/tripletp1_state_measurements_werner.npy")
    dec_vals4, p_tol_idx4 = lsvm4.get_desicion_values_for_data("Plots/SpinEW/data/tripletn1_state_measurements_werner.npy")

    plt.plot(dec_vals1)
    plt.plot(dec_vals2)
    plt.plot(dec_vals3)
    plt.plot(dec_vals4)
    plt.title("Decision values for different Werner states")
    plt.xlabel("Sample index")
    plt.ylabel("Decision value")
    plt.legend([r"$\Psi^-$", r"$\Psi^+$", r"$\Phi^-$", r"$\Phi^+$"])
    plt.show()

print_decision_boundaries()
print_accuracies()
plot_single_state_EWs()
