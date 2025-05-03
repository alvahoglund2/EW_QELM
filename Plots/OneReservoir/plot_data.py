import sys
import numpy as np
import matplotlib.pyplot as plt

sys.path.append('SVM')
from linear_svm import linear_svm
from rbf_svm import rbf_svm

palette = [
    (100/255, 143/255, 255/255),  # Blue 
    (120/255, 94/255, 240/255),  # Purple
    (220/255, 38/255, 127/255),  # Pink
    (254/255, 97/255, 0/255),    # Red-Orange
    (255/255, 176/255, 0/255),   # Orange
]

labels ={"singlet_state": r"$\Psi_p^-$", "triplet0_state": r"$\Psi_p^+$", "tripletp1_state": r"$\Phi_p^+$", "tripletn1_state": r"$\Phi_p^-$"}

def train_lsvm():
    ent_state_types = ["singlet_state", "triplet0_state", "tripletp1_state", "tripletn1_state"]
    seed = 4
    res_qd = 4
    qn = 1
    y_min, y_max = float('inf'), float('-inf')

    for ent_state_type in ent_state_types:
        lsvm = linear_svm(
                        f"Plots/OneReservoir/data/data_{ent_state_type}/measurements_train_res_{res_qd}_qn_{qn}_seed_{seed}.npy",
                        f"Plots/OneReservoir/data/data_{ent_state_type}/labels_train_res_{res_qd}_qn_{qn}_{seed}.npy",
                        f"Plots/OneReservoir/data/data_{ent_state_type}/measurements_test_res_{res_qd}_qn_{qn}_{seed}.npy",
                        f"Plots/OneReservoir/data/data_{ent_state_type}/labels_test_res_{res_qd}_qn_{qn}_{seed}.npy",
                        class_weights={-1: 1, 1: 1000},
                        C=1,
                    )
        print("Finished training of:")
        print(ent_state_type)
        lsvm.print_accuracy()
        dec_val, p_idx = lsvm.get_desicion_values_for_data(f"Plots/OneReservoir/data/data_werner_{ent_state_type}/measurements_{res_qd}_qn_{qn}_seed_{seed}.npy",)
        p_range = np.linspace(0, 1, len(dec_val))
        p_tol = p_range[p_idx]

        print("p_tol:")
        print(p_tol)
        plt.plot(p_range, dec_val, ':', linewidth = 2,  label=labels[ent_state_type], color=palette[ent_state_types.index(ent_state_type)])
        plt.axvline(x = p_tol, color='grey',  linestyle='-', linewidth=1, alpha=0.5)
        print("Finished getting decision values")
        y_min = min(y_min, np.min(dec_val))
        y_max = max(y_max, np.max(dec_val))
    plt.xlabel("p")
    plt.ylabel("Decision Value")
    #plt.axvline(x = 1/3, color='black', linestyle='--', label='p=1/3')
    y_min = y_min - 0.2
    y_max = y_max + 0.2
    plt.ylim(y_min, y_max)
    plt.xlim(0, 1)  
    y_zero_norm = (0 - y_min) / (y_max - y_min)  
    #plt.axvspan(0, 1/3, ymin=y_zero_norm, ymax=1, facecolor=palette[0], alpha=0.2) 
    plt.axvspan(0, 1/3, ymin=0, ymax=y_zero_norm, facecolor="lightgrey")
    plt.axvspan(1/3, 1, ymin=y_zero_norm, ymax=1, facecolor="lightgrey")
    #plt.axvspan(1/3, 1, ymin=0, ymax=y_zero_norm, facecolor=palette[0], alpha=0.2)
    plt.text(0.05, 0.5, 'True Separable', fontsize=10,)
    plt.text(0.05, -1, 'False Entangled', fontsize=10,)
    plt.text(0.77, 0.5, 'False Separable', fontsize=10,)
    plt.text(0.77, -1, 'True Entangled', fontsize=10,)
    plt.title(f"Linear EWs - Single State")
    plt.legend()
    #plt.grid(True)
    plt.show()

def train_rsvm():
    ent_state_types = ["singlet_state", "triplet0_state", "tripletp1_state", "tripletn1_state"]
    seed = 4
    res_qd = 4
    qn = 1
    y_min, y_max = float('inf'), float('-inf')

    for ent_state_type in ent_state_types:
        rsvm = rbf_svm(
                        f"Plots/OneReservoir/data/data_{ent_state_type}/measurements_train_res_{res_qd}_qn_{qn}_seed_{seed}.npy",
                        f"Plots/OneReservoir/data/data_{ent_state_type}/labels_train_res_{res_qd}_qn_{qn}_{seed}.npy",
                        f"Plots/OneReservoir/data/data_{ent_state_type}/measurements_test_res_{res_qd}_qn_{qn}_{seed}.npy",
                        f"Plots/OneReservoir/data/data_{ent_state_type}/labels_test_res_{res_qd}_qn_{qn}_{seed}.npy",
                        class_weights={-1: 1, 1: 1000},
                        C=1,
                        gamma = 1.5
                    )

        print("gamma: ")
        print(rsvm.clf._gamma)
        print("Finished training")
        dec_val, p_idx = rsvm.get_desicion_values_for_data(f"Plots/OneReservoir/data/data_werner_{ent_state_type}/measurements_{res_qd}_qn_{qn}_seed_{seed}.npy",)
        p_range = np.linspace(0, 1, len(dec_val))
        p_tol = p_range[p_idx]
        print("p_tol:")
        print(p_tol)
        plt.plot(p_range, dec_val,':', linewidth = 2,  label=labels[ent_state_type], color=palette[ent_state_types.index(ent_state_type)])
        plt.axvline(x = p_tol, color='grey',  linestyle='-', linewidth=1, alpha=0.5)
        print("Finished getting decision values")
        y_min = min(y_min, np.min(dec_val))
        y_max = max(y_max, np.max(dec_val))
        rsvm.print_accuracy()
    y_min = y_min - 0.2
    y_max = y_max + 0.2
    plt.ylim(y_min, y_max)
    plt.xlim(0, 1)
    y_zero_norm = (0 - y_min) / (y_max - y_min)  
    #plt.axvspan(0, 1/3, ymin=y_zero_norm, ymax=1, facecolor=palette[0], alpha=0.2) 
    plt.axvspan(0, 1/3, ymin=0, ymax=y_zero_norm, facecolor="lightgrey")
    plt.axvspan(1/3, 1, ymin=y_zero_norm, ymax=1, facecolor="lightgrey")
    #plt.axvspan(1/3, 1, ymin=0, ymax=y_zero_norm, facecolor=palette[0], alpha=0.2)
    plt.xlabel("p")
    plt.ylabel("Decision Value") 
    plt.title(f"Nonlinear EW - Single State")
    plt.text(0.05, 0.25, 'True Separable', fontsize=10,)
    plt.text(0.05, -0.5, 'False Entangled', fontsize=10,)
    plt.text(0.77, 0.25, 'False Separable', fontsize=10,)
    plt.text(0.77, -0.5, 'True Entangled', fontsize=10,)
    #plt.grid(True)
    plt.legend()
    plt.show()

def train_rsvm_multiple_states(): 
    ent_state_types = ["singlet_state", "triplet0_state", "tripletp1_state", "tripletn1_state"]
    seed = 4
    res_qd = 4
    qn = 1
    y_min, y_max = float('inf'), float('-inf')
    rsvm = rbf_svm(f"Plots/OneReservoir/data/data_all_states/measurements_train_res_{res_qd}_qn_{qn}_seed_{seed}.npy",
                    f"Plots/OneReservoir/data/data_all_states/labels_train_res_{res_qd}_qn_{qn}_{seed}.npy",
                    f"Plots/OneReservoir/data/data_all_states/measurements_test_res_{res_qd}_qn_{qn}_{seed}.npy",
                    f"Plots/OneReservoir/data/data_all_states/labels_test_res_{res_qd}_qn_{qn}_{seed}.npy",
                    class_weights={-1: 1, 1: 1000},
                    C=1,
                    gamma=0.5
                    )
    
    print("Finished training")
    rsvm.print_accuracy()
    for ent_state_type in ent_state_types:
        dec_val, p_idx = rsvm.get_desicion_values_for_data(f"Plots/OneReservoir/data/data_werner_{ent_state_type}/measurements_{res_qd}_qn_{qn}_seed_{seed}.npy",)
        p_range = np.linspace(0, 1, len(dec_val))
        p_tol = p_range[p_idx]
        plt.axvline(x = p_tol, color='grey',  linestyle='-', linewidth=1, alpha=0.5)
        plt.plot(np.linspace(0,1, len(dec_val)), dec_val, ':', linewidth = 2,  label=labels[ent_state_type], color=palette[ent_state_types.index(ent_state_type)])
        print("State:" + ent_state_type)
        print("p_tol:")
        print(p_tol)
        print()

        y_min = min(y_min, np.min(dec_val))
        y_max = max(y_max, np.max(dec_val))
    rsvm.print_accuracy()

    y_min = y_min - 0.2
    y_max = y_max + 0.2
    plt.ylim(y_min, y_max)  
    plt.xlim(0, 1)
    y_zero_norm = (0 - y_min) / (y_max - y_min)  
    #plt.axvspan(0, 1/3, ymin=y_zero_norm, ymax=1, facecolor=palette[0], alpha=0.2) 
    plt.axvspan(0, 1/3, ymin=0, ymax=y_zero_norm, facecolor="lightgrey")
    plt.axvspan(1/3, 1, ymin=y_zero_norm, ymax=1, facecolor="lightgrey")
    #plt.axvspan(1/3, 1, ymin=0, ymax=y_zero_norm, facecolor=palette[0], alpha=0.2)
    plt.xlabel("p")
    plt.ylabel("Decision Value") 
    plt.title(f"Nonlinear EW - Multiple States")
    plt.text(0.05, 0.25, 'True Separable', fontsize=10,)
    plt.text(0.05, -0.5, 'False Entangled', fontsize=10,)
    plt.text(0.77, 0.25, 'False Separable', fontsize=10,)
    plt.text(0.77, -0.5, 'True Entangled', fontsize=10,)
    plt.legend()
    #plt.grid(True)
    plt.show()

def print_EW():
    ent_state_type = "singlet_state"
    seed = 4
    res_qd = 4
    qn = 1
    lsvm = linear_svm(
                f"Plots/OneReservoir/data/data_{ent_state_type}/measurements_train_res_{res_qd}_qn_{qn}_seed_{seed}.npy",
                f"Plots/OneReservoir/data/data_{ent_state_type}/labels_train_res_{res_qd}_qn_{qn}_{seed}.npy",
                f"Plots/OneReservoir/data/data_{ent_state_type}/measurements_test_res_{res_qd}_qn_{qn}_{seed}.npy",
                f"Plots/OneReservoir/data/data_{ent_state_type}/labels_test_res_{res_qd}_qn_{qn}_{seed}.npy",
                class_weights={-1: 1, 1: 1000},
                C=1
            )
    lsvm.print_weights()

def plot_p ():
    ent_state_type = "singlet_state"
    seed = 4
    res_qd = 4
    qn = 1
    lsvm = linear_svm(
                f"Plots/OneReservoir/data/data_{ent_state_type}/measurements_train_res_{res_qd}_qn_{qn}_seed_{seed}.npy",
                f"Plots/OneReservoir/data/data_{ent_state_type}/labels_train_res_{res_qd}_qn_{qn}_{seed}.npy",
                f"Plots/OneReservoir/data/data_{ent_state_type}/measurements_test_res_{res_qd}_qn_{qn}_{seed}.npy",
                f"Plots/OneReservoir/data/data_{ent_state_type}/labels_test_res_{res_qd}_qn_{qn}_{seed}.npy",
                class_weights={-1: 1, 1: 1000},
                C=1
            )
    p_range, y_pred = lsvm.get_decision_values(False)
    plt.plot(p_range, y_pred, label=ent_state_type)
    plt.show()


#train_lsvm()
#train_rsvm()
train_rsvm_multiple_states()
#print_EW()