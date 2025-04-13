import sys
import numpy as np
import matplotlib.pyplot as plt

sys.path.append('SVM')
from linear_svm import linear_svm
from rbf_svm import rbf_svm
from poly_svm import poly_svm

palette = [
    (100/255, 143/255, 255/255),  # Blue 
    (120/255, 94/255, 240/255),  # Purple
    (220/255, 38/255, 127/255),  # Pink
    (254/255, 97/255, 0/255),    # Red-Orange
    (255/255, 176/255, 0/255),   # Orange
]

def plot_all_avg(ent_state_types, seeds, res_qds):

    qn_values = []
    for res_qd in res_qds:
        if res_qd <= 4:
            qn_values.append([i for i in range(0, res_qd*2+1)])
            #qn_values.append([1,2])
        elif res_qd == 5:
            qn_values.append([0,1,2,3])
        elif res_qd == 6:
            qn_values.append([0,1])

    for i, res_qd in enumerate(res_qds):
        accuracies_ent_qd = []
        robustness_qd = []

        for j, ent_state_type in enumerate(ent_state_types):
            for seed in seeds:
                accuracies_ent_qn = []
                robustness_qn = []
                
                qns = qn_values[i]
                
                for qn in qns:
                    print("====================")
                    print(f"res_qd: {res_qd}, qn: {qn}, seed: {seed}")
                    rsvm = rbf_svm(
                        f"Plots/Varying_qn/data_{ent_state_type}/measurements_train_{ent_state_type}_res_{res_qd}_qn_{qn}_seed_{seed}.npy",
                        f"Plots/Varying_qn/States/train_labels.npy",
                        f"Plots/Varying_qn/data_{ent_state_type}/measurements_test_{ent_state_type}_res_{res_qd}_qn_{qn}_{seed}.npy",
                        f"Plots/Varying_qn/States/test_labels.npy",
                        class_weights={-1: 1, 1: 10^4},
                        C=1,
                        gamma=0.5
                    )
                    accuracy, accuracy_sep, accuracy_ent, _ = rsvm.evaluate_model(evaluate_train=False)
                    dec_val, p_idx = rsvm.get_desicion_values_for_data(f"Plots/Varying_qn/data_{ent_state_type}/measurements_werner_{ent_state_type}_res_{res_qd}_qn_{qn}_{seed}.npy",)
                    p_range = np.linspace(0, 1, len(dec_val))
                    p_tol = p_range[p_idx]

                    print(f"RBF:")
                    rsvm.print_accuracy()
                    print(f"Noise tol: {p_tol}")

                    print("Linear")
                    lsvm = linear_svm(
                        f"Plots/Varying_qn/data_{ent_state_type}/measurements_train_{ent_state_type}_res_{res_qd}_qn_{qn}_seed_{seed}.npy",
                        f"Plots/Varying_qn/States/train_labels.npy",
                        f"Plots/Varying_qn/data_{ent_state_type}/measurements_test_{ent_state_type}_res_{res_qd}_qn_{qn}_{seed}.npy",
                        f"Plots/Varying_qn/States/test_labels.npy",
                        class_weights={-1: 1, 1: 5000},
                        C=1
                    )
                    accuracy, accuracy_sep, accuracy_ent, _ = rsvm.evaluate_model(evaluate_train=False)
                    dec_val, p_idx = lsvm.get_desicion_values_for_data(f"Plots/Varying_qn/data_{ent_state_type}/measurements_werner_{ent_state_type}_res_{res_qd}_qn_{qn}_{seed}.npy",)
                    p_range = np.linspace(0, 1, len(dec_val))
                    p_tol = p_range[p_idx]
                    lsvm.print_accuracy()
                    print(f"Noise tol: {p_tol}")


                accuracies_ent_qd.append(accuracies_ent_qn)
                robustness_qd.append(robustness_qn)
            
        mean_rob = np.mean(robustness_qd, axis=0)
        var_rob = np.var(robustness_qd, axis=0)

seeds = [2]
ent_state_types = ["singlet_state"]
res_qds = [3]

plot_all_avg(ent_state_types, seeds, res_qds)