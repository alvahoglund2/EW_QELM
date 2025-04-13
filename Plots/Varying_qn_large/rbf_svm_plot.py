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
            #qn_values.append([i for i in range(0, res_qd*2+1)])
            qn_values.append([0,1,2,3,4])
        elif res_qd == 5:
            qn_values.append([0,1,2,3])
        elif res_qd == 6:
            qn_values.append([0,1])

    fig, axes = plt.subplots(1, len(res_qds), figsize=(15, 8))

    for i, res_qd in enumerate(res_qds):
        accuracies_ent_qd = []
        robustness_qd = []

        for j, ent_state_type in enumerate(ent_state_types):
            for seed in seeds:
                accuracies_ent_qn = []
                robustness_qn = []
                
                qns = qn_values[i]
                
                for qn in qns:
                    rsvm = rbf_svm(
                        f"Plots/varying_qn_large/data_{ent_state_type}/measurements_train_{ent_state_type}_res_{res_qd}_qn_{qn}_seed_{seed}.npy",
                        f"Plots/varying_qn_large/States/train_labels.npy",
                        f"Plots/varying_qn_large/data_{ent_state_type}/measurements_test_{ent_state_type}_res_{res_qd}_qn_{qn}_{seed}.npy",
                        f"Plots/varying_qn_large/States/test_labels.npy",
                        class_weights={-1: 1, 1: 10^5},
                        C=1,
                        gamma=1.5
                    )
                    accuracy, accuracy_sep, accuracy_ent, _ = rsvm.evaluate_model(evaluate_train=False)
                    dec_val, p_idx = rsvm.get_desicion_values_for_data(f"Plots/varying_qn_large/data_{ent_state_type}/measurements_werner_{ent_state_type}_res_{res_qd}_qn_{qn}_{seed}.npy",)
                    p_range = np.linspace(0, 1, len(dec_val))
                    p_tol = p_range[p_idx]

                    accuracies_ent_qn.append(accuracy_ent)
                    robustness_qn.append(p_tol)
                    print("fres_qd: {res_qd}, qn: {qn}, seed: {seed}, state: {ent_state_type}")
                    rsvm.print_accuracy()
                    print(f"res_qd: {res_qd}, qn: {qn}, seed: {seed}, state: {ent_state_type} accuracy_sep: {accuracy_sep} accuracy_ent: {accuracy_ent}, robustness: {p_tol}")
                    
                accuracies_ent_qd.append(accuracies_ent_qn)
                robustness_qd.append(robustness_qn)
            
        mean_rob = np.mean(robustness_qd, axis=0)
        var_rob = np.var(robustness_qd, axis=0)

        axes[i].plot(mean_rob, '-o', color=palette[i], markersize=8, label=f'res_qd {res_qd}')
        axes[i].errorbar(
            qn_values[i], mean_rob, yerr=np.sqrt(var_rob), fmt='-o', 
            label=f'res_qd {res_qd}', color=palette[i], markersize=8, 
            elinewidth=2, capsize=4, alpha=0.6
        )
        axes[i].set_ylim(0.35, 1.15)
        axes[i].set_xticks(qn_values[i])
        if res_qd == 1:
            axes[i].set_title(f'{res_qd} QD in Reservoir')
        else:
            axes[i].set_title(f'{res_qd} QDs in Reservoir')
        axes[i].set_xlabel('Particles in Reservoir', fontsize=12)
        axes[i].set_ylabel('Noise Tolerance', fontsize=12)
    plt.suptitle('Noise Tolerance for different Reservoir Settings')

    plt.tight_layout(pad=3.0)
    plt.show()



seeds = [2]
ent_state_types = ["singlet_state"]
res_qds = [1]

plot_all_avg(ent_state_types, seeds, res_qds)