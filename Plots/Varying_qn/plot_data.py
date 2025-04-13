import sys
import numpy as np
import matplotlib.pyplot as plt

sys.path.append('SVM')
from linear_svm import linear_svm
from rbf_svm import rbf_svm

palette = [
    (90/255, 200/255, 250/255), # Turquoise
    (100/255, 143/255, 255/255),  # Blue 
    (120/255, 94/255, 240/255),  # Purple
    (220/255, 38/255, 127/255),  # Pink
    (254/255, 97/255, 0/255),    # Red-Orange
    (255/255, 176/255, 0/255),   # Orange
]

def plot_seeds(ent_state_type, seeds, res_qd):
    qn_values = None
    if res_qd <= 4:
        qn_values = [i for i in range(res_qd*2+1)]
    else:
        qn_values = [0, 1, 2]
    fig, axes = plt.subplots( figsize=(10, 8))
    
    accuracies_seed = []
    robustness_seed = []
        
    for seed in seeds:
        accuracies_ent_qn = []
        robustness_qn = []        
        for qn in qn_values:
            lsvm = linear_svm(
                f"Plots/Varying_qn/data_{ent_state_type}/measurements_train_{ent_state_type}_res_{res_qd}_qn_{qn}_seed_{seed}.npy",
                f"Plots/Varying_qn/States/train_labels.npy",
                f"Plots/Varying_qn/data_{ent_state_type}/measurements_test_{ent_state_type}_res_{res_qd}_qn_{qn}_{seed}.npy",
                f"Plots/Varying_qn/States/test_labels.npy",
                class_weights={-1: 1, 1: 5000},
                C=1
            )
            accuracy, accuracy_sep, accuracy_ent, _ = lsvm.evaluate_model(evaluate_train=False)

            dec_val, p_idx = lsvm.get_desicion_values_for_data(f"Plots/Varying_qn/data_{ent_state_type}/measurements_werner_{ent_state_type}_res_{res_qd}_qn_{qn}_{seed}.npy",)
            p_range = np.linspace(0, 1, len(dec_val))
            p_tol = p_range[p_idx]
            accuracies_ent_qn.append(accuracy_ent)
            robustness_qn.append(p_tol)
            print(f"res_qd: {res_qd}, qn: {qn}, seed: {seed}, accuracy_sep: {accuracy_sep} accuracy_ent: {accuracy_ent}, robustness: {p_tol}")
            
        accuracies_seed.append(accuracies_ent_qn)
        robustness_seed.append(robustness_qn)

        axes.plot(qn_values, robustness_qn, label = f'seed: {seed}', marker='o')
        axes.set_xticks(qn_values)
        axes.set_title(f'Noise tolerace for {res_qd} dot reservoir')
        axes.set_xlabel('Quantum Number (qn)')
        axes.set_ylabel('p_tol')
        axes.grid(True)
        axes.legend()
    
    plt.tight_layout()
    plt.show()

def plot_ent_seed(ent_state_type, seeds, res_qds):
    qn_values = []
    for i, res_qd in enumerate(res_qds):
        if res_qd <= 4:
            qn_values.append([i for i in range(res_qd*2+1)])
        else:
            qn_values.append([0, 1, 2])
    fig, axes = plt.subplots(1, len(res_qds), figsize=(15, 10))
    
    for i, res_qd in enumerate(res_qds):
        accuracies_ent_qd = []
        robustness_qd = []
        
        for seed in seeds:
            accuracies_ent_qn = []
            robustness_qn = []
            
            qns = qn_values[i]
            
            for qn in qns:
                lsvm = linear_svm(
                    f"Plots/Varying_qn/data_{ent_state_type}/measurements_train_res_{res_qd}_qn_{qn}_seed_{seed}.npy",
                    f"Plots/Varying_qn/data_{ent_state_type}/labels_train_res_{res_qd}_qn_{qn}_{seed}.npy",
                    f"Plots/Varying_qn/data_{ent_state_type}/measurements_test_res_{res_qd}_qn_{qn}_{seed}.npy",
                    f"Plots/Varying_qn/data_{ent_state_type}/labels_test_res_{res_qd}_qn_{qn}_{seed}.npy",
                    class_weights={-1: 1, 1: 5000},
                    C=1
                )
                accuracy, accuracy_sep, accuracy_ent, _ = lsvm.evaluate_model(evaluate_train=False)
                robustness = lsvm.get_robustness(evaluate_train=False, p_min=1/3)

                accuracies_ent_qn.append(accuracy_ent)
                robustness_qn.append(robustness)
                print(f"res_qd: {res_qd}, qn: {qn}, seed: {seed}, accuracy_sep: {accuracy_sep} accuracy_ent: {accuracy_ent}, robustness: {robustness}")
                
            accuracies_ent_qd.append(accuracies_ent_qn)
            robustness_qd.append(robustness_qn)
        
        mean_acc = np.mean(accuracies_ent_qd, axis=0)
        var_acc = np.var(accuracies_ent_qd, axis=0)
        mean_rob = np.mean(robustness_qd, axis=0)
        var_rob = np.var(robustness_qd, axis=0)

        axes[i].errorbar(qn_values[i], mean_rob, yerr=np.sqrt(var_rob), fmt='-o', label=f'res_qd {res_qd}')
        axes[i].set_title(f'Robustness for res_qd {res_qd}')
        axes[i].set_xlabel('Quantum Number (qn)')
        axes[i].set_ylabel('p_tol')
        axes[i].grid(True)
    
    plt.tight_layout()
    plt.show()

def plot_all(ent_state_types, seeds):
    res_qds = [1, 2, 3, 4, 5]
    qn_values = [[0, 1, 2], [0, 1, 2, 3, 4], [0, 1, 2, 3, 4, 5, 6], [0, 1, 2, 3, 4, 5, 6, 7, 8], [0, 1, 2, 3]]
    
    fig, axes = plt.subplots(len(ent_state_types), len(res_qds), figsize=(15, 10))

    for j, ent_state_type in enumerate(ent_state_types):

        for i, res_qd in enumerate(res_qds):
            accuracies_ent_qd = []
            robustness_qd = []
            
            for seed in seeds:
                accuracies_ent_qn = []
                robustness_qn = []
                
                qns = qn_values[i]
                
                for qn in qns:
                    lsvm = linear_svm(
                        f"Plots/Varying_qn/data_{ent_state_type}/measurements_train_res_{res_qd}_qn_{qn}_seed_{seed}.npy",
                        f"Plots/Varying_qn/data_{ent_state_type}/labels_train_res_{res_qd}_qn_{qn}_{seed}.npy",
                        f"Plots/Varying_qn/data_{ent_state_type}/measurements_test_res_{res_qd}_qn_{qn}_{seed}.npy",
                        f"Plots/Varying_qn/data_{ent_state_type}/labels_test_res_{res_qd}_qn_{qn}_{seed}.npy",
                        class_weights={-1: 1, 1: 5000},
                        C=1
                    )
                    accuracy, accuracy_sep, accuracy_ent, _ = lsvm.evaluate_model(evaluate_train=False)
                    robustness = lsvm.get_robustness(evaluate_train=False, p_min=0.5)

                    accuracies_ent_qn.append(accuracy_ent)
                    robustness_qn.append(robustness)
                    print(f"res_qd: {res_qd}, qn: {qn}, seed: {seed}, state: {ent_state_type} accuracy_sep: {accuracy_sep} accuracy_ent: {accuracy_ent}, robustness: {robustness}")
                    
                accuracies_ent_qd.append(accuracies_ent_qn)
                robustness_qd.append(robustness_qn)
            
            mean_acc = np.mean(accuracies_ent_qd, axis=0)
            var_acc = np.var(accuracies_ent_qd, axis=0)
            mean_rob = np.mean(robustness_qd, axis=0)
            var_rob = np.var(robustness_qd, axis=0)

            axes[j, i].errorbar(qn_values[i], mean_rob, yerr=np.sqrt(var_rob), fmt='-o', label=f'res_qd {res_qd}')
            axes[j, i].set_title(f'Robustness for res_qd {res_qd}')
            axes[j, i].set_xlabel('Quantum Number (qn)')
            axes[j, i].set_ylabel('p_tol')
            axes[j, i].grid(True)
        
    plt.tight_layout()
    plt.show()

def plot_all_avg(ent_state_types, seeds):
    res_qds = [1, 2, 3, 4, 5, 6]
    qn_values = [[0, 1, 2], [0, 1, 2, 3, 4], [0, 1, 2, 3, 4, 5, 6], [0, 1, 2, 3, 4, 5, 6, 7, 8], [0, 1, 2, 3], [0,1]]
    
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
                    lsvm = linear_svm(
                        f"Plots/Varying_qn/data_{ent_state_type}/measurements_train_{ent_state_type}_res_{res_qd}_qn_{qn}_seed_{seed}.npy",
                        f"Plots/Varying_qn/States/train_labels.npy",
                        f"Plots/Varying_qn/data_{ent_state_type}/measurements_test_{ent_state_type}_res_{res_qd}_qn_{qn}_{seed}.npy",
                        f"Plots/Varying_qn/States/test_labels.npy",
                        class_weights={-1: 1, 1: 5000},
                        C=1
                    )
                    accuracy, accuracy_sep, accuracy_ent, _ = lsvm.evaluate_model(evaluate_train=False)
                    
                    dec_val, p_idx = lsvm.get_desicion_values_for_data(f"Plots/Varying_qn/data_{ent_state_type}/measurements_werner_{ent_state_type}_res_{res_qd}_qn_{qn}_{seed}.npy",)
                    p_range = np.linspace(0, 1, len(dec_val))
                    p_tol = p_range[p_idx]
                    accuracies_ent_qn.append(accuracy_ent)
                    robustness_qn.append(p_tol)
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

seeds = [1, 2, 4, 5, 6, 7]
ent_state_types = ["singlet_state", "triplet0_state", "tripletn1_state", "Tripletp1_state"]
res_qds = [1,2,3,4,5,6]
#ent_state_types = ["singlet_state"]
#plot_all(ent_state_types, seeds)
#plot_all_avg(ent_state_types, seeds)
#plot_ent_seed(ent_state_types[0], seeds, res_qds)
#plot_seeds(ent_state_types[0], seeds, 2)

plot_all_avg(ent_state_types, seeds)