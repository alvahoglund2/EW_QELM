includet("../src/basis.jl")
includet("../src/hamiltonian.jl")
includet("../src/time_evolution.jl")
includet("../src/measurments.jl")
includet("../src/main_system_state.jl")
includet("../src/reservoir_state.jl")
includet("../src/effective_measurments.jl")
includet("generate_effective_measurments.jl")

## -------------- Define System ----------------
ent_state_types = [singlet_state]
noise_level_min = 0.8

t_eval = 1.0

nbr_ent_states = 100
nbr_sep_states = 10000

sys_qd = 2
res_qd = 2

conserved_qn = IndexConservation(:↑) * IndexConservation(:↓)
d, d_main, dA_main, dB_main, d_res = total_basis(sys_qd, res_qd, conserved_qn = conserved_qn)
hamiltonian_type = Hdot_b

hamiltonian = random_hamiltonian(d, hamiltonian_type, seed=4)

qn = (1,1)
focknbrs = [(3,1), (2,2), (1,3)]

ρ_R = res_ground_state(hamiltonian, d, d_res, qn)

## ------------ Save data ----------------
sep_measurments_train, ent_measurments_train = charge_measurments(hamiltonian, ρ_R, t_eval, d, d_main, dA_main, dB_main, d_res, nbr_ent_states, nbr_sep_states, ent_state_types, noise_level_min, focknbrs)
sep_measurments_test, ent_measurments_test = charge_measurments(hamiltonian, ρ_R, t_eval, d, d_main, dA_main, dB_main, d_res, nbr_ent_states, nbr_sep_states, ent_state_types, noise_level_min, focknbrs) 

labels_train = vcat([-1 for i in 1:ent_measurments_train.size[1]], [1 for i in 1:sep_measurments_train.size[1]])
measurments_train = vcat(hcat(ent_measurments_train), hcat(sep_measurments_train))

labels_test = vcat([-1 for i in 1:ent_measurments_test.size[1]], [1 for i in 1:sep_measurments_test.size[1]])
measurments_test = vcat(hcat(ent_measurments_test), hcat(sep_measurments_test))


np.save("Charge_EW/data/measurments_train.npy", measurments_train)
np.save("Charge_EW/data/labels_train.npy", labels_train)

np.save("Charge_EW/data/measurments_test.npy", measurments_test)
np.save("Charge_EW/data/labels_test.npy", labels_test)