includet("../src/basis.jl")
includet("../src/hamiltonian.jl")
includet("../src/time_evolution.jl")
includet("../src/measurments.jl")
includet("../src/main_system_state.jl")
includet("../src/reservoir_state.jl")
includet("../src/effective_measurments.jl")
includet("generate_spin_measurments.jl")

ent_state_types = [singlet_state, triplet0_state, tripletn1_state, tripletp1_state]
noise_level_min = 0.5
nbr_ent_states = 10000
nbr_sep_states = 10000

sep_measurments_train, ent_measurments_train = generate_spin_measurments(ent_state_types, noise_level_min, nbr_ent_states, nbr_sep_states)
sep_measurments_test, ent_measurments_test = generate_spin_measurments(ent_state_types, noise_level_min, nbr_ent_states, nbr_sep_states)
## -------------- Save ----------------

ent_measurments_matrix_train = hcat(ent_measurments_train...)
sep_measurments_matrix_train = hcat(sep_measurments_train...)

ent_measurments_matrix_test = hcat(ent_measurments_test...)
sep_measurments_matrix_test = hcat(sep_measurments_test...)

labels_train = vcat([-1 for i in 1:ent_measurments_matrix_train.size[1]], [1 for i in 1:sep_measurments_matrix_train.size[1]])
measurments_train = vcat(hcat(ent_measurments_matrix_train), hcat(sep_measurments_matrix_train))

labels_test = vcat([-1 for i in 1:ent_measurments_matrix_test.size[1]], [1 for i in 1:sep_measurments_matrix_test.size[1]])
measurments_test = vcat(hcat(ent_measurments_matrix_test), hcat(sep_measurments_matrix_test))


np.save("Spin_EW/data/measurments_train.npy", measurments_train)
np.save("Spin_EW/data/labels_train.npy", labels_train)

np.save("Spin_EW/data/measurments_test.npy", measurments_test)
np.save("Spin_EW/data/labels_test.npy", labels_test)