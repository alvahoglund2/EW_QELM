includet("../src/basis.jl")
includet("../src/hamiltonian.jl")
includet("../src/time_evolution.jl")
includet("../src/measurements.jl")
includet("../src/main_system_state.jl")
includet("../src/reservoir_state.jl")
includet("../src/effective_measurements.jl")
includet("generate_spin_measurements.jl")

ent_state_types = [singlet_state, triplet0_state, tripletn1_state, tripletp1_state]
noise_level_min = 0.5
nbr_ent_states = 10000
nbr_sep_states = 10000

sep_measurements_train, ent_measurements_train = generate_spin_measurements(ent_state_types, noise_level_min, nbr_ent_states, nbr_sep_states)
sep_measurements_test, ent_measurements_test = generate_spin_measurements(ent_state_types, noise_level_min, nbr_ent_states, nbr_sep_states)
## -------------- Save ----------------

ent_measurements_matrix_train = hcat(ent_measurements_train...)
sep_measurements_matrix_train = hcat(sep_measurements_train...)

ent_measurements_matrix_test = hcat(ent_measurements_test...)
sep_measurements_matrix_test = hcat(sep_measurements_test...)

labels_train = vcat([-1 for i in 1:ent_measurements_matrix_train.size[1]], [1 for i in 1:sep_measurements_matrix_train.size[1]])
measurements_train = vcat(hcat(ent_measurements_matrix_train), hcat(sep_measurements_matrix_train))

labels_test = vcat([-1 for i in 1:ent_measurements_matrix_test.size[1]], [1 for i in 1:sep_measurements_matrix_test.size[1]])
measurements_test = vcat(hcat(ent_measurements_matrix_test), hcat(sep_measurements_matrix_test))


np.save("Spin_EW/data/measurements_train.npy", measurements_train)
np.save("Spin_EW/data/labels_train.npy", labels_train)

np.save("Spin_EW/data/measurements_test.npy", measurements_test)
np.save("Spin_EW/data/labels_test.npy", labels_test)