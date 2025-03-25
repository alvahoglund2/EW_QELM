includet("../src/basis.jl")
includet("../src/hamiltonian.jl")
includet("../src/time_evolution.jl")
includet("../src/measurements.jl")
includet("../src/main_system_state.jl")
includet("../src/reservoir_state.jl")
includet("../src/effective_measurements.jl")
includet("../src/generate_data.jl")
## -------------- Define System ----------------

function define_system_parameters()
    ent_state_types = [triplet0_state]
    sep_state_types = [random_separable_state]
    state_types = [ent_state_types, sep_state_types]

    noise_level_min = 0.8

    t_eval = 1.0

    nbr_sep_states = 10000
    nbr_mix_sep_states = 100000
    nbr_ent_states = nbr_sep_states+ nbr_mix_sep_states

    nbr_states = [nbr_ent_states, nbr_sep_states, nbr_mix_sep_states]

    sys_qd = 2
    res_qd = 2

    conserved_qn = QuantumDots.fermionnumber
    d_tot = total_basis(sys_qd, res_qd, conserved_qn = conserved_qn)
    d, d_main, dA_main, dB_main, d_res = d_tot
    hamiltonian_type = Hdot_so_b

    hamiltonian = random_hamiltonian(d, hamiltonian_type, seed=4)

    qn = 2
    focknbrs = 2+qn

    ρ_R = res_ground_state(hamiltonian, d, d_res, qn)

    return hamiltonian, ρ_R, t_eval, d_tot, nbr_states, state_types, noise_level_min, focknbrs
end

function save_data()
    ent_measurements_train, sep_measurements_train, mix_sep_measurements_train = get_charge_measurements(define_system_parameters()...)
    ent_measurements_test, sep_measurements_test, mix_sep_measurements_test = get_charge_measurements(define_system_parameters()...)
    
    labels_train = vcat([-1 for i in 1:ent_measurements_train.size[1]], [1 for i in 1:sep_measurements_train.size[1]], [1 for i in 1:mix_sep_measurements_test.size[1]])
    measurements_train = vcat(hcat(ent_measurements_train), hcat(sep_measurements_train), hcat(mix_sep_measurements_train))
    
    labels_test = vcat([-1 for i in 1:ent_measurements_test.size[1]], [1 for i in 1:sep_measurements_test.size[1]], [1 for i in 1:mix_sep_measurements_test.size[1]])
    measurements_test = vcat(hcat(ent_measurements_test), hcat(sep_measurements_test), hcat(mix_sep_measurements_test))
    
    
    np.save("Charge_EW/data/measurements_train.npy", measurements_train)
    np.save("Charge_EW/data/labels_train.npy", labels_train)
    
    np.save("Charge_EW/data/measurements_test.npy", measurements_test)
    np.save("Charge_EW/data/labels_test.npy", labels_test)
end

save_data()