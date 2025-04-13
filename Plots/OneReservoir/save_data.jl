includet("../../src/basis.jl")
includet("../../src/hamiltonian.jl")
includet("../../src/time_evolution.jl")
includet("../../src/measurements.jl")
includet("../../src/main_system_state.jl")
includet("../../src/reservoir_state.jl")
includet("../../src/effective_measurements.jl")
includet("../../src/generate_data.jl")
## -------------- Define System ----------------

function define_system_parameters(res_qd, qn, seed, ent_state_types, noise_level_min)
    ent_state_types = ent_state_types

    noise_level_min = noise_level_min
    t_eval = 1.0

    nbr_sep_states = 40000*4
    nbr_mix_sep_states = 300000*4
    nbr_ent_states = nbr_sep_states+ nbr_mix_sep_states

    nbr_states = [nbr_ent_states, nbr_sep_states, nbr_mix_sep_states]

    sys_qd = 2
    res_qd = res_qd

    conserved_qn = QuantumDots.fermionnumber
    d_tot = total_basis(sys_qd, res_qd, conserved_qn = conserved_qn)
    d, d_main, dA_main, dB_main, d_res = d_tot
    hamiltonian_type = Hdot_so_b

    hamiltonian = random_hamiltonian(d, hamiltonian_type, seed=seed)

    qn = qn
    focknbrs = 2+qn

    ρ_R = res_ground_state(hamiltonian, d, d_res, qn)

    return hamiltonian, ρ_R, t_eval, d_tot, nbr_states, ent_state_types, noise_level_min, focknbrs
end

function save_data_single_state_EW()
    seed = 2
    res_qd = 1
    qn = 0
    noise_level_min = 1/2
    ent_state_types = [singlet_state, triplet0_state, tripletn1_state, tripletp1_state]
    for ent_state_type in ent_state_types
        ent_measurements_train, sep_measurements_train, mix_sep_measurements_train = get_charge_measurements(define_system_parameters(res_qd, qn, seed, [ent_state_type], noise_level_min)...)
        ent_measurements_test, sep_measurements_test, mix_sep_measurements_test = get_charge_measurements(define_system_parameters(res_qd, qn, seed, [ent_state_type], noise_level_min)...)
        
        labels_train = vcat([-1 for i in 1:ent_measurements_train.size[1]], [1 for i in 1:sep_measurements_train.size[1]], [1 for i in 1:mix_sep_measurements_test.size[1]])
        measurements_train = vcat(hcat(ent_measurements_train), hcat(sep_measurements_train), hcat(mix_sep_measurements_train))
        
        labels_test = vcat([-1 for i in 1:ent_measurements_test.size[1]], [1 for i in 1:sep_measurements_test.size[1]], [1 for i in 1:mix_sep_measurements_test.size[1]])
        measurements_test = vcat(hcat(ent_measurements_test), hcat(sep_measurements_test), hcat(mix_sep_measurements_test))
        println("ent_state_type: ", ent_state_type)
        np.save("Plots/OneReservoir/data_large_smallres/data_$(ent_state_type)/measurements_train_res_$(res_qd)_qn_$(qn)_seed_$(seed).npy", measurements_train)
        np.save("Plots/OneReservoir/data_large_smallres/data_$(ent_state_type)/labels_train_res_$(res_qd)_qn_$(qn)_$(seed).npy", labels_train)
        np.save("Plots/OneReservoir/data_large_smallres/data_$(ent_state_type)/measurements_test_res_$(res_qd)_qn_$(qn)_$(seed).npy", measurements_test)
        np.save("Plots/OneReservoir/data_large_smallres/data_$(ent_state_type)/labels_test_res_$(res_qd)_qn_$(qn)_$(seed).npy", labels_test)
    end
end

function save_data_multiple_state_EW()
    seed = 4
    res_qd = 4
    qn = 1
    noise_level_min = 1/2
    ent_state_types = [singlet_state, triplet0_state, tripletn1_state, tripletp1_state]

    ent_measurements_train, sep_measurements_train, mix_sep_measurements_train = get_charge_measurements(define_system_parameters(res_qd, qn, seed, ent_state_types, noise_level_min)...)
    ent_measurements_test, sep_measurements_test, mix_sep_measurements_test = get_charge_measurements(define_system_parameters(res_qd, qn, seed, ent_state_types, noise_level_min)...)
    
    labels_train = vcat([-1 for i in 1:ent_measurements_train.size[1]], [1 for i in 1:sep_measurements_train.size[1]], [1 for i in 1:mix_sep_measurements_test.size[1]])
    measurements_train = vcat(hcat(ent_measurements_train), hcat(sep_measurements_train), hcat(mix_sep_measurements_train))
    
    labels_test = vcat([-1 for i in 1:ent_measurements_test.size[1]], [1 for i in 1:sep_measurements_test.size[1]], [1 for i in 1:mix_sep_measurements_test.size[1]])
    measurements_test = vcat(hcat(ent_measurements_test), hcat(sep_measurements_test), hcat(mix_sep_measurements_test))
    np.save("Plots/OneReservoir/data_large_smallres/data_all_states/measurements_train_res_$(res_qd)_qn_$(qn)_seed_$(seed).npy", measurements_train)
    np.save("Plots/OneReservoir/data_large_smallres/data_all_states/labels_train_res_$(res_qd)_qn_$(qn)_$(seed).npy", labels_train)
    np.save("Plots/OneReservoir/data_large_smallres/data_all_states/measurements_test_res_$(res_qd)_qn_$(qn)_$(seed).npy", measurements_test)
    np.save("Plots/OneReservoir/data_large_smallres/data_all_states/labels_test_res_$(res_qd)_qn_$(qn)_$(seed).npy", labels_test)
end

function save_test_data()
    seed = 4
    res_qd = 4
    qn = 1
    noise_level_min = 0.0
    ent_state_types = [singlet_state, triplet0_state, tripletn1_state, tripletp1_state]
    for ent_state_type in ent_state_types
        ent_measurements_train, sep_measurements_train, mix_sep_measurements_train = get_charge_measurements(define_system_parameters(res_qd, qn, seed, ent_state_type, noise_level_min)...)
        np.save("Plots/OneReservoir/data_large_smallres/data_werner_$(ent_state_type)/measurements_$(res_qd)_qn_$(qn)_seed_$(seed).npy", ent_measurements_train)
    end
end

save_data_single_state_EW()
save_data_multiple_state_EW()