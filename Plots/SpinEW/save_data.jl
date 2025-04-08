includet("../../src/basis.jl")
includet("../../src/hamiltonian.jl")
includet("../../src/time_evolution.jl")
includet("../../src/measurements.jl")
includet("../../src/main_system_state.jl")
includet("../../src/reservoir_state.jl")
includet("../../src/effective_measurements.jl")
includet("../../src/generate_data.jl")

function save_data_single_state_EW()
    d, d_main, dA_main, dB_main, d_res = total_basis(2, 0, conserved_qn = QuantumDots.fermionnumber)

    nbr_sep_states = 40000
    nbr_mix_sep_states = 300000
    nbr_ent_states = nbr_sep_states+ nbr_mix_sep_states
    nbr_werner_states = 10000

    nbr_states = [nbr_ent_states, nbr_sep_states, nbr_mix_sep_states, nbr_werner_states]
    
    ent_state_types = [singlet_state, triplet0_state, tripletn1_state, tripletp1_state]
    p_min = 1/2

    for ent_state_type in ent_state_types

        ent_measurements_train, sep_measurements_train, mix_sep_measurements_train, werner_measurements = get_spin_measurements(d_main, dA_main, dB_main, nbr_states, [ent_state_type], p_min, )
        ent_measurements_test, sep_measurements_test, mix_sep_measurements_test, werner_measurements = get_spin_measurements(d_main, dA_main, dB_main, nbr_states, [ent_state_type], p_min, )
        
        labels_train = vcat([-1 for i in 1:ent_measurements_train.size[1]], [1 for i in 1:sep_measurements_train.size[1]], [1 for i in 1:mix_sep_measurements_train.size[1]])
        measurements_train = vcat(hcat(ent_measurements_train), hcat(sep_measurements_train), hcat(mix_sep_measurements_train))

        
        labels_test = vcat([-1 for i in 1:ent_measurements_test.size[1]], [1 for i in 1:sep_measurements_test.size[1]], [1 for i in 1:mix_sep_measurements_test.size[1]])
        measurements_test = vcat(hcat(ent_measurements_test), hcat(sep_measurements_test), hcat(mix_sep_measurements_test))
        
        np.save("Plots/SpinEW/data/$(ent_state_type)_measurements_train.npy", measurements_train)
        np.save("Plots/SpinEW/data/$(ent_state_type)_labels_train.npy", labels_train)
        
        np.save("Plots/SpinEW/data/$(ent_state_type)_measurements_test.npy", measurements_test)
        np.save("Plots/SpinEW/data/$(ent_state_type)_labels_test.npy", labels_test)

        np.save("Plots/SpinEW/data/$(ent_state_type)_measurements_werner.npy", werner_measurements)
    end
end

function save_data_multiple_state_EW()
    d, d_main, dA_main, dB_main, d_res = total_basis(2, 0, conserved_qn = QuantumDots.fermionnumber)

    nbr_sep_states = 40000
    nbr_mix_sep_states = 300000
    nbr_ent_states = nbr_sep_states+ nbr_mix_sep_states
    nbr_werner_states = 10000

    nbr_states = [nbr_ent_states, nbr_sep_states, nbr_mix_sep_states, nbr_werner_states]
    
    ent_state_types = [singlet_state, triplet0_state, tripletn1_state, tripletp1_state]
    p_min = 1/2

        ent_measurements_train, sep_measurements_train, mix_sep_measurements_train, werner_measurements = get_spin_measurements(d_main, dA_main, dB_main, nbr_states, ent_state_types, p_min, )
        ent_measurements_test, sep_measurements_test, mix_sep_measurements_test, werner_measurements = get_spin_measurements(d_main, dA_main, dB_main, nbr_states, ent_state_types, p_min, )
        
        labels_train = vcat([-1 for i in 1:ent_measurements_train.size[1]], [1 for i in 1:sep_measurements_train.size[1]], [1 for i in 1:mix_sep_measurements_train.size[1]])
        measurements_train = vcat(hcat(ent_measurements_train), hcat(sep_measurements_train), hcat(mix_sep_measurements_train))

        
        labels_test = vcat([-1 for i in 1:ent_measurements_test.size[1]], [1 for i in 1:sep_measurements_test.size[1]], [1 for i in 1:mix_sep_measurements_test.size[1]])
        measurements_test = vcat(hcat(ent_measurements_test), hcat(sep_measurements_test), hcat(mix_sep_measurements_test))
        
        np.save("Plots/SpinEW/data/multiple_state_measurements_train.npy", measurements_train)
        np.save("Plots/SpinEW/data/multiple_state_labels_train.npy", labels_train)
        
        np.save("Plots/SpinEW/data/multiple_state_measurements_test.npy", measurements_test)
        np.save("Plots/SpinEW/data/multiple_state_labels_test.npy", labels_test)

        np.save("Plots/SpinEW/data/multiple_state_measurements_werner.npy", werner_measurements)
    
end

save_data_single_state_EW()
#save_data_multiple_state_EW()

