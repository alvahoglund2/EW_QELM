function generate_spin_measurements(ent_state_types, noise_level_min, nbr_ent_states, nbr_sep_states)
    main_sys_qd = 2
    res_qd = 0
    d, d_main, dA_main, dB_main, d_res = total_basis(main_sys_qd, res_qd)

    sep_measurements = measure_sep_states(d_main, dA_main, dB_main, nbr_sep_states)
    ent_measurements = measure_ent_states(d_main, dA_main, dB_main, nbr_ent_states, ent_state_types, noise_level_min)

    return sep_measurements, ent_measurements
end

function measurements(state_list, d_main, dA_main, dB_main)
    sx_list = [expectation_value(state, pauli_string(d_main, dA_main, dB_main, sx, sx)) for state in state_list]
    sy_list = [expectation_value(state, pauli_string(d_main, dA_main, dB_main, sy, sy)) for state in state_list]
    sz_list = [expectation_value(state, pauli_string(d_main, dA_main, dB_main, sz, sz)) for state in state_list]
    return sx_list, sy_list, sz_list
end

function measure_sep_states(d_main, dA_main, dB_main, nbr_sep_states)
    rand_sep_states = [random_separable_state(d_main, dA_main, dB_main) for i in 1:nbr_sep_states]
    sx_list_sep, sy_list_sep, sz_list_sep = measurements(rand_sep_states, d_main, dA_main, dB_main)
    return sx_list_sep, sy_list_sep, sz_list_sep
end

function measure_ent_states(d_main, dA_main, dB_main, nbr_ent_states, types, noise_level_min)
    ent_states = vcat([werner_state_list(d_main, nbr_ent_states, type, noise_level_min) for type in types]...)
    sx_list_ent, sy_list_ent, sz_list_ent = measurements(ent_states, d_main, dA_main, dB_main)
    return sx_list_ent, sy_list_ent, sz_list_ent
end

function a_test()
    print("a")
end