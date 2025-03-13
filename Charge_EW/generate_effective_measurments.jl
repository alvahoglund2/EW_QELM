function charge_measurments(hamiltonian :: AbstractMatrix, ρ_R :: AbstractMatrix , t_eval, 
    d :: FermionBasis, d_main :: FermionBasis, dA_main :: FermionBasis, dB_main :: FermionBasis, d_res :: FermionBasis, 
    nbr_ent_states :: Integer, nbr_sep_states :: Integer, ent_state_types, p_min, fock_nbrs)
    N_sep = nbr_sep_states^(1/4) |> round |> Int
    ent_states = vcat([werner_state_list(d_main, nbr_ent_states, type, p_min) for type in ent_state_types]...)
    sep_states = uniform_separable_two_qubit_states(d_main, dA_main, dB_main, N_sep)
    eff_measurments = get_effective_measurments(hamiltonian, ρ_R, t_eval, d, d_main, d_res, fock_nbrs)

    ent_states_measurments = measure_states(ent_states, eff_measurments,d_main)
    sep_states_measurments = measure_states(sep_states, eff_measurments, d_main)

    return sep_states_measurments, ent_states_measurments
end

function get_effective_measurments(hamiltonian :: AbstractMatrix, ρ_R :: AbstractMatrix , t_eval, 
    d :: FermionBasis, d_main :: FermionBasis, d_res :: FermionBasis, fock_nbrs)
    
    dot_labels = get_spatial_labels(d)
    nbr_dots = length(dot_labels)
    
    ops = vcat([nbr_op(i, d) for i in 1:nbr_dots], [nbr2_op(i, d) for i in 1:nbr_dots])
    eff_measurments = get_eff_measurments(ops, ρ_R, hamiltonian, t_eval, d, d_main, d_res, fock_nbrs)
    return eff_measurments
end

function measure_states(state_list, eff_measurments, d_main)
    n_states = length(state_list)
    n_measurements = length(eff_measurments)
    result = zeros(n_states, n_measurements)
    for (i, state) in enumerate(state_list)
        for (j, eff_measurment) in enumerate(eff_measurments)
            trunc_state = state[get_qubit_idx(d_main),get_qubit_idx(d_main)]
            result[i, j] = expectation_value(trunc_state, eff_measurment)
        end
    end    
    return result
end