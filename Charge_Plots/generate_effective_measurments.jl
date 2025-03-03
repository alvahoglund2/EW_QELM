function charge_measurments(hamiltonian :: AbstractMatrix, ρ_R :: AbstractMatrix , t_eval, 
    d :: FermionBasis, d_main :: FermionBasis, dA_main :: FermionBasis, dB_main :: FermionBasis, d_res :: FermionBasis, nbr_ent_states :: Integer, nbr_sep_states :: Integer, ent_state_types, p_min)

    ent_states = vcat([werner_state_list(d:main, nbr_ent_states, type, p_min) for type in ent_state_types]...)
    sep_states = [random_separable_state(d_main, dA_main, dB_main) for i in 1:nbr_sep_states]
    eff_measurments = get_effective_measurments(hamiltonian, ρ_R, t_eval, d, d_main, d_res)

    ent_states_measurments = measure_states(ent_states, eff_measurments)
    sep_states_measurments = measure_states(sep_states, eff_measurments)

    return sep_states_measurments, ent_states_measurments
end

function get_effective_measurments(hamiltonian :: AbstractMatrix, ρ_R :: AbstractMatrix , t_eval, d :: FermionBasis, d_main :: FermionBasis, d_res :: FermionBasis)
    
    dot_labels = get_spatial_labels(d)
    nbr_dots = length(dot_labels)
    
    ops = vcat([nbr_op(i, d) for i in 1:nbr_dots], [nbr2_op(i, d) for i in 1:nbr_dots])
    eff_measurments = [get_eff_measurment(op, ρ_R, hamiltonian, t_eval, d, d_main, d_res) for op in ops]
    return eff_measurments
end

function measure_states(state_list, eff_measurments)
    n_states = length(state_list)
    n_measurements = length(eff_measurments)
    result = zeros(n_states, n_measurements)
    for (i, state) in enumerate(state_list)
        for (j, eff_measurment) in enumerate(eff_measurments)
            trunc_state = state[get_qubit_idx(),get_qubit_idx()]
            result[i, j] = expectation_value(trunc_state, eff_measurment)
        end
    end    
    return result
end