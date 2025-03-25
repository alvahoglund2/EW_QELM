includet("../src/basis.jl")
includet("../src/hamiltonian.jl")
includet("../src/time_evolution.jl")
includet("../src/measurments.jl")
includet("../src/main_system_state.jl")
includet("../src/reservoir_state.jl")
includet("../src/effective_measurments.jl")

## -------------- Define System ----------------
t_eval = 1.0
nbr_states = 10

sys_qd = 2
res_qd = 3

d, d_main, dA_main, dB_main, d_res = total_basis(sys_qd, res_qd)
hamiltonian_type = Hdot_so_b
hamiltonian = random_hamiltonian(d, hamiltonian_type)

qn = 3
ρ_R = res_ground_state(hamiltonian, d, d_res, qn)

## ------------ Define states ----------------

states = [random_separable_state(d_main, dA_main, dB_main) for i in 1:nbr_states]
states_tot = [wedge([state, ρ_R], [d_main, d_res], d) for state in states]

## ------------ Define Measurments ----------------
dot_labels = get_spatial_labels(d)
nbr_dots = length(dot_labels)
ops = vcat([nbr_op(i, d) for i in 1:nbr_dots], [nbr2_op(i, d) for i in 1:nbr_dots])

function measure_states(state_list, measurment_list)
    n_states = length(state_list)
    n_measurements = length(measurment_list)
    result = zeros(n_states, n_measurements)
    for (i, state) in enumerate(state_list)
        for (j, measurment) in enumerate(measurment_list)
            result[i, j] = expectation_value(state, measurment)
        end
    end    
    return result
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

## ------------ Evolve states ----------------
evolved_states = [state_evolution(state, t_eval, hamiltonian) for state in states_tot]
measurments_schrödinger = measure_states(evolved_states, ops)

## ------------ Evolve measurments ----------------
evolved_ops = [operator_evolution(op, t_eval, hamiltonian) for op in ops]
measurments_heisenberg = measure_states(states_tot, evolved_ops)

## ------------ Effective measurments ----------------
eff_measurments = [get_eff_measurment(op, ρ_R, hamiltonian, t_eval, d, d_main, d_res) for op in ops]
states_trunc = [state[get_qubit_idx(d_main),get_qubit_idx(d_main)] for state in states]
measurments_effective = measure_states(states_trunc, eff_measurments)

## ------------ Evolve only specified sector ----------------

fock_nbrs = 2+qn
ops = vcat([nbr_op(i, d) for i in 1:nbr_dots], [nbr2_op(i, d) for i in 1:nbr_dots])
eff_measurments = get_eff_measurments(ops, ρ_R, hamiltonian, t_eval, d, d_main, d_res, fock_nbrs)
measurment_focknbr = measure_states(states, eff_measurments, d_main)

## ------------ Compare ----------------
println(measurments_heisenberg  ≈ measurments_schrödinger)
println(measurments_effective ≈ measurments_schrödinger)
println(measurments_effective ≈ measurments_heisenberg)
println(measurment_focknbr ≈ measurments_effective)