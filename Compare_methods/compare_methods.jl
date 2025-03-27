includet("../src/basis.jl")
includet("../src/hamiltonian.jl")
includet("../src/time_evolution.jl")
includet("../src/measurements.jl")
includet("../src/main_system_state.jl")
includet("../src/reservoir_state.jl")
includet("../src/effective_measurements.jl")

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

## ------------ Define measurements ----------------
dot_labels = get_spatial_labels(d)
nbr_dots = length(dot_labels)
ops = vcat([nbr_op(i, d) for i in 1:nbr_dots], [nbr2_op(i, d) for i in 1:nbr_dots])

function measure_states(state_list, measurement_list)
    n_states = length(state_list)
    n_measurements = length(measurement_list)
    result = zeros(n_states, n_measurements)
    for (i, state) in enumerate(state_list)
        for (j, measurement) in enumerate(measurement_list)
            result[i, j] = expectation_value(state, measurement)
        end
    end    
    return result
end

function measure_states(state_list, eff_measurements, d_main)
    n_states = length(state_list)
    n_measurements = length(eff_measurements)
    result = zeros(n_states, n_measurements)
    for (i, state) in enumerate(state_list)
        for (j, eff_measurement) in enumerate(eff_measurements)
            idx = get_qubit_idx(d_main)
            trunc_state = state[idx,idx]
            result[i, j] = expectation_value(trunc_state, eff_measurement)
        end
    end    
    return result
end

## ------------ Evolve states ----------------
evolved_states = [state_evolution(state, t_eval, hamiltonian) for state in states_tot]
measurements_schrödinger = measure_states(evolved_states, ops)

## ------------ Evolve measurements ----------------
evolved_ops = [operator_evolution(op, t_eval, hamiltonian) for op in ops]
measurements_heisenberg = measure_states(states_tot, evolved_ops)

## ------------ Effective measurements ----------------
eff_measurements = [get_eff_measurement(op, ρ_R, hamiltonian, t_eval, d, d_main, d_res) for op in ops]
idx = get_qubit_idx(d_main)
states_trunc = [state[idx, idx] for state in states]
measurements_effective = measure_states(states_trunc, eff_measurements)

## ------------ Evolve only specified sector ----------------

fock_nbrs = 2+qn
ops = vcat([nbr_op(i, d) for i in 1:nbr_dots], [nbr2_op(i, d) for i in 1:nbr_dots])
eff_measurements = get_eff_measurements(ops, ρ_R, hamiltonian, t_eval, d, d_main, d_res, fock_nbrs)
measurement_focknbr = measure_states(states, eff_measurements, d_main)

## ------------ Compare ----------------
println(measurements_heisenberg  ≈ measurements_schrödinger)
println(measurements_effective ≈ measurements_schrödinger)
println(measurements_effective ≈ measurements_heisenberg)
println(measurement_focknbr ≈ measurements_effective)