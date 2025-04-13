includet("../../src/basis.jl")
includet("../../src/hamiltonian.jl")
includet("../../src/time_evolution.jl")
includet("../../src/measurements.jl")
includet("../../src/main_system_state.jl")
includet("../../src/reservoir_state.jl")
includet("../../src/effective_measurements.jl")
includet("../../src/generate_data.jl")

using JLD2

function define_system_parameters(res_qd, qn, seed)
    sys_qd = 2
    res_qd = res_qd
    t_eval = 1.0

    conserved_qn = QuantumDots.fermionnumber
    d_tot = total_basis(sys_qd, res_qd, conserved_qn = conserved_qn)
    d, d_main, dA_main, dB_main, d_res = d_tot
    hamiltonian_type = Hdot_so_b

    hamiltonian = random_hamiltonian(d, hamiltonian_type, seed=seed)
    qn = qn
    focknbrs = 2+qn
    ρ_R = res_ground_state(hamiltonian, d, d_res, qn)
    return hamiltonian, ρ_R, t_eval, d, d_main, d_res, focknbrs
end


function save_measurement_data()
    res_qds = [6]
    seeds = [1,2,4]
    for seed in seeds
        for res_qd in res_qds
            qns = nothing
            if res_qd <= 4
                qns = [i for i in 0:res_qd*2]
            elseif res_qd == 5
                qns = [i for i in 0:3]
            elseif res_qd == 6
                qns = [i for i in 0:1]
            end
            for qn in qns
                measurements = get_effective_measurements(define_system_parameters(res_qd, qn, seed)...)
                flattened_matrices = [vec(measurement) for measurement in measurements]
                measurements_matrix = hcat(flattened_matrices...)
                save("Plots/Evaluate_measurements/data/measmat_resqd_$(res_qd)_qn_$(qn)_seed_$(seed).jld2", "A", measurements_matrix)
            end
        end
    end
end

save_measurement_data()