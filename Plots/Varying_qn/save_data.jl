includet("../../src/basis.jl")
includet("../../src/hamiltonian.jl")
includet("../../src/time_evolution.jl")
includet("../../src/measurements.jl")
includet("../../src/main_system_state.jl")
includet("../../src/reservoir_state.jl")
includet("../../src/effective_measurements.jl")
includet("../../src/generate_data.jl")
using JLD2


## -------------- Define System ----------------
function define_system_parameters(res_qd, qn, seed)
    t_eval = 1.0
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

    return hamiltonian, ρ_R, t_eval, d, d_main, d_res, focknbrs
end
 
function save_states()
    nbr_sep_states = 40000
    nbr_mixed_sep_states = 300000
    nbr_ent_states = nbr_sep_states+ nbr_mixed_sep_states
    p_min = 1/2
#
    #Defining the main system basis separately and assume this main system basis is defined in the same waylater on
    d_main, dA_main, dB_main =  main_system_basis([1,2], 1, conserved_qn = QuantumDots.fermionnumber)
    
    ## Define separable states
    sep_pure_states_train, sep_mixed_states_train = generate_sep_states(d_main, dA_main, dB_main, nbr_sep_states, nbr_mixed_sep_states)
    sep_states_train = vcat(sep_pure_states_train, sep_mixed_states_train)
    save("Plots/Varying_qn/States/train_sep.jld2", "A", sep_states_train)
    sep_pure_states_test,sep_mixed_states_test = generate_sep_states(d_main, dA_main, dB_main, nbr_sep_states, nbr_mixed_sep_states)
    sep_states_test = vcat(sep_pure_states_test, sep_mixed_states_test)
    save("Plots/Varying_qn/States/test_sep.jld2", "A", sep_states_test)

    ### Define entangled states
    ent_state_types = [singlet_state, triplet0_state, tripletn1_state, tripletp1_state]
    ent_states_train = nothing
    ent_states_test = nothing
    for ent_state_type in ent_state_types
         ent_states_train = generate_werner_states(d_main, nbr_ent_states, [ent_state_type], p_min)
    #    save("Plots/Varying_qn/States/train_ent_$(ent_state_type).jld2", "A", ent_states_train)
         ent_states_test = generate_werner_states(d_main, nbr_ent_states, [ent_state_type], p_min)
    #    save("Plots/Varying_qn/States/test_ent_$(ent_state_type).jld2", "A", ent_states_test)
    end
#
    ##Define labels for the states
    labels_train = vcat([-1 for i in 1:size(ent_states_train)[1]], [1 for i in 1:size(sep_states_train)[1]])
    np.save("Plots/Varying_qn/States/train_labels.npy", labels_train)
    labels_test = vcat([-1 for i in 1:size(ent_states_train)[1]], [1 for i in 1:size(sep_states_test)[1]])
    np.save("Plots/Varying_qn/States/test_labels.npy", labels_test)
#
    ## Define range of Werner states
    for ent_state_type in ent_state_types
        pmin = 0.0
        nbr_states = 10000
        ent_states = generate_werner_states(d_main, nbr_states, [ent_state_type], pmin)
        save("Plots/Varying_qn/States/werner_$(ent_state_type).jld2", "A", ent_states)
    end
end

function save_data()
    res_qds = [1, 2, 3, 4 , 5, 6]
    seeds = [1, 2, 4, 5, 6, 7]
    ent_state_types = [singlet_state, triplet0_state, tripletn1_state, tripletp1_state]

    sep_states_train = load("Plots/Varying_qn/States/train_sep.jld2")["A"]
    sep_states_test = load("Plots/Varying_qn/States/test_sep.jld2")["A"]

    for ent_state_type in ent_state_types
        ent_states_train = load("Plots/Varying_qn/States/train_ent_$(ent_state_type).jld2")["A"]
        ent_states_test = load("Plots/Varying_qn/States/test_ent_$(ent_state_type).jld2")["A"]
        werner_states = load("Plots/Varying_qn/States/werner_$(ent_state_type).jld2")["A"]
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
                    
                    hamiltonian, ρ_R, t_eval, d, d_main, d_res, focknbrs = define_system_parameters(res_qd, qn, seed)
                    
                    eff_measurements = get_effective_measurements(hamiltonian, ρ_R, t_eval, d, d_main, d_res, focknbrs)
                    save("Plots/Varying_qn/Measurement_Operators/measurementop_res_$(res_qd)_qn_$(qn)_seed_$(seed).jld2", "A", eff_measurements)
                    
                    ent_measurements_train = measure_states(ent_states_train, eff_measurements, d_main)
                    sep_measurements_train = measure_states(sep_states_train, eff_measurements, d_main)
                    ent_measurements_test = measure_states(ent_states_test, eff_measurements, d_main)
                    sep_measurements_test = measure_states(sep_states_test, eff_measurements, d_main)
                    werner_measurements = measure_states(werner_states, eff_measurements, d_main)
                    
                    measurements_train = vcat(hcat(ent_measurements_train), hcat(sep_measurements_train))
                    measurements_test = vcat(hcat(ent_measurements_test), hcat(sep_measurements_test))
                    
                    np.save("Plots/Varying_qn/data_$(ent_state_type)/measurements_train_$(ent_state_type)_res_$(res_qd)_qn_$(qn)_seed_$(seed).npy", measurements_train)                        
                    np.save("Plots/Varying_qn//data_$(ent_state_type)/measurements_test_$(ent_state_type)_res_$(res_qd)_qn_$(qn)_$(seed).npy", measurements_test)
                    np.save("Plots/Varying_qn//data_$(ent_state_type)/measurements_werner_$(ent_state_type)_res_$(res_qd)_qn_$(qn)_$(seed).npy", werner_measurements)
                    GC.gc()

                end
            end
        end
    end
end

save_states()
#save_data()