includet("../src/basis.jl")
includet("../src/hamiltonian.jl")
includet("../src/time_evolution.jl")
includet("../src/measurments.jl")
includet("../src/main_system_state.jl")
includet("../src/reservoir_state.jl")
includet("../src/effective_measurments.jl")
includet("../src/generate_data.jl")

## -------------- Define System ----------------

function define_system_parameters()

    #------- State parameters -------
    # State types
    ent_state_types = [triplet0_state]
    sep_state_types = [random_separable_state]
    state_types = [ent_state_types, sep_state_types]

    #Nuber of states
    nbr_sep_states = 10000
    nbr_mix_sep_states = 40000
    nbr_ent_states = nbr_sep_states+ nbr_mix_sep_states
    nbr_states = [nbr_ent_states, nbr_sep_states, nbr_mix_sep_states]

    #Noise level for Werner states
    noise_level_min = 1/3
    t_eval = 1.0

    #------- System parameters -------
    # Number of quantum dots in reservoir
    res_qd = 1
c  
    # Interactions in reservoir
    conserved_qn = QuantumDots.fermionnumber
    hamiltonian_type = Hdot_so_b

    # System basis
    d_tot = total_basis(2, res_qd, conserved_qn = conserved_qn)
    d, d_main, dA_main, dB_main, d_res = d_tot

    # Hamiltonian
    hamiltonian = random_hamiltonian(d, hamiltonian_type, seed=4)
    
    # Reservoir state
    res_qn = 1
    focknbrs = 2+res_qn
    ρ_R = res_ground_state(hamiltonian, d, d_res, res_qn)

    return hamiltonian, ρ_R, t_eval, d_tot, nbr_states, state_types, noise_level_min, focknbrs
end

function get_data()
    ent_measurments, sep_measurments, mix_sep_measurements= get_charge_measurments(define_system_parameters()...)
    return ent_measurments, sep_measurments, mix_sep_measurements
end

## ------------ Measure states ----------------

function pca_plot(all_measurments, ent_measurments, sep_measurments, mix_sep_measurements)
    pca = fit(PCA, real(all_measurments)', maxoutdim=3, pratio=1.0)
    pca_sep_measurments = predict(pca, real(sep_measurments)')
    pca_ent_measurments = predict(pca, real(ent_measurments)')
    pca_mix_sep_measurements = predict(pca, real(mix_sep_measurements)')

    scatter(pca_sep_measurments[1,:], pca_sep_measurments[2,:], pca_sep_measurments[3,:], label = "sep")
    scatter!(pca_ent_measurments[1,:], pca_ent_measurments[2,:], pca_ent_measurments[3,:], label="ent")
    scatter!(pca_mix_sep_measurements[1,:], pca_mix_sep_measurements[2,:], pca_mix_sep_measurements[3,:], label = "mix_sep")
end

function measurements_plot()

    scatter(sep_measurments[:,1], sep_measurments[:,2], sep_measurments[:,6], label="sep")
    scatter!(ent_measurments[:,1], ent_measurments[:,2], ent_measurments[:,6], label="ent")
    scatter!(mix_sep_measurements[:,1], mix_sep_measurements[:,2], mix_sep_measurements[:,6], label="mix_sep")
end

ent_measurments, sep_measurments, mix_sep_measurements = get_data()
all_measurments = vcat(ent_measurments, sep_measurments, mix_sep_measurements)
pca_plot(all_measurments, ent_measurments, sep_measurments, mix_sep_measurements)
