includet("../src/basis.jl")
includet("../src/hamiltonian.jl")
includet("../src/time_evolution.jl")
includet("../src/measurements.jl")
includet("../src/main_system_state.jl")
includet("../src/reservoir_state.jl")
includet("../src/effective_measurements.jl")
includet("../src/generate_data.jl")

## -------------- Define System ----------------

function define_system_parameters()

    #------- State parameters -------
    # State types
    ent_state_types = [triplet0_state]

    #Nuber of states
    nbr_sep_states = 40000
    nbr_mix_sep_states = 100000
    nbr_ent_states = nbr_sep_states+ nbr_mix_sep_states
    nbr_states = [nbr_ent_states, nbr_sep_states, nbr_mix_sep_states]

    #Noise level for Werner states
    noise_level_min = 1/2
    t_eval = 1.0

    #------- System parameters -------
    # Number of quantum dots in reservoir
    res_qd = 1
 
    # Interactions in reservoir
    conserved_qn = QuantumDots.fermionnumber
    hamiltonian_type = Hdot_so_b


    # System basis
    d_tot = total_basis(2, res_qd, conserved_qn = conserved_qn)
    d, d_main, dA_main, dB_main, d_res = d_tot

    # Hamiltonian
    hamiltonian = random_hamiltonian(d, hamiltonian_type, seed=140)
    
    # Reservoir state
    res_qn = 1
    focknbrs = 2+res_qn
    ρ_R = res_ground_state(hamiltonian, d, d_res, res_qn)

    return hamiltonian, ρ_R, t_eval, d_tot, nbr_states, ent_state_types, noise_level_min, focknbrs
end

function get_data()
    ent_measurements, sep_measurements, mix_sep_measurements= get_charge_measurements(define_system_parameters()...)
    return ent_measurements, sep_measurements, mix_sep_measurements
end

## ------------ Measure states ----------------

function pca_plot(all_measurements, ent_measurements, sep_measurements, mix_sep_measurements)
    pca = fit(PCA, real(all_measurements)', maxoutdim=3, pratio=1.0)
    pca_sep_measurements = predict(pca, real(sep_measurements)')
    pca_ent_measurements = predict(pca, real(ent_measurements)')
    pca_mix_sep_measurements = predict(pca, real(mix_sep_measurements)')

    scatter(pca_sep_measurements[1,:], pca_sep_measurements[2,:], pca_sep_measurements[3,:], label = "sep")
    scatter!(pca_ent_measurements[1,:], pca_ent_measurements[2,:], pca_ent_measurements[3,:], label="ent")
    scatter!(pca_mix_sep_measurements[1,:], pca_mix_sep_measurements[2,:], pca_mix_sep_measurements[3,:], label = "mix_sep")
end

function pca_plot2(all_measurements, ent_measurements, sep_measurements, mix_sep_measurements)
    pca = fit(PCA, real(all_measurements)', maxoutdim=2, pratio=1.0)
    pca_sep_measurements = predict(pca, real(sep_measurements)')
    pca_ent_measurements = predict(pca, real(ent_measurements)')
    pca_mix_sep_measurements = predict(pca, real(mix_sep_measurements)')

    scatter(pca_sep_measurements[1,:], pca_sep_measurements[2,:], label = "sep")
    scatter!(pca_ent_measurements[1,:], pca_ent_measurements[2,:], label="ent")
    scatter!(pca_mix_sep_measurements[1,:], pca_mix_sep_measurements[2,:], label = "mix_sep")
end

function measurements_plot(sep_measurements, ent_measurements, mix_sep_measurements)

    scatter(sep_measurements[:,1], sep_measurements[:,2], sep_measurements[:,6], label="sep")
    scatter!(ent_measurements[:,1], ent_measurements[:,2], ent_measurements[:,6], label="ent")
    scatter!(mix_sep_measurements[:,1], mix_sep_measurements[:,2], mix_sep_measurements[:,6], label="mix_sep")
end


function measurements_plot2(ent_measurements, sep_measurements, mix_sep_measurements)

    scatter(sep_measurements[:,6], sep_measurements[:,1], label="sep")
    #scatter!(ent_measurements[:,1], ent_measurements[:,2], label="ent")
    scatter!(mix_sep_measurements[:,6], mix_sep_measurements[:,1], label="mix_sep")
end

ent_measurements, sep_measurements, mix_sep_measurements = get_data()
all_measurements = vcat(ent_measurements, sep_measurements, mix_sep_measurements)
pca_plot2(all_measurements, ent_measurements, sep_measurements, mix_sep_measurements)

#measurements_plot2(ent_measurements, sep_measurements, mix_sep_measurements)


