function get_charge_measurments(hamiltonian :: AbstractMatrix, ρ_R :: AbstractMatrix , t_eval, 
    total_basis, nbr_states, state_types, p_min, fock_nbrs)
    """
        Generate the measurements for the entangled and separable states.
    """

    d, d_main, dA_main, dB_main, d_res = total_basis
    nbr_ent_states, nbr_sep_states, nbr_mixed_sep_states = nbr_states
    ent_state_types, sep_state_types = state_types

    ent_states = vcat([werner_state_list(d_main, nbr_ent_states, type, p_min) for type in ent_state_types]...)
    sep_states = vcat([[sep_state_type(d_main, dA_main, dB_main) for i in 1:nbr_sep_states] for sep_state_type in sep_state_types]...)
    
    eff_measurments = get_effective_measurments(hamiltonian, ρ_R, t_eval, d, d_main, d_res, fock_nbrs)

    ent_states_measurments = measure_states(ent_states, eff_measurments, d_main)
    sep_states_measurments = measure_states(sep_states, eff_measurments, d_main)
    mix_sep_states_measurments = get_mixed_measurements(sep_states_measurments, nbr_mixed_sep_states)

    return ent_states_measurments, sep_states_measurments, mix_sep_states_measurments
end

function get_effective_measurments(hamiltonian :: AbstractMatrix, ρ_R :: AbstractMatrix , t_eval, 
    d :: FermionBasis, d_main :: FermionBasis, d_res :: FermionBasis, fock_nbrs)
    """
        Generate the effective measurements corresponding to expectation value 
        of one and two particles on each dot.
    """
    
    dot_labels = get_spatial_labels(d)
    nbr_dots = length(dot_labels)
    
    ops = vcat([nbr_op(i, d) for i in 1:nbr_dots], [nbr2_op(i, d) for i in 1:nbr_dots])
    eff_measurments = get_eff_measurments(ops, ρ_R, hamiltonian, t_eval, d, d_main, d_res, fock_nbrs)
    return eff_measurments
end

function get_mixed_measurements(sep_states_measurments, nbr_mixed_sep_states)
    """
        Generate the measurements for the mixed separable states by linear combination of measurments of pure states. 
        Start of by removing pure states that are not edge states.          
    """
    data_batches = split_data(sep_states_measurments)
    extreme_points = collect(vcat([get_extreme_points(batch) for batch in data_batches]...))
    nbr_mix_measurements2 = nbr_mixed_sep_states ÷ 2
    nbr_mix_measurements3 = nbr_mixed_sep_states - nbr_mix_measurements2
    mix_measurements2 = get_mixed_measurement(extreme_points, 2, nbr_mix_measurements2)
    mix_measurements3 = get_mixed_measurement(extreme_points, 3, nbr_mix_measurements3)
    return vcat(mix_measurements2, mix_measurements3)
end

function get_mixed_measurement(extreme_points, nbr_combinations, nbr_mix_measurements)
    """
    Generate the mixed measurements by linear combination of nbr_mix_measurements extreme points.
    """
    mix_measurements = zeros(nbr_mix_measurements, size(extreme_points, 2))
    nbr_extreme_points = size(extreme_points, 1)

    for i in 1:nbr_mix_measurements
        points = [extreme_points[rand(1:nbr_extreme_points), :] for i in 1:nbr_combinations]
        probs = rand(nbr_combinations)
        probs /= sum(probs) 
        mix_measurements[i, :] = probs' * points
    end

    return mix_measurements
end

function get_extreme_points(data)
    """
    Get the extreme points of the convex hull of the data.
    """
    v = vrep(data)
    p = polyhedron(v, CDDLib.Library())
    removevredundancy!(p)
    return reduce(hcat, points(p))'
end

function split_data(data)
    """
    Split the data into batches of 1000 points.
    """
    nbr_points = size(data, 1)
    batch_size = 1000
    points_batches = [data[i:min(i + batch_size - 1, nbr_points), :] for i in 1:batch_size:nbr_points]
    return points_batches
end