function get_eff_measurement(measurement_op :: AbstractMatrix, ρ_r :: AbstractMatrix,
    hamiltonian :: AbstractMatrix, t_ev, 
    d :: FermionBasis, d_main :: FermionBasis, d_res :: FermionBasis; 
    truncate_density_matrix :: Bool = true,) 

    op_ev = operator_evolution(measurement_op, t_ev, hamiltonian)

    exp_value(ρ_I) = tr(wedge([sparse(ρ_I), ρ_r], [d_main, d_res], d)*op_ev)

    function f_vec_trunc(ρ_I_vec_trunc)
        ρ_I_trunc = reshape(ρ_I_vec_trunc, 4, 4)'
        ρ_I = zeros(Complex{Float64}, 16, 16)
        idx = get_qubit_idx(d_main)
        ρ_I[idx, idx] = ρ_I_trunc
        return exp_value(ρ_I)
    end
    
    function f_vec(ρ_I_vec)
        ρ_I = reshape(ρ_I_vec, 16, 16)'
        return exp_value(ρ_I)
    end

    n_eff = nothing
    if truncate_density_matrix
        lmap = LinearMaps.LinearMap(f_vec_trunc, 1, 16)
        n_eff = sparse(reshape(Matrix{Complex{Float64}}(lmap), 4, 4))
    else
        lmap = LinearMaps.LinearMap(f_vec, 1, 16^2)
        n_eff = sparse(reshape(Matrix{Complex{Float64}}(lmap), 16, 16))
    end

    return n_eff
end


function get_eff_measurement(op_ev :: AbstractMatrix, ρ_r :: AbstractMatrix,
    d :: FermionBasis, d_main :: FermionBasis, d_res :: FermionBasis; )
    """
        get_eff_measurement(op_ev :: AbstractMatrix, ρ_r :: AbstractMatrix,
        d :: FermionBasis, d_main :: FermionBasis, d_res :: FermionBasis; )

    Returns the effective measurement for the given evolved measurement operator.
    """
    exp_value(ρ_I) = tr(wedge([sparse(ρ_I), ρ_r], [d_main, d_res], d)*op_ev)

    function f_vec_trunc(ρ_I_vec_trunc)
        ρ_I_trunc = reshape(ρ_I_vec_trunc, 4, 4)'
        ρ_I = zeros(Complex{Float64}, 16, 16)
        idx = get_qubit_idx(d_main)
        ρ_I[idx, idx] = ρ_I_trunc
        return exp_value(ρ_I)
    end

    lmap = LinearMaps.LinearMap(f_vec_trunc, 1, 16)
    n_eff = sparse(reshape(Matrix{Complex{Float64}}(lmap), 4, 4))
    return n_eff

end

function get_eff_measurements(measurement_ops, ρ_r :: AbstractMatrix,
    hamiltonian :: AbstractMatrix, t_eval, 
    d :: FermionBasis, d_main :: FermionBasis, d_res :: FermionBasis, 
    focknbrs)
    """
        get_eff_measurements(measurement_ops, ρ_r :: AbstractMatrix,
        hamiltonian :: AbstractMatrix, t_ev, 
        d :: FermionBasis, d_main :: FermionBasis, d_res :: FermionBasis; 
        focknbrs)

    Returns the effective measurements for the given measurement operators 
    and specified focknbrs.
    """
    u_block = get_propagator_block(t_eval, hamiltonian, d, focknbrs)
    eff_measurements = []
    for op in measurement_ops
        op_ev = operator_evolution_blocks(op, u_block, d, focknbrs)
        push!(eff_measurements, get_eff_measurement(op_ev, ρ_r, d, d_main, d_res))
    end
    return eff_measurements

end