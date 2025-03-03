function get_eff_measurment(measurment_op :: AbstractMatrix, ρ_r :: AbstractMatrix,
    hamiltonian :: AbstractMatrix, t_ev, 
    d :: FermionBasis, d_main :: FermionBasis, d_res :: FermionBasis, 
    truncate_density_matrix :: Bool = true,) 

    op_ev = operator_evolution(measurment_op, t_ev, hamiltonian)

    exp_value(ρ_I) = tr(wedge([ρ_I, ρ_r], [d_main, d_res], d)*op_ev)

    function f_vec_trunc(ρ_I_vec_trunc)
        ρ_I_trunc = reshape(ρ_I_vec_trunc, 4, 4)
        ρ_I = zeros(Complex{Float64}, 16, 16)
        ρ_I[get_qubit_idx(),get_qubit_idx()] = ρ_I_trunc
        return exp_value(ρ_I)
    end
    
    function f_vec(ρ_I_vec)
        ρ_I = reshape(ρ_I_vec, 16, 16)'
        return exp_value(ρ_I)
    end


    lmap = if truncate_density_matrix
        lmap = LinearMap(f_vec_trunc, 1, 16)
    else
        lmap = LinearMap(f_vec, 1, 16^2)
    end

    n_eff_trunc = sparse(reshape(Matrix{Complex{Float64}}(lmap), 4, 4))
    return n_eff_trunc
end